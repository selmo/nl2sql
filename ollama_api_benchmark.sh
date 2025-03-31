#!/bin/bash

# Ollama API 기반 성능 측정 자동화 스크립트
# 사용법: ./ollama_api_benchmark.sh [모델명] [프롬프트 길이] [테스트 반복 횟수] [콜드스타트] [웜업]

set -e

# 사용법 함수
usage() {
    echo "Ollama API 기반 성능 측정 자동화 스크립트"
    echo "사용법: $0 [모델명] [프롬프트 길이] [테스트 반복 횟수] [콜드스타트] [웜업]"
    echo ""
    echo "파라미터:"
    echo "  모델명         - Ollama에 저장된 모델명 또는 'all'(기본값: all)"
    echo "  프롬프트 길이   - 테스트에 사용할 프롬프트 길이 (기본값: 100)"
    echo "  테스트 반복 횟수 - 각 모델당 테스트 반복 횟수 (기본값: 5)"
    echo "  콜드스타트     - 'true', 'false' 중 선택 (기본값: false)"
    echo "  웜업           - 'true', 'false' 중 선택 (기본값: true)"
    echo ""
    echo "환경변수:"
    echo "  OLLAMA_API_HOST - Ollama API 호스트 (기본값: localhost)"
    echo "  OLLAMA_API_PORT - Ollama API 포트 (기본값: 11434)"
    echo ""
    echo "예시:"
    echo "  $0                            # 모든 모델 테스트 (localhost)"
    echo "  $0 phi3                       # phi3 모델만 테스트 (localhost)"
    echo "  OLLAMA_API_HOST=192.168.1.100 $0 phi3  # 원격 서버로 테스트"
    exit 1
}

# 명령줄 인자 처리
[ "$1" = "-h" ] || [ "$1" = "--help" ] && usage

# 기본 설정값
MODEL=${1:-"all"}
PROMPT_LENGTH=${2:-100}
REPEAT=${3:-5}
COLD_START=${4:-"false"}
WARMUP=${5:-"true"}

# 환경변수에서 API 호스트 및 포트 설정 읽기
API_HOST=${OLLAMA_API_HOST:-"localhost"}
API_PORT=${OLLAMA_API_PORT:-"11434"}
API_BASE_URL="http://$API_HOST:$API_PORT/api"

# 결과 디렉토리 생성
RESULT_DIR="ollama_api_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# 환경 정보 보여주기
show_env_info() {
    echo "==== 벤치마크 환경 정보 ===="
    echo "모델: ${MODEL:-모든 모델}"
    echo "프롬프트 길이: $PROMPT_LENGTH"
    echo "반복 횟수: $REPEAT"
    echo "콜드 스타트: $COLD_START"
    echo "웜업: $WARMUP"
    echo "API 주소: $API_HOST:$API_PORT"
    echo "==========================="
}

# 스크립트 시작 시 환경 정보 표시
show_env_info

# API 접근 테스트
test_api_connection() {
    local api_url="$API_BASE_URL/tags"
    local response

    echo "API 연결 테스트 중..."
    response=$(curl -s "$api_url" 2>/dev/null)

    # 응답 검증
    if [[ "$response" == *"models"* ]]; then
        echo "API 연결 성공"
        return 0
    else
        echo "오류: API 연결 실패. Ollama 서버가 실행 중인지 확인하세요."
        echo "응답: $response"
        return 1
    fi
}

# API를 통해 모델 목록 가져오기
get_models_api() {
    local api_url="$API_BASE_URL/tags"
    local response=$(curl -s "$api_url")

    # jq 사용 가능 여부 확인
    if command -v jq &> /dev/null; then
        echo "$response" | jq -r '.models[].name'
    else
        # jq 없는 경우 간단한 패턴 매칭
        echo "$response" | grep -o '"name":"[^"]*"' | cut -d'"' -f4
    fi
}

# API를 통해 모델 언로드
unload_model_api() {
    local model=$1
    local api_url="$API_BASE_URL/delete"
    local request="{\"name\":\"$model\"}"

    echo "모델 언로드 중: $model"
    curl -s -X DELETE "$api_url" -H "Content-Type: application/json" -d "$request" > /dev/null

    # 잠시 대기 (언로드 완료 확인)
    sleep 1
}

# 프롬프트 생성 함수
generate_prompt() {
    local length=$1
    echo "질문: 간단한 문장을 요약해주세요. 이것은 Ollama 모델 성능 테스트를 위한 프롬프트입니다. 길이는 $length 정도로 설정되어 있으며, 모델의 추론 속도와 토큰 생성 속도를 측정하기 위한 목적입니다."
}

# API를 통한 모델 벤치마크 실행
benchmark_model_api() {
    local model=$1
    local prompt=$2
    local output_file=$3
    local is_cold_start=$4

    # API 요청 JSON 생성
    local request_json=$(cat <<EOF
{
  "model": "$model",
  "prompt": "$prompt",
  "stream": false,
  "options": {
    "num_predict": 512
  }
}
EOF
)

    # API 호출 타이밍 측정 시작
    local start_time=$(date +%s.%N)

    # API 호출 실행
    curl -s -X POST "$API_BASE_URL/generate" -H "Content-Type: application/json" -d "$request_json" > "$output_file.json"

    # API 호출 타이밍 측정 종료
    local end_time=$(date +%s.%N)
    local total_time=$(echo "$end_time - $start_time" | bc)

    # API 응답 분석
    analyze_api_response "$output_file.json" "$total_time" "$output_file.stats"

    # 통계 파일이 존재하는지 확인
    if [ -f "$output_file.stats" ]; then
        # 통계 파일 로드
        source "$output_file.stats"

        # 결과 요약 출력
        echo "---- $model 벤치마크 결과 ----"
        echo "시작 유형: $([ "$is_cold_start" = "true" ] && echo "콜드 스타트" || echo "웜 스타트")"
        echo "모델 로드 시간: ${LOAD_TIME:-N/A} 초"
        echo "첫 토큰 생성 시간: ${FIRST_TOKEN_TIME:-N/A} 초"
        echo "총 응답 시간: ${TOTAL_TIME:-N/A} 초"
        echo "생성된 토큰 수: ${TOKENS_GENERATED:-N/A}"
        echo "토큰 생성 속도: ${TOKENS_PER_SEC:-N/A} tokens/sec"
        echo "---------------------------"

        return 0
    else
        echo "오류: API 응답 분석 실패"
        return 1
    fi
}

# API 응답 분석 함수
analyze_api_response() {
    local json_file=$1
    local measured_time=$2
    local output_file=$3

    # 출력 파일 초기화
    > "$output_file"

    # jq 사용 가능 여부 확인
    if command -v jq &> /dev/null; then
        # JSON 파일 유효성 검사
        if ! jq empty "$json_file" 2>/dev/null; then
            echo "오류: 유효하지 않은 JSON 응답" >&2
            echo "LOAD_TIME=N/A" >> "$output_file"
            echo "FIRST_TOKEN_TIME=N/A" >> "$output_file"
            echo "TOTAL_TIME=$measured_time" >> "$output_file"
            echo "TOKENS_GENERATED=N/A" >> "$output_file"
            echo "TOKENS_PER_SEC=N/A" >> "$output_file"
            return 1
        fi

        # API 오류 확인
        local error=$(jq -r '.error // empty' "$json_file")
        if [ -n "$error" ]; then
            echo "API 오류: $error" >&2
            echo "LOAD_TIME=N/A" >> "$output_file"
            echo "FIRST_TOKEN_TIME=N/A" >> "$output_file"
            echo "TOTAL_TIME=$measured_time" >> "$output_file"
            echo "TOKENS_GENERATED=N/A" >> "$output_file"
            echo "TOKENS_PER_SEC=N/A" >> "$output_file"
            return 1
        fi

        # 성능 지표 추출
        local total_duration=$(jq -r '.total_duration // 0' "$json_file")
        local load_duration=$(jq -r '.prompt_eval_duration // 0' "$json_file")
        local eval_duration=$(jq -r '.eval_duration // 0' "$json_file")
        local eval_count=$(jq -r '.eval_count // 0' "$json_file")

        # 나노초를 초로 변환
        local load_time="N/A"
        local first_token_time="N/A"
        local total_duration_sec="$measured_time"  # 기본값은 측정된 시간

        if [ "$load_duration" != "0" ]; then
            load_time=$(echo "scale=6; $load_duration / 1000000000" | bc)
        fi

        if [ "$eval_duration" != "0" ] && [ "$total_duration" != "0" ]; then
            # 첫 토큰 시간 = 전체 시간 - 토큰 생성 시간
            first_token_time=$(echo "scale=6; ($total_duration - $eval_duration) / 1000000000" | bc)
            total_duration_sec=$(echo "scale=6; $total_duration / 1000000000" | bc)
        fi

        # 토큰 생성 속도 계산
        local tokens_per_sec="N/A"
        if [ "$eval_count" != "0" ] && [ "$total_duration_sec" != "N/A" ]; then
            if (( $(echo "$total_duration_sec > 0" | bc -l) )); then
                tokens_per_sec=$(echo "scale=2; $eval_count / $total_duration_sec" | bc)
            fi
        fi

        # 결과 저장
        echo "LOAD_TIME=$load_time" >> "$output_file"
        echo "FIRST_TOKEN_TIME=$first_token_time" >> "$output_file"
        echo "TOTAL_TIME=$total_duration_sec" >> "$output_file"
        echo "TOKENS_GENERATED=$eval_count" >> "$output_file"
        echo "TOKENS_PER_SEC=$tokens_per_sec" >> "$output_file"
    else
        # jq 없는 경우 간단한 grep 기반 파싱
        echo "경고: jq가 설치되어 있지 않아 제한된 정보만 추출합니다."

        # eval_count 추출 시도
        local response=$(cat "$json_file")
        local eval_count=$(echo "$response" | grep -o '"eval_count":[0-9]*' | grep -o '[0-9]*')

        # 토큰 생성 속도 계산
        local tokens_per_sec="N/A"
        if [ -n "$eval_count" ] && [ "$eval_count" != "0" ] && [ "$measured_time" != "0" ]; then
            tokens_per_sec=$(echo "scale=2; $eval_count / $measured_time" | bc)
        fi

        # 기본 정보만 저장
        echo "LOAD_TIME=N/A" >> "$output_file"
        echo "FIRST_TOKEN_TIME=N/A" >> "$output_file"
        echo "TOTAL_TIME=$measured_time" >> "$output_file"
        echo "TOKENS_GENERATED=${eval_count:-N/A}" >> "$output_file"
        echo "TOKENS_PER_SEC=$tokens_per_sec" >> "$output_file"
    fi

    return 0
}

# 시스템 정보 수집 및 표시 부분 개선

# API를 통해 Ollama 서버 정보 가져오기
get_server_info() {
    echo "==== Ollama 서버 정보 ($API_HOST:$API_PORT) ===="

    # Ollama 버전 정보 가져오기 (가능한 경우)
    local version_response
    version_response=$(curl -s "$API_BASE_URL/version" 2>/dev/null)

    if [[ "$version_response" == *"version"* ]]; then
        if command -v jq &> /dev/null; then
            local ollama_version=$(echo "$version_response" | jq -r '.version')
            echo "Ollama 버전: $ollama_version"
        else
            # jq 없는 경우 간단한 패턴 매칭
            local ollama_version=$(echo "$version_response" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
            echo "Ollama 버전: $ollama_version"
        fi
    else
        echo "Ollama 버전: 정보를 가져올 수 없음"
    fi

    # API를 통해 가용한 모델 수 계산
    local models_count=$(get_models_api | wc -l)
    echo "사용 가능한 모델 수: $models_count"

    # 원격 서버 여부 확인
    if [ "$API_HOST" = "localhost" ] || [ "$API_HOST" = "127.0.0.1" ]; then
        echo "서버 위치: 로컬"

        # 로컬인 경우에만 시스템 정보 수집
        echo ""
        echo "==== 로컬 시스템 정보 ===="
        echo "운영체제: $(uname -a)"

        # CPU 정보 (OS별 처리)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # Mac OS
            echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
            echo "코어 수: $(sysctl -n hw.ncpu)"
            echo "메모리: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')"
        else
            # Linux
            echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
            echo "코어 수: $(nproc)"
            echo "메모리: $(free -h | grep Mem | awk '{print $2}')"
        fi

        # GPU 정보 수집 (NVIDIA GPU가 있는 경우)
        if command -v nvidia-smi &> /dev/null; then
            echo "==== GPU 정보 ===="
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        fi
    else
        echo "서버 위치: 원격 ($API_HOST)"
        echo "참고: 원격 서버의 상세 시스템 정보는 사용할 수 없습니다."
    fi
}

# 벤치마크 시작 부분 수정
# 기존의 시스템 정보 수집 부분을 get_server_info() 함수 호출로 대체
get_server_info

# API 연결 테스트
test_api_connection || { echo "Ollama API 연결에 실패했습니다. 종료합니다."; exit 1; }

# API 연결 테스트
test_api_connection || { echo "Ollama API 연결에 실패했습니다. 종료합니다."; exit 1; }

# 모델 목록 가져오기
echo "==== Ollama 모델 정보 ===="

# 모델 목록 추출
if [ "$MODEL" = "all" ]; then
    MODEL_LIST=$(get_models_api)
    echo "발견된 모델 목록:"
    echo "$MODEL_LIST"
else
    MODEL_LIST="$MODEL"
    echo "선택된 모델: $MODEL"
fi

# 결과 저장 파일
RESULT_FILE="$RESULT_DIR/api_benchmark_results.csv"
echo "모델,테스트번호,프롬프트길이,시작타입,모델로드시간(초),첫토큰생성시간(초),총생성시간(초),생성된토큰수,토큰생성속도(tokens/sec),CPU사용률(%),메모리사용량(MB),GPU메모리사용량(MB)" > "$RESULT_FILE"

echo "==== 벤치마크 시작 (모델당 $REPEAT 회 반복) ===="

# 모든 모델에 대해 반복
for CURRENT_MODEL in $MODEL_LIST; do
    echo "==== $CURRENT_MODEL 모델 벤치마크 시작 ====="

    # 모델별 결과 디렉토리 생성
    MODEL_RESULT_DIR="$RESULT_DIR/$CURRENT_MODEL"
    mkdir -p "$MODEL_RESULT_DIR"

    # 웜업 실행 (선택적)
    if [ "$WARMUP" = "true" ]; then
        echo "모델 웜업 실행 중... (측정에서 제외됨)"

        # 웜업 프롬프트 생성
        WARMUP_PROMPT="간단한 웜업 프롬프트입니다. 모델을 준비하기 위한 메시지입니다."

        # 웜업용 API 요청 JSON 생성
        warmup_request_json=$(cat <<EOF
{
  "model": "$CURRENT_MODEL",
  "prompt": "$WARMUP_PROMPT",
  "stream": false
}
EOF
)

        # 웜업 API 호출
        curl -s -X POST "$API_BASE_URL/generate" -H "Content-Type: application/json" -d "$warmup_request_json" > /dev/null

        echo "웜업 완료. 성능 측정 시작..."
        # 잠시 대기 (리소스 안정화를 위해)
        sleep 2
    fi

    for i in $(seq 1 $REPEAT); do
        echo "$CURRENT_MODEL 모델 테스트 $i/$REPEAT 실행 중..."

        # 프롬프트 생성
        PROMPT=$(generate_prompt $PROMPT_LENGTH)

        # 콜드 스타트 처리
        is_cold_start="false"
        if [ "$COLD_START" = "true" ]; then
            echo "콜드 스타트 측정을 위해 모델 언로드 중..."
            unload_model_api "$CURRENT_MODEL"
            is_cold_start="true"
        elif [ "$i" -eq 1 ]; then
            # 첫 번째 실행은 콜드 스타트로 간주
            is_cold_start="true"
        fi

        # CPU/메모리 모니터링 시작
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # Mac OS
            ps -p $(pgrep ollama) -o %cpu,%mem > "$MODEL_RESULT_DIR/stats_$i.txt" 2>/dev/null || echo "0,0" > "$MODEL_RESULT_DIR/stats_$i.txt" &
        else
            # Linux
            ps -p $(pgrep -f "ollama") -o %cpu,%mem > "$MODEL_RESULT_DIR/stats_$i.txt" 2>/dev/null || echo "0,0" > "$MODEL_RESULT_DIR/stats_$i.txt" &
        fi
        STATS_PID=$!

        # GPU 상태 모니터링 (NVIDIA GPU가 있는 경우)
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_before_$i.txt"
        fi

        # 벤치마크 실행
        OLLAMA_OUTPUT_FILE="$MODEL_RESULT_DIR/api_output_$i"
        benchmark_model_api "$CURRENT_MODEL" "$PROMPT" "$OLLAMA_OUTPUT_FILE" "$is_cold_start"

        # 통계 파일 로드
        if [ -f "$OLLAMA_OUTPUT_FILE.stats" ]; then
            source "$OLLAMA_OUTPUT_FILE.stats"
        else
            echo "오류: 벤치마크 결과를 찾을 수 없습니다."
            LOAD_TIME="N/A"
            FIRST_TOKEN_TIME="N/A"
            TOTAL_TIME="N/A"
            TOKENS_GENERATED="N/A"
            TOKENS_PER_SEC="N/A"
        fi

        # GPU 사용 후 상태 확인
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_after_$i.txt"
            GPU_MEM_USED=$(cat "$MODEL_RESULT_DIR/gpu_mem_after_$i.txt" | head -1)
        else
            GPU_MEM_USED="N/A"
        fi

        # CPU/메모리 모니터링 중지
        kill $STATS_PID 2>/dev/null || true

        # CPU 및 메모리 사용량 파싱
        if [ -f "$MODEL_RESULT_DIR/stats_$i.txt" ]; then
            # 로컬 환경에서의 CPU/메모리 사용량 파싱
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # Mac OS
                CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $1}' || echo "N/A")
                MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $2}' || echo "N/A")
            else
                # Linux
                CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $1}' || echo "N/A")
                MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $2}' || echo "N/A")
            fi
        else
            CPU_USAGE="N/A"
            MEM_USAGE="N/A"
        fi

        # 결과 기록
        START_TYPE=$([ "$is_cold_start" = "true" ] && echo "cold" || echo "warm")
        echo "$CURRENT_MODEL,$i,$PROMPT_LENGTH,$START_TYPE,$LOAD_TIME,$FIRST_TOKEN_TIME,$TOTAL_TIME,$TOKENS_GENERATED,$TOKENS_PER_SEC,$CPU_USAGE,$MEM_USAGE,$GPU_MEM_USED" >> "$RESULT_FILE"

        # 잠시 대기
        sleep 2
    done

    # 모델별 결과 요약
    echo "==== $CURRENT_MODEL 모델 벤치마크 완료 ===="

    # 웜 스타트와 콜드 스타트 결과 분리
    echo "$CURRENT_MODEL 모델의 콜드 스타트 평균 성능 지표:"
    awk -F, -v model="$CURRENT_MODEL" '$1==model && $4=="cold" {load_sum+=$5; first_token_sum+=$6; total_time_sum+=$7; tokens_sum+=$8; token_rate_sum+=$9; count++} END {if(count>0) printf "모델 로드 시간: %.2f초\n첫 토큰 생성 시간: %.2f초\n총 생성 시간: %.2f초\n평균 생성 토큰 수: %.2f\n평균 토큰 생성 속도: %.2f tokens/sec\n", load_sum/count, first_token_sum/count, total_time_sum/count, tokens_sum/count, token_rate_sum/count; else print "데이터 없음"}' "$RESULT_FILE"

    echo "$CURRENT_MODEL 모델의 웜 스타트 평균 성능 지표:"
    awk -F, -v model="$CURRENT_MODEL" '$1==model && $4=="warm" {load_sum+=$5; first_token_sum+=$6; total_time_sum+=$7; tokens_sum+=$8; token_rate_sum+=$9; count++} END {if(count>0) printf "모델 로드 시간: %.2f초\n첫 토큰 생성 시간: %.2f초\n총 생성 시간: %.2f초\n평균 생성 토큰 수: %.2f\n평균 토큰 생성 속도: %.2f tokens/sec\n", load_sum/count, first_token_sum/count, total_time_sum/count, tokens_sum/count, token_rate_sum/count; else print "데이터 없음"}' "$RESULT_FILE"

    echo "----------------------------------------"
done

# 결과 요약
echo "==== 벤치마크 결과 요약 ===="
echo "결과는 $RESULT_FILE 에 저장되었습니다."

# HTML 보고서를 생성하기 전 Ollama 버전 정보 다시 가져오기
OLLAMA_VERSION="알 수 없음"
version_response=$(curl -s "$API_BASE_URL/version" 2>/dev/null)
if [[ "$version_response" == *"version"* ]]; then
    if command -v jq &> /dev/null; then
        OLLAMA_VERSION=$(echo "$version_response" | jq -r '.version')
    else
        OLLAMA_VERSION=$(echo "$version_response" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    fi
fi

# HTML 보고서 생성
HTML_REPORT="$RESULT_DIR/api_benchmark_report.html"
echo "HTML 보고서 생성 중..."

cat > "$HTML_REPORT" << EOF
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama API 벤치마크 결과</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:hover { background-color: #f5f5f5; }
        .chart-container { margin: 20px 0; }
        .model-section { margin: 30px 0; padding: 20px; border: 1px solid #eee; border-radius: 5px; }
        .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .note { font-style: italic; color: #666; }
        .server-info { background-color: #e6f7ff; border-left: 4px solid #1890ff; padding: 15px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Ollama API 벤치마크 결과</h1>
    <p>테스트 일시: $(date)</p>

    <div class="server-info">
        <h2>Ollama 서버 정보</h2>
        <p><strong>서버 주소:</strong> $API_HOST:$API_PORT</p>
        <p><strong>Ollama 버전:</strong> $OLLAMA_VERSION</p>
        <p><strong>서버 위치:</strong> $(if [ "$API_HOST" = "localhost" ] || [ "$API_HOST" = "127.0.0.1" ]; then echo "로컬"; else echo "원격 ($API_HOST)"; fi)</p>
        <p><strong>사용 가능한 모델 수:</strong> $(get_models_api | wc -l | xargs)</p>
    </div>

EOF

# 로컬 서버인 경우에만 시스템 정보 추가
if [ "$API_HOST" = "localhost" ] || [ "$API_HOST" = "127.0.0.1" ]; then
    cat >> "$HTML_REPORT" << EOF
    <div class="summary">
        <h2>시스템 정보</h2>
        <p>운영체제: $(uname -a)</p>
        $(if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "<p>CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'N/A')</p>";
            echo "<p>코어 수: $(sysctl -n hw.ncpu 2>/dev/null || echo 'N/A')</p>";
            echo "<p>메모리: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 " GB"}' || echo 'N/A')</p>";
        else
            echo "<p>CPU: $(lscpu 2>/dev/null | grep 'Model name' | cut -d':' -f2 | xargs || echo 'N/A')</p>";
            echo "<p>코어 수: $(nproc 2>/dev/null || echo 'N/A')</p>";
            echo "<p>메모리: $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo 'N/A')</p>";
        fi)
        $(if command -v nvidia-smi &> /dev/null; then echo "<p>GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)</p>"; fi)
    </div>
EOF
else
    cat >> "$HTML_REPORT" << EOF
    <div class="note">
        <p>참고: 원격 서버를 사용하기 때문에 상세 시스템 정보는 보고서에 포함되지 않았습니다.</p>
    </div>
EOF
fi

# 테스트 설정 정보 추가
cat >> "$HTML_REPORT" << EOF
    <div class="summary">
        <h2>테스트 설정 정보</h2>
        <ul>
            <li><strong>프롬프트 길이:</strong> $PROMPT_LENGTH 자</li>
            <li><strong>테스트 반복 횟수:</strong> $REPEAT 회/모델</li>
            <li><strong>콜드 스타트 모드:</strong> $(if [ "$COLD_START" = "true" ]; then echo "활성화 (각 테스트마다 모델 언로드 후 재로드)"; else echo "비활성화 (첫 번째 실행만 콜드 스타트로 간주)"; fi)</li>
            <li><strong>웜업 모드:</strong> $(if [ "$WARMUP" = "true" ]; then echo "활성화 (측정 전 웜업 실행)"; else echo "비활성화 (웜업 없이 측정)"; fi)</li>
            <li><strong>벤치마크 실행시간:</strong> $(date)</li>
        </ul>
    </div>

    <h2>모델별 성능 비교</h2>

    <table>
        <tr>
            <th>모델명</th>
            <th>시작 유형</th>
            <th>모델 로드 시간(초)</th>
            <th>첫 토큰 생성 시간(초)</th>
            <th>총 생성 시간(초)</th>
            <th>평균 토큰 수</th>
            <th>토큰 생성 속도(tokens/sec)</th>
        </tr>
EOF

# 모델별 평균 데이터를 HTML 테이블에 추가 (콜드 스타트)
for CURRENT_MODEL in $MODEL_LIST; do
    MODEL_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '
    $1==model && $4=="cold" {
        load_sum+=$5;
        first_token_sum+=$6;
        total_time_sum+=$7;
        tokens_sum+=$8;
        token_rate_sum+=$9;
        count++
    }
    END {
        if(count>0) {
            printf "<tr><td>%s</td><td>콜드 스타트</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>",
            model,
            load_sum/count,
            first_token_sum/count,
            total_time_sum/count,
            tokens_sum/count,
            token_rate_sum/count
        } else {
            printf "<tr><td>%s</td><td>콜드 스타트</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>",
            model
        }
    }' "$RESULT_FILE")
    echo "$MODEL_AVERAGES" >> "$HTML_REPORT"
done

# 모델별 평균 데이터를 HTML 테이블에 추가 (웜 스타트)
for CURRENT_MODEL in $MODEL_LIST; do
    MODEL_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '
    $1==model && $4=="warm" {
        load_sum+=$5;
        first_token_sum+=$6;
        total_time_sum+=$7;
        tokens_sum+=$8;
        token_rate_sum+=$9;
        count++
    }
    END {
        if(count>0) {
            printf "<tr><td>%s</td><td>웜 스타트</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>",
            model,
            load_sum/count,
            first_token_sum/count,
            total_time_sum/count,
            tokens_sum/count,
            token_rate_sum/count
        } else {
            printf "<tr><td>%s</td><td>웜 스타트</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>",
            model
        }
    }' "$RESULT_FILE")
    echo "$MODEL_AVERAGES" >> "$HTML_REPORT"
done

# HTML 마무리
cat >> "$HTML_REPORT" << EOF
    </table>
</body>
</html>
EOF

echo "HTML 보고서가 $HTML_REPORT 에 생성되었습니다."
echo "벤치마크 완료! 결과는 $RESULT_DIR 디렉토리에서 확인할 수 있습니다."