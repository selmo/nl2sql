#!/bin/bash

# Ollama 성능 측정 자동화 스크립트 (Docker 및 로컬 환경 지원)
# 사용법: ./ollama_benchmark.sh [모델명] [프롬프트 길이] [테스트 반복 횟수] [환경타입] [콜드스타트] [웜업]

set -e

# 사용법 함수
usage() {
    echo "Ollama 성능 측정 자동화 스크립트"
    echo "사용법: $0 [모델명] [프롬프트 길이] [테스트 반복 횟수] [환경타입] [콜드스타트] [웜업]"
    echo ""
    echo "파라미터:"
    echo "  모델명         - Ollama에 저장된 모델명 또는 'all'(기본값: all)"
    echo "  프롬프트 길이   - 테스트에 사용할 프롬프트 길이 (기본값: 100)"
    echo "  테스트 반복 횟수 - 각 모델당 테스트 반복 횟수 (기본값: 5)"
    echo "  환경타입       - 'docker', 'local', 'auto' 중 선택 (기본값: auto)"
    echo "  콜드스타트     - 'true', 'false' 중 선택 (기본값: false)"
    echo "                  true: 각 테스트 전에 모델을 언로드하여 콜드 스타트 측정"
    echo "                  false: 모델 로드 상태를 유지하여 웜 스타트만 측정"
    echo "  웜업           - 'true', 'false' 중 선택 (기본값: true)"
    echo "                  true: 성능 측정 전 웜업 실행 (측정에서 제외)"
    echo "                  false: 웜업 없이 바로 측정 시작"
    echo ""
    echo "예시:"
    echo "  $0                       # 모든 모델 테스트 (웜업 포함, 웜 스타트 측정)"
    echo "  $0 llama2                # llama2 모델만 테스트 (웜업 포함)"
    echo "  $0 all 100 5 auto true true  # 콜드 스타트 측정 + 웜업 실행"
    echo "  $0 all 100 5 auto false false # 웜업 없이 웜 스타트 측정"
    exit 1
}

# 도움말 옵션 처리
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

# 기본 설정값
MODEL=${1:-"all"}  # 기본값을 "all"로 변경
PROMPT_LENGTH=${2:-100}
REPEAT=${3:-5}
ENV_TYPE=${4:-"auto"}  # 환경 타입: docker, local, auto(자동감지)
COLD_START=${5:-"false"}  # 콜드 스타트 측정 여부: true, false (기본값: false)
WARMUP=${6:-"true"}  # 성능 측정 전 웜업 실행 여부 (기본값: true)

# 환경 타입 자동 감지
if [ "$ENV_TYPE" = "auto" ]; then
    if command -v docker &> /dev/null && docker ps | grep -q ollama; then
        ENV_TYPE="docker"
        echo "Docker 환경이 감지되었습니다."
    else
        ENV_TYPE="local"
        echo "로컬 환경으로 설정합니다."
    fi
fi

# Docker 컨테이너 확인 (Docker 환경인 경우)
if [ "$ENV_TYPE" = "docker" ]; then
    DOCKER_CONTAINER=$(docker ps | grep ollama | awk '{print $1}')
    if [ -z "$DOCKER_CONTAINER" ]; then
        echo "경고: Ollama Docker 컨테이너를 찾을 수 없습니다. 로컬 환경으로 전환합니다."
        ENV_TYPE="local"
    else
        echo "Ollama Docker 컨테이너 ID: $DOCKER_CONTAINER"
    fi
fi

# ollama 커맨드 확인 (로컬 환경인 경우)
if [ "$ENV_TYPE" = "local" ]; then
    if ! command -v ollama &> /dev/null; then
        echo "오류: ollama 커맨드를 찾을 수 없습니다. ollama가 설치되어 있는지 확인하세요."
        exit 1
    fi
fi

# 랜덤 프롬프트 생성 함수 (Mac OS 호환성 개선)
generate_prompt() {
    local length=$1
    local chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    local result=""

    # Mac OS 호환 방식으로 랜덤 문자열 생성
    for i in $(seq 1 $length); do
        local rand=$((RANDOM % ${#chars}))
        result="${result}${chars:$rand:1}"
    done

    echo "$result"
}

# 시스템 정보 수집 (Mac OS 호환성 개선)
echo "==== 시스템 정보 ===="
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

# Docker 컨테이너 정보 수집 (Docker 환경인 경우)
if [ "$ENV_TYPE" = "docker" ]; then
    echo "==== Docker 컨테이너 정보 ===="
    docker inspect "$DOCKER_CONTAINER" | jq '.[] | {Name: .Name, Image: .Config.Image, CPUs: .HostConfig.NanoCpus, Memory: .HostConfig.Memory}' || echo "Docker 정보를 가져올 수 없습니다."
fi

# 결과 저장 디렉토리 생성
RESULT_DIR="ollama_benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# Ollama 모델 정보 수집 및 모델 목록 가져오기
echo "==== Ollama 모델 정보 ===="

# 환경에 따라 명령 실행
if [ "$ENV_TYPE" = "docker" ]; then
    docker exec "$DOCKER_CONTAINER" ollama list

    # 모델 목록 추출
    if [ "$MODEL" = "all" ]; then
        # 모든 모델을 테스트할 경우 모델 목록 추출
        MODEL_LIST=$(docker exec "$DOCKER_CONTAINER" ollama list | tail -n +2 | awk '{print $1}')
        echo "발견된 모델 목록: $MODEL_LIST"
    else
        # 단일 모델만 테스트
        MODEL_LIST="$MODEL"
    fi
else
    ollama list

    # 모델 목록 추출
    if [ "$MODEL" = "all" ]; then
        # 모든 모델을 테스트할 경우 모델 목록 추출
        MODEL_LIST=$(ollama list | tail -n +2 | awk '{print $1}')
        echo "발견된 모델 목록: $MODEL_LIST"
    else
        # 단일 모델만 테스트
        MODEL_LIST="$MODEL"
    fi
fi

# 측정 결과 저장 파일
RESULT_FILE="$RESULT_DIR/benchmark_results.csv"
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
        WARMUP_PROMPT=$(generate_prompt 50)

        # 웜업을 위한 실행 (결과는 저장하지 않음)
        if [ "$ENV_TYPE" = "docker" ]; then
            docker exec -i "$DOCKER_CONTAINER" ollama run $CURRENT_MODEL "$WARMUP_PROMPT" > /dev/null 2>&1
        else
            ollama run $CURRENT_MODEL "$WARMUP_PROMPT" > /dev/null 2>&1
        fi

        echo "웜업 완료. 성능 측정 시작..."
        # 잠시 대기 (리소스 안정화를 위해)
        sleep 2
    fi

    for i in $(seq 1 $REPEAT); do
        echo "$CURRENT_MODEL 모델 테스트 $i/$REPEAT 실행 중..."

        # 프롬프트 생성
        PROMPT=$(generate_prompt $PROMPT_LENGTH)

        # 콜드 스타트 측정을 위해 모델 언로드 (선택적)
        if [ "$COLD_START" = "true" ]; then
            echo "콜드 스타트 측정을 위해 모델 언로드 중..."
            if [ "$ENV_TYPE" = "docker" ]; then
                docker exec "$DOCKER_CONTAINER" ollama rm -f $CURRENT_MODEL 2>/dev/null || true
                docker exec "$DOCKER_CONTAINER" ollama pull $CURRENT_MODEL >/dev/null 2>&1
            else
                ollama rm -f $CURRENT_MODEL 2>/dev/null || true
                ollama pull $CURRENT_MODEL >/dev/null 2>&1
            fi
            START_TYPE="cold"

            # 콜드 스타트 후 웜업이 필요한 경우 다시 진행
            if [ "$WARMUP" = "true" ] && [ "$i" -gt 1 ]; then
                echo "모델 재웜업 실행 중..."
                WARMUP_PROMPT=$(generate_prompt 50)

                if [ "$ENV_TYPE" = "docker" ]; then
                    docker exec -i "$DOCKER_CONTAINER" ollama run $CURRENT_MODEL "$WARMUP_PROMPT" > /dev/null 2>&1
                else
                    ollama run $CURRENT_MODEL "$WARMUP_PROMPT" > /dev/null 2>&1
                fi

                sleep 2
            fi
        else
            # 첫 번째 실행은 콜드 스타트로 간주, 이후는 웜 스타트
            if [ "$i" -eq 1 ]; then
                START_TYPE="cold"
            else
                START_TYPE="warm"
            fi
        fi

        # 컨테이너 스탯 모니터링 시작 (Docker 환경인 경우)
        if [ "$ENV_TYPE" = "docker" ]; then
            docker stats "$DOCKER_CONTAINER" --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" > "$MODEL_RESULT_DIR/stats_$i.txt" &
            STATS_PID=$!
        else
            # 로컬 환경에서는 OS에 맞는 방식으로 CPU/메모리 사용량 모니터링
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # Mac OS용
                ps -p $(pgrep ollama) -o %cpu,%mem > "$MODEL_RESULT_DIR/stats_$i.txt" 2>/dev/null || echo "0,0" > "$MODEL_RESULT_DIR/stats_$i.txt" &
            else
                # Linux용
                ps -p $(pgrep -f "ollama") -o %cpu,%mem > "$MODEL_RESULT_DIR/stats_$i.txt" 2>/dev/null || echo "0,0" > "$MODEL_RESULT_DIR/stats_$i.txt" &
            fi
            STATS_PID=$!
        fi

        # GPU 상태 모니터링 (NVIDIA GPU가 있는 경우)
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_before_$i.txt"
        fi

        # 모델 로드 시간 및 추론 측정
        START_TIME=$(date +%s.%N)
        OLLAMA_OUTPUT_FILE="$MODEL_RESULT_DIR/ollama_output_$i.txt"

        # Docker 또는 로컬 환경에서 Ollama 실행하고 결과 수집
        if [ "$ENV_TYPE" = "docker" ]; then
            docker exec -i "$DOCKER_CONTAINER" bash -c "time ollama run $CURRENT_MODEL \"$PROMPT\" --verbose" > "$OLLAMA_OUTPUT_FILE" 2>&1
        else
            time ollama run $CURRENT_MODEL "$PROMPT" --verbose > "$OLLAMA_OUTPUT_FILE" 2>&1
        fi

        END_TIME=$(date +%s.%N)
        TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

        # GPU 사용 후 상태 확인
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_after_$i.txt"
            GPU_MEM_USED=$(cat "$MODEL_RESULT_DIR/gpu_mem_after_$i.txt" | head -1)
        else
            GPU_MEM_USED="N/A"
        fi

        # Docker 스탯 모니터링 중지
        kill $STATS_PID 2>/dev/null || true

        # 결과 파싱
        if [ -f "$OLLAMA_OUTPUT_FILE" ]; then
            # 첫 토큰 생성 시간 추출 (로그에서 패턴 찾기)
            LOAD_TIME=$(grep -o "load [0-9.]*ms" "$OLLAMA_OUTPUT_FILE" | awk '{print $2}' | sed 's/ms//' | awk '{print $1/1000}' || echo "N/A")
            FIRST_TOKEN_TIME=$(grep -o "first token [0-9.]*ms" "$OLLAMA_OUTPUT_FILE" | awk '{print $3}' | sed 's/ms//' | awk '{print $1/1000}' || echo "N/A")

            # 생성된 토큰 수 계산
            TOKENS_GENERATED=$(grep -o "eval [0-9]* tokens" "$OLLAMA_OUTPUT_FILE" | awk '{print $2}' | sort -n | tail -1 || echo "N/A")

            # 토큰 생성 속도 계산
            if [ "$TOKENS_GENERATED" != "N/A" ] && [ "$TOTAL_TIME" != "N/A" ]; then
                TOKENS_PER_SEC=$(echo "scale=2; $TOKENS_GENERATED / $TOTAL_TIME" | bc)
            else
                TOKENS_PER_SEC="N/A"
            fi

            # CPU 및 메모리 사용량 파싱
            if [ -f "$MODEL_RESULT_DIR/stats_$i.txt" ]; then
                if [ "$ENV_TYPE" = "docker" ]; then
                    CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | cut -d',' -f1 | sed 's/%//')
                    MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | cut -d',' -f2 | grep -o "[0-9.]*MiB" | sed 's/MiB//' || echo "N/A")
                else
                    # 로컬 환경에서의 CPU/메모리 사용량 파싱 (OS 호환성 개선)
                    if [[ "$OSTYPE" == "darwin"* ]]; then
                        # Mac OS
                        CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $1}' || echo "N/A")
                        MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $2}' || echo "N/A")
                    else
                        # Linux
                        CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $1}' || echo "N/A")
                        MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $2}' || echo "N/A")
                    fi
                fi
            else
                CPU_USAGE="N/A"
                MEM_USAGE="N/A"
            fi

            # 결과 기록
            echo "$CURRENT_MODEL,$i,$PROMPT_LENGTH,$START_TYPE,$LOAD_TIME,$FIRST_TOKEN_TIME,$TOTAL_TIME,$TOKENS_GENERATED,$TOKENS_PER_SEC,$CPU_USAGE,$MEM_USAGE,$GPU_MEM_USED" >> "$RESULT_FILE"
        else
            echo "오류: Ollama 출력 파일을 찾을 수 없습니다."
        fi

        # 잠시 대기
        sleep 3
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

# HTML 보고서에 오류 처리 추가
# 대체 GNUPLOT을 사용할 수 없는 경우 메시지 표시
if ! command -v gnuplot &> /dev/null; then
    GNUPLOT_MESSAGE="<p class='note'>참고: gnuplot이 설치되어 있지 않아 그래프가 생성되지 않았습니다.</p>"
else
    GNUPLOT_MESSAGE=""
fi

# 모델 비교 표 생성
echo "모델별 평균 성능 비교:"
echo "------------------------------------------------------"
echo "모델명 | 로드유형 | 로드시간(초) | 첫토큰(초) | 총시간(초) | 토큰속도(t/s)"
echo "------------------------------------------------------"

# 모델별 콜드 스타트 평균 계산하여 표시
for CURRENT_MODEL in $MODEL_LIST; do
    COLD_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '$1==model && $4=="cold" {load_sum+=$5; first_token_sum+=$6; total_time_sum+=$7; tokens_sum+=$8; token_rate_sum+=$9; count++} END {if(count>0) printf "%s | 콜드 | %.2f | %.2f | %.2f | %.2f", model, load_sum/count, first_token_sum/count, total_time_sum/count, token_rate_sum/count; else printf "%s | 콜드 | N/A | N/A | N/A | N/A", model}' "$RESULT_FILE")
    echo "$COLD_AVERAGES"
done

# 모델별 웜 스타트 평균 계산하여 표시
for CURRENT_MODEL in $MODEL_LIST; do
    WARM_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '$1==model && $4=="warm" {load_sum+=$5; first_token_sum+=$6; total_time_sum+=$7; tokens_sum+=$8; token_rate_sum+=$9; count++} END {if(count>0) printf "%s | 웜 | %.2f | %.2f | %.2f | %.2f", model, load_sum/count, first_token_sum/count, total_time_sum/count, token_rate_sum/count; else printf "%s | 웜 | N/A | N/A | N/A | N/A", model}' "$RESULT_FILE")
    echo "$WARM_AVERAGES"
done
echo "------------------------------------------------------"

# 결과 시각화 (옵션)
if command -v gnuplot &> /dev/null; then
    echo "결과 그래프 생성 중..."

    # 콜드 스타트 vs 웜 스타트 비교 그래프
    cat > "$RESULT_DIR/cold_warm_comparison.gnu" << EOF
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/cold_warm_comparison.png'
set title "모델별 콜드 스타트 vs 웜 스타트 로딩 시간 비교"
set xlabel "모델"
set ylabel "로드 시간 (초)"
set yrange [0:*]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
set grid y
set key top left
set datafile separator ","

# 데이터 파일 생성
system("awk -F, '\$4==\"cold\" {print \$1,\$5}' $RESULT_FILE | sort | uniq -f 1 > $RESULT_DIR/cold_start.dat")
system("awk -F, '\$4==\"warm\" {print \$1,\$5}' $RESULT_FILE | sort | uniq -f 1 > $RESULT_DIR/warm_start.dat")

plot '$RESULT_DIR/cold_start.dat' using 2:xtic(1) title 'Cold Start' lc rgb '#ff0000', \
     '$RESULT_DIR/warm_start.dat' using 2:xtic(1) title 'Warm Start' lc rgb '#0000ff'
EOF
    gnuplot "$RESULT_DIR/cold_warm_comparison.gnu" || echo "콜드/웜 스타트 비교 그래프 생성 실패"

    # 첫 토큰 생성 시간 비교 그래프
    cat > "$RESULT_DIR/first_token_comparison.gnu" << EOF
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/first_token_comparison.png'
set title "모델별 콜드 스타트 vs 웜 스타트 첫 토큰 생성 시간 비교"
set xlabel "모델"
set ylabel "첫 토큰 생성 시간 (초)"
set yrange [0:*]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
set grid y
set key top left
set datafile separator ","

# 데이터 파일 생성
system("awk -F, '\$4==\"cold\" {print \$1,\$6}' $RESULT_FILE | sort | uniq -f 1 > $RESULT_DIR/cold_token.dat")
system("awk -F, '\$4==\"warm\" {print \$1,\$6}' $RESULT_FILE | sort | uniq -f 1 > $RESULT_DIR/warm_token.dat")

plot '$RESULT_DIR/cold_token.dat' using 2:xtic(1) title 'Cold Start' lc rgb '#ff0000', \
     '$RESULT_DIR/warm_token.dat' using 2:xtic(1) title 'Warm Start' lc rgb '#0000ff'
EOF
    gnuplot "$RESULT_DIR/first_token_comparison.gnu" || echo "첫 토큰 생성 시간 비교 그래프 생성 실패"

    # 기존 그래프도 생성
    cat > "$RESULT_DIR/plot_speed.gnu" << EOF
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/token_speed_comparison.png'
set title "모델별 토큰 생성 속도 비교 (tokens/sec)"
set xlabel "테스트 번호"
set ylabel "토큰/초"
set grid
set key outside
set datafile separator ","

# 막대 그래프용 스크립트 (모델별 평균 속도)
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/model_comparison_bar.png'
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
set title "모델별 평균 토큰 생성 속도"
set ylabel "토큰/초"
set grid y
set auto x
plot '$RESULT_FILE' using 9:xtic(1) title "Token Speed (tokens/sec)" group by 1
EOF
    gnuplot "$RESULT_DIR/plot_speed.gnu" || echo "토큰 속도 비교 그래프 생성 실패"

    echo "그래프가 $RESULT_DIR 디렉토리에 생성되었습니다."
fi

# 모델별 여러 지표 비교 차트 (가능한 경우)
if command -v python3 &> /dev/null; then
    cat > "$RESULT_DIR/comparison_chart.py" << EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 결과 파일 읽기
df = pd.read_csv('$RESULT_FILE')

# 콜드
EOF
    python3 "$RESULT_DIR/comparison_chart.py" || echo "비교 차트 생성 실패"
fi

# HTML 보고서 생성
HTML_REPORT="$RESULT_DIR/benchmark_report.html"
echo "HTML 보고서 생성 중..."

cat > "$HTML_REPORT" << EOF
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama 벤치마크 결과</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:hover { background-color: #f5f5f5; }
        .chart-container { margin: 20px 0; }
        .model-section { margin: 30px 0; padding: 20px; border: 1px solid #eee; border-radius: 5px; }
        .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
        .note { font-style: italic; color: #666; }
    </style>
</head>
<body>
    <h1>Ollama 모델 벤치마크 결과</h1>
    <p>테스트 일시: $(date)</p>

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

    <div class="chart-container">
        <h2>시각화</h2>
        $GNUPLOT_MESSAGE
        $(if [ -f "$RESULT_DIR/cold_warm_comparison.png" ]; then echo "<img src=\"cold_warm_comparison.png\" alt=\"콜드/웜 스타트 로딩 시간 비교\" style=\"max-width:100%; margin-bottom:20px;\">"; fi)
        $(if [ -f "$RESULT_DIR/first_token_comparison.png" ]; then echo "<img src=\"first_token_comparison.png\" alt=\"콜드/웜 스타트 첫 토큰 시간 비교\" style=\"max-width:100%; margin-bottom:20px;\">"; fi)
        $(if [ -f "$RESULT_DIR/cold_warm_comparison_chart.png" ]; then echo "<img src=\"cold_warm_comparison_chart.png\" alt=\"상세 성능 비교 차트\" style=\"max-width:100%; margin-bottom:20px;\">"; fi)
        $(if [ -f "$RESULT_DIR/cold_warm_radar_chart.png" ]; then echo "<img src=\"cold_warm_radar_chart.png\" alt=\"콜드/웜 스타트 레이더 차트\" style=\"max-width:100%; margin-bottom:20px;\">"; fi)
        $(if [ -f "$RESULT_DIR/token_speed_comparison.png" ]; then echo "<img src=\"token_speed_comparison.png\" alt=\"토큰 생성 속도 비교\" style=\"max-width:100%; margin-bottom:20px;\">"; fi)
        $(if [ -f "$RESULT_DIR/model_comparison_bar.png" ]; then echo "<img src=\"model_comparison_bar.png\" alt=\"모델별 평균 속도\" style=\"max-width:100%; margin-bottom:20px;\">"; fi)
    </div>

    <div class="note">
        <p>참고: 테스트는 각 모델당 $REPEAT회 반복되었으며, 프롬프트 길이는 $PROMPT_LENGTH 자로 설정되었습니다.</p>
        <p>콜드 스타트 모드: $(if [ "$COLD_START" = "true" ]; then echo "활성화 (각 테스트마다 모델 언로드 후 재로드)"; else echo "비활성화 (첫 번째 실행만 콜드 스타트로 간주)"; fi)</p>
        <p>웜업 모드: $(if [ "$WARMUP" = "true" ]; then echo "활성화 (측정 전 웜업 실행)"; else echo "비활성화 (웜업 없이 측정)"; fi)</p>
        <p>생성 시간: $(date)</p>
        <p>환경 유형: $(if [ "$ENV_TYPE" = "docker" ]; then echo "Docker"; else echo "로컬"; fi)</p>
    </div>
</body>
</html>
EOF

echo "HTML 보고서가 $HTML_REPORT 에 생성되었습니다."
echo "벤치마크 완료! 결과는 $RESULT_DIR 디렉토리에서 확인할 수 있습니다."#!/bin/bash

# Ollama 성능 측정 자동화 스크립트 (Docker 및 로컬 환경 지원)
# 사용법: ./ollama_benchmark.sh [모델명] [프롬프트 길이] [테스트 반복 횟수] [환경타입] [콜드스타트] [웜업]

set -e

# 사용법 함수
usage() {
    echo "Ollama 성능 측정 자동화 스크립트"
    echo "사용법: $0 [모델명] [프롬프트 길이] [테스트 반복 횟수] [환경타입] [콜드스타트] [웜업]"
    echo ""
    echo "파라미터:"
    echo "  모델명         - Ollama에 저장된 모델명 또는 'all'(기본값: all)"
    echo "  프롬프트 길이   - 테스트에 사용할 프롬프트 길이 (기본값: 100)"
    echo "  테스트 반복 횟수 - 각 모델당 테스트 반복 횟수 (기본값: 5)"
    echo "  환경타입       - 'docker', 'local', 'auto' 중 선택 (기본값: auto)"
    echo "  콜드스타트     - 'true', 'false' 중 선택 (기본값: false)"
    echo "                  true: 각 테스트 전에 모델을 언로드하여 콜드 스타트 측정"
    echo "                  false: 모델 로드 상태를 유지하여 웜 스타트만 측정"
    echo "  웜업           - 'true', 'false' 중 선택 (기본값: true)"
    echo "                  true: 성능 측정 전 웜업 실행 (측정에서 제외)"
    echo "                  false: 웜업 없이 바로 측정 시작"
    echo ""
    echo "예시:"
    echo "  $0                       # 모든 모델 테스트 (웜업 포함, 웜 스타트 측정)"
    echo "  $0 llama2                # llama2 모델만 테스트 (웜업 포함)"
    echo "  $0 all 100 5 auto true true  # 콜드 스타트 측정 + 웜업 실행"
    echo "  $0 all 100 5 auto false false # 웜업 없이 웜 스타트 측정"
    exit 1
}

# 도움말 옵션 처리
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

# 기본 설정값
MODEL=${1:-"all"}  # 기본값을 "all"로 변경
PROMPT_LENGTH=${2:-100}
REPEAT=${3:-5}
ENV_TYPE=${4:-"auto"}  # 환경 타입: docker, local, auto(자동감지)
COLD_START=${5:-"false"}  # 콜드 스타트 측정 여부: true, false (기본값: false)
WARMUP=${6:-"true"}  # 성능 측정 전 웜업 실행 여부 (기본값: true)

# 환경 타입 자동 감지
if [ "$ENV_TYPE" = "auto" ]; then
    if command -v docker &> /dev/null && docker ps | grep -q ollama; then
        ENV_TYPE="docker"
        echo "Docker 환경이 감지되었습니다."
    else
        ENV_TYPE="local"
        echo "로컬 환경으로 설정합니다."
    fi
fi

# Docker 컨테이너 확인 (Docker 환경인 경우)
if [ "$ENV_TYPE" = "docker" ]; then
    DOCKER_CONTAINER=$(docker ps | grep ollama | awk '{print $1}')
    if [ -z "$DOCKER_CONTAINER" ]; then
        echo "경고: Ollama Docker 컨테이너를 찾을 수 없습니다. 로컬 환경으로 전환합니다."
        ENV_TYPE="local"
    else
        echo "Ollama Docker 컨테이너 ID: $DOCKER_CONTAINER"
    fi
fi

# ollama 커맨드 확인 (로컬 환경인 경우)
if [ "$ENV_TYPE" = "local" ]; then
    if ! command -v ollama &> /dev/null; then
        echo "오류: ollama 커맨드를 찾을 수 없습니다. ollama가 설치되어 있는지 확인하세요."
        exit 1
    fi
fi

# 랜덤 프롬프트 생성 함수 (Mac OS 호환성 개선)
generate_prompt() {
    local length=$1
    local chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    local result=""

    # Mac OS 호환 방식으로 랜덤 문자열 생성
    for i in $(seq 1 $length); do
        local rand=$((RANDOM % ${#chars}))
        result="${result}${chars:$rand:1}"
    done

    echo "$result"
}

# 시스템 정보 수집 (Mac OS 호환성 개선)
echo "==== 시스템 정보 ===="
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

# Docker 컨테이너 정보 수집 (Docker 환경인 경우)
if [ "$ENV_TYPE" = "docker" ]; then
    echo "==== Docker 컨테이너 정보 ===="
    docker inspect "$DOCKER_CONTAINER" | jq '.[] | {Name: .Name, Image: .Config.Image, CPUs: .HostConfig.NanoCpus, Memory: .HostConfig.Memory}' || echo "Docker 정보를 가져올 수 없습니다."
fi

# 결과 저장 디렉토리 생성
RESULT_DIR="ollama_benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# Ollama 모델 정보 수집 및 모델 목록 가져오기
echo "==== Ollama 모델 정보 ===="

# 환경에 따라 명령 실행
if [ "$ENV_TYPE" = "docker" ]; then
    docker exec "$DOCKER_CONTAINER" ollama list

    # 모델 목록 추출
    if [ "$MODEL" = "all" ]; then
        # 모든 모델을 테스트할 경우 모델 목록 추출
        MODEL_LIST=$(docker exec "$DOCKER_CONTAINER" ollama list | tail -n +2 | awk '{print $1}')
        echo "발견된 모델 목록: $MODEL_LIST"
    else
        # 단일 모델만 테스트
        MODEL_LIST="$MODEL"
    fi
else
    ollama list

    # 모델 목록 추출
    if [ "$MODEL" = "all" ]; then
        # 모든 모델을 테스트할 경우 모델 목록 추출
        MODEL_LIST=$(ollama list | tail -n +2 | awk '{print $1}')
        echo "발견된 모델 목록: $MODEL_LIST"
    else
        # 단일 모델만 테스트
        MODEL_LIST="$MODEL"
    fi
fi

# 측정 결과 저장 파일
RESULT_FILE="$RESULT_DIR/benchmark_results.csv"
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
        WARMUP_PROMPT=$(generate_prompt 50)

        # 웜업을 위한 실행 (결과는 저장하지 않음)
        if [ "$ENV_TYPE" = "docker" ]; then
            docker exec -i "$DOCKER_CONTAINER" ollama run $CURRENT_MODEL "$WARMUP_PROMPT" > /dev/null 2>&1
        else
            ollama run $CURRENT_MODEL "$WARMUP_PROMPT" > /dev/null 2>&1
        fi

        echo "웜업 완료. 성능 측정 시작..."
        # 잠시 대기 (리소스 안정화를 위해)
        sleep 2
    fi

    for i in $(seq 1 $REPEAT); do
        echo "$CURRENT_MODEL 모델 테스트 $i/$REPEAT 실행 중..."

        # 프롬프트 생성
        PROMPT=$(generate_prompt $PROMPT_LENGTH)

        # 콜드 스타트 측정을 위해 모델 언로드 (선택적)
        if [ "$COLD_START" = "true" ]; then
            echo "콜드 스타트 측정을 위해 모델 언로드 중..."
            if [ "$ENV_TYPE" = "docker" ]; then
                docker exec "$DOCKER_CONTAINER" ollama rm -f $CURRENT_MODEL 2>/dev/null || true
                docker exec "$DOCKER_CONTAINER" ollama pull $CURRENT_MODEL >/dev/null 2>&1
            else
                ollama rm -f $CURRENT_MODEL 2>/dev/null || true
                ollama pull $CURRENT_MODEL >/dev/null 2>&1
            fi
            START_TYPE="cold"

            # 콜드 스타트 후 웜업이 필요한 경우 다시 진행
            if [ "$WARMUP" = "true" ] && [ "$i" -gt 1 ]; then
                echo "모델 재웜업 실행 중..."
                WARMUP_PROMPT=$(generate_prompt 50)

                if [ "$ENV_TYPE" = "docker" ]; then
                    docker exec -i "$DOCKER_CONTAINER" ollama run $CURRENT_MODEL "$WARMUP_PROMPT" > /dev/null 2>&1
                else
                    ollama run $CURRENT_MODEL "$WARMUP_PROMPT" > /dev/null 2>&1
                fi

                sleep 2
            fi
        else
            # 첫 번째 실행은 콜드 스타트로 간주, 이후는 웜 스타트
            if [ "$i" -eq 1 ]; then
                START_TYPE="cold"
            else
                START_TYPE="warm"
            fi
        fi

        # 컨테이너 스탯 모니터링 시작 (Docker 환경인 경우)
        if [ "$ENV_TYPE" = "docker" ]; then
            docker stats "$DOCKER_CONTAINER" --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" > "$MODEL_RESULT_DIR/stats_$i.txt" &
            STATS_PID=$!
        else
            # 로컬 환경에서는 OS에 맞는 방식으로 CPU/메모리 사용량 모니터링
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # Mac OS용
                ps -p $(pgrep ollama) -o %cpu,%mem > "$MODEL_RESULT_DIR/stats_$i.txt" 2>/dev/null || echo "0,0" > "$MODEL_RESULT_DIR/stats_$i.txt" &
            else
                # Linux용
                ps -p $(pgrep -f "ollama") -o %cpu,%mem > "$MODEL_RESULT_DIR/stats_$i.txt" 2>/dev/null || echo "0,0" > "$MODEL_RESULT_DIR/stats_$i.txt" &
            fi
            STATS_PID=$!
        fi

        # GPU 상태 모니터링 (NVIDIA GPU가 있는 경우)
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_before_$i.txt"
        fi

        # 모델 로드 시간 및 추론 측정
        START_TIME=$(date +%s.%N)
        OLLAMA_OUTPUT_FILE="$MODEL_RESULT_DIR/ollama_output_$i.txt"

        # Docker 또는 로컬 환경에서 Ollama 실행하고 결과 수집
        if [ "$ENV_TYPE" = "docker" ]; then
            docker exec -i "$DOCKER_CONTAINER" bash -c "time ollama run $CURRENT_MODEL \"$PROMPT\" --verbose" > "$OLLAMA_OUTPUT_FILE" 2>&1
        else
            time ollama run $CURRENT_MODEL "$PROMPT" --verbose > "$OLLAMA_OUTPUT_FILE" 2>&1
        fi

        END_TIME=$(date +%s.%N)
        TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

        # GPU 사용 후 상태 확인
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_after_$i.txt"
            GPU_MEM_USED=$(cat "$MODEL_RESULT_DIR/gpu_mem_after_$i.txt" | head -1)
        else
            GPU_MEM_USED="N/A"
        fi

        # Docker 스탯 모니터링 중지
        kill $STATS_PID 2>/dev/null || true

        # 결과 파싱
        if [ -f "$OLLAMA_OUTPUT_FILE" ]; then
            # 첫 토큰 생성 시간 추출 (로그에서 패턴 찾기)
            LOAD_TIME=$(grep -o "load [0-9.]*ms" "$OLLAMA_OUTPUT_FILE" | awk '{print $2}' | sed 's/ms//' | awk '{print $1/1000}' || echo "N/A")
            FIRST_TOKEN_TIME=$(grep -o "first token [0-9.]*ms" "$OLLAMA_OUTPUT_FILE" | awk '{print $3}' | sed 's/ms//' | awk '{print $1/1000}' || echo "N/A")

            # 생성된 토큰 수 계산
            TOKENS_GENERATED=$(grep -o "eval [0-9]* tokens" "$OLLAMA_OUTPUT_FILE" | awk '{print $2}' | sort -n | tail -1 || echo "N/A")

            # 토큰 생성 속도 계산
            if [ "$TOKENS_GENERATED" != "N/A" ] && [ "$TOTAL_TIME" != "N/A" ]; then
                TOKENS_PER_SEC=$(echo "scale=2; $TOKENS_GENERATED / $TOTAL_TIME" | bc)
            else
                TOKENS_PER_SEC="N/A"
            fi

            # CPU 및 메모리 사용량 파싱
            if [ -f "$MODEL_RESULT_DIR/stats_$i.txt" ]; then
                if [ "$ENV_TYPE" = "docker" ]; then
                    CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | cut -d',' -f1 | sed 's/%//')
                    MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | cut -d',' -f2 | grep -o "[0-9.]*MiB" | sed 's/MiB//' || echo "N/A")
                else
                    # 로컬 환경에서의 CPU/메모리 사용량 파싱 (OS 호환성 개선)
                    if [[ "$OSTYPE" == "darwin"* ]]; then
                        # Mac OS
                        CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $1}' || echo "N/A")
                        MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $2}' || echo "N/A")
                    else
                        # Linux
                        CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $1}' || echo "N/A")
                        MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | tail -1 | awk '{print $2}' || echo "N/A")
                    fi
                fi
            else
                CPU_USAGE="N/A"
                MEM_USAGE="N/A"
            fi

            # 결과 기록
            echo "$CURRENT_MODEL,$i,$PROMPT_LENGTH,$START_TYPE,$LOAD_TIME,$FIRST_TOKEN_TIME,$TOTAL_TIME,$TOKENS_GENERATED,$TOKENS_PER_SEC,$CPU_USAGE,$MEM_USAGE,$GPU_MEM_USED" >> "$RESULT_FILE"
        else
            echo "오류: Ollama 출력 파일을 찾을 수 없습니다."
        fi

        # 잠시 대기
        sleep 3
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

# HTML 보고서에 오류 처리 추가
# 대체 GNUPLOT을 사용할 수 없는 경우 메시지 표시
if ! command -v gnuplot &> /dev/null; then
    GNUPLOT_MESSAGE="<p class='note'>참고: gnuplot이 설치되어 있지 않아 그래프가 생성되지 않았습니다.</p>"
else
    GNUPLOT_MESSAGE=""
fi

# 모델 비교 표 생성
echo "모델별 평균 성능 비교:"
echo "------------------------------------------------------"
echo "모델명 | 로드유형 | 로드시간(초) | 첫토큰(초) | 총시간(초) | 토큰속도(t/s)"
echo "------------------------------------------------------"

# 모델별 콜드 스타트 평균 계산하여 표시
for CURRENT_MODEL in $MODEL_LIST; do
    COLD_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '$1==model && $4=="cold" {load_sum+=$5; first_token_sum+=$6; total_time_sum+=$7; tokens_sum+=$8; token_rate_sum+=$9; count++} END {if(count>0) printf "%s | 콜드 | %.2f | %.2f | %.2f | %.2f", model, load_sum/count, first_token_sum/count, total_time_sum/count, token_rate_sum/count; else printf "%s | 콜드 | N/A | N/A | N/A | N/A", model}' "$RESULT_FILE")
    echo "$COLD_AVERAGES"
done

# 모델별 웜 스타트 평균 계산하여 표시
for CURRENT_MODEL in $MODEL_LIST; do
    WARM_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '$1==model && $4=="warm" {load_sum+=$5; first_token_sum+=$6; total_time_sum+=$7; tokens_sum+=$8; token_rate_sum+=$9; count++} END {if(count>0) printf "%s | 웜 | %.2f | %.2f | %.2f | %.2f", model, load_sum/count, first_token_sum/count, total_time_sum/count, token_rate_sum/count; else printf "%s | 웜 | N/A | N/A | N/A | N/A", model}' "$RESULT_FILE")
    echo "$WARM_AVERAGES"
done
echo "------------------------------------------------------"

# 결과 시각화 (옵션)
if command -v gnuplot &> /dev/null; then
    echo "결과 그래프 생성 중..."

    # 콜드 스타트 vs 웜 스타트 비교 그래프
    cat > "$RESULT_DIR/cold_warm_comparison.gnu" << EOF
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/cold_warm_comparison.png'
set title "모델별 콜드 스타트 vs 웜 스타트 로딩 시간 비교"
set xlabel "모델"
set ylabel "로드 시간 (초)"
set yrange [0:*]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
set grid y
set key top left
set datafile separator ","

# 데이터 파일 생성
system("awk -F, '\$4==\"cold\" {print \$1,\$5}' $RESULT_FILE | sort | uniq -f 1 > $RESULT_DIR/cold_start.dat")
system("awk -F, '\$4==\"warm\" {print \$1,\$5}' $RESULT_FILE | sort | uniq -f 1 > $RESULT_DIR/warm_start.dat")

plot '$RESULT_DIR/cold_start.dat' using 2:xtic(1) title 'Cold Start' lc rgb '#ff0000', \
     '$RESULT_DIR/warm_start.dat' using 2:xtic(1) title 'Warm Start' lc rgb '#0000ff'
EOF
    gnuplot "$RESULT_DIR/cold_warm_comparison.gnu" || echo "콜드/웜 스타트 비교 그래프 생성 실패"

    # 첫 토큰 생성 시간 비교 그래프
    cat > "$RESULT_DIR/first_token_comparison.gnu" << EOF
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/first_token_comparison.png'
set title "모델별 콜드 스타트 vs 웜 스타트 첫 토큰 생성 시간 비교"
set xlabel "모델"
set ylabel "첫 토큰 생성 시간 (초)"
set yrange [0:*]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
set grid y
set key top left
set datafile separator ","

# 데이터 파일 생성
system("awk -F, '\$4==\"cold\" {print \$1,\$6}' $RESULT_FILE | sort | uniq -f 1 > $RESULT_DIR/cold_token.dat")
system("awk -F, '\$4==\"warm\" {print \$1,\$6}' $RESULT_FILE | sort | uniq -f 1 > $RESULT_DIR/warm_token.dat")

plot '$RESULT_DIR/cold_token.dat' using 2:xtic(1) title 'Cold Start' lc rgb '#ff0000', \
     '$RESULT_DIR/warm_token.dat' using 2:xtic(1) title 'Warm Start' lc rgb '#0000ff'
EOF
    gnuplot "$RESULT_DIR/first_token_comparison.gnu" || echo "첫 토큰 생성 시간 비교 그래프 생성 실패"

    # 기존 그래프도 생성
    cat > "$RESULT_DIR/plot_speed.gnu" << EOF
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/token_speed_comparison.png'
set title "모델별 토큰 생성 속도 비교 (tokens/sec)"
set xlabel "테스트 번호"
set ylabel "토큰/초"
set grid
set key outside
set datafile separator ","

# 막대 그래프용 스크립트 (모델별 평균 속도)
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/model_comparison_bar.png'
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
set title "모델별 평균 토큰 생성 속도"
set ylabel "토큰/초"
set grid y
set auto x
plot '$RESULT_FILE' using 9:xtic(1) title "Token Speed (tokens/sec)" group by 1
EOF
    gnuplot "$RESULT_DIR/plot_speed.gnu" || echo "토큰 속도 비교 그래프 생성 실패"

    echo "그래프가 $RESULT_DIR 디렉토리에 생성되었습니다."
fi

# 모델별 여러 지표 비교 차트 (가능한 경우)
if command -v python3 &> /dev/null; then
    cat > "$RESULT_DIR/comparison_chart.py" << EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 결과 파일 읽기
df = pd.read_csv('$RESULT_FILE')

# 콜드 스타트와 웜 스타트 분리하여 모델별 평균 계산
df_cold = df[df['시작타입'] == 'cold']
df_warm = df[df['시작타입'] == 'warm']

# 모델별 평균 계산
cold_stats = df_cold.groupby('모델').mean()
warm_stats = df_warm.groupby('모델').mean()

# 콜드 스타트와 웜 스타트 비교 차트 생성
plt.figure(figsize=(15, 10))

# 스타일 설정
plt.style.use('ggplot')
bar_width = 0.35
index = np.arange(len(cold_stats.index))

# 콜드 스타트 vs 웜 스타트 막대 그래프
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('콜드 스타트 vs 웜 스타트 성능 비교', fontsize=16)

# 모델 로드 시간
axes[0, 0].bar(index, cold_stats['모델로드시간(초)'], bar_width, label='Cold Start', color='red')
if not warm_stats.empty:
    axes[0, 0].bar(index + bar_width, warm_stats['모델로드시간(초)'], bar_width, label='Warm Start', color='blue')
axes[0, 0].set_xlabel('모델')
axes[0, 0].set_ylabel('로드 시간 (초)')
axes[0, 0].set_title('모델 로드 시간 비교')
axes[0, 0].set_xticks(index + bar_width / 2)
axes[0, 0].set_xticklabels(cold_stats.index, rotation=45)
axes[0, 0].legend()

# 첫 토큰 생성 시간
axes[0, 1].bar(index, cold_stats['첫토큰생성시간(초)'], bar_width, label='Cold Start', color='red')
if not warm_stats.empty:
    axes[0, 1].bar(index + bar_width, warm_stats['첫토큰생성시간(초)'], bar_width, label='Warm Start', color='blue')
axes[0, 1].set_xlabel('모델')
axes[0, 1].set_ylabel('첫 토큰 생성 시간 (초)')
axes[0, 1].set_title('첫 토큰 생성 시간 비교')
axes[0, 1].set_xticks(index + bar_width / 2)
axes[0, 1].set_xticklabels(cold_stats.index, rotation=45)
axes[0, 1].legend()

# 총 생성 시간
axes[1, 0].bar(index, cold_stats['총생성시간(초)'], bar_width, label='Cold Start', color='red')
if not warm_stats.empty:
    axes[1, 0].bar(index + bar_width, warm_stats['총생성시간(초)'], bar_width, label='Warm Start', color='blue')
axes[1, 0].set_xlabel('모델')
axes[1, 0].set_ylabel('총 생성 시간 (초)')
axes[1, 0].set_title('총 생성 시간 비교')
axes[1, 0].set_xticks(index + bar_width / 2)
axes[1, 0].set_xticklabels(cold_stats.index, rotation=45)
axes[1, 0].legend()

# 토큰 생성 속도
axes[1, 1].bar(index, cold_stats['토큰생성속도(tokens/sec)'], bar_width, label='Cold Start', color='red')
if not warm_stats.empty:
    axes[1, 1].bar(index + bar_width, warm_stats['토큰생성속도(tokens/sec)'], bar_width, label='Warm Start', color='blue')
axes[1, 1].set_xlabel('모델')
axes[1, 1].set_ylabel('토큰 생성 속도 (tokens/sec)')
axes[1, 1].set_title('토큰 생성 속도 비교')
axes[1, 1].set_xticks(index + bar_width / 2)
axes[1, 1].set_xticklabels(cold_stats.index, rotation=45)
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('$RESULT_DIR/cold_warm_comparison_chart.png')

# 또한 레이더 차트도 생성
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, polar=True)

# 필요한 지표만 선택
metrics = ['모델로드시간(초)', '첫토큰생성시간(초)', '총생성시간(초)', '토큰생성속도(tokens/sec)']

# 각도 설정
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 원형으로 닫기

# 모델별로 그래프 그리기
for model in cold_stats.index:
    if model in warm_stats.index:
        # 콜드 스타트
        values_cold = cold_stats.loc[model][metrics].values.flatten().tolist()
        values_cold += values_cold[:1]  # 원형으로 닫기
        ax.plot(angles, values_cold, linewidth=1, label=f'{model} (Cold)', linestyle='-')
        ax.fill(angles, values_cold, alpha=0.1)

        # 웜 스타트
        values_warm = warm_stats.loc[model][metrics].values.flatten().tolist()
        values_warm += values_warm[:1]  # 원형으로 닫기
        ax.plot(angles, values_warm, linewidth=1, label=f'{model} (Warm)', linestyle='--')
        ax.fill(angles, values_warm, alpha=0.1)

# 차트 설정
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_title('모델별 콜드/웜 스타트 성능 비교 (레이더 차트)')
ax.legend(loc='upper right')

plt.savefig('$RESULT_DIR/cold_warm_radar_chart.png')