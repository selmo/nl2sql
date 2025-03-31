#!/bin/bash

# Ollama 성능 측정 자동화 스크립트 (Docker 환경용)
# 사용법: ./ollama_benchmark.sh [모델명] [프롬프트 길이] [테스트 반복 횟수]

set -e

# 기본 설정값
MODEL=${1:-"all"}  # 기본값을 "all"로 변경
PROMPT_LENGTH=${2:-100}
REPEAT=${3:-5}
DOCKER_CONTAINER=$(docker ps | grep ollama | awk '{print $1}')

if [ -z "$DOCKER_CONTAINER" ]; then
    echo "Error: Ollama Docker 컨테이너를 찾을 수 없습니다."
    exit 1
fi

# 시스템 정보 수집
echo "==== 시스템 정보 ===="
echo "운영체제: $(uname -a)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "코어 수: $(nproc)"
echo "메모리: $(free -h | grep Mem | awk '{print $2}')"

# GPU 정보 수집 (NVIDIA GPU가 있는 경우)
if command -v nvidia-smi &> /dev/null; then
    echo "==== GPU 정보 ===="
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# 랜덤 프롬프트 생성 함수
generate_prompt() {
    local length=$1
    cat /dev/urandom | tr -dc 'a-zA-Z0-9 ' | fold -w "$length" | head -n 1
}

# 결과 저장 디렉토리 생성
RESULT_DIR="ollama_benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# Docker 컨테이너 정보 수집
echo "==== Docker 컨테이너 정보 ===="
docker inspect "$DOCKER_CONTAINER" | jq '.[] | {Name: .Name, Image: .Config.Image, CPUs: .HostConfig.NanoCpus, Memory: .HostConfig.Memory}'

# Ollama 모델 정보 수집 및 모델 목록 가져오기
echo "==== Ollama 모델 정보 ===="
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

# 측정 결과 저장 파일
RESULT_FILE="$RESULT_DIR/benchmark_results.csv"
echo "모델,테스트번호,프롬프트길이,모델로드시간(초),첫토큰생성시간(초),총생성시간(초),생성된토큰수,토큰생성속도(tokens/sec),CPU사용률(%),메모리사용량(MB),GPU메모리사용량(MB)" > "$RESULT_FILE"

echo "==== 벤치마크 시작 (모델당 $REPEAT 회 반복) ===="

# 모든 모델에 대해 반복
for CURRENT_MODEL in $MODEL_LIST; do
    echo "===== $CURRENT_MODEL 모델 벤치마크 시작 ====="

    # 모델별 결과 디렉토리 생성
    MODEL_RESULT_DIR="$RESULT_DIR/$CURRENT_MODEL"
    mkdir -p "$MODEL_RESULT_DIR"

    for i in $(seq 1 $REPEAT); do
        echo "$CURRENT_MODEL 모델 테스트 $i/$REPEAT 실행 중..."
        PROMPT=$(generate_prompt $PROMPT_LENGTH)

        # 컨테이너 스탯 모니터링 시작
        docker stats "$DOCKER_CONTAINER" --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" > "$MODEL_RESULT_DIR/stats_$i.txt" &
        STATS_PID=$!

        # GPU 상태 모니터링 (NVIDIA GPU가 있는 경우)
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_before_$i.txt"
        fi

        # 모델 로드 시간 및 추론 측정
        START_TIME=$(date +%s.%N)
        OLLAMA_OUTPUT_FILE="$MODEL_RESULT_DIR/ollama_output_$i.txt"

        # Docker를 통해 Ollama 실행하고 결과 수집
        docker exec -i "$DOCKER_CONTAINER" bash -c "time ollama run $CURRENT_MODEL \"$PROMPT\" --verbose" > "$OLLAMA_OUTPUT_FILE" 2>&1

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
            CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | cut -d',' -f1 | sed 's/%//')
            MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | cut -d',' -f2 | grep -o "[0-9.]*MiB" | sed 's/MiB//')
        else
            CPU_USAGE="N/A"
            MEM_USAGE="N/A"
        fi

        # 결과 기록
        echo "$CURRENT_MODEL,$i,$PROMPT_LENGTH,$LOAD_TIME,$FIRST_TOKEN_TIME,$TOTAL_TIME,$TOKENS_GENERATED,$TOKENS_PER_SEC,$CPU_USAGE,$MEM_USAGE,$GPU_MEM_USED" >> "$RESULT_FILE"
    else
        echo "오류: Ollama 출력 파일을 찾을 수 없습니다."
    fi

    # 잠시 대기
    sleep 3
    done

    # 모델별 결과 요약
    echo "==== $CURRENT_MODEL 모델 벤치마크 완료 ===="
    echo "$CURRENT_MODEL 모델의 평균 성능 지표:"
    awk -F, -v model="$CURRENT_MODEL" '$1==model {load_sum+=$4; first_token_sum+=$5; total_time_sum+=$6; tokens_sum+=$7; token_rate_sum+=$8; count++} END {printf "모델 로드 시간: %.2f초\n첫 토큰 생성 시간: %.2f초\n총 생성 시간: %.2f초\n평균 생성 토큰 수: %.2f\n평균 토큰 생성 속도: %.2f tokens/sec\n", load_sum/count, first_token_sum/count, total_time_sum/count, tokens_sum/count, token_rate_sum/count}' "$RESULT_FILE"
    echo "----------------------------------------"
done

# 결과 요약
echo "==== 벤치마크 결과 요약 ===="
echo "결과는 $RESULT_FILE 에 저장되었습니다."

# 모델 비교 표 생성
echo "모델별 평균 성능 비교:"
echo "------------------------------------------------------"
echo "모델명 | 로드시간(초) | 첫토큰(초) | 총시간(초) | 토큰속도(t/s)"
echo "------------------------------------------------------"

# 모델별 평균 계산하여 표시
for CURRENT_MODEL in $MODEL_LIST; do
    MODEL_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '$1==model {load_sum+=$4; first_token_sum+=$5; total_time_sum+=$6; tokens_sum+=$7; token_rate_sum+=$8; count++} END {printf "%s | %.2f | %.2f | %.2f | %.2f", model, load_sum/count, first_token_sum/count, total_time_sum/count, token_rate_sum/count}' "$RESULT_FILE")
    echo "$MODEL_AVERAGES"
done
echo "------------------------------------------------------"

# 결과 시각화 (옵션)
if command -v gnuplot &> /dev/null; then
    echo "결과 그래프 생성 중..."

    # 모델 목록으로 플롯 명령 생성
    PLOT_CMD=""
    for CURRENT_MODEL in $MODEL_LIST; do
        if [ -z "$PLOT_CMD" ]; then
            PLOT_CMD="plot '$RESULT_FILE' using 2:8 with linespoints title '$CURRENT_MODEL' smooth bezier"
        else
            PLOT_CMD="$PLOT_CMD, '$RESULT_FILE' using 2:8:(stringcolumn(1) eq '$CURRENT_MODEL' ? 1 : 0) smooth bezier with linespoints title '$CURRENT_MODEL'"
        fi
    done

    # 토큰 생성 속도 그래프
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
plot '$RESULT_FILE' using 8:xtic(1) title "Token Speed (tokens/sec)" group by 1
EOF
    gnuplot "$RESULT_DIR/plot_speed.gnu"

    # 모델별 여러 지표 비교 레이더 차트 (가능한 경우)
    if command -v python3 &> /dev/null; then
        cat > "$RESULT_DIR/radar_chart.py" << EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 결과 파일 읽기
df = pd.read_csv('$RESULT_FILE')

# 모델별 평균 계산
model_stats = df.groupby('모델').mean()

# 필요한 지표만 선택
metrics = ['모델로드시간(초)', '첫토큰생성시간(초)', '총생성시간(초)', '토큰생성속도(tokens/sec)']
stats = model_stats[metrics]

# 레이더 차트 그리기
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# 각도 설정
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 원형으로 닫기

# 각 모델별로 그리기
for model in stats.index:
    values = stats.loc[model].values.flatten().tolist()
    values += values[:1]  # 원형으로 닫기
    ax.plot(angles, values, linewidth=1, label=model)
    ax.fill(angles, values, alpha=0.1)

# 차트 설정
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_title('모델별 성능 비교')
ax.legend(loc='upper right')

plt.savefig('$RESULT_DIR/model_radar_chart.png')
EOF
        python3 "$RESULT_DIR/radar_chart.py"
    fi

    echo "그래프가 $RESULT_DIR 디렉토리에 생성되었습니다."
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
        <p>CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)</p>
        <p>코어 수: $(nproc)</p>
        <p>메모리: $(free -h | grep Mem | awk '{print $2}')</p>
        $(if command -v nvidia-smi &> /dev/null; then echo "<p>GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)</p>"; fi)
    </div>

    <h2>모델별 성능 비교</h2>

    <table>
        <tr>
            <th>모델명</th>
            <th>모델 로드 시간(초)</th>
            <th>첫 토큰 생성 시간(초)</th>
            <th>총 생성 시간(초)</th>
            <th>평균 토큰 수</th>
            <th>토큰 생성 속도(tokens/sec)</th>
        </tr>
EOF

# 모델별 평균 데이터를 HTML 테이블에 추가
for CURRENT_MODEL in $MODEL_LIST; do
    MODEL_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '
    $1==model {
        load_sum+=$4;
        first_token_sum+=$5;
        total_time_sum+=$6;
        tokens_sum+=$7;
        token_rate_sum+=$8;
        count++
    }
    END {
        printf "<tr><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>",
        model,
        load_sum/count,
        first_token_sum/count,
        total_time_sum/count,
        tokens_sum/count,
        token_rate_sum/count
    }' "$RESULT_FILE")
    echo "$MODEL_AVERAGES" >> "$HTML_REPORT"
done

# HTML 마무리
cat >> "$HTML_REPORT" << EOF
    </table>

    <div class="chart-container">
        <h2>시각화</h2>
        $(if [ -f "$RESULT_DIR/token_speed_comparison.png" ]; then echo "<img src=\"token_speed_comparison.png\" alt=\"토큰 생성 속도 비교\" style=\"max-width:100%;\">"; fi)
        $(if [ -f "$RESULT_DIR/model_comparison_bar.png" ]; then echo "<img src=\"model_comparison_bar.png\" alt=\"모델별 평균 속도\" style=\"max-width:100%;\">"; fi)
        $(if [ -f "$RESULT_DIR/model_radar_chart.png" ]; then echo "<img src=\"model_radar_chart.png\" alt=\"모델별 성능 비교\" style=\"max-width:100%;\">"; fi)
    </div>

    <div class="note">
        <p>참고: 테스트는 각 모델당 $REPEAT회 반복되었으며, 프롬프트 길이는 $PROMPT_LENGTH 자로 설정되었습니다.</p>
        <p>생성 시간: $(date)</p>
    </div>
</body>
</html>
EOF

echo "HTML 보고서가 $HTML_REPORT 에 생성되었습니다."
echo "벤치마크 완료! 결과는 $RESULT_DIR 디렉토리에서 확인할 수 있습니다."