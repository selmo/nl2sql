#!/bin/bash
# run_nl2sql_models.sh
# NL2SQL 모델 조합 배치 실행을 위한 단순 스크립트

# 기본 설정
OLLAMA_URL="172.16.15.112"
PREFIX="OUTPUT"
BATCH_SIZE=50
MAX_CONCURRENT=20
MAX_RETRIES=10
MODE="ollama-api"

# 사용법 출력
function show_usage {
  echo "사용법: $0 [base_models] [verifying_models]"
  echo "예시: $0 \"sqlcoder:70b llama3:70b\" \"gemma3:27b gemma3:8b\""
  echo ""
  echo "각 모델 이름은 공백으로 구분합니다."
  exit 1
}

# 인수 확인
if [ $# -ne 2 ]; then
  show_usage
fi

# 모델 목록 설정
BASE_MODELS=($1)
VERIFYING_MODELS=($2)

# 모델 목록 확인
if [ ${#BASE_MODELS[@]} -eq 0 ] || [ ${#VERIFYING_MODELS[@]} -eq 0 ]; then
  echo "오류: 모델 목록이 비어 있습니다."
  show_usage
fi

echo "===== NL2SQL 배치 실행 시작 ====="
echo "실행 시간: $(date)"
echo "Base 모델: ${BASE_MODELS[@]}"
echo "Verifying 모델: ${VERIFYING_MODELS[@]}"
echo "결과 저장 경로: $PREFIX"

# 디렉토리 생성
mkdir -p "$PREFIX/logs"

# 각 모델 조합 실행
total_combinations=$((${#BASE_MODELS[@]} * ${#VERIFYING_MODELS[@]}))
current=1

for base_model in "${BASE_MODELS[@]}"; do
  for verifying_model in "${VERIFYING_MODELS[@]}"; do
    echo ""
    echo "[$current/$total_combinations] 실행: $base_model → $verifying_model"

    # 실행 명령
    cmd="python run_evaluation.py $MODE \
      --ollama-url $OLLAMA_URL \
      --base-model $base_model \
      --verifying-model $verifying_model \
      --prefix $PREFIX \
      --batch-size $BATCH_SIZE \
      --max-concurrent $MAX_CONCURRENT \
      --max-retries $MAX_RETRIES"

    echo "명령: $cmd"

    # 실행
    start_time=$(date +%s)
    eval $cmd
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # 결과 출력
    if [ $exit_code -eq 0 ]; then
      echo "완료: 성공 (${duration}초 소요)"
    else
      echo "완료: 실패 (종료 코드: $exit_code)"
    fi

    # 잠시 대기
    sleep 3

    current=$((current + 1))
  done
done

echo ""
echo "===== 배치 실행 완료 ====="
echo "완료 시간: $(date)"
echo "결과 저장 경로: $PREFIX/stats/"

# 통계 파일 확인
stats_file="$PREFIX/stats/nl2sql_verification_stats.csv"
if [ -f "$stats_file" ]; then
  echo "통계 파일: $stats_file"

  # 결과 요약 (간단하게 파일만 표시)
  echo "결과 파일을 확인하세요. 예: cat $stats_file"
else
  echo "통계 파일을 찾을 수 없습니다."
fi

exit 0