#!/bin/bash
# run_nl2sql.sh
# NL2SQL 모델 조합 배치 실행을 위한 단순 스크립트

# 기본 설정
OLLAMA_URL = "172.16.15.112"
PREFIX = "OUTPUT"
BATCH_SIZE = 50
MAX_CONCURRENT = 20
MAX_RETRIES = 10
MODE = "ollama-api"
LOG_DIR = "${PREFIX}/logs"

# 사용법 출력
function
show_usage
{
    echo
"사용법: $0 [base_models] [verifying_models]"
echo
"예시: $0 \"sqlcoder:70b llama3:70b\" \"gemma3:27b gemma3:8b\""
echo
""
echo
"각 모델 이름은 공백으로 구분합니다."
exit
1
}

# 로그 디렉토리 생성 함수
function
setup_logging
{
    mkdir - p
"${LOG_DIR}"
LOG_FILE = "${LOG_DIR}/run_nl2sql_$(date +%Y%m%d_%H%M%S).log"

# 로그 파일 생성 및 초기 메시지
echo
"=============== NL2SQL 배치 실행 로그 ===============" > "${LOG_FILE}"
echo
"시작 시간: $(date)" >> "${LOG_FILE}"
echo
"설정: OLLAMA_URL=${OLLAMA_URL}, PREFIX=${PREFIX}" >> "${LOG_FILE}"
echo
"BASE_MODELS: ${BASE_MODELS[@]}" >> "${LOG_FILE}"
echo
"VERIFYING_MODELS: ${VERIFYING_MODELS[@]}" >> "${LOG_FILE}"
echo
"===================================================" >> "${LOG_FILE}"

# 로그 정보 출력
echo
"로그 파일이 생성되었습니다: ${LOG_FILE}"
}

# 로그 출력 함수 (콘솔과 파일에 모두 출력)
function
log
{
    echo
"[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee - a
"${LOG_FILE}"
}

# 인수 확인
if [ $  # -ne 2 ]; then
show_usage
fi

# 모델 목록 설정
BASE_MODELS=($1)
VERIFYING_MODELS=($2)

# 모델 목록 확인
if[${  # BASE_MODELS[@]} -eq 0 ] || [ ${#VERIFYING_MODELS[@]} -eq 0 ]; then
echo "오류: 모델 목록이 비어 있습니다."
show_usage
fi

# 로깅 설정
setup_logging

log "===== NL2SQL 배치 실행 시작 ====="
log "Base 모델: ${BASE_MODELS[@]}"
log "Verifying 모델: ${VERIFYING_MODELS[@]}"
log "결과 저장 경로: $PREFIX"

# 디렉토리 생성
mkdir -p "$PREFIX/logs"

# 각 모델 조합 실행
total_combinations=$((${  # BASE_MODELS[@]} * ${#VERIFYING_MODELS[@]}))
current=1

for base_model in "${BASE_MODELS[@]}"; do
for verifying_model in "${VERIFYING_MODELS[@]}"; do
log ""
log "[$current/$total_combinations] 실행: $base_model → $verifying_model"

# 실행 명령
cmd="python run_evaluation.py $MODE \
      --ollama-url $OLLAMA_URL \
      --base-model $base_model \
      --verifying-model $verifying_model \
      --prefix $PREFIX \
      --batch-size $BATCH_SIZE \
      --max-concurrent $MAX_CONCURRENT \
      --max-retries $MAX_RETRIES"

log "명령: $cmd"

# 실행 (출력을 로그 파일에도 저장)
start_time=$(date + %s)
log "실행 시작..."
eval $cmd 2 > & 1 | tee -a "${LOG_FILE}"
exit_code=${PIPESTATUS[0]}
end_time=$(date + %s)
duration=$((end_time - start_time))

# 결과 출력
if[$exit_code -eq 0]; then
log "완료: 성공 (${duration}초 소요)"
else
log "완료: 실패 (종료 코드: $exit_code, ${duration}초 소요)"
fi

# 잠시 대기
log "다음 모델 조합으로 진행하기 전 3초 대기..."
sleep 3

current=$((current + 1))
done
done

log ""
log "===== 배치 실행 완료 ====="
log "완료 시간: $(date)"
log "결과 저장 경로: $PREFIX/stats/"

# 통계 파일 확인
stats_file="$PREFIX/stats/nl2sql_verification_stats.csv"
if[-f "$stats_file"]; then
log "통계 파일: $stats_file"

# 결과 요약 보여주기 (간단한 통계)
log "통계 파일 내용 미리보기:"
head -n 1 "$stats_file" | tee -a "${LOG_FILE}"  # 헤더
tail -n 5 "$stats_file" | tee -a "${LOG_FILE}"  # 마지막 5개 결과
else
log "통계 파일을 찾을 수 없습니다: $stats_file"
fi

log "전체 로그는 다음 파일에서 확인할 수 있습니다: ${LOG_FILE}"

exit 0