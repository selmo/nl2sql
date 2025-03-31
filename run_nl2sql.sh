#!/bin/bash
# run_nl2sql.sh
# NL2SQL 모델 조합 배치 실행을 위한 스크립트

# 기본 설정 (환경변수가 없는 경우 기본값)
OLLAMA_URL=${OLLAMA_URL:-"172.16.15.112"}
PREFIX=${PREFIX:-"OUTPUT"}
BATCH_SIZE=${BATCH_SIZE:-50}
MAX_CONCURRENT=${MAX_CONCURRENT:-20}
MAX_RETRIES=${MAX_RETRIES:-10}
MODE=${MODE:-"ollama-api"}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-300}  # 기본값: 300초, 0=무제한
LOG_DIR="${PREFIX}/logs"
# TEST_SIZE 환경변수가 설정되어 있으면 해당 값을 사용
TEST_SIZE_OPT=""
if [ ! -z "$TEST_SIZE" ]; then
  TEST_SIZE_OPT="--test-size $TEST_SIZE"
fi
# TEST_DATASET 환경변수가 설정되어 있으면 해당 값을 사용
TEST_DATASET_OPT=""
if [ ! -z "$TEST_DATASET" ]; then
  TEST_DATASET_OPT="--test-dataset $TEST_DATASET"
fi

# 사용법 출력
function show_usage {
  echo "사용법: $0 [옵션] <base_models> <verifying_models>"
  echo "또는:  $0 [옵션] -f <base_models_file> <verifying_models_file>"
  echo ""
  echo "옵션:"
  echo "  -h, --help           도움말 표시"
  echo "  -f, --file           모델 목록을 파일에서 읽기 (한 줄에 하나의 모델)"
  echo "  -d, --dataset NAME   테스트 데이터셋 이름 지정"
  echo "  -s, --size NUM       테스트 크기 지정"
  echo ""
  echo "예시:"
  echo "  $0 \"sqlcoder:70b llama3:70b\" \"gemma3:27b gemma3:8b\""
  echo "  $0 -f base_models.txt verify_models.txt"
  echo "  $0 -d shangrilar/ko_text2sql:origin:test -s 100 \"sqlcoder:70b\" \"gemma3:27b\""
  echo ""
  echo "환경변수 설정:"
  echo "  OLLAMA_URL     - Ollama 서버 URL (기본값: 172.16.15.112)"
  echo "  PREFIX         - 출력 디렉토리 경로 (기본값: OUTPUT)"
  echo "  BATCH_SIZE     - 배치 크기 (기본값: 50)"
  echo "  MAX_CONCURRENT - 최대 동시 요청 수 (기본값: 20)"
  echo "  MAX_RETRIES    - 최대 재시도 횟수 (기본값: 10)"
  echo "  MODE           - 실행 모드 (기본값: ollama-api)"
  echo "  TEST_SIZE      - 테스트 크기 (설정된 경우 --test-size 옵션으로 전달)"
  echo "  TEST_DATASET   - 테스트 데이터셋 (설정된 경우 --test-dataset 옵션으로 전달)"
  echo ""
  echo "각 모델 이름은 공백으로 구분합니다."
  exit 1
}

# 로그 디렉토리 생성 함수
function setup_logging {
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/run_nl2sql_$(date +%Y%m%d_%H%M%S).log"

  # 로그 파일 생성 및 초기 메시지
  echo "=============== NL2SQL 배치 실행 로그 ===============" > "${LOG_FILE}"
  echo "시작 시간: $(date)" >> "${LOG_FILE}"
  echo "설정:" >> "${LOG_FILE}"
  echo "  OLLAMA_URL=${OLLAMA_URL}" >> "${LOG_FILE}"
  echo "  PREFIX=${PREFIX}" >> "${LOG_FILE}"
  echo "  BATCH_SIZE=${BATCH_SIZE}" >> "${LOG_FILE}"
  echo "  MAX_CONCURRENT=${MAX_CONCURRENT}" >> "${LOG_FILE}"
  echo "  MAX_RETRIES=${MAX_RETRIES}" >> "${LOG_FILE}"
  echo "  MODE=${MODE}" >> "${LOG_FILE}"
  if [ ! -z "$TEST_SIZE" ]; then
    echo "  TEST_SIZE=${TEST_SIZE}" >> "${LOG_FILE}"
  fi
  if [ ! -z "$TEST_DATASET" ]; then
    echo "  TEST_DATASET=${TEST_DATASET}" >> "${LOG_FILE}"
  fi
  echo "BASE_MODELS: ${BASE_MODELS[@]}" >> "${LOG_FILE}"
  echo "VERIFYING_MODELS: ${VERIFYING_MODELS[@]}" >> "${LOG_FILE}"
  echo "===================================================" >> "${LOG_FILE}"

  # 로그 정보 출력
  echo "로그 파일이 생성되었습니다: ${LOG_FILE}"
}

# 로그 출력 함수 (콘솔과 파일에 모두 출력)
function log {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# 현재 환경변수 설정 출력 함수
function print_config {
  log "현재 설정:"
  log "  OLLAMA_URL     = ${OLLAMA_URL}"
  log "  PREFIX         = ${PREFIX}"
  log "  BATCH_SIZE     = ${BATCH_SIZE}"
  log "  MAX_CONCURRENT = ${MAX_CONCURRENT}"
  log "  MAX_RETRIES    = ${MAX_RETRIES}"
  log "  MODE           = ${MODE}"
  if [ ! -z "$TEST_SIZE" ]; then
    log "  TEST_SIZE      = ${TEST_SIZE}"
  fi
  if [ ! -z "$TEST_DATASET" ]; then
    log "  TEST_DATASET   = ${TEST_DATASET}"
  fi
}

# 파일에서 모델 목록 읽기 함수
function read_models_from_file {
  local file=$1
  if [ ! -f "$file" ]; then
    log "오류: 모델 파일을 찾을 수 없습니다: $file"
    exit 1
  fi

  # 파일에서 모델 읽기 (빈 줄과 주석 제외)
  local models=()
  while IFS= read -r line || [ -n "$line" ]; do
    # '#'으로 시작하는 줄 또는 빈 줄 제외
    if [[ ! "$line" =~ ^[[:space:]]*# && -n "${line// /}" ]]; then
      models+=("$line")
    fi
  done < "$file"

  # 모델 배열 반환
  echo "${models[@]}"
}

# 명령행 인수 파싱
USE_FILES=false
TEST_DATASET_CLI=""
TEST_SIZE_CLI=""
NO_EVAL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_usage
      ;;
    -f|--file)
      USE_FILES=true
      shift
      ;;
    -e|--no-eval)
      NO_EVAL=true
      shift
      ;;
    -d|--dataset)
      TEST_DATASET_CLI="$2"
      shift 2
      ;;
    -s|--size)
      TEST_SIZE_CLI="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# 인수 확인
if [ $# -ne 2 ]; then
  echo "오류: 필수 인수가 빠졌습니다."
  show_usage
fi

# 명령행 옵션이 환경변수보다 우선
if [ ! -z "$TEST_DATASET_CLI" ]; then
  TEST_DATASET="$TEST_DATASET_CLI"
  TEST_DATASET_OPT="--test-dataset $TEST_DATASET"
fi

if [ ! -z "$TEST_SIZE_CLI" ]; then
  TEST_SIZE="$TEST_SIZE_CLI"
  TEST_SIZE_OPT="--test-size $TEST_SIZE"
fi

if [ "$NO_EVAL" = true]; then
  NO_EVAL_OPT="--no-evaluation"
fi

# 모델 목록 설정
if [ "$USE_FILES" = true ]; then
  # 파일에서 모델 리스트 읽기
  BASE_MODELS_RAW=$(read_models_from_file "$1")
  VERIFY_MODELS_RAW=$(read_models_from_file "$2")

  # 문자열을 배열로 변환
  read -ra BASE_MODELS <<< "$BASE_MODELS_RAW"
  read -ra VERIFYING_MODELS <<< "$VERIFY_MODELS_RAW"

  log "파일에서 모델 리스트를 읽었습니다."
  log "BASE_MODELS 파일: $1 (${#BASE_MODELS[@]}개 모델)"
  log "VERIFYING_MODELS 파일: $2 (${#VERIFYING_MODELS[@]}개 모델)"
else
  # 명령행에서 직접 모델 리스트 사용
  BASE_MODELS=($1)
  VERIFYING_MODELS=($2)
fi

# 모델 목록 확인
if [ ${#BASE_MODELS[@]} -eq 0 ] || [ ${#VERIFYING_MODELS[@]} -eq 0 ]; then
  echo "오류: 모델 목록이 비어 있습니다."
  show_usage
fi

# 로깅 설정
setup_logging

log "===== NL2SQL 배치 실행 시작 ====="
print_config
log "Base 모델 (${#BASE_MODELS[@]}개): ${BASE_MODELS[@]}"
log "Verifying 모델 (${#VERIFYING_MODELS[@]}개): ${VERIFYING_MODELS[@]}"
log "결과 저장 경로: $PREFIX"

# 디렉토리 생성
mkdir -p "$PREFIX/logs"

# 각 모델 조합 실행
total_combinations=$((${#BASE_MODELS[@]} * ${#VERIFYING_MODELS[@]}))
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
      --max-retries $MAX_RETRIES \
      --request-timeout $REQUEST_TIMEOUT \
      $TEST_SIZE_OPT \
      $TEST_DATASET_OPT \
      $NO_EVAL_OPT"

    log "명령: $cmd"

    # 실행 (출력을 로그 파일에도 저장)
    start_time=$(date +%s)
    log "실행 시작..."
    eval $cmd 2>&1 | tee -a "${LOG_FILE}"
    exit_code=${PIPESTATUS[0]}
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # 결과 출력
    if [ $exit_code -eq 0 ]; then
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
if [ -f "$stats_file" ]; then
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