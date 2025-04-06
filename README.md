# NL2SQL 평가 및 텍스트 번역 프레임워크

자연어를 SQL(NL2SQL)로 변환하는 언어 모델의 훈련, 테스트 및 평가를 위한 종합 프레임워크입니다. 또한 데이터셋의 특정 컬럼에 대한 텍스트 번역 일괄처리 기능도 지원합니다.

## 개요

이 프로젝트는 다음과 같은 도구를 제공합니다:
- 자연어 쿼리에서 SQL 생성을 위한 언어 모델 훈련 및 미세 조정
- 미세 조정된 모델과 기본 모델 병합
- 테스트셋을 기반으로 NL2SQL 모델의 정확도 평가
- 데이터셋의 지정된 컬럼에 대한 텍스트 번역 일괄처리
- 효율적인 대규모 처리를 위한 병렬 API 요청 처리
- 평가 결과 자동 기록 및 분석 도구
- Hugging Face에 처리된 데이터셋 직접 업로드

## 주요 기능

- **NL2SQL 평가**: 테스트셋을 사용하여 자연어-SQL 변환 모델의 정확도 측정
- **데이터셋 기반 번역**: 데이터프레임의 특정 컬럼에 대한 번역 일괄처리
- **병렬 API 처리**: 제어된 동시성으로 대규모 데이터셋을 효율적으로 처리
- **진행 상황 추적**: 처리 중 실시간 진행 상황 모니터링
- **다양한 모델 지원**: OpenAI 모델(GPT 계열)과 로컬 Ollama 모델 모두 호환
- **종합적인 시간 통계**: 모든 작업의 성능 지표 추적
- **성능 분석 도구**: 모델 평가 결과를 CSV로 자동 저장하고 분석하는 기능
- **Hugging Face 통합**: 처리된 데이터셋을 Hugging Face에 직접 업로드
- **성능 벤치마크**: Ollama 모델의 성능 측정 및 시각화 도구

## 프로젝트 구조

```
.
├── run_evaluation.py                 # 평가 명령을 위한 주요 진입점
├── run_transform.py                  # 데이터셋 변환 도구
├── nl2sql_core.py                    # NL2SQL 핵심 기능 구현
├── scripts/                          # 스크립트 파일 디렉토리
│   ├── run_nl2sql.sh                 # 모델 조합 배치 실행 스크립트
│   ├── ollama_benchmark.sh           # Ollama 성능 측정 스크립트
│   └── ollama_api_benchmark.sh       # Ollama API 기반 성능 측정 스크립트
├── configs/                          # 설정 파일 디렉토리
│   ├── base_models.config            # 기본 모델 목록 설정
│   ├── cpu_models.config             # CPU용 모델 목록 설정
│   └── eval_models.config            # 평가 모델 목록 설정
├── requirements.txt                  # 프로젝트 의존성
├── llms/                             # LLM 관련 유틸리티
│   ├── __init__.py                   # 패키지 정의
│   ├── client.py                     # LLM 클라이언트 구현
│   └── templates.py                  # 프롬프트 템플릿 정의
└── utils/                            # 유틸리티 모듈
    ├── __init__.py                   # 패키지 정의
    ├── common.py                     # 공통 유틸리티 함수
    ├── config.py                     # 구성 및 인수 파싱
    ├── parallel.py                   # 병렬 처리 유틸리티
    ├── reporting.py                  # 평가 결과 로깅 및 리포팅
    ├── sql_extractor.py              # SQL 쿼리 추출 유틸리티
    └── tracking.py                   # 진행 상황 추적 유틸리티
```

## 설치

### 전제 조건

- Python 3.8+
- Ollama (로컬 모델 평가용)
- AutoTrain (모델 미세 조정용)

### 설정

1. 저장소 복제:
   ```bash
   git clone [저장소-URL]
   cd nl2sql-evaluation
   ```

2. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 사용법

이 프레임워크는 NL2SQL 및 텍스트 처리 워크플로우의 다양한 단계에 대한 여러 명령을 제공합니다:

### 훈련

새 모델을 훈련하거나 기존 모델을 미세 조정합니다:

```bash
python run_evaluation.py train --base-model [기본-모델-이름] --finetuned-model [출력-모델-이름] --prefix [데이터-디렉토리]
```

### 모델 병합

미세 조정된 모델을 기본 모델과 병합합니다:

```bash
python run_evaluation.py merge --base-model [기본-모델-이름] --finetuned-model [미세-조정-모델-이름] --prefix [출력-디렉토리]
```

### NL2SQL 평가

테스트 데이터셋에서 모델의 NL2SQL 성능을 평가합니다:

```bash
python run_evaluation.py eval --base-model [테스트할-모델] --verifying-model [검증-모델] --prefix [출력-디렉토리] --test-size [샘플-수] --test-dataset [데이터셋-이름]
```

### Ollama API를 사용한 테스트

Ollama API를 사용하여 모델을 테스트합니다:

```bash
python run_evaluation.py ollama-api --base-model [모델-이름] --ollama-url [ollama-서버-URL] --batch-size [배치-크기] --max-concurrent [동시-요청] --max-retries [최대-재시도] --prefix [출력-디렉토리] --test-dataset [데이터셋-이름] --timeout [요청-타임아웃]
```

### 배치 처리 (NL2SQL 또는 번역)

지정된 모드에 따라 배치 처리를 실행합니다:

```bash
python run_evaluation.py batch --mode [nl2sql|translate] --base-model [모델-이름] --ollama-url [ollama-서버-URL] --batch-size [배치-크기] --max-concurrent [동시-요청] --prefix [출력-디렉토리]
```

#### 번역 모드 사용 예시 (데이터셋 컬럼 기반):

```bash
python run_evaluation.py batch --mode translate --base-model [모델-이름] --input-column [입력-컬럼] --output-column [출력-컬럼] --prefix [출력-디렉토리]
```

이 명령은 데이터셋에서 `input-column`으로 지정된 컬럼의 텍스트를 번역하여 `output-column`에 저장합니다.

### Hugging Face 업로드

처리된 결과를 Hugging Face에 업로드합니다:

```bash
python run_evaluation.py upload --from-file [입력-파일] --upload-to-hf [huggingface-저장소-ID]
```

### 데이터셋 변환 및 DBMS 감지

`run_transform.py`를 사용하여 데이터셋 구조를 변환하고 SQL 문의 DBMS 유형을 감지할 수 있습니다:

```bash
# 프롬프트를 (context, question, answer) 형식으로 변환
python run_transform.py transform --input_dataset [입력-데이터셋] --output_dataset [출력-데이터셋] --split [스플릿-이름]

# SQL 쿼리의 DBMS 유형 감지
python run_transform.py dbms --input_dataset [입력-데이터셋] --output_dataset [출력-데이터셋] --sql_field [SQL-컬럼] --dbms_field [DBMS-컬럼]

# 변환 및 DBMS 감지를 한 번에 수행
python run_transform.py trans-dbms --input_dataset [입력-데이터셋] --output_dataset [출력-데이터셋] --sql_field [SQL-컬럼] --dbms_field [DBMS-컬럼]
```

## 구성 옵션

| 옵션 | 설명 | 기본값 |
|--------|-------------|---------|
| `--prefix` | 데이터 및 출력 파일 디렉토리 | `.` |
| `--ollama-url` | Ollama 서버 주소 | `""` |
| `--base-model` | 기본 모델 이름 | `qwq` |
| `--finetuned-model` | 미세 조정된 모델 이름 | `""` |
| `--verifying-model` | 검증에 사용되는 모델 | `gemma3:27b` |
| `--batch-size` | API 요청 배치 크기 | `10` |
| `--max-concurrent` | 최대 동시 API 요청 | `10` |
| `--max-retries` | 실패한 요청에 대한 최대 재시도 횟수 | `10` |
| `--request-timeout` | API 요청 타임아웃 (초, 0 = 무제한) | `300` |
| `--test-size` | 사용할 테스트 샘플 수(null = 전체) | `None` |
| `--test-dataset` | 테스트 데이터셋 이름 | `shangrilar/ko_text2sql:origin:test` |
| `--mode` | 배치 처리 모드 (`nl2sql` 또는 `translate`) | `nl2sql` |
| `--input-column` | 입력 데이터 컬럼 이름 | 모드별 기본값 |
| `--output-column` | 출력 데이터 컬럼 이름 | 모드별 기본값 |
| `--warmup-model` | 배치 처리 전 모델 예열 활성화 | `True` |
| `--question-column` | 질문 컬럼명 | `question` |
| `--answer-column` | 응답 컬럼명 | `answer` |
| `--upload-to-hf` | 업로드할 Hugging Face 저장소 ID | `None` |
| `--no-evaluation` | 평가 절차 제외 | `False` |

## 여러 모델 조합 실행

`scripts/run_nl2sql.sh` 스크립트를 사용하여 여러 모델 조합을 실행할 수 있습니다:

### 기본 사용법

```bash
./scripts/run_nl2sql.sh "sqlcoder:70b llama3:70b" "gemma3:27b gemma3:8b"
```

### 환경변수 사용

```bash
TEST_SIZE=100 TEST_DATASET="shangrilar/ko_text2sql:clean:test" ./scripts/run_nl2sql.sh "sqlcoder:70b" "gemma3:27b"
```

### 명령줄 옵션 사용

```bash
./scripts/run_nl2sql.sh -s 50 -d "shangrilar/ko_text2sql:clean:test" "sqlcoder:70b" "gemma3:27b"
```

### 모델 파일 사용

모델 목록이 포함된 파일을 생성하고 사용할 수 있습니다:

```bash
# 미리 구성된 모델 목록 파일 사용
./scripts/run_nl2sql.sh -f configs/base_models.config configs/eval_models.config
```

`run_nl2sql.sh` 스크립트는 두 개의 인자(base 모델 목록과 verifying 모델 목록)를 받으며, 이들 사이의 모든 조합에 대해 평가를 수행합니다. `-f` 옵션을 사용하면 이 목록들을 파일에서 읽을 수 있습니다.

### run_nl2sql.sh 스크립트 환경변수

| 환경변수 | 설명 | 기본값 |
|--------|-------------|---------|
| `OLLAMA_URL` | Ollama 서버 URL | `172.16.15.112` |
| `PREFIX` | 출력 디렉토리 경로 | `OUTPUT` |
| `BATCH_SIZE` | 배치 크기 | `50` |
| `MAX_CONCURRENT` | 최대 동시 요청 수 | `20` |
| `MAX_RETRIES` | 최대 재시도 횟수 | `10` |
| `MODE` | 실행 모드 | `ollama-api` |
| `TEST_SIZE` | 테스트 크기 | (설정된 경우만 사용) |
| `TEST_DATASET` | 테스트 데이터셋 | (설정된 경우만 사용) |
| `REQUEST_TIMEOUT` | API 요청 타임아웃 (초) | `300` |

## Ollama 모델 벤치마크

Ollama 모델의 성능을 측정하기 위한 두 가지 스크립트가 제공됩니다:

### Ollama 로컬 벤치마크

```bash
./scripts/ollama_benchmark.sh [모델명] [프롬프트-길이] [테스트-반복-횟수] [환경타입] [콜드스타트] [웜업]
```

매개변수 설명:
- 모델명: Ollama에 저장된 모델명 또는 'all'(기본값: all)
- 프롬프트 길이: 테스트에 사용할 프롬프트 길이 (기본값: 100)
- 테스트 반복 횟수: 각 모델당 테스트 반복 횟수 (기본값: 5)
- 환경타입: 'docker', 'local', 'auto' 중 선택 (기본값: auto)
- 콜드스타트: 'true', 'false' 중 선택 (기본값: false)
- 웜업: 'true', 'false' 중 선택 (기본값: true)

### Ollama API 벤치마크

```bash
./scripts/ollama_api_benchmark.sh [모델명] [프롬프트-길이] [테스트-반복-횟수] [콜드스타트] [웜업]
```

매개변수 설명:
- 모델명: Ollama에 저장된 모델명 또는 'all'(기본값: all)
- 프롬프트 길이: 테스트에 사용할 프롬프트 길이 (기본값: 100)
- 테스트 반복 횟수: 각 모델당 테스트 반복 횟수 (기본값: 5)
- 콜드스타트: 'true', 'false' 중 선택 (기본값: false)
- 웜업: 'true', 'false' 중 선택 (기본값: true)

벤치마크 결과는 HTML 보고서 형태로 생성되며, 다음 정보를 포함합니다:
- 모델 로드 시간 (초)
- 첫 토큰 생성 시간 (초)
- 총 생성 시간 (초)
- 토큰 생성 속도 (tokens/sec)
- CPU/GPU 사용량 모니터링

## 평가 결과 로깅

모든 평가 실행은 자동으로 결과를 기록하고 CSV 파일에 저장합니다:

- **NL2SQL 평가 결과**: `[prefix]/stats/nl2sql_verification_stats.csv` (기본값)
  - 모델 이름, 테스트셋, 정확도, 처리 시간, 배치 처리량 등의 성능 지표
  - 테스트 설정 (배치 크기, 동시 요청 수 등)

- **번역 처리 결과**: `[prefix]/stats/nl2sql_translation_stats.csv` (번역 모드)
  - 모델 이름, 처리 시간, 배치 처리량
  - 데이터셋 크기, 성공/실패 항목 수
  - 배치 처리 설정 (배치 크기, 동시 요청 수 등)

- **처리된 데이터셋**: `[prefix]/batch_results/[mode]/[model_name]_results.{jsonl|csv}`
  - 원본 데이터와 처리 결과가 포함된 데이터셋

여러 조합의 평가가 실행되면, 각 실행 결과는 동일한 통계 파일에 추가됩니다. `utils/reporting.py` 모듈을 통해 결과를 다양한 형식(Markdown, Wiki 표 등)으로 내보내거나 요약 보고서를 생성할 수 있습니다.

### 결과 분석

저장된 CSV 파일을 사용하여 다양한 모델의 성능을 비교하고 분석할 수 있습니다:

```python
import pandas as pd

# CSV 파일 로드
results = pd.read_csv('OUTPUT/stats/nl2sql_verification_stats.csv')

# 모델별 평균 성능
model_performance = results.groupby('nl2sql_model').agg({
    'accuracy': 'mean',
    'avg_processing_time': 'mean',
    'batch_throughput': 'mean'
}).sort_values(by='accuracy', ascending=False)

print(model_performance)

# 데이터셋별 성능 분석
dataset_performance = results.pivot_table(
    index='nl2sql_model',
    columns='test_dataset',
    values='accuracy',
    aggfunc='mean'
)

print(dataset_performance)
```

## 모니터링 및 로깅

이 프레임워크는 포괄적인 로깅과 진행 상황 추적을 제공합니다:
- 실시간 진행 표시줄은 완료 상태를 보여줍니다
- 상세한 로그는 프로세스 정보와 오류를 기록합니다
- 테이블 형식의 결과 요약이 콘솔에 표시됩니다
- 모든 실행 결과는 CSV 파일에 누적되어 저장됩니다
- `run_nl2sql.sh` 실행 시 별도의 로그 파일이 생성됩니다 (`[prefix]/logs/run_nl2sql_*.log`)

## 구성 파일

`configs/` 디렉토리에 있는 모델 구성 파일:

- **base_models.config**: NL2SQL 생성에 사용될 기본 모델 목록
- **cpu_models.config**: CPU에서 실행할 수 있는 모델 목록
- **eval_models.config**: 평가에 사용될 모델 목록

이 파일들은 `run_nl2sql.sh` 스크립트와 함께 `-f` 옵션을 사용할 때 참조할 수 있습니다.

## 라이센스

[라이센스 정보]
