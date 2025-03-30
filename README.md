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

## 주요 기능

- **NL2SQL 평가**: 테스트셋을 사용하여 자연어-SQL 변환 모델의 정확도 측정
- **데이터셋 기반 번역**: 데이터프레임의 특정 컬럼에 대한 번역 일괄처리
- **병렬 API 처리**: 제어된 동시성으로 대규모 데이터셋을 효율적으로 처리
- **진행 상황 추적**: 처리 중 실시간 진행 상황 모니터링
- **다양한 모델 지원**: OpenAI 모델(GPT 계열)과 로컬 Ollama 모델 모두 호환
- **종합적인 시간 통계**: 모든 작업의 성능 지표 추적
- **성능 분석 도구**: 모델 평가 결과를 CSV로 자동 저장하고 분석하는 기능

## 프로젝트 구조

```
.
├── run_evaluation.py                 # 평가 명령을 위한 주요 진입점
├── evaluator.py                      # 핵심 평가 기능
├── api_request_parallel_processor.py # 속도 제한이 있는 병렬 API 요청 처리
├── requirements.txt                  # 프로젝트 의존성
├── llms/                             # LLM 관련 유틸리티
│   ├── __init__.py                   # 패키지 정의
│   ├── ollama_api.py                 # Ollama API 통합
│   ├── prompt_generator.py           # 모델용 프롬프트 생성
│   └── response_processor.py         # 모델 응답 처리
└── util/                             # 유틸리티 모듈
    ├── __init__.py                   # 패키지 정의
    ├── config.py                     # 구성 및 인수 파싱
    ├── eval_results_logger.py        # 평가 결과 로깅 및 저장
    ├── progress.py                   # 진행 상황 추적 유틸리티
    └── util_common.py                # 공통 유틸리티 함수
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
python run_evaluation.py ollama-api --base-model [모델-이름] --ollama-url [ollama-서버-URL] --batch-size [배치-크기] --max-concurrent [동시-요청] --max-retries [최대-재시도] --prefix [출력-디렉토리] --test-dataset [데이터셋-이름]
```

### 배치 처리 (NL2SQL 또는 번역)

지정된 모드에 따라 배치 처리를 실행합니다:

```bash
python run_evaluation.py batch --mode [nl2sql|translate] --base-model [모델-이름] --ollama-url [ollama-서버-URL] --batch-size [배치-크기] --max-concurrent [동시-요청] --prefix [출력-디렉토리]
```

#### 번역 모드 사용 예시 (데이터셋 컬럼 기반):

```bash
python run_evaluation.py batch --mode translate --base-model [모델-이름] --input-column [입력-컬럼] --output-column [출력-컬럼] --source-lang en --target-lang ko --prefix [출력-디렉토리]
```

이 명령은 데이터셋에서 `input-column`으로 지정된 컬럼의 텍스트를 `source-lang`에서 `target-lang`으로 번역하여 `output-column`에 저장합니다.

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
| `--test-size` | 사용할 테스트 샘플 수(null = 전체) | `None` |
| `--test-dataset` | 테스트 데이터셋 이름 | `Spider` |
| `--mode` | 배치 처리 모드 (`nl2sql` 또는 `translate`) | `nl2sql` |
| `--input-column` | 입력 데이터 컬럼 이름 | 모드별 기본값 |
| `--output-column` | 출력 데이터 컬럼 이름 | 모드별 기본값 |
| `--source-lang` | 번역 모드의 원본 언어 코드 | `en` |
| `--target-lang` | 번역 모드의 대상 언어 코드 | `ko` |
| `--warmup-model` | 배치 처리 전 모델 예열 활성화 | `True` |
| `--results-file` | 평가 결과를 저장할 CSV 파일 이름 | `nl2sql_eval_results.csv` |

## 추가 사용 예제

### NL2SQL 평가

```bash
python run_evaluation.py eval --base-model sqlcoder2:latest --verifying-model gemma3:27b --prefix ./results --test-size 100 --test-dataset Spider
```

### 데이터셋 컬럼 번역 일괄처리

```bash
python run_evaluation.py batch --mode translate --base-model mistral:7b --ollama-url http://localhost:11434 --batch-size 20 --input-column english_text --output-column korean_text --source-lang en --target-lang ko --prefix ./translations
```

이 예제는 데이터셋의 'english_text' 컬럼을 영어에서 한국어로 번역하여 'korean_text' 컬럼에 저장합니다.

### Ollama로 대량의 쿼리 일괄 처리

```bash
python run_evaluation.py ollama-api --base-model sqlcoder2:latest --ollama-url http://localhost:11434 --batch-size 20 --max-concurrent 5 --max-retries 3 --prefix ./batch-results
```

## 평가 결과 로깅

모든 평가 실행은 자동으로 결과를 기록하고 CSV 파일에 저장합니다:

- **NL2SQL 평가 결과**: `[prefix]/stats/nl2sql_eval_results.csv` (기본값)
  - 모델 이름, 테스트셋, 정확도, 처리 시간, 배치 처리량 등의 성능 지표
  - 실행 환경 정보 (메모리 사용량, 리소스 등)
  - 테스트 설정 (배치 크기, 동시 요청 수 등)

- **번역 처리 결과**: `[prefix]/stats/translation_stats.csv` (번역 모드)
  - 원본 및 대상 언어, 모델 이름, 처리 시간, 배치 처리량
  - 데이터셋 크기, 성공/실패 항목 수
  - 배치 처리 설정 (배치 크기, 동시 요청 수 등)

- **처리된 데이터셋**: `[prefix]/batch_results/[mode]/[model_name]_[timestamp].{jsonl|csv}`
  - 원본 데이터와 처리 결과가 포함된 데이터셋

### 결과 분석

저장된 CSV 파일을 사용하여 다양한 모델의 성능을 비교하고 분석할 수 있습니다:

```python
import pandas as pd

# CSV 파일 로드
results = pd.read_csv('results/stats/nl2sql_eval_results.csv')

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

## 확장 계획

- **번역 품질 평가**: 번역 결과의 품질을 자동으로 평가하는 기능 추가
- **다국어 데이터셋 지원**: 다양한 언어의 데이터셋 처리 및 언어 감지 기능
- **추가 작업 모드**: QA(질의응답), 요약, 텍스트 분류 등의 작업 지원
- **자동 리포트 생성**: 평가 결과를 기반으로 한 PDF 또는 HTML 보고서 자동 생성
- **대시보드 통합**: 평가 결과를 시각화하는 웹 기반 대시보드

## 고급 사용법

### 사용자 정의 프롬프트 템플릿

다양한 모델 및 평가 시나리오에 맞게 `llms/prompt_generator.py`에서 프롬프트 템플릿을 사용자 정의할 수 있습니다.

### 성능 튜닝

최적의 성능을 위해 다음 매개변수를 조정하세요:
- `batch-size`: 각 배치의 요청 수
- `max-concurrent`: 최대 동시 API 요청 수
- `max-retries`: 실패한 요청에 대한 재시도 횟수

### 요약 보고서 생성

평가 실행 후 결과를 요약한 HTML 보고서를 생성할 수 있습니다:

```python
from util.eval_results_logger import EvalResultsLogger

# 로거 초기화
logger = EvalResultsLogger(output_dir='./results/stats')

# 요약 보고서 생성
summary = logger.generate_summary_report('./results/summary_report.html')
```

## 모니터링 및 로깅

이 프레임워크는 포괄적인 로깅과 진행 상황 추적을 제공합니다:
- 실시간 진행 표시줄은 완료 상태를 보여줍니다
- 상세한 로그는 프로세스 정보와 오류를 기록합니다
- 테이블 형식의 결과 요약이 콘솔에 표시됩니다
- 모든 실행 결과는 CSV 파일에 누적되어 저장됩니다

## 기여

기여는 환영합니다! Pull Request를 제출해 주세요.

## 라이센스

[라이센스 정보]