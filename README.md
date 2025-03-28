# NL2SQL 평가 프레임워크

자연어를 SQL(NL2SQL)로 변환하는 언어 모델의 훈련, 테스트 및 평가를 위한 종합 프레임워크입니다.

## 개요

이 프로젝트는 다음과 같은 도구를 제공합니다:
- 자연어 쿼리에서 SQL 생성을 위한 언어 모델 훈련 및 미세 조정
- 미세 조정된 모델과 기본 모델 병합
- NL2SQL 작업에서 모델 성능 평가
- 효율적인 평가를 위한 병렬 API 요청 처리

## 주요 기능

- **병렬 API 처리**: 제어된 동시성으로 대규모 데이터셋을 효율적으로 평가
- **진행 상황 추적**: 평가 중 실시간 진행 상황 모니터링
- **다양한 모델 지원**: OpenAI 모델(GPT 계열)과 로컬 Ollama 모델 모두 호환
- **종합적인 시간 통계**: 모든 작업의 성능 지표 추적

## 프로젝트 구조

```
.
├── api_request_parallel_processor.py  # 속도 제한이 있는 병렬 API 요청 처리
├── evaluator.py                      # 핵심 평가 기능
├── run_evaluation.py                 # 평가 명령을 위한 주요 진입점
├── config.py                         # 구성 및 인수 파싱
├── llms/                             # LLM 관련 유틸리티
│   ├── ollama_api.py                 # Ollama API 통합
│   ├── prompt_generator.py           # 모델용 프롬프트 생성
│   └── response_processor.py         # 모델 응답 처리
└── util/                             # 유틸리티 모듈
    ├── progress.py                   # 진행 상황 추적 유틸리티
    ├── timing_stats.py               # 성능 지표 추적
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

이 프레임워크는 NL2SQL 워크플로우의 다양한 단계에 대한 여러 명령을 제공합니다:

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

### 평가

테스트 데이터셋에서 모델 성능을 평가합니다:

```bash
python run_evaluation.py eval --base-model [테스트할-모델] --verifying-model [검증-모델] --prefix [출력-디렉토리] --test-size [샘플-수]
```

### Ollama API를 사용한 테스트

Ollama API를 사용하여 모델을 테스트합니다:

```bash
python run_evaluation.py ollama-api --base-model [모델-이름] --ollama-url [ollama-서버-URL] --batch-size [배치-크기] --max-concurrent [동시-요청] --max-retries [최대-재시도] --prefix [출력-디렉토리]
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
| `--test-size` | 사용할 테스트 샘플 수(null = 전체) | `None` |

## 예제

### 모델 미세 조정

```bash
python run_evaluation.py train --base-model sqlcoder2:latest --finetuned-model sqlcoder2-finetuned --prefix ./models
```

### 테스트 데이터셋에 대한 모델 평가

```bash
python run_evaluation.py eval --base-model sqlcoder2:latest --verifying-model gemma3:27b --prefix ./results --test-size 100
```

### Ollama로 대량의 쿼리 일괄 처리

```bash
python run_evaluation.py ollama-api --base-model sqlcoder2:latest --ollama-url http://localhost:11434 --batch-size 20 --max-concurrent 5 --max-retries 3 --prefix ./batch-results
```

## 고급 사용법

### 사용자 정의 프롬프트 템플릿

다양한 모델 및 평가 시나리오에 맞게 `llms/prompt_generator.py`에서 프롬프트 템플릿을 사용자 정의할 수 있습니다.

### 성능 튜닝

최적의 성능을 위해 다음 매개변수를 조정하세요:
- `batch-size`: 각 배치의 요청 수
- `max-concurrent`: 최대 동시 API 요청 수
- `max-retries`: 실패한 요청에 대한 재시도 횟수

## 모니터링 및 로깅

이 프레임워크는 포괄적인 로깅과 진행 상황 추적을 제공합니다:
- 실시간 진행 표시줄은 완료 상태를 보여줍니다
- 상세한 로그는 프로세스 정보와 오류를 기록합니다
- 시간 통계는 성능 지표를 기록합니다

## 기여

기여는 환영합니다! Pull Request를 제출해 주세요.

## 라이센스

[라이센스 정보]