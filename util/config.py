from argparse import ArgumentParser
from enum import Enum, auto


class BatchMode(Enum):
    """배치 처리 모드 열거형"""
    NL2SQL = "nl2sql"  # 자연어를 SQL로 변환
    TRANSLATION = "translate"  # 텍스트 번역

    def __str__(self):
        return self.value


def parse_arguments():
    parser = ArgumentParser(description='모델 학습 및 평가 도구')

    # 공통 인수
    parser.add_argument('command', choices=['train', 'merge', 'test', 'eval', 'ollama-api', 'batch', 'upload'],
                        help='실행할 명령 (train, merge, test, eval, ollama-api, batch, upload')
    parser.add_argument('--prefix', type=str, default=".",
                        help='실행 데이터 디렉토리 접두사')
    parser.add_argument('--ollama-url', type=str, default="",
                        help='ollama server 주소')
    parser.add_argument('--base-model', type=str, default='qwq',
                        help='기본 모델 이름')
    parser.add_argument('--finetuned-model', type=str, default="",
                        help='파인튜닝된 모델 이름')
    parser.add_argument('--verifying-model', type=str, default="gemma3:27b",
                        help='검증용 모델 이름 (기본값: deepseek-r1:70b)')

    # ollama-api 명령에 대한 추가 인수
    parser.add_argument('--batch-size', type=int, default=10,
                        help='배치 크기 (ollama-api 명령 시 사용, 기본값: 10)')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='최대 동시 요청 수 (ollama-api 명령 시 사용, 기본값: 10)')
    parser.add_argument('--max-retries', type=int, default=10,
                        help='최대 재시도 횟수 (ollama-api 명령 시 사용, 기본값: 10)')
    parser.add_argument('--test-size', type=int, default=None,
                        help='테스트집합 크기 (ollama-api 명령 시 사용)')
    parser.add_argument('--test-dataset', type=str, default="shangrilar/ko_text2sql:origin:test",
                        help='테스트 데이터셋 이름 (기본값: "shangrilar/ko_text2sql:origin:test")')
    parser.add_argument('--results-file', type=str, default="[PREFIX]/nl2sql_eval_results.csv",
                        help='평가 결과를 저장할 CSV 파일 이름')
    parser.add_argument('--question-column', type=str, default="question",
                        help='질문 컬럼명')
    parser.add_argument('--answer-column', type=str, default="answer",
                        help='응답 컬럼명')

    # 모델 예열 관련 옵션 추가
    parser.add_argument('--warmup-model', action='store_true', default=True,
                        help='배치 처리 전 모델 예열 실행 (기본값: True)')
    parser.add_argument('--no-warmup-model', action='store_false', dest='warmup_model',
                        help='배치 처리 전 모델 예열 비활성화')

    # 배치 모드 선택 (batch 명령 사용 시)
    parser.add_argument('--mode', type=str, choices=[mode.value for mode in BatchMode],
                        default=BatchMode.NL2SQL.value,
                        help='배치 처리 모드 (기본값: nl2sql)')

    # 입력 및 출력 컬럼 지정 (custom 모드나 translate 모드 등에서 사용)
    parser.add_argument('--input-column', type=str, default=None,
                        help='입력 데이터 컬럼 이름')
    parser.add_argument('--output-column', type=str, default=None,
                        help='출력 데이터 컬럼 이름')
    parser.add_argument('--upload-to-hf', type=str, default=None,
                        help='huggingface로 업로드 (repo-id를 지정)')
    parser.add_argument('--from-file', type=str, default=None,
                        help='데이터 업로드')

    # # 번역 모드 옵션
    # parser.add_argument('--source-lang', type=str, default='en',
    #                     help='원본 언어 코드 (번역 모드에서 사용, 기본값: en)')
    # parser.add_argument('--target-lang', type=str, default='ko',
    #                     help='대상 언어 코드 (번역 모드에서 사용, 기본값: ko)')

    args = parser.parse_args()

    # 명령어에 따른 인수 유효성 검사
    if args.command == 'batch':
        # 배치 모드에 따른 필수 인수 확인
        mode = BatchMode(args.mode)

        if mode in [BatchMode.TRANSLATION]:
            if not args.input_column:
                parser.error(f"{mode.value} 모드에서는 --input-column이 필요합니다.")
            if not args.output_column:
                parser.error(f"{mode.value} 모드에서는 --output-column이 필요합니다.")

    return args

api_key = "crR2uHiE9awuVzimCtwmCXG6apq_rsPhHBBfjt1PSts4VmcZyLEwCJv3FFWqCD4hp20KGDL6oeT3BlbkFJBR4xBLE6TLPOwXaUdRiEgzqwE96hHs6xNKZTVXdWrEbxuqUHUZe3neqOYSrHghB8K3NOzVrXMA"

def get_apikey():
    return f"sk-proj-{api_key}"

token = "NRSlvHHFBPntnmidVYoMVOOXgUxjEBMMaM"
def get_hf_token():
    return f"hf_{token}"