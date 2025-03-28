from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser(description='모델 학습 및 평가 도구')

    # 공통 인수
    parser.add_argument('command', choices=['train', 'merge', 'test', 'eval', 'ollama-api'],
                        help='실행할 명령 (train, merge, test, eval, ollama-api')
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
    parser.add_argument('--test-dataset', type=str, default="Spider",
                        help='테스트 데이터셋 이름 (예: Spider, WikiSQL, BIRD)')
    parser.add_argument('--results-file', type=str, default="[PREFIX]/nl2sql_eval_results.csv",
                        help='평가 결과를 저장할 CSV 파일 이름')

    # 모델 예열 관련 옵션 추가
    parser.add_argument('--warmup-model', action='store_true', default=True,
                        help='배치 처리 전 모델 예열 실행 (기본값: True)')
    parser.add_argument('--no-warmup-model', action='store_false', dest='warmup_model',
                        help='배치 처리 전 모델 예열 비활성화')

    return parser.parse_args()

api_key = "crR2uHiE9awuVzimCtwmCXG6apq_rsPhHBBfjt1PSts4VmcZyLEwCJv3FFWqCD4hp20KGDL6oeT3BlbkFJBR4xBLE6TLPOwXaUdRiEgzqwE96hHs6xNKZTVXdWrEbxuqUHUZe3neqOYSrHghB8K3NOzVrXMA"

def get_apikey():
    return f"sk-proj-{api_key}"