import time
import argparse
import aux_local
import logging
import sys

from aux_common import merge_model, evaluation
from nl2sql_processor import prepare_test_dataset
from timing_stats import timing_stats_manager, log_timing_stats

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='모델 학습 및 평가 도구')

    # 공통 인수
    parser.add_argument('command', choices=['ollama-langchain', 'ollama-api', 'ollama-http'],
                        help='실행할 명령 (test, eval, ollama-api, ollama-http, eval-csv)')
    parser.add_argument('--prefix', type=str, default=".",
                        help='실행 데이터 디렉토리 접두사')
    parser.add_argument('--ollama-url', type=str, default="172.16.15.112",
                        help='ollama server 주소')
    parser.add_argument('--base-model', type=str, default='qwq',
                        help='기본 모델 이름')
    parser.add_argument('--verifying-model', type=str, default="deepseek-r1:70b",
                        help='검증용 모델 이름 (기본값: deepseek-r1:70b)')

    # ollama-api 명령에 대한 추가 인수
    parser.add_argument('--batch-size', type=int, default=10,
                        help='배치 크기 (ollama-api 명령 시 사용, 기본값: 10)')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='최대 동시 요청 수 (ollama-api 명령 시 사용, 기본값: 10)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='최대 재시도 횟수 (ollama-api 명령 시 사용, 기본값: 3)')
    parser.add_argument('--test-size', type=int, default=0,
                        help='테스트집합 크기 (ollama-api 명령 시 사용, 기본값: 0[전체])')
    parser.add_argument('--eval-api', action='store_true', help='평가에 api 사용')

    return parser.parse_args()


# main routine
if __name__ == "__main__":
    api_key = "crR2uHiE9awuVzimCtwmCXG6apq_rsPhHBBfjt1PSts4VmcZyLEwCJv3FFWqCD4hp20KGDL6oeT3BlbkFJBR4xBLE6TLPOwXaUdRiEgzqwE96hHs6xNKZTVXdWrEbxuqUHUZe3neqOYSrHghB8K3NOzVrXMA"

    args = parse_arguments()

    prefix = args.prefix
    gen_model = args.base_model
    eval_model = args.verifying_model
    test_size = args.test_size

    # 병렬 처리 매개변수 설정
    batch_size = args.batch_size
    max_concurrent = args.max_concurrent
    max_retries = args.max_retries

    gen_prefix = f"{prefix}_{gen_model.replace(':', '-')}" if prefix != "" or prefix != "." else gen_model.replace(':', '-')
    result_prefix = f"{prefix}_{gen_model.replace(':', '-')}_{eval_model.replace(':', '-')}"

    program_start_time = time.time()
    timing_stats_manager.start_process("main")

    command_start_time = time.time()
    timing_stats_manager.start_process(f"command_{args.command}", "main")

    if args.command == 'ollama-api':
        logging.info(f"병렬 처리 설정: 배치 크기 {batch_size}, 최대 동시 요청 {max_concurrent}, 최대 재시도 {max_retries}")

        # 테스트 데이터셋 준비 시간 측정 (병렬 처리)
        timing_stats_manager.start_process("prepare_test_dataset", f"command_{args.command}")
        # test_dataset = (
        prepare_test_dataset(
            gen_model,
            gen_prefix,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            ollama_url=args.ollama_url,
            test_size=test_size,
        )
        prepare_time = timing_stats_manager.stop_process("prepare_test_dataset")
        logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

        # # 평가 시간 측정
        # timing_stats_manager.start_process("evaluation", f"command_{args.command}")
        #
        # evaluation(gen_model, eval_model, test_dataset, result_prefix, api_key=f"sk-proj-{api_key}")
        #
        # eval_time = timing_stats_manager.stop_process("evaluation")
        # logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    elif args.command == 'ollama-http':
        # 모델 병합 시간 측정
        timing_stats_manager.start_process("merge_model", f"command_{args.command}")
        gen_model, gen_prefix = merge_model(gen_model, eval_model, prefix)
        merge_time = timing_stats_manager.stop_process("merge_model")
        logging.info(f"모델 병합 완료: {merge_time:.2f}초 소요")

        # HTTP 테스트 준비 시간 측정
        timing_stats_manager.start_process("prepare_test_ollama", f"command_{args.command}")
        test_dataset = aux_local.prepare_test_ollama(gen_model, gen_prefix, test_size)
        prepare_time = timing_stats_manager.stop_process("prepare_test_ollama")
        logging.info(f"HTTP 테스트 준비 완료: {prepare_time:.2f}초 소요")

        # 평가 시간 측정
        timing_stats_manager.start_process("evaluation", f"command_{args.command}")
        evaluation(gen_model, eval_model, test_dataset, result_prefix)
        eval_time = timing_stats_manager.stop_process("evaluation")
        logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    elif args.command == 'ollama-langchain':
        # 병렬 처리 매개변수 설정
        batch_size = args.batch_size
        max_concurrent = args.max_concurrent
        max_retries = args.max_retries

        logging.info(f"병렬 처리 설정: 배치 크기 {batch_size}, 최대 동시 요청 {max_concurrent}, 최대 재시도 {max_retries}")

        # 테스트 데이터셋 준비 시간 측정 (병렬 처리)
        timing_stats_manager.start_process("prepare_test_dataset", f"command_{args.command}")
        test_dataset = aux_local.prepare_test_dataset_langchain(
            gen_model,
            gen_prefix,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            test_size=test_size,
        )
        prepare_time = timing_stats_manager.stop_process("prepare_test_dataset")
        logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

        # 평가 시간 측정
        timing_stats_manager.start_process("evaluation", f"command_{args.command}")
        result_prefix = f"{prefix}_{gen_model.replace(':', '-')}_{eval_model.replace(':', '-')}"
        api_key = "crR2uHiE9awuVzimCtwmCXG6apq_rsPhHBBfjt1PSts4VmcZyLEwCJv3FFWqCD4hp20KGDL6oeT3BlbkFJBR4xBLE6TLPOwXaUdRiEgzqwE96hHs6xNKZTVXdWrEbxuqUHUZe3neqOYSrHghB8K3NOzVrXMA"
        evaluation(gen_model, eval_model, test_dataset, result_prefix, api_key=f"sk-proj-{api_key}")
        eval_time = timing_stats_manager.stop_process("evaluation")
        logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    else:
        print('사용 가능한 명령: ollama-api, ollama-http, eval-csv')
        sys.exit(1)

    # 명령어 실행 종료 및 시간 측정
    command_end_time = time.time()
    command_elapsed = command_end_time - command_start_time
    timing_stats_manager.stop_process(f"command_{args.command}")
    logging.info(f"명령어 '{args.command}' 실행 완료: {command_elapsed:.2f}초 소요")

    # 전체 실행 시간 측정 종료
    program_end_time = time.time()
    program_elapsed = program_end_time - program_start_time
    timing_stats_manager.stop_process("main")

    # 전체 시간 통계 출력
    logging.info(f"프로그램 실행 완료: 총 {program_elapsed:.2f}초 소요")
    log_timing_stats()