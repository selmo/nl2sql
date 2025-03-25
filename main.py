import time
import argparse
import aux_local
import logging
import sys
import os.path as path

from aux_common import prepare_train_dataset, prepare_test_dataset, merge_model, evaluation
from timing_stats import timing_stats_manager, log_timing_stats
from util_common import check_and_create_directory, autotrain
from utils import load_csv

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='모델 학습 및 평가 도구')

    # 공통 인수
    parser.add_argument('command', choices=['train', 'test', 'eval', 'ollama-langchain', 'ollama-api', 'ollama-http', 'eval-csv'],
                        help='실행할 명령 (train, test, eval, ollama-api, ollama-http, eval-csv)')
    parser.add_argument('--prefix', type=str, default=".",
                        help='실행 데이터 디렉토리 접두사')
    parser.add_argument('--ollama-url', type=str, default="172.16.15.112",
                        help='ollama server 주소')
    parser.add_argument('--base-model', type=str, default='qwq',
                        help='기본 모델 이름')
    parser.add_argument('--finetuned-model', type=str, default="",
                        help='파인튜닝된 모델 이름')
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

    # eval-csv 명령에 대한 추가 인수
    parser.add_argument('--csv-path', type=str, help='평가할 CSV 파일 경로 (eval-csv 명령 시 필수)')

    return parser.parse_args()


# # base_model = 'defog/sqlcoder-7b-2'
# # finetuned_model = "sqlcoder-finetuned"
# verifying_model = "deepseek-r1:70b"  # "llama3.3:70b"
# base_model = ''  # 'qwq'
# finetuned_model = "Qwen/QwQ-32B-AWQ"  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# main routine
if __name__ == "__main__":
    args = parse_arguments()
    prefix = args.prefix

    base_model = args.base_model
    finetuned_model = args.finetuned_model
    verifying_model = args.verifying_model
    test_size = args.test_size

    program_start_time = time.time()
    timing_stats_manager.start_process("main")

    command_start_time = time.time()
    timing_stats_manager.start_process(f"command_{args.command}", "main")

    if args.command == 'train':
        datapath = path.join(prefix, 'data')
        check_and_create_directory(datapath)

        timing_stats_manager.start_process("prepare_train_dataset", f"command_{args.command}")
        prepare_train_dataset(prefix)
        timing_stats_manager.stop_process("prepare_train_dataset")

        # 모델 학습 시간 측정
        timing_stats_manager.start_process("autotrain", f"command_{sys.argv[1]}")
        autotrain(
            base_model,
            finetuned_model,
            data_path=path.join(prefix, 'data'),
            text_column='text',
            lr=2e-4,
            batch_size=14,
            gradient_accumulation=5,
            block_size=1024,
            warmup_ratio=0.1,
            epochs=5,
            lora_r=16,
            lora_alpha=32,
            # epochs=1,
            # lora_r=8,
            # lora_alpha=16,
            lora_dropout=0.05,
            weight_decay=0.01,
            mixed_precision='fp16',
            peft=True,
            quantization='int4',
            trainer='sft',
        )
        timing_stats_manager.stop_process("autotrain")

    elif args.command == 'test' or args.command == 'eval':
        # finetuned_model = "LGAI-EXAONE/EXAONE-Deep-32B-AWQ"
        # 모델 병합 시간 측정
        timing_stats_manager.start_process("merge_model", f"command_{args.command}")
        merge_model(base_model, finetuned_model, prefix)
        timing_stats_manager.stop_process("merge_model")

        # 테스트 데이터셋 준비 시간 측정
        timing_stats_manager.start_process("prepare_test_dataset", f"command_{args.command}")
        test_dataset = aux_local.prepare_test_dataset_origin(base_model, prefix, test_size)
        prepare_time = timing_stats_manager.stop_process("prepare_test_dataset")
        logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

        # 평가 시간 측정
        timing_stats_manager.start_process("evaluation", f"command_{args.command}")
        evaluation(finetuned_model, verifying_model, test_dataset, prefix)
        eval_time = timing_stats_manager.stop_process("evaluation")
        logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    elif args.command == 'ollama-api':
        # 병렬 처리 매개변수 설정
        batch_size = args.batch_size
        max_concurrent = args.max_concurrent
        max_retries = args.max_retries

        logging.info(f"병렬 처리 설정: 배치 크기 {batch_size}, 최대 동시 요청 {max_concurrent}, 최대 재시도 {max_retries}")

        # 모델 병합 시간 측정
        timing_stats_manager.start_process("merge_model", f"command_{args.command}")
        model_id, model_prefix = merge_model(base_model, finetuned_model, prefix)
        merge_time = timing_stats_manager.stop_process("merge_model")
        logging.info(f"모델 병합 완료: {merge_time:.2f}초 소요")

        # 테스트 데이터셋 준비 시간 측정 (병렬 처리)
        timing_stats_manager.start_process("prepare_test_dataset", f"command_{args.command}")
        test_dataset = aux_local.prepare_test_dataset(
            model_id,
            model_prefix,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            ollama_url=args.ollama_url,
            test_size=test_size,
        )
        prepare_time = timing_stats_manager.stop_process("prepare_test_dataset")
        logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

        # 평가 시간 측정
        timing_stats_manager.start_process("evaluation", f"command_{args.command}")
        result_prefix = f"{prefix}_{model_id.replace(':', '-')}_{verifying_model.replace(':', '-')}"
        api_key = "crR2uHiE9awuVzimCtwmCXG6apq_rsPhHBBfjt1PSts4VmcZyLEwCJv3FFWqCD4hp20KGDL6oeT3BlbkFJBR4xBLE6TLPOwXaUdRiEgzqwE96hHs6xNKZTVXdWrEbxuqUHUZe3neqOYSrHghB8K3NOzVrXMA"

        if args.eval_api:
            aux_local.evaluation_api(verifying_model, test_dataset, result_prefix, api_key=f"sk-proj-{api_key}")
        else:
            evaluation(model_id, verifying_model, test_dataset, result_prefix, api_key=f"sk-proj-{api_key}")

        eval_time = timing_stats_manager.stop_process("evaluation")
        logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    elif args.command == 'ollama-http':
        # finetuned_model = "gemma3:27b"
        # verifying_model = ""
        # prefix = f"{prefix}_{finetuned_model.replace(':', '-')}_{verifying_model}"

        # 모델 병합 시간 측정
        timing_stats_manager.start_process("merge_model", f"command_{args.command}")
        model_id, model_prefix = merge_model(base_model, finetuned_model, prefix)
        merge_time = timing_stats_manager.stop_process("merge_model")
        logging.info(f"모델 병합 완료: {merge_time:.2f}초 소요")

        # HTTP 테스트 준비 시간 측정
        timing_stats_manager.start_process("prepare_test_ollama", f"command_{args.command}")
        test_dataset = aux_local.prepare_test_ollama(model_id, model_prefix, test_size)
        prepare_time = timing_stats_manager.stop_process("prepare_test_ollama")
        logging.info(f"HTTP 테스트 준비 완료: {prepare_time:.2f}초 소요")

        # 평가 시간 측정
        timing_stats_manager.start_process("evaluation", f"command_{args.command}")
        result_prefix = f"{prefix}_{model_id.replace(':', '-')}_{verifying_model.replace(':', '-')}"
        evaluation(model_id, verifying_model, test_dataset, result_prefix)
        eval_time = timing_stats_manager.stop_process("evaluation")
        logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    elif args.command == 'eval-csv':
        if not args.csv_path:
            logging.error("eval-csv 명령에는 --csv-path 인수가 필요합니다")
            sys.exit(1)

        # CSV 로딩 시간 측정
        timing_stats_manager.start_process("load_csv", f"command_{args.command}")
        base_eval = load_csv(args.csv_path)
        load_time = timing_stats_manager.stop_process("load_csv")
        logging.info(f"CSV 로딩 완료: {load_time:.2f}초 소요")

        # 평가 분석 시간 측정
        timing_stats_manager.start_process("analyze_results", f"command_{args.command}")
        num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]
        analyze_time = timing_stats_manager.stop_process("analyze_results")

        logging.info("Evaluation CSV:\n%s", base_eval)
        logging.info("Number of correct answers: %s", num_correct_answers)
        logging.info(f"결과 분석 완료: {analyze_time:.2f}초 소요")

    elif args.command == 'ollama-langchain':
        # 병렬 처리 매개변수 설정
        batch_size = args.batch_size
        max_concurrent = args.max_concurrent
        max_retries = args.max_retries

        logging.info(f"병렬 처리 설정: 배치 크기 {batch_size}, 최대 동시 요청 {max_concurrent}, 최대 재시도 {max_retries}")

        # 모델 병합 시간 측정
        timing_stats_manager.start_process("merge_model", f"command_{args.command}")
        model_id, model_prefix = merge_model(base_model, finetuned_model, prefix)
        merge_time = timing_stats_manager.stop_process("merge_model")
        logging.info(f"모델 병합 완료: {merge_time:.2f}초 소요")

        # 테스트 데이터셋 준비 시간 측정 (병렬 처리)
        timing_stats_manager.start_process("prepare_test_dataset", f"command_{args.command}")
        test_dataset = aux_local.prepare_test_dataset_langchain(
            model_id,
            model_prefix,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            test_size=test_size,
        )
        prepare_time = timing_stats_manager.stop_process("prepare_test_dataset")
        logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

        # 평가 시간 측정
        timing_stats_manager.start_process("evaluation", f"command_{args.command}")
        result_prefix = f"{prefix}_{model_id.replace(':', '-')}_{verifying_model.replace(':', '-')}"
        # evaluation(model_id, verifying_model, test_dataset, result_prefix)
        api_key = "crR2uHiE9awuVzimCtwmCXG6apq_rsPhHBBfjt1PSts4VmcZyLEwCJv3FFWqCD4hp20KGDL6oeT3BlbkFJBR4xBLE6TLPOwXaUdRiEgzqwE96hHs6xNKZTVXdWrEbxuqUHUZe3neqOYSrHghB8K3NOzVrXMA"
        evaluation(model_id, verifying_model, test_dataset, result_prefix, api_key=f"sk-proj-{api_key}")
        eval_time = timing_stats_manager.stop_process("evaluation")
        logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    else:
        print('사용 가능한 명령: train, test, eval, ollama-api, ollama-http, eval-csv')
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