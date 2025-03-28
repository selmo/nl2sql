import time
import logging
import sys
import os.path as path
import evaluator
from evaluator import prepare_finetuning, merge_model
from util import config
from util.timing_stats import timing_stats_manager, log_timing_stats
from util.util_common import check_and_create_directory, autotrain


def setup_root_logger():
    # 콘솔 핸들러 설정 (INFO 이상만 콘솔로 출력)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(name)s][%(levelname)s] %(message)s'
    ))

    # 루트 로거에 콘솔 핸들러만 등록
    root_logger = logging.getLogger()
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG)  # 모든 레벨 처리
    root_logger.addHandler(console_handler)


# 함수별 파일 핸들러 추가 함수
def add_file_handler(log_filepath, level=logging.DEBUG):
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(name)s][%(levelname)s] %(message)s'
    ))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    return file_handler


if __name__ == "__main__":
    args = config.parse_arguments()
    program_start_time = time.time()
    timing_stats_manager.start_process("main")

    check_and_create_directory(args.prefix)

    # 루트 로거 초기 설정
    setup_root_logger()
    add_file_handler(path.join(args.prefix, "nl2sql.log"))
    command_start_time = time.time()
    timing_stats_manager.start_process(f"command_{args.command}", "main")


    if args.command == 'train':
        datapath = path.join(args.prefix, 'data')
        check_and_create_directory(datapath)

        timing_stats_manager.start_process("prepare_train_dataset", f"command_{args.command}")
        prepare_finetuning(args)
        timing_stats_manager.stop_process("prepare_train_dataset")

        # 모델 학습 시간 측정
        timing_stats_manager.start_process("autotrain", f"command_{args.command}")
        autotrain(args,
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

    elif args.command == 'merge':
        # 모델 병합 시간 측정
        timing_stats_manager.start_process("merge_model", f"command_{args.command}")
        merge_model(args.base_model, args.finetuned_model, args.prefix)
        timing_stats_manager.stop_process("merge_model")

    elif args.command == 'test' or args.command == 'eval':
        # 테스트 데이터셋 준비 시간 측정
        timing_stats_manager.start_process("prepare_test_dataset", f"command_{args.command}")
        test_dataset = evaluator.prepare_evaluation_hf(args.base_model, args.prefix, args.test_size)
        prepare_time = timing_stats_manager.stop_process("prepare_test_dataset")
        logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

        evaluator.perform_evaluation(args, test_dataset)

    elif args.command == 'ollama-api':
        test_dataset = evaluator.prepare_evaluation(args)
        evaluator.perform_evaluation(args, test_dataset)

    else:
        print('사용 가능한 명령: train, merge, test, eval, ollama-api')
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