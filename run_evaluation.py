import time
import logging
import sys
import os
import os.path as path
import evaluator
from evaluator import prepare_finetuning, merge_model
from util import config
from util.util_common import check_and_create_directory, autotrain


def setup_root_logger():
    """루트 로거 초기화 함수"""
    # 기존에 설정된 모든 핸들러 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 루트 로거 레벨 설정
    root_logger.setLevel(logging.DEBUG)

    # 콘솔 핸들러 설정 (INFO 이상만 콘솔로 출력)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(name)s][%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 로그 메시지 필터링 설정
    class DuplicateFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self.last_log = {}

        def filter(self, record):
            # 로그 레코드의 고유 키 생성
            current_log = (record.module, record.levelno, record.getMessage())
            if current_log in self.last_log:
                return False
            self.last_log = current_log
            return True

    # 루트 로거에 중복 필터 추가
    # root_logger.addFilter(DuplicateFilter())

    return root_logger


# 함수별 파일 핸들러 추가 함수
def add_file_handler(log_filepath, level=logging.DEBUG):
    """로그 파일 핸들러 추가 함수"""
    # 절대 경로로 변환
    log_filepath = path.abspath(log_filepath)

    # 로그 디렉토리 생성
    log_dir = path.dirname(log_filepath)
    check_and_create_directory(log_dir)

    # 파일 핸들러 생성
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(name)s][%(levelname)s] %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # 루트 로거에 핸들러 추가
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # 추가된 핸들러 정보 로깅
    logging.debug(f"파일 로그 핸들러 추가됨: {log_filepath}")

    return file_handler


if __name__ == "__main__":
    args = config.parse_arguments()
    program_start_time = time.time()

    check_and_create_directory(args.prefix)

    # 로그 디렉토리 준비
    log_dir = path.join(args.prefix, "logs")
    check_and_create_directory(log_dir)

    # 절대 경로로 변환
    abs_log_path = os.path.abspath(os.path.join(log_dir, "nl2sql.log"))

    # 루트 로거 초기 설정
    setup_root_logger()

    # 파일 핸들러 추가
    log_handler = add_file_handler(abs_log_path)

    # 프로그램 시작 로그
    if args.command == 'batch':
        logging.info(f"배치 처리 프로그램 시작 (모드: {args.mode}, 로그 파일: {abs_log_path})")
    else:
        logging.info(f"NL2SQL 평가 프로그램 시작 (명령: {args.command}, 로그 파일: {abs_log_path})")

    command_start_time = time.time()

    # 프로그램 종료 시 핸들러 정리
    try:
        if args.command == 'train':
            datapath = path.join(args.prefix, 'data')
            check_and_create_directory(datapath)
            prepare_finetuning(args)

            # 모델 학습 시간 측정
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

        elif args.command == 'merge':
            merge_model(args.base_model, args.finetuned_model, args.prefix)

        elif args.command == 'test' or args.command == 'eval':
            test_dataset = evaluator.prepare_evaluation_hf(args.base_model, args.prefix, args.test_size)
            evaluator.perform_evaluation(args, test_dataset)

        elif args.command == 'ollama-api':
            test_dataset = evaluator.prepare_evaluation(args)
            evaluator.perform_evaluation(args, test_dataset)


        elif args.command == 'batch':
            # 배치 처리 실행
            results = evaluator.process_batch(args)

            # 결과 요약
            logging.info(f"배치 처리 완료: {len(results)}개 항목 처리됨")

        elif args.command == 'upload':
            # 배치 처리 실행
            evaluator.process_upload(args.from_file, args.upload_to_hf)

        else:
            print('사용 가능한 명령: train, merge, test, eval, ollama-api, batch, upload')
            sys.exit(1)
    finally:
        # 명령어 실행 종료 및 시간 측정
        command_end_time = time.time()
        command_elapsed = command_end_time - command_start_time
        logging.info(f"명령어 '{args.command}' 실행 완료: {command_elapsed:.2f}초 소요")

        # 전체 실행 시간 측정 종료
        program_end_time = time.time()
        program_elapsed = program_end_time - program_start_time

        # 전체 시간 통계 출력
        logging.info(f"프로그램 실행 완료: 총 {program_elapsed:.2f}초 소요")

        # 파일 핸들러 정리
        if log_handler:
            log_handler.close()
            logging.getLogger().removeHandler(log_handler)