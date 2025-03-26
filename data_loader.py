import logging
import pandas as pd
from os.path import join as join
from pathlib import Path
from datasets import load_dataset

import util_common
from timing_stats import timing_stats_manager
from util_common import check_and_create_directory
from llms.ollama_api import llm_invoke_parallel
from llms.response_processor import make_result


def prepare_evaluation(option):
    timing_stats_manager.start_process("prepare_test_dataset", f"command_{option.command}")
    """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""
    model_prefix = join(option.prefix, "test")
    check_and_create_directory(model_prefix)
    filepath = join(model_prefix, "saved_results.jsonl")

    # 로그 파일 설정
    # log_filepath = join(option.prefix, "test_processing.log")
    # file_handler = logging.FileHandler(log_filepath)
    # file_handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s][%(levelname)s] %(message)s'))
    #
    # root_logger = logging.getLogger()
    # root_logger.addHandler(file_handler)
    # root_logger.setLevel(logging.INFO)

    if not Path(filepath).exists():
        logging.info(f"파일이 존재하지 않습니다. 데이터 파일 생성 중: {filepath}")

        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()
        if option.test_size is not None:
            df = df[:option.test_size]

        # 프롬프트 목록 생성
        datasets = []
        for _, row in df.iterrows():
            datasets.append(row)

        # 병렬 호출 실행
        logging.info(f"총 {len(datasets)}개 데이터셋에 대한 병렬 처리를 시작합니다.")

        # 로그 파일 핸들러 설정 (DEBUG 포함 모든 레벨 기록)
        log_filepath = join(option.prefix, "test_processing.log")
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.DEBUG)  # DEBUG부터 상위 레벨까지 파일로 기록
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(name)s][%(levelname)s] %(message)s'
        ))

        root_logger = logging.getLogger()
        # 가장 낮은 레벨로 잡아야 DEBUG 메시지가 파일 핸들러까지 전달됨
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        logging.info(f"병렬 처리 로그를 {log_filepath}에 기록합니다.")
        logging.info(
            f"병렬 처리 설정: 배치 크기 {option.batch_size}, 최대 동시 요청 {option.max_concurrent}, 최대 재시도 {option.max_retries}")
        responses = llm_invoke_parallel(
            option.base_model,
            datasets,
            batch_size=option.batch_size,
            max_retries=option.max_retries,
            max_concurrent=option.max_concurrent,
            url=util_common.get_api_url(option.ollama_url, option.base_model)
        )

        # 결과 처리
        results, success_count, error_count = make_result(responses, df)

        logging.info(f"결과 처리 완료: 성공 {success_count}개, 오류 {error_count}개")

        # 결과 저장
        results.to_json(filepath, orient='records', lines=True)
        logging.info(f"파일 저장 완료: {filepath}")

        # 로그 핸들러 제거
        root_logger.removeHandler(file_handler)
    else:
        logging.info(f"파일이 존재합니다. 데이터 파일 로딩 중: {filepath}")
        results = pd.read_json(filepath, lines=True)
        logging.info(f"데이터 컬럼: {results.keys()}")
        logging.info("파일 로딩 완료.")

    prepare_time = timing_stats_manager.stop_process("prepare_test_dataset")
    logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

    return results
