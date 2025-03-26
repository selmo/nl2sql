import logging
import json
import pandas as pd
import api_request_parallel_processor
import util_common

from os.path import join, dirname
from pathlib import Path
from config import get_apikey
from timing_stats import timing_stats_manager
from util_common import clean_filepath, check_and_create_directory
from llms.prompt_generator import make_request_jobs
from utils import change_jsonl_to_csv, load_csv


def perform_evaluation(option, dataset):
    # 평가 시간 측정
    model = option.verifying_model
    result_prefix = join(option.prefix, 'eval')
    if not Path(result_prefix).exists():
        Path(result_prefix).mkdir(parents=True)
    timing_stats_manager.start_process("evaluation", f"command_{option.command}")

    eval_filename = "text2sql"
    logging.info("DataFrame:\n%s", dataset)
    logging.info("Evaluation file path: %s", eval_filename)

    # requests_filepath = clean_filepath(f"{eval_filename}_requests.jsonl", prefix=result_prefix)
    # save_filepath = clean_filepath(f"{eval_filename}_results.jsonl", prefix=result_prefix)
    # output_file = clean_filepath(f"{eval_filename}_{option.base_model}-{model}.csv", prefix=result_prefix)
    requests_filepath = join(result_prefix, f"{eval_filename}_requests.jsonl")
    save_filepath = join(result_prefix, f"{eval_filename}_results.jsonl")
    output_file = join(result_prefix, f"{eval_filename}_{option.base_model}-{model}.csv")

    # 로그 파일 핸들러 설정 (DEBUG 포함 모든 레벨 기록)
    log_filepath = join(option.prefix, "eval_processing.log")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)  # DEBUG부터 상위 레벨까지 파일로 기록
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(name)s][%(levelname)s] %(message)s'
    ))

    root_logger = logging.getLogger()
    # 가장 낮은 레벨로 잡아야 DEBUG 메시지가 파일 핸들러까지 전달됨
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    if not Path(requests_filepath).exists():
        # 평가를 위한 requests.jsonl 생성
        jobs = make_request_jobs(option.verifying_model, dataset, evaluation=True)
        # logging.info("jobs: %s", jobs)

        with open(requests_filepath, "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")

    if not Path(save_filepath).exists():
        api_request_parallel_processor.process_by_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=util_common.get_api_url(option.ollama_url, option.verifying_model),
            api_key=get_apikey(),
            max_requests_per_minute=2500,
            max_tokens_per_minute=100000,
            token_encoding_name="cl100k_base",
            max_attempts=10,
            logging_level=20
        )

    if not Path(output_file).exists():
        base_eval = change_jsonl_to_csv(
            save_filepath,
            output_file,
            response_column="resolve_yn",
            model=model
        )
    else:
        base_eval = pd.read_csv(output_file)

    base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]

    logging.info("Evaluation CSV:\n%s", base_eval)
    logging.info("Number of correct answers: %s", num_correct_answers)

    eval_time = timing_stats_manager.stop_process("evaluation")
    logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    root_logger.removeHandler(file_handler)

