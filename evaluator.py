import logging
import json
from os import path

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import api_request_parallel_processor
from llms.ollama_api import llm_invoke_parallel, llm_invoke_jobs_parallel
from llms.response_processor import make_result
from util import util_common

from os.path import join
from pathlib import Path
from util.config import get_apikey
from util.timing_stats import timing_stats_manager
from llms.prompt_generator import make_request_jobs, make_prompt
from util.util_common import check_and_create_directory, change_jsonl_to_csv


def perform_evaluation(option, dataset):
    # 평가 시간 측정
    base_model = option.base_model
    model = option.verifying_model
    result_prefix = join(option.prefix, 'eval')
    if not Path(result_prefix).exists():
        Path(result_prefix).mkdir(parents=True)
    timing_stats_manager.start_process("perform_evaluation", f"command_{option.command}")

    requests_filepath = join(result_prefix, f"{base_model}_{model}_requests.jsonl")
    save_filepath = join(result_prefix, f"{base_model}_{model}_results.jsonl")
    output_file = join(result_prefix, f"{base_model}-{model}.csv")

    logging.info("DataFrame:\n%s", dataset)
    logging.info("Evaluation file path: %s", output_file)


    if not Path(requests_filepath).exists():
        # 평가를 위한 requests.jsonl 생성
        jobs = make_request_jobs(model, dataset, evaluation=True)

        with open(requests_filepath, "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")

    if not Path(save_filepath).exists():
        api_request_parallel_processor.process_by_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=util_common.get_api_url(option.ollama_url, model),
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

    eval_time = timing_stats_manager.stop_process("perform_evaluation")
    logging.info(f"평가 완료: {eval_time:.2f}초 소요")


def prepare_evaluation(option):
    timing_stats_manager.start_process("prepare_evaluation", f"command_{option.command}")
    """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""
    model_prefix = join(option.prefix, "test")
    check_and_create_directory(model_prefix)
    filepath = join(model_prefix, f"{option.base_model}.jsonl")

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
    else:
        logging.info(f"파일이 존재합니다. 데이터 파일 로딩 중: {filepath}")
        results = pd.read_json(filepath, lines=True)
        logging.info(f"데이터 컬럼: {results.keys()}")
        logging.info("파일 로딩 완료.")

    prepare_time = timing_stats_manager.stop_process("prepare_evaluation")
    logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

    return results


def prepare_evaluation_hf(model, prefix='', test_size=0):
    df_filepath = path.join(prefix, f"{model}.jsonl")
    util_common.check_and_create_directory(path.dirname(df_filepath))

    if path.exists(df_filepath):
        logging.info("File exists. Loading file: %s", df_filepath)
        df = pd.read_json(df_filepath, lines=True)
        logging.info(f"Data Columns: {df.keys()}")
        logging.info(f"Data: {df}")
    else:
        logging.info(f"File not exists. Creating data file: {df_filepath}")
        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()
        if test_size > 0:
            df = df[:test_size]
        for idx, row in df.iterrows():
            prompt = make_prompt(model, row['context'], row['question'])
            df.loc[idx, 'prompt'] = prompt

        # sql 생성
        hf_pipe = make_inference_pipeline(model)

        gen_sqls = []
        # tqdm의 total은 전체 prompt 수
        batch_size = 10
        prompts = df['prompt'].tolist()
        from tqdm import tqdm
        for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
            batch = prompts[i: i + batch_size]
            # pipeline에 한 번에 batch_size개씩 넣어주기
            outputs = hf_pipe(
                batch,
                do_sample=False,
                return_full_text=False,
                truncation=True
            )
            # 파이프라인 반환값이 리스트 형태일 것이므로
            if batch_size == 1:
                # 단일 prompt만 줬을 경우도 리스트로 나오므로
                outputs = [outputs]
            gen_sqls.extend(outputs)

        # gen_sqls = hf_pipe(
        #     df['prompt'].tolist(),
        #     do_sample=False,
        #     return_full_text=False,
        #     # max_length=512,
        #     truncation=True
        # )
        gen_sqls = [x[0][('generated_text')] for x in gen_sqls]
        df['gen_sql'] = gen_sqls
        logging.info(f"Data Columns: {df.keys()}")
        logging.info(f"Data: {df}")
        df.to_json(df_filepath, orient='records', lines=True)
        logging.info(f"Data file saved: {df_filepath}")

    return df


def change_responses_to_csv(dataset, output_file='', prompt_column="prompt", response_column="response", model="gpt"):
    prompts = []
    responses = []

    for json_data in dataset:
        prompts.append(json_data[0]['messages'][0]['content'])
        if model.lower().startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
            responses.append(json_data[1]['choices'][0]['message']['content'])
        else:
            responses.append(json_data[1]['message']['content'])

    dfs = pd.DataFrame({prompt_column: prompts, response_column: responses})
    if not output_file == '':
        dfs.to_csv(output_file, index=False)
    return dfs


def evaluation_api(model, dataset, prefix='', batch_size=10, max_concurrent=10, max_retries=3, size=None, api_key=""):
    """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""
    check_and_create_directory(prefix)
    filepath = path.join(prefix, "text2sql.jsonl")
    requests_path = path.join(prefix, 'requests')
    if not Path(requests_path).exists():
        Path(requests_path).mkdir(parents=True)
    # requests_filepath = clean_filepath(filepath, prefix=requests_path)
    requests_filepath = join(requests_path, filepath)
    check_and_create_directory(path.dirname(requests_filepath))

    # prompts = make_prompts_for_evaluation(dataset)
    # jobs = util_common.make_request_jobs(model, prompts)
    jobs = make_request_jobs(model, dataset, evaluation=True)

    with open(requests_filepath, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

    # 로그 파일 설정
    log_filepath = path.join(prefix, "parallel_processing_eval.log")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logging.info(f"병렬 처리 로그를 {log_filepath}에 기록합니다.")

    url = "https://api.openai.com/v1/chat/completions" if model.lower().startswith(
        'gpt') or model.startswith('o1') or model.startswith('o3') else "http://172.16.15.112:11434/api/chat"

    # 데이터셋 불러오기
    if size is not None and size > 0:
        jobs = jobs[:size]

    # 병렬 호출 실행
    logging.info(f"총 {len(jobs)}개 데이터셋에 대한 병렬 처리를 시작합니다.")
    responses = llm_invoke_jobs_parallel(
        model,
        jobs,
        url,
        batch_size=batch_size,
        max_retries=max_retries,
        max_concurrent=max_concurrent
    )

    output_file = path.join(prefix, "result.csv")
    base_eval = change_responses_to_csv(
        responses,
        output_file,
        # "prompt",
        response_column="resolve_yn",
        model=model
    )

    base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]

    logging.info("Evaluation CSV:\n%s", base_eval)
    logging.info("Number of correct answers: %s", num_correct_answers)


def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    #, load_in_4bit)=True,
    #                                    bnb_4bit_compute_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


def prepare_finetuning(option):
    filepath = path.join(option.prefix, "train_dataset.jsonl")

    if path.exists(filepath):
        logging.info("File exists. Loading file: %s", filepath)
        df_sql = pd.read_json(filepath, lines=True)
    else:
        logging.info("File not exists. Creating data file.")

        df_sql = load_dataset("shangrilar/ko_text2sql", "clean")["train"]
        df_sql = df_sql.to_pandas()
        df_sql = df_sql.dropna().sample(frac=1, random_state=42)
        df_sql = df_sql.query("db_id != 1")

        for idx, row in df_sql.iterrows():
            df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'], option.base_model)

        df_sql.to_json(filepath, orient='records', lines=True)

    csv_filepath = path.join(option.prefix, "data/train.csv")

    if path.exists(csv_filepath):
        logging.info("File exists: %s", csv_filepath)
    else:
        df_sql.to_csv(csv_filepath, index=False)


def merge_model(base_model, finetuned_model, prefix=''):
    if finetuned_model == '':
        return base_model, f"{prefix}_{base_model.replace(':', '-')}"

    filepath = path.join(prefix, finetuned_model)

    if path.exists(filepath):
        logging.info(f"Skip creating merged model: {filepath}")
    else:
        logging.info("File not exists. Creating merged model.")

        model_name = base_model

        logging.info(f"AutoModelForCausalLM.from_pretrained: {model_name}")
        # LoRA와 기초 모델 파라미터 합치기
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        logging.info(f"AutoTokenizer.from_pretrained: {model_name}")
        # 토크나이저 설정
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        #
        base_model.resize_token_embeddings(len(tokenizer))

        logging.info(f"PeftModel.from_pretrained: {base_model}, {finetuned_model}")
        model = PeftModel.from_pretrained(base_model, finetuned_model)

        logging.info("start merge and unload")
        model = model.merge_and_unload()

        logging.info("start saving model and tokenizer.")
        model.save_pretrained(filepath)
        tokenizer.save_pretrained(filepath)

    return finetuned_model, f"{prefix}_{finetuned_model.replace(':', '-')}"
