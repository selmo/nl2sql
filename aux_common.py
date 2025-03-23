import json

import api_request_parallel_processor
import util_common
import pandas as pd
import logging
import torch

from os import path
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import make_prompt, change_jsonl_to_csv
from util_common import check_and_create_directory, clean_filepath, make_requests_for_evaluation, make_request_jobs
from pathlib import Path

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    #, load_in_4bit)=True,
    #                                    bnb_4bit_compute_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


def prepare_train_dataset(prefix):
    filepath = path.join(prefix, "train_dataset.jsonl")

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
            df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'], llm='sqlcoder')

        df_sql.to_json(filepath, orient='records', lines=True)

    csv_filepath = path.join(prefix, "data/train.csv")

    if path.exists(csv_filepath):
        logging.info("File exists: %s", csv_filepath)
    else:
        df_sql.to_csv(csv_filepath, index=False)


def prepare_test_dataset(model, prefix=''):
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
        for idx, row in df.iterrows():
            prompt = make_prompt(row['context'], row['question'], llm='sqlcoder')
            df.loc[idx, 'prompt'] = prompt

        # sql 생성
        hf_pipe = make_inference_pipeline(model)

        gen_sqls = hf_pipe(
            df['prompt'].tolist(),
            do_sample=False,
            return_full_text=False,
            # max_length=512,
            truncation=True
        )
        gen_sqls = [x[0][('generated_text')] for x in gen_sqls]
        df['gen_sql'] = gen_sqls
        logging.info(f"Data Columns: {df.keys()}")
        logging.info(f"Data: {df}")
        df.to_json(df_filepath, orient='records', lines=True)
        logging.info(f"Data file saved: {df_filepath}")

    return df


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


def evaluation(ft_model, verifying_model, dataset, prefix, api_key=""):
    eval_filepath = "text2sql.jsonl"

    logging.info("DataFrame:\n%s", dataset)
    logging.info("Evaluation file path: %s", eval_filepath)

    requests_path = path.join(prefix, 'requests')
    if not Path(requests_path).exists():
        Path(requests_path).mkdir(parents=True)
    results_path = path.join(prefix, 'results')
    requests_filepath = clean_filepath(eval_filepath, prefix=requests_path)
    save_filepath = clean_filepath(eval_filepath, prefix=results_path)
    output_file = clean_filepath(f"{ft_model}.csv", prefix=results_path)
    check_and_create_directory(path.dirname(requests_filepath))
    check_and_create_directory(path.dirname(save_filepath))
    check_and_create_directory(path.dirname(output_file))

    # 평가를 위한 requests.jsonl 생성
    prompts = make_requests_for_evaluation(dataset)

    jobs = make_request_jobs(verifying_model, prompts)

    with open(requests_filepath, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

    url = "https://api.openai.com/v1/chat/completions" if verifying_model.lower().startswith(
        'gpt') or verifying_model.startswith('o1') or verifying_model.startswith('o3') else "http://172.16.15.112:11434/api/chat"

    logging.info('URL: %s', url)
    api_request_parallel_processor.process_by_file(
        requests_filepath=requests_filepath,
        save_filepath=save_filepath,
        request_url=url,
        api_key=api_key,
        max_requests_per_minute=2500,
        max_tokens_per_minute=100000,
        token_encoding_name="cl100k_base",
        max_attempts=10,
        logging_level=20
    )
    base_eval = change_jsonl_to_csv(
        save_filepath,
        output_file,
        # "prompt",
        response_column="resolve_yn",
        model=verifying_model
    )

    base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]

    logging.info("Evaluation CSV:\n%s", base_eval)
    logging.info("Number of correct answers: %s", num_correct_answers)