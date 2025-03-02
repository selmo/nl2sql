from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import api_request_parallel_processor
import utils
import json
import logging

import os
import os.path as path
import pandas as pd
import torch
from pathlib import Path

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def autotrain(
        model: str,
        project_name: str,
        data_path: str,
        text_column: str,
        lr: float,
        batch_size: int,
        epochs: int,
        block_size: int,
        warmup_ratio: float,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        weight_decay: float,
        gradient_accumulation: int,
        mixed_precision: str,
        peft: bool,
        quantization: str,
        trainer: str
):
    args = [
        "llm",
        "--train",
        "--model", model,
        "--project_name", project_name,
        "--data_path", data_path,
        "--text_column", text_column,
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--block_size", str(block_size),
        "--warmup_ratio", str(warmup_ratio),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--weight_decay", str(weight_decay),
        "--gradient_accumulation", str(gradient_accumulation),
        "--mixed_precision", str(mixed_precision),
        "--peft",
        "--quantization", str(quantization),
        "--trainer", str(trainer)
    ]
    # import subprocess
    # process = subprocess.Popen(["autotrain"] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    path = "/mnt/drv21/gtone/.conda/envs/nl2sql/bin/autotrain"
    import os
    os.execv(path, [path] + args)


def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.float16)
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
            df_sql.loc[idx, 'text'] = utils.make_prompt(row['context'], row['question'], row['answer'], llm='sqlcoder')

        df_sql.to_json(filepath, orient='records', lines=True)

    csv_filepath = path.join(prefix, "data/train.csv")

    if path.exists(csv_filepath):
        logging.info("File exists: %s", csv_filepath)
    else:
        df_sql.to_csv(csv_filepath, index=False)

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
        # epochs=5,
        # lora_r=16,
        # lora_alpha=32,
        epochs=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        weight_decay=0.01,
        mixed_precision='fp16',
        peft=True,
        quantization='int4',
        trainer='sft',
    )


def prepare_test_dataset(model, prefix=''):
    df_filepath = path.join(prefix, f"{model}.jsonl")

    if path.exists(df_filepath):
        logging.info("File exists. Loading file: %s", df_filepath)
        df = pd.read_json(df_filepath, lines=True)
    else:
        logging.info(f"File not exists. Creating data file: {df_filepath}")
        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()
        for idx, row in df.iterrows():
            prompt = utils.make_prompt(row['context'], row['question'], llm='sqlcoder')
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
        df.to_json(df_filepath, orient='records', lines=True)

    return df


def make_requests_for_evaluation(df, filename, dir='requests', model="gpt-4o-mini"):
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)
    prompts = []
    for idx, row in df.iterrows():
        prompts.append(
            """Based on below DDL and Question, evaluate gen_sql can resolve Question. 
If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" +
            f"""

DDL: {row['context']}
Question: {row['question']}
gt_sql: {row['answer']}
gen_sql: {row['gen_sql']}"""
        )

    if model.startswith('gpt'):
        jobs = [{"model": model,
                 "response_format": {
                     "type": "json_object"
                 },
                 "messages": [
                     {"role": "system",
                      "content": prompt}
                 ]
                 } for prompt in prompts
                ]
    else:
        jobs = [{"model": model,
                 "stream": False,
                 "messages": [
                     {"role": "system",
                      "content": prompt}
                 ],
                 "format": {
                     "type": "object",
                     "properties": {
                         "resolve_yn": {
                             "type": "string"
                         }
                     },
                     "required": [
                         "resolve_yn"
                     ]
                 }
                 } for prompt in prompts
                ]
    with open(Path(dir, filename), "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


def check_and_create(filepath):
    if not Path(filepath).exists():
        Path(filepath).mkdir(parents=True)


def clean_filepath(filepath, prefix=''):
    filepath = path.join(prefix, filepath)
    if path.exists(filepath):
        os.remove(filepath)


def evaluation(ft_model, verifying_model, dataset, prefix):
    eval_filepath = "text2sql.jsonl"

    logging.info("DataFrame:\n%s", dataset)
    logging.info("Evaluation file path: %s", eval_filepath)

    clean_filepath(eval_filepath, prefix=path.join(prefix, 'requests'))
    clean_filepath(eval_filepath, prefix=path.join(prefix, 'results'))
    clean_filepath(f"{ft_model}.csv", prefix=path.join(prefix, 'results'))

    # 평가를 위한 requests.jsonl 생성
    make_requests_for_evaluation(dataset, eval_filepath, model=verifying_model)

    url = "https://api.openai.com/v1/chat/completions" if verifying_model.lower().startswith(
        'gpt') else "http://172.16.15.112:11434/api/chat"

    api_request_parallel_processor.process(
        requests_filepath=f"requests/{eval_filepath}",
        save_filepath=f"results/{eval_filepath}",
        request_url=url,
        max_requests_per_minute=2500,
        max_tokens_per_minute=100000,
        token_encoding_name="cl100k_base",
        max_attempts=10,
        logging_level=20
    )
    base_eval = utils.change_jsonl_to_csv(
        f"results/{eval_filepath}",
        f"results/{ft_model}.csv",
        # "prompt",
        response_column="resolve_yn",
        model=verifying_model
    )

    base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]

    logging.info("Evaluation CSV:\n%s", base_eval)
    logging.info("Number of correct answers: %s", num_correct_answers)


def merge_model(base_model, finetuned_model, prefix=''):
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


import sys

if __name__ == "__main__":
    prefix = "20250302"
    datapath = path.join(prefix, 'data')

    base_model = 'defog/sqlcoder-7b-2'
    finetuned_model = "sqlcoder-finetuned"
    verifying_model = "llama3.3:70b"

    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            check_and_create(datapath)
            prepare_train_dataset(prefix)
        elif sys.argv[1] == 'test' or sys.argv[1] == 'eval':
            merge_model(base_model, finetuned_model)
            test_dataset = prepare_test_dataset(finetuned_model, prefix)
            evaluation(finetuned_model, verifying_model, test_dataset, prefix)
        else:
            print('Arg:\n\ttrain: Finetuning model\n\ttest|eval: Evaluation model')
