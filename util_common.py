import json
import os
from os import path
from pathlib import Path


def check_and_create_directory(filepath):
    if not Path(filepath).exists():
        Path(filepath).mkdir(parents=True)
    return filepath


def clean_filepath(filepath, prefix=''):
    filepath = path.join(prefix, filepath)
    if path.exists(filepath):
        os.remove(filepath)

    return filepath


def make_request_jobs(model, prompts):
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
    return jobs


def make_requests_for_evaluation(df, directory='requests'):
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True)
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

    return prompts


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