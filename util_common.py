import argparse
import os
import re
from os.path import join, exists
from pathlib import Path
from llms.prompt_generator import make_request


def get_api_url(input_url: str, model: str = '') -> str:
    if model.lower().startswith('gpt') or model.lower().startswith('o1') or model.lower().startswith('o3'):
        return "https://api.openai.com/v1/chat/completions"

    # 기본값 설정
    default_protocol = "http://"
    default_port = "11434"
    api_path = "api/generate"

    # 빈 문자열일 경우 기본 URL 반환
    if not input_url:
        return f"{default_protocol}localhost:{default_port}/{api_path}"

    # URL에 프로토콜이 없는 경우 기본 프로토콜을 붙여준다
    if not re.match(r"^[a-zA-Z]+://", input_url):
        # Fix: removed the extra slash
        input_url = f"{default_protocol}{input_url}"

    # 호스트 부분 추출
    match = re.match(r"^(https?://)?([^:/]+)(?::(\d+))?", input_url)
    if match:
        protocol = match.group(1) or default_protocol
        host = match.group(2)
        port = match.group(3) or default_port
        normalized_url = f"{protocol}{host}:{port}/{api_path}"
        return normalized_url

    # 기본 처리
    return f"{default_protocol}localhost:{default_port}/{api_path}"
# def get_api_url(input_url: str, model:str = '') -> str:
#     if model.lower().startswith('gpt') or model.lower().startswith('o1') or model.lower().startswith('o3'):
#         return "https://api.openai.com/v1/chat/completions"
#
#     # 기본값 설정
#     default_protocol = "http://"
#     default_port = "11434"
#     api_path = "api/generate"
#
#     # 빈 문자열일 경우 기본 URL 반환
#     if not input_url:
#         return f"{default_protocol}localhost:{default_port}/{api_path}"
#
#     # URL에 프로토콜이 없는 경우 추가
#     if not re.match(r"^[a-zA-Z]+://", input_url):
#         input_url = f"{default_protocol}/{input_url}"
#
#     # 호스트 부분 추출
#     match = re.match(r"^(https?://)?([^:/]+)(?::(\d+))?", input_url)
#     if match:
#         protocol = match.group(1) or default_protocol
#         host = match.group(2)
#         port = match.group(3) or "11434"
#         normalized_url = f"{protocol}{host}:{port}/{api_path}"
#         return normalized_url
#
#     # 기본 처리
#     return f"{default_protocol}localhost:{default_port}/{api_path}"


def check_and_create_directory(filepath):
    if not Path(filepath).exists():
        Path(filepath).mkdir(parents=True)
    return filepath


def clean_filepath(filepath, prefix=''):
    filepath = join(prefix, filepath)
    if exists(filepath):
        os.remove(filepath)

    return filepath


def make_request_jobs(model, prompts):
    if model.startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
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
        jobs = [ make_request(model, prompt, evaluation=True)
        #   {"model": model,
        #          "stream": False,
        #          "messages": [
        #              {"role": "system",
        #               "content": prompt}
        #          ],
        #          "format": {
        #              "type": "object",
        #              "properties": {
        #                  "resolve_yn": {
        #                      "type": "string"
        #                  }
        #              },
        #              "required": [
        #                  "resolve_yn"
        #              ]
        #          }
        #          }
            for prompt in prompts
                ]
    return jobs


def make_prompts_for_evaluation(df):
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


# model: str,
# project_name: str,
# data_path: str,
def autotrain(
        option,
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
        "llms",
        "--train",
        "--model", option.base_model,
        "--project_name", option.finetuned_model,
        "--data_path", join(option.prefix, 'data'),
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