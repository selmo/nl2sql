import pandas as pd
import json
import logging
import re
from os.path import join
from pathlib import Path


def is_gpt_model(model: str):
    return (True
            if model.lower().startswith('gpt') or model.lower().startswith('o1') or model.lower().startswith('o3')
            else False)


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


def check_and_create_directory(filepath):
    if not Path(filepath).exists():
        Path(filepath).mkdir(parents=True)
    return filepath


# def clean_filepath(filepath, prefix=''):
#     filepath = join(prefix, filepath)
#     if exists(filepath):
#         os.remove(filepath)
#
#     return filepath
#
#
# def make_request_jobs(model, prompts):
#     if is_gpt_model(model):
#         jobs = [{"model": model,
#                  "response_format": {
#                      "type": "json_object"
#                  },
#                  "messages": [
#                      {"role": "system",
#                       "content": prompt}
#                  ]
#                  } for prompt in prompts
#                 ]
#     else:
#         jobs = [make_request(model, prompt, evaluation=True) for prompt in prompts]
#     return jobs
#
#
# def make_prompts_for_evaluation(df):
#     prompts = []
#     for idx, row in df.iterrows():
#         prompts.append(
#             """Based on below DDL and Question, evaluate gen_sql can resolve Question.
# If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" +
#             f"""
#
# DDL: {row['context']}
# Question: {row['question']}
# gt_sql: {row['answer']}
# gen_sql: {row['gen_sql']}"""
#         )
#
#     return prompts


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


def change_jsonl_to_csv(input_file, output_file='', prompt_column="prompt", response_column="response", model="gpt"):
    prompts = []
    responses = []

    with open(input_file, 'r') as json_file:
        for data in json_file:
            json_data = json.loads(data)
            if model.lower().startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
                prompts.append(json_data[0]['messages'][0]['content'])
                responses.append(json_data[1]['choices'][0]['message']['content'])
            else:
                prompts.append(json_data[0]['prompt'])
                responses.append(json_data[1]['response'])

    dfs = pd.DataFrame({prompt_column: prompts, response_column: responses})
    logging.info(f"change_jsonl_to_csv: input_file={input_file}, output_file={output_file}")
    if not output_file == '':
        dfs.to_csv(output_file, index=False)
    return dfs


def sanitize_filename(filename):
    """
    Windows 및 기타 OS에서 사용 불가능한 문자를 제거하거나 대체하여 안전한 파일명 생성

    Args:
        filename: 변환할 파일명 문자열

    Returns:
        안전한 파일명 문자열
    """
    # Windows에서 허용되지 않는 문자
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    # 각 문자를 대체
    safe_name = filename
    for char in invalid_chars:
        if char == ':':
            safe_name = safe_name.replace(char, '-')  # 콜론은 하이픈으로 대체
        else:
            safe_name = safe_name.replace(char, '_')  # 다른 특수문자는 언더스코어로 대체

    # 파일명 끝의 공백과 마침표 제거 (Windows에서 문제 발생 가능)
    safe_name = safe_name.rstrip('. ')

    # 파일명이 비어있거나 Windows 예약어인 경우 대체
    if not safe_name or safe_name.upper() in [
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    ]:
        safe_name = f"_{safe_name}"

    return safe_name