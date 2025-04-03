import pandas as pd
import json
import logging
import re
from os.path import join
from pathlib import Path

from huggingface_hub import login
from pandas import DataFrame

def load_dataset(options) -> DataFrame:
    """
    옵션에 따라 데이터셋 로드

    데이터셋 지정 형식:
    1. 이름만 사용: "origin", "spider" 등
    2. 전체 경로 지정: "경로:설정:분할" (예: "shangrilar/ko_text2sql:origin:test")

    Args:
        options: 명령행 옵션

    Returns:
        DataFrame: 로드된 데이터셋
    """
    from datasets import load_dataset

    # 기본 데이터셋 설정
    dataset_path = "shangrilar/ko_text2sql"
    dataset_name = "clean"
    dataset_split = "train"

    # 테스트 데이터셋 설정 가져오기
    dataset_option = getattr(options, 'test_dataset', None)

    if dataset_option:
        parts = dataset_option.split(':')
        dataset_path = parts[0]
        dataset_name = parts[1] if len(parts) > 1 else None
        dataset_split = parts[2] if len(parts) > 2 else "test"

    try:
        # 데이터셋 로드
        logging.info(f"데이터셋 로드 중: {dataset_path}, 설정: {dataset_name}, 분할: {dataset_split}")
        df = load_dataset(dataset_path, dataset_name)[dataset_split]
        df = df.to_pandas()

        # 테스트 크기 제한 적용
        if hasattr(options, 'test_size') and options.test_size is not None:
            try:
                size = int(options.test_size)
                df = df[:size]
                logging.info(f"테스트 크기 제한 적용: {size}개 샘플")
            except ValueError:
                # test_size가 정수가 아닌 경우, 데이터셋 경로로 간주
                logging.info(f"test_size '{options.test_size}'는 숫자가 아닙니다. 데이터셋 경로로 처리합니다.")
                return load_dataset_from_path(options.test_size)

        logging.info(f"데이터셋 로드 완료: {len(df)}개 데이터")
        return df

    except Exception as e:
        logging.error(f"데이터셋 '{dataset_path}:{dataset_name}:{dataset_split}' 로드 중 오류 발생: {str(e)}")

        # 경로 문자열 형식으로 전달된 값인지 확인
        if hasattr(options, 'test_size') and options.test_size is not None:
            if not isinstance(options.test_size, (int, float)) and ':' in options.test_size:
                logging.info(f"test_size 값을 데이터셋 경로로 시도합니다: {options.test_size}")
                return load_dataset_from_path(options.test_size)

        # 기본 데이터셋으로 대체
        logging.info("기본 'shangrilar/ko_text2sql:origin:test' 데이터셋으로 대체합니다.")
        df = load_dataset("shangrilar/ko_text2sql", "origin")["test"]
        df = df.to_pandas()

        if (hasattr(options, 'test_size') and
                options.test_size is not None and
                isinstance(options.test_size, (int, float))):
            df = df[:int(options.test_size)]

        return df


def load_dataset_from_path(dataset_path_str):
    """
    문자열 경로에서 데이터셋 로드 (형식: '경로:설정:분할')

    Args:
        dataset_path_str: 데이터셋 경로 문자열

    Returns:
        DataFrame: 로드된 데이터셋
    """
    from datasets import load_dataset
    parts = dataset_path_str.split(':')
    if len(parts) < 2:
        raise ValueError(f"데이터셋 경로 형식이 잘못되었습니다: {dataset_path_str}. '경로:설정:분할' 형식이어야 합니다.")

    # 기본값 설정
    path = parts[0]
    name = parts[1] if len(parts) > 1 else None
    split = parts[2] if len(parts) > 2 else "test"

    logging.info(f"경로에서 데이터셋 로드 중: {path}, 설정: {name}, 분할: {split}")

    # name이 None이면 name 인자 없이 호출
    if name:
        df = load_dataset(path, name)[split]
    else:
        df = load_dataset(path)[split]

    return df.to_pandas()


# def load_dataset(options):
#     """
#     옵션에 따라 데이터셋 로드
#
#     Args:
#         options: 명령행 옵션
#
#     Returns:
#         DataFrame: 로드된 데이터셋
#     """
#     if options.mode == BatchMode.NL2SQL.value:
#         from datasets import load_dataset
#         df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
#         df = df.to_pandas()
#         if options.test_size is not None:
#             df = df[:options.test_size]
#         return df


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
        for line in json_file:
            try:
                json_data = json.loads(line)

                # 프롬프트 추출
                if len(json_data) >= 1 and isinstance(json_data[0], dict):
                    if 'messages' in json_data[0]:
                        # OpenAI 형식
                        prompts.append(json_data[0]['messages'][0]['content'])
                    elif 'prompt' in json_data[0]:
                        # Ollama 형식
                        prompts.append(json_data[0]['prompt'])
                    else:
                        prompts.append(str(json_data[0]))
                else:
                    prompts.append("")

                # 응답 추출
                if len(json_data) >= 2 and isinstance(json_data[1], dict):
                    # resolve_yn 처리 로직 개선
                    if response_column == "resolve_yn":
                        if 'resolve_yn' in json_data[1]:
                            # 응답 처리 함수에서 직접 처리된 경우
                            resolve_yn = json_data[1]['resolve_yn']
                            # 소문자로 통일하고 공백 제거
                            responses.append(str(resolve_yn).lower().strip())
                        else:
                            # OpenAI 응답 형식에서 추출
                            if 'choices' in json_data[1] and len(json_data[1]['choices']) > 0:
                                content = json_data[1]['choices'][0]['message']['content']
                                # JSON 추출 시도
                                try:
                                    parsed = json.loads(content)
                                    if 'resolve_yn' in parsed:
                                        responses.append(parsed['resolve_yn'].lower().strip())
                                    else:
                                        # 정규식으로 찾기
                                        match = re.search(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', content,
                                                          re.IGNORECASE)
                                        if match:
                                            responses.append(match.group(1).lower().strip())
                                        else:
                                            # 간단한 yes/no 포함 여부
                                            if "yes" in content.lower():
                                                responses.append("yes")
                                            elif "no" in content.lower():
                                                responses.append("no")
                                            else:
                                                responses.append("unknown")
                                except json.JSONDecodeError:
                                    # 정규식으로 찾기
                                    match = re.search(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', content, re.IGNORECASE)
                                    if match:
                                        responses.append(match.group(1).lower().strip())
                                    else:
                                        # 간단한 yes/no 포함 여부
                                        if "yes" in content.lower():
                                            responses.append("yes")
                                        elif "no" in content.lower():
                                            responses.append("no")
                                        else:
                                            responses.append("unknown")

                            # Ollama 응답 형식에서 추출
                            elif 'response' in json_data[1]:
                                content = json_data[1]['response']
                                # JSON 추출 시도
                                try:
                                    parsed = json.loads(content)
                                    if 'resolve_yn' in parsed:
                                        responses.append(parsed['resolve_yn'].lower().strip())
                                    else:
                                        # 정규식으로 찾기
                                        match = re.search(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', content,
                                                          re.IGNORECASE)
                                        if match:
                                            responses.append(match.group(1).lower().strip())
                                        else:
                                            # 간단한 yes/no 포함 여부
                                            if "yes" in content.lower():
                                                responses.append("yes")
                                            elif "no" in content.lower():
                                                responses.append("no")
                                            else:
                                                responses.append("unknown")
                                except json.JSONDecodeError:
                                    # 정규식으로 찾기
                                    match = re.search(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', content, re.IGNORECASE)
                                    if match:
                                        responses.append(match.group(1).lower().strip())
                                    else:
                                        # 간단한 yes/no 포함 여부
                                        if "yes" in content.lower():
                                            responses.append("yes")
                                        elif "no" in content.lower():
                                            responses.append("no")
                                        else:
                                            responses.append("unknown")
                            else:
                                responses.append("unknown")
                    else:
                        # 일반 응답 처리
                        if 'choices' in json_data[1] and len(json_data[1]['choices']) > 0:
                            # OpenAI 응답
                            responses.append(json_data[1]['choices'][0]['message']['content'])
                        elif 'response' in json_data[1]:
                            # Ollama 응답
                            responses.append(json_data[1]['response'])
                        else:
                            # 그 외 응답
                            responses.append(str(json_data[1]))
                else:
                    responses.append("")
            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 오류: {e}, 라인: {line}")
                prompts.append("")
                responses.append("error")
            except Exception as e:
                logging.error(f"일반 오류: {e}, 라인: {line}")
                prompts.append("")
                responses.append("error")

    # 로깅 추가
    logging.info(f"JSONL 파일에서 {len(prompts)}개 항목 추출")

    # response_column이 "resolve_yn"인 경우 값 분포 로깅
    if response_column == "resolve_yn":
        yes_count = sum(1 for r in responses if r == "yes")
        no_count = sum(1 for r in responses if r == "no")
        unknown_count = sum(1 for r in responses if r != "yes" and r != "no")
        logging.info(f"resolve_yn 값 분포: yes={yes_count}, no={no_count}, 기타={unknown_count}")

    dfs = pd.DataFrame({prompt_column: prompts, response_column: responses})
    logging.info(f"change_jsonl_to_csv: input_file={input_file}, output_file={output_file}")

    if output_file:
        dfs.to_csv(output_file, index=False)

    return dfs
# def change_jsonl_to_csv(input_file, output_file='', prompt_column="prompt", response_column="response", model="gpt"):
#     prompts = []
#     responses = []
#
#     with open(input_file, 'r') as json_file:
#         for data in json_file:
#             json_data = json.loads(data)
#             if model.lower().startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
#                 prompts.append(json_data[0]['messages'][0]['content'])
#                 responses.append(json_data[1]['choices'][0]['message']['content'])
#             else:
#                 prompts.append(json_data[0]['prompt'])
#                 responses.append(json_data[1]['response'])
#
#     dfs = pd.DataFrame({prompt_column: prompts, response_column: responses})
#     logging.info(f"change_jsonl_to_csv: input_file={input_file}, output_file={output_file}")
#     if not output_file == '':
#         dfs.to_csv(output_file, index=False)
#     return dfs


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


def upload_to_huggingface(data, dataset_name, token, split_name=None):
    """
    데이터프레임 또는 DatasetDict을 Hugging Face에 업로드

    Args:
        data: DataFrame 또는 DatasetDict 객체
        dataset_name: Hugging Face에 업로드할 데이터셋 이름 (e.g., 'username/dataset-name')
        token: Hugging Face API 토큰
        split_name: 단일 DataFrame을 업로드할 때의 스플릿 이름 (기본값: None)
    """
    from datasets import Dataset, DatasetDict

    # Hugging Face 로그인
    login(token)

    # 데이터 유형에 따른 처리
    if isinstance(data, pd.DataFrame):
        # 단일 DataFrame인 경우
        dataset = Dataset.from_pandas(data)

        if split_name:
            # 스플릿 이름이 지정된 경우 DatasetDict으로 업로드
            dataset_dict = DatasetDict({split_name: dataset})
            dataset_dict.push_to_hub(
                dataset_name,
                private=False,
            )
            logging.info(f"데이터셋이 '{split_name}' 스플릿으로 {dataset_name}에 업로드되었습니다.")
        else:
            # 스플릿 이름이 없는 경우 단일 데이터셋으로 업로드
            dataset.push_to_hub(
                dataset_name,
                private=False,
            )
            logging.info(f"데이터셋이 성공적으로 {dataset_name}에 업로드되었습니다.")

    elif isinstance(data, dict):
        # 스플릿별 DataFrame 딕셔너리인 경우
        dataset_dict = DatasetDict()

        # 각 스플릿 변환
        for split, df in data.items():
            if df is not None:
                dataset_dict[split] = Dataset.from_pandas(df)

        # Hugging Face에 업로드
        dataset_dict.push_to_hub(
            dataset_name,
            private=False,
        )
        logging.info(f"데이터셋 딕셔너리가 성공적으로 {dataset_name}에 업로드되었습니다.")

    elif hasattr(data, 'push_to_hub'):
        # 이미 Dataset 또는 DatasetDict인 경우
        data.push_to_hub(
            dataset_name,
            private=False,
        )
        logging.info(f"데이터셋이 성공적으로 {dataset_name}에 업로드되었습니다.")

    else:
        raise TypeError(f"지원되지 않는 데이터 유형: {type(data)}")