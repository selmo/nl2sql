import pandas as pd
import json
import logging
import re
from os.path import join
from pathlib import Path

from huggingface_hub import login
from pandas import DataFrame

from utils.config import BatchMode


def load_dataset(options):
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

        return df, dataset_path

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

        return df, dataset_path


def get_options(options, dataset_path):
    dataset_path = dataset_path.split(':')[0]
    # 옵션에 모드 명시적 설정
    if dataset_path.endswith('/synthetic_text_to_sql'):
        ds_options = {
            'context_column': 'sql_context',
            'question_column': 'sql_prompt',
            'answer_column': 'sql',
        }
    else:
        ds_options = {
            'context_column': options.answer_column or 'context',
            'question_column': options.question_column or 'question',
            'answer_column': options.answer_column or 'answer',
        }

    ds_options['mode'] = BatchMode.NL2SQL  # 열거형 직접 사용
    ds_options['input_column'] = options.input_column or 'input'
    ds_options['output_column'] = options.output_column or 'gen_sql'
    ds_options['batch_size'] = options.batch_size
    ds_options['max_retries'] = options.max_retries
    ds_options['max_concurrent'] = options.max_concurrent

    return ds_options


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


def change_jsonl_to_csv(input_file, prompt_column="prompt", response_column="response", model="gpt"):
    prompts = []
    responses = []
    task_ids = []
    prompts_and_responses = []

    with open(input_file, 'r') as json_file:
        # logging.info(f'input_file: {input_file}')
        for line in json_file:
            # logging.info(f"line: {line}")
            try:
                json_data = json.loads(line)
                # logging.info(f"json_data: {json_data}\n\n")

                # 프롬프트 추출
                if len(json_data) >= 1 and isinstance(json_data[0], dict):
                    if 'messages' in json_data[0]:
                        # OpenAI 형식
                        prompt = json_data[0]['messages'][0]['content']
                    elif 'prompt' in json_data[0]:
                        # Ollama 형식
                        prompt = json_data[0]['prompt']
                    else:
                        prompt = str(json_data[0])
                else:
                    prompt = ""

                prompts.append(prompt)
                prompt_and_response = {prompt_column: prompt}

                # 응답 추출
                if len(json_data) >= 2 and isinstance(json_data[1], dict):
                    # resolve_yn 처리 로직 개선
                    if response_column == "resolve_yn":
                        if 'resolve_yn' in json_data[1]:
                            # 응답 처리 함수에서 직접 처리된 경우
                            resolve_yn = json_data[1]['resolve_yn']
                            # 소문자로 통일하고 공백 제거
                            response = str(resolve_yn).lower().strip()
                        else:
                            # OpenAI 응답 형식에서 추출
                            if 'choices' in json_data[1] and len(json_data[1]['choices']) > 0:
                                content = json_data[1]['choices'][0]['message']['content']
                                # JSON 추출 시도
                                try:
                                    parsed = json.loads(content)
                                    if 'resolve_yn' in parsed:
                                        response = parsed['resolve_yn'].lower().strip()
                                    else:
                                        # 정규식으로 찾기
                                        match = re.search(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', content,
                                                          re.IGNORECASE)
                                        if match:
                                            response = match.group(1).lower().strip()
                                        else:
                                            # 간단한 yes/no 포함 여부
                                            if "yes" in content.lower():
                                                response = "yes"
                                            elif "no" in content.lower():
                                                response = "no"
                                            else:
                                                response = "unknown"
                                except json.JSONDecodeError:
                                    # 정규식으로 찾기
                                    match = re.search(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', content, re.IGNORECASE)
                                    if match:
                                        response = match.group(1).lower().strip()
                                    else:
                                        # 간단한 yes/no 포함 여부
                                        if "yes" in content.lower():
                                            response = "yes"
                                        elif "no" in content.lower():
                                            response = "no"
                                        else:
                                            response = "unknown"

                            # Ollama 응답 형식에서 추출
                            elif 'response' in json_data[1]:
                                content = json_data[1]['response']
                                # JSON 추출 시도
                                try:
                                    parsed = json.loads(content)
                                    if 'resolve_yn' in parsed:
                                        response = parsed['resolve_yn'].lower().strip()
                                    else:
                                        # 정규식으로 찾기
                                        match = re.search(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', content,
                                                          re.IGNORECASE)
                                        if match:
                                            response = match.group(1).lower().strip()
                                        else:
                                            # 간단한 yes/no 포함 여부
                                            if "yes" in content.lower():
                                                response = "yes"
                                            elif "no" in content.lower():
                                                response = "no"
                                            else:
                                                response = "unknown"
                                except json.JSONDecodeError:
                                    # 정규식으로 찾기
                                    match = re.search(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', content, re.IGNORECASE)
                                    if match:
                                        response = match.group(1).lower().strip()
                                    else:
                                        # 간단한 yes/no 포함 여부
                                        if "yes" in content.lower():
                                            response = "yes"
                                        elif "no" in content.lower():
                                            response = "no"
                                        else:
                                            response = "unknown"
                            else:
                                response = "unknown"
                    else:
                        # 일반 응답 처리
                        if 'choices' in json_data[1] and len(json_data[1]['choices']) > 0:
                            # OpenAI 응답
                            response = json_data[1]['choices'][0]['message']['content']
                        elif 'response' in json_data[1]:
                            # Ollama 응답
                            response = json_data[1]['response']
                        else:
                            # 그 외 응답
                            response = str(json_data[1])
                else:
                    response = ""

                responses.append(response)
                prompt_and_response[response_column] = response

                # 응답 추출
                if len(json_data) >= 3 and isinstance(json_data[2], dict):
                    task_id = json_data[2]['task_id']
                else:
                    task_id = ''

                task_ids.append(task_id)

                prompt_and_response['task_id'] = task_id
                prompts_and_responses.append(prompt_and_response)

            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 오류: {e}, 라인: {line}")
                prompts.append("")
                responses.append("error")
                task_ids.append("error")

                prompt_and_response[prompt_column] = prompt
                prompt_and_response[response_column] = response
                prompt_and_response['task_id'] = task_id
                prompts_and_responses.append(prompt_and_response)

            except Exception as e:
                logging.error(f"일반 오류: {e}, 라인: {line}")
                prompts.append("")
                responses.append("error")
                task_ids.append("error")

                prompt_and_response[prompt_column] = prompt
                prompt_and_response[response_column] = response
                prompt_and_response['task_id'] = task_id
                prompts_and_responses.append(prompt_and_response)

    # 로깅 추가
    logging.info(f"JSONL 파일에서 {len(prompts)}개 항목 추출")

    # response_column이 "resolve_yn"인 경우 값 분포 로깅
    if response_column == "resolve_yn":
        yes_count = sum(1 for r in responses if r == "yes")
        no_count = sum(1 for r in responses if r == "no")
        unknown_count = sum(1 for r in responses if r != "yes" and r != "no")
        logging.info(f"resolve_yn 값 분포: yes={yes_count}, no={no_count}, 기타={unknown_count}")

    # dfs = pd.DataFrame({'task_id': task_ids, prompt_column: prompts, response_column: responses})
    dfs = pd.DataFrame(prompts_and_responses)

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


def extract_sql_queries(text):
    """텍스트에서 SQL 쿼리를 추출하는 통합 유틸리티 함수"""
    # 1) ```sql ... ``` 패턴
    pattern_triple_backticks_sql = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL)

    # 2) ``` ... ``` 패턴 (언어 지정 없음)
    pattern_backticks = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)

    # 3) SELECT ... 패턴
    pattern_select = re.compile(r"\bSELECT\b.+?(?:;|$)", re.DOTALL | re.IGNORECASE)

    # 1) SQL 코드 블록 패턴
    matches = pattern_triple_backticks_sql.findall(text)
    if matches:
        return matches[0].strip()

    # 2) 일반 코드 블록 패턴
    matches = pattern_backticks.findall(text)
    if matches:
        # 중첩된 경우 가장 긴 것 선택
        longest_match = max(matches, key=len)
        return longest_match.strip()

    # 3) SELECT 문 패턴
    matches = pattern_select.findall(text)
    if matches:
        return matches[0].strip()

    # 4) JSON 객체에서 추출 시도
    try:
        json_obj = json.loads(text)
        if isinstance(json_obj, dict) and 'gen_sql' in json_obj:
            return json_obj['gen_sql']
    except (json.JSONDecodeError, TypeError):
        pass

    return ""


def extract_resolve_yn_from_text(content):
    """텍스트에서 resolve_yn 값 추출"""
    # 1. JSON 블록 추출 시도
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)

    # 2. JSON 파싱 시도
    try:
        json_data = json.loads(content)
        if 'resolve_yn' in json_data:
            return {"resolve_yn": json_data['resolve_yn'].lower().strip()}
    except json.JSONDecodeError:
        pass

    # 3. 직접 키-값 패턴 검색
    if re.search(r'[\'\"]resolve_yn[\'\"]:\s*[\'\"]yes[\'\"]', content, re.IGNORECASE):
        return {"resolve_yn": "yes"}
    elif re.search(r'[\'\"]resolve_yn[\'\"]:\s*[\'\"]no[\'\"]', content, re.IGNORECASE):
        return {"resolve_yn": "no"}

    # 4. 단순 텍스트 검색
    pattern = re.compile(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', re.IGNORECASE)
    match = pattern.search(content)
    if match:
        return {"resolve_yn": match.group(1).lower().strip()}

    # 5. 단순 yes/no 단어 검색
    if re.search(r'\byes\b', content, re.IGNORECASE):
        return {"resolve_yn": "yes"}
    elif re.search(r'\bno\b', content, re.IGNORECASE):
        return {"resolve_yn": "no"}

    # 결정할 수 없는 경우
    return {"resolve_yn": "unknown"}
