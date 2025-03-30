from langchain_core.output_parsers import PydanticOutputParser
from pandas import DataFrame
from pydantic import BaseModel, Field

from llms.prompts import template, translation_template, evaluation_template
from util.config import BatchMode
from util.util_common import is_gpt_model


# 기존 SQL 변환 관련 코드는 유지
class SQL(BaseModel):
    reasoning: str = Field(description="your chain-of-thought or reasoning")
    description: str = Field(description="a short high-level description")
    gen_sql: str = Field(description="the final SQL")


sql_parser = PydanticOutputParser(pydantic_object=SQL)


def make_prompt(model: str, data, options=None):
    """
    다양한 모드를 지원하는 확장된 프롬프트 생성 함수

    Args:
        model: 모델 이름
        data: 입력 데이터 (Series 또는 dict)
        batch_mode: 배치 처리 모드 (BatchMode enum)
        options: 추가 옵션 (dict)

    Returns:
        str: 생성된 프롬프트
    """
    if options is None:
        options = {}

    batch_mode = options.get('mode', BatchMode.NL2SQL)
    evaluation = options.get('evaluation', False)
    field_question = options.get('question_column', 'question')
    field_answer = options.get('answer_column', 'answer')

    # NL2SQL 모드 (기존 로직)
    if batch_mode == BatchMode.NL2SQL:
        if evaluation:
            return evaluation_template.format(
                schema=data.get('context', ''),
                question=data.get(field_question, ''),
                gt_sql=data.get(field_answer, ''),
                gen_sql=data.get('gen_sql', '')
            )
        else:
            return template.format(
                schema=data.get('context', ''),
                question=data.get(field_question, ''),
                format_instructions=options.get('format_instructions', '')
            )
    # 번역 모드
    elif batch_mode == BatchMode.TRANSLATION:
        input_column = options.get('input_column', 'text')
        text = data.get(input_column, '')

        return translation_template.format(
            question=text,
            answer=data.get('answer', ''),
        )
    else:
        raise ValueError(f"지원되지 않는 모드: {batch_mode}")


def make_prompts(dataset: DataFrame, model: str, options=None):
    """
    데이터셋으로부터 프롬프트 리스트 생성

    Args:
        dataset: 데이터셋
        model: 모델 이름
        options: 추가 옵션

    Returns:
        list: 프롬프트 리스트
    """
    if options is None:
        options = {}

    prompts = []

    for _, data in dataset.iterrows() if hasattr(dataset, 'iterrows') else enumerate(dataset):
        prompt = make_prompt(model, data, options)
        prompts.append(prompt)

    return prompts


def make_request(model: str, prompt: str, options=None):
    """
    다양한 모드를 지원하는 요청 객체 생성 함수

    Args:
        model: 모델 이름
        prompt: 생성된 프롬프트
        batch_mode: 배치 처리 모드
        options: 추가 옵션

    Returns:
        dict: 요청 객체
    """
    if options is None:
        options = {}

    batch_mode = options.get('mode', BatchMode.NL2SQL)

    # JSON 응답 포맷 요구 여부
    requires_json = batch_mode == BatchMode.NL2SQL

    if is_gpt_model(model):
        request = {
            "model": model,
            "messages": [{"role": "system", "content": prompt}]
        }

        # NL2SQL 모드는 JSON 응답 필요
        if requires_json:
            request["response_format"] = {"type": "json_object"}

        return request

    # logging.info('batch_mode: %s', batch_mode)

    # Ollama 모델용 요청 형식
    if batch_mode == BatchMode.NL2SQL:
        if 'evaluation' in options and options['evaluation']:
            # 평가 모드
            format_obj = {
                "type": "object",
                "properties": {
                    "resolve_yn": {"type": "string"}
                },
                "required": ["resolve_yn"]
            }
        elif options.get('only_sql', True):
            # SQL만 반환
            format_obj = {
                "type": "object",
                "properties": {
                    "gen_sql": {"type": "string"}
                },
                "required": ["gen_sql"]
            }
        else:
            # 추론 과정 포함
            format_obj = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "description": {"type": "string"},
                    "gen_sql": {"type": "string"}
                },
                "required": ["reasoning", "description", "gen_sql"]
            }

        return {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "format": format_obj
        }
    elif batch_mode == BatchMode.TRANSLATION:
        return {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "format": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "sql_query": {"type": "string"},
                },
                # "required": ["question"]
                "required": ["question", "sql_query"]
            }
        }
    else:
        # 다른 모드는 일반 텍스트 응답
        return {
            "model": model,
            "stream": False,
            "prompt": prompt
        }


def make_request_jobs(model: str, dataset: DataFrame, options=None):
    """
    데이터셋으로부터 요청 작업 리스트 생성

    Args:
        model: 모델 이름
        dataset: 데이터셋
        mode: 배치 처리 모드
        options: 추가 옵션

    Returns:
        list: 요청 작업 리스트
    """
    if options is None:
        options = {}

    jobs = []

    for _, data in dataset.iterrows() if hasattr(dataset, 'iterrows') else enumerate(dataset):
        prompt = make_prompt(model, data, options)
        request = make_request(model, prompt, options)
        jobs.append(request)

    return jobs