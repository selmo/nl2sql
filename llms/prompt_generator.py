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


def get_dbms_specific_instructions(dbms):
    """Generate DBMS-specific instructions"""
    if not dbms:
        return ""

    instructions = {
        "PostgreSQL": """
Special Instructions (PostgreSQL):
- Use PostgreSQL syntax
- Use 'YYYY-MM-DD' format for dates
- Use ILIKE for case-insensitive text searches
- Use -> or ->> operators for JSON field access
""",
        "MySQL": """
Special Instructions (MySQL):
- Use MySQL syntax
- Use 'YYYY-MM-DD' format for dates
- Use LIKE for text searches
- Use -> or ->> operators for JSON field access
- Implement pagination using LIMIT and OFFSET
""",
        "Oracle": """
Special Instructions (Oracle):
- Use Oracle syntax
- Use TO_DATE function for date formatting
- Implement result limitation using ROWNUM
- Use DUAL table appropriately
- Use LIKE for text searches
""",
        "SQLite": """
Special Instructions (SQLite):
- Use SQLite syntax
- Use 'YYYY-MM-DD' format for dates
- Consider SQLite's limited function set
- Use basic inner and outer joins instead of advanced join techniques
"""
    }

    # Case-insensitive DBMS search
    for key, value in instructions.items():
        if key.lower() == dbms.lower():
            return value

    return ""  # Return empty string if no matching DBMS


import logging
import numpy as np


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

    # DBMS 정보 추출
    dbms = data.get('dbms', None)

    if isinstance(dbms, list) or isinstance(dbms, np.ndarray):
        if len(dbms) > 2:
            dbms = "General DBMS"
        elif len(dbms) > 1:
            dbms = dbms[0]
        else:
            dbms = ""
    else:
        logging.info(f'dbms type: {type(dbms)}, {dbms}')
        raise

    # DBMS 특화 지시사항 생성
    dbms_instructions = get_dbms_specific_instructions(dbms)

    # NL2SQL 모드
    if batch_mode == BatchMode.NL2SQL:
        if evaluation:
            return evaluation_template.format(
                schema=data.get('context', ''),
                question=data.get(field_question, ''),
                gt_sql=data.get(field_answer, ''),
                gen_sql=data.get('gen_sql', ''),
                dbms=dbms or "SQL"  # 기본값으로 일반 SQL 지정
            )
        else:
            # 기본 템플릿에 DBMS 정보 추가
            return template.format(
                schema=data.get('context', ''),
                question=data.get(field_question, ''),
                format_instructions=options.get('format_instructions', ''),
                dbms=dbms or "SQL",  # 기본값으로 일반 SQL 지정
                dbms_instructions=dbms_instructions
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
        # o1-mini 모델인 경우와 그 외 모델 구분
        if model.lower().startswith('o1-mini'):
            request = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]  # system 대신 user role 사용
            }
        else:
            request = {
                "model": model,
                "messages": [{"role": "system", "content": prompt}]
            }

        # NL2SQL 모드는 JSON 응답 필요
        if requires_json and not model.lower().startswith('o1-mini'):
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