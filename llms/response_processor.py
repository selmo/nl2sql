import re
import logging
import pandas as pd
from llms.prompt_generator import sql_parser, SQL
from typing import List, Tuple
from langchain_core.exceptions import OutputParserException


def extract_sql_queries(text: str) -> str:
    # 1) ```sql ... ``` 패턴
    pattern_triple_backticks = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL)

    # 2) \boxed{ ... } 패턴
    pattern_boxed = re.compile(r"\\boxed\s*\{\s*(.*?)\s*\}", re.DOTALL)

    # (1) triple backticks 안의 SQL 추출
    matches_backticks = pattern_triple_backticks.findall(text)
    if matches_backticks:
        return matches_backticks[0].strip()

    # (2) \boxed 안의 SQL 추출
    matches_boxed = pattern_boxed.findall(text)
    if matches_boxed:
        return matches_boxed[0].strip()

    return ""


def clean_response(parser, response):
    clean_output = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    try:
        result = parser.parse(clean_output)
        logging.debug("result: %s\n", result)
        return result
    except OutputParserException as e:
        logging.error("response: %s", response)
        logging.error("clean_output: %s", clean_output)
        query_match = extract_sql_queries(clean_output)
        logging.error("query: %s", query_match)
        sql_obj = SQL(
            reasoning="",
            description="Parsing error.",
            gen_sql=query_match
        )
        logging.error("error msg: %s\n", e)
        return sql_obj
# result:
# {'model': 'sqlcoder:15b',
#  'created_at': '2025-03-21T01:35:28.250463879Z',
#  'response': '{
#       "reasoning": "...",
#       "description": "...",
#       "gen_sql": "SELECT * FROM quests WHERE name LIKE \'%사냥-고블린%\' OR name LIKE \'%수집-꽃%\';"
#       }',
#   'done': True,
#   'done_reason': 'stop',
#   'context': [...],
#   'total_duration': 20305647401,
#   'load_duration': 26541045,
#   'prompt_eval_count': 309,
#   'prompt_eval_duration': 165000000,
#   'eval_count': 241,
#   'eval_duration': 6011000000,
#   'task_id': 8}
def is_valid_content(result: dict) -> bool:
    return bool(result) and 'message' in result and 'content' in result['message']

def is_valid_response(result: dict) -> bool:
    return bool(result) and 'response' in result and 'gen_sql' in result['response']

def has_error(result: dict) -> bool:
    return bool(result) and 'error' in result


def make_result(
        responses: List[dict],
        dataset: pd.DataFrame,
        success_count: int = 0,
        error_count: int = 0
) -> Tuple[pd.DataFrame, int, int]:
    """
    responses 리스트를 순회하며 dataset에 'gen_sql' 값을 업데이트한다.
    성공적으로 SQL 문이 생성되면 success_count를, 오류 발생 시 error_count를 증가시킨다.

    Parameters
    ----------
    responses : List[dict]
        응답 데이터 리스트
    dataset : pd.DataFrame
        결과를 업데이트할 데이터프레임
    success_count : int
        초기 성공 카운트
    error_count : int
        초기 에러 카운트

    Returns
    -------
    dataset : pd.DataFrame
        'gen_sql' 열이 업데이트된 데이터프레임
    success_count : int
        업데이트 후의 성공 카운트
    error_count : int
        업데이트 후의 에러 카운트
    """
    gen_sql_list = []

    for idx, result in enumerate(responses):
        logging.debug("Result index %d: %s", idx, result)
        try:
            if is_valid_content(result):
                content = result['message']['content']
                logging.debug("Content: %s", content)
                # clean_response는 예시로 둡니다. 실제 로직은 필요에 맞춰 구현
                cleaned_result = clean_response(sql_parser, content)
                gen_sql_list.append(cleaned_result.gen_sql)
                success_count += 1

            elif is_valid_response(result):
                content = result['response']
                logging.debug("Content: %s", content)
                # clean_response는 예시로 둡니다. 실제 로직은 필요에 맞춰 구현
                cleaned_result = clean_response(sql_parser, content)
                gen_sql_list.append(cleaned_result.gen_sql)
                success_count += 1

            elif has_error(result):
                logging.error("오류 응답 (인덱스 %d): %s", idx, result['error'])
                logging.debug("Full error result: %s", result)
                gen_sql_list.append('')
                error_count += 1

            else:
                logging.warning("예상치 못한 응답 형식 (인덱스 %d): %s", idx, result)
                logging.debug("Unhandled response: %s", result)
                gen_sql_list.append('')
                error_count += 1

        except KeyError as ke:
            logging.error("KeyError (인덱스 %d): %s", idx, str(ke))
            gen_sql_list.append('')
            error_count += 1
        except Exception as e:
            logging.error("결과 처리 중 알 수 없는 오류 (인덱스 %d): %s", idx, str(e))
            gen_sql_list.append('')
            error_count += 1

    # 마지막에 한 번에 업데이트
    dataset['gen_sql'] = gen_sql_list

    return dataset, success_count, error_count


# def make_result(responses, dataset, success_count, error_count):
#     for idx, result in enumerate(responses):
#         logging.info("result: %s", result)
#         try:
#             if result and 'message' in result and 'content' in result['message']:
#                 content = result['message']['content']
#                 logging.debug("content: %s", content)
#                 cleaned_result = clean_response(sql_parser, content)
#                 dataset.loc[idx, 'gen_sql'] = cleaned_result.gen_sql
#                 success_count += 1
#             elif result and 'error' in result:
#                 logging.error(f"오류 응답 (인덱스 {idx}): {result['error']}")
#                 logging.debug("error 1: %s", result)
#                 dataset.loc[idx, 'gen_sql'] = ''
#                 error_count += 1
#             else:
#                 logging.warning(f"예상치 못한 응답 형식 (인덱스 {idx}): {result}")
#                 logging.debug("error 2: %s", result)
#                 dataset.loc[idx, 'gen_sql'] = ''
#                 error_count += 1
#         except Exception as e:
#             logging.error(f"결과 처리 중 오류 발생 (인덱스 {idx}): {str(e)}")
#             logging.debug("error 3: %s", str(e))
#             dataset.loc[idx, 'gen_sql'] = ''
#             error_count += 1
#
#     return dataset, success_count, error_count