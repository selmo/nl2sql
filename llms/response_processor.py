import json
import re
import logging
import pandas as pd
from llms.prompt_generator import sql_parser, SQL
from typing import List, Tuple, Any
from langchain_core.exceptions import OutputParserException

from util.config import BatchMode


def extract_sql_queries(text: str) -> str:
    """텍스트에서 SQL 쿼리 추출"""
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


def extract_json(text: str) -> dict:
    """텍스트에서 JSON 추출 시도"""
    try:
        # JSON 블록 추출 패턴
        json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
        match = json_pattern.search(text)

        if match:
            json_text = match.group(1)
            return json.loads(json_text)

        # 전체 텍스트가 JSON인 경우
        return json.loads(text)

    except json.JSONDecodeError:
        # JSON 추출 실패 시 빈 딕셔너리 반환
        return {}


# response 컬럼에서 JSON 데이터 추출
def extract_from_response(response_str):
    try:
        # response_str이 문자열 형태의 딕셔너리라고 가정
        # 작은 따옴표를 큰 따옴표로 바꿔서 JSON 형식으로 변환
        response_str = response_str.replace("'", '"')
        response_dict = json.loads(response_str)

        # 'question'과 'answer' 키가 있는지 확인하고 없으면 빈 문자열 반환
        e_question = response_dict.get('question', '')
        e_answer = response_dict.get('answer', '')

        return e_question, e_answer
    except (json.JSONDecodeError, TypeError):
        # JSON 파싱 실패 시 빈 문자열 반환
        return '', ''


# def clean_response(response, only_sql: bool = True):
def clean_response(response: str, mode: BatchMode = BatchMode.NL2SQL) -> Any:
    """
    모드별 응답 정리 함수

    Args:
        response: 모델 응답 텍스트
        mode: 배치 처리 모드
        **kwargs: 추가 옵션

    Returns:
        Any: 정리된 응답
    """
    # <think> 태그 제거
    clean_output = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    logging.debug(f'clean_output: {clean_output}')
    # NL2SQL 모드
    if mode == BatchMode.NL2SQL:
        try:
            result = json.loads(clean_output)
            return result.get('gen_sql', '')
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 SQL 쿼리 추출 시도
            return extract_sql_queries(clean_output)
    # 번역, 요약, QA 모드
    elif mode in [BatchMode.TRANSLATION]:
        try:
            result = json.loads(clean_output)
            # logging.info(f'result: [{type(result)}] {result}')
            return result
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 SQL 쿼리 추출 시도
            return extract_sql_queries(clean_output)

    # 기본 응답 처리
    return clean_output.strip()

    # clean_output = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # logging.debug('response: %s', response)
    # logging.debug('clean_output: %s', clean_output)
    # if only_sql:
    #     # logging.info("cleaned_output 1: [%s] %s", type(clean_output), clean_output)
    #     clean_output = json.loads(clean_output)
    #     return clean_output['gen_sql']
    # else:
    #     # logging.info("cleaned_output 2: [%s] %s", type(clean_output), clean_output)
    #     try:
    #         result = sql_parser.parse(clean_output)
    #         # logging.info("result 0: [%s] %s\n", type(result), result)
    #         return result
    #     except TypeError as te:
    #         logging.error("response 1: [TypeError] %s", response)
    #         logging.error("error: %s", te)
    #         logging.error("clean_output 3: %s", clean_output)
    #         exit(0)
    #     except OutputParserException as e:
    #         logging.error("response 2: %s", response)
    #         logging.error("clean_output 4: %s", clean_output)
    #         query_match = extract_sql_queries(clean_output)
    #         logging.error("query: %s", query_match)
    #         sql_obj = SQL(
    #             reasoning="",
    #             description="Parsing error.",
    #             gen_sql=query_match
    #         )
    #         logging.error("error msg: %s\n", e)
    #         return sql_obj


def is_valid_content(result: dict) -> bool:
    """응답에 올바른 콘텐츠가 있는지 확인"""
    return bool(result) and 'message' in result and 'content' in result['message']


def is_valid_response(result: dict) -> bool:
    """Ollama 응답이 올바른지 확인"""
    return bool(result) and 'response' in result


def has_error(result: dict) -> bool:
    """응답에 오류가 있는지 확인"""
    return bool(result) and 'error' in result


def process_response_by_mode(
        responses: List[dict],
        dataset,
        options=None
) -> pd.DataFrame:
    """
    모드별 응답 처리 함수

    Args:
        responses: 모델 응답 리스트
        dataset: 원본 데이터셋
        options: 추가 옵션

    Returns:
        DataFrame: 처리 결과가 추가된 데이터프레임
    """
    if options is None:
        options = {}

    result_list = []
    success_count = 0
    error_count = 0

    mode = options.get('mode', BatchMode.NL2SQL)

    # 데이터프레임 복사
    if isinstance(dataset, pd.DataFrame):
        result_df = dataset.copy()
    else:
        # 리스트 등 다른 형태의 데이터셋인 경우 데이터프레임으로 변환
        result_df = pd.DataFrame(dataset)

    output_column = options.get('output_column', None)

    # 출력 컬럼 설정
    if output_column is None:
        if mode == BatchMode.NL2SQL:
            output_column = 'gen_sql'
        elif mode == BatchMode.TRANSLATION:
            output_column = 'translation'
        else:
            output_column = 'response'

    # 응답 처리
    for idx, result in enumerate(responses):
        try:
            if is_valid_content(result):
                # GPT 스타일 응답
                content = result['message']['content']
                cleaned_result = clean_response(content, mode)
                result_list.append(cleaned_result)
                success_count += 1

            elif is_valid_response(result):
                # Ollama 스타일 응답
                content = result['response']
                cleaned_result = clean_response(content, mode)
                # logging.info(f'cleaned_result: [{type(cleaned_result)}]{cleaned_result}')
                if mode == BatchMode.TRANSLATION:
                    result_list.append((idx, cleaned_result))
                else:
                    result_list.append(cleaned_result)
                success_count += 1

            elif has_error(result):
                # 오류 응답
                logging.error(f"오류 응답 (인덱스 #{idx}): {result.get('error', 'unknown error')}")
                result_list.append('')
                error_count += 1

            else:
                # 예상치 못한 응답 형식
                logging.warning(f"예상치 못한 응답 형식 (인덱스 #{idx}): [{type(result)}], {result}")
                result_list.append('')
                error_count += 1

        except Exception as e:
            logging.error(f"응답 처리 중 오류 발생 (인덱스 #{idx}): {str(e)}")
            result_list.append('')
            error_count += 1

    # 데이터프레임에 결과 추가
    if len(result_list) > 0:
        if mode == BatchMode.TRANSLATION:
            # 새 컬럼 초기화
            result_df['e_question'] = None
            result_df['e_answer'] = None

            for i, row in result_list:
                if i < len(result_df):
                    result_df.at[i, 'e_question'] = row.get('question', '')
                    result_df.at[i, 'e_answer'] = row.get('sql_query', '')

            print(f"처리된 데이터셋 크기: {len(result_df)}")
            print(f"컬럼: {result_df.columns.tolist()}")
        else:
            result_df[output_column] = result_list

    # 성공/실패 통계 정보 추가
    result_df.attrs['success_count'] = success_count
    result_df.attrs['error_count'] = error_count

    logging.info(f"응답 처리 완료: 성공 {success_count}개, 오류 {error_count}개")

    return result_df


def make_result(
        responses: List[dict],
        dataset: pd.DataFrame,
        options: dict = None
) -> Tuple[pd.DataFrame, int, int]:
    """
    NL2SQL 응답 처리 함수 (이전 버전과의 호환성 유지)

    Args:
        responses: 모델 응답 리스트
        dataset: 원본 데이터셋
        options:

    Returns:
        Tuple[DataFrame, int, int]: 처리 결과 데이터프레임, 성공 카운트, 에러 카운트
    """
    result_df = process_response_by_mode(
        responses,
        dataset,
        options=options
    )

    return result_df, result_df.attrs.get('success_count', 0), result_df.attrs.get('error_count', 0)

# def make_result(
#         responses: List[dict],
#         dataset: pd.DataFrame,
#         success_count: int = 0,
#         error_count: int = 0,
#         only_sql: bool = True,
# ) -> Tuple[pd.DataFrame, int, int]:
#     """
#     responses 리스트를 순회하며 dataset에 'gen_sql' 값을 업데이트한다.
#     성공적으로 SQL 문이 생성되면 success_count를, 오류 발생 시 error_count를 증가시킨다.
#
#     Parameters
#     ----------
#     responses : List[dict]
#         응답 데이터 리스트
#     dataset : pd.DataFrame
#         결과를 업데이트할 데이터프레임
#     success_count : int
#         초기 성공 카운트
#     error_count : int
#         초기 에러 카운트
#
#     Returns
#     -------
#     dataset : pd.DataFrame
#         'gen_sql' 열이 업데이트된 데이터프레임
#     success_count : int
#         업데이트 후의 성공 카운트
#     error_count : int
#         업데이트 후의 에러 카운트
#     """
#     gen_sql_list = []
#
#     for idx, result in enumerate(responses):
#         logging.debug("Result index #%d: %s", idx, result)
#         try:
#             if is_valid_content(result):
#                 content = result['message']['content']
#                 logging.debug("Content: %s", content)
#                 cleaned_result = clean_response(content, only_sql)
#                 if only_sql:
#                     gen_sql_list.append(cleaned_result)
#                 else:
#                     gen_sql_list.append(cleaned_result.gen_sql)
#                 success_count += 1
#
#             elif is_valid_response(result):
#                 content = result['response']
#                 logging.debug("Content: %s", content)
#                 cleaned_result = clean_response(content, only_sql)
#                 if only_sql:
#                     gen_sql_list.append(cleaned_result)
#                 else:
#                     gen_sql_list.append(cleaned_result.gen_sql)
#                 success_count += 1
#
#             elif has_error(result):
#                 logging.error("오류 응답 (인덱스 #%d): %s", idx, result['error'])
#                 logging.debug("Full error result: %s", result)
#                 gen_sql_list.append('')
#                 error_count += 1
#
#             else:
#                 logging.warning("예상치 못한 응답 형식 (인덱스 #%d): %s", idx, result)
#                 logging.debug("Unhandled response: %s", result)
#                 gen_sql_list.append('')
#                 error_count += 1
#
#         except KeyError as ke:
#             logging.error("KeyError (인덱스 #%d): %s", idx, str(ke))
#             gen_sql_list.append('')
#             error_count += 1
#         except Exception as e:
#             logging.error("결과 처리 중 알 수 없는 오류 (인덱스 #%d): %s", idx, str(e))
#             gen_sql_list.append('')
#             error_count += 1
#
#     # 마지막에 한 번에 업데이트
#     dataset['gen_sql'] = gen_sql_list
#
#     return dataset, success_count, error_count


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