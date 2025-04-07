import asyncio
import json
import logging
import time
import aiohttp
import pandas as pd

from typing import Union, Dict, Any, Optional, List
from pandas import DataFrame

from llms.templates import make_prompt, make_request
from utils.config import BatchMode
from utils.common import extract_sql_queries, extract_resolve_yn_from_text
from utils.tracking import ProgressTracker


async def warmup_model(model: str, url: str = "http://localhost:11434/api/generate"):
    """
    Ollama 모델 로딩을 위한 예열(warm-up) 요청 전송

    Args:
        model: 모델 이름
        url: API URL

    Returns:
        dict: 로딩 시간 및 상태 정보
    """
    logging.info(f"모델 '{model}' 로딩 중...")
    start_time = time.time()

    # 간단한 예열 프롬프트 생성
    warmup_prompt = "Hello, I'm testing if you're loaded. Please respond with 'OK'."

    # 요청 객체 생성
    request = {
        "model": model,
        "prompt": warmup_prompt,
        "stream": False,
    }

    try:
        # 비동기 HTTP 클라이언트 세션 생성
        conn = aiohttp.TCPConnector(limit=1)
        timeout = aiohttp.ClientTimeout(total=300)  # 모델 로딩에 시간이 걸릴 수 있으므로 타임아웃 충분히 설정

        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            async with session.post(url, json=request) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logging.error(f"모델 로딩 실패: {error_text}")
                    return {
                        "success": False,
                        "error": error_text,
                        "loading_time": time.time() - start_time
                    }

                result = await response.json()

                # 로딩 완료 시간 계산
                loading_time = time.time() - start_time
                logging.info(f"모델 '{model}' 로딩 완료: {loading_time:.2f}초 소요")

                return {
                    "success": True,
                    "loading_time": loading_time,
                    "response": result.get("response", "")
                }
    except Exception as e:
        loading_time = time.time() - start_time
        logging.error(f"모델 로딩 중 오류 발생: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "loading_time": loading_time
        }


def warmup_model_sync(model: str, url: str = "http://localhost:11434/api/generate"):
    """
    동기 방식의 모델 예열 함수 (비동기 함수의 동기 래퍼)

    Args:
        model: 모델 이름
        url: API URL

    Returns:
        dict: 로딩 시간 및 상태 정보
    """
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(warmup_model(model, url))
    return result


def llm_invoke_parallel(model: str,
                        dataset: Union[list, DataFrame],
                        url: str = "http://localhost:11434/api/generate",
                        log_dir: str = "logs",
                        warmup: bool = True,
                        response_processor=None,
                        options: dict = None):  # 응답 처리 함수 매개변수 추가
    """병렬 처리를 위한 래퍼 함수 (로깅 기능 추가)"""
    logging.info(
        f"병렬 처리 시작: 총 {len(dataset)}개 요청 (배치 크기: {options['batch_size']}, 최대 동시 요청: {options['max_concurrent']})")

    # 로그 디렉터리를 절대 경로로 변환
    import os
    abs_log_dir = os.path.abspath(log_dir)
    logging.debug(f"로그 디렉터리: {abs_log_dir}")

    # 전체 실행 시간 측정 시작
    total_start_time = time.time()

    # 1. 모델 예열 (옵션에 따라 실행)
    loading_time = 0
    if warmup:
        warmup_result = warmup_model_sync(model, url)
        loading_time = warmup_result.get("loading_time", 0)

        if not warmup_result.get("success", False):
            logging.warning(f"모델 예열 중 오류가 발생했습니다: {warmup_result.get('error', '알 수 없는 오류')}")
            logging.warning("오류가 발생했지만 계속 진행합니다...")

    # 2. 실제 배치 처리 시간 측정 시작
    batch_start_time = time.time()

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        llm_invoke_batch(
            model=model,
            dataset=dataset,
            url=url,
            log_dir=abs_log_dir,  # 절대 경로 사용
            response_processor=response_processor,  # 응답 처리 함수 전달
            options=options
        )
    )

    # 배치 처리 시간 계산
    batch_elapsed_time = time.time() - batch_start_time

    # 전체 시간 계산 (모델 로딩 시간 + 배치 처리 시간)
    total_elapsed_time = time.time() - total_start_time

    # 요약 정보 출력
    logging.info(f"\n===== 병렬 처리 결과 요약 =====")
    if warmup:
        logging.info(f"모델 '{model}' 로딩 시간: {loading_time:.2f}초")
    logging.info(f"배치 처리 시간: {batch_elapsed_time:.2f}초")
    logging.info(f"총 실행 시간: {total_elapsed_time:.2f}초")
    logging.info(f"처리된 요청 수: {len(results)}개")
    if len(dataset) > 0:
        logging.info(f"평균 요청 처리 시간: {(batch_elapsed_time / len(dataset)):.3f}초/요청")
        logging.info(f"초당 처리량: {(len(dataset) / batch_elapsed_time):.2f}개/초")
    logging.info(f"===============================\n")

    return results


async def llm_invoke_batch(model: str,
                           dataset: Union[list, DataFrame],
                           url: str = "http://localhost:11434/api/generate",
                           log_dir: str = ".",
                           response_processor=None,
                           options: dict = None):  # 응답 처리 함수 매개변수 추가
    """프롬프트 배치에 대한 병렬 LLM 호출 (진행률 로깅 기능 추가)"""
    all_results = [None] * len(dataset)  # 순서 보존을 위한 결과 저장 리스트
    total_prompts = len(dataset)

    if options is None:
        options = {}

    # 문자열 모드를 열거형으로 변환
    if 'mode' in options and isinstance(options['mode'], str):
        try:
            options['mode'] = BatchMode.from_string(options['mode'])
        except ValueError:
            # 기본값 사용
            options['mode'] = BatchMode.NL2SQL
            logging.warning(f"지원되지 않는 모드 문자열: {options['mode']}, 기본값 NL2SQL 사용")

    logging.info(f'options: {options}')

    batch_size = options.get('batch_size', 10)
    max_concurrent = options.get('max_concurrent', 10)
    max_retries = options.get('max_retries', 3)

    # 재시도 정보 추적
    retry_counts = [0] * total_prompts  # 각 작업별 재시도 횟수
    failed_tasks = set()  # 실패한 작업 ID
    processed_tasks = set()  # 처리 완료된 작업 ID

    # 배치 수 계산
    total_batches = (total_prompts + batch_size - 1) // batch_size

    # 진행률 추적 객체 생성
    progress_tracker = ProgressTracker(total_prompts, batch_size, log_dir=log_dir)

    # 동시 연결 제한을 위한 세마포어
    semaphore = asyncio.Semaphore(max_concurrent)

    # 커넥션 풀 설정으로 TCP 연결 재사용
    conn = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)

    # 응답 처리 함수를 llm_invoke_single에 전달하기 위한 래퍼 함수
    async def llm_invoke_single_with_processor(model, data, progress_tracker, session, task_id, url, options):
        """진행률 추적 기능이 추가된 단일 프롬프트 비동기 LLM 호출"""
        # 요청 시작 로깅
        progress_tracker.update_task_progress(task_id, "start")
        start_time = time.time()

        prompt = make_prompt(model, data, options=options)
        request = make_request(model, prompt, options=options)

        try:
            async with session.post(url, json=request) as response:
                if response.status != 200:
                    error_text = await response.text()
                    elapsed = time.time() - start_time
                    progress_tracker.update_task_progress(task_id, "failed", elapsed, error_text)

                    # 재시도 횟수 업데이트
                    retry_counts[task_id] += 1

                    # 재시도 여부 결정
                    if retry_counts[task_id] <= max_retries:
                        return {"error": error_text, "task_id": task_id, "retry": True}
                    else:
                        failed_tasks.add(task_id)
                        processed_tasks.add(task_id)
                        return {"error": error_text, "task_id": task_id}

                result = await response.json()
                logging.debug(f'request: {request}')
                logging.debug(f'response: {result}')

                elapsed = time.time() - start_time


                # # 클라이언트 코드에서만 task_id 관리
                # local_result = {"orig_response": result, "task_id": task_id}
                #
                # 응답 처리 시도 (response_processor가 있는 경우)
                if response_processor:
                    try:
                        processed_result = response_processor(result, None)

                        # 처리된 결과는 내부 관리용으로만 task_id 추가
                        progress_tracker.update_task_progress(task_id, "success", elapsed)
                        processed_tasks.add(task_id)
                        # 저장/반환용 결과에는 task_id 포함
                        return {"processed_result": processed_result, "task_id": task_id}
                    except Exception as process_error:
                        # 응답 처리 오류 시 기록하고 원본 결과 반환
                        logging.error(f"응답 처리 중 오류 발생 (요청 #{task_id}: {str(process_error)}")

                        # 재시도 횟수 업데이트
                        retry_counts[task_id] += 1

                        # 재시도 여부 결정
                        if retry_counts[task_id] <= max_retries:
                            return {"error": str(process_error), "task_id": task_id, "retry": True}
                        else:
                            result["processing_error"] = str(process_error)
                            progress_tracker.update_task_progress(task_id, "success", elapsed)

                            # 처리 완료 표시
                            failed_tasks.add(task_id)
                            processed_tasks.add(task_id)
                            return result
                else:
                    # 응답 처리 함수가 없는 경우 원본 결과 반환
                    progress_tracker.update_task_progress(task_id, "success", elapsed)

                    # 처리 완료 표시
                    processed_tasks.add(task_id)
                    return result
        except Exception as e:
            elapsed = time.time() - start_time
            progress_tracker.update_task_progress(task_id, "failed", elapsed, str(e))

            # 재시도 횟수 업데이트
            retry_counts[task_id] += 1

            # 재시도 여부 결정
            if retry_counts[task_id] <= max_retries:
                return {"error": str(e), "task_id": task_id, "retry": True}
            else:
                # 처리 완료 표시
                failed_tasks.add(task_id)
                processed_tasks.add(task_id)
                return {"error": str(e), "task_id": task_id}

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        for batch_idx in range(0, total_prompts, batch_size):
            batch_end = min(batch_idx + batch_size, total_prompts)

            # DataFrame과 리스트 모두 지원하기 위한 배치 처리 로직 수정
            if hasattr(dataset, 'iterrows'):  # DataFrame인 경우
                batch_items = []
                for idx, row in dataset.iloc[batch_idx:batch_end].iterrows():
                    batch_items.append((idx - batch_idx, row))
            else:  # 리스트인 경우
                batch_items = [(i, dataset[batch_idx + i]) for i in range(batch_end - batch_idx)]

            # 배치 시작 로깅
            progress_tracker.update_batch_progress(len(batch_items), batch_idx // batch_size, total_batches)

            # 비동기 작업 생성
            tasks = []
            for i, data in batch_items:
                task_id = batch_idx + i

                # 이미 처리된 작업 건너뛰기
                if task_id in processed_tasks:
                    continue

                # 세마포어 사용한 비동기 작업 생성
                async with semaphore:
                    task = asyncio.create_task(
                        llm_invoke_single_with_processor(model, data, progress_tracker, session, task_id, url, options)
                    )
                    tasks.append(task)

            # 모든 작업 완료 대기
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 재시도가 필요한 작업 처리
            retry_tasks = []

            for result in batch_results:
                # 예외 처리
                if isinstance(result, Exception):
                    logging.error(f"작업 예외 발생: {str(result)}")
                    continue

                # 결과가 없는 경우 건너뛰기
                if result is None:
                    continue

                task_id = result.get("task_id")
                if task_id is None:
                    logging.error(f"태스크 ID가 없는 결과: {result}")
                    continue

                # 재시도 필요 여부 확인
                if result.get("retry", False):
                    retry_tasks.append(task_id)
                else:
                    # 결과 저장 (원래 순서대로)
                    all_results[task_id] = result

            # 재시도 필요한 작업이 있는 경우, 배치 완료 후 별도 처리
            if retry_tasks:
                logging.info(f"배치 {batch_idx // batch_size + 1}/{total_batches}에서 {len(retry_tasks)}개 작업 재시도 필요")

            # 배치 완료 로깅
            progress_tracker.update_batch_completion(batch_results)

            # 서버 부하 방지를 위한 짧은 대기
            if batch_end < total_prompts:
                await asyncio.sleep(1.0)

        # 미처리 작업 확인 (모든 배치 처리 후)
        all_tasks = set(range(total_prompts))
        missing_tasks = all_tasks - processed_tasks

        if missing_tasks:
            logging.warning(f"처리되지 않은 작업 {len(missing_tasks)}개 발견: {missing_tasks}")

            # 누락된 작업에 대한 결과 항목 생성
            for task_id in missing_tasks:
                all_results[task_id] = {
                    "error": "요청이 처리되지 않음 (유실)",
                    "task_id": task_id
                }

    # 진행률 표시 종료
    progress_tracker.close()

    # 요청 ID 처리 통계 로깅
    logging.info(f"처리 완료된 작업: {len(processed_tasks)}개")
    logging.info(f"실패한 작업: {len(failed_tasks)}개")

    return all_results


def is_valid_openai_response(response: Dict[str, Any]) -> bool:
    """OpenAI 형식의 응답인지 확인"""
    return isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0


def is_valid_ollama_response(response: Dict[str, Any]) -> bool:
    """Ollama 형식의 응답인지 확인"""
    return isinstance(response, dict) and 'response' in response


def extract_content_from_response(response: Dict[str, Any]) -> Optional[str]:
    """다양한 응답 형식에서 콘텐츠 추출"""
    if is_valid_openai_response(response):
        return response['choices'][0]['message']['content']
    elif is_valid_ollama_response(response):
        return response['response']
    return None


def process_nl2sql_response(response: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    NL2SQL 응답을 처리하는 함수

    Args:
        response: LLM 응답
        metadata: 요청 메타데이터

    Returns:
        dict: 처리된 결과 (gen_sql 필드 포함)
    """
    # 이미 처리된 결과가 있는지 확인
    if 'processed_result' in response:
        return response['processed_result']

    # 요청 ID 추출
    task_id = response.get('task_id', None)

    # 콘텐츠 추출
    content = extract_content_from_response(response)
    if not content:
        raise ValueError(f"응답에서 콘텐츠를 추출할 수 없습니다: {response}")

    # JSON 파싱 시도
    try:
        result = json.loads(content)
        if 'gen_sql' in result:
            if task_id:
                result['task_id'] = task_id
            return result
    except json.JSONDecodeError:
        pass

    # SQL 쿼리 직접 추출 시도
    sql_query = extract_sql_queries(content)
    if sql_query:
        result = {"gen_sql": sql_query}
        if task_id:
            result['task_id'] = task_id
        return result

    # 처리할 수 없는 응답
    raise ValueError(f"응답에서 SQL 쿼리를 추출할 수 없습니다: {response}")


def process_evaluation_response(response: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    평가 응답을 처리하는 함수

    Args:
        response: LLM 응답
        metadata: 요청 메타데이터

    Returns:
        dict: 처리된 결과 (resolve_yn 필드 포함)
    """
    # 요청 ID 추출
    task_id = response.get('task_id', None)

    # 콘텐츠 추출
    content = extract_content_from_response(response)
    if not content:
        raise ValueError(f"응답에서 콘텐츠를 추출할 수 없습니다: {response}")

    # resolve_yn 값 추출
    result = extract_resolve_yn_from_text(content)

    # 요청 ID 추가
    if task_id:
        result['task_id'] = task_id

    return result


def get_processor_for_mode(mode: str):
    """모드에 따른 적절한 응답 처리기 반환"""
    processors = {
        'nl2sql': process_nl2sql_response,
        'evaluation': process_evaluation_response,
    }

    return processors.get(mode, process_nl2sql_response)


def process_response_by_mode(
        responses: List[Dict[str, Any]],
        dataset,
        options: Optional[Dict[str, Any]] = None
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

    # 결과 저장 변수
    result_list = []
    success_count = 0
    error_count = 0

    # 모드 및 컬럼 설정
    mode = options.get('mode', BatchMode.NL2SQL)
    output_column = options.get('output_column', None)

    # 데이터프레임 복사 또는 생성
    if isinstance(dataset, pd.DataFrame):
        result_df = dataset.copy()
    else:
        result_df = pd.DataFrame(dataset)

    # 출력 컬럼 설정
    if output_column is None:
        if mode == BatchMode.NL2SQL or str(mode) == "nl2sql":
            output_column = 'gen_sql'
        elif mode == BatchMode.TRANSLATION or str(mode) == "translate":
            output_column = 'translation'
        else:
            output_column = 'response'

    # 적절한 응답 처리기 선택
    processor = get_processor_for_mode(mode.__str__())

    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.info(f"응답 처리 시작: 모드={mode}, 출력 컬럼={output_column}")

    # 응답 처리
    for idx, response in enumerate(responses):
        # 요청 ID 추출
        task_id = response.get('task_id', idx)

        try:
            # 이미 처리된 결과가 있는지 확인
            if 'processed_result' in response:
                processed_result = response['processed_result']
            else:
                # 응답 처리
                processed_result = processor(response, None)

            # 결과 추가
            result_list.append((task_id, processed_result))
            success_count += 1

        except Exception as e:
            logger.error(f"응답 처리 중 오류 발생 (태스크 #{task_id}): {str(e)}")
            result_list.append((idx, None))
            error_count += 1

    # 결과 정렬 및 데이터프레임에 추가
    result_list.sort(key=lambda x: x[0])  # task_id 기준 정렬

    def extract_sql(data, column_name: str = 'gen_sql'):
        """Extract SQL query from various data formats"""
        if isinstance(data, dict) and column_name in data:
            return data[column_name]

        try:
            if isinstance(data, str):
                parsed_data = json.loads(data)
                if isinstance(parsed_data, dict) and column_name in parsed_data:
                    return parsed_data[column_name]
                return parsed_data
        except:
            raise
            pass

        return data

    # 번역 모드 처리
    if mode == BatchMode.TRANSLATION:
        # 새 컬럼 초기화
        result_df['e_question'] = None
        result_df['e_answer'] = None

        for task_id, result in result_list:
            if task_id < len(result_df) and result:
                result_df.at[task_id, 'e_question'] = result.get('question', '')
                result_df.at[task_id, 'e_answer'] = result.get('sql_query', '')
    else:
        # 다른 모드 처리
        result_df[output_column] = [extract_sql(result, output_column) for _, result in result_list]
        result_df['task_id'] = [idx for idx, _ in result_list]

    # 성공/실패 통계 정보 추가
    result_df.attrs['success_count'] = success_count
    result_df.attrs['error_count'] = error_count

    logger.info(f"응답 처리 완료: 성공 {success_count}개, 오류 {error_count}개")
    return result_df


def make_result(
        responses: List[Dict[str, Any]],
        dataset: pd.DataFrame,
        options: Dict[str, Any] = None
) -> tuple:
    """
    NL2SQL 응답 처리 함수 (이전 버전과의 호환성 유지 위한 래퍼)

    Args:
        responses: 모델 응답 리스트
        dataset: 원본 데이터셋
        options: 추가 옵션

    Returns:
        tuple: 처리 결과 데이터프레임, 성공 카운트, 에러 카운트
    """
    result_df = process_response_by_mode(responses, dataset, options)
    return result_df, result_df.attrs.get('success_count', 0), result_df.attrs.get('error_count', 0)


def process_evaluation_results(base_eval, dataset, output_path, model_name):
    """
    Process evaluation results and save success/failure cases

    Args:
        base_eval: DataFrame containing evaluation results
        dataset: Original dataset with questions and contexts
        output_path: Base path for saving output files
        model_name: Name of the model used for evaluation

    Returns:
        tuple: (processed_df, success_count, failure_count)
    """
    import json
    import ast
    import pandas as pd
    from pathlib import Path

    # File paths for success and failure cases
    success_cases_file = Path(output_path) / f"{model_name}_success_cases.csv"
    failure_cases_file = Path(output_path) / f"{model_name}_failure_cases.csv"

    # Step 1: Clean and normalize the resolve_yn values
    base_eval = _normalize_resolve_yn_values(base_eval)

    # Step 2: Extract task_ids and clean prompts
    base_eval = _extract_metadata_from_prompts(base_eval)

    # Log distribution of resolve_yn values
    logging.info(f"resolve_yn distribution: {base_eval['resolve_yn'].value_counts().to_dict()}")

    # Step 3: Save success and failure cases
    logging.info(f'dataset: {dataset.keys()}')
    dataset_selected = dataset[['task_id', 'question', 'answer', 'gen_sql', 'context']]
    merged = base_eval.merge(dataset_selected, on='task_id', how='inner')
    merged = merged.rename(columns={'answer': 'gt_sql'})
    success_count = _save_filtered_cases(merged, 'yes', success_cases_file)
    failure_count = _save_filtered_cases(merged, 'no', failure_cases_file)

    # Log results
    if success_count > 0:
        logging.info(f"Saved {success_count} success cases to {success_cases_file}")
    else:
        logging.info("No success cases found")

    if failure_count > 0:
        logging.info(f"Saved {failure_count} failure cases to {failure_cases_file}")
    else:
        logging.info("No failure cases found")

    return base_eval, success_count, failure_count


def _normalize_resolve_yn_values(df):
    """Normalize resolve_yn values to consistent format"""

    def safe_extract_resolve_yn(x):
        if not isinstance(x, str):
            return x

        # Return as is if already in the expected format
        x_lower = x.lower().strip()
        if x_lower in ['yes', 'no', 'unknown']:
            return x_lower

        # Simple text matching
        if x.lower().startswith('yes'):
            return 'yes'
        elif x.lower().startswith('no'):
            return 'no'

        # Pattern matching for common formats
        if "'resolve_yn': 'yes'" in x or '"resolve_yn": "yes"' in x:
            return 'yes'
        elif "'resolve_yn': 'no'" in x or '"resolve_yn": "no"' in x:
            return 'no'

        # Try JSON parsing
        try:
            parsed = json.loads(x)
            if isinstance(parsed, dict) and 'resolve_yn' in parsed:
                return parsed['resolve_yn'].lower().strip()
        except:
            pass

        return x

    df_copy = df.copy()
    df_copy['resolve_yn'] = df_copy['resolve_yn'].apply(
        lambda x: safe_extract_resolve_yn(x) if pd.notna(x) else x
    )
    return df_copy


def _extract_metadata_from_prompts(df):
    """Extract task_ids and clean prompts from prompt column"""
    import ast

    def extract_prompt(prompt_data):
        # Handle dictionary objects
        if isinstance(prompt_data, dict):
            # Check common paths for prompt
            for path in ['prompt', 'request.prompt']:
                parts = path.split('.')
                value = prompt_data
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                if value is not None:
                    return value

        # Handle string representations
        try:
            data = ast.literal_eval(prompt_data)
            if isinstance(data, dict):
                return extract_prompt(data)
        except:
            return prompt_data

        return None

    df_copy = df.copy()

    # Apply transformations with error handling
    if 'resolve_yn' in df_copy.columns:
        df_copy['resolve_yn'] = df_copy['resolve_yn'].apply(
            lambda x: x if pd.notna(x) else x
        )

    if 'prompt' in df_copy.columns:
        df_copy['prompt'] = df_copy['prompt'].apply(
            lambda x: extract_prompt(x) if pd.notna(x) else None
        )

    return df_copy


def _save_filtered_cases(dataset, result_value, output_file):
    """Save filtered cases (success or failure) to CSV file"""

    # Extract cases matching the result value
    cases = dataset[dataset['resolve_yn'] == result_value].copy()

    if cases.empty:
        return 0

    # Debug logging to check task_id availability
    task_id_count = cases['task_id'].apply(lambda x: x != "").sum()
    logging.info(f"Including {task_id_count} task_ids in the {result_value} cases file")

    # Save to CSV
    cases.sort_values(by='task_id', inplace=True)
    cases.to_csv(output_file, index=False)

    return len(cases)