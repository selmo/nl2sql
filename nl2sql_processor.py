import logging
import asyncio
import re
import aiohttp
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

import api_request_parallel_processor
from util_common import check_and_create_directory
from os.path import join as pathjoin
from datetime import time
from llms.prompt_generator import sql_parser, SQL, make_prompt, make_request
from typing import List, Tuple
from langchain_core.exceptions import OutputParserException


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def prepare_test_dataset(model, prefix='', batch_size=10, max_concurrent=10, max_retries=3, test_size=0, ollama_url=None, api_key=None):
    """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""
    test_path = pathjoin(prefix, 'test')
    check_and_create_directory(test_path)
    requests_file = pathjoin(test_path, 'requests.jsonl')
    results_file = pathjoin(test_path, 'results.jsonl')

    # 로그 파일 설정
    log_filepath = pathjoin(prefix, "test_processing.log")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logging.info(f"병렬 처리 로그를 {log_filepath}에 기록합니다.")

    if not Path(requests_file).exists():
        logging.info(f"파일이 존재하지 않습니다. 데이터 파일 생성 중: {requests_file}")

        # 데이터셋 불러오기
        dataset = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        dataset = dataset.to_pandas()
        if test_size > 0:
            dataset = dataset[:test_size]

        # 병렬 호출 실행
        logging.info(f"총 {len(dataset)}개 데이터셋에 대한 병렬 처리를 시작합니다.")

        # df.loc[idx, 'prompt'] = prompt
        jobs = []
        for idx, data in dataset.iterrows():
            # 평가를 위한 requests.jsonl 생성
            prompt = make_prompt(model, data['context'], data['question'])
            job = make_request(model, prompt)
            jobs.append(job)

        with open(requests_file, "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")

        url = "https://api.openai.com/v1/chat/completions" if model.lower().startswith(
            'gpt') or model.startswith('o1') or model.startswith(
            'o3') else "http://172.16.15.112:11434/api/generate"

        logging.info('URL: %s', url)
        api_request_parallel_processor.process_by_file(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url=url,
            api_key=api_key,
            max_requests_per_minute=2500,
            max_tokens_per_minute=100000,
            token_encoding_name="cl100k_base",
            max_attempts=10,
            logging_level=20,
            max_concurrent_requests=max_concurrent,
            batch_size=batch_size,
        )
        # 결과 처리
        # results, success_count, error_count = make_result(responses, df)
        #
        # logging.info(f"결과 처리 완료: 성공 {success_count}개, 오류 {error_count}개")

        # 결과 저장
        # results.to_json(requests_file, orient='records', lines=True)
        # logging.info(f"파일 저장 완료: {requests_file}")
    else:
        # logging.info(f"파일이 존재합니다. 데이터 파일 로딩 중: {requests_file}")
        # results = pd.read_json(requests_file, lines=True)
        # logging.info(f"데이터 컬럼: {results.keys()}")
        logging.info("파일 로딩 완료.")

    # 로그 핸들러 제거
    root_logger.removeHandler(file_handler)

    # return results


def llm_invoke_parallel(model, datasets, batch_size=10, max_retries=3, max_concurrent=10, url="http://localhost"):
    """병렬 처리를 위한 래퍼 함수 (로깅 기능 추가)"""
    logging.info(f"병렬 처리 시작: 총 {len(datasets)}개 요청 (배치 크기: {batch_size}, 최대 동시 요청: {max_concurrent})")

    start_time = time.time()

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        llm_invoke_batch(
            datasets,
            model,
            f"{url}:11434/api/generate",
            batch_size=batch_size,
            max_retries=max_retries,
            max_concurrent=max_concurrent
        )
    )

    elapsed_time = time.time() - start_time

    logging.info(f"병렬 처리 완료: {len(results)}개 응답 수신")
    logging.info(f"평균 요청 처리 시간: {elapsed_time / len(datasets)}초")

    return results


async def llm_invoke_single(session, data, model_url, model_name, task_id, progress_tracker):
    """진행률 추적 기능이 추가된 단일 프롬프트 비동기 LLM 호출"""
    # 요청 시작 로깅
    progress_tracker.update_task_progress(task_id, "start")
    start_time = time.time()

    prompt = make_prompt(model_name, data['context'], data['question'])
    request = make_request(model_name, prompt)

    try:
        async with session.post(model_url, json=request) as response:
            if response.status != 200:
                error_text = await response.text()
                elapsed = time.time() - start_time
                progress_tracker.update_task_progress(task_id, "failed", elapsed, error_text)
                return {"error": error_text, "task_id": task_id}

            result = await response.json()
            elapsed = time.time() - start_time
            progress_tracker.update_task_progress(task_id, "success", elapsed)
            result["task_id"] = task_id  # 태스크 ID 추가
            return result
    except Exception as e:
        elapsed = time.time() - start_time
        progress_tracker.update_task_progress(task_id, "failed", elapsed, str(e))
        return {"error": str(e), "task_id": task_id}


async def llm_invoke_batch(datasets, model_name, model_url="http://172.16.15.112:11434/api/chat",
                           batch_size=10, max_retries=3, max_concurrent=10):
    """프롬프트 배치에 대한 병렬 LLM 호출 (진행률 로깅 기능 추가)"""
    all_results = [None] * len(datasets)  # 순서 보존을 위한 결과 저장 리스트
    total_prompts = len(datasets)

    # 배치 수 계산
    total_batches = (total_prompts + batch_size - 1) // batch_size

    # 진행률 추적 객체 생성
    progress_tracker = ProgressTracker(total_prompts, batch_size)

    # 동시 연결 제한을 위한 세마포어
    semaphore = asyncio.Semaphore(max_concurrent)

    # 커넥션 풀 설정으로 TCP 연결 재사용
    conn = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        for batch_idx in range(0, total_prompts, batch_size):
            batch_end = min(batch_idx + batch_size, total_prompts)
            batch = datasets[batch_idx:batch_end]

            # 배치 시작 로깅
            progress_tracker.update_batch_progress(len(batch), batch_idx // batch_size, total_batches)

            # 비동기 작업 생성
            tasks = []
            for i, data in enumerate(batch):
                task_id = batch_idx + i
                task = asyncio.create_task(
                    llm_invoke_single(
                        session, data, model_url, model_name, task_id, progress_tracker
                    )
                )
                tasks.append(task)

            # 모든 작업 완료 대기
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 재시도가 필요한 작업 처리
            retry_tasks = []

            for result in batch_results:
                if isinstance(result, Exception):
                    logging.error(f"작업 예외 발생: {str(result)}")
                    continue

                task_id = result.get("task_id")
                if task_id is None:
                    logging.error(f"태스크 ID가 없는 결과: {result}")
                    continue

                # 오류 발생 여부 확인 및 재시도 결정
                if "error" in result and max_retries > 0:
                    retry_tasks.append((task_id, datasets[task_id], 0))  # (task_id, prompt, retry_count)
                else:
                    # 결과 저장 (원래 순서대로)
                    all_results[task_id] = result

            # 실패한 작업 재시도
            if retry_tasks and max_retries > 0:
                await retry_failed_tasks(session, retry_tasks, model_url, model_name,
                                         max_retries, all_results, progress_tracker)

            # 배치 완료 로깅
            progress_tracker.update_batch_completion(batch_results)

            # 서버 부하 방지를 위한 짧은 대기
            if batch_end < total_prompts:
                await asyncio.sleep(0.5)

    # 진행률 표시 종료
    progress_tracker.close()

    return all_results


async def retry_failed_tasks(session, retry_tasks, model_url, model_name,
                             max_retries, all_results, progress_tracker):
    """실패한 작업을 재시도하는 함수"""
    retry_logger = logging.getLogger("RetryProcessor")

    for retry_round in range(max_retries):
        if not retry_tasks:
            break

        retry_logger.info(f"{len(retry_tasks)}개 작업 재시도 중 (라운드: {retry_round + 1}/{max_retries})")

        current_tasks = retry_tasks
        retry_tasks = []

        # 재시도 작업 생성
        tasks = []
        for task_id, prompt, retry_count in current_tasks:
            # 재시도 로깅
            progress_tracker.update_task_progress(task_id, "retry", error=retry_count + 1)

            # 재시도 요청 준비
            task = asyncio.create_task(
                llm_invoke_single(
                    session, prompt, model_url, model_name, task_id, progress_tracker
                )
            )
            tasks.append((task, task_id, retry_count, prompt))

        # 재시도 작업 처리
        for task, task_id, retry_count, prompt in tasks:
            try:
                result = await task
                if "error" in result and retry_count < max_retries - 1:
                    # 다음 재시도 라운드에 추가
                    retry_tasks.append((task_id, prompt, retry_count + 1))
                else:
                    # 최종 결과 저장
                    all_results[task_id] = result
            except Exception as e:
                if retry_count < max_retries - 1:
                    retry_tasks.append((task_id, prompt, retry_count + 1))
                else:
                    all_results[task_id] = {"error": str(e), "task_id": task_id}

        # 다음 재시도 라운드 전에 짧은 대기
        if retry_tasks:
            await asyncio.sleep(1)


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
        logging.debug("Result index #%d: %s", idx, result)
        try:
            if is_valid_content(result):
                content = result['message']['content']
                logging.debug("Content: %s", content)
                # clean_response는 예시로 둡니다. 실제 로직은 필요에 맞춰 구현
                cleaned_result = clean_response(content, sql_parser)
                gen_sql_list.append(cleaned_result.gen_sql)
                success_count += 1

            elif is_valid_response(result):
                content = result['response']
                logging.debug("Content: %s", content)
                # clean_response는 예시로 둡니다. 실제 로직은 필요에 맞춰 구현
                cleaned_result = clean_response(content, sql_parser)
                gen_sql_list.append(cleaned_result.gen_sql)
                success_count += 1

            elif has_error(result):
                logging.error("오류 응답 (인덱스 #%d): %s", idx, result['error'])
                logging.debug("Full error result: %s", result)
                gen_sql_list.append('')
                error_count += 1

            else:
                logging.warning("예상치 못한 응답 형식 (인덱스 #%d): %s", idx, result)
                logging.debug("Unhandled response: %s", result)
                gen_sql_list.append('')
                error_count += 1

        except KeyError as ke:
            logging.error("KeyError (인덱스 #%d): %s", idx, str(ke))
            gen_sql_list.append('')
            error_count += 1
        except Exception as e:
            logging.error("결과 처리 중 알 수 없는 오류 (인덱스 #%d): %s", idx, str(e))
            gen_sql_list.append('')
            error_count += 1

    # 마지막에 한 번에 업데이트
    dataset['gen_sql'] = gen_sql_list

    return dataset, success_count, error_count


def is_valid_content(result: dict) -> bool:
    return bool(result) and 'message' in result and 'content' in result['message']


def is_valid_response(result: dict) -> bool:
    return bool(result) and 'response' in result and 'gen_sql' in result['response']


def has_error(result: dict) -> bool:
    return bool(result) and 'error' in result


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


def clean_response(response, parser=None):
    cleaned_text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    if parser is None:
        return cleaned_text
    else:
        try:
            result = parser.parse(cleaned_text)
            return result
        except TypeError as te:
            logging.error("def clean_response [TypeError] %s", response)
            logging.error("error: %s", te)
            logging.error("clean_output: %s", cleaned_text)
            exit(0)
        except OutputParserException as e:
            query_match = extract_sql_queries(cleaned_text)
            sql_obj = SQL(
                reasoning="",
                description="Parsing error.",
                gen_sql=query_match
            )
            logging.error("error msg: %s\n", e)
            return sql_obj


class ProgressTracker:
    """병렬 처리 진행 상황을 추적하는 클래스"""

    def __init__(self, total_prompts, batch_size):
        self.total_prompts = total_prompts
        self.batch_size = batch_size
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time()
        self.pbar = tqdm(total=total_prompts, desc="전체 진행률")

        # 로그 설정
        self.logger = logging.getLogger("BatchProcessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def update_batch_progress(self, batch_size, batch_idx, total_batches):
        """배치 시작 시 로깅"""
        self.logger.info(f"배치 처리 시작: {batch_idx + 1}/{total_batches} (전체 진행률: {self.completed}/{self.total_prompts})")

    def update_task_progress(self, task_id, status, elapsed=None, error=None):
        """개별 태스크 진행 상황 업데이트"""
        if status == "start":
            self.logger.debug(f"태스크 #{task_id} 시작")
        elif status == "success":
            self.completed += 1
            self.successful += 1
            self.pbar.update(1)
            if elapsed:
                self.logger.debug(f"태스크 #{task_id} 완료: {elapsed:.2f}초 소요")
        elif status == "failed":
            self.completed += 1
            self.failed += 1
            # self.pbar.update(1)
            error_msg = f": {error}" if error else ""
            self.logger.warning(f"태스크 #{task_id} 실패{error_msg}")
        elif status == "retry":
            retry_count = error if error else "?"
            self.logger.info(f"태스크 #{task_id} 재시도 중 (시도 횟수: {retry_count})")

    def update_batch_completion(self, batch_results):
        """배치 완료 시 진행 상황 업데이트"""
        batch_success = sum(1 for r in batch_results if not isinstance(r, Exception) and 'error' not in r)
        batch_failed = len(batch_results) - batch_success

        elapsed = time() - self.start_time
        requests_per_second = self.completed / elapsed if elapsed > 0 else 0

        # 현재 상태 로깅
        self.logger.info(
            f"배치 완료: 성공 {batch_success}, 실패 {batch_failed} "
            f"(전체: {self.completed}/{self.total_prompts}, "
            f"속도: {requests_per_second:.2f} 요청/초)"
        )

    def close(self):
        """진행률 추적 종료"""
        self.pbar.close()
        elapsed = time() - self.start_time

        self.logger.info(
            f"처리 완료: 총 {self.total_prompts}개 요청, "
            f"{elapsed:.2f}초 소요, "
            f"성공: {self.successful}, 실패: {self.failed}, "
            f"최종 속도: {self.total_prompts / elapsed:.2f} 요청/초"
        )
