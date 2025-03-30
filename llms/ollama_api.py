import time
import asyncio
import logging
import aiohttp
from aiohttp import ClientSession
from pandas import DataFrame, Series

from llms.prompt_generator import make_request, make_prompt
from util.config import BatchMode
from util.progress import ProgressTracker


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
                        dataset: list|DataFrame,
                        url: str = "http://localhost:11434/api/generate",
                        # batch_size: int = 10,
                        # max_retries: int = 3,
                        # max_concurrent: int = 10,
                        log_dir: str = "logs",
                        warmup: bool = True,
                        response_processor=None,
                        options: dict=None):  # 응답 처리 함수 매개변수 추가
    """병렬 처리를 위한 래퍼 함수 (로깅 기능 추가)"""
    logging.info(f"병렬 처리 시작: 총 {len(dataset)}개 요청 (배치 크기: {options['batch_size']}, 최대 동시 요청: {options['max_concurrent']})")

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
            response_processor=response_processor, # 응답 처리 함수 전달
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
                           dataset: list|DataFrame,
                           url: str = "http://localhost:11434/api/generate",
                           log_dir: str = ".",
                           response_processor=None,
                           options: dict=None):  # 응답 처리 함수 매개변수 추가

    # model, datasets, url="http://localhost:11434/api/generate",
    #                        batch_size=10, max_retries=3, max_concurrent=10):
    """프롬프트 배치에 대한 병렬 LLM 호출 (진행률 로깅 기능 추가)"""
    all_results = [None] * len(dataset)  # 순서 보존을 위한 결과 저장 리스트
    total_prompts = len(dataset)

    batch_size = options.get('batch_size')
    max_concurrent = options.get('max_concurrent')

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

        logging.debug("llm_invoke_single_with_processor: url=%s, request=%s", url, request)
        try:
            async with session.post(url, json=request) as response:
                if response.status != 200:
                    error_text = await response.text()
                    elapsed = time.time() - start_time
                    progress_tracker.update_task_progress(task_id, "failed", elapsed, error_text)
                    return {"error": error_text, "task_id": task_id}

                result = await response.json()
                elapsed = time.time() - start_time

                # 응답 처리 시도 (response_processor가 있는 경우)
                if response_processor:
                    try:
                        processed_result = response_processor(result, None)
                        result["processed_result"] = processed_result
                        progress_tracker.update_task_progress(task_id, "success", elapsed)
                        result["task_id"] = task_id  # 태스크 ID 추가
                        return result
                    except Exception as process_error:
                        # 응답 처리 오류 시 기록하고 원본 결과 반환
                        logging.error(f"응답 처리 중 오류 발생 (요청 #{task_id}): {str(process_error)}")
                        result["processing_error"] = str(process_error)
                        progress_tracker.update_task_progress(task_id, "success", elapsed)
                        result["task_id"] = task_id
                        return result
                else:
                    # 응답 처리 함수가 없는 경우 원본 결과 반환
                    progress_tracker.update_task_progress(task_id, "success", elapsed)
                    result["task_id"] = task_id  # 태스크 ID 추가
                    return result
        except Exception as e:
            elapsed = time.time() - start_time
            progress_tracker.update_task_progress(task_id, "failed", elapsed, str(e))
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
                task = asyncio.create_task(
                    llm_invoke_single_with_processor(model, data, progress_tracker, session, task_id, url, options)
                )
                tasks.append(task)

            # batch = dataset[batch_idx:batch_end]
            # # 배치 시작 로깅
            # progress_tracker.update_batch_progress(len(batch), batch_idx // batch_size, total_batches)
            #
            # # 비동기 작업 생성
            # tasks = []
            # # DataFrame과 list 모두 지원하도록 반복 처리
            # for i, data in enumerate(batch) if not hasattr(batch, 'iterrows') else batch.iterrows():
            #     # DataFrame의 경우 iterrows()는 (index, row) 튜플을 반환하므로 두 번째 요소 사용
            #     if hasattr(batch, 'iterrows'):
            #         _, data = data  # index와 row 추출
            # # for i, data in batch.iterrows():
            #     task_id = batch_idx + i
            #     task = asyncio.create_task(
            #         llm_invoke_single_with_processor(model, data, progress_tracker, session, task_id, url, options)
            #     )
            #     tasks.append(task)

            # 모든 작업 완료 대기
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 재시도가 필요한 작업 처리
            retry_tasks = []
            max_retries = options.get('max_retries')

            for result in batch_results:
                if isinstance(result, Exception):
                    logging.error(f"작업 예외 발생 1: {str(result)}")
                    continue

                task_id = result.get("task_id")
                if task_id is None:
                    logging.error(f"태스크 ID가 없는 결과: {result}")
                    continue

                # 오류 발생 여부 확인 및 재시도 결정
                if "error" in result and max_retries > 0:
                    retry_tasks.append((task_id, dataset[task_id], 0))  # (task_id, prompt, retry_count)
                else:
                    # 결과 저장 (원래 순서대로)
                    all_results[task_id] = result

            # 실패한 작업 재시도
            if retry_tasks and max_retries > 0:
                await retry_failed_tasks(session, retry_tasks, url, model,
                                         max_retries, all_results, progress_tracker,
                                         response_processor, options)  # 응답 처리 함수 전달

            # 배치 완료 로깅
            progress_tracker.update_batch_completion(batch_results)

            # 서버 부하 방지를 위한 짧은 대기
            if batch_end < total_prompts:
                await asyncio.sleep(1.0)

    # 진행률 표시 종료
    progress_tracker.close()

    return all_results


# async def llm_invoke_batch(model: str,
#                            dataset: list,
#                            url: str = "http://localhost:11434/api/generate",
#                            batch_size: int = 10,
#                            max_retries: int = 3,
#                            max_concurrent: int = 10,
#                            log_dir: str = "."):
#
#     # model, datasets, url="http://localhost:11434/api/generate",
#     #                        batch_size=10, max_retries=3, max_concurrent=10):
#     """프롬프트 배치에 대한 병렬 LLM 호출 (진행률 로깅 기능 추가)"""
#     all_results = [None] * len(dataset)  # 순서 보존을 위한 결과 저장 리스트
#     total_prompts = len(dataset)
#
#     # 배치 수 계산
#     total_batches = (total_prompts + batch_size - 1) // batch_size
#
#     # 진행률 추적 객체 생성
#     progress_tracker = ProgressTracker(total_prompts, batch_size, log_dir=log_dir)
#
#     # 동시 연결 제한을 위한 세마포어
#     semaphore = asyncio.Semaphore(max_concurrent)
#
#     # 커넥션 풀 설정으로 TCP 연결 재사용
#     conn = aiohttp.TCPConnector(limit=max_concurrent)
#     timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
#
#     async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
#         for batch_idx in range(0, total_prompts, batch_size):
#             batch_end = min(batch_idx + batch_size, total_prompts)
#             batch = dataset[batch_idx:batch_end]
#
#             # 배치 시작 로깅
#             progress_tracker.update_batch_progress(len(batch), batch_idx // batch_size, total_batches)
#
#             # 비동기 작업 생성
#             tasks = []
#             for i, data in enumerate(batch):
#                 task_id = batch_idx + i
#                 task = asyncio.create_task(
#                     llm_invoke_single(model, data, progress_tracker, session, task_id, url)
#                 )
#                 tasks.append(task)
#
#             # 모든 작업 완료 대기
#             batch_results = await asyncio.gather(*tasks, return_exceptions=True)
#
#             # 재시도가 필요한 작업 처리
#             retry_tasks = []
#
#             for result in batch_results:
#                 if isinstance(result, Exception):
#                     logging.error(f"작업 예외 발생 1: {str(result)}")
#                     continue
#
#                 task_id = result.get("task_id")
#                 if task_id is None:
#                     logging.error(f"태스크 ID가 없는 결과: {result}")
#                     continue
#
#                 # 오류 발생 여부 확인 및 재시도 결정
#                 if "error" in result and max_retries > 0:
#                     retry_tasks.append((task_id, dataset[task_id], 0))  # (task_id, prompt, retry_count)
#                 else:
#                     # 결과 저장 (원래 순서대로)
#                     all_results[task_id] = result
#
#             # 실패한 작업 재시도
#             if retry_tasks and max_retries > 0:
#                 await retry_failed_tasks(session, retry_tasks, url, model,
#                                          max_retries, all_results, progress_tracker)
#
#             # 배치 완료 로깅
#             progress_tracker.update_batch_completion(batch_results)
#
#             # 서버 부하 방지를 위한 짧은 대기
#             if batch_end < total_prompts:
#                 await asyncio.sleep(1.0)
#
#     # 진행률 표시 종료
#     progress_tracker.close()
#
#     return all_results


async def llm_invoke_single(model: str,
                            data: Series,
                            progress_tracker: ProgressTracker,
                            session: ClientSession,
                            task_id: int,
                            url: str = "http://localhost:11434/api/generate",
                            ):
    """진행률 추적 기능이 추가된 단일 프롬프트 비동기 LLM 호출"""
    # 요청 시작 로깅
    progress_tracker.update_task_progress(task_id, "start")
    start_time = time.time()

    # prompt = make_prompt(model_name, data['context'], data['question'])
    # request = make_request(model_name, prompt)
    # logging.info("data: [%s] %s", type(data), data['context'])

    prompt = make_prompt(model, data)
    # prompt = make_prompt(model, data, evaluation=False)
    request = make_request(model, prompt)

    logging.debug("llm_invoke_single: url=%s, request=%s", url, request)
    try:
        async with session.post(url, json=request) as response:
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


def llm_invoke_jobs_parallel(model, jobs, url, batch_size=10, max_retries=3, max_concurrent=10):
    """병렬 처리를 위한 래퍼 함수 (로깅 기능 추가)"""
    logging.info(f"병렬 처리 시작: 총 {len(jobs)}개 요청 (배치 크기: {batch_size}, 최대 동시 요청: {max_concurrent})")

    start_time = time.time()

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        llm_invoke_jobs_batch(
            model,
            jobs,
            url,
            batch_size=batch_size,
            max_retries=max_retries,
            max_concurrent=max_concurrent
        )
    )

    elapsed_time = time.time() - start_time

    logging.info(f"병렬 처리 완료: {len(results)}개 응답 수신")
    logging.info(f"평균 요청 처리 시간: {(elapsed_time / len(jobs)):.3f}초")

    return results


async def llm_invoke_jobs_batch(model_name, jobs, model_url="http://localhost:11434/api/chat",
                                batch_size=10, max_retries=3, max_concurrent=10):
    """프롬프트 배치에 대한 병렬 LLM 호출 (진행률 로깅 기능 추가)"""
    all_results = [None] * len(jobs)  # 순서 보존을 위한 결과 저장 리스트
    total_prompts = len(jobs)

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
            batch = jobs[batch_idx:batch_end]

            # 배치 시작 로깅
            progress_tracker.update_batch_progress(len(batch), batch_idx // batch_size, total_batches)

            # 비동기 작업 생성
            tasks = []
            for i, job in enumerate(batch):
                task_id = batch_idx + i
                task = asyncio.create_task(
                    llm_invoke_job(
                        session, job, model_url, task_id, progress_tracker
                    )
                )
                tasks.append(task)

            # 모든 작업 완료 대기
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 재시도가 필요한 작업 처리
            retry_tasks = []

            for result in batch_results:
                if isinstance(result, Exception):
                    logging.error(f"작업 예외 발생 2: {str(result)}")
                    continue

                task_id = result.get("task_id")
                if task_id is None:
                    logging.error(f"태스크 ID가 없는 결과: {result}")
                    continue

                # 오류 발생 여부 확인 및 재시도 결정
                if "error" in result and max_retries > 0:
                    retry_tasks.append((task_id, jobs[task_id], 0))  # (task_id, prompt, retry_count)
                else:
                    # 결과 저장 (원래 순서대로)
                    all_results[task_id] = result

            # 실패한 작업 재시도
            if retry_tasks and max_retries > 0:
                await retry_failed_job_tasks(session, retry_tasks, model_url,
                                             max_retries, all_results, progress_tracker)

            # 배치 완료 로깅
            progress_tracker.update_batch_completion(batch_results)

            # 서버 부하 방지를 위한 짧은 대기
            if batch_end < total_prompts:
                await asyncio.sleep(0.5)

    # 진행률 표시 종료
    progress_tracker.close()

    return all_results


async def llm_invoke_job(session, job, model_url, task_id, progress_tracker):
    """진행률 추적 기능이 추가된 단일 프롬프트 비동기 LLM 호출"""
    # 요청 시작 로깅
    progress_tracker.update_task_progress(task_id, "start")
    start_time = time.time()

    try:
        async with session.post(model_url, json=job) as response:
            if response.status != 200:
                error_text = await response.text()
                elapsed = time.time() - start_time
                progress_tracker.update_task_progress(task_id, "failed", elapsed, error_text)
                return {"error": error_text, "task_id": task_id}

            result = await response.json()
            logging.info("url: %s, request: %s", model_url, job)
            logging.info("result: %s", result['response'])
            elapsed = time.time() - start_time
            progress_tracker.update_task_progress(task_id, "success", elapsed)
            result["task_id"] = task_id  # 태스크 ID 추가
            return result
    except Exception as e:
        elapsed = time.time() - start_time
        progress_tracker.update_task_progress(task_id, "failed", elapsed, str(e))
        return {"error": str(e), "task_id": task_id}


async def retry_failed_tasks(session, retry_tasks, model_url, model_name,
                             max_retries, all_results, progress_tracker,
                             response_processor=None, options=None):  # 응답 처리 함수 매개변수 추가
    """실패한 작업을 재시도하는 함수"""
    retry_logger = logging.getLogger("RetryProcessor")

    # llm_invoke_single_with_processor와 유사한 재시도 래퍼 함수 정의
    async def retry_invoke_single(model, data, tracker, task_id, retry_count, options):
        """진행률 추적 기능이 추가된 재시도용 단일 LLM 호출"""
        # 재시도 상태 로깅
        tracker.update_task_progress(task_id, "retry", error=retry_count + 1)
        start_time = time.time()

        prompt = make_prompt(model, data, options)
        request = make_request(model, prompt, options)

        try:
            async with session.post(model_url, json=request) as response:
                if response.status != 200:
                    error_text = await response.text()
                    elapsed = time.time() - start_time
                    tracker.update_task_progress(task_id, "failed", elapsed, error_text)
                    return {"error": error_text, "task_id": task_id}

                result = await response.json()
                elapsed = time.time() - start_time

                # 응답 처리 시도 (response_processor가 있는 경우)
                if response_processor:
                    try:
                        processed_result = response_processor(result, None)
                        result["processed_result"] = processed_result
                        tracker.update_task_progress(task_id, "success", elapsed)
                        result["task_id"] = task_id
                        return result
                    except Exception as process_error:
                        # 응답 처리 오류 시 기록하고 원본 결과 반환
                        retry_logger.error(f"재시도 응답 처리 중 오류 발생 (요청 #{task_id}): {str(process_error)}")
                        result["processing_error"] = str(process_error)
                        tracker.update_task_progress(task_id, "success", elapsed)
                        result["task_id"] = task_id
                        return result
                else:
                    # 응답 처리 함수가 없는 경우 원본 결과 반환
                    tracker.update_task_progress(task_id, "success", elapsed)
                    result["task_id"] = task_id
                    return result
        except Exception as e:
            elapsed = time.time() - start_time
            tracker.update_task_progress(task_id, "failed", elapsed, str(e))
            return {"error": str(e), "task_id": task_id}

    for retry_round in range(max_retries):
        if not retry_tasks:
            break

        retry_logger.info(f"{len(retry_tasks)}개 작업 재시도 중 (라운드: {retry_round + 1}/{max_retries})")

        current_tasks = retry_tasks
        retry_tasks = []

        # 재시도 작업 생성
        tasks = []
        for task_id, data, retry_count in current_tasks:
            # 재시도 작업 생성
            task = asyncio.create_task(
                retry_invoke_single(model_name, data, progress_tracker, task_id, retry_count, options)
            )
            tasks.append((task, task_id, retry_count, data))

        # 재시도 작업 처리
        for task, task_id, retry_count, data in tasks:
            try:
                result = await task
                if "error" in result and retry_count < max_retries - 1:
                    # 다음 재시도 라운드에 추가
                    retry_tasks.append((task_id, data, retry_count + 1))
                else:
                    # 최종 결과 저장
                    all_results[task_id] = result
            except Exception as e:
                if retry_count < max_retries - 1:
                    retry_tasks.append((task_id, data, retry_count + 1))
                else:
                    all_results[task_id] = {"error": str(e), "task_id": task_id}

        # 다음 재시도 라운드 전에 짧은 대기
        if retry_tasks:
            await asyncio.sleep(1)


# def llm_invoke_parallel(model: str,
#                         dataset: list,
#                         url: str = "http://localhost:11434/api/generate",
#                         batch_size: int = 10,
#                         max_retries: int = 3,
#                         max_concurrent: int = 10,
#                         log_dir: str = "logs",
#                         warmup: bool = True):
#     """병렬 처리를 위한 래퍼 함수 (로깅 기능 추가)"""
#     logging.info(f"병렬 처리 시작: 총 {len(dataset)}개 요청 (배치 크기: {batch_size}, 최대 동시 요청: {max_concurrent})")
#
#     # 로그 디렉터리를 절대 경로로 변환
#     import os
#     abs_log_dir = os.path.abspath(log_dir)
#     logging.debug(f"로그 디렉터리: {abs_log_dir}")
#
#     # 전체 실행 시간 측정 시작
#     total_start_time = time.time()
#
#     # 1. 모델 예열 (옵션에 따라 실행)
#     loading_time = 0
#     if warmup:
#         warmup_result = warmup_model_sync(model, url)
#         loading_time = warmup_result.get("loading_time", 0)
#
#         if not warmup_result.get("success", False):
#             logging.warning(f"모델 예열 중 오류가 발생했습니다: {warmup_result.get('error', '알 수 없는 오류')}")
#             logging.warning("오류가 발생했지만 계속 진행합니다...")
#
#     # 2. 실제 배치 처리 시간 측정 시작
#     batch_start_time = time.time()
#
#     loop = asyncio.get_event_loop()
#     results = loop.run_until_complete(
#         llm_invoke_batch(
#             model=model,
#             dataset=dataset,
#             url=url,
#             batch_size=batch_size,
#             max_retries=max_retries,
#             max_concurrent=max_concurrent,
#             log_dir=abs_log_dir  # 절대 경로 사용
#         )
#     )
#
#     # 배치 처리 시간 계산
#     batch_elapsed_time = time.time() - batch_start_time
#
#     # 전체 시간 계산 (모델 로딩 시간 + 배치 처리 시간)
#     total_elapsed_time = time.time() - total_start_time
#
#     # 요약 정보 출력
#     logging.info(f"\n===== 병렬 처리 결과 요약 =====")
#     if warmup:
#         logging.info(f"모델 '{model}' 로딩 시간: {loading_time:.2f}초")
#     logging.info(f"배치 처리 시간: {batch_elapsed_time:.2f}초")
#     logging.info(f"총 실행 시간: {total_elapsed_time:.2f}초")
#     logging.info(f"처리된 요청 수: {len(results)}개")
#     if len(dataset) > 0:
#         logging.info(f"평균 요청 처리 시간: {(batch_elapsed_time / len(dataset)):.3f}초/요청")
#         logging.info(f"초당 처리량: {(len(dataset) / batch_elapsed_time):.2f}개/초")
#     logging.info(f"===============================\n")
#
#     return results





# async def retry_failed_tasks(session, retry_tasks, model_url, model_name,
#                              max_retries, all_results, progress_tracker):
#     """실패한 작업을 재시도하는 함수"""
#     retry_logger = logging.getLogger("RetryProcessor")
#
#     for retry_round in range(max_retries):
#         if not retry_tasks:
#             break
#
#         retry_logger.info(f"{len(retry_tasks)}개 작업 재시도 중 (라운드: {retry_round + 1}/{max_retries})")
#
#         current_tasks = retry_tasks
#         retry_tasks = []
#
#         # 재시도 작업 생성
#         tasks = []
#         for task_id, prompt, retry_count in current_tasks:
#             # 재시도 로깅
#             progress_tracker.update_task_progress(task_id, "retry", error=retry_count + 1)
#
#             # 재시도 요청 준비
#             task = asyncio.create_task(
#                 llm_invoke_single(
#                     session, prompt, model_url, model_name, task_id, progress_tracker
#                 )
#             )
#             tasks.append((task, task_id, retry_count, prompt))
#
#         # 재시도 작업 처리
#         for task, task_id, retry_count, prompt in tasks:
#             try:
#                 result = await task
#                 if "error" in result and retry_count < max_retries - 1:
#                     # 다음 재시도 라운드에 추가
#                     retry_tasks.append((task_id, prompt, retry_count + 1))
#                 else:
#                     # 최종 결과 저장
#                     all_results[task_id] = result
#             except Exception as e:
#                 if retry_count < max_retries - 1:
#                     retry_tasks.append((task_id, prompt, retry_count + 1))
#                 else:
#                     all_results[task_id] = {"error": str(e), "task_id": task_id}
#
#         # 다음 재시도 라운드 전에 짧은 대기
#         if retry_tasks:
#             await asyncio.sleep(1)


async def retry_failed_job_tasks(session, retry_tasks, model_url,
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
        for task_id, job, retry_count in current_tasks:
            # 재시도 로깅
            progress_tracker.update_task_progress(task_id, "retry", error=retry_count + 1)

            # 재시도 요청 준비
            task = asyncio.create_task(
                llm_invoke_job(
                    session, job, model_url, task_id, progress_tracker
                )
            )
            tasks.append((task, task_id, retry_count, job))

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
