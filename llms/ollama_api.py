import time
import asyncio
import logging
import aiohttp

from llms.prompt_generator import make_request, make_prompt
from util.progress import ProgressTracker


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
            logging.info("url: %s, request: %s", model_url,job)
            logging.info("result: %s", result['response'])
            elapsed = time.time() - start_time
            progress_tracker.update_task_progress(task_id, "success", elapsed)
            result["task_id"] = task_id  # 태스크 ID 추가
            return result
    except Exception as e:
        elapsed = time.time() - start_time
        progress_tracker.update_task_progress(task_id, "failed", elapsed, str(e))
        return {"error": str(e), "task_id": task_id}


async def llm_invoke_single(session, data, model_url, model_name, task_id, progress_tracker):
    """진행률 추적 기능이 추가된 단일 프롬프트 비동기 LLM 호출"""
    # 요청 시작 로깅
    progress_tracker.update_task_progress(task_id, "start")
    start_time = time.time()

    prompt = make_prompt(model_name, data['context'], data['question'])
    request = make_request(model_name, prompt)

    logging.info("request: %", request)

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


async def llm_invoke_batch(datasets, model_name, model_url="http://localhost:11434/api/generate",
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


async def llm_invoke_jobs_batch(model_name, jobs, model_url="http://172.16.15.112:11434/api/chat",
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
                    logging.error(f"작업 예외 발생: {str(result)}")
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


def llm_invoke_jobs_parallel(model, jobs, url ,batch_size=10, max_retries=3, max_concurrent=10):
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
    logging.info(f"평균 요청 처리 시간: {elapsed_time / len(jobs)}초")

    return results


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