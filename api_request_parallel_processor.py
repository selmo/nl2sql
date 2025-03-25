import json
import time
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
from dataclasses import dataclass, field
from util.progress import ProgressTracker


async def process_api_requests_from_file(
        requests_filepath: str,
        save_filepath: str,
        request_url: str,
        api_key: str,
        max_requests_per_minute: float,
        max_tokens_per_minute: float,
        token_encoding_name: str,
        max_attempts: int,
        logging_level: int,
        max_concurrent_requests: int = 10,  # 추가: 동시 요청 수 제한
        progress_tracker: ProgressTracker = None,  # 추가: 진행률 추적기
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if '/deployments' in request_url:
        request_header = {"api-key": f"{api_key}"}

    # 총 요청 수 계산
    total_requests = 0
    with open(requests_filepath) as file:
        for _ in file:
            total_requests += 1

    # ProgressTracker 초기화 (인자로 받지 않았을 경우)
    if progress_tracker is None:
        batch_size = min(20, total_requests)  # 기본 배치 크기
        progress_tracker = ProgressTracker(total_requests, batch_size)

    logging.debug(f"ProgressTracker initialized with {total_requests} total requests")

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 0, 1, 2, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call
    active_tasks = set()  # 현재 활성화된 작업 추적
    completed_tasks = set()  # 완료된 작업 추적

    # 동시 요청 제한을 위한 세마포어
    request_semaphore = asyncio.Semaphore(max_concurrent_requests)

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # 배치 처리 추적 변수
    current_batch_size = 0
    batch_start_index = 0
    total_batches = (total_requests + progress_tracker.batch_size - 1) // progress_tracker.batch_size
    current_batch = 0

    # API 호출 래퍼 함수
    async def execute_api_call(request_obj):
        async with request_semaphore:  # 세마포어로 동시 요청 제한
            start_time = time.time()
            # 요청 시작 시 상태 업데이트
            progress_tracker.update_task_progress(request_obj.task_id, "start")

            try:
                result = await request_obj.call_api(
                    session=session,
                    request_url=request_url,
                    request_header=request_header,
                    retry_queue=queue_of_requests_to_retry,
                    save_filepath=save_filepath,
                    status_tracker=status_tracker,
                )

                # 성공 시 상태 업데이트
                if result is not None:  # None이 아니면 성공 (None은 재시도 필요)
                    elapsed = time.time() - start_time
                    progress_tracker.update_task_progress(request_obj.task_id, "success", elapsed=elapsed)
                    completed_tasks.add(request_obj.task_id)
                return result
            except Exception as e:
                # 실패 시 상태 업데이트
                elapsed = time.time() - start_time
                progress_tracker.update_task_progress(request_obj.task_id, "failed", elapsed=elapsed, error=str(e))

                # 재시도 가능한 경우
                if request_obj.attempts_left > 0:
                    logging.warning(f"Error in request #{request_obj.task_id}: {str(e)}. Will retry.")
                    await queue_of_requests_to_retry.put(request_obj)
                else:
                    # 최종 실패
                    logging.error(f"Request #{request_obj.task_id} failed after all attempts: {str(e)}")
                    completed_tasks.add(request_obj.task_id)
                return None

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = await queue_of_requests_to_retry.get()
                        logging.debug(
                            f"Retrying request #{next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))

                            # "index" 키가 있는 경우에만 삭제
                            if "index" in request_json:
                                del request_json["index"]

                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                max_attempts=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request #{next_request.task_id}: {next_request}"
                            )

                            # 배치 처리 로직
                            current_batch_size += 1
                            if current_batch_size == 1:
                                # 배치 시작 로깅
                                progress_tracker.update_batch_progress(
                                    progress_tracker.batch_size,
                                    current_batch,
                                    total_batches
                                )

                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                            # 마지막 배치 완료 처리
                            if current_batch_size > 0:
                                batch_results = [{"success": True} for _ in range(current_batch_size)]
                                progress_tracker.update_batch_completion(batch_results)
                                current_batch_size = 0
                                current_batch += 1

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                            available_request_capacity >= 1
                            and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API with semaphore and progress tracking
                        task = asyncio.create_task(execute_api_call(next_request))
                        active_tasks.add(task)

                        # 작업 완료 시 active_tasks에서 제거하는 콜백
                        def remove_task(completed_task):
                            if completed_task in active_tasks:
                                active_tasks.remove(completed_task)

                        task.add_done_callback(remove_task)
                        next_request = None  # reset next_request to empty

                        # 배치 크기가 채워졌으면 배치 완료 처리
                        if current_batch_size >= progress_tracker.batch_size:
                            batch_results = [{"success": True} for _ in range(current_batch_size)]
                            progress_tracker.update_batch_completion(batch_results)
                            current_batch_size = 0
                            batch_start_index += progress_tracker.batch_size
                            current_batch += 1

                # 완료된 작업 정리 및 실패한 작업 확인
                tasks_to_remove = set()
                for task in active_tasks:
                    if task.done():
                        tasks_to_remove.add(task)
                        try:
                            task.result()  # 예외 발생 여부 확인
                        except Exception as e:
                            logging.error(f"Task failed with exception: {e}")

                for task in tasks_to_remove:
                    active_tasks.remove(task)

                # 종료 조건 개선: 모든 작업이 완료되었는지 확인
                if (status_tracker.num_tasks_in_progress == 0 and
                        len(active_tasks) == 0 and
                        queue_of_requests_to_retry.empty() and
                        not file_not_finished):
                    logging.info("All tasks completed - process terminating")
                    break

                # 추가 안전장치: 모든 요청이 처리되었지만 추적 카운터가 일치하지 않는 경우
                if (not file_not_finished and
                        queue_of_requests_to_retry.empty() and
                        len(active_tasks) == 0 and
                        status_tracker.num_tasks_in_progress > 0):
                    logging.warning(
                        f"Mismatch in tracking: {status_tracker.num_tasks_in_progress} tasks still marked as in progress "
                        f"but no active tasks remain. Forcing completion."
                    )
                    status_tracker.num_tasks_in_progress = 0
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                        time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                        seconds_since_rate_limit_error
                        < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                            seconds_to_pause_after_rate_limit_error
                            - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warning(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )

    # 최종 진행 상태 마무리
    progress_tracker.close()


def process_by_file(
        requests_filepath: str,
        save_filepath: str = None,
        request_url: str = "https://api.openai.com/v1/chat/completions",
        api_key: str = os.getenv("OPENAI_API_KEY"),
        max_requests_per_minute: float = 1500,
        max_tokens_per_minute: float = 125000,
        token_encoding_name: str = "cl100k_base",
        max_attempts: int = 5,
        logging_level: int = logging.INFO,
        max_concurrent_requests: int = 10,  # 추가: 동시 요청 수 제한
        batch_size: int = 20,  # 추가: 배치 크기 매개변수
):
    """
    Process a batch of API requests from a JSONL file with rate limiting and parallel execution.

    Args:
        requests_filepath: Path to JSONL file with API requests.
        save_filepath: Path where results will be saved. Defaults to requests_filepath with '_results' suffix.
        request_url: API endpoint URL.
        api_key: API key for authentication.
        max_requests_per_minute: Maximum number of requests allowed per minute.
        max_tokens_per_minute: Maximum number of tokens allowed per minute.
        token_encoding_name: Name of token encoding for counting tokens.
        max_attempts: Maximum number of retry attempts per request.
        logging_level: Logging level (e.g., logging.INFO).
        max_concurrent_requests: Maximum number of concurrent API requests.
        batch_size: Size of processing batches for progress tracking.
    """
    # 결과 파일 경로 설정
    if save_filepath is None:
        save_filepath = requests_filepath.replace(".jsonl", "_results.jsonl")

    # 총 요청 수 계산
    total_requests = 0
    try:
        with open(requests_filepath) as file:
            for _ in file:
                total_requests += 1
    except Exception as e:
        logging.error(f"Error counting requests in file: {e}")
        return

    logging.info(f"Processing {total_requests} requests from {requests_filepath}")
    logging.info(f"Results will be saved to {save_filepath}")
    logging.info(f"Rate limits: {max_requests_per_minute} requests/min, {max_tokens_per_minute} tokens/min")
    logging.info(f"Concurrent requests limit: {max_concurrent_requests}")

    # 배치 크기 조정 - 너무 작거나 크지 않도록
    batch_size = min(max(5, batch_size), 100)  # 최소 5, 최대 100
    batch_size = min(batch_size, total_requests)  # 총 요청 수보다 크지 않도록

    # ProgressTracker 초기화
    progress_tracker = ProgressTracker(total_requests, batch_size)
    logging.info(f"Using batch size of {batch_size} for progress tracking")

    try:
        # 비동기 처리 실행
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=requests_filepath,
                save_filepath=save_filepath,
                request_url=request_url,
                api_key=api_key,
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                token_encoding_name=token_encoding_name,
                max_attempts=max_attempts,
                logging_level=logging_level,
                max_concurrent_requests=max_concurrent_requests,  # 동시 요청 수 제한 전달
                progress_tracker=progress_tracker,  # 진행률 추적기 전달
            )
        )
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    except Exception as e:
        logging.error(f"Error in process_by_file: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # 진행률 추적기 종료 - 에러가 발생해도 항상 실행
        try:
            progress_tracker.close()
        except Exception as e:
            logging.error(f"Error closing progress tracker: {e}")

    # 처리 결과 확인
    try:
        successful_count = 0
        failed_count = 0

        with open(save_filepath) as f:
            for line in f:
                result = json.loads(line)
                if isinstance(result, list) and len(result) > 1:
                    # 두 번째 요소가 에러 객체인지 확인
                    if isinstance(result[1], dict) and "error" in result[1]:
                        failed_count += 1
                    else:
                        successful_count += 1

        completion_percentage = (successful_count + failed_count) / total_requests * 100 if total_requests > 0 else 0
        success_percentage = successful_count / total_requests * 100 if total_requests > 0 else 0

        logging.info(f"Processing summary:")
        logging.info(f"  Total requests: {total_requests}")
        logging.info(f"  Completed: {successful_count + failed_count} ({completion_percentage:.1f}%)")
        logging.info(f"  Successful: {successful_count} ({success_percentage:.1f}%)")
        logging.info(f"  Failed: {failed_count} ({(failed_count / total_requests * 100):.1f}%)")
    except Exception as e:
        logging.error(f"Error analyzing results: {e}")


# async def process_api_requests_from_file(
#         requests_filepath: str,
#         save_filepath: str,
#         request_url: str,
#         api_key: str,
#         max_requests_per_minute: float,
#         max_tokens_per_minute: float,
#         token_encoding_name: str,
#         max_attempts: int,
#         logging_level: int,
#         total_requests: int,
#         max_concurrent_requests: int = 10,
#         progress_tracker: ProgressTracker = None,  # 추가: 진행률 추적기
# ):
#     # 함수 시작 부분, 변수 초기화 섹션에 추가
#     # 동시 요청 제한을 위한 세마포어
#     request_semaphore = asyncio.Semaphore(max_concurrent_requests)
#
#     # ProgressTracker 초기화 (인자로 받지 않았을 경우)
#     if progress_tracker is None:
#         batch_size = min(10, total_requests)  # 기본 배치 크기
#         progress_tracker = ProgressTracker(total_requests, batch_size)
#
#     logging.debug(f"ProgressTracker initialized with {total_requests} total requests")
#
#     # 활성 작업 추적
#     active_tasks = set()
#
#     # StatusTracker 초기화 후, next_request 변수 정의 전에 추가
#     # API 호출 래퍼 함수
#     async def execute_api_call(request_obj):
#         async with request_semaphore:  # 세마포어로 동시 요청 제한
#             return await request_obj.call_api(
#                 session=session,
#                 request_url=request_url,
#                 request_header=request_header,
#                 retry_queue=queue_of_requests_to_retry,
#                 save_filepath=save_filepath,
#                 status_tracker=status_tracker,
#             )
#
#     """Processes API requests in parallel, throttling to stay under rate limits."""
#     # constants
#     seconds_to_pause_after_rate_limit_error = 15
#     seconds_to_sleep_each_loop = (
#         0.001  # 1 ms limits max throughput to 1,000 requests per second
#     )
#
#     # initialize logging
#     logging.basicConfig(level=logging_level)
#     logging.debug(f"Logging initialized at level {logging_level}")
#
#     # infer API endpoint and construct request header
#     api_endpoint = api_endpoint_from_url(request_url)
#     request_header = {"Authorization": f"Bearer {api_key}"}
#     # use api-key header for Azure deployments
#     if '/deployments' in request_url:
#         request_header = {"api-key": f"{api_key}"}
#
#     # initialize trackers
#     queue_of_requests_to_retry = asyncio.Queue()
#     task_id_generator = (
#         task_id_generator_function()
#     )  # generates integer IDs of 0, 1, 2, ...
#     status_tracker = (
#         StatusTracker()
#     )  # single instance to track a collection of variables
#     next_request = None  # variable to hold the next request to call
#
#     # initialize available capacity counts
#     available_request_capacity = max_requests_per_minute
#     available_token_capacity = max_tokens_per_minute
#     last_update_time = time.time()
#
#     # initialize flags
#     file_not_finished = True  # after file is empty, we'll skip reading it
#     logging.debug(f"Initialization complete.")
#
#     # initialize file reading
#     with open(requests_filepath) as file:
#         # `requests` will provide requests one at a time
#         requests = file.__iter__()
#         logging.debug(f"File opened. Entering main loop")
#         async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
#             while True:
#                 # get next request (if one is not already waiting for capacity)
#                 if next_request is None:
#                     if not queue_of_requests_to_retry.empty():
#                         next_request = queue_of_requests_to_retry.get_nowait()
#                         logging.debug(
#                             f"Retrying request #{next_request.task_id}: {next_request}"
#                         )
#                     elif file_not_finished:
#                         try:
#                             # get new request
#                             request_json = json.loads(next(requests))
#                             next_request = APIRequest(
#                                 task_id=next(task_id_generator),
#                                 request_json=request_json,
#                                 token_consumption=num_tokens_consumed_from_request(
#                                     request_json, api_endpoint, token_encoding_name
#                                 ),
#                                 attempts_left=max_attempts,
#                                 max_attempts=max_attempts,  # 이 매개변수를 추가했습니다
#                                 metadata=request_json.pop("metadata", None),
#                             )
#                             status_tracker.num_tasks_started += 1
#                             status_tracker.num_tasks_in_progress += 1
#                             logging.debug(
#                                 f"Reading request #{next_request.task_id}: {next_request}"
#                             )
#                         except StopIteration:
#                             # if file runs out, set flag to stop reading it
#                             logging.debug("Read file exhausted")
#                             file_not_finished = False
#
#                 # update available capacity
#                 current_time = time.time()
#                 seconds_since_update = current_time - last_update_time
#                 available_request_capacity = min(
#                     available_request_capacity
#                     + max_requests_per_minute * seconds_since_update / 60.0,
#                     max_requests_per_minute,
#                 )
#                 available_token_capacity = min(
#                     available_token_capacity
#                     + max_tokens_per_minute * seconds_since_update / 60.0,
#                     max_tokens_per_minute,
#                 )
#                 last_update_time = current_time
#
#                 # if enough capacity available, call API
#                 if next_request:
#                     next_request_tokens = next_request.token_consumption
#                     if (
#                         available_request_capacity >= 1
#                         and available_token_capacity >= next_request_tokens
#                     ):
#                         # update counters
#                         available_request_capacity -= 1
#                         available_token_capacity -= next_request_tokens
#                         next_request.attempts_left -= 1
#
#
#                         # call API with semaphore
#                         task = asyncio.create_task(
#                             execute_api_call(next_request)
#                         )
#                         # call API
#                         # asyncio.create_task(
#                         #     next_request.call_api(
#                         #         session=session,
#                         #         request_url=request_url,
#                         #         request_header=request_header,
#                         #         retry_queue=queue_of_requests_to_retry,
#                         #         save_filepath=save_filepath,
#                         #         status_tracker=status_tracker,
#                         #     )
#                         # )
#                         next_request = None  # reset next_request to empty
#
#                 # if all tasks are finished, break
#                 if status_tracker.num_tasks_in_progress == 0:
#                     break
#
#                 # main loop sleeps briefly so concurrent tasks can run
#                 await asyncio.sleep(seconds_to_sleep_each_loop)
#
#                 # if a rate limit error was hit recently, pause to cool down
#                 seconds_since_rate_limit_error = (
#                     time.time() - status_tracker.time_of_last_rate_limit_error
#                 )
#                 if (
#                     seconds_since_rate_limit_error
#                     < seconds_to_pause_after_rate_limit_error
#                 ):
#                     remaining_seconds_to_pause = (
#                         seconds_to_pause_after_rate_limit_error
#                         - seconds_since_rate_limit_error
#                     )
#                     await asyncio.sleep(remaining_seconds_to_pause)
#                     # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
#                     logging.warning(
#                         f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
#                     )
#
#         # after finishing, log final status
#         logging.info(
#             f"""Parallel processing complete. Results saved to {save_filepath}"""
#         )
#         if status_tracker.num_tasks_failed > 0:
#             logging.warning(
#                 f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
#             )
#         if status_tracker.num_rate_limit_errors > 0:
#             logging.warning(
#                 f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
#             )


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


# APIRequest 클래스에 max_attempts 필드 추가하기
@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    max_attempts: int  # 이 필드 추가
    metadata: dict = None
    result: list = field(default_factory=list)

    async def call_api(
            self,
            session: aiohttp.ClientSession,
            request_url: str,
            request_header: dict,
            retry_queue: asyncio.Queue,
            status_tracker: StatusTracker,
            save_filepath: str = None,
            timeout: int = 60,
            progress_tracker: ProgressTracker = None,  # 추가: 진행률 추적기
    ):
        """Calls the API and handles various error conditions."""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting request #{self.task_id}")

        # 요청 시작 시 상태 업데이트
        if progress_tracker:
            progress_tracker.update_task_progress(self.task_id, "start")

        start_time = time.time()

        error = None
        result_data = None

        try:
            # Make the API call
            try:
                async with session.post(
                        url=request_url,
                        headers=request_header,
                        json=self.request_json,
                        timeout=timeout,
                        raise_for_status=False  # Changed to handle status codes explicitly
                ) as http_response:
                    # Check for HTTP error status codes
                    if http_response.status >= 400:
                        error_text = await http_response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=http_response.request_info,
                            history=http_response.history,
                            status=http_response.status,
                            message=f"HTTP error {http_response.status}: {error_text}",
                            headers=http_response.headers
                        )

                    response_json = await http_response.json()

                    # 성공 시 상태 업데이트
                    if progress_tracker:
                        elapsed = time.time() - start_time
                        progress_tracker.update_task_progress(self.task_id, "success", elapsed=elapsed)

            except aiohttp.ClientConnectionError as e:
                logger.warning(f"Request #{self.task_id} connection error: {str(e)}")
                raise ConnectionError(f"Server connection error: {str(e)}")
            except aiohttp.ClientResponseError as e:
                logger.warning(f"Request #{self.task_id} HTTP error: {e.status} - {e.message}")
                raise ConnectionError(f"HTTP error {e.status}: {e.message}")
            except asyncio.TimeoutError:
                logger.warning(f"Request #{self.task_id} timeout error")
                # 타임아웃 발생 시 명확한 처리
                if self.attempts_left > 0:
                    self.attempts_left -= 1
                    # 직접 재시도 큐에 추가
                    await retry_queue.put(self)
                    logger.info(f"Request #{self.task_id} queued for retry ({self.attempts_left} attempts left)")
                    return None
                else:
                    # 재시도 횟수 소진
                    error_data = [self.request_json, {"error": "Request timed out after all retries"}, self.metadata]
                    if save_filepath:
                        append_to_jsonl(error_data, save_filepath)
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1
                    return error_data
                # logger.warning(f"Request #{self.task_id} timeout error")
                # raise TimeoutError("Request timed out")
            except Exception as e:
                logger.error(f"Request #{self.task_id} unexpected error during HTTP request: {str(e)}")
                if progress_tracker:
                    elapsed = time.time() - start_time
                    progress_tracker.update_task_progress(self.task_id, "failed", elapsed=elapsed, error=str(e))
                raise

            # Process API response
            if "error" in response_json:
                logger.warning(f"Request #{self.task_id} failed with API error: {response_json['error']}")
                status_tracker.num_api_errors += 1
                error = response_json

                # Check for rate limit errors
                error_message = response_json["error"].get("message", "")
                if "Rate limit" in error_message or "rate_limit" in error_message.lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
                    logger.warning(f"Rate limit error for request #{self.task_id}: {error_message}")
                    raise ConnectionError(f"Rate limit error: {error_message}")
            else:
                logger.info(f"Response #{self.task_id}: OK")
                result_data = [self.request_json, response_json, self.metadata] if self.metadata else [
                    self.request_json, response_json]

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Request #{self.task_id} failed with error: {str(e)}")
            status_tracker.num_other_errors += 1
            error = e

            # Store the error in the result list
            self.result.append(str(e))

            # If we have attempts left, return None to signal retry is needed
            if self.attempts_left > 0:
                return None

        except Exception as e:
            logger.error(f"Request #{self.task_id} failed with unexpected exception: {str(e)}")
            status_tracker.num_other_errors += 1
            error = e
            self.result.append(str(e))

        # Handle errors and prepare result
        if error:
            if self.attempts_left > 0:
                # Return None to signal caller that retry is needed
                return None
            else:
                # No more retries, create error result
                logger.error(f"Request #{self.task_id} failed after all attempts. Final error: {error}")
                data = (
                    [self.request_json, {"error": [str(e) for e in self.result]}, self.metadata]
                    if self.metadata
                    else [self.request_json, {"error": [str(e) for e in self.result]}]
                )
                result_data = data

                # Save to file if needed
                if save_filepath is not None:
                    append_to_jsonl(data, save_filepath)

                # Update status
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            # Successful result
            data = result_data
            if save_filepath is not None:
                append_to_jsonl(data, save_filepath)
                logger.debug(f"Request #{self.task_id} saved to {save_filepath}")

            # Update status
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

        return result_data


def api_endpoint_from_url(request_url: str) -> str:
    """
    Extract the API endpoint from the request_url.

    처리 예시:
      - Azure:
          https://<azure_host>/openai/deployments/<dep>/chat/completions?param=foo
            -> "chat/completions"
      - OpenAI:
          https://api.openai.com/v1/chat/completions -> "chat/completions"
          https://api.openai.com/v1/embeddings      -> "embeddings"
      - 로컬/사설 IP (포트:11434, /api/...):
          http://...:11434/api/generate             -> "generate"
          http://...:11434/api/chat                 -> "chat"
      - 위에 해당되지 않으면 None
    """

    # 1) Azure 스타일
    #    예: https://<azure_host>/openai/deployments/mydep/chat/completions?foo=bar
    match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    if match:
        return match.group(1)

    # 2) OpenAI 스타일
    #    예: https://api.openai.com/v1/chat/completions
    #        https://api.openai.com/v1/embeddings
    match = re.search(r"^https://[^/]+/v\d+/(.+)$", request_url)
    if match:
        return match.group(1)

    # 3) 로컬 혹은 사설 IP 형태
    #    예: http://localhost:11434/api/generate
    #         http://172.16.15.112:11434/api/chat
    #         http://127.0.0.1:11434/api/generate
    match = re.search(r"^http://[^/]+:\d+/api/(.+)$", request_url)
    if match:
        return match.group(1)

    # 4) 위 패턴에 해당하지 않으면 None
    return None


# def api_endpoint_from_url(request_url):
#     """Extract the API endpoint from the request URL."""
#     match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
#     if match is None:
#         # for Azure OpenAI deployment urls
#         match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
#     if match is None:
#         return "generate"
#     else:
#         return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions") or api_endpoint.endswith("generate") or api_endpoint.endswith("chat"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# def process_by_file(
#         requests_filepath: str,
#         save_filepath: str = None,
#         request_url: str = "https://api.openai.com/v1/chat/completions",
#         api_key: str = os.getenv("OPENAI_API_KEY"),
#         max_requests_per_minute: float = 1500,
#         max_tokens_per_minute: float = 125000,
#         token_encoding_name: str = "cl100k_base",
#         max_attempts: int = 5,
#         logging_level: int = logging.INFO,
#         max_concurrent_requests: int = 10,
# ):
#     if save_filepath is None:
#         save_filepath = requests_filepath.replace(".jsonl", "_results.jsonl")
#
#     # run script
#     asyncio.run(
#         process_api_requests_from_file(
#             requests_filepath,
#             save_filepath,
#             request_url,
#             api_key,
#             max_requests_per_minute,
#             max_tokens_per_minute,
#             token_encoding_name,
#             max_attempts,
#             logging_level,
#             max_concurrent_requests,  # 추가: 동시 요청 수 제한
#         )
#     )


# async def process_api_requests_in_memory(
#         requests_list,
#         request_url: str,
#         max_requests_per_minute: float,
#         max_tokens_per_minute: float,
#         token_encoding_name: str,
#         max_attempts: int,
#         api_key: str = '',
#         request_timeout: int = 300,  # Individual request timeout
#         max_concurrent_requests: int = 100  # Maximum concurrent requests
# ):
#     """
#     Process a list of API requests in memory with parallel execution and robust retry mechanism.
#
#     Args:
#         requests_list: List of API request parameters
#         request_url: API endpoint URL
#         max_requests_per_minute: Rate limit for requests per minute
#         max_tokens_per_minute: Rate limit for tokens per minute
#         token_encoding_name: Token encoding name
#         max_attempts: Maximum retry attempts per request
#         api_key: API key for authentication
#         request_timeout: Timeout for individual requests in seconds (None for no timeout)
#         max_concurrent_requests: Maximum number of concurrent requests
#
#     Returns:
#         List of (request, response) tuples
#     """
#     logger = logging.getLogger(__name__)
#
#     # Process start time
#     process_start_time = time.time()
#
#     # Initialize status tracker
#     status_tracker = StatusTracker()
#     task_id_gen = task_id_generator_function()
#
#     # API endpoint and headers
#     api_endpoint = api_endpoint_from_url(request_url)
#     request_header = {"Authorization": f"Bearer {api_key}"}
#     if "/deployments" in request_url:
#         request_header = {"api-key": f"{api_key}"}
#
#     # Rate limiting parameters
#     seconds_to_pause_after_rate_limit_error = 15
#     seconds_to_sleep_each_loop = 0.001
#
#     # Capacity tracking
#     available_request_capacity = max_requests_per_minute
#     available_token_capacity = max_tokens_per_minute
#     last_update_time = time.time()
#
#     # Queues and tracking
#     queue_of_requests_to_retry = asyncio.Queue()
#     current_index = 0
#     total_requests = len(requests_list)
#
#     # Results storage
#     results = []
#
#     # Active task tracking
#     active_tasks = {}  # task_id -> asyncio.Task
#
#     # Completed task tracking
#     completed_tasks = set()  # Set of completed task IDs
#
#     # Track scheduled retries
#     retry_scheduled = {}  # task_id -> Future
#
#     # Progress tracking
#     last_progress_time = time.time()
#     progress_interval = 10  # Log progress every 10 seconds
#
#     # Define timeout for HTTP connection
#     conn_timeout = aiohttp.ClientTimeout(
#         total=request_timeout if request_timeout else None,
#         connect=min(10, request_timeout if request_timeout else 60),
#         sock_read=min(90, request_timeout if request_timeout else 300)
#     )
#
#     # Maximum concurrent connections
#     conn_limit = min(max_concurrent_requests, max_requests_per_minute // 10)
#
#     logger.info(f"Starting processing {total_requests} requests with {max_concurrent_requests} concurrent limit")
#
#     # Create a ClientSession with improved settings
#     async with aiohttp.ClientSession(
#             timeout=conn_timeout,
#             connector=aiohttp.TCPConnector(
#                 limit=conn_limit,
#                 force_close=False,
#                 keepalive_timeout=60
#             )
#     ) as session:
#         # Helper function to schedule retries with backoff
#         async def schedule_retry(task_id, req_obj, delay):
#             try:
#                 if req_obj.attempts_left <= 0:
#                     logger.warning(f"Request #{task_id} has no attempts left. Not scheduling retry.")
#                     return
#
#                 logger.info(
#                     f"Scheduling retry for request #{task_id} after {delay}s delay ({req_obj.attempts_left} attempts left)")
#
#                 # Apply delay before retry
#                 await asyncio.sleep(delay)
#
#                 # Put back in queue
#                 await queue_of_requests_to_retry.put(req_obj)
#                 logger.info(f"Added request #{task_id} back to retry queue after delay")
#             except Exception as e:
#                 logger.error(f"Error in schedule_retry for task #{task_id}: {e}")
#             finally:
#                 # Remove from tracking once scheduled
#                 if task_id in retry_scheduled:
#                     del retry_scheduled[task_id]
#
#         # Define task callback function
#         def make_callback(task_id, request_obj):
#             def callback(task):
#                 try:
#                     # Process completed task
#                     if not task.cancelled():
#                         try:
#                             api_result = task.result()
#                             if api_result:
#                                 # Success case - store result
#                                 results.append(api_result)
#                                 logger.info(f"Request #{task_id} completed successfully")
#                                 completed_tasks.add(task_id)
#                             else:
#                                 # None result means retry is needed
#                                 handle_retry(task_id, request_obj, "request returned None")
#                         except asyncio.TimeoutError:
#                             logger.warning(f"Request #{task_id} timed out")
#                             handle_retry(task_id, request_obj, "timeout error")
#                         except aiohttp.ClientConnectionError as e:
#                             logger.warning(f"Request #{task_id} connection error: {str(e)}")
#                             handle_retry(task_id, request_obj, f"connection error: {str(e)}")
#                         except ConnectionError as e:
#                             logger.warning(f"Request #{task_id} connection error: {str(e)}")
#                             handle_retry(task_id, request_obj, f"connection error: {str(e)}")
#                         except Exception as e:
#                             logger.error(f"Error in request #{task_id}: {str(e)}")
#                             handle_retry(task_id, request_obj, f"error: {str(e)}")
#                     else:
#                         logger.warning(f"Request #{task_id} was cancelled")
#                         handle_retry(task_id, request_obj, "task cancelled")
#                 except Exception as e:
#                     logger.error(f"Error in callback for task #{task_id}: {e}")
#                     handle_retry(task_id, request_obj, f"callback error: {str(e)}")
#                 finally:
#                     # Always clean up task tracking
#                     active_tasks.pop(task_id, None)
#
#                     # Update status tracker
#                     status_tracker.num_tasks_in_progress = len(active_tasks) + queue_of_requests_to_retry.qsize() + len(
#                         retry_scheduled)
#
#             return callback
#
#         # Helper function to handle retry logic
#         def handle_retry(task_id, req_obj, error_msg):
#             # Make sure we're not retrying a task that's already scheduled
#             if task_id in retry_scheduled:
#                 logger.debug(f"Request #{task_id} already has a retry scheduled, skipping additional retry")
#                 return
#
#             if req_obj.attempts_left > 0:
#                 # Calculate exponential backoff delay
#                 backoff_delay = min(30, 2 ** (req_obj.max_attempts - req_obj.attempts_left))
#
#                 # Schedule the retry with exponential backoff
#                 retry_future = asyncio.ensure_future(schedule_retry(task_id, req_obj, backoff_delay))
#                 retry_scheduled[task_id] = retry_future
#             else:
#                 # No more retries left, mark as failed
#                 logger.warning(f"Request #{task_id} failed after all retries")
#                 error_result = (req_obj.request_json, {"error": f"Max retries reached: {error_msg}"})
#                 results.append(error_result)
#                 completed_tasks.add(task_id)
#
#                 # Update failure counter
#                 status_tracker.num_tasks_failed += 1
#
#         # Main processing loop
#         while True:
#             # Log progress periodically
#             current_time = time.time()
#             if current_time - last_progress_time > progress_interval:
#                 logger.info(f"Progress: {len(completed_tasks)}/{total_requests} requests completed, "
#                             f"{len(active_tasks)} active, "
#                             f"{queue_of_requests_to_retry.qsize()} in retry queue, "
#                             f"{len(retry_scheduled)} retries scheduled")
#                 last_progress_time = current_time
#
#             # Check if we're done - all requests have been processed or failed
#             if len(completed_tasks) >= total_requests:
#                 logger.info(f"All {total_requests} requests processed. Exiting loop.")
#                 break
#
#             # Check if there's nothing more to do but some tasks aren't completed
#             # (This would indicate that we've lost track of some tasks)
#             all_requests_started = current_index >= total_requests
#             no_active_tasks = len(active_tasks) == 0
#             no_retries_pending = queue_of_requests_to_retry.empty() and len(retry_scheduled) == 0
#
#             if all_requests_started and no_active_tasks and no_retries_pending and len(
#                     completed_tasks) < total_requests:
#                 logger.warning(
#                     f"No more active tasks or retries but only {len(completed_tasks)}/{total_requests} completed. Check for lost tasks.")
#
#                 # Add failsafe - consider missing tasks as failed
#                 for i in range(total_requests):
#                     task_id = i
#                     if task_id not in completed_tasks:
#                         logger.error(f"Request #{task_id} was lost during processing. Marking as failed.")
#                         error_result = ({"lost_request": True}, {"error": "Request was lost during processing"})
#                         results.append(error_result)
#                         completed_tasks.add(task_id)
#
#                 logger.info("All lost tasks accounted for. Exiting loop.")
#                 break
#
#             # Check if we have all tasks started, nothing active, and no retries
#             if all_requests_started and no_active_tasks and no_retries_pending:
#                 logger.info("All tasks either completed or failed. Exiting loop.")
#                 break
#
#             # Get the next request if we're under the concurrency limit
#             next_request = None
#             if len(active_tasks) < max_concurrent_requests:
#                 if not queue_of_requests_to_retry.empty():
#                     next_request = await queue_of_requests_to_retry.get()
#                     logger.debug(f"Got request #{next_request.task_id} from retry queue")
#                 elif current_index < total_requests:
#                     # Create new request
#                     request_json = requests_list[current_index]
#                     request_obj = APIRequest(
#                         task_id=next(task_id_gen),
#                         request_json=request_json,
#                         token_consumption=num_tokens_consumed_from_request(
#                             request_json, api_endpoint, token_encoding_name
#                         ),
#                         attempts_left=max_attempts,
#                         max_attempts=max_attempts
#                     )
#                     next_request = request_obj
#                     current_index += 1
#                     logger.debug(f"Starting new request #{next_request.task_id}")
#
#             # Update available capacity
#             now = time.time()
#             elapsed = now - last_update_time
#             available_request_capacity = min(
#                 max_requests_per_minute,
#                 available_request_capacity + max_requests_per_minute * elapsed / 60.0
#             )
#             available_token_capacity = min(
#                 max_tokens_per_minute,
#                 available_token_capacity + max_tokens_per_minute * elapsed / 60.0
#             )
#             last_update_time = now
#
#             # Process next request if we have capacity
#             if next_request:
#                 needed_tokens = next_request.token_consumption
#                 if available_request_capacity >= 1 and available_token_capacity >= needed_tokens:
#                     # Consume capacity
#                     available_request_capacity -= 1
#                     available_token_capacity -= needed_tokens
#                     next_request.attempts_left -= 1
#                     status_tracker.num_tasks_started += 1
#                     status_tracker.num_tasks_in_progress += 1
#
#                     # Create execution wrapper
#                     async def execute_request(req):
#                         try:
#                             # Actually make the API call
#                             result = await req.call_api(
#                                 session=session,
#                                 request_url=request_url,
#                                 request_header=request_header,
#                                 retry_queue=queue_of_requests_to_retry,
#                                 status_tracker=status_tracker,
#                             )
#                             return result
#                         except Exception as e:
#                             logger.error(f"Unhandled error in execute_request for task #{req.task_id}: {e}")
#                             # Return None to signal retry needed
#                             return None
#
#                     # Create task with timeout handling
#                     task_id = next_request.task_id
#                     if request_timeout:
#                         task = asyncio.create_task(
#                             asyncio.wait_for(
#                                 execute_request(next_request),
#                                 timeout=request_timeout
#                             )
#                         )
#                     else:
#                         # No timeout
#                         task = asyncio.create_task(execute_request(next_request))
#
#                     # Store task in tracking dictionary
#                     active_tasks[task_id] = task
#
#                     # Add done callback
#                     task.add_done_callback(make_callback(task_id, next_request))
#                 else:
#                     # Not enough capacity - put back in queue
#                     logger.debug(f"Not enough capacity for request #{next_request.task_id}, returning to queue")
#                     await queue_of_requests_to_retry.put(next_request)
#
#             # Clean up any completed tasks from active_tasks
#             tasks_to_remove = []
#             for task_id, task in active_tasks.items():
#                 if task.done():
#                     tasks_to_remove.append(task_id)
#
#             for task_id in tasks_to_remove:
#                 try:
#                     active_tasks.pop(task_id, None)
#                     logger.debug(f"Removed completed task #{task_id} from active_tasks")
#                 except Exception as e:
#                     logger.error(f"Error removing task #{task_id}: {e}")
#
#             # Rate limit error cooldown
#             time_since_rate_limit = time.time() - status_tracker.time_of_last_rate_limit_error
#             if time_since_rate_limit < seconds_to_pause_after_rate_limit_error:
#                 to_sleep = seconds_to_pause_after_rate_limit_error - time_since_rate_limit
#                 logger.warning(f"Pausing {to_sleep:.1f} seconds due to rate limit error")
#                 await asyncio.sleep(to_sleep)
#
#             # Brief sleep to allow other tasks to run
#             await asyncio.sleep(seconds_to_sleep_each_loop)
#
#     # Calculate elapsed time
#     process_elapsed = time.time() - process_start_time
#
#     # Log completion statistics
#     logger.info(f"Processing completed in {process_elapsed:.2f} seconds")
#     logger.info(f"Processed {len(completed_tasks)}/{total_requests} requests")
#     logger.info(f"Successful: {status_tracker.num_tasks_succeeded}, Failed: {status_tracker.num_tasks_failed}")
#
#     return results
#
#
# def setup_logging(level, log_file=None):
#     """Configure logging to output to both console and file if specified."""
#     log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#
#     # Reset previous handlers to avoid duplicate logs
#     root_logger = logging.getLogger()
#     for handler in root_logger.handlers[:]:
#         root_logger.removeHandler(handler)
#
#     handlers = [logging.StreamHandler()]
#
#     if log_file:
#         # Ensure directory exists
#         log_dir = os.path.dirname(log_file)
#         if log_dir and not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#
#         file_handler = logging.FileHandler(log_file, mode='a')
#         file_handler.setFormatter(logging.Formatter(log_format))
#         handlers.append(file_handler)
#
#     # Configure root logger
#     logging.basicConfig(
#         level=level,
#         format=log_format,
#         handlers=handlers
#     )
#
#     # Create and return a named logger
#     logger = logging.getLogger(__name__)
#     logger.info(f"Logging initialized at level {logging.getLevelName(level)}" +
#                 (f", writing to {log_file}" if log_file else ""))
#
#     return logger
#
#
# def process_in_memory(
#         dataset,
#         request_url: str,
#         max_requests_per_minute: float = 1500,
#         max_tokens_per_minute: float = 125000,
#         token_encoding_name: str = "cl100k_base",
#         max_attempts: int = 5,
#         api_key: str = '',
#         logging_level: int = logging.INFO,
#         request_timeout: int = 300,  # Individual request timeout (5 minutes or None for no limit)
#         overall_timeout: int = 3600,  # Overall process timeout (1 hour or None for no limit)
#         max_concurrent_requests: int = 100,  # Maximum concurrent requests
#         log_file: str = None,  # Log file path for debug logging
#         wait_for_all: bool = True  # Wait for all requests to complete
# ):
#     """
#     Process API requests from a dataset with configurable concurrency and timeouts.
#     """
#     # Configure logging
#     logger = setup_logging(logging_level, log_file)
#
#     # Extract requests from dataset
#     requests_list = dataset['job'].tolist()
#     logger.info(f"Starting to process {len(requests_list)} requests with {max_concurrent_requests} concurrent limit")
#
#     # Define async wrapper function with overall timeout
#     async def run_with_timeout():
#         start_time = time.time()
#         # 여기서 overall_timeout 파라미터 제거 - 호환성 문제 해결
#         result = await process_api_requests_in_memory(
#             requests_list=requests_list,
#             request_url=request_url,
#             max_requests_per_minute=max_requests_per_minute,
#             max_tokens_per_minute=max_tokens_per_minute,
#             token_encoding_name=token_encoding_name,
#             max_attempts=max_attempts,
#             api_key=api_key,
#             request_timeout=request_timeout,
#             max_concurrent_requests=max_concurrent_requests
#         )
#         elapsed = time.time() - start_time
#         logger.info(f"Processing completed in {elapsed:.2f} seconds")
#         return result
#
#     # 전체 타임아웃 처리 로직
#     start_time = time.time()
#
#     def is_overall_timeout_exceeded():
#         if overall_timeout is None:
#             return False
#         return (time.time() - start_time) > overall_timeout
#
#     # Run with overall timeout
#     try:
#         if overall_timeout is not None and wait_for_all:
#             logger.info(f"Running with overall timeout of {overall_timeout}s")
#             responses = asyncio.run(asyncio.wait_for(run_with_timeout(), overall_timeout))
#         else:
#             if wait_for_all:
#                 logger.info("Running without overall timeout until all requests complete")
#             else:
#                 logger.info(f"Running with internal timeout control of {overall_timeout}s")
#             responses = asyncio.run(run_with_timeout())
#     except asyncio.TimeoutError:
#         logger.error(f"Overall operation timed out after {overall_timeout}s")
#         # Only return partial results if wait_for_all is False
#         if not wait_for_all:
#             responses = []
#             logger.warning("Returning partial or no results due to timeout")
#         else:
#             # Re-raise to terminate
#             raise
#     except Exception as e:
#         logger.error(f"Error in overall process: {e}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         responses = []
#
#     # Process results
#     num_responses = len(responses)
#     if num_responses == 0:
#         logger.warning("No responses received!")
#
#     logger.info(f"Processing {num_responses} responses")
#     reasonings = []
#     descriptions = []
#     gen_sqls = []
#
#     # Create a progress counter
#     progress_step = max(1, num_responses // 10)  # Log every 10% of responses
#
#     for i, response in enumerate(responses):
#         # Log progress periodically
#         if i % progress_step == 0 or i == num_responses - 1:
#             logger.info(f"Processing response {i + 1}/{num_responses} ({(i + 1) / num_responses * 100:.1f}% complete)")
#
#         try:
#             # Extract response content
#             if len(response) >= 2 and isinstance(response[1], dict):
#                 # Handle error responses
#                 if "error" in response[1]:
#                     error_info = response[1]["error"]
#                     error_str = str(error_info)
#                     if len(error_str) > 100:
#                         error_str = error_str[:100] + "..."
#
#                     logger.warning(f"Response {i + 1} contains error: {error_str}")
#                     reasonings.append('')
#                     descriptions.append(f"Error: {error_str}")
#                     gen_sqls.append('')
#                     continue
#
#                 # Process successful responses
#                 if 'message' in response[1] and 'content' in response[1]['message']:
#                     content = response[1]['message']['content']
#
#                     try:
#                         json_data = json.loads(content)
#                         logger.debug(f"Successfully parsed JSON response {i + 1}")
#
#                         # Extract fields with fallbacks
#                         reasoning = json_data.get('reasoning', '')
#                         description = json_data.get('description', '')
#                         gen_sql = json_data.get('gen_sql', '')
#
#                         reasonings.append(reasoning)
#                         descriptions.append(description)
#                         gen_sqls.append(gen_sql)
#                     except json.JSONDecodeError as e:
#                         logger.error(f"Failed to parse JSON in response {i + 1}: {str(e)}")
#                         content_for_error = content[:100] + "..." if len(content) > 100 else content
#                         logger.debug(f"Failed JSON content: {content_for_error}")
#
#                         reasonings.append('')
#                         descriptions.append(f'JSON parse error')
#                         gen_sqls.append('')
#                 else:
#                     logger.error(f"Unexpected response format for response {i + 1}")
#                     reasonings.append('')
#                     descriptions.append('Unexpected response format')
#                     gen_sqls.append('')
#             else:
#                 logger.error(f"Invalid response structure for response {i + 1}")
#                 reasonings.append('')
#                 descriptions.append('Invalid response')
#                 gen_sqls.append('')
#         except Exception as e:
#             logger.error(f"Error processing response {i + 1}: {e}")
#             reasonings.append('')
#             descriptions.append(f'Processing error')
#             gen_sqls.append('')
#
#     # Ensure all lists match the dataset length
#     expected_length = len(dataset)
#     logger.info(f"Filling missing responses to match dataset length ({expected_length})")
#
#     # Calculate how many responses were missing
#     missing_count = expected_length - len(reasonings)
#     if missing_count > 0:
#         logger.warning(
#             f"Missing {missing_count} out of {expected_length} responses ({missing_count / expected_length * 100:.1f}%)")
#
#     # Fill missing values
#     reasonings = reasonings + [''] * (expected_length - len(reasonings))
#     descriptions = descriptions + [''] * (expected_length - len(descriptions))
#     gen_sqls = gen_sqls + [''] * (expected_length - len(gen_sqls))
#
#     # Trim if we somehow have too many
#     if len(reasonings) > expected_length:
#         logger.warning(f"Trimming excess responses: {len(reasonings)} received, {expected_length} expected")
#         reasonings = reasonings[:expected_length]
#         descriptions = descriptions[:expected_length]
#         gen_sqls = gen_sqls[:expected_length]
#
#     # Update the dataset
#     dataset['reasoning'] = reasonings
#     dataset['description'] = descriptions
#     dataset['gen_sql'] = gen_sqls
#
#     logger.info(f"Processing complete. {len(responses)} responses processed for {expected_length} requests")
#     return dataset
#
#
# async def llm_invoke_batch_langchain(model_name, prompts, template, model_url="http://172.16.15.112:11434",
#                                      batch_size=10, max_retries=3, max_concurrent=10):
#     """프롬프트 배치에 대한 병렬 LLM 호출 (LangChain 사용, 진행률 로깅 기능 추가)"""
#     logging.info(f"LangChain 사용하여 {len(prompts)}개 프롬프트 처리 시작 (모델: {model_name}, 배치크기: {batch_size})")
#
#     # 결과 저장용 리스트 초기화 (순서 보존)
#     all_results = [None] * len(prompts)
#     total_prompts = len(prompts)
#
#     # 배치 수 계산
#     total_batches = (total_prompts + batch_size - 1) // batch_size
#
#     # 진행률 추적 객체 생성
#     progress_tracker = ProgressTracker(total_prompts, batch_size)
#
#     # LangChain 모델 및 체인 준비 (한 번만 생성)
#     logging.info(f"LangChain 모델 초기화 중 (URL: {model_url})")
#
#     model = OllamaLLM(
#         model=model_name,
#         temperature=0.0,
#         base_url=model_url
#     )
#
#     prompt_template = PromptTemplate(
#         template=template,
#         input_variables=["request", "ddl", "sql"],
#         partial_variables={"format_instructions": sql_parser.get_format_instructions()},
#     )
#
#     chain = prompt_template | model
#
#     # 동시 연결 제한을 위한 세마포어
#     semaphore = asyncio.Semaphore(max_concurrent)
#
#     # LLM에 단일 프롬프트 전송하는 함수
#     async def llm_invoke_single(prompt, task_id):
#         async with semaphore:
#             try:
#                 # 진행 상태 업데이트 - "시작" 상태로 변경
#                 progress_tracker.update_task_progress(task_id, "start")
#
#                 start_time = time.time()
#
#                 # LangChain을 사용하여 비동기 호출
#                 response = await chain.ainvoke({
#                     'ddl': prompt.get('context', ''),
#                     'request': prompt.get('question', ''),
#                     'sql': ''
#                 })
#
#                 elapsed = time.time() - start_time
#
#                 # 응답 결과 구성
#                 result = {
#                     "task_id": task_id,
#                     "response": response,
#                     "prompt": prompt
#                 }
#
#                 # 진행 상태 업데이트 - "성공" 상태로 변경
#                 progress_tracker.update_task_progress(task_id, "success", elapsed=elapsed)
#
#                 return result
#
#             except Exception as e:
#                 # 오류 처리
#                 error_msg = str(e)
#                 # 진행 상태 업데이트 - "실패" 상태로 변경
#                 progress_tracker.update_task_progress(task_id, "failed", error=error_msg)
#                 logging.error(f"Task {task_id} failed: {error_msg}")
#                 return {
#                     "task_id": task_id,
#                     "error": error_msg,
#                     "prompt": prompt
#                 }
#
#     # 실패한 작업 재시도 함수
#     async def retry_failed_tasks(retry_tasks):
#         """실패한 작업 재시도"""
#         retry_round = 0
#         while retry_tasks and retry_round < max_retries:
#             logging.info(f"재시도 라운드 {retry_round + 1}/{max_retries}, {len(retry_tasks)} 작업 재시도 중")
#
#             # 재시도 작업 비동기 실행
#             retry_coroutines = []
#             for task_id, prompt, retry_count in retry_tasks:
#                 # 재시도 상태 업데이트
#                 progress_tracker.update_task_progress(task_id, "retry", error=retry_count + 1)
#                 retry_coroutines.append(llm_invoke_single(prompt, task_id))
#
#             retry_results = await asyncio.gather(*retry_coroutines, return_exceptions=True)
#
#             # 다음 라운드 재시도 작업 목록 초기화
#             next_retry_tasks = []
#
#             # 결과 처리
#             for (task_id, prompt, retry_count), result in zip(retry_tasks, retry_results):
#                 if isinstance(result, Exception):
#                     logging.error(f"재시도 중 예외 발생 (작업 {task_id}): {str(result)}")
#                     if retry_round < max_retries - 1:
#                         next_retry_tasks.append((task_id, prompt, retry_count + 1))
#                     continue
#
#                 if "error" in result and retry_round < max_retries - 1:
#                     logging.warning(f"재시도 실패 (작업 {task_id}): {result.get('error', 'Unknown error')}")
#                     next_retry_tasks.append((task_id, prompt, retry_count + 1))
#                 else:
#                     # 결과 저장
#                     logging.info(f"재시도 성공 (작업 {task_id})")
#                     all_results[task_id] = result
#
#             # 재시도 목록 업데이트
#             retry_tasks = next_retry_tasks
#
#             # 다음 라운드로 진행
#             retry_round += 1
#
#             # 짧은 대기 후 다음 재시도 라운드
#             if retry_tasks and retry_round < max_retries:
#                 await asyncio.sleep(1)
#
#     # 메인 처리 로직
#     for batch_idx in range(0, total_prompts, batch_size):
#         batch_end = min(batch_idx + batch_size, total_prompts)
#         batch = prompts[batch_idx:batch_end]
#
#         # 배치 시작 로깅
#         batch_num = batch_idx // batch_size + 1
#         logging.info(f"배치 {batch_num}/{total_batches} 처리 시작 ({len(batch)} 프롬프트)")
#         progress_tracker.update_batch_progress(len(batch), batch_idx // batch_size, total_batches)
#
#         # 비동기 작업 생성
#         tasks = []
#         for i, prompt in enumerate(batch):
#             task_id = batch_idx + i
#             task = asyncio.create_task(llm_invoke_single(prompt, task_id))
#             tasks.append(task)
#
#         # 모든 작업 완료 대기
#         batch_results = await asyncio.gather(*tasks, return_exceptions=True)
#
#         # 재시도가 필요한 작업 처리
#         retry_tasks = []
#         success_count = 0
#
#         for result in batch_results:
#             if isinstance(result, Exception):
#                 logging.error(f"작업 예외 발생: {str(result)}")
#                 continue
#
#             task_id = result.get("task_id")
#             if task_id is None:
#                 logging.error(f"태스크 ID가 없는 결과: {result}")
#                 continue
#
#             # 오류 발생 여부 확인 및 재시도 결정
#             if "error" in result and max_retries > 0:
#                 retry_tasks.append((task_id, prompts[task_id], 0))  # (task_id, prompt, retry_count)
#             else:
#                 # 결과 저장 (원래 순서대로)
#                 all_results[task_id] = result
#                 success_count += 1
#
#         # 실패한 작업 재시도
#         if retry_tasks and max_retries > 0:
#             logging.info(f"배치 {batch_num}에서 {len(retry_tasks)}개 작업 재시도 필요")
#             await retry_failed_tasks(retry_tasks)
#
#         # 배치 완료 로깅
#         logging.info(f"배치 {batch_num}/{total_batches} 처리 완료 (성공: {success_count}/{len(batch)})")
#         progress_tracker.update_batch_completion(batch_results)
#
#         # 서버 부하 방지를 위한 짧은 대기
#         if batch_end < total_prompts:
#             await asyncio.sleep(0.5)
#
#     # 진행률 표시 종료
#     progress_tracker.close()
#
#     # 전체 통계
#     successful_results = sum(1 for r in all_results if r and "error" not in r)
#     logging.info(f"전체 처리 완료: {successful_results}/{total_prompts} 성공")
#
#     return all_results
#
#
# def llm_invoke_parallel_langchain(model, prompts, template, batch_size=10, max_retries=3, max_concurrent=10):
#     """병렬 처리를 위한 래퍼 함수 (로깅 기능 추가)"""
#     logging.info(f"병렬 처리 시작: 총 {len(prompts)}개 요청 (배치 크기: {batch_size}, 최대 동시 요청: {max_concurrent})")
#
#     start_time = time.time()
#
#     loop = asyncio.get_event_loop()
#     results = loop.run_until_complete(
#         llm_invoke_batch_langchain(
#             model,
#             prompts,
#             template,
#             batch_size=batch_size,
#             max_retries=max_retries,
#             max_concurrent=max_concurrent
#         )
#     )
#
#     elapsed_time = time.time() - start_time
#
#     logging.info(f"병렬 처리 완료: {len(results)}개 응답 수신")
#     logging.info(f"평균 요청 처리 시간: {elapsed_time / len(prompts)}초")
#
#     return results
#
#
# if __name__ == "__main__":
#     # parse command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--requests_filepath")
#     parser.add_argument("--save_filepath", default=None)
#     parser.add_argument("--request_url", default="https://api.openai.com/v1/chat/completions")
#     parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
#     parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
#     parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
#     parser.add_argument("--token_encoding_name", default="cl100k_base")
#     parser.add_argument("--max_attempts", type=int, default=5)
#     parser.add_argument("--logging_level", default=logging.INFO)
#     args = parser.parse_args()
#
#     if args.save_filepath is None:
#         args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")
#
#     # run script
#     # asyncio.run(
#     #     process_api_requests_from_file(
#     #         requests_filepath=args.requests_filepath,
#     #         save_filepath=args.save_filepath,
#     #         request_url=args.request_url,
#     #         api_key=args.api_key,
#     #         max_requests_per_minute=float(args.max_requests_per_minute),
#     #         max_tokens_per_minute=float(args.max_tokens_per_minute),
#     #         token_encoding_name=args.token_encoding_name,
#     #         max_attempts=int(args.max_attempts),
#     #         logging_level=int(args.logging_level),
#     #     )
#     # )
#     process_by_file(
#         requests_filepath=args.requests_filepath,
#         save_filepath=args.save_filepath,
#         request_url=args.request_url,
#         api_key=args.api_key,
#         max_requests_per_minute=float(args.max_requests_per_minute),
#         max_tokens_per_minute=float(args.max_tokens_per_minute),
#         token_encoding_name=args.token_encoding_name,
#         max_attempts=int(args.max_attempts),
#         logging_level=int(args.logging_level),
#     )
