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
            timeout: int = None,
            progress_tracker: ProgressTracker = None,
            response_processor=None,  # 추가: 응답 처리 함수
    ):
        """Calls the API and handles various error conditions."""
        logger = logging.getLogger(__name__)

        # 요청 시작 시 상태 업데이트
        if progress_tracker:
            progress_tracker.update_task_progress(self.task_id, "start")

        start_time = time.time()

        error = None
        result_data = None
        processed_result = None  # 처리된 결과를 저장할 변수

        try:
            # API 호출
            try:
                async with session.post(
                        url=request_url,
                        headers=request_header,
                        json=self.request_json,
                        timeout=timeout,
                        raise_for_status=False
                ) as http_response:
                    # HTTP 에러 상태 코드 확인
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

                    # 응답 처리 (즉시 처리)
                    if response_processor and "error" not in response_json:
                        try:
                            processed_result = response_processor(response_json, self.metadata)
                            # logging.info(f"processed_result: [{type(processed_result)}] {processed_result}")
                        except Exception as process_error:
                            logger.error(f"응답 처리 중 오류 발생 (요청 #{self.task_id}): {str(process_error)}")
                            # 처리 오류도 재시도 요인으로 간주
                            raise ValueError(f"응답 처리 오류: {str(process_error)}")

                    # 성공 시 상태 업데이트
                    if progress_tracker:
                        elapsed = time.time() - start_time
                        progress_tracker.update_task_progress(self.task_id, "success", elapsed=elapsed)

            except aiohttp.ClientConnectionError as e:
                logger.warning(f"요청 #{self.task_id} 연결 오류: {str(e)}")
                raise ConnectionError(f"서버 연결 오류: {str(e)}")
            except aiohttp.ClientResponseError as e:
                logger.warning(f"요청 #{self.task_id} HTTP 오류: {e.status} - {e.message}")
                raise ConnectionError(f"HTTP 오류 {e.status}: {e.message}")
            except asyncio.TimeoutError:
                logger.warning(f"요청 #{self.task_id} 시간 초과")
                # 타임아웃 발생 시 재시도 처리
                if self.attempts_left > 0:
                    self.attempts_left -= 1
                    await retry_queue.put(self)
                    logger.info(f"요청 #{self.task_id} 재시도 큐에 추가됨 (남은 시도: {self.attempts_left})")
                    return None
                else:
                    # 재시도 횟수 소진
                    error_data = [self.request_json, {"error": "모든 재시도 후 요청 시간 초과"}, self.metadata]
                    if save_filepath:
                        append_to_jsonl(error_data, save_filepath)
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1
                    return error_data
            except ValueError as e:
                # 응답 처리 오류 처리
                logger.error(f"요청 #{self.task_id} 응답 처리 오류: {str(e)}")
                if self.attempts_left > 0:
                    self.attempts_left -= 1
                    await retry_queue.put(self)
                    logger.info(f"요청 #{self.task_id} 재시도 큐에 추가됨 (남은 시도: {self.attempts_left}, 응답 처리 오류)")
                    return None
                else:
                    error = e
            except Exception as e:
                logger.error(f"요청 #{self.task_id} HTTP 요청 중 예상치 못한 오류: {str(e)}")
                if progress_tracker:
                    elapsed = time.time() - start_time
                    progress_tracker.update_task_progress(self.task_id, "failed", elapsed=elapsed, error=str(e))
                raise

            # API 응답 처리
            if "error" in response_json:
                logger.warning(f"요청 #{self.task_id} API 오류 발생: {response_json['error']}")
                status_tracker.num_api_errors += 1
                error = response_json

                # 속도 제한 오류 확인
                error_message = response_json["error"].get("message", "")
                if "Rate limit" in error_message or "rate_limit" in error_message.lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
                    logger.warning(f"요청 #{self.task_id}에 대한 속도 제한 오류: {error_message}")
                    raise ConnectionError(f"속도 제한 오류: {error_message}")
            else:
                # 성공한 응답
                # 중요: 여기서 저장할 데이터에 processed_result가 있으면 그것을 포함
                if self.metadata:
                    if processed_result is not None:
                        result_data = [self.request_json, processed_result, self.metadata]
                    else:
                        result_data = [self.request_json, response_json, self.metadata]
                else:
                    if processed_result is not None:
                        result_data = [self.request_json, processed_result]
                    else:
                        result_data = [self.request_json, response_json]

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.warning(f"요청 #{self.task_id} 오류 발생: {str(e)}")
            status_tracker.num_other_errors += 1
            error = e

            # 오류 내용을 결과 리스트에 저장
            self.result.append(str(e))

            # 재시도 가능 여부 확인
            if self.attempts_left > 0:
                return None

        except Exception as e:
            logger.error(f"요청 #{self.task_id} 예상치 못한 예외 발생: {str(e)}")
            status_tracker.num_other_errors += 1
            error = e
            self.result.append(str(e))

        # 오류 처리 및 결과 준비
        if error:
            if self.attempts_left > 0:
                # 재시도 신호를 반환
                return None
            else:
                # 더 이상 재시도 불가, 오류 결과 생성
                logger.error(f"요청 #{self.task_id} 모든 시도 후 실패. 최종 오류: {error}")
                data = (
                    [self.request_json, {"error": [str(e) for e in self.result]}, self.metadata]
                    if self.metadata
                    else [self.request_json, {"error": [str(e) for e in self.result]}]
                )
                result_data = data

                # 파일에 저장 (필요한 경우)
                if save_filepath is not None:
                    append_to_jsonl(data, save_filepath)

                # 상태 업데이트
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            # 성공한 결과
            data = result_data
            if save_filepath is not None:
                # logging.info(f'append_to_jsonl: {data}')
                append_to_jsonl(data, save_filepath)
                logger.debug(f"요청 #{self.task_id} 결과가 {save_filepath}에 저장됨")

            # 상태 업데이트
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

        # 처리된 결과가 있으면 그것을 반환, 없으면 원본 결과 반환
        return processed_result if processed_result is not None else result_data


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
        max_concurrent_requests: int = 10,  # 동시 요청 수 제한
        progress_tracker: ProgressTracker = None,  # 진행률 추적기
        response_processor=None,  # 응답 처리 함수 추가
        request_timeout=300
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

    # 총 요청 수와 요청 ID 추적을 위한 추가 변수
    expected_ids = set()  # 추가: 예상되는 모든 요청 ID를 추적하는 세트
    total_requests = 0
    with open(requests_filepath) as file:
        for _ in file:
            expected_ids.add(total_requests)  # 요청 ID 추적 세트에 추가
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
    processed_ids = set()  # 추가: 처리된 요청 ID를 추적하는 세트

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

    # 배치 크기가 0인 경우를 방지
    progress_tracker.batch_size = max(1, progress_tracker.batch_size)

    # 그 후 total_batches 계산
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
                    response_processor=response_processor,  # 응답 처리 함수 전달
                    timeout=request_timeout
                )

                # 성공 시 상태 업데이트
                if result is not None:  # None이 아니면 성공 (None은 재시도 필요)
                    elapsed = time.time() - start_time
                    progress_tracker.update_task_progress(request_obj.task_id, "success", elapsed=elapsed)
                    completed_tasks.add(request_obj.task_id)
                    processed_ids.add(request_obj.task_id)  # 요청 ID 추적 세트에 추가
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
                    processed_ids.add(request_obj.task_id)  # 추가: 요청 ID 추적 세트에 추가

                    # 실패 결과를 파일에 저장
                    error_data = [request_obj.request_json, {"error": str(e)},
                                  request_obj.metadata] if request_obj.metadata else [request_obj.request_json,
                                                                                      {"error": str(e)}]
                    append_to_jsonl(error_data, save_filepath)

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

                    # 추가: 예상된 모든 ID가 처리되었는지 확인
                    missing_ids = expected_ids - processed_ids
                    if missing_ids:
                        logging.warning(f"발견된 {len(missing_ids)}개의 미처리 요청: {missing_ids}")

                        # 미처리 요청에 대한 오류 기록
                        for missing_id in missing_ids:
                            error_data = [
                                {"task_id": missing_id},  # 원본 요청 정보가 없으므로 최소한의 정보만 포함
                                {"error": "요청이 처리되지 않음 (유실)"},
                                None
                            ]
                            append_to_jsonl(error_data, save_filepath)
                            processed_ids.add(missing_id)

                            # 상태 업데이트
                            status_tracker.num_tasks_failed += 1

                        logging.info(f"유실된 {len(missing_ids)}개 요청에 대한 오류 기록 완료")

                    logging.debug("All tasks completed - process terminating")
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

                    # 추가: 처리되지 않은 요청 확인
                    missing_ids = expected_ids - processed_ids
                    if missing_ids:
                        logging.warning(f"발견된 {len(missing_ids)}개의 미처리 요청: {missing_ids}")

                        # 미처리 요청에 대한 오류 기록
                        for missing_id in missing_ids:
                            error_data = [
                                {"task_id": missing_id},
                                {"error": "요청이 처리되지 않음 (유실)"},
                                None
                            ]
                            append_to_jsonl(error_data, save_filepath)

                            # 상태 업데이트
                            status_tracker.num_tasks_failed += 1

                        logging.info(f"유실된 {len(missing_ids)}개 요청에 대한 오류 기록 완료")

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

        # 최종 확인: 모든 요청이 처리되었는지 확인
        missing_ids = expected_ids - processed_ids
        if missing_ids:
            logging.warning(f"최종 확인: {len(missing_ids)}개의 미처리 요청이 발견됨: {missing_ids}")

            # 미처리 요청에 대한 오류 기록
            for missing_id in missing_ids:
                error_data = [
                    {"task_id": missing_id},
                    {"error": "최종 확인에서 요청이 처리되지 않은 것으로 확인됨"},
                    None
                ]
                append_to_jsonl(error_data, save_filepath)

                # 상태 업데이트
                status_tracker.num_tasks_failed += 1

            logging.info(f"최종 확인: 유실된 {len(missing_ids)}개 요청에 대한 오류 기록 완료")

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
        batch_size: int = 20,  # 배치 크기 매개변수
        response_processor=None,  # 응답 처리 함수 추가
        prefix: str = ".",  # 로그 디렉토리 경로
        request_timeout: int = 300,  # 요청 타임아웃 (초)
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
        response_processor: Function to process API responses.
        prefix: Directory prefix for logs and other files.
    """
    # 결과 파일 경로 설정
    if save_filepath is None:
        save_filepath = requests_filepath.replace(".jsonl", "_results.jsonl")

    # 총 요청 수 계산
    total_requests = 0

    # 타임아웃 설정 (0 이하면 None으로 설정하여 무제한)
    timeout = None if request_timeout <= 0 else request_timeout

    try:
        with open(requests_filepath) as file:
            for _ in file:
                total_requests += 1
    except Exception as e:
        logging.error(f"Error counting requests in file: {e}")
        return

    logging.info(f"API URL: %s", request_url)
    logging.info(f"Processing {total_requests} requests from {requests_filepath}")
    logging.info(f"Results will be saved to {save_filepath}")
    logging.info(f"Rate limits: {max_requests_per_minute} requests/min, {max_tokens_per_minute} tokens/min")
    logging.info(f"Concurrent requests limit: {max_concurrent_requests}")

    # 배치 크기 조정 - 너무 작거나 크지 않도록
    # 배치 크기 조정 - 너무 작거나 크지 않도록 (0인 경우 포함)
    batch_size = min(max(5, batch_size), 100)  # 최소 5, 최대 100
    batch_size = max(1, min(batch_size, total_requests))  # 최소 1, 최대 total_requests

    # ProgressTracker 초기화
    progress_tracker = ProgressTracker(total_requests, batch_size)
    logging.debug(f"Using batch size of {batch_size} for progress tracking")

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
                response_processor=response_processor,  # 응답 처리 함수 전달
                request_timeout=timeout,  # 타임아웃 전달
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
    #         http://172.16.15.109:11434/api/chat
    #         http://127.0.0.1:11434/api/generate
    match = re.search(r"^http://[^/]+:\d+/api/(.+)$", request_url)
    if match:
        return match.group(1)

    # 4) 위 패턴에 해당하지 않으면 None
    return None


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
