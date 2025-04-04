import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Set, List, Tuple, Callable

import aiohttp
import tiktoken

from utils.tracking import ProgressTracker


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
    request_map: Dict[str, int] = field(default_factory=dict)  # request_id -> task_id 매핑
    task_map: Dict[int, str] = field(default_factory=dict)  # task_id -> request_id 매핑


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    max_attempts: int
    request_id: str  # 고유 요청 ID 추가
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
            response_processor=None,
    ):
        """Calls the API and handles various error conditions."""
        logger = logging.getLogger(__name__)

        # 요청 시작 시 로깅만 하고 진행률은 아직 업데이트하지 않음
        if progress_tracker:
            # 진행률 업데이트 주석 처리 - 응답 처리 후에 업데이트할 예정
            # progress_tracker.update_task_progress(self.task_id, "start", request_id=self.request_id)
            # 로깅만 수행
            logger.debug(f"태스크 #{self.task_id} (ID: {self.request_id}) 시작")

        start_time = time.time()

        error = None
        result_data = None
        processed_result = None  # 처리된 결과를 저장할 변수

        try:
            # API 호출
            try:
                # request_json에 request_id 추가
                request_with_id = self.request_json.copy() if isinstance(self.request_json, dict) else self.request_json

                # 메타데이터가 있는 경우에만 request_id 추가
                if isinstance(request_with_id, dict):
                    # 메타데이터가 없으면 생성
                    if 'metadata' not in request_with_id:
                        request_with_id['metadata'] = {}
                    # 메타데이터에 request_id 추가
                    request_with_id['metadata']['request_id'] = self.request_id

                async with session.post(
                        url=request_url,
                        headers=request_header,
                        json=request_with_id,  # request_id가 포함된 요청 사용
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

                    # 응답에 request_id 추가하여 추적성 확보
                    if isinstance(response_json, dict):
                        response_json['request_id'] = self.request_id

                    # 응답 처리 (즉시 처리)
                    if response_processor and "error" not in response_json:
                        try:
                            processed_result = response_processor(response_json, self.metadata)
                            # 처리된 결과에도 request_id 추가
                            if isinstance(processed_result, dict) and 'request_id' not in processed_result:
                                processed_result['request_id'] = self.request_id
                        except Exception as process_error:
                            logger.error(
                                f"응답 처리 중 오류 발생 (요청 #{self.task_id}, ID: {self.request_id}): {str(process_error)}")
                            # 처리 오류도 재시도 요인으로 간주
                            raise ValueError(f"응답 처리 오류: {str(process_error)}")

                    # 성공 시 상태 업데이트 - 여기서 진행률 업데이트
                    if progress_tracker:
                        elapsed = time.time() - start_time
                        progress_tracker.update_task_progress(self.task_id, "success", elapsed=elapsed,
                                                              request_id=self.request_id)
            # 연결 오류 처리
            except (aiohttp.ClientConnectionError, ConnectionError) as e:
                logging.warning(f"요청 #{self.task_id} (ID: {self.request_id}) 연결 오류: {str(e)}")

                # 재시도 처리
                if self.attempts_left > 0:
                    self.attempts_left -= 1
                    # 재시도 큐에 추가
                    await retry_queue.put(self)
                    logger.info(
                        f"요청 #{self.task_id} (ID: {self.request_id}) 연결 오류로 재시도 큐에 추가됨 (남은 시도: {self.attempts_left})")
                    return None
                else:
                    error_data = [self.request_json, {"error": f"모든 재시도 후 연결 오류: {str(e)}", "request_id": self.request_id},
                                  self.metadata]
                    if save_filepath:
                        append_to_jsonl(error_data, save_filepath)
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1

                    # 실패 시 진행률 업데이트
                    if progress_tracker:
                        elapsed = time.time() - start_time
                        progress_tracker.update_task_progress(self.task_id, "failed", elapsed=elapsed,
                                                              error=f"연결 오류 - 모든 재시도 실패: {str(e)}",
                                                              request_id=self.request_id)
                    return error_data
            # HTTP 오류 처리 부분
            except aiohttp.ClientResponseError as e:
                logger.warning(f"요청 #{self.task_id} (ID: {self.request_id}) HTTP 오류: {e.status} - {e.message}")

                # HTTP 500 오류는 서버 측 오류이므로 재시도
                if e.status >= 500:
                    if self.attempts_left > 0:
                        self.attempts_left -= 1
                        # 재시도 큐에 추가
                        await retry_queue.put(self)
                        logger.info(
                            f"요청 #{self.task_id} (ID: {self.request_id}) 서버 오류로 재시도 큐에 추가됨 (남은 시도: {self.attempts_left})")
                        return None

                # 그 외의 경우 일반 연결 오류로 처리
                raise ConnectionError(f"HTTP 오류 {e.status}: {e.message}")
            except asyncio.TimeoutError:
                logger.warning(f"요청 #{self.task_id} (ID: {self.request_id}) 시간 초과")
                # 타임아웃 발생 시 재시도 처리
                if self.attempts_left > 0:
                    self.attempts_left -= 1
                    await retry_queue.put(self)
                    logger.info(f"요청 #{self.task_id} (ID: {self.request_id}) 재시도 큐에 추가됨 (남은 시도: {self.attempts_left})")
                    return None
                else:
                    # 재시도 횟수 소진
                    error_data = [self.request_json, {"error": "모든 재시도 후 요청 시간 초과", "request_id": self.request_id},
                                  self.metadata]
                    if save_filepath:
                        append_to_jsonl(error_data, save_filepath)
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1

                    # 실패 시 진행률 업데이트
                    if progress_tracker:
                        elapsed = time.time() - start_time
                        progress_tracker.update_task_progress(self.task_id, "failed", elapsed=elapsed,
                                                              error="시간 초과", request_id=self.request_id)

                    return error_data
            except ValueError as e:
                # 응답 처리 오류 처리
                logger.error(f"요청 #{self.task_id} (ID: {self.request_id}) 응답 처리 오류: {str(e)}")
                if self.attempts_left > 0:
                    self.attempts_left -= 1
                    await retry_queue.put(self)
                    logger.info(
                        f"요청 #{self.task_id} (ID: {self.request_id}) 재시도 큐에 추가됨 (남은 시도: {self.attempts_left}, 응답 처리 오류)")
                    return None
                else:
                    error = e
            except Exception as e:
                logger.error(f"요청 #{self.task_id} (ID: {self.request_id}) HTTP 요청 중 예상치 못한 오류: {str(e)}")
                if progress_tracker:
                    elapsed = time.time() - start_time
                    progress_tracker.update_task_progress(self.task_id, "failed", elapsed=elapsed, error=str(e),
                                                          request_id=self.request_id)
                raise

            # API 응답 처리
            if "error" in response_json:
                logger.warning(f"요청 #{self.task_id} (ID: {self.request_id}) API 오류 발생: {response_json['error']}")
                status_tracker.num_api_errors += 1
                error = response_json

                # 속도 제한 오류 확인
                error_message = str(response_json["error"].get("message", ""))
                if "Rate limit" in error_message or "rate_limit" in error_message.lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
                    logger.warning(f"요청 #{self.task_id} (ID: {self.request_id})에 대한 속도 제한 오류: {error_message}")
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
            logger.warning(f"요청 #{self.task_id} (ID: {self.request_id}) 오류 발생: {str(e)}")
            status_tracker.num_other_errors += 1
            error = e

            # 오류 내용을 결과 리스트에 저장
            self.result.append(str(e))

            # 재시도 가능 여부 확인
            if self.attempts_left > 0:
                return None

        except Exception as e:
            logger.error(f"요청 #{self.task_id} (ID: {self.request_id}) 예상치 못한 예외 발생: {str(e)}")
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
                logger.error(f"요청 #{self.task_id} (ID: {self.request_id}) 모든 시도 후 실패. 최종 오류: {error}")
                error_result = {"error": [str(e) for e in self.result], "request_id": self.request_id}

                data = (
                    [self.request_json, error_result, self.metadata]
                    if self.metadata
                    else [self.request_json, error_result]
                )
                result_data = data

                # 파일에 저장 (필요한 경우)
                if save_filepath is not None:
                    append_to_jsonl(data, save_filepath)

                # 상태 업데이트
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1

                # 실패 시 진행률 업데이트
                if progress_tracker:
                    elapsed = time.time() - start_time
                    progress_tracker.update_task_progress(self.task_id, "failed", elapsed=elapsed,
                                                          error=str(error), request_id=self.request_id)
        else:
            # 성공한 결과
            data = result_data
            if save_filepath is not None:
                append_to_jsonl(data, save_filepath)
                logger.debug(f"요청 #{self.task_id} (ID: {self.request_id}) 결과가 {save_filepath}에 저장됨")

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
    original_requests = {}  # 원본 요청 정보 저장
    total_requests = 0
    with open(requests_filepath) as file:
        for line_num, line in enumerate(file):
            expected_ids.add(line_num)  # 요청 ID 추적 세트에 추가
            try:
                # 원본 요청 정보 저장
                original_requests[line_num] = json.loads(line)
            except json.JSONDecodeError:
                pass  # 파싱 실패시 무시
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
    processed_ids = set()  # 추가: 처리된 요청 ID를 추적하는 세트 (성공 또는 실패 모두 포함)

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

            # 요청 시작 시 로깅만 수행하고 진행률 업데이트는 나중에
            logging.debug(f"요청 #{request_obj.task_id} (ID: {request_obj.request_id}) 시작")

            try:
                result = await request_obj.call_api(
                    session=session,
                    request_url=request_url,
                    request_header=request_header,
                    retry_queue=queue_of_requests_to_retry,
                    save_filepath=save_filepath,
                    status_tracker=status_tracker,
                    response_processor=response_processor,  # 응답 처리 함수 전달
                    timeout=request_timeout,
                    progress_tracker=progress_tracker
                )

                # 성공 또는 최종 실패 시 처리됨으로 표시
                if result is not None:  # None이 아니면 성공 또는 최종 실패 (None은 재시도 필요)
                    # 진행률 업데이트는 call_api 내에서 이미 처리됨
                    completed_tasks.add(request_obj.task_id)
                    processed_ids.add(request_obj.task_id)  # 요청 ID 추적 세트에 추가

                    # 요청 ID 매핑 추가 (성공한 경우에만)
                    if isinstance(result, dict) and 'request_id' in result:
                        status_tracker.request_map[result['request_id']] = request_obj.task_id
                        status_tracker.task_map[request_obj.task_id] = result['request_id']
                return result
            except Exception as e:
                # 실패 시 상태 업데이트
                elapsed = time.time() - start_time
                if progress_tracker:
                    progress_tracker.update_task_progress(request_obj.task_id, "failed", elapsed=elapsed, error=str(e),
                                                          request_id=request_obj.request_id)

                # 재시도 가능한 경우
                if request_obj.attempts_left > 0:
                    logging.warning(
                        f"Error in request #{request_obj.task_id} (ID: {request_obj.request_id}): {str(e)}. Will retry.")
                    await queue_of_requests_to_retry.put(request_obj)
                else:
                    # 최종 실패
                    logging.error(
                        f"Request #{request_obj.task_id} (ID: {request_obj.request_id}) failed after all attempts: {str(e)}")
                    completed_tasks.add(request_obj.task_id)
                    processed_ids.add(request_obj.task_id)  # 요청 ID 추적 세트에 추가

                    # 실패 결과를 파일에 저장
                    error_data = [
                        request_obj.request_json,
                        {"error": str(e), "request_id": request_obj.request_id},
                        request_obj.metadata
                    ]
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
                            f"Retrying request #{next_request.task_id} (ID: {next_request.request_id}): {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            current_task_id = next(task_id_generator)

                            # 고유 요청 ID 생성
                            request_id = str(uuid.uuid4())

                            # "index" 키가 있는 경우에만 삭제
                            if "index" in request_json:
                                del request_json["index"]

                            # 요청 ID 매핑 추가
                            status_tracker.request_map[request_id] = current_task_id
                            status_tracker.task_map[current_task_id] = request_id

                            next_request = APIRequest(
                                task_id=current_task_id,
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                max_attempts=max_attempts,
                                request_id=request_id,  # UUID 할당
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request #{next_request.task_id} (ID: {next_request.request_id}): {next_request}"
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
                            # 원본 요청 정보가 있으면 사용, 없으면 최소 정보만 포함
                            request_data = original_requests.get(missing_id, {"task_id": missing_id})
                            error_data = [
                                request_data,
                                {"error": "요청이 처리되지 않음 (유실)", "request_id": f"missing_{missing_id}"},
                                None
                            ]
                            append_to_jsonl(error_data, save_filepath)
                            processed_ids.add(missing_id)  # 처리됨으로 표시

                            # 상태 업데이트
                            status_tracker.num_tasks_failed += 1

                            # 진행률 업데이트
                            if progress_tracker:
                                progress_tracker.update_task_progress(missing_id, "failed", error="요청 처리 실패 (유실)")

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

                    # 불일치 정보 기록
                    logging.warning(f"요청 ID 맵 상태: {len(status_tracker.request_map)}개 항목")
                    logging.warning(f"작업 ID 맵 상태: {len(status_tracker.task_map)}개 항목")

                    # 미처리 요청 강제 처리
                    for task_id in range(total_requests):
                        if task_id not in processed_ids:
                            error_data = [
                                original_requests.get(task_id, {"task_id": task_id}),
                                {"error": "요청 처리 추적 불일치로 인한 강제 종료", "request_id": f"forced_{task_id}"},
                                None
                            ]
                            append_to_jsonl(error_data, save_filepath)
                            processed_ids.add(task_id)

                            # 진행률 업데이트
                            if progress_tracker:
                                progress_tracker.update_task_progress(task_id, "failed", error="추적 불일치로 인한 강제 종료")

                    # 이미 모든 미처리 요청을 기록했으므로 중복 기록 방지
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

    # 전달받은 prefix가 존재하는지 확인하고 없으면 생성
    os.makedirs(prefix, exist_ok=True)

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

    # 배치 크기 조정 - 너무 작거나 크지 않도록 (0인 경우 포함)
    batch_size = min(max(5, batch_size), 100)  # 최소 5, 최대 100
    batch_size = max(1, min(batch_size, total_requests))  # 최소 1, 최대 total_requests

    # ProgressTracker 초기화
    progress_tracker = ProgressTracker(total_requests, batch_size, log_dir=prefix)
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


@dataclass
class BatchMetrics:
    """배치 처리 지표를 저장하는 데이터 클래스"""
    total_requests: int = 0
    completed_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limit_errors: int = 0
    other_errors: int = 0
    start_time: float = 0.0
    last_rate_limit_time: float = 0.0

    def elapsed_time(self) -> float:
        """배치 처리 시작부터 현재까지의 경과 시간"""
        return time.time() - self.start_time

    def completion_percentage(self) -> float:
        """요청 완료율"""
        if self.total_requests == 0:
            return 0.0
        return (self.completed_requests / self.total_requests) * 100

    def success_rate(self) -> float:
        """요청 성공률"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def avg_time_per_request(self) -> float:
        """요청당 평균 처리 시간"""
        if self.completed_requests == 0:
            return 0.0
        return self.elapsed_time() / self.completed_requests

    def requests_per_second(self) -> float:
        """초당 처리된 요청 수"""
        elapsed = self.elapsed_time()
        if elapsed == 0:
            return 0.0
        return self.completed_requests / elapsed

    def summary(self) -> Dict[str, Any]:
        """배치 처리 요약 정보"""
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "completion_percentage": self.completion_percentage(),
            "success_rate": self.success_rate(),
            "elapsed_time": self.elapsed_time(),
            "avg_time_per_request": self.avg_time_per_request(),
            "requests_per_second": self.requests_per_second(),
            "rate_limit_errors": self.rate_limit_errors,
            "other_errors": self.other_errors
        }


class RequestManager:
    """API 요청 관리 클래스 - 고유 ID 기반 요청 추적"""

    def __init__(self):
        self.requests: Dict[str, Dict] = {}  # 요청 ID -> 요청 정보
        self.original_indices: Dict[int, str] = {}  # 원본 인덱스 -> 요청 ID
        self.processed_ids: Set[str] = set()  # 처리된 요청 ID
        self.metrics = BatchMetrics()
        self.metrics.start_time = time.time()

    def register_request(self, request_data: Dict, original_idx: int) -> str:
        """새 요청 등록 및 고유 ID 할당"""
        # UUID 생성 - 고유성 보장
        request_id = str(uuid.uuid4())

        # 요청 정보 저장
        self.requests[request_id] = {
            "data": request_data,
            "original_idx": original_idx,
            "status": "pending",
            "attempts": 0,
            "max_attempts": 3,  # 기본값, 변경 가능
            "register_time": time.time(),
            "start_time": None,
            "end_time": None,
            "duration": None,
            "response": None,
            "error": None
        }

        # 원본 인덱스 매핑
        self.original_indices[original_idx] = request_id

        # 지표 업데이트
        self.metrics.total_requests += 1

        return request_id

    def mark_request_start(self, request_id: str) -> None:
        """요청 시작 표시"""
        if request_id in self.requests:
            self.requests[request_id]["status"] = "in_progress"
            self.requests[request_id]["start_time"] = time.time()
            self.requests[request_id]["attempts"] += 1

    def mark_request_end(self, request_id: str, response: Any = None, error: str = None) -> None:
        """요청 완료 표시"""
        if request_id in self.requests:
            end_time = time.time()
            req_info = self.requests[request_id]

            # 시간 정보 업데이트
            req_info["end_time"] = end_time
            if req_info["start_time"]:
                req_info["duration"] = end_time - req_info["start_time"]

            if error:
                # 오류 발생 시
                req_info["error"] = error

                # 재시도 가능 여부 확인
                if req_info["attempts"] < req_info["max_attempts"]:
                    req_info["status"] = "retry"
                else:
                    req_info["status"] = "failed"
                    self.processed_ids.add(request_id)
                    self.metrics.completed_requests += 1
                    self.metrics.failed_requests += 1

                # 속도 제한 오류 여부 확인
                if "rate limit" in error.lower() or "rate_limit" in error.lower():
                    self.metrics.rate_limit_errors += 1
                    self.metrics.last_rate_limit_time = time.time()
                else:
                    self.metrics.other_errors += 1
            else:
                # 성공 시
                req_info["status"] = "success"
                req_info["response"] = response
                self.processed_ids.add(request_id)
                self.metrics.completed_requests += 1
                self.metrics.successful_requests += 1

    def get_retry_requests(self) -> List[Tuple[str, Dict]]:
        """재시도가 필요한 요청 목록 반환"""
        return [(req_id, req_info) for req_id, req_info in self.requests.items()
                if req_info["status"] == "retry"]

    def get_pending_requests(self) -> List[Tuple[str, Dict]]:
        """대기 중인 요청 목록 반환"""
        return [(req_id, req_info) for req_id, req_info in self.requests.items()
                if req_info["status"] == "pending"]

    def get_results(self) -> List[Tuple[int, Any]]:
        """처리 결과 목록 반환 (원본 인덱스, 응답)"""
        results = []
        for req_id, req_info in self.requests.items():
            if req_info["status"] == "success":
                results.append((req_info["original_idx"], req_info["response"]))

        # 원본 인덱스 기준 정렬
        results.sort(key=lambda x: x[0])
        return results

    def get_unprocessed_indices(self) -> Set[int]:
        """처리되지 않은 원본 인덱스 목록 반환"""
        processed_indices = {self.requests[req_id]["original_idx"] for req_id in self.processed_ids}
        all_indices = set(self.original_indices.keys())
        return all_indices - processed_indices

    def all_completed(self) -> bool:
        """모든 요청이 처리 완료되었는지 확인"""
        return len(self.processed_ids) == len(self.requests)


async def process_api_requests_parallel(
        requests_data: List[Dict],
        request_url: str,
        api_key: str = None,
        headers: Dict = None,
        max_concurrent: int = 10,
        max_attempts: int = 3,
        batch_size: int = 20,
        rate_limit_pause: float = 15.0,
        response_processor: Callable = None,
        timeout: float = 60.0
) -> Tuple[List, Dict]:
    """
    API 요청을 병렬로 처리하는 함수

    Args:
        requests_data: 요청 데이터 목록
        request_url: API 엔드포인트 URL
        api_key: API 키 (있는 경우)
        headers: 요청 헤더 (있는 경우)
        max_concurrent: 최대 동시 요청 수
        max_attempts: 최대 재시도 횟수
        batch_size: 배치 크기
        rate_limit_pause: 속도 제한 오류 후 대기 시간 (초)
        response_processor: 응답 처리 함수 (있는 경우)
        timeout: 요청 타임아웃 (초)

    Returns:
        Tuple[List, Dict]: 처리 결과 목록과 처리 지표
    """
    # 요청 관리자 초기화
    request_manager = RequestManager()

    # 모든 요청 등록
    for idx, request_data in enumerate(requests_data):
        request_id = request_manager.register_request(request_data, idx)
        # 최대 재시도 횟수 설정
        request_manager.requests[request_id]["max_attempts"] = max_attempts

    # 기본 헤더 설정
    if headers is None:
        headers = {}

    if api_key and "Authorization" not in headers:
        # OpenAI 스타일 인증 헤더
        headers["Authorization"] = f"Bearer {api_key}"

    # 세마포어 초기화 - 동시 요청 제한
    semaphore = asyncio.Semaphore(max_concurrent)

    # 단일 요청 처리 함수
    async def process_single_request(session, request_id):
        request_info = request_manager.requests[request_id]
        request_data = request_info["data"]

        # 요청 메타데이터 설정 - request_id 포함
        if "metadata" not in request_data:
            request_data["metadata"] = {}
        request_data["metadata"]["request_id"] = request_id

        # 세마포어 획득 (동시 요청 제한)
        async with semaphore:
            # 요청 시작 표시
            request_manager.mark_request_start(request_id)

            try:
                # API 요청 실행
                async with session.post(
                        url=request_url,
                        json=request_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    # HTTP 상태 코드 확인
                    if response.status >= 400:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP error {response.status}: {error_text}",
                            headers=response.headers
                        )

                    # 응답 JSON 파싱
                    response_json = await response.json()

                    # 응답 처리 (커스텀 프로세서가 있는 경우)
                    processed_response = response_json
                    if response_processor:
                        try:
                            processed_response = response_processor(
                                response_json,
                                request_data.get("metadata", {})
                            )
                        except Exception as e:
                            # 응답 처리 실패 시 재시도를 위해 오류 발생
                            raise ValueError(f"응답 처리 오류: {str(e)}")

                    # 응답 저장
                    request_manager.mark_request_end(request_id, processed_response)

                    # 진행 상황 로깅 (진행률이 5%씩 변할 때마다)
                    progress = request_manager.metrics.completion_percentage()
                    if progress % 5 < 0.5:  # 5%씩 로깅
                        logging.info(
                            f"진행률: {progress:.1f}% ({request_manager.metrics.completed_requests}/{request_manager.metrics.total_requests})")

                    return processed_response

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                error_msg = f"네트워크 오류: {str(e)}"
                request_manager.mark_request_end(request_id, error=error_msg)
                return None
            except Exception as e:
                error_msg = f"요청 처리 오류: {str(e)}"
                request_manager.mark_request_end(request_id, error=error_msg)
                return None

    # 배치 처리 메인 루프
    async with aiohttp.ClientSession() as session:
        while not request_manager.all_completed():
            # 속도 제한 후 대기 시간 확인
            if request_manager.metrics.last_rate_limit_time > 0:
                time_since_rate_limit = time.time() - request_manager.metrics.last_rate_limit_time
                if time_since_rate_limit < rate_limit_pause:
                    # 대기 시간 미충족 시 대기
                    await asyncio.sleep(rate_limit_pause - time_since_rate_limit)

            # 재시도 요청 처리
            retry_requests = request_manager.get_retry_requests()
            if retry_requests:
                retry_tasks = []
                for request_id, _ in retry_requests[:batch_size]:
                    task = asyncio.create_task(process_single_request(session, request_id))
                    retry_tasks.append(task)

                if retry_tasks:
                    await asyncio.gather(*retry_tasks, return_exceptions=True)
                    continue  # 재시도 요청 우선 처리

            # 대기 중인 요청 처리
            pending_requests = request_manager.get_pending_requests()
            if pending_requests:
                batch_tasks = []
                for request_id, _ in pending_requests[:batch_size]:
                    task = asyncio.create_task(process_single_request(session, request_id))
                    batch_tasks.append(task)

                if batch_tasks:
                    await asyncio.gather(*batch_tasks, return_exceptions=True)
                    continue

            # 대기 중이거나 재시도할 요청이 없으면 짧게 대기
            if not request_manager.all_completed():
                await asyncio.sleep(0.1)

    # 처리되지 않은 요청 확인
    unprocessed = request_manager.get_unprocessed_indices()
    if unprocessed:
        logging.warning(f"{len(unprocessed)}개 요청이 처리되지 않았습니다: {sorted(unprocessed)}")

    # 결과 반환
    results = [result for _, result in request_manager.get_results()]
    metrics = request_manager.metrics.summary()

    # 최종 요약 로깅
    logging.info(f"병렬 처리 완료: {metrics['successful_requests']}/{metrics['total_requests']} 성공 "
                 f"({metrics['success_rate']:.1f}%), {metrics['elapsed_time']:.2f}초 소요")

    return results, metrics


def process_requests_from_file(
        request_file: str,
        result_file: str,
        request_url: str,
        api_key: str = None,
        max_concurrent: int = 10,
        batch_size: int = 20,
        max_attempts: int = 3,
        response_processor: Callable = None,
        timeout: float = 60.0
) -> Dict:
    """
    파일에서 요청을 읽어 병렬로 처리하는 함수

    Args:
        request_file: 요청 데이터가 있는 JSONL 파일 경로
        result_file: 결과를 저장할 JSONL 파일 경로
        request_url: API 엔드포인트 URL
        api_key: API 키 (있는 경우)
        max_concurrent: 최대 동시 요청 수
        batch_size: 배치 크기
        max_attempts: 최대 재시도 횟수
        response_processor: 응답 처리 함수 (있는 경우)
        timeout: 요청 타임아웃 (초)

    Returns:
        Dict: 처리 지표
    """
    # 요청 파일 읽기
    requests_data = []
    with open(request_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                request_data = json.loads(line.strip())
                # 인덱스 정보 추가
                if isinstance(request_data, dict) and "metadata" not in request_data:
                    request_data["metadata"] = {"original_index": i}
                requests_data.append(request_data)
            except json.JSONDecodeError:
                logging.warning(f"라인 {i + 1}: JSON 파싱 실패, 건너뜁니다.")

    logging.info(f"{len(requests_data)}개 요청을 로드했습니다.")

    # 병렬 처리 실행
    loop = asyncio.get_event_loop()
    results, metrics = loop.run_until_complete(
        process_api_requests_parallel(
            requests_data=requests_data,
            request_url=request_url,
            api_key=api_key,
            max_concurrent=max_concurrent,
            max_attempts=max_attempts,
            batch_size=batch_size,
            response_processor=response_processor,
            timeout=timeout
        )
    )

    # 결과 파일 저장
    with open(result_file, 'w') as f:
        for idx, result in enumerate(results):
            # 원본 요청과 결과 함께 저장
            output = {
                "request": requests_data[idx],
                "response": result,
                "index": idx
            }
            f.write(json.dumps(output) + '\n')

    logging.info(f"처리 결과를 {result_file}에 저장했습니다.")

    return metrics
