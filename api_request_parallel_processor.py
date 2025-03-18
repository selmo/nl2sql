import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field


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

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

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
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

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

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
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


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
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
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        result_data = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

            logging.info(f"Response #{self.task_id}: OK")
        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                result_data = data
                if save_filepath is not None:
                    append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            result_data = data
            if save_filepath is not None:
                append_to_jsonl(data, save_filepath)
                logging.debug(f"Request {self.task_id} saved to {save_filepath}")
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

        return result_data


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    if match is None:
        return "chat/completions"
    else:
        return match[1]


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
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
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
):
    if save_filepath is None:
        save_filepath = requests_filepath.replace(".jsonl", "_results.jsonl")

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath,
            save_filepath,
            request_url,
            api_key,
            max_requests_per_minute,
            max_tokens_per_minute,
            token_encoding_name,
            max_attempts,
            logging_level,
        )
    )


async def process_api_requests_in_memory(
    requests_list,
    request_url: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    api_key: str = '',
    logging_level: int = logging.INFO
):
    """
    파일이 아닌 '메모리 상의 요청 리스트'를 받아 비동기로 병렬 호출하고,
    최종 응답을 리스트 형태로 반환한다.
      - requests_list: List[dict], 각 원소는 OpenAI API 규격의 JSON (model, messages...) 등
      - return 값: List[Tuple(request_json, response_json or error_info)]
    """

    # -- 초기 세팅 --
    logging.basicConfig(level=logging_level)
    logger = logging.getLogger(__name__)

    status_tracker = StatusTracker()
    task_id_gen = task_id_generator_function()

    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # Azure OpenAI deployments의 경우 header 달라질 수 있음
    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}

    # 속도제어(Throttle) 파라미터
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001

    # capacity
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # 큐(재시도용) 및 인덱스
    queue_of_requests_to_retry = asyncio.Queue()
    current_index = 0
    total_requests = len(requests_list)

    # 결과 저장용 리스트
    results = []

    # 태스크별 상태 관리용
    in_progress_tasks = set()  # asyncio.Task 집합

    async with aiohttp.ClientSession() as session:
        while True:
            # 1) 새 request를 꺼내 올지 결정
            if (
                (current_index < total_requests) or (not queue_of_requests_to_retry.empty())
            ):
                # 우선 재시도 큐가 있으면 그걸 우선
                if not queue_of_requests_to_retry.empty():
                    next_request = await queue_of_requests_to_retry.get()
                else:
                    request_json = requests_list[current_index]
                    request_obj = APIRequest(
                        task_id=next(task_id_gen),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(
                            request_json, api_endpoint, token_encoding_name
                        ),
                        attempts_left=max_attempts
                    )
                    next_request = request_obj
                    current_index += 1

                status_tracker.num_tasks_started += 1
                status_tracker.num_tasks_in_progress += 1
            else:
                # 더 이상 보낼 request 없음 + 재시도도 없음 => 끝났는지 체크
                if status_tracker.num_tasks_in_progress == 0:
                    break
                else:
                    next_request = None

            # 2) 사용가능 capacity 업데이트
            now = time.time()
            elapsed = now - last_update_time
            # 분당 제한이므로, elapsed 초 동안 얼마나 capacity가 회복되었는지 계산
            available_request_capacity = min(
                max_requests_per_minute,
                available_request_capacity + max_requests_per_minute * elapsed / 60.0
            )
            available_token_capacity = min(
                max_tokens_per_minute,
                available_token_capacity + max_tokens_per_minute * elapsed / 60.0
            )
            last_update_time = now

            # 3) next_request를 실제로 호출할지 결정
            if next_request:
                needed_tokens = next_request.token_consumption
                if available_request_capacity >= 1 and available_token_capacity >= needed_tokens:
                    # capacity 소비
                    available_request_capacity -= 1
                    available_token_capacity -= needed_tokens
                    next_request.attempts_left -= 1

                    # 결과를 처리하는 콜백 함수 정의
                    def process_completed_task(completed_task):
                        try:
                            # 태스크에서 결과 가져오기
                            api_result = completed_task.result()
                            if api_result:  # None이 아닌 경우 결과 저장
                                results.append(api_result)
                        except Exception as e:
                            logging.error(f"Error processing task result: {e}")
                        finally:
                            # 완료된 태스크는 항상 set에서 제거
                            in_progress_tasks.remove(completed_task)

                    # 실제 비동기 호출 태스크 생성
                    task = asyncio.create_task(next_request.call_api(
                        session=session,
                        request_url=request_url,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                    ))

                    in_progress_tasks.add(task)
                    # 태스크 완료시 set에서 제거
                    task.add_done_callback(process_completed_task)
                else:
                    # capacity 모자라면 다시 큐에 넣고 대기
                    if next_request.attempts_left >= 0:
                        await queue_of_requests_to_retry.put(next_request)
                    else:
                        # 재시도 없다면 실패 처리
                        results.append((next_request.request_json, {"error": "No capacity + attempts=0"}))
                        status_tracker.num_tasks_in_progress -= 1
                        status_tracker.num_tasks_failed += 1

            # 4) 모든 태스크 완료 시점인지?
            if status_tracker.num_tasks_in_progress == 0:
                break

            # 5) rate limit 에러 쿨다운
            time_since_rate_limit = time.time() - status_tracker.time_of_last_rate_limit_error
            if time_since_rate_limit < seconds_to_pause_after_rate_limit_error:
                to_sleep = seconds_to_pause_after_rate_limit_error - time_since_rate_limit
                logger.warning(f"Pausing {to_sleep:.1f} seconds due to rate limit error.")
                await asyncio.sleep(to_sleep)

            # 6) 조금 쉰 후 반복
            await asyncio.sleep(seconds_to_sleep_each_loop)

        # 모든 태스크가 끝날 때까지 대기
        if in_progress_tasks:
            await asyncio.gather(*in_progress_tasks)

    # 최종 결과: (request_json, response_json or {"error": ...}) 리스트
    return results


def process_in_memory(
    dataset,
    request_url: str,
    max_requests_per_minute: float = 1500,
    max_tokens_per_minute: float = 125000,
    token_encoding_name: str = "cl100k_base",
    max_attempts: int = 5,
    logging_level: int = logging.INFO
):
    """
    위의 async 함수를 asyncio.run으로 감싸서 간단히 호출하고 결과를 반환하는 함수.
    """

    requests_list = dataset['job'].tolist()

    responses = asyncio.run(
        process_api_requests_in_memory(
            requests_list=requests_list,
            request_url=request_url,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
        )
    )

    reasonings = []
    descriptions = []
    gen_sqls = []

    for response in responses:
        json_data = json.loads(response[1]['message']['content'])
        logging.info('json_data: %s', json_data)
        reasonings.append(json_data['reasoning'])
        descriptions.append(json_data['description'])
        gen_sqls.append(json_data['gen_sql'])

    dataset['reasoning'] = reasonings
    dataset['description'] = descriptions
    dataset['gen_sql'] = gen_sqls

    return dataset


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    # run script
    # asyncio.run(
    #     process_api_requests_from_file(
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
    # )
    process_by_file(
        requests_filepath=args.requests_filepath,
        save_filepath=args.save_filepath,
        request_url=args.request_url,
        api_key=args.api_key,
        max_requests_per_minute=float(args.max_requests_per_minute),
        max_tokens_per_minute=float(args.max_tokens_per_minute),
        token_encoding_name=args.token_encoding_name,
        max_attempts=int(args.max_attempts),
        logging_level=int(args.logging_level),
    )
