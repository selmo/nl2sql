"""
요청 추적 및 진행 상황 표시 모듈

이 모듈은 병렬 처리 과정에서 요청을 추적하고
진행 상황을 표시하는 기능을 제공합니다.
"""

import logging
import os
from time import time
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Callable
from tqdm import tqdm

class RequestStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"


class RequestTracker:
    """
    병렬 요청을 추적하고 결과를 관리하는 클래스
    """

    def __init__(self, total_requests: int, max_retries: int = 3):
        self.total_requests = total_requests
        self.max_retries = max_retries
        self.results = [None] * total_requests
        self.statuses = [RequestStatus.PENDING] * total_requests
        self.retry_counts = [0] * total_requests
        self.processed_ids: Set[int] = set()
        self.expected_ids: Set[int] = set(range(total_requests))
        self.error_msgs: Dict[int, str] = {}
        self.processing_times: Dict[int, float] = {}

        self.start_time = time.time()
        self.stats = {
            "success_count": 0,
            "failure_count": 0,
            "retry_count": 0,
            "in_progress_count": 0
        }

        self.logger = logging.getLogger(__name__)

    def start_request(self, request_id: int) -> None:
        """요청 시작 상태 업데이트"""
        if request_id not in self.expected_ids:
            raise ValueError(f"Invalid request ID: {request_id}")

        self.statuses[request_id] = RequestStatus.IN_PROGRESS
        self.stats["in_progress_count"] += 1
        self.logger.debug(f"Request #{request_id} started")

    def complete_request(self, request_id: int, result: Any, success: bool = True) -> None:
        """요청 완료 처리"""
        if request_id not in self.expected_ids:
            raise ValueError(f"Invalid request ID: {request_id}")

        self.results[request_id] = result
        self.processed_ids.add(request_id)
        self.processing_times[request_id] = time.time() - self.start_time

        if self.statuses[request_id] == RequestStatus.IN_PROGRESS:
            self.stats["in_progress_count"] -= 1

        if success:
            self.statuses[request_id] = RequestStatus.SUCCESS
            self.stats["success_count"] += 1
            self.logger.debug(f"Request #{request_id} completed successfully")
        else:
            self.statuses[request_id] = RequestStatus.FAILED
            self.stats["failure_count"] += 1
            self.logger.debug(f"Request #{request_id} failed")

    def mark_for_retry(self, request_id: int, error_msg: str = "") -> bool:
        """재시도 처리"""
        if request_id not in self.expected_ids:
            raise ValueError(f"Invalid request ID: {request_id}")

        if self.retry_counts[request_id] >= self.max_retries:
            self.complete_request(request_id, None, success=False)
            self.error_msgs[request_id] = f"Max retries exceeded: {error_msg}"
            return False

        self.retry_counts[request_id] += 1
        self.statuses[request_id] = RequestStatus.RETRY
        self.stats["retry_count"] += 1
        self.error_msgs[request_id] = error_msg

        self.logger.debug(
            f"Request #{request_id} marked for retry ({self.retry_counts[request_id]}/{self.max_retries})")
        return True

    def get_retry_requests(self) -> List[int]:
        """재시도가 필요한 요청 ID 목록 반환"""
        return [i for i, status in enumerate(self.statuses) if status == RequestStatus.RETRY]

    def get_pending_requests(self) -> List[int]:
        """대기 중인 요청 ID 목록 반환"""
        return [i for i, status in enumerate(self.statuses) if status == RequestStatus.PENDING]

    def get_missing_requests(self) -> List[int]:
        """처리되지 않은 요청 ID 목록 반환"""
        return list(self.expected_ids - self.processed_ids)

    def get_summary(self) -> Dict[str, Any]:
        """요청 처리 요약 정보 반환"""
        elapsed_time = time.time() - self.start_time
        return {
            "total_requests": self.total_requests,
            "processed_requests": len(self.processed_ids),
            "success_count": self.stats["success_count"],
            "failure_count": self.stats["failure_count"],
            "retry_count": self.stats["retry_count"],
            "in_progress_count": self.stats["in_progress_count"],
            "elapsed_time": elapsed_time,
            "requests_per_second": len(self.processed_ids) / elapsed_time if elapsed_time > 0 else 0
        }

    def print_summary(self) -> None:
        """요약 정보 출력"""
        summary = self.get_summary()
        self.logger.info("\n===== Request Tracker Summary =====")
        self.logger.info(f"Total Requests: {summary['total_requests']}")
        self.logger.info(
            f"Processed: {summary['processed_requests']} ({summary['processed_requests'] / summary['total_requests'] * 100:.1f}%)")
        self.logger.info(
            f"Success: {summary['success_count']} ({summary['success_count'] / summary['total_requests'] * 100:.1f}%)")
        self.logger.info(
            f"Failures: {summary['failure_count']} ({summary['failure_count'] / summary['total_requests'] * 100:.1f}%)")
        self.logger.info(f"Retries: {summary['retry_count']}")
        self.logger.info(f"In Progress: {summary['in_progress_count']}")
        self.logger.info(f"Elapsed Time: {summary['elapsed_time']:.2f} seconds")
        self.logger.info(f"Throughput: {summary['requests_per_second']:.2f} requests/second")
        self.logger.info("==================================\n")

    def is_complete(self) -> bool:
        """모든 요청이 처리되었는지 확인"""
        return len(self.processed_ids) == self.total_requests

class ProgressTracker:
    """병렬 처리 진행 상황을 추적하는 클래스"""

    # tracking.py 파일 내 ProgressTracker 클래스의 __init__ 메서드 수정
    def __init__(self, total_prompts, batch_size, log_dir="."):
        self.total_prompts = total_prompts
        self.batch_size = max(1, batch_size)  # 배치 크기가 최소 1이 되도록 보장
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time()
        self.pbar = tqdm(total=total_prompts, desc="전체 진행률")
        self.processed_tasks = set()  # 처리된 작업 추적을 위한 세트 추가

        # 로그 설정
        self.logger = logging.getLogger("BatchProcessor")

        # 1. 이미 핸들러가 설정되어 있으면 모든 핸들러 제거
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

        # 2. 로그 레벨 설정
        self.logger.setLevel(logging.DEBUG)

        # 3. 로그 디렉토리 확인 및 생성 (항상 절대 경로로 처리)
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # 변경 부분: 로그 파일 경로를 설정할 때 log_dir 내에 파일 생성
        self.log_file_path = os.path.join(log_dir, 'batch_processor.log')

        # 4. 파일 핸들러 설정 - 모든 로그 메시지를 파일에 기록
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s [%(name)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 5. 콘솔 핸들러 설정 - 배치 처리 정보를 제외한 다른 메시지만 출력
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s [%(name)s][%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)

        # 6. 배치 처리 시작 메시지를 필터링하는 필터 추가
        class BatchInfoFilter(logging.Filter):
            def filter(self, record):
                return not (record.getMessage().startswith('배치 처리 시작:') or
                            '배치 완료:' in record.getMessage())

        console_handler.addFilter(BatchInfoFilter())
        self.logger.addHandler(console_handler)

        # 7. 중요: 로거가 부모 로거로 메시지를 전파하지 않도록 설정
        self.logger.propagate = False

        # 8. 시작 로그 메시지 기록
        self.logger.debug(f"ProgressTracker 초기화: 총 {total_prompts}개 작업, 로그 파일: {self.log_file_path}")

    def update_batch_progress(self, batch_size, batch_idx, total_batches):
        """배치 시작 시 로깅"""
        self.logger.debug(f"배치 처리 시작: {batch_idx + 1}/{total_batches} (전체 진행률: {self.completed}/{self.total_prompts})")

    def update_task_progress(self, task_id, status, elapsed=None, error=None, request_id=None):
        """
        개별 태스크 진행 상황 업데이트

        Args:
            task_id: 작업 ID
            status: 상태 ('start', 'success', 'failed', 'retry')
            elapsed: 경과 시간 (초)
            error: 오류 메시지
            request_id: 요청 ID (옵션)
        """
        # 요청 ID 정보 추가 (있는 경우)
        id_info = f" (ID: {request_id})" if request_id else ""

        if status == "start":
            self.logger.debug(f"태스크 #{task_id}{id_info} 시작")
        elif status == "success":
            # 이미 처리된 작업은 다시 업데이트하지 않음
            if task_id not in self.processed_tasks:
                self.completed += 1
                self.successful += 1
                self.processed_tasks.add(task_id)  # 처리된 작업 추적을 위해 세트 추가
                self.pbar.update(1)
                if elapsed:
                    self.logger.debug(f"태스크 #{task_id}{id_info} 완료: {elapsed:.2f}초 소요")
        elif status == "failed":
            # 이미 처리된 작업은 다시 업데이트하지 않음
            if task_id not in self.processed_tasks:
                self.completed += 1
                self.failed += 1
                self.processed_tasks.add(task_id)  # 처리된 작업 추적
                self.pbar.update(1)  # 실패한 경우에도 진행률 업데이트
                error_msg = f": {error}" if error else ""
                self.logger.warning(f"태스크 #{task_id}{id_info} 실패{error_msg}")
        elif status == "retry":
            retry_count = error if error else "?"
            self.logger.info(f"태스크 #{task_id}{id_info} 재시도 중 (시도 횟수: {retry_count})")

    def update_batch_completion(self, batch_results):
        """배치 완료 시 진행 상황 업데이트"""
        batch_success = sum(1 for r in batch_results if not isinstance(r, Exception) and 'error' not in r)
        batch_failed = len(batch_results) - batch_success

        elapsed = time() - self.start_time
        requests_per_second = self.completed / elapsed if elapsed > 0 else 0

        # 현재 상태 로깅
        self.logger.debug(
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

        # 핸들러 정리
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
