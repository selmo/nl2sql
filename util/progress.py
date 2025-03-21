import logging
from time import time
from tqdm import tqdm


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
            self.pbar.update(1)
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
