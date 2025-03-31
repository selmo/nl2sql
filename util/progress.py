import logging
import os
from time import time
from tqdm import tqdm


class ProgressTracker:
    """병렬 처리 진행 상황을 추적하는 클래스"""

    def __init__(self, total_prompts, batch_size, log_dir="."):
        self.total_prompts = total_prompts
        self.batch_size = max(1, batch_size)  # 배치 크기가 최소 1이 되도록 보장
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time()
        self.pbar = tqdm(total=total_prompts, desc="전체 진행률")

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
        # 이 설정은 새 핸들러를 추가한 후에 해야 합니다
        self.logger.propagate = False

        # 8. 시작 로그 메시지 기록
        self.logger.debug(f"ProgressTracker 초기화: 총 {total_prompts}개 작업, 로그 파일: {self.log_file_path}")

    def update_batch_progress(self, batch_size, batch_idx, total_batches):
        """배치 시작 시 로깅"""
        self.logger.debug(f"배치 처리 시작: {batch_idx + 1}/{total_batches} (전체 진행률: {self.completed}/{self.total_prompts})")

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
            self.pbar.update(1)  # 실패한 경우에도 진행률 업데이트
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