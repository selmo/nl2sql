import time
import logging
import functools
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class TimingStats:
    """프로세스 시간 측정을 위한 통계 클래스"""
    process_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    sub_processes: Dict[str, "TimingStats"] = field(default_factory=dict)
    count: int = 0
    total_time: float = 0
    min_time: float = float('inf')
    max_time: float = 0

    def start(self) -> None:
        """프로세스 측정 시작"""
        self.start_time = time.time()

    def stop(self) -> float:
        """프로세스 측정 종료 및 소요 시간 반환"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        # 통계 업데이트
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

        return elapsed

    def add_sub_process(self, name: str) -> "TimingStats":
        """하위 프로세스 추가"""
        if name not in self.sub_processes:
            self.sub_processes[name] = TimingStats(process_name=name)
        return self.sub_processes[name]

    def get_summary(self, indent: int = 0) -> str:
        """시간 통계 요약 문자열 반환"""
        if self.count == 0:
            avg_time = 0
        else:
            avg_time = self.total_time / self.count

        indent_str = " " * indent
        summary = [
            f"{indent_str}프로세스: {self.process_name}",
            f"{indent_str}  실행 횟수: {self.count}",
            f"{indent_str}  총 소요 시간: {self.total_time:.2f}초",
            f"{indent_str}  평균 소요 시간: {avg_time:.2f}초",
        ]

        if self.count > 0:
            summary.extend([
                f"{indent_str}  최소 소요 시간: {self.min_time:.2f}초",
                f"{indent_str}  최대 소요 시간: {self.max_time:.2f}초",
            ])

        # 하위 프로세스 통계 추가
        if self.sub_processes:
            summary.append(f"{indent_str}  하위 프로세스:")
            for sub_proc in self.sub_processes.values():
                summary.append(sub_proc.get_summary(indent + 4))

        return "\n".join(summary)

    def log_summary(self, level=logging.INFO) -> None:
        """통계 요약을 로그로 출력"""
        summary = self.get_summary()
        for line in summary.split("\n"):
            logging.log(level, line)


class TimingStatsManager:
    """전체 프로세스의 시간 통계를 관리하는 클래스"""

    def __init__(self):
        self.processes: Dict[str, TimingStats] = {}
        self.active_processes: Dict[str, TimingStats] = {}

    def start_process(self, name: str, parent: Optional[str] = None) -> TimingStats:
        """프로세스 측정 시작"""
        if parent:
            if parent not in self.processes:
                self.processes[parent] = TimingStats(process_name=parent)

            process = self.processes[parent].add_sub_process(name)
        else:
            if name not in self.processes:
                self.processes[name] = TimingStats(process_name=name)
            process = self.processes[name]

        process.start()
        self.active_processes[name] = process
        return process

    def stop_process(self, name: str) -> float:
        """프로세스 측정 종료 및 소요 시간 반환"""
        if name in self.active_processes:
            process = self.active_processes[name]
            elapsed = process.stop()
            del self.active_processes[name]
            return elapsed
        else:
            logging.warning(f"프로세스 '{name}'가 시작되지 않았거나 이미 종료되었습니다.")
            return 0

    def get_process(self, name: str) -> Optional[TimingStats]:
        """이름으로 프로세스 통계 객체 가져오기"""
        if name in self.processes:
            return self.processes[name]

        # 하위 프로세스 검색
        for process in self.processes.values():
            if name in process.sub_processes:
                return process.sub_processes[name]

        return None

    def get_summary(self) -> str:
        """모든 프로세스의 통계 요약 문자열 반환"""
        summary = ["=== 프로세스 시간 통계 ==="]
        for process in self.processes.values():
            summary.append(process.get_summary())
        return "\n".join(summary)

    def log_summary(self, level=logging.INFO) -> None:
        """모든 프로세스의 통계 요약을 로그로 출력"""
        logging.log(level, "=== 프로세스 시간 통계 ===")
        for process in self.processes.values():
            process.log_summary(level)


# 전역 인스턴스 생성
timing_stats_manager = TimingStatsManager()


def time_process(process_name: str, parent: Optional[str] = None):
    """프로세스 시간을 측정하는 데코레이터"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 프로세스 시작
            timing_stats_manager.start_process(process_name, parent)

            try:
                # 원래 함수 실행
                result = func(*args, **kwargs)
                return result
            finally:
                # 프로세스 종료
                elapsed = timing_stats_manager.stop_process(process_name)
                logging.info(f"프로세스 '{process_name}' 완료: {elapsed:.2f}초 소요")

        return wrapper

    return decorator


def log_timing_stats():
    """모든 프로세스의 시간 통계를 로그로 출력"""
    timing_stats_manager.log_summary()