import os
import csv
import logging
import pandas as pd
from datetime import datetime
from tabulate import tabulate


class EvalResultsLogger:
    """NL2SQL 평가 결과를 로깅하고 저장하는 클래스"""

    def __init__(self, output_dir="./results", filename=None):
        """
        결과 로거 초기화

        Args:
            output_dir: 결과 파일을 저장할 디렉토리
            filename: 결과 파일명 (지정하지 않으면 자동 생성)
        """
        # 절대 경로로 변환
        self.output_dir = os.path.abspath(output_dir)

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

        # 결과 파일 이름 설정
        if filename is None:
            filename = f"nl2sql_eval_results.csv"
        self.csv_path = os.path.join(self.output_dir, filename)

        # 로거 설정 - 로거 이름을 고유하게 설정하여 다른 로거와 분리
        self.logger = logging.getLogger("EvalResultsLogger")

        # 핸들러 중복 방지
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

        # 로거 레벨 설정
        self.logger.setLevel(logging.INFO)

        # 콘솔 핸들러 추가 (메시지 중복 방지를 위해 로거만의 핸들러 설정)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s [%(name)s][%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 중요: propagate를 False로 설정하여 메시지가 루트 로거로 전파되지 않도록 함
        self.logger.propagate = False

        self.logger.info(f"평가 결과를 {self.csv_path}에 저장합니다.")

    def log_evaluation_result(self, eval_data):
        """
        평가 결과를 로그로 출력하고 CSV 파일에 추가

        Args:
            eval_data: 평가 결과 데이터 (딕셔너리)
        """
        # 실행 시간 추가
        eval_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ms 단위의 시간을 s 단위로 변환
        if 'avg_translation_time_ms' in eval_data:
            eval_data['avg_translation_time_s'] = eval_data['avg_translation_time_ms'] / 1000
            del eval_data['avg_translation_time_ms']

        if 'avg_verification_time_ms' in eval_data:
            eval_data['avg_verification_time_s'] = eval_data['avg_verification_time_ms'] / 1000
            del eval_data['avg_verification_time_ms']

        # 메모리 사용량 및 eval_environment 삭제
        if 'memory_usage' in eval_data:
            del eval_data['memory_usage']

        if 'eval_environment' in eval_data:
            del eval_data['eval_environment']

        # 콘솔에 결과 테이블 출력
        self._print_table(eval_data)

        # CSV 파일에 결과 추가
        self._append_to_csv(eval_data)

        return eval_data

    def _print_table(self, eval_data):
        """평가 결과를 테이블 형식으로 콘솔에 출력"""
        # 데이터 정리
        headers = ["항목", "값"]
        table_data = []

        # 기본 정보
        table_data.append(["실행 시간", eval_data.get('timestamp', 'N/A')])
        table_data.append(["NL2SQL 모델", eval_data.get('nl2sql_model', 'N/A')])
        if 'evaluator_model' in eval_data and eval_data.get('evaluator_model'):
            table_data.append(["평가자 모델", eval_data.get('evaluator_model', 'N/A')])
        table_data.append(["테스트셋", eval_data.get('test_dataset', 'N/A')])
        table_data.append(["테스트 크기", eval_data.get('test_size', 'N/A')])

        # 성공 수 추가
        if 'successful_count' in eval_data:
            table_data.append(["성공 수", f"{eval_data.get('successful_count', 0)}/{eval_data.get('test_size', 0)}"])

        # 성능 지표
        table_data.append(["정확도 (%)", f"{eval_data.get('accuracy', 0):.2f}"])
        table_data.append(["평균 처리 시간 (s)", f"{eval_data.get('avg_processing_time', 0):.3f}"])
        table_data.append(["배치 처리량 (query/s)", f"{eval_data.get('batch_throughput', 0):.2f}"])

        # 추가 시간 지표 (ms -> s 변환된 값)
        if 'avg_translation_time_s' in eval_data:
            table_data.append(["평균 변환 시간 (s)", f"{eval_data.get('avg_translation_time_s', 0):.3f}"])
        if 'avg_verification_time_s' in eval_data:
            table_data.append(["평균 검증 시간 (s)", f"{eval_data.get('avg_verification_time_s', 0):.3f}"])

        # # 설정 정보
        # table_data.append(["배치 크기", eval_data.get('batch_size', 'N/A')])
        # table_data.append(["최대 동시 요청", eval_data.get('max_concurrent', 'N/A')])
        # table_data.append(["최대 재시도 횟수", eval_data.get('max_retries', 'N/A')])

        # 테이블 출력
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        # 로깅 레벨을 INFO로 설정하고 tabulate 결과를 출력
        # print() 대신 logger를 사용해 일관성 유지
        self.logger.info(f"\n평가 결과 요약:\n{table}")

    def _append_to_csv(self, eval_data):
        """평가 결과를 CSV 파일에 추가"""
        # 파일 존재 여부 확인
        file_exists = os.path.isfile(self.csv_path)

        fieldnames = [
            'timestamp', 'nl2sql_model', 'evaluator_model', 'test_dataset',
            'test_size', 'successful_count', 'accuracy', 'avg_processing_time',
            'batch_throughput',
            # 'batch_size', 'max_concurrent', 'max_retries',
            'model_size', 'quantization', 'comments',
            'phase', 'avg_translation_time_s', 'throughput', 'success_rate', 'error_rate',
            'avg_verification_time_s'  # ms -> s로 변경
        ]

        # CSV 파일에 데이터 추가
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 파일이 새로 생성된 경우 헤더 작성
            if not file_exists:
                writer.writeheader()

            # 데이터 작성
            writer.writerow(eval_data)

        self.logger.info(f"평가 결과가 {self.csv_path}에 추가되었습니다.")

    def generate_summary_report(self, output_path=None):
        """
        지금까지의 평가 결과를 요약한 리포트 생성

        Args:
            output_path: 요약 리포트 저장 경로 (없으면 자동 생성)

        Returns:
            DataFrame: 요약 데이터
        """
        if not os.path.isfile(self.csv_path):
            self.logger.warning(f"결과 파일 {self.csv_path}가 존재하지 않습니다.")
            return None

        # CSV 파일 로드
        df = pd.read_csv(self.csv_path)

        if output_path is None:
            output_path = os.path.join(self.output_dir, "nl2sql_eval_summary.html")

        # 모델별 평균 성능
        model_summary = df.groupby('nl2sql_model').agg({
            'accuracy': 'mean',
            'avg_processing_time': 'mean',
            'batch_throughput': 'mean',
            'test_size': 'sum'
        }).reset_index()

        # 데이터셋별 성능
        dataset_summary = df.groupby(['nl2sql_model', 'test_dataset']).agg({
            'accuracy': 'mean',
            'avg_processing_time': 'mean'
        }).reset_index()

        # 평가자 모델별 성능
        evaluator_summary = df.groupby(['nl2sql_model', 'evaluator_model']).agg({
            'accuracy': 'mean'
        }).reset_index()

        # HTML 리포트 생성
        html_content = f"""
        <html>
        <head>
            <title>NL2SQL 평가 결과 요약</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>NL2SQL 평가 결과 요약</h1>
            <p>생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <h2>모델별 평균 성능</h2>
            {model_summary.to_html(index=False, float_format=lambda x: f'{x:.3f}')}

            <h2>데이터셋별 성능</h2>
            {dataset_summary.to_html(index=False, float_format=lambda x: f'{x:.3f}')}

            <h2>평가자 모델별 성능</h2>
            {evaluator_summary.to_html(index=False, float_format=lambda x: f'{x:.3f}')}
        </body>
        </html>
        """

        # HTML 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"요약 리포트가 {output_path}에 생성되었습니다.")

        return {
            'model_summary': model_summary,
            'dataset_summary': dataset_summary,
            'evaluator_summary': evaluator_summary
        }