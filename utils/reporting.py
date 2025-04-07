import argparse
import csv
import logging
import os
from datetime import datetime

import pandas as pd
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

    def log_evaluation_result(self, eval_data, evaluation: bool = False):
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
        self._append_to_csv(eval_data, evaluation)

        return eval_data

    def _print_table(self, eval_data):
        """Displays evaluation results in a tabular format on the console"""
        # Prepare data
        headers = ["Metric", "Value"]
        table_data = []

        # Function to add section headers
        def add_section_header(title):
            table_data.append([f"=== {title} ===", ""])

        # Basic Information Section
        add_section_header("Basic Information")
        table_data.append(["Execution Time", eval_data.get('timestamp', 'N/A')])
        table_data.append(["NL2SQL Model", eval_data.get('nl2sql_model', 'N/A')])
        if 'evaluator_model' in eval_data and eval_data.get('evaluator_model'):
            table_data.append(["Evaluator Model", eval_data.get('evaluator_model', 'N/A')])
        table_data.append(["Test Dataset", eval_data.get('test_dataset', 'N/A')])
        table_data.append(["Test Size", eval_data.get('test_size', 'N/A')])

        # Performance Metrics Section
        add_section_header("Performance Metrics")

        # Add success count
        if 'successful_count' in eval_data:
            success_rate = (eval_data.get('successful_count', 0) / eval_data.get('test_size', 1)) * 100
            table_data.append(["Success Count",
                               f"{eval_data.get('successful_count', 0)}/{eval_data.get('test_size', 0)} ({success_rate:.1f}%)"])

        table_data.append(["Accuracy", f"{eval_data.get('accuracy', 0):.2f}%"])

        # Time Metrics
        add_section_header("Time Metrics")
        table_data.append(["Avg Processing Time", f"{eval_data.get('avg_processing_time', 0):.3f} sec/query"])
        table_data.append(["Batch Throughput", f"{eval_data.get('batch_throughput', 0):.2f} queries/sec"])

        # Additional time metrics (ms -> s converted values)
        if 'avg_translation_time_s' in eval_data:
            table_data.append(["Avg Translation Time", f"{eval_data.get('avg_translation_time_s', 0):.3f} sec"])
        if 'avg_verification_time_s' in eval_data:
            table_data.append(["Avg Verification Time", f"{eval_data.get('avg_verification_time_s', 0):.3f} sec"])

        # Output table - using grid format for better alignment
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        # Logging
        self.logger.info(f"\nEvaluation Results Summary:\n{table}")

    def _append_to_csv(self, eval_data, evaluation: bool = False):
        """평가 결과를 CSV 파일에 추가"""
        # 파일 존재 여부 확인
        file_exists = os.path.isfile(self.csv_path)

        if evaluation:
            fieldnames = [
                'timestamp', 'nl2sql_model', 'evaluator_model', 'test_dataset',
                'test_size', 'successful_count', 'accuracy', 'comments', 'phase'
            ]
        else:
            fieldnames = [
                'timestamp', 'nl2sql_model', 'test_dataset', 'test_size', 'successful_count', 'accuracy',
                'avg_processing_time', 'batch_throughput', 'model_size', 'comments', 'phase',
                'throughput', 'success_rate', 'error_rate'
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


def export_stats_file(input_file, output_dir=None, formats=None):
    """
    NL2SQL 통계 파일을 여러 형식으로 내보내기

    Args:
        input_file: 입력 CSV 파일 경로
        output_dir: 출력 파일을 저장할 디렉토리 (기본값: 입력 파일과 같은 디렉토리)
        formats: 내보낼 형식 목록 (기본값: ['md', 'wiki'])
    """
    if formats is None:
        formats = ['md', 'wiki']

    if not os.path.exists(input_file):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_file}")
        return

    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)

    # 기본 파일명 (확장자 제외)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # CSV 파일 로드
    df = pd.read_csv(input_file)

    print(f"'{input_file}' 파일 로드: {len(df)}개 행")

    # 결과 내보내기
    for fmt in formats:
        if fmt.lower() == 'md':
            output_file = os.path.join(output_dir, f"{base_name}.md")
            export_markdown(df, output_file)

        elif fmt.lower() == 'wiki' or fmt.lower() == 'redmine':
            output_file = os.path.join(output_dir, f"{base_name}.wiki")
            export_wiki(df, output_file)

        else:
            print(f"경고: 지원되지 않는 형식: {fmt}")

    # 모델별 성능 요약 내보내기
    if len(df) > 0 and 'nl2sql_model' in df.columns:
        summary_file = os.path.join(output_dir, f"{base_name}_summary.md")
        export_model_summary(df, summary_file)


def export_markdown(df, output_file):
    """데이터프레임을 Markdown 테이블로 내보내기"""
    # 테이블 제목
    md_content = "# NL2SQL 통계 보고서\n\n"

    # 테이블 생성 (tabulate 사용)
    md_content += tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Markdown 형식으로 저장: {output_file}")


def export_wiki(df, output_file):
    """데이터프레임을 Redmine Wiki 테이블로 내보내기"""
    headers = df.columns.tolist()

    # 테이블 시작
    wiki_content = "h1. NL2SQL 통계 보고서\n\n"

    # 테이블 헤더 행
    wiki_content += "|_. " + " |_. ".join(headers) + " |\n"

    # 데이터 행
    for _, row in df.iterrows():
        values = []
        for col in headers:
            # 숫자인 경우 소수점 형식 조정
            cell_value = row[col]
            if pd.api.types.is_numeric_dtype(type(cell_value)):
                # 소수점 둘째 자리까지 반올림
                if isinstance(cell_value, float):
                    cell_value = f"{cell_value:.2f}"
                else:
                    cell_value = str(cell_value)
            else:
                cell_value = str(cell_value)
            values.append(cell_value)

        wiki_content += "| " + " | ".join(values) + " |\n"

    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(wiki_content)

    print(f"Redmine Wiki 형식으로 저장: {output_file}")


def export_model_summary(df, output_file):
    """모델별 성능 요약을 Markdown으로 내보내기"""
    # 모델별 그룹화
    if 'nl2sql_model' not in df.columns:
        print("경고: 'nl2sql_model' 열이 없어 모델 요약을 생성할 수 없습니다")
        return

    # 요약할 지표 목록
    metrics = ['success_rate', 'accuracy', 'avg_processing_time', 'throughput', 'avg_translation_time_s',
               'avg_verification_time_s']
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("경고: 요약할 지표가 없습니다")
        return

    # 모델별 통계 계산
    summary = df.groupby('nl2sql_model')[available_metrics].agg(['mean', 'std', 'count'])

    # 요약 내용 생성
    md_content = "# NL2SQL 모델별 성능 요약\n\n"

    # 각 지표별 테이블 생성
    for metric in available_metrics:
        if metric not in summary.columns:
            continue

        metric_df = summary[metric].copy()

        # 열 이름 변경
        metric_df.columns = ['평균', '표준편차', '실행 횟수']

        # 테이블 제목
        if metric == 'success_rate' or metric == 'accuracy':
            md_content += f"## 성공률/정확도 ({metric})\n\n"
        elif metric == 'avg_processing_time':
            md_content += "## 평균 처리 시간 (초)\n\n"
        elif metric == 'throughput':
            md_content += "## 처리량 (쿼리/초)\n\n"
        elif metric == 'avg_translation_time_s':
            md_content += "## 번역 시간 (초)\n\n"
        elif metric == 'avg_verification_time_s':
            md_content += "## 검증 시간 (초)\n\n"
        else:
            md_content += f"## {metric}\n\n"

        # 테이블 생성
        md_content += tabulate(metric_df, headers='keys', tablefmt='pipe') + "\n\n"

    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"모델별 성능 요약 저장: {output_file}")

def main():
    """명령행에서 독립 실행될 때 사용되는 기능"""
    parser = argparse.ArgumentParser(description="NL2SQL 통계 파일을 여러 형식으로 내보내기")
    parser.add_argument("input", help="입력 CSV 파일 경로")
    parser.add_argument("--output-dir", "-o", help="출력 디렉토리 (기본값: 입력 파일과 같은 디렉토리)")
    parser.add_argument("--formats", "-f", nargs="+", default=["md", "wiki"],
                        help="내보낼 형식 (지원: md, wiki/redmine, 기본값: md wiki)")

    args = parser.parse_args()

    export_stats_file(args.input, args.output_dir, args.formats)

if __name__ == "__main__":
    main()