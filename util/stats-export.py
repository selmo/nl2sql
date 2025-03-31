import pandas as pd
import os
import argparse
from tabulate import tabulate


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL2SQL 통계 파일을 여러 형식으로 내보내기")
    parser.add_argument("input", help="입력 CSV 파일 경로")
    parser.add_argument("--output-dir", "-o", help="출력 디렉토리 (기본값: 입력 파일과 같은 디렉토리)")
    parser.add_argument("--formats", "-f", nargs="+", default=["md", "wiki"],
                        help="내보낼 형식 (지원: md, wiki/redmine, 기본값: md wiki)")

    args = parser.parse_args()

    export_stats_file(args.input, args.output_dir, args.formats)

# 사용 예시:
# python stats_exporter.py [PREFIX]/stats/nl2sql_translation_stats.csv --output-dir [PREFIX]/reports
# python stats_exporter.py [PREFIX]/stats/nl2sql_verification_stats.csv --formats md wiki