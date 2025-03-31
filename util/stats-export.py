import pandas as pd
import os
import argparse
from tabulate import tabulate


def export_stats_file(input_file, output_dir=None, formats=None):
    """
    NL2SQL ��� ������ ���� �������� ��������

    Args:
        input_file: �Է� CSV ���� ���
        output_dir: ��� ������ ������ ���丮 (�⺻��: �Է� ���ϰ� ���� ���丮)
        formats: ������ ���� ��� (�⺻��: ['md', 'wiki'])
    """
    if formats is None:
        formats = ['md', 'wiki']

    if not os.path.exists(input_file):
        print(f"����: �Է� ������ ã�� �� �����ϴ�: {input_file}")
        return

    # ��� ���丮 ����
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)

    # �⺻ ���ϸ� (Ȯ���� ����)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # CSV ���� �ε�
    df = pd.read_csv(input_file)

    print(f"'{input_file}' ���� �ε�: {len(df)}�� ��")

    # ��� ��������
    for fmt in formats:
        if fmt.lower() == 'md':
            output_file = os.path.join(output_dir, f"{base_name}.md")
            export_markdown(df, output_file)

        elif fmt.lower() == 'wiki' or fmt.lower() == 'redmine':
            output_file = os.path.join(output_dir, f"{base_name}.wiki")
            export_wiki(df, output_file)

        else:
            print(f"���: �������� �ʴ� ����: {fmt}")

    # �𵨺� ���� ��� ��������
    if len(df) > 0 and 'nl2sql_model' in df.columns:
        summary_file = os.path.join(output_dir, f"{base_name}_summary.md")
        export_model_summary(df, summary_file)


def export_markdown(df, output_file):
    """�������������� Markdown ���̺�� ��������"""
    # ���̺� ����
    md_content = "# NL2SQL ��� ����\n\n"

    # ���̺� ���� (tabulate ���)
    md_content += tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

    # ���� ����
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Markdown �������� ����: {output_file}")


def export_wiki(df, output_file):
    """�������������� Redmine Wiki ���̺�� ��������"""
    headers = df.columns.tolist()

    # ���̺� ����
    wiki_content = "h1. NL2SQL ��� ����\n\n"

    # ���̺� ��� ��
    wiki_content += "|_. " + " |_. ".join(headers) + " |\n"

    # ������ ��
    for _, row in df.iterrows():
        values = []
        for col in headers:
            # ������ ��� �Ҽ��� ���� ����
            cell_value = row[col]
            if pd.api.types.is_numeric_dtype(type(cell_value)):
                # �Ҽ��� ��° �ڸ����� �ݿø�
                if isinstance(cell_value, float):
                    cell_value = f"{cell_value:.2f}"
                else:
                    cell_value = str(cell_value)
            else:
                cell_value = str(cell_value)
            values.append(cell_value)

        wiki_content += "| " + " | ".join(values) + " |\n"

    # ���� ����
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(wiki_content)

    print(f"Redmine Wiki �������� ����: {output_file}")


def export_model_summary(df, output_file):
    """�𵨺� ���� ����� Markdown���� ��������"""
    # �𵨺� �׷�ȭ
    if 'nl2sql_model' not in df.columns:
        print("���: 'nl2sql_model' ���� ���� �� ����� ������ �� �����ϴ�")
        return

    # ����� ��ǥ ���
    metrics = ['success_rate', 'accuracy', 'avg_processing_time', 'throughput', 'avg_translation_time_s',
               'avg_verification_time_s']
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("���: ����� ��ǥ�� �����ϴ�")
        return

    # �𵨺� ��� ���
    summary = df.groupby('nl2sql_model')[available_metrics].agg(['mean', 'std', 'count'])

    # ��� ���� ����
    md_content = "# NL2SQL �𵨺� ���� ���\n\n"

    # �� ��ǥ�� ���̺� ����
    for metric in available_metrics:
        if metric not in summary.columns:
            continue

        metric_df = summary[metric].copy()

        # �� �̸� ����
        metric_df.columns = ['���', 'ǥ������', '���� Ƚ��']

        # ���̺� ����
        if metric == 'success_rate' or metric == 'accuracy':
            md_content += f"## ������/��Ȯ�� ({metric})\n\n"
        elif metric == 'avg_processing_time':
            md_content += "## ��� ó�� �ð� (��)\n\n"
        elif metric == 'throughput':
            md_content += "## ó���� (����/��)\n\n"
        elif metric == 'avg_translation_time_s':
            md_content += "## ���� �ð� (��)\n\n"
        elif metric == 'avg_verification_time_s':
            md_content += "## ���� �ð� (��)\n\n"
        else:
            md_content += f"## {metric}\n\n"

        # ���̺� ����
        md_content += tabulate(metric_df, headers='keys', tablefmt='pipe') + "\n\n"

    # ���� ����
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"�𵨺� ���� ��� ����: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL2SQL ��� ������ ���� �������� ��������")
    parser.add_argument("input", help="�Է� CSV ���� ���")
    parser.add_argument("--output-dir", "-o", help="��� ���丮 (�⺻��: �Է� ���ϰ� ���� ���丮)")
    parser.add_argument("--formats", "-f", nargs="+", default=["md", "wiki"],
                        help="������ ���� (����: md, wiki/redmine, �⺻��: md wiki)")

    args = parser.parse_args()

    export_stats_file(args.input, args.output_dir, args.formats)

# ��� ����:
# python stats_exporter.py [PREFIX]/stats/nl2sql_translation_stats.csv --output-dir [PREFIX]/reports
# python stats_exporter.py [PREFIX]/stats/nl2sql_verification_stats.csv --formats md wiki