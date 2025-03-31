#!/bin/bash

# Ollama ���� ���� �ڵ�ȭ ��ũ��Ʈ (Docker ȯ���)
# ����: ./ollama_benchmark.sh [�𵨸�] [������Ʈ ����] [�׽�Ʈ �ݺ� Ƚ��]

set -e

# �⺻ ������
MODEL=${1:-"all"}  # �⺻���� "all"�� ����
PROMPT_LENGTH=${2:-100}
REPEAT=${3:-5}
DOCKER_CONTAINER=$(docker ps | grep ollama | awk '{print $1}')

if [ -z "$DOCKER_CONTAINER" ]; then
    echo "Error: Ollama Docker �����̳ʸ� ã�� �� �����ϴ�."
    exit 1
fi

# �ý��� ���� ����
echo "==== �ý��� ���� ===="
echo "�ü��: $(uname -a)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "�ھ� ��: $(nproc)"
echo "�޸�: $(free -h | grep Mem | awk '{print $2}')"

# GPU ���� ���� (NVIDIA GPU�� �ִ� ���)
if command -v nvidia-smi &> /dev/null; then
    echo "==== GPU ���� ===="
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# ���� ������Ʈ ���� �Լ�
generate_prompt() {
    local length=$1
    cat /dev/urandom | tr -dc 'a-zA-Z0-9 ' | fold -w "$length" | head -n 1
}

# ��� ���� ���丮 ����
RESULT_DIR="ollama_benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# Docker �����̳� ���� ����
echo "==== Docker �����̳� ���� ===="
docker inspect "$DOCKER_CONTAINER" | jq '.[] | {Name: .Name, Image: .Config.Image, CPUs: .HostConfig.NanoCpus, Memory: .HostConfig.Memory}'

# Ollama �� ���� ���� �� �� ��� ��������
echo "==== Ollama �� ���� ===="
docker exec "$DOCKER_CONTAINER" ollama list

# �� ��� ����
if [ "$MODEL" = "all" ]; then
    # ��� ���� �׽�Ʈ�� ��� �� ��� ����
    MODEL_LIST=$(docker exec "$DOCKER_CONTAINER" ollama list | tail -n +2 | awk '{print $1}')
    echo "�߰ߵ� �� ���: $MODEL_LIST"
else
    # ���� �𵨸� �׽�Ʈ
    MODEL_LIST="$MODEL"
fi

# ���� ��� ���� ����
RESULT_FILE="$RESULT_DIR/benchmark_results.csv"
echo "��,�׽�Ʈ��ȣ,������Ʈ����,�𵨷ε�ð�(��),ù��ū�����ð�(��),�ѻ����ð�(��),��������ū��,��ū�����ӵ�(tokens/sec),CPU����(%),�޸𸮻�뷮(MB),GPU�޸𸮻�뷮(MB)" > "$RESULT_FILE"

echo "==== ��ġ��ũ ���� (�𵨴� $REPEAT ȸ �ݺ�) ===="

# ��� �𵨿� ���� �ݺ�
for CURRENT_MODEL in $MODEL_LIST; do
    echo "===== $CURRENT_MODEL �� ��ġ��ũ ���� ====="

    # �𵨺� ��� ���丮 ����
    MODEL_RESULT_DIR="$RESULT_DIR/$CURRENT_MODEL"
    mkdir -p "$MODEL_RESULT_DIR"

    for i in $(seq 1 $REPEAT); do
        echo "$CURRENT_MODEL �� �׽�Ʈ $i/$REPEAT ���� ��..."
        PROMPT=$(generate_prompt $PROMPT_LENGTH)

        # �����̳� ���� ����͸� ����
        docker stats "$DOCKER_CONTAINER" --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" > "$MODEL_RESULT_DIR/stats_$i.txt" &
        STATS_PID=$!

        # GPU ���� ����͸� (NVIDIA GPU�� �ִ� ���)
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_before_$i.txt"
        fi

        # �� �ε� �ð� �� �߷� ����
        START_TIME=$(date +%s.%N)
        OLLAMA_OUTPUT_FILE="$MODEL_RESULT_DIR/ollama_output_$i.txt"

        # Docker�� ���� Ollama �����ϰ� ��� ����
        docker exec -i "$DOCKER_CONTAINER" bash -c "time ollama run $CURRENT_MODEL \"$PROMPT\" --verbose" > "$OLLAMA_OUTPUT_FILE" 2>&1

        END_TIME=$(date +%s.%N)
        TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    # GPU ��� �� ���� Ȯ��
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$MODEL_RESULT_DIR/gpu_mem_after_$i.txt"
        GPU_MEM_USED=$(cat "$MODEL_RESULT_DIR/gpu_mem_after_$i.txt" | head -1)
    else
        GPU_MEM_USED="N/A"
    fi

    # Docker ���� ����͸� ����
    kill $STATS_PID 2>/dev/null || true

    # ��� �Ľ�
    if [ -f "$OLLAMA_OUTPUT_FILE" ]; then
        # ù ��ū ���� �ð� ���� (�α׿��� ���� ã��)
        LOAD_TIME=$(grep -o "load [0-9.]*ms" "$OLLAMA_OUTPUT_FILE" | awk '{print $2}' | sed 's/ms//' | awk '{print $1/1000}' || echo "N/A")
        FIRST_TOKEN_TIME=$(grep -o "first token [0-9.]*ms" "$OLLAMA_OUTPUT_FILE" | awk '{print $3}' | sed 's/ms//' | awk '{print $1/1000}' || echo "N/A")

        # ������ ��ū �� ���
        TOKENS_GENERATED=$(grep -o "eval [0-9]* tokens" "$OLLAMA_OUTPUT_FILE" | awk '{print $2}' | sort -n | tail -1 || echo "N/A")

        # ��ū ���� �ӵ� ���
        if [ "$TOKENS_GENERATED" != "N/A" ] && [ "$TOTAL_TIME" != "N/A" ]; then
            TOKENS_PER_SEC=$(echo "scale=2; $TOKENS_GENERATED / $TOTAL_TIME" | bc)
        else
            TOKENS_PER_SEC="N/A"
        fi

        # CPU �� �޸� ��뷮 �Ľ�
        if [ -f "$MODEL_RESULT_DIR/stats_$i.txt" ]; then
            CPU_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | cut -d',' -f1 | sed 's/%//')
            MEM_USAGE=$(cat "$MODEL_RESULT_DIR/stats_$i.txt" | cut -d',' -f2 | grep -o "[0-9.]*MiB" | sed 's/MiB//')
        else
            CPU_USAGE="N/A"
            MEM_USAGE="N/A"
        fi

        # ��� ���
        echo "$CURRENT_MODEL,$i,$PROMPT_LENGTH,$LOAD_TIME,$FIRST_TOKEN_TIME,$TOTAL_TIME,$TOKENS_GENERATED,$TOKENS_PER_SEC,$CPU_USAGE,$MEM_USAGE,$GPU_MEM_USED" >> "$RESULT_FILE"
    else
        echo "����: Ollama ��� ������ ã�� �� �����ϴ�."
    fi

    # ��� ���
    sleep 3
    done

    # �𵨺� ��� ���
    echo "==== $CURRENT_MODEL �� ��ġ��ũ �Ϸ� ===="
    echo "$CURRENT_MODEL ���� ��� ���� ��ǥ:"
    awk -F, -v model="$CURRENT_MODEL" '$1==model {load_sum+=$4; first_token_sum+=$5; total_time_sum+=$6; tokens_sum+=$7; token_rate_sum+=$8; count++} END {printf "�� �ε� �ð�: %.2f��\nù ��ū ���� �ð�: %.2f��\n�� ���� �ð�: %.2f��\n��� ���� ��ū ��: %.2f\n��� ��ū ���� �ӵ�: %.2f tokens/sec\n", load_sum/count, first_token_sum/count, total_time_sum/count, tokens_sum/count, token_rate_sum/count}' "$RESULT_FILE"
    echo "----------------------------------------"
done

# ��� ���
echo "==== ��ġ��ũ ��� ��� ===="
echo "����� $RESULT_FILE �� ����Ǿ����ϴ�."

# �� �� ǥ ����
echo "�𵨺� ��� ���� ��:"
echo "------------------------------------------------------"
echo "�𵨸� | �ε�ð�(��) | ù��ū(��) | �ѽð�(��) | ��ū�ӵ�(t/s)"
echo "------------------------------------------------------"

# �𵨺� ��� ����Ͽ� ǥ��
for CURRENT_MODEL in $MODEL_LIST; do
    MODEL_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '$1==model {load_sum+=$4; first_token_sum+=$5; total_time_sum+=$6; tokens_sum+=$7; token_rate_sum+=$8; count++} END {printf "%s | %.2f | %.2f | %.2f | %.2f", model, load_sum/count, first_token_sum/count, total_time_sum/count, token_rate_sum/count}' "$RESULT_FILE")
    echo "$MODEL_AVERAGES"
done
echo "------------------------------------------------------"

# ��� �ð�ȭ (�ɼ�)
if command -v gnuplot &> /dev/null; then
    echo "��� �׷��� ���� ��..."

    # �� ������� �÷� ��� ����
    PLOT_CMD=""
    for CURRENT_MODEL in $MODEL_LIST; do
        if [ -z "$PLOT_CMD" ]; then
            PLOT_CMD="plot '$RESULT_FILE' using 2:8 with linespoints title '$CURRENT_MODEL' smooth bezier"
        else
            PLOT_CMD="$PLOT_CMD, '$RESULT_FILE' using 2:8:(stringcolumn(1) eq '$CURRENT_MODEL' ? 1 : 0) smooth bezier with linespoints title '$CURRENT_MODEL'"
        fi
    done

    # ��ū ���� �ӵ� �׷���
    cat > "$RESULT_DIR/plot_speed.gnu" << EOF
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/token_speed_comparison.png'
set title "�𵨺� ��ū ���� �ӵ� �� (tokens/sec)"
set xlabel "�׽�Ʈ ��ȣ"
set ylabel "��ū/��"
set grid
set key outside
set datafile separator ","

# ���� �׷����� ��ũ��Ʈ (�𵨺� ��� �ӵ�)
set terminal pngcairo size 1200,800 enhanced font 'Verdana,10'
set output '$RESULT_DIR/model_comparison_bar.png'
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
set title "�𵨺� ��� ��ū ���� �ӵ�"
set ylabel "��ū/��"
set grid y
set auto x
plot '$RESULT_FILE' using 8:xtic(1) title "Token Speed (tokens/sec)" group by 1
EOF
    gnuplot "$RESULT_DIR/plot_speed.gnu"

    # �𵨺� ���� ��ǥ �� ���̴� ��Ʈ (������ ���)
    if command -v python3 &> /dev/null; then
        cat > "$RESULT_DIR/radar_chart.py" << EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ��� ���� �б�
df = pd.read_csv('$RESULT_FILE')

# �𵨺� ��� ���
model_stats = df.groupby('��').mean()

# �ʿ��� ��ǥ�� ����
metrics = ['�𵨷ε�ð�(��)', 'ù��ū�����ð�(��)', '�ѻ����ð�(��)', '��ū�����ӵ�(tokens/sec)']
stats = model_stats[metrics]

# ���̴� ��Ʈ �׸���
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# ���� ����
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # �������� �ݱ�

# �� �𵨺��� �׸���
for model in stats.index:
    values = stats.loc[model].values.flatten().tolist()
    values += values[:1]  # �������� �ݱ�
    ax.plot(angles, values, linewidth=1, label=model)
    ax.fill(angles, values, alpha=0.1)

# ��Ʈ ����
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_title('�𵨺� ���� ��')
ax.legend(loc='upper right')

plt.savefig('$RESULT_DIR/model_radar_chart.png')
EOF
        python3 "$RESULT_DIR/radar_chart.py"
    fi

    echo "�׷����� $RESULT_DIR ���丮�� �����Ǿ����ϴ�."
fi

# HTML ���� ����
HTML_REPORT="$RESULT_DIR/benchmark_report.html"
echo "HTML ���� ���� ��..."

cat > "$HTML_REPORT" << EOF
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama ��ġ��ũ ���</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:hover { background-color: #f5f5f5; }
        .chart-container { margin: 20px 0; }
        .model-section { margin: 30px 0; padding: 20px; border: 1px solid #eee; border-radius: 5px; }
        .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
        .note { font-style: italic; color: #666; }
    </style>
</head>
<body>
    <h1>Ollama �� ��ġ��ũ ���</h1>
    <p>�׽�Ʈ �Ͻ�: $(date)</p>

    <div class="summary">
        <h2>�ý��� ����</h2>
        <p>�ü��: $(uname -a)</p>
        <p>CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)</p>
        <p>�ھ� ��: $(nproc)</p>
        <p>�޸�: $(free -h | grep Mem | awk '{print $2}')</p>
        $(if command -v nvidia-smi &> /dev/null; then echo "<p>GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)</p>"; fi)
    </div>

    <h2>�𵨺� ���� ��</h2>

    <table>
        <tr>
            <th>�𵨸�</th>
            <th>�� �ε� �ð�(��)</th>
            <th>ù ��ū ���� �ð�(��)</th>
            <th>�� ���� �ð�(��)</th>
            <th>��� ��ū ��</th>
            <th>��ū ���� �ӵ�(tokens/sec)</th>
        </tr>
EOF

# �𵨺� ��� �����͸� HTML ���̺� �߰�
for CURRENT_MODEL in $MODEL_LIST; do
    MODEL_AVERAGES=$(awk -F, -v model="$CURRENT_MODEL" '
    $1==model {
        load_sum+=$4;
        first_token_sum+=$5;
        total_time_sum+=$6;
        tokens_sum+=$7;
        token_rate_sum+=$8;
        count++
    }
    END {
        printf "<tr><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>",
        model,
        load_sum/count,
        first_token_sum/count,
        total_time_sum/count,
        tokens_sum/count,
        token_rate_sum/count
    }' "$RESULT_FILE")
    echo "$MODEL_AVERAGES" >> "$HTML_REPORT"
done

# HTML ������
cat >> "$HTML_REPORT" << EOF
    </table>

    <div class="chart-container">
        <h2>�ð�ȭ</h2>
        $(if [ -f "$RESULT_DIR/token_speed_comparison.png" ]; then echo "<img src=\"token_speed_comparison.png\" alt=\"��ū ���� �ӵ� ��\" style=\"max-width:100%;\">"; fi)
        $(if [ -f "$RESULT_DIR/model_comparison_bar.png" ]; then echo "<img src=\"model_comparison_bar.png\" alt=\"�𵨺� ��� �ӵ�\" style=\"max-width:100%;\">"; fi)
        $(if [ -f "$RESULT_DIR/model_radar_chart.png" ]; then echo "<img src=\"model_radar_chart.png\" alt=\"�𵨺� ���� ��\" style=\"max-width:100%;\">"; fi)
    </div>

    <div class="note">
        <p>����: �׽�Ʈ�� �� �𵨴� $REPEATȸ �ݺ��Ǿ�����, ������Ʈ ���̴� $PROMPT_LENGTH �ڷ� �����Ǿ����ϴ�.</p>
        <p>���� �ð�: $(date)</p>
    </div>
</body>
</html>
EOF

echo "HTML ������ $HTML_REPORT �� �����Ǿ����ϴ�."
echo "��ġ��ũ �Ϸ�! ����� $RESULT_DIR ���丮���� Ȯ���� �� �ֽ��ϴ�."