import json
import logging
import pandas as pd


# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def change_responses_to_csv(responses, output_file='', prompt_column="prompt", response_column="response", model="gpt"):
    prompts = []
    responses = []

    for json_data in responses:
        # logging.debug(f"json_data: {json_data}")
        prompts.append(json_data[0]['messages'][0]['content'])
        if model.lower().startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
            responses.append(json_data[1]['choices'][0]['message']['content'])
        else:
            responses.append(json_data[1]['message']['content'])

    dfs = pd.DataFrame({prompt_column: prompts, response_column: responses})
    if not output_file == '':
        dfs.to_csv(output_file, index=False)
    return dfs


def change_jsonl_to_csv(input_file, output_file='', prompt_column="prompt", response_column="response", model="gpt"):
    prompts = []
    responses = []

    logging.info(f"change_jsonl_to_csv: input_file={input_file}")
    with open(input_file, 'r') as json_file:
        for data in json_file:
            json_data = json.loads(data)

            # logging.info(f"json_data: {json_data}")
            # prompts.append(json_data[0]['messages'][0]['content'])
            prompts.append(json_data[0]['prompt'])
            if model.lower().startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
                responses.append(json_data[1]['choices'][0]['message']['content'])
            else:
                # responses.append(json_data[1]['message']['content'])
                responses.append(json_data[1]['response'])

    dfs = pd.DataFrame({prompt_column: prompts, response_column: responses})
    logging.info(f"change_jsonl_to_csv: input_file={input_file}, output_file={output_file}")
    if not output_file == '':
        dfs.to_csv(output_file, index=False)
    return dfs


def merge_gt_and_gen_result(df_gt, df_gen):
    results = []
    for idx, row in df_gen.iterrows():
        with_sql_gt = df_gt.loc[df_gt['without_sql'] == row['without_sql']] 
        gt_sql = with_sql_gt['sql'].values[0]
        gen_sql = row['gen_sql']
        results.append((with_sql_gt['ddl'].values[0], with_sql_gt['request'].values[0], gt_sql, gen_sql))
    df_result = pd.DataFrame(results, columns=["ddl", "request", "gt_sql", "gen_sql"])
    return df_result

def make_evaluation_requests(df, filename, model="gpt-4-1106-preview"):
    prompts = []
    for idx, row in df.iterrows():
        prompts.append(f"""Based on provided ddl, request, gen_sql, ground_truth_sql if gen_sql eqauls to ground_truth_sql, output "yes" else "no"
DDL:
{row['ddl']}
Request:
{row['request']}
ground_truth_sql:
{row['gt_sql']}
gen_sql:
{row['gen_sql']}

Answer:""")

    jobs = [{"model": model, "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]
    with open(filename, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

def make_prompt(ddl, request, sql="", llm="common"):
    if llm == "sqlcoder":
        prompt = f"""
            ### Task
            Generate a SQL query to answer [QUESTION]{request}[/QUESTION]

            ### Database Schema
            The query will run on a database with the following schema:
            {ddl}

            ### Answer
            Given the database schema, here is the SQL query that [QUESTION]{request}[/QUESTION]
            [SQL]
            {sql}"""
    else:
        prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL과 요청사항을 바탕으로 적절한 SQL 쿼리를 생성하세요.

    DDL:
    {ddl}

    요청사항:
    {request}

    SQL:
    {sql}"""

    return prompt


def load_csv(filepath):
    base_eval = pd.read_csv(filepath)
    base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])

    return base_eval


if __name__ == '__main__':
    df = pd.read_csv('./nl2sql_validation.csv')
    df.sample(100).to_csv('nl2sql_validation_sample.csv', index=False)