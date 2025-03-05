import logging

from datasets import load_dataset

import util_common

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from os import path
from pathlib import Path

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

template = """
### Task
Generate a SQL query to answer [QUESTION]{request}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{ddl}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{request}[/QUESTION]
[SQL]
{sql}"""


def prepare_dataset(model, messages):
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=model, verbose=True)
    chain = prompt | model

    for message in messages:
        logging.info("message: %s", message)
        result = chain.invoke(message)
        logging.info("results: %s", result)


def make_request(model, prompts):
    if model.startswith('gpt'):
        jobs = [{"model": model,
                 "response_format": {
                     "type": "json_object"
                 },
                 "messages": [
                     {"role": "system",
                      "content": prompt}
                 ]
                 } for prompt in prompts
                ]
    else:
        jobs = [{"model": model,
                 "stream": False,
                 "messages": [
                     {"role": "system",
                      "content": prompt}
                 ],
                 "format": {
                     "type": "object",
                     "properties": {
                         "sql": {
                             "type": "string"
                         }
                     },
                     "required": [
                         "sql"
                     ]
                 }
                 } for prompt in prompts
                ]
    return jobs


def prepare_test_dataset(model, verifying_model, prefix=''):
    # template = """Question: {question}
    #
    # Answer: Let's think step by step."""
    #
    # prompt = ChatPromptTemplate.from_template(template)
    #
    # model = OllamaLLM(model="deepseek-r1:70b")
    #
    # chain = prompt | model
    #
    # response = chain.invoke({"question": "What is LangChain?"})
    #
    # print(response)

    util_common.check_and_create_directory(prefix)
    requests_filepath = path.join(prefix, "requests.jsonl")
    save_filepath = path.join(prefix, "results.jsonl")
    filepath = path.join(prefix, "saved_results.jsonl")

    if not Path(filepath).exists():
        logging.info(f"File not exists. Creating data file: {filepath}")
        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()
        messages = []
        for idx, row in df.iterrows():
            msg = {'ddl': row['context'], 'request': row['question'], 'sql': ''}
            # logging.info(f"{idx}: {msg}")
            messages.append(msg)
            # prompt = utils.make_prompt(row['context'], row['question'], llm='sqlcoder')
            # df.loc[idx, 'prompt'] = prompt
        prepare_dataset(model, messages)

        # logging.info(f"Data Columns: {df.keys()}")
        # logging.info(f"Data: {df}")
        # # 평가를 위한 requests.jsonl 생성
        # jobs = make_request(model=model, prompts=df['prompt'])
        #
    #     with open(requests_filepath, "w") as f:
    #         for job in jobs:
    #             json_string = json.dumps(job)
    #             f.write(json_string + "\n")
    #
    #     url = "http://172.16.15.112:11434/api/chat"
    #
    #     api_request_parallel_processor.process(
    #         requests_filepath=requests_filepath,
    #         save_filepath=save_filepath,
    #         request_url=url,
    #         max_requests_per_minute=2500,
    #         max_tokens_per_minute=100000,
    #         token_encoding_name="cl100k_base",
    #         max_attempts=10,
    #         logging_level=20
    #     )
    #
    #     base_eval = utils.change_jsonl_to_csv(
    #         save_filepath,
    #         # "prompt",
    #         response_column="sql",
    #         model=verifying_model
    #     )
    #
    #     base_eval.to_json(filepath, orient='records', lines=True)
    #     logging.info(f"Base eval: {base_eval}")
    #
    #     return base_eval
    # else:
    #     logging.info(f"File exists. Loading data file: {filepath}")
    #     df_sql = pd.read_json(filepath, lines=True)
    #     logging.info(f"Data Columns: {df_sql.keys()}")
    #     logging.info(f"Data: {df_sql}")
    #     return df_sql
