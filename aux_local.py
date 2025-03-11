import logging

from langchain_core.output_parsers import SimpleJsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, model_validator

import util_common
from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
from os import path
from pathlib import Path

import re

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# template = """
# ### Task
# Generate a SQL query to answer.
# {format_instructions}
# [QUESTION]{request}[/QUESTION]
#
# ### Database Schema
# The query will run on a database with the following schema:
# {ddl}
#
# ### Answer
# Given the database schema, here is the SQL query that [QUESTION]{request}[/QUESTION]
# [SQL]
# {sql}"""
template = """\
You are a helpful language model. Please follow these rules:
1. Output must be valid JSON, matching the schema below exactly (no extra text outside the JSON).
2. Put your chain-of-thought or reasoning in the "reasoning" field. 
3. Provide a short high-level description in the "description" field.
4. Provide the final SQL in the "sql" field.

{format_instructions}

The user request is: "{request}"
The database schema is:
{ddl}

Now provide your answer strictly in the JSON format (no extra text!):
SQL to resolve the question: "{sql}"
"""


def prepare_dataset(model, messages):
    model = OllamaLLM(model=model, temperature=0.0)

    class SQL(BaseModel):
        reasoning: str = Field(description="your chain-of-thought or reasoning")
        description: str = Field(description="a short high-level description")
        sql: str = Field(description="the final SQL")

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=SQL)

    prompt = PromptTemplate(
        template=template, #"Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["request", "ddl", "sql"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model
    for message in messages:
        logging.info("message: %s", message)
        output = chain.invoke(message)
        logging.info("output: %s", output)

        clean_output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        result = parser.parse(clean_output)

        # for chunk in chain.stream(message):
        #     print(chunk, end="|", flush=True)
        logging.info("result: %s", result)


def prepare_dataset_test(model):
    model = OllamaLLM(model=model, temperature=0.0)

    # Define your desired data structure.
    class Joke(BaseModel):
        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

        # You can add custom validation logic easily with Pydantic.
        @model_validator(mode="before")
        @classmethod
        def question_ends_with_question_mark(cls, values: dict) -> dict:
            setup = values.get("setup")
            if setup and setup[-1] != "?":
                raise ValueError("Badly formed question!")
            return values

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=Joke)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    logging.info('prompt: %s', parser.get_format_instructions())

    # And a query intended to prompt a language model to populate the data structure.
    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"query": "Tell me a joke."})
    logging.info('output: %s', output)
    result = parser.invoke(output)
    logging.info('result: %s', result)


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
        # prepare_dataset(model, messages)
        prepare_dataset(model, [{'ddl': df['context'][0], 'request': df['question'][0], 'sql': ''}])
        # prepare_dataset_test(model)

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
