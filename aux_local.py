import json
import pandas as pd
import logging
import re

from langchain_core.exceptions import OutputParserException

import api_request_parallel_processor
from util_common import clean_filepath, check_and_create_directory
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator

from datasets import load_dataset
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from os import path
from pathlib import Path

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


class SQL(BaseModel):
    reasoning: str = Field(description="your chain-of-thought or reasoning")
    description: str = Field(description="a short high-level description")
    gen_sql: str = Field(description="the final SQL")


def prepare_model_and_parser(model_id):
    model = OllamaLLM(model=model_id, temperature=0.0)

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=SQL)

    prompt = PromptTemplate(
        template=template,
        input_variables=["request", "ddl", "sql"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model

    return chain, parser


def llm_invoke(model, prompt):
    # logging.info("prompt: %s", prompt)
    output = model.invoke({'ddl': prompt['context'], 'request': prompt['question'], 'sql': ''})
    # logging.info("response: %s", output)

    return output


def extract_sql_queries(text: str) -> str:
    # 1) ```sql ... ``` 패턴
    pattern_triple_backticks = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL)

    # 2) \boxed{ ... } 패턴
    pattern_boxed = re.compile(r"\\boxed\s*\{\s*(.*?)\s*\}", re.DOTALL)

    # (1) triple backticks 안의 SQL 추출
    matches_backticks = pattern_triple_backticks.findall(text)
    if matches_backticks:
        return matches_backticks[0].strip()

    # (2) \boxed 안의 SQL 추출
    matches_boxed = pattern_boxed.findall(text)
    if matches_boxed:
        return matches_boxed[0].strip()

    return ""


def clean_response(parser, response):
    clean_output = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    try:
        result = parser.parse(clean_output)
        logging.info("result: %s\n", result)
        return result
    except OutputParserException as e:
        logging.error("response: %s", clean_output)
        query_match = extract_sql_queries(clean_output)
        logging.info("query: %s", query_match)
        sql_obj = SQL(
            reasoning="",
            description="Parsing error.",
            gen_sql=query_match
        )
        logging.error("error msg: %s\n", e)
        return sql_obj


def prepare_dataset(model, parser, messages):
    results = []
    for message in messages:
        logging.info("message: %s", message)
        output = model.invoke(message)
        # logging.info("output: %s", output)

        clean_output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        result = parser.parse(clean_output)
        results.append(result)
        logging.info("result: %s", result.gen_sql)

    return results


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


def prepare_test_dataset(model, prefix=''):
    check_and_create_directory(prefix)
    filepath = path.join(prefix, "saved_results.jsonl")

    if not Path(filepath).exists():
        logging.info(f"File not exists. Creating data file: {filepath}")
        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()

        model, parser = prepare_model_and_parser(model)

        for idx, row in df.iterrows():
            gen_sql = llm_invoke(model, row)
            cleaned_sql = clean_response(parser, gen_sql)

            logging.info(f'cleaned sql #{idx}: {cleaned_sql.gen_sql}')

            df.loc[idx, 'gen_sql'] = cleaned_sql.gen_sql

        logging.info(f"Data Columns: {df.keys()}")
        logging.info(f"Data: {df}")

        df.to_json(filepath, orient='records', lines=True)
        logging.info(f"File saved: {filepath}")
    else:
        logging.info(f"File exists. Loading data file: {filepath}")
        df = pd.read_json(filepath, lines=True)
        logging.info(f"Data Columns: {df.keys()}")
        logging.info(f"Data: {df}")
        logging.info("File loaded.")

    return df


def make_requests_for_generation(df, directory='requests'):
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True)
    prompts = []
    format_instructions = '''The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"reasoning": {"description": "your chain-of-thought or reasoning", "title": "Reasoning", "type": "string"}, "description": {"description": "a short high-level description", "title": "Description", "type": "string"}, "gen_sql": {"description": "the final SQL", "title": "Gen Sql", "type": "string"}}, "required": ["reasoning", "description", "gen_sql"]}
```'''
    for idx, row in df.iterrows():
        prompts.append(
            """\
        You are a helpful language model. Please follow these rules:
        1. Output must be valid JSON, matching the schema below exactly (no extra text outside the JSON).
        2. Put your chain-of-thought or reasoning in the "reasoning" field. 
        3. Provide a short high-level description in the "description" field.
        4. Provide the final SQL in the "sql" field.
""" + f"""
{format_instructions}

        The user request is: "{row['question']}"
        The database schema is:
        {row['context']}

        Now provide your answer strictly in the JSON format (no extra text!):
        SQL to resolve the question: 
        """
        )

    return prompts


# def make_requests_for_evaluation(df, directory='requests'):
#     if not Path(directory).exists():
#         Path(directory).mkdir(parents=True)
#     prompts = []
#     for idx, row in df.iterrows():
#         prompts.append(
#             """Based on below DDL and Question, evaluate gen_sql can resolve Question.
# If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" +
#             f"""
#
# DDL: {row['context']}
# Question: {row['question']}
# gt_sql: {row['answer']}
# gen_sql: {row['gen_sql']}"""
#         )
#
#     return prompts

def make_request_jobs(model, prompts):
    jobs = [{"model": model,
             "stream": False,
             "messages": [
                 {"role": "system",
                  "content": prompt}
             ],
             "format": {
                 "type": "object",
                 "properties": {
                     "reasoning": {
                         "type": "string"
                     },
                     "description": {
                         "type": "string"
                     },
                     "gen_sql": {
                         "type": "string"
                     }
                 },
                 "required": [
                     "reasoning", "description", "gen_sql"
                 ]
             }
             } for prompt in prompts
            ]

    return jobs

def prepare_test_ollama(model, prefix=''):
    check_and_create_directory(prefix)
    filepath = path.join(prefix, "saved_results.jsonl")

    if not Path(filepath).exists():
        requests_path = path.join(prefix, 'requests')
        results_path = path.join(prefix, 'results')
        requests_filepath = clean_filepath(f'{model}.req', prefix=requests_path)
        save_filepath = clean_filepath(f'{model}.sav', prefix=results_path)
        output_file = clean_filepath(f"{model}.csv", prefix=results_path)
        check_and_create_directory(path.dirname(requests_filepath))
        check_and_create_directory(path.dirname(save_filepath))
        check_and_create_directory(path.dirname(output_file))

        logging.info(f"File not exists. Creating data file: {filepath}")
        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()

        # 평가를 위한 requests.jsonl 생성
        prompts = make_requests_for_generation(df, directory=requests_path)
        jobs = make_request_jobs(model, prompts)

        with open(requests_filepath, "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")

        url = "https://api.openai.com/v1/chat/completions" if model.lower().startswith(
            'gpt') else "http://172.16.15.112:11434/api/chat"

        api_request_parallel_processor.process(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=url,
            max_requests_per_minute=2500,
            max_tokens_per_minute=100000,
            token_encoding_name="cl100k_base",
            max_attempts=10,
            logging_level=20
        )

        prompts = []
        responses = []
        with open(save_filepath, 'r') as json_file:
            for data in json_file:
                json_data = json.loads(data)

                logging.info(f"json_data: {json_data}")
                prompts.append(json_data[0]['messages'][0]['content'])
                responses.append(json_data[1]['message']['content'])

        dfs = pd.DataFrame({"prompt": prompts, "response": responses})

        dfs.to_json(filepath, orient='records', lines=True)
        logging.info(f"File saved: {filepath}")
    else:
        logging.info(f"File exists. Loading data file: {filepath}")
        dfs = pd.read_json(filepath, lines=True)
        logging.info("File loaded.")

    results_path = path.join(prefix, 'results')
    save_filepath = path.join(results_path, f'{model}.sav')
    logging.info('loading file: %s', save_filepath)
    data = pd.read_csv(save_filepath)
    logging.info('keys: %s', data.keys())
    logging.info('data: %s', data)

    return dfs
