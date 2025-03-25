import pandas as pd
import logging
import re
import json

import util_common
# from api_request_parallel_processor import process_in_memory, llm_invoke_parallel_langchain
from aux_common import make_inference_pipeline
from llms import prompt_generator
from llms.ollama_api import llm_invoke_parallel, llm_invoke_jobs_parallel
from llms.prompt_generator import sql_parser, SQL, make_prompt
from llms.response_processor import make_result, clean_response
from util_common import clean_filepath, check_and_create_directory, make_requests_for_evaluation
from langchain_core.output_parsers import PydanticOutputParser

from datasets import load_dataset
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from os import path
from pathlib import Path

from utils import change_jsonl_to_csv, change_responses_to_csv

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def prepare_model(model_id):
    model = OllamaLLM(model=model_id, temperature=0.0, base_url='172.16.15.112')

    prompt = PromptTemplate(
        template=prompt_generator.template,
        input_variables=["request", "ddl", "sql"],
        partial_variables={"format_instructions": sql_parser.get_format_instructions()},
    )
    chain = prompt | model

    return chain


def llm_invoke(model, prompt):
    # logging.info("prompt: %s", prompt)
    output = model.invoke({'ddl': prompt['context'], 'request': prompt['question'], 'sql': ''})
    # logging.info("response: %s", output)

    return output


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


# def prepare_dataset_test(model):
#     model = OllamaLLM(model=model, temperature=0.0)
#
#     # Define your desired data structure.
#     class Joke(BaseModel):
#         setup: str = Field(description="question to set up a joke")
#         punchline: str = Field(description="answer to resolve the joke")
#
#         # You can add custom validation logic easily with Pydantic.
#         @model_validator(mode="before")
#         @classmethod
#         def question_ends_with_question_mark(cls, values: dict) -> dict:
#             setup = values.get("setup")
#             if setup and setup[-1] != "?":
#                 raise ValueError("Badly formed question!")
#             return values
#
#     # Set up a parser + inject instructions into the prompt template.
#     parser = PydanticOutputParser(pydantic_object=Joke)
#
#     prompt = PromptTemplate(
#         template="Answer the user query.\n{format_instructions}\n{query}\n",
#         input_variables=["query"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )
#
#     logging.info('prompt: %s', parser.get_format_instructions())
#
#     # And a query intended to prompt a language model to populate the data structure.
#     prompt_and_model = prompt | model
#     output = prompt_and_model.invoke({"query": "Tell me a joke."})
#     logging.info('output: %s', output)
#     result = parser.invoke(output)
#     logging.info('result: %s', result)


# def prepare_test_dataset_parallel(model, prefix=''):
#     """병렬 처리를 사용한 테스트 데이터셋 준비"""
#     check_and_create_directory(prefix)
#     filepath = path.join(prefix, "saved_results.jsonl")
#
#     if not Path(filepath).exists():
#         logging.info(f"파일이 존재하지 않습니다. 데이터 파일 생성 중: {filepath}")
#
#         # 데이터셋 불러오기
#         df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
#         df = df.to_pandas()
#         df = df[:50]  # 필요한 경우 샘플 수 조정
#
#         # 프롬프트 목록 생성
#         prompts = []
#         for _, row in df.iterrows():
#             prompts.append(row)
#
#         # 결과 처리
#         parser = PydanticOutputParser(pydantic_object=SQL)
#
#         # 병렬 호출 실행
#         results = llm_invoke_parallel(model, prompts, template, parser)
#
#         for idx, result in enumerate(results):
#             try:
#                 if 'message' in result and 'content' in result['message']:
#                     content = result['message']['content']
#                     cleaned_result = clean_response(parser, content)
#                     df.loc[idx, 'gen_sql'] = cleaned_result.gen_sql
#                 elif 'error' in result:
#                     logging.error(f"오류 응답: {result['error']}")
#                     df.loc[idx, 'gen_sql'] = ''
#                 else:
#                     logging.warning(f"예상치 못한 응답 형식: {result}")
#                     df.loc[idx, 'gen_sql'] = ''
#             except Exception as e:
#                 logging.error(f"결과 처리 중 오류 발생: {str(e)}")
#                 df.loc[idx, 'gen_sql'] = ''
#
#         # 결과 저장
#         df.to_json(filepath, orient='records', lines=True)
#         logging.info(f"파일 저장 완료: {filepath}")
#     else:
#         logging.info(f"파일이 존재합니다. 데이터 파일 로딩 중: {filepath}")
#         df = pd.read_json(filepath, lines=True)
#         logging.info(f"데이터 컬럼: {df.keys()}")
#         logging.info(f"데이터: {df}")
#         logging.info("파일 로딩 완료.")
#
#     return df
def prepare_test_dataset_origin(model, prefix='', test_size=0):
    df_filepath = path.join(prefix, f"{model}.jsonl")
    util_common.check_and_create_directory(path.dirname(df_filepath))

    if path.exists(df_filepath):
        logging.info("File exists. Loading file: %s", df_filepath)
        df = pd.read_json(df_filepath, lines=True)
        logging.info(f"Data Columns: {df.keys()}")
        logging.info(f"Data: {df}")
    else:
        logging.info(f"File not exists. Creating data file: {df_filepath}")
        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()
        if test_size > 0:
            df = df[:test_size]
        for idx, row in df.iterrows():
            prompt = make_prompt(model, row['context'], row['question'])
            df.loc[idx, 'prompt'] = prompt

        # sql 생성
        hf_pipe = make_inference_pipeline(model)

        gen_sqls = hf_pipe(
            df['prompt'].tolist(),
            do_sample=False,
            return_full_text=False,
            # max_length=512,
            truncation=True
        )
        gen_sqls = [x[0][('generated_text')] for x in gen_sqls]
        df['gen_sql'] = gen_sqls
        logging.info(f"Data Columns: {df.keys()}")
        logging.info(f"Data: {df}")
        df.to_json(df_filepath, orient='records', lines=True)
        logging.info(f"Data file saved: {df_filepath}")

    return df


def prepare_test_dataset(model, prefix='', batch_size=10, max_concurrent=10, max_retries=3, test_size=0, ollama_url=None):
    """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""
    check_and_create_directory(prefix)
    filepath = path.join(prefix, "saved_results.jsonl")

    # 로그 파일 설정
    log_filepath = path.join(prefix, "parallel_processing.log")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logging.info(f"병렬 처리 로그를 {log_filepath}에 기록합니다.")

    if not Path(filepath).exists():
        logging.info(f"파일이 존재하지 않습니다. 데이터 파일 생성 중: {filepath}")

        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()
        if test_size > 0:
            df = df[:test_size]

        # 프롬프트 목록 생성
        datasets = []
        for _, row in df.iterrows():
            datasets.append(row)

        # 병렬 호출 실행
        logging.info(f"총 {len(datasets)}개 데이터셋에 대한 병렬 처리를 시작합니다.")
        responses = llm_invoke_parallel(
            model,
            datasets,
            batch_size=batch_size,
            max_retries=max_retries,
            max_concurrent=max_concurrent,
            url=ollama_url
        )

        # 결과 처리
        results, success_count, error_count = make_result(responses, df)

        logging.info(f"결과 처리 완료: 성공 {success_count}개, 오류 {error_count}개")

        # 결과 저장
        results.to_json(filepath, orient='records', lines=True)
        logging.info(f"파일 저장 완료: {filepath}")
    else:
        logging.info(f"파일이 존재합니다. 데이터 파일 로딩 중: {filepath}")
        results = pd.read_json(filepath, lines=True)
        logging.info(f"데이터 컬럼: {results.keys()}")
        logging.info("파일 로딩 완료.")

    # 로그 핸들러 제거
    root_logger.removeHandler(file_handler)

    return results


def make_requests_for_generation(df):
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


# def prepare_test_ollama(model, prefix='', test_size=0):
#     check_and_create_directory(prefix)
#     filepath = path.join(prefix, "saved_results.jsonl")
#
#     if not Path(filepath).exists():
#         requests_path = path.join(prefix, 'requests')
#         results_path = path.join(prefix, 'results')
#         requests_filepath = clean_filepath(f'{model}.req', prefix=requests_path)
#         save_filepath = clean_filepath(f'{model}.sav', prefix=results_path)
#         output_file = clean_filepath(f"{model}.csv", prefix=results_path)
#         check_and_create_directory(path.dirname(requests_filepath))
#         check_and_create_directory(path.dirname(save_filepath))
#         check_and_create_directory(path.dirname(output_file))
#
#         logging.info(f"File not exists. Creating data file: {filepath}")
#         # 데이터셋 불러오기
#         testset = load_dataset("shangrilar/ko_text2sql", "origin")['test']
#         testset = testset.to_pandas()
#         if test_size > 0:
#             testset = testset[:test_size]
#         logging.info('column: %s', testset.keys())
#
#         if not Path(requests_path).exists():
#             Path(requests_path).mkdir(parents=True)
#
#         # 평가를 위한 requests.jsonl 생성
#         prompts = make_requests_for_generation(testset)
#         jobs = make_request_jobs(model, prompts)
#         testset['prompt'] = prompts
#         testset['job'] = jobs
#
#         url = "http://172.16.15.112:11434/api/chat"
#
#         responses = process_in_memory(
#             dataset=testset,
#             request_url=url,
#             max_requests_per_minute=50,
#             max_tokens_per_minute=100000,
#             token_encoding_name="cl100k_base",
#             max_attempts=10,
#             logging_level=20,
#             max_concurrent_requests=10,
#             request_timeout=None,
#             overall_timeout=None)
#
#         responses.to_json(filepath, orient='records', lines=True)
#         logging.info(f"File saved: {filepath}")
#     else:
#         logging.info(f"File exists. Loading data file: {filepath}")
#         responses = pd.read_json(filepath, lines=True)
#         logging.info("Colums: %s", responses.keys())
#         logging.info("File loaded.")
#
#     # results_path = path.join(prefix, 'results')
#     # save_filepath = path.join(results_path, f'{model}.sav')
#     # logging.info('loading file: %s', save_filepath)
#     # data = pd.read_csv(save_filepath)
#     # logging.info('keys: %s', data.keys())
#     # logging.info('data: %s', data)
#
#     return responses
#
# def prepare_test_dataset_langchain(model, prefix='', batch_size=10, max_concurrent=10, max_retries=3, test_size=0):
#     """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""
#     check_and_create_directory(prefix)
#     filepath = path.join(prefix, "saved_results.jsonl")
#
#     # 로그 파일 설정
#     log_filepath = path.join(prefix, "parallel_processing.log")
#     file_handler = logging.FileHandler(log_filepath)
#     file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
#
#     root_logger = logging.getLogger()
#     root_logger.addHandler(file_handler)
#     root_logger.setLevel(logging.INFO)
#
#     logging.info(f"병렬 처리 로그를 {log_filepath}에 기록합니다.")
#
#     if not Path(filepath).exists():
#         logging.info(f"파일이 존재하지 않습니다. 데이터 파일 생성 중: {filepath}")
#
#         # 데이터셋 불러오기
#         df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
#         df = df.to_pandas()
#         if test_size > 0:
#             df = df[:test_size]
#
#         # 프롬프트 목록 생성
#         prompts = []
#         for _, row in df.iterrows():
#             prompts.append(row)
#
#         # 병렬 호출 실행
#         logging.info(f"총 {len(prompts)}개 프롬프트에 대한 병렬 처리를 시작합니다.")
#         results = llm_invoke_parallel_langchain(
#             model,
#             prompts,
#             prompt_generator.template,
#             batch_size=batch_size,
#             max_retries=max_retries,
#             max_concurrent=max_concurrent
#         )
#
#         # 결과 처리
#         success_count = 0
#         error_count = 0
#
#         for idx, result in enumerate(results):
#             # logging.info("Result: [%s] %s", type(result), result)
#             try:
#                 if result and 'response' in result:
#                     content = result['response']
#                     # logging.info("content: %s", content)
#                     cleaned_result = clean_response(content, sql_parser)
#                     df.loc[idx, 'gen_sql'] = cleaned_result.gen_sql
#                     logging.info("df.loc[idx, 'gen_sql'] = cleaned_result")
#                     success_count += 1
#                 elif result and 'error' in result:
#                     logging.error(f"오류 응답 (인덱스 {idx}): {result['error']}")
#                     df.loc[idx, 'gen_sql'] = ''
#                     error_count += 1
#                 else:
#                     logging.warning(f"예상치 못한 응답 형식 (인덱스 {idx}): {result}")
#                     df.loc[idx, 'gen_sql'] = ''
#                     error_count += 1
#             except Exception as e:
#                 logging.error(f"결과 처리 중 오류 발생 (인덱스 {idx}): [{type(e)}] {str(e)}")
#                 df.loc[idx, 'gen_sql'] = ''
#                 error_count += 1
#
#         logging.info(f"결과 처리 완료: 성공 {success_count}개, 오류 {error_count}개")
#
#         # 결과 저장
#         df.to_json(filepath, orient='records', lines=True)
#         logging.info(f"파일 저장 완료: {filepath}")
#     else:
#         logging.info(f"파일이 존재합니다. 데이터 파일 로딩 중: {filepath}")
#         df = pd.read_json(filepath, lines=True)
#         logging.info(f"데이터 컬럼: {df.keys()}")
#         logging.info("파일 로딩 완료.")
#
#     # 로그 핸들러 제거
#     root_logger.removeHandler(file_handler)
#
#     return df


def evaluation_api(model, dataset, prefix='', batch_size=10, max_concurrent=10, max_retries=3, size=None, api_key=""):
    """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""
    check_and_create_directory(prefix)
    filepath = path.join(prefix, "text2sql.jsonl")
    requests_path = path.join(prefix, 'requests')
    if not Path(requests_path).exists():
        Path(requests_path).mkdir(parents=True)
    requests_filepath = clean_filepath(filepath, prefix=requests_path)
    check_and_create_directory(path.dirname(requests_filepath))

    prompts = make_requests_for_evaluation(dataset)
    jobs = util_common.make_request_jobs(model, prompts)

    with open(requests_filepath, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

    # 로그 파일 설정
    log_filepath = path.join(prefix, "parallel_processing_eval.log")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logging.info(f"병렬 처리 로그를 {log_filepath}에 기록합니다.")

    url = "https://api.openai.com/v1/chat/completions" if model.lower().startswith(
        'gpt') or model.startswith('o1') or model.startswith('o3') else "http://172.16.15.112:11434/api/chat"

    # 데이터셋 불러오기
    if size is not None and size > 0:
        jobs = jobs[:size]

    # 병렬 호출 실행
    logging.info(f"총 {len(jobs)}개 데이터셋에 대한 병렬 처리를 시작합니다.")
    responses = llm_invoke_jobs_parallel(
        model,
        jobs,
        url,
        batch_size=batch_size,
        max_retries=max_retries,
        max_concurrent=max_concurrent
    )

    output_file = path.join(prefix, "result.csv")
    base_eval = change_responses_to_csv(
        responses,
        output_file,
        # "prompt",
        response_column="resolve_yn",
        model=model
    )

    base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]

    logging.info("Evaluation CSV:\n%s", base_eval)
    logging.info("Number of correct answers: %s", num_correct_answers)




    # eval_filepath = "text2sql.jsonl"
    #
    # logging.info("DataFrame:\n%s", dataset)
    # logging.info("Evaluation file path: %s", eval_filepath)
    #
    # requests_path = path.join(prefix, 'requests')
    # results_path = path.join(prefix, 'results')
    # requests_filepath = clean_filepath(eval_filepath, prefix=requests_path)
    # save_filepath = clean_filepath(eval_filepath, prefix=results_path)
    # output_file = clean_filepath(f"{ft_model}.csv", prefix=results_path)
    # check_and_create_directory(path.dirname(requests_filepath))
    # check_and_create_directory(path.dirname(save_filepath))
    # check_and_create_directory(path.dirname(output_file))
    #
    # # 평가를 위한 requests.jsonl 생성
    # prompts = make_requests_for_evaluation(dataset, directory=requests_path)
    #
    # jobs = make_request_jobs(verifying_model, prompts)
    #
    # with open(requests_filepath, "w") as f:
    #     for job in jobs:
    #         json_string = json.dumps(job)
    #         f.write(json_string + "\n")
    #
    # url = "https://api.openai.com/v1/chat/completions" if verifying_model.lower().startswith(
    #     'gpt') or verifying_model.startswith('o1') or verifying_model.startswith('o3') else "http://172.16.15.112:11434/api/chat"
    #
    # logging.info('URL: %s', url)
    # api_request_parallel_processor.process_by_file(
    #     requests_filepath=requests_filepath,
    #     save_filepath=save_filepath,
    #     request_url=url,
    #     api_key=api_key,
    #     max_requests_per_minute=2500,
    #     max_tokens_per_minute=100000,
    #     token_encoding_name="cl100k_base",
    #     max_attempts=10,
    #     logging_level=20
    # )
    # base_eval = change_jsonl_to_csv(
    #     save_filepath,
    #     output_file,
    #     # "prompt",
    #     response_column="resolve_yn",
    #     model=verifying_model
    # )
    #
    # base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    # num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]
    #
    # logging.info("Evaluation CSV:\n%s", base_eval)
    # logging.info("Number of correct answers: %s", num_correct_answers)