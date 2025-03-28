from datasets.packaged_modules.pandas import pandas
from langchain_core.output_parsers import PydanticOutputParser
from pandas import DataFrame
from pydantic import BaseModel, Field

from util.util_common import is_gpt_model

template_1 = """\
You are a helpful language model. Please follow these rules:
1. Output must be valid JSON, matching the schema below exactly (no extra text outside the JSON).
2. Put your chain-of-thought or reasoning in the "reasoning" field. 
3. Provide a short high-level description in the "description" field.
4. Provide the final SQL in the "gen_sql" field.

{format_instructions}

The user request is: "{question}"
The database schema is:
{schema}

Now provide your answer strictly in the JSON format (no extra text!):
SQL to resolve the question:
"""

template = """\
You are a helpful language model. Please follow these rules:
- Output must be valid JSON, matching the schema below exactly (no extra text outside the JSON).
- Provide the final SQL in the "gen_sql" field.

{format_instructions}

The user request is: "{question}"
The database schema is:
{schema}

Now provide your answer strictly in the JSON format (no extra text!):
SQL to resolve the question:
"""
# SQL to resolve the question: "{sql}"


# def make_requests_for_generation(df):
#     prompts = []
#     format_instructions = '''The output should be formatted as a JSON instance that conforms to the JSON schema below.
#
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
#
# Here is the output schema:
# ```
# {"properties": {"reasoning": {"description": "your chain-of-thought or reasoning", "title": "Reasoning", "type": "string"}, "description": {"description": "a short high-level description", "title": "Description", "type": "string"}, "gen_sql": {"description": "the final SQL", "title": "Gen Sql", "type": "string"}}, "required": ["reasoning", "description", "gen_sql"]}
# ```'''
#     for idx, row in df.iterrows():
#         prompts.append(
#             """\
#         You are a helpful language model. Please follow these rules:
#         1. Output must be valid JSON, matching the schema below exactly (no extra text outside the JSON).
#         2. Put your chain-of-thought or reasoning in the "reasoning" field.
#         3. Provide a short high-level description in the "description" field.
#         4. Provide the final SQL in the "sql" field.
# """ + f"""
# {format_instructions}
#
#         The user request is: "{row['question']}"
#         The database schema is:
#         {row['context']}
#
#         Now provide your answer strictly in the JSON format (no extra text!):
#         SQL to resolve the question:
#         """
#         )
#
#     return prompts


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

# """
# ### Instructions:
# Your task is to convert a question into a SQL query, given a Postgres database schema.
# Adhere to these rules:
# - **Deliberately go through the question and database schema word by word** to appropriately answer the question
# - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
# - When creating a ratio, always cast the numerator as float
#
# ### Input:
# Generate a SQL query that answers the question `{question}`.
# This query will run on a database whose schema is represented in this string:
# CREATE TABLE products (
#   product_id INTEGER PRIMARY KEY, -- Unique ID for each product
#   name VARCHAR(50), -- Name of the product
#   price DECIMAL(10,2), -- Price of each unit of the product
#   quantity INTEGER  -- Current quantity in stock
# );
#
# CREATE TABLE customers (
#    customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
#    name VARCHAR(50), -- Name of the customer
#    address VARCHAR(100) -- Mailing address of the customer
# );
#
# CREATE TABLE salespeople (
#   salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
#   name VARCHAR(50), -- Name of the salesperson
#   region VARCHAR(50) -- Geographic sales region
# );
#
# CREATE TABLE sales (
#   sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
#   product_id INTEGER, -- ID of product sold
#   customer_id INTEGER,  -- ID of customer who made purchase
#   salesperson_id INTEGER, -- ID of salesperson who made the sale
#   sale_date DATE, -- Date the sale occurred
#   quantity INTEGER -- Quantity of product sold
# );
#
# CREATE TABLE product_suppliers (
#   supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
#   product_id INTEGER, -- Product ID supplied
#   supply_price DECIMAL(10,2) -- Unit price charged by supplier
# );
#
# -- sales.product_id can be joined with products.product_id
# -- sales.customer_id can be joined with customers.customer_id
# -- sales.salesperson_id can be joined with salespeople.salesperson_id
# -- product_suppliers.product_id can be joined with products.product_id
#
# ### Response:
# Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
# ```sql
# """
sqlcoder_template = """### Instructions:
Your task is to convert a question into a SQL query, given a database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose schema is represented in this string:
{schema}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
```sql
"""

sqlcoder_template_2 = """### Task
Generate a SQL query to answer [QUESTION]{request}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
{sql}"""

k2sql_template = """당신은 SQL을 생성하는 SQL 봇입니다. DDL과 요청사항을 바탕으로 적절한 SQL 쿼리를 생성하세요.

DDL:
{ddl}

요청사항:
{question}

SQL:
{sql}"""


# evaluation_template = """Based on below DDL and Question, evaluate gen_sql can resolve Question.
# If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""
#
# DDL: {schema}
# Question: {question}
# gt_sql: {answer}
# gen_sql: {gen_sql}"""
evaluation_template = """Based on below DDL and Question, evaluate if gen_sql correctly resolves the Question.
If gen_sql and gt_sql produce the same results (functionally equivalent), return "yes" else return "no". 
Note that SQL queries might have different syntax but still return the same results.
Output JSON Format: {{"resolve_yn": ""}}

DDL: {schema}
Question: {question}
gt_sql (ground truth): {gt_sql}
gen_sql (generated): {gen_sql}"""

class SQL(BaseModel):
    reasoning: str = Field(description="your chain-of-thought or reasoning")
    description: str = Field(description="a short high-level description")
    gen_sql: str = Field(description="the final SQL")


sql_parser = PydanticOutputParser(pydantic_object=SQL)


def make_prompt(model: str, data, evaluation: bool = False, only_sql: bool =True):
    if evaluation:
        return evaluation_template.format(schema=data['context'],
                                          question=data['question'],
                                          gt_sql=data['answer'],
                                          gen_sql=data['gen_sql'])
    if model.lower().startswith('sqlcoder'):
        return sqlcoder_template.format(schema=data['context'], question=data['question'])
    else:
        return template.format(schema=data['context'],
                               question=data['question'],
                               format_instructions='' if only_sql else sql_parser.get_format_instructions())


def make_prompts(dataset: DataFrame, model: str, evaluation: bool = False, only_sql: bool =True):
    prompts = []
    for _, data in dataset.iterrows():
        prompts.append(make_prompt(model, data, evaluation, only_sql))

    return prompts

# def make_prompts_for_evaluation(df):
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

def make_request(model: str, prompt: str, evaluation: bool = False, only_sql: bool = False):
    if is_gpt_model(model):
        return {"model": model,
                "response_format": {
                    "type": "json_object"
                },
                "messages": [
                    {"role": "system",
                     "content": prompt}]
                }

    eval_format = {"type": "object",
                   "properties": {
                       "resolve_yn": {"type": "string"}
                   },
                   "required": ["resolve_yn"]
                  }
    trans_format = {"type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "description": {"type": "string"},
                        "gen_sql": {"type": "string"}
                    },
                    "required": ["reasoning", "description", "gen_sql"]
                   }
    only_sql_format = {"type": "object",
                    "properties": {
                        "gen_sql": {"type": "string"}
                    },
                    "required": ["gen_sql"]
                   }
    return {"model": model,
            "stream": False,
            "prompt": prompt,
            "format": eval_format if evaluation else (only_sql_format if only_sql else trans_format)}


def make_request_jobs(model: str, dataset: DataFrame, evaluation: bool = False):
    prompts = make_prompts(dataset, model, evaluation=evaluation, only_sql=True)

    return [make_request(model, prompt, evaluation=evaluation, only_sql=True) for prompt in prompts]