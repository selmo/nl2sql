"""
LLM 프롬프트 템플릿 및 생성 모듈

이 모듈은 다양한 LLM 프롬프트 템플릿을 정의하고,
데이터 및 모델에 적합한 프롬프트를 생성하는 기능을 제공합니다.
"""

import logging
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional
from langchain_core.output_parsers import PydanticOutputParser
from pandas import DataFrame
from pydantic import BaseModel, Field

from utils.config import BatchMode
from utils.common import is_gpt_model

def get_dbms_specific_instructions(dbms):
    """Generate DBMS-specific instructions"""
    if not dbms:
        return ""

    instructions = {
        "PostgreSQL": """
Special Instructions (PostgreSQL):
- Use PostgreSQL syntax
- Use 'YYYY-MM-DD' format for dates
- Use ILIKE for case-insensitive text searches
- Use -> or ->> operators for JSON field access
""",
        "MySQL": """
Special Instructions (MySQL):
- Use MySQL syntax
- Use 'YYYY-MM-DD' format for dates
- Use LIKE for text searches
- Use -> or ->> operators for JSON field access
- Implement pagination using LIMIT and OFFSET
""",
        "Oracle": """
Special Instructions (Oracle):
- Use Oracle syntax
- Use TO_DATE function for date formatting
- Implement result limitation using ROWNUM
- Use DUAL table appropriately
- Use LIKE for text searches
""",
        "SQLite": """
Special Instructions (SQLite):
- Use SQLite syntax
- Use 'YYYY-MM-DD' format for dates
- Consider SQLite's limited function set
- Use basic inner and outer joins instead of advanced join techniques
"""
    }

    # Case-insensitive DBMS search
    for key, value in instructions.items():
        if key.lower() == dbms.lower():
            return value

    return ""  # Return empty string if no matching DBMS


def make_prompt(model: str, data, options=None):
    """
    다양한 모드를 지원하는 확장된 프롬프트 생성 함수

    Args:
        model: 모델 이름
        data: 입력 데이터 (Series 또는 dict)
        batch_mode: 배치 처리 모드 (BatchMode enum)
        options: 추가 옵션 (dict)

    Returns:
        str: 생성된 프롬프트
    """
    if options is None:
        options = {}

    batch_mode = options.get('mode', BatchMode.NL2SQL)
    evaluation = options.get('evaluation', False)
    field_question = options.get('question_column', 'question')
    field_answer = options.get('answer_column', 'answer')
    field_output = options.get('output_column', 'gen_sql')
    field_context = options.get('context_column', 'context')

    # DBMS 정보 추출
    dbms = data.get('dbms', None)

    if isinstance(dbms, list) or isinstance(dbms, np.ndarray):
        if len(dbms) > 2:
            dbms = "General DBMS"
        elif len(dbms) > 1:
            dbms = dbms[0]
        else:
            dbms = None
    else:
        dbms = ""

    # DBMS 특화 지시사항 생성
    dbms_instructions = get_dbms_specific_instructions(dbms)

    logging.info(f'batch_mode: {batch_mode}')

    # NL2SQL 모드
    if batch_mode == BatchMode.NL2SQL or str(batch_mode) == "nl2sql":
        if evaluation:
            return evaluation_template.format(
            # return eval_template_1.format(
                schema=data.get(field_context, ''),
                question=data.get(field_question, ''),
                gt_sql=data.get(field_answer, ''),
                gen_sql=data.get(field_output, ''),
                # dbms=dbms or "SQL"  # 기본값으로 일반 SQL 지정
            )
        else:
            # return sqlcoder_template_2.format(
            #     schema=data.get('context', ''),
            #     question=data.get(field_question, ''),
            # )
            if model.lower().startswith('sqlcoder'):
                return sqlcoder_template_2.format(
                    schema=data.get(field_context, ''),
                    question=data.get(field_question, ''),
                )
            else:
                # 기본 템플릿에 DBMS 정보 추가
                return template_0.format(
                    schema=data.get(field_context, ''),
                    question=data.get(field_question, ''),
                    # format_instructions=options.get('format_instructions', ''),
                    # dbms=dbms or "SQL",  # 기본값으로 일반 SQL 지정
                    # dbms_instructions=dbms_instructions
                )
    # 번역 모드
    elif batch_mode == BatchMode.TRANSLATION:
        input_column = options.get('input_column', 'text')
        text = data.get(input_column, '')

        return translation_template.format(
            question=text,
            answer=data.get('answer', ''),
        )
    else:
        raise ValueError(f"지원되지 않는 모드: {batch_mode}")


def make_prompts(dataset: DataFrame, model: str, options=None):
    """
    데이터셋으로부터 프롬프트 리스트 생성

    Args:
        dataset: 데이터셋
        model: 모델 이름
        options: 추가 옵션

    Returns:
        list: 프롬프트 리스트
    """
    if options is None:
        options = {}

    prompts = []

    for _, data in dataset.iterrows() if hasattr(dataset, 'iterrows') else enumerate(dataset):
        prompt = make_prompt(model, data, options)
        prompts.append(prompt)

    return prompts


def make_request(model: str, prompt: str, options=None):
    """
    다양한 모드를 지원하는 요청 객체 생성 함수

    Args:
        model: 모델 이름
        prompt: 생성된 프롬프트
        batch_mode: 배치 처리 모드
        options: 추가 옵션

    Returns:
        dict: 요청 객체
    """
    if options is None:
        options = {}

    batch_mode = options.get('mode', BatchMode.NL2SQL)

    # JSON 응답 포맷 요구 여부
    requires_json = batch_mode == BatchMode.NL2SQL

    if is_gpt_model(model):
        # o1-mini 모델인 경우와 그 외 모델 구분
        if model.lower().startswith('o1-mini'):
            request = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]  # system 대신 user role 사용
            }
        else:
            request = {
                "model": model,
                "messages": [{"role": "system", "content": prompt}]
            }

        # NL2SQL 모드는 JSON 응답 필요
        if requires_json and not model.lower().startswith('o1-mini'):
            request["response_format"] = {"type": "json_object"}

        return request

    # logging.info('batch_mode: %s', batch_mode)

    # Ollama 모델용 요청 형식
    if batch_mode == BatchMode.NL2SQL:
        if 'evaluation' in options and options['evaluation']:
            # 평가 모드
            format_obj = {
                "type": "object",
                "properties": {
                    "resolve_yn": {"type": "string"}
                },
                "required": ["resolve_yn"]
            }
        elif options.get('only_sql', True):
            # SQL만 반환
            format_obj = {
                "type": "object",
                "properties": {
                    "gen_sql": {"type": "string"}
                },
                "required": ["gen_sql"]
            }
        else:
            # 추론 과정 포함
            format_obj = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "description": {"type": "string"},
                    "gen_sql": {"type": "string"}
                },
                "required": ["reasoning", "description", "gen_sql"]
            }

        return {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "format": format_obj
        }
    elif batch_mode == BatchMode.TRANSLATION:
        return {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "format": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "sql_query": {"type": "string"},
                },
                # "required": ["question"]
                "required": ["question", "sql_query"]
            }
        }
    else:
        # 다른 모드는 일반 텍스트 응답
        return {
            "model": model,
            "stream": False,
            "prompt": prompt
        }


def make_request_jobs(model: str, dataset: DataFrame, options=None):
    """
    데이터셋으로부터 요청 작업 리스트 생성

    Args:
        model: 모델 이름
        dataset: 데이터셋
        mode: 배치 처리 모드
        options: 추가 옵션

    Returns:
        list: 요청 작업 리스트
    """
    if options is None:
        options = {}

    jobs = []

    for _, data in dataset.iterrows() if hasattr(dataset, 'iterrows') else enumerate(dataset):
        task_id = data.get('task_id')
        prompt = make_prompt(model, data, options)
        request = make_request(model, prompt, options)
        jobs.append({ 'task_id' : task_id, 'request': request })

    return jobs


class Template(Enum):
    """배치 처리 모드 열거형"""
    NL2SQL = "nl2sql"  # 자연어를 SQL로 변환
    NL2SQL_SHORT = "nl2sql-short"
    TRANSLATION = "translate"  # 텍스트 번역
    SQLCODER = "sqlcoder"

    def __str__(self):
        return self.value


class SQL(BaseModel):
    reasoning: str = Field(description="your chain-of-thought or reasoning")
    description: str = Field(description="a short high-level description")
    gen_sql: str = Field(description="the final SQL")

sql_parser = PydanticOutputParser(pydantic_object=SQL)

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

template_long = """You are a SQL generator that converts natural language questions into valid SQL queries.

INPUT:
- User question: {question}
- Database schema: {schema}

TASK:
1. Analyze the user's question and the database schema.
2. Generate a valid SQL query that answers the user's question.
3. Return ONLY a valid JSON object matching the format below.

GUIDELINES:
- Ensure the SQL query uses only tables and columns defined in the schema.
- Use appropriate JOINs, WHERE clauses, and aggregate functions as needed.
- If the question cannot be answered with the given schema, provide a helpful error message in the "error" field.

EXAMPLE:
User question: "Show me all employees in the Sales department"
Schema: 
CREATE TABLE employees (
  id INTEGER PRIMARY KEY,
  name VARCHAR(100),
  department_id INTEGER,
  salary DECIMAL(10,2),
  FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE departments (
  id INTEGER PRIMARY KEY,
  name VARCHAR(50)
);

Output:
{{
  "gen_sql": "SELECT e.* FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Sales'"
}}

OUTPUT FORMAT:
{{
  "gen_sql": "<SQL query that answers the user's question>"
}}"""

template = """You are a SQL generator for {dbms}. Convert natural language to SQL.

INPUT:
- DBMS: {dbms}
- Question: {question}
- Schema: {schema}

{dbms_instructions}

RULES:
- Return ONLY valid JSON with the SQL query
- Use only tables and columns from the schema
- No explanations or comments
- Follow {dbms} syntax rules exactly

EXAMPLE:
Question: "Find employees in Sales department"
Schema: 
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100), department_id INTEGER);
CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR(50));

Output:
{{
  "gen_sql": "SELECT e.* FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Sales'"
}}

OUTPUT FORMAT:
{{
  "gen_sql": "<SQL query>"
}}"""

template_0 = """You are a SQL generator. Convert natural language to SQL.

INPUT:
- Question: {question}
- Schema: {schema}

RULES:
- Return ONLY valid JSON with the SQL query
- Use only tables and columns from the schema
- No explanations or comments

EXAMPLE:
Question: "Find employees in Sales department"
Schema: 
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100), department_id INTEGER);
CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR(50));

Output:
{{
  "gen_sql": "SELECT e.* FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Sales'"
}}

OUTPUT FORMAT:
{{
  "gen_sql": "<SQL query>"
}}"""

translation_template = """
You are a professional Korean-to-English translator specializing in database queries.

INPUT:
- Source language: Korean
- Target language: English
- Korean question: {question}
- SQL query containing Korean text: {answer}

TASK:
1. Translate the Korean question to natural, fluent English.
2. In the SQL query, translate ONLY:
   - Korean string literals (text inside quotes: '한국어' → 'English')
   - Korean comments (after -- or between /* */)
   
3. DO NOT modify:
   - SQL keywords (SELECT, FROM, WHERE, etc.)
   - Table names, column names, or database identifiers
   - Numeric values, dates, or special characters
   - SQL syntax structure or operators

4. Ensure the translated question and SQL remain logically consistent.

5. For ambiguous terms, prioritize database/technical context over literal translation.

EXAMPLE:
Input:
{{
  "question": "서울에 있는 상위 5개 식당을 찾아줘",
  "answer": "SELECT name, rating FROM restaurants WHERE city = '서울' ORDER BY rating DESC LIMIT 5 -- 서울의 최고 식당"
}}

Output:
{{
  "question": "Find the top 5 restaurants in Seoul",
  "sql_query": "SELECT name, rating FROM restaurants WHERE city = 'Seoul' ORDER BY rating DESC LIMIT 5 -- Seoul's best restaurants"
}}

OUTPUT FORMAT:
{{
  "question": "<Translated question in English>",
  "sql_query": "<SQL query with translated Korean text>"
}}"""

sqlcoder_template = """
### Instructions:
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
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
"""

k2sql_template = """당신은 SQL을 생성하는 SQL 봇입니다. DDL과 요청사항을 바탕으로 적절한 SQL 쿼리를 생성하세요.

DDL:
{ddl}

요청사항:
{question}

SQL:
{sql}"""

evaluation_template = """Based on below DDL and Question, evaluate if gen_sql correctly resolves the Question.
If gen_sql and gt_sql produce the same results (functionally equivalent), return "yes" else return "no". 
Note that SQL queries might have different syntax but still return the same results.
Output JSON Format: {{"resolve_yn": ""}}

DDL: {schema}
Question: {question}
gt_sql (ground truth): {gt_sql}
gen_sql (generated): {gen_sql}"""

eval_template_1 = """Based on below DDL and Question, evaluate if gen_sql correctly resolves the Question.
If gen_sql and gt_sql produce the same results (functionally equivalent), return "yes" else return "no".
Note that SQL queries might have different syntax but still return the same results.

Consider the following as functionally equivalent:
- Numeric values compared as strings ('1' vs 1)
- Column names with or without quotes when they don't contain spaces
- Different case sensitivity in SQL keywords (SELECT vs select)
- Different spacing or formatting that doesn't affect execution

Output JSON Format: {{"resolve_yn": ""}}

DDL: {schema}
Question: {question}
gt_sql (ground truth): {gt_sql}
gen_sql (generated): {gen_sql}"""

eval_template_2 = """Based on below DDL and Question, evaluate if gen_sql correctly resolves the Question.
If gen_sql and gt_sql would return the same result set when executed against the same database, return "yes" else return "no".

Functional equivalence means:
- Both queries select the same columns from the same tables
- Both queries apply logically equivalent filtering conditions
- Type differences that would be automatically converted (e.g., numeric 1 vs string '1') should be considered equivalent
- Syntactic differences such as quoting identifiers or not should be ignored if they don't affect the result

Output JSON Format: {{"resolve_yn": ""}}

DDL: {schema}
Question: {question}
gt_sql (ground truth): {gt_sql}
gen_sql (generated): {gen_sql}"""

eval_template_3 = """Based on below DDL and Question, evaluate if gen_sql correctly answers the Question.
Focus on whether both queries are trying to retrieve the same information, not on syntactic differences.

For numeric comparisons, treat numeric literals (1) and string literals ('1') as equivalent when comparing with numeric columns.
For identifier quoting, consider "Column" and Column as equivalent notations when they refer to the same column.

If both queries would return the same rows when executed against the same database, return "yes" else return "no".

Output JSON Format: {{"resolve_yn": ""}}

DDL: {schema}
Question: {question}
gt_sql (ground truth): {gt_sql}
gen_sql (generated): {gen_sql}"""


