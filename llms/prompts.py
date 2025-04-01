from enum import Enum


class Template(Enum):
    """배치 처리 모드 열거형"""
    NL2SQL = "nl2sql"  # 자연어를 SQL로 변환
    NL2SQL_SHORT = "nl2sql-short"
    TRANSLATION = "translate"  # 텍스트 번역
    SQLCODER = "sqlcoder"

    def __str__(self):
        return self.value

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

# template = """\
# You are a helpful language model. Please follow these rules:
# - Output must be valid JSON, matching the schema below exactly (no extra text outside the JSON).
# - Provide the final SQL in the "gen_sql" field.
#
# {format_instructions}
#
# The user request is: "{question}"
# The database schema is:
# {schema}
#
# Now provide your answer strictly in the JSON format (no extra text!):
# SQL to resolve the question:
# """

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

# template = """You are a SQL generator. Convert natural language to SQL.
#
# INPUT:
# - Question: {question}
# - Schema: {schema}
#
# RULES:
# - Return ONLY valid JSON with the SQL query
# - Use only tables and columns from the schema
# - No explanations or comments
#
# EXAMPLE:
# Question: "Find employees in Sales department"
# Schema:
# CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100), department_id INTEGER);
# CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR(50));
#
# Output:
# {{
#   "gen_sql": "SELECT e.* FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Sales'"
# }}
#
# OUTPUT FORMAT:
# {{
#   "gen_sql": "<SQL query>"
# }}"""

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


evaluation_template = """Based on below DDL and Question, evaluate if gen_sql correctly resolves the Question.
If gen_sql and gt_sql produce the same results (functionally equivalent), return "yes" else return "no". 
Note that SQL queries might have different syntax but still return the same results.
Output JSON Format: {{"resolve_yn": ""}}

DDL: {schema}
Question: {question}
gt_sql (ground truth): {gt_sql}
gen_sql (generated): {gen_sql}"""


# 번역 모드용 템플릿
# translation_template = """\
# You are a professional translator. Please follow these rules:
# - Translate the following text from {source_lang} to {target_lang}.
# - This is a text2sql query related to the database schema provided below.
# - DO NOT translate any table names, column names, or database values mentioned in the schema.
# - Return your response as a valid JSON object with the translation in the "translate" field
# - Maintain all SQL syntax elements and database identifiers in their original form
# - Provide ONLY the direct translation without any explanations, notes, or additional text
#
# Database Schema:
# {schema}
#
# Text to translate:
# {text}
#
# Translation:
# """

# SQL to resolve the question: "{sql}"


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

# evaluation_template = """Based on below DDL and Question, evaluate gen_sql can resolve Question.
# If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""
#
# DDL: {schema}
# Question: {question}
# gt_sql: {answer}
# gen_sql: {gen_sql}"""