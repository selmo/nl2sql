from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator

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
SQL to resolve the question:
"""
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

class SQL(BaseModel):
    reasoning: str = Field(description="your chain-of-thought or reasoning")
    description: str = Field(description="a short high-level description")
    gen_sql: str = Field(description="the final SQL")

sql_parser = PydanticOutputParser(pydantic_object=SQL)

def make_prompt(model_id: str, schema:str, question:str):
    if model_id.lower().startswith('sqlcoder'):
        return sqlcoder_template.format(schema=schema, question=question)
    else:
        return template.format(ddl=schema, request=question, format_instructions=sql_parser.get_format_instructions())


def make_requests(model, prompts):
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

def make_request(model_id, content: str):
    return {"model": model_id,
            "stream": False,
            "prompt": content,
            "format": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "description": {"type": "string"},
                    "gen_sql": {"type": "string"}
                },
                "required": ["reasoning", "description", "gen_sql"]
            }}

    # if model_id.lower().startswith('sqlcoder') or model_id.lower().startswith('exaone'):
    #     payload = {
    #         "model": model_id,
    #         "stream": False,
    #         "prompt": content,
    #         "format": {
    #             "type": "object",
    #             "properties": {
    #                 "reasoning": {"type": "string"},
    #                 "description": {"type": "string"},
    #                 "gen_sql": {"type": "string"}
    #             },
    #             "required": ["reasoning", "description", "gen_sql"]
    #         }
    #     }
    # else:
    #     payload = {
    #         "model": model_id,
    #         "stream": False,
    #         "messages": [
    #             {"role": "system", "content": content}
    #         ],
    #         "format": {
    #             "type": "object",
    #             "properties": {
    #                 "reasoning": {"type": "string"},
    #                 "description": {"type": "string"},
    #                 "gen_sql": {"type": "string"}
    #             },
    #             "required": ["reasoning", "description", "gen_sql"]
    #         }
    #     }
    # return payload
        # payload = {
        #         "model": model_name,
        #         "stream": False,
        #         "messages": [
        #             {"role": "system", "content": template.format(
        #                 format_instructions=parser.get_format_instructions(),
        #                 ddl=data['context'],
        #                 request=data['question'],
        #                 sql=''
        #             )}
        #         ],
        #         "format": {
        #             "type": "object",
        #             "properties": {
        #                 "reasoning": {"type": "string"},
        #                 "description": {"type": "string"},
        #                 "gen_sql": {"type": "string"}
        #             },
        #             "required": ["reasoning", "description", "gen_sql"]
        #         }
        #     }