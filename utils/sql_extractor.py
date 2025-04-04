import re
import json
import logging


def extract_sql_queries(text):
    """텍스트에서 SQL 쿼리를 추출하는 통합 유틸리티 함수"""
    # 1) ```sql ... ``` 패턴
    pattern_triple_backticks_sql = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL)

    # 2) ``` ... ``` 패턴 (언어 지정 없음)
    pattern_backticks = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)

    # 3) SELECT ... 패턴
    pattern_select = re.compile(r"\bSELECT\b.+?(?:;|$)", re.DOTALL | re.IGNORECASE)

    # 1) SQL 코드 블록 패턴
    matches = pattern_triple_backticks_sql.findall(text)
    if matches:
        return matches[0].strip()

    # 2) 일반 코드 블록 패턴
    matches = pattern_backticks.findall(text)
    if matches:
        # 중첩된 경우 가장 긴 것 선택
        longest_match = max(matches, key=len)
        return longest_match.strip()

    # 3) SELECT 문 패턴
    matches = pattern_select.findall(text)
    if matches:
        return matches[0].strip()

    # 4) JSON 객체에서 추출 시도
    try:
        json_obj = json.loads(text)
        if isinstance(json_obj, dict) and 'gen_sql' in json_obj:
            return json_obj['gen_sql']
    except (json.JSONDecodeError, TypeError):
        pass

    return ""


def extract_resolve_yn_from_text(content):
    """텍스트에서 resolve_yn 값 추출"""
    # 1. JSON 블록 추출 시도
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)

    # 2. JSON 파싱 시도
    try:
        json_data = json.loads(content)
        if 'resolve_yn' in json_data:
            return {"resolve_yn": json_data['resolve_yn'].lower().strip()}
    except json.JSONDecodeError:
        pass

    # 3. 직접 키-값 패턴 검색
    if re.search(r'[\'\"]resolve_yn[\'\"]:\s*[\'\"]yes[\'\"]', content, re.IGNORECASE):
        return {"resolve_yn": "yes"}
    elif re.search(r'[\'\"]resolve_yn[\'\"]:\s*[\'\"]no[\'\"]', content, re.IGNORECASE):
        return {"resolve_yn": "no"}

    # 4. 단순 텍스트 검색
    pattern = re.compile(r'resolve_yn\s*:?\s*[\'"]?(yes|no)[\'"]?', re.IGNORECASE)
    match = pattern.search(content)
    if match:
        return {"resolve_yn": match.group(1).lower().strip()}

    # 5. 단순 yes/no 단어 검색
    if re.search(r'\byes\b', content, re.IGNORECASE):
        return {"resolve_yn": "yes"}
    elif re.search(r'\bno\b', content, re.IGNORECASE):
        return {"resolve_yn": "no"}

    # 결정할 수 없는 경우
    return {"resolve_yn": "unknown"}