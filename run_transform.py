import re
import json
import logging
import argparse
from typing import List, Dict, Set
from datasets import load_dataset, Dataset

from utils.config import get_hf_token

try:
    import sqlglot
    from sqlglot.errors import ParseError, TokenError
except ImportError:
    raise ImportError("sqlglot 라이브러리가 필요합니다. pip install sqlglot 로 설치하세요.")


logging.basicConfig(
    format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
    level=logging.INFO
)


# -------------------------------------------------
# Prompt Handling for transform
# -------------------------------------------------
class PromptHandler:
    def can_handle(self, prompt: str) -> bool:
        raise NotImplementedError

    def parse(self, prompt: str, completion: str):
        """Return dict with (context, question, answer)."""
        raise NotImplementedError


class QuestionMetadataHandler(PromptHandler):
    """Example Handler: 'question:' and 'metadata:' 구문 파싱."""
    def can_handle(self, prompt: str) -> bool:
        return "question:" in prompt and "metadata:" in prompt

    def parse(self, prompt: str, completion: str):
        try:
            q_idx = prompt.index("question:")
            m_idx = prompt.index("metadata:")
        except ValueError:
            return None
        question_text = prompt[q_idx + len("question:"): m_idx].strip()
        context_text = prompt[m_idx + len("metadata:"):].strip()
        if context_text.startswith("```"):
            context_text = context_text[3:]
            end_idx = context_text.rfind("```")
            if end_idx != -1:
                context_text = context_text[:end_idx]
        context_text = context_text.strip()
        answer_text = completion.strip()
        return {
            "context": context_text,
            "question": question_text,
            "answer": answer_text
        }


# -------------------------------------------------
# DBMS Detection (no classification field)
# -------------------------------------------------

DBMS_PATTERNS: Dict[str, List[str]] = {
    "MySQL": [
        r"\bauto_increment\b",
        r"\blimit\s+\d+\s*,\s*\d+",
        r"\bifnull\s*\(",
        r"`\w+`",
    ],
    "PostgreSQL": [
        r"\boffset\s+\d+",
        r"\breturning\b",
        r"\bserial\b|\bbigserial\b",
        r"\bilike\b",
        r"\'[^']*\'::\w+",
        r"\barray\s*\[",
    ],
    "SQLite": [
        r"\bsqlite_master\b",
        r"\bpragma\b",
        r"\browid\b",
        r"\binsert\s+or\s+(?:replace|ignore)\b",
        r"\bautoincrement\b",
    ],
    "SQL Server": [
        r"\btop\s+\d+",
        r"\bisnull\s*\(",
        r"@@\w+",
        r"\bidentity\s*\(\d+\s*,\s*\d+\)",
        r"\[[^\]]+\]",
    ],
    "Oracle": [
        r"\brownum\b",
        r"\bconnect\s+by\b",
        r"\bfrom\s+dual\b",
        r"\bsysdate\b",
        r"\bnvl\s*\(",
        r"\bminus\b",
        r"\bvarchar2\s*\(\d+",
        r"\bnumber\s*\(\d+",
        r"\bdecode\s*\(",
    ],
}

SQLGLOT_DIALECT_MAP: Dict[str, str] = {
    "MySQL": "mysql",
    "PostgreSQL": "postgres",
    "SQLite": "sqlite",
    "SQL Server": "tsql",
    "Oracle": "oracle",
}


def detect_dbms_no_class(sql_query: str) -> List[str]:
    """
    DBMS 감지. 다섯 가지 DBMS(MySQL, PostgreSQL, SQLite, SQL Server, Oracle) 중
    정규표현식/파싱에 의해 "호환 가능"으로 보이는 DBMS 목록을 반환.

    - 하나도 잡히지 않으면 => INVALID SQL (빈 리스트)
    - 전부(5)면 => 사실상 GENERAL
    - 중간만 잡히면 => SPECIFIC
    """
    sql_lower = sql_query.lower()

    # (1) 정규표현식 패턴 매칭
    pattern_candidates: Set[str] = set()
    for dbms_name, patterns in DBMS_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, sql_lower):
                pattern_candidates.add(dbms_name)
                break

    # (2) sqlglot 파싱
    parse_candidates: Set[str] = set()
    for dbms_name, dialect in SQLGLOT_DIALECT_MAP.items():
        try:
            sqlglot.parse_one(sql_query, read=dialect)
            parse_candidates.add(dbms_name)
        except (ParseError, TokenError):
            pass

    # 종합
    if not pattern_candidates and not parse_candidates:
        # 완전히 감지 실패 => INVALID
        return []

    if pattern_candidates and parse_candidates:
        inter = pattern_candidates.intersection(parse_candidates)
        if inter:
            # 교집합이 있으면 우선적으로
            final = inter
        else:
            # 합집합
            final = pattern_candidates.union(parse_candidates)
    elif pattern_candidates:
        final = pattern_candidates
    else:
        final = parse_candidates

    return sorted(final)


# -------------------------------------------------
# 공통 로직 (중복 제거)
# -------------------------------------------------
def apply_prompt_transform(dataset, handlers=None):
    """
    prompt -> (context, question, answer)
    """
    if handlers is None:
        handlers = [QuestionMetadataHandler()]
    out_data = {
        "context": [],
        "question": [],
        "answer": [],
    }
    skipped = 0
    for i in range(len(dataset)):
        example = dataset[i]
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        parsed = None
        for h in handlers:
            if h.can_handle(prompt):
                res = h.parse(prompt, completion)
                if res:
                    parsed = res
                break
        if not parsed:
            skipped += 1
            continue
        out_data["context"].append(parsed["context"])
        out_data["question"].append(parsed["question"])
        out_data["answer"].append(parsed["answer"])
    return Dataset.from_dict(out_data), skipped


def apply_dbms_detection(dataset, sql_field: str, dbms_field: str):
    """
    (1) dataset[sql_field]에서 SQL 추출
    (2) detect_dbms_no_class -> DBMS list
    (3) 해당 값을 dbms_field에 저장
    (4) INVALID SQL인 경우(빈 리스트) 로깅
    """
    data_dict = dataset.to_dict()
    if sql_field not in data_dict:
        logging.warning(
            f"[apply_dbms_detection] '{sql_field}' not found in dataset. Returning original dataset."
        )
        return dataset

    new_dbms_list = []
    valid_indices = []  # 유효한 데이터의 인덱스를 저장할 리스트

    for i in range(len(dataset)):
        sql_text = data_dict[sql_field][i]
        dbms_list = detect_dbms_no_class(sql_text)

        if not dbms_list:
            # 빈 dbms 리스트인 경우 로깅하고 해당 데이터 제외
            logging.warning(
                f"[apply_dbms_detection] INVALID SQL at index {i} => {repr(sql_text)}"
            )
        else:
            # 유효한 dbms 리스트가 있는 경우 데이터 포함
            valid_indices.append(i)
            new_dbms_list.append(dbms_list)

    # 유효한 데이터만 필터링
    filtered_data = {}
    for key in data_dict:
        filtered_data[key] = [data_dict[key][i] for i in valid_indices]

    # dbms 필드 추가
    filtered_data[dbms_field] = new_dbms_list

    filtered_dataset = Dataset.from_dict(filtered_data)
    logging.info(
        f"[apply_dbms_detection] Filtered dataset: {len(filtered_dataset)}/{len(dataset)} items retained"
    )

    return filtered_dataset


def save_dataset_to_json(dataset, file_name: str):
    records = []
    for i in range(len(dataset)):
        row = {}
        for col in dataset.column_names:
            row[col] = dataset[col][i]
        records.append(row)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def push_dataset_to_hub(dataset, repo_name: str):
    logging.info(f"Pushing dataset to '{repo_name}'...")
    dataset.push_to_hub(repo_name, token=get_hf_token())
    logging.info("Push completed successfully.")


# -------------------------------------------------
# 서브커맨드 함수들
# -------------------------------------------------
def cmd_transform(args):
    ds = load_dataset(args.input_dataset, args.config, split=args.split)
    logging.info(f"[transform] Loaded {len(ds)} from '{args.input_dataset}'")
    new_ds, skipped = apply_prompt_transform(ds)
    logging.info(f"[transform] transform done => size={len(new_ds)}, skipped={skipped}")

    if args.save_json:
        fn = args.output_dataset.replace("/", "_") + f"_{args.split}.json"
        save_dataset_to_json(new_ds, fn)
        logging.info(f"[transform] Saved to {fn}")

    push_dataset_to_hub(new_ds, args.output_dataset)


def cmd_detect_dbms(args):
    ds = load_dataset(args.input_dataset, args.config, split=args.split)
    logging.info(f"[detect-dbms] Loaded {len(ds)} from '{args.input_dataset}'")

    new_ds = apply_dbms_detection(ds, args.sql_field, args.dbms_field)
    logging.info(f"[detect-dbms] detection done => size={len(new_ds)}")

    if args.save_json:
        fn = args.output_dataset.replace("/", "_") + f"_{args.split}_dbms.json"
        logging.info(f"[detect-dbms] Save to {fn}")
        save_dataset_to_json(new_ds, fn)
        logging.info(f"[detect-dbms] Saved.")

    push_dataset_to_hub(new_ds, args.output_dataset)


def cmd_transform_detect(args):
    ds = load_dataset(args.input_dataset, args.config, split=args.split)
    logging.info(f"[transform-detect] Loaded {len(ds)} from '{args.input_dataset}'")

    # 1) prompt transform
    trans_ds, skipped = apply_prompt_transform(ds)
    logging.info(f"[transform-detect] transform => size={len(trans_ds)}, skipped={skipped}")

    # 2) dbms detection
    final_ds = apply_dbms_detection(trans_ds, args.sql_field, args.dbms_field)
    logging.info(f"[transform-detect] detection => size={len(final_ds)}")

    if args.save_json:
        fn = args.output_dataset.replace("/", "_") + f"_{args.split}_transform_detect.json"
        save_dataset_to_json(final_ds, fn)
        logging.info(f"[transform-detect] Saved to {fn}")

    push_dataset_to_hub(final_ds, args.output_dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Perform transformations and/or DBMS detection on an NL2SQL dataset (no separate class field)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # transform
    p_trans = subparsers.add_parser("transform", help="Prompt -> (context, question, answer).")
    p_trans.add_argument("--input_dataset", required=True)
    p_trans.add_argument("--output_dataset", required=True)
    p_trans.add_argument("--split", default="train")
    p_trans.add_argument("--save_json", action="store_true")

    # detect-dbms
    p_dbms = subparsers.add_parser("dbms", help="Read dataset, detect DBMS from sql_field -> dbms_field.")
    p_dbms.add_argument("--input_dataset", required=True)
    p_dbms.add_argument("--output_dataset", required=True)
    p_dbms.add_argument("--config", default=None)
    p_dbms.add_argument("--split", default="train")
    p_dbms.add_argument("--sql_field", default="sql", help="Which field has the SQL query?")
    p_dbms.add_argument("--dbms_field", default="dbms", help="Field name to store the DBMS list.")
    p_dbms.add_argument("--save_json", action="store_true")

    # transform-detect
    p_td = subparsers.add_parser("trans-dbms", help="Do prompt transform & DBMS detect in one pass.")
    p_td.add_argument("--input_dataset", required=True)
    p_td.add_argument("--output_dataset", required=True)
    p_td.add_argument("--config", default=None)
    p_td.add_argument("--split", default="train")
    p_td.add_argument("--sql_field", default="sql")
    p_td.add_argument("--dbms_field", default="dbms")
    p_td.add_argument("--save_json", action="store_true")

    args = parser.parse_args()

    if args.command == "transform":
        cmd_transform(args)
    elif args.command == "dbms":
        cmd_detect_dbms(args)
    elif args.command == "trans-dbms":
        cmd_transform_detect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
