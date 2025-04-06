import torch
import re
import logging
import time
import json
import pandas as pd

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import utils.parallel
from llms.client import get_processor_for_mode, llm_invoke_parallel, process_response_by_mode, \
    process_evaluation_results
from llms.templates import make_request_jobs, make_prompt
from utils import common
from os.path import join
from utils.config import get_apikey, BatchMode, get_hf_token
from utils.common import change_jsonl_to_csv, upload_to_huggingface
from os import path
from pathlib import Path
from utils.common import check_and_create_directory, get_api_url, sanitize_filename, load_dataset
from utils.reporting import EvalResultsLogger
from utils.sql_extractor import extract_resolve_yn_from_text


def prepare_evaluation(options):
    """병렬 처리를 사용한 테스트 데이터셋 준비 (UUID 기반 요청 추적 포함)"""
    # 시작 시간 측정
    start_time = time.time()

    # 기본 경로 설정
    model_prefix = path.join(options.prefix, "test")
    check_and_create_directory(model_prefix)

    # 파일 경로 생성
    filepath_base = options.test_dataset + "." + options.base_model
    safe_model_name = sanitize_filename(filepath_base)
    filepath = path.join(model_prefix, f"{safe_model_name}.jsonl")

    # 이미 처리된 결과 파일이 있는지 확인
    if Path(filepath).exists():
        logging.info(f"기존 처리 결과 사용: {filepath}")
        results = pd.read_json(filepath, lines=True)
        logging.info(f"데이터 로드 완료: {len(results)}행, 컬럼={results.columns.tolist()}")

        prepare_time = time.time() - start_time
        logging.info(f"테스트 데이터셋 준비 완료 (캐시 사용): {prepare_time:.2f}초 소요")
        return results

    logging.info(f"처리 결과 파일 없음, 새로 생성: {filepath}")

    # 데이터셋 로드
    df = load_dataset(options)
    logging.info(f"데이터셋 로드 완료: {len(df)}행, 컬럼={df.columns.tolist()}")

    # NL2SQL 변환 시작 시간
    translation_start_time = time.time()

    # 로그 디렉터리 설정
    abs_log_dir = path.abspath(path.join(options.prefix, "logs"))

    # 응답 처리기 생성
    response_processor = get_processor_for_mode('nl2sql')

    # 옵션에 모드 명시적 설정
    batch_options = {
        'mode': BatchMode.NL2SQL,  # 열거형 직접 사용
        'input_column': options.input_column or 'question',
        'output_column': options.output_column or 'gen_sql',
        'question_column': options.question_column or 'question',
        'answer_column': options.answer_column or 'answer',
        'batch_size': options.batch_size,
        'max_retries': options.max_retries,
        'max_concurrent': options.max_concurrent,
    }

    # API URL 설정
    api_url = get_api_url(options.ollama_url, options.base_model)

    # 모델 예열 옵션
    warmup_enabled = getattr(options, 'warmup_model', True)

    # 병렬 호출 실행
    logging.info(f"총 {len(df)}개 데이터셋에 대한 병렬 처리를 시작합니다.")
    logging.info(
        f"병렬 처리 설정: 배치 크기 {options.batch_size}, 최대 동시 요청 {options.max_concurrent}, 최대 재시도 {options.max_retries}")

    responses = llm_invoke_parallel(
        options.base_model,
        df,
        url=api_url,
        log_dir=abs_log_dir,
        warmup=warmup_enabled,
        response_processor=response_processor,
        options=batch_options
    )

    # NL2SQL 변환 종료 시간 측정
    translation_end_time = time.time()
    translation_time = translation_end_time - translation_start_time

    # 결과 처리
    from llms.client import make_result
    results, success_count, error_count = make_result(responses, df, batch_options)

    # 결과 저장
    results.to_json(filepath, orient='records', lines=True)
    logging.info(f"결과 파일 저장 완료: {filepath}")

    # 통계 계산
    accuracy = (success_count / len(df)) * 100 if len(df) > 0 else 0
    avg_time = translation_time / len(df) if len(df) > 0 else 0
    throughput = len(df) / translation_time if translation_time > 0 else 0

    # 모델 크기 추정
    model_size = "Unknown"
    model_name = options.base_model.lower()
    if '7b' in model_name:
        model_size = '7B'
    elif '8b' in model_name:
        model_size = '8B'
    elif '13b' in model_name or '14b' in model_name:
        model_size = '13-14B'
    elif '27b' in model_name or '30b' in model_name:
        model_size = '27-30B'
    elif '70b' in model_name:
        model_size = '70B'

    # NL2SQL 변환 성능 측정 결과 기록
    nl2sql_stats = {
        'nl2sql_model': options.base_model,
        'test_dataset': getattr(options, 'test_dataset', ''),
        'test_size': len(df),
        'successful_count': success_count,
        'accuracy': accuracy,
        'avg_processing_time': avg_time,
        'batch_throughput': throughput,
        'comments': f"성공: {success_count}, 실패: {error_count}",
        'phase': 'translation',
        'success_rate': accuracy,
        'error_rate': (error_count / len(df)) * 100 if len(df) > 0 else 0,
        'avg_translation_time_s': avg_time,
        'throughput': throughput,
        'model_size': model_size
    }

    # 결과 요약 출력
    logging.info("\n===== NL2SQL 변환 결과 요약 =====")
    logging.info(f"모델: {options.base_model}")
    logging.info(f"성공률: {nl2sql_stats['success_rate']:.2f}% ({success_count}/{len(df)})")
    logging.info(f"평균 처리 시간: {avg_time:.3f}초/쿼리")
    logging.info(f"총 소요 시간: {translation_time:.2f}초")
    logging.info("===============================\n")

    # 통계 기록
    stats_dir = path.join(options.prefix, 'stats')
    check_and_create_directory(stats_dir)
    translation_logger = EvalResultsLogger(output_dir=stats_dir, filename='nl2sql_translation_stats.csv')
    translation_logger.log_evaluation_result(nl2sql_stats)

    prepare_time = time.time() - start_time
    logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

    return results


def perform_evaluation(options, dataset):
    """평가 수행 함수 (UUID 기반 요청 추적 포함)"""
    # 모델 예열 옵션 사용 (기본값: True)
    no_evaluation = getattr(options, 'no_evaluation', False)

    if no_evaluation:
        return []

    # 검증 시작 시간 측정
    start_time = time.time()

    base_model = options.base_model
    model = options.verifying_model

    # 이 부분이 중요: 로그 디렉토리를 명시적으로 지정하고 절대 경로로 변환
    log_dir = path.join(options.prefix, "logs")
    abs_log_dir = path.abspath(log_dir)

    # 타임아웃 옵션 추출 (없으면 기본값 사용)
    request_timeout = getattr(options, 'request_timeout', 300)

    filepath_base = options.test_dataset + "." + base_model

    # 모델명을 안전한 파일명으로 변환
    safe_base_model = sanitize_filename(filepath_base)
    safe_model = sanitize_filename(model)

    result_prefix = join(options.prefix, 'eval')
    if not Path(result_prefix).exists():
        Path(result_prefix).mkdir(parents=True)

    # 안전한 파일명으로 경로 생성
    requests_filepath = join(result_prefix, f"{safe_base_model}_{safe_model}_requests.jsonl")
    save_filepath = join(result_prefix, f"{safe_base_model}_{safe_model}_results.jsonl")
    output_file = join(result_prefix, f"{safe_base_model}-{safe_model}.csv")

    # # 성공/실패 케이스용 파일 경로 추가
    # success_cases_file = join(result_prefix, f"{safe_base_model}-{safe_model}_success_cases.csv")
    # failure_cases_file = join(result_prefix, f"{safe_base_model}-{safe_model}_failure_cases.csv")

    # logging.info("DataFrame:\n%s", dataset)
    logging.info("Evaluation file path: %s", output_file)

    if not Path(requests_filepath).exists():
        # 평가를 위한 requests.jsonl 생성
        jobs_with_id = make_request_jobs(model, dataset, options={'evaluation': True})

        with open(requests_filepath, "w") as f:
            for job_with_id in jobs_with_id:
                json_string = json.dumps(job_with_id)
                f.write(json_string + "\n")

    verification_start_time = time.time()

    if not Path(save_filepath).exists():
        # 응답 처리 함수 정의
        def eval_response_processor(response, metadata=None):
            """평가 응답을 즉시 처리하는 함수"""
            try:
                if isinstance(response, dict):
                    # OpenAI 응답 처리 (choices 필드가 있는 경우)
                    if 'choices' in response and len(response['choices']) > 0:
                        content = response['choices'][0]['message']['content']
                        result = extract_resolve_yn_from_text(content)

                        return result

                    # Ollama 응답 처리 (response 필드가 있는 경우)
                    elif 'response' in response:
                        content = response['response']
                        result = extract_resolve_yn_from_text(content)

                        return result

                # 처리할 수 없는 형식
                raise ValueError(f"응답 형식이 올바르지 않습니다: {response}")
            except Exception as e:
                raise ValueError(f"응답 처리 중 오류 발생: {str(e)}, 응답: {response}")

        # 배치 처리 옵션 설정
        batch_options = {
            'batch_size': options.batch_size if hasattr(options, 'batch_size') else 20,
            'max_concurrent': options.max_concurrent if hasattr(options, 'max_concurrent') else 10,
            'max_retries': options.max_retries if hasattr(options, 'max_retries') else 3,
            'request_timeout': request_timeout
        }

        logging.info(
            f"API 요청 설정: 배치={batch_options['batch_size']}, 동시={batch_options['max_concurrent']}, 재시도={batch_options['max_retries']}")

        utils.parallel.process_by_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=get_api_url(options.ollama_url, model),
            api_key=get_apikey(),
            max_requests_per_minute=2500,
            max_tokens_per_minute=100000,
            token_encoding_name="cl100k_base",
            max_attempts=batch_options['max_retries'],
            logging_level=20,
            max_concurrent_requests=batch_options['max_concurrent'],
            batch_size=batch_options['batch_size'],
            response_processor=eval_response_processor,  # 응답 처리 함수 전달
            prefix=abs_log_dir,  # 로그 디렉토리를 option.prefix로 설정
            request_timeout=request_timeout,  # 타임아웃 옵션 전달
        )

    verification_end_time = time.time()
    verification_time = verification_end_time - verification_start_time

    if not Path(output_file).exists():
        base_eval = change_jsonl_to_csv(
            save_filepath,
            response_column="resolve_yn",
            model=model
        )

        # Process results and save success/failure cases
        base_eval, success_count, failure_count = process_evaluation_results(
            base_eval,
            dataset,
            result_prefix,
            f"{safe_base_model}-{safe_model}"
        )

        # Now save the fully processed data
        base_eval.to_csv(output_file, index=False)
        logging.info(f"Processed evaluation data saved to {output_file}, {len(base_eval)} rows, columns={base_eval.columns.tolist()}")
    else:
        base_eval = pd.read_csv(output_file)
        logging.info(f"Evaluation data loaded: {len(base_eval)} rows, columns={base_eval.columns.tolist()}")

    # 정확도 계산
    num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]
    total_size = len(dataset)
    accuracy = (num_correct_answers / total_size) * 100 if total_size > 0 else 0

    logging.debug("Evaluation CSV:\n%s", base_eval)
    logging.info("Number of correct answers: %s", num_correct_answers)

    eval_time = time.time() - start_time
    logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    # 검증 성능 측정 결과 기록
    verification_stats = {
        'nl2sql_model': options.base_model,
        'evaluator_model': options.verifying_model,
        'test_dataset': getattr(options, 'test_dataset', ''),
        'test_size': total_size,
        'successful_count': num_correct_answers,
        'accuracy': accuracy,
        'comments': f"정답 수: {num_correct_answers}/{total_size}",
        'phase': 'verification',
        # 'avg_verification_time': verification_time / total_size if total_size > 0 else 0,
        # 'verification_throughput': total_size / verification_time if verification_time > 0 else 0,
    }

    # 모델 크기 추정
    model_name = options.base_model.lower()
    if '7b' in model_name:
        verification_stats['model_size'] = '7B'
    elif '8b' in model_name:
        verification_stats['model_size'] = '8B'
    elif '13b' in model_name or '14b' in model_name:
        verification_stats['model_size'] = '13-14B'
    elif '27b' in model_name or '30b' in model_name:
        verification_stats['model_size'] = '27-30B'
    elif '70b' in model_name:
        verification_stats['model_size'] = '70B'

    # 직접 로그 출력을 위한 상세 정보 출력
    logging.info("\n===== 검증 결과 요약 =====")
    logging.info(f"모델: {options.base_model}, 평가자: {options.verifying_model}")
    logging.info(f"정확도: {accuracy:.2f}% ({num_correct_answers}/{total_size})")
    logging.info(f"평균 처리 시간: {(verification_time / total_size):.3f}초/쿼리")
    logging.info(f"총 소요 시간: {verification_time:.2f}초")
    logging.info(f"성공 케이스: {num_correct_answers}개, 실패 케이스: {total_size - num_correct_answers}개")
    logging.info("==========================\n")

    # EvalResultsLogger 객체 생성 및 결과 로깅
    stats_dir = path.join(options.prefix, 'stats')
    check_and_create_directory(stats_dir)
    verification_logger = EvalResultsLogger(output_dir=stats_dir,
                                            filename='nl2sql_verification_stats.csv')

    # 로깅 수행
    verification_logger.log_evaluation_result(verification_stats)

    return base_eval, accuracy, eval_time


def change_responses_to_csv(dataset, output_file='', prompt_column="prompt", response_column="response", model="gpt"):
    prompts = []
    responses = []

    for json_data in dataset:
        prompts.append(json_data[0]['messages'][0]['content'])
        if model.lower().startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
            responses.append(json_data[1]['choices'][0]['message']['content'])
        else:
            responses.append(json_data[1]['message']['content'])

    dfs = pd.DataFrame({prompt_column: prompts, response_column: responses})
    if not output_file == '':
        dfs.to_csv(output_file, index=False)
    return dfs


def evaluation_api(model, dataset, prefix='', batch_size=10, max_concurrent=10, max_retries=3, size=None, api_key="",
                   llm_invoke_jobs_parallel=None):
    """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""
    check_and_create_directory(prefix)
    filepath = path.join(prefix, "text2sql.jsonl")
    requests_path = path.join(prefix, 'requests')
    if not Path(requests_path).exists():
        Path(requests_path).mkdir(parents=True)
    # requests_filepath = clean_filepath(filepath, prefix=requests_path)
    requests_filepath = join(requests_path, filepath)
    check_and_create_directory(path.dirname(requests_filepath))

    # prompts = make_prompts_for_evaluation(dataset)
    # jobs = util_common.make_request_jobs(model, prompts)
    jobs = make_request_jobs(model, dataset, options={'evaluation': True})

    with open(requests_filepath, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

    url = "https://api.openai.com/v1/chat/completions" if model.lower().startswith(
        'gpt') or model.startswith('o1') or model.startswith('o3') else "http://localhost:11434/api/chat"

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


def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    #, load_in_4bit)=True,
    #                                    bnb_4bit_compute_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


def prepare_finetuning(option):
    filepath = path.join(option.prefix, "train_dataset.jsonl")

    if path.exists(filepath):
        logging.info("File exists. Loading file: %s", filepath)
        df_sql = pd.read_json(filepath, lines=True)
    else:
        logging.info("File not exists. Creating data file.")

        df_sql = load_dataset("shangrilar/ko_text2sql", "clean")["train"]
        df_sql = df_sql.to_pandas()
        df_sql = df_sql.dropna().sample(frac=1, random_state=42)
        df_sql = df_sql.query("db_id != 1")

        for idx, row in df_sql.iterrows():
            df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'], option.base_model)

        df_sql.to_json(filepath, orient='records', lines=True)

    csv_filepath = path.join(option.prefix, "data/train.csv")

    if path.exists(csv_filepath):
        logging.info("File exists: %s", csv_filepath)
    else:
        df_sql.to_csv(csv_filepath, index=False)


def merge_model(base_model, finetuned_model, prefix=''):
    if finetuned_model == '':
        # 안전한 파일명으로 변환
        safe_base_model = common.sanitize_filename(base_model)
        return base_model, f"{prefix}_{safe_base_model}"

    filepath = path.join(prefix, finetuned_model)

    if path.exists(filepath):
        logging.info(f"Skip creating merged model: {filepath}")
    else:
        logging.info("File not exists. Creating merged model.")

        model_name = base_model

        logging.info(f"AutoModelForCausalLM.from_pretrained: {model_name}")
        # LoRA와 기초 모델 파라미터 합치기
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        logging.info(f"AutoTokenizer.from_pretrained: {model_name}")
        # 토크나이저 설정
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        #
        base_model.resize_token_embeddings(len(tokenizer))

        logging.info(f"PeftModel.from_pretrained: {base_model}, {finetuned_model}")
        model = PeftModel.from_pretrained(base_model, finetuned_model)

        logging.info("start merge and unload")
        model = model.merge_and_unload()

        logging.info("start saving model and tokenizer.")
        model.save_pretrained(filepath)
        tokenizer.save_pretrained(filepath)

    return finetuned_model, f"{prefix}_{finetuned_model.replace(':', '-')}"


def process_batch(options):
    """
    명령행 옵션에 따라 배치 처리 수행 - 전체 splits에 대해 처리

    Args:
        options: 명령행 옵션

    Returns:
        dict: 각 split별 처리 결과 데이터프레임
    """
    # 시작 시간 측정
    start_time = time.time()

    # 모드 확인
    batch_mode = BatchMode(options.mode)
    logging.info(f"배치 처리 모드: {batch_mode.value}")

    # 결과 저장 디렉토리 설정
    result_prefix = join(options.prefix, 'batch_results', options.mode)
    check_and_create_directory(result_prefix)

    # 모델명을 안전한 파일명으로 변환
    safe_model_name = sanitize_filename(options.base_model)

    # 입력/출력 컬럼 확인
    input_column = options.input_column
    output_column = options.output_column

    if batch_mode == BatchMode.NL2SQL:
        # NL2SQL 모드는 기본 컬럼 사용
        input_column = input_column or 'question'
        output_column = output_column or 'gen_sql'
    elif batch_mode == BatchMode.TRANSLATION:
        # NL2SQL 모드는 기본 컬럼 사용
        input_column = input_column or 'question'
        output_column = output_column or 'response'

    # 추가 옵션 설정
    batch_options = {
        'mode': batch_mode,
        'input_column': input_column,
        'output_column': output_column,
        'batch_size': options.batch_size,
        'max_retries': options.max_retries,
        'max_concurrent': options.max_concurrent,
    }

    # SQL 쿼리 추출 유틸리티 함수 (현재 함수에서는 내부 함수로 정의)
    def extract_sql_queries(text):
        """텍스트에서 SQL 쿼리 추출"""
        # 1) ```sql ... ``` 패턴
        pattern_triple_backticks = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL)
        # 2) ``` ... ``` 패턴 (언어 지정 없음)
        pattern_backticks = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)
        # 3) SELECT ... 패턴
        pattern_select = re.compile(r"\bSELECT\b.+?(?:;|$)", re.DOTALL | re.IGNORECASE)

        # (1) triple backticks with sql 안의 SQL 추출
        matches_sql_backticks = pattern_triple_backticks.findall(text)
        if matches_sql_backticks:
            return matches_sql_backticks[0].strip()

        # (2) triple backticks 안의 내용 추출
        matches_backticks = pattern_backticks.findall(text)
        if matches_backticks:
            # 중첩된 경우 가장 긴 것 선택
            longest_match = max(matches_backticks, key=len)
            return longest_match.strip()

        # (3) SELECT 문 추출
        matches_select = pattern_select.findall(text)
        if matches_select:
            return matches_select[0].strip()

        return ""

    # 응답 처리 함수 정의
    def batch_response_processor(response, metadata=None):
        """배치 모드에 따른 응답 처리 함수"""
        try:
            if batch_mode == BatchMode.NL2SQL:
                # NL2SQL 모드 응답 처리
                if isinstance(response, dict):
                    # Ollama 응답 처리
                    if 'response' in response:
                        content = response['response']
                        try:
                            result = json.loads(content)
                            if 'gen_sql' in result:
                                return result
                            else:
                                # SQL 쿼리만 추출 시도
                                sql_query = extract_sql_queries(content)
                                if sql_query:
                                    return {"gen_sql": sql_query}
                        except json.JSONDecodeError:
                            # JSON 파싱에 실패하면 SQL 쿼리 직접 추출 시도
                            sql_query = extract_sql_queries(content)
                            if sql_query:
                                return {"gen_sql": sql_query}

                    # OpenAI 응답 처리
                    elif 'choices' in response and len(response['choices']) > 0:
                        content = response['choices'][0]['message']['content']
                        try:
                            result = json.loads(content)
                            if 'gen_sql' in result:
                                return result
                            else:
                                # SQL 쿼리만 추출 시도
                                sql_query = extract_sql_queries(content)
                                if sql_query:
                                    return {"gen_sql": sql_query}
                        except json.JSONDecodeError:
                            # JSON 파싱에 실패하면 SQL 쿼리 직접 추출 시도
                            sql_query = extract_sql_queries(content)
                            if sql_query:
                                return {"gen_sql": sql_query}

            elif batch_mode == BatchMode.TRANSLATION:
                # 번역 모드 응답 처리
                if isinstance(response, dict):
                    # Ollama 응답 처리
                    if 'response' in response:
                        translation = response['response'].strip()
                        return {"translation": translation}

                    # OpenAI 응답 처리
                    elif 'choices' in response and len(response['choices']) > 0:
                        translation = response['choices'][0]['message']['content'].strip()
                        return {"translation": translation}

            # 처리할 수 없는 응답 형식
            raise ValueError(f"응답 형식이 올바르지 않습니다: {response}")
        except Exception as e:
            raise ValueError(f"응답 처리 중 오류: {str(e)}")

    # API URL 확인
    api_url = get_api_url(options.ollama_url, options.base_model)

    # === 여기서부터 수정: 모든 스플릿 처리 ===
    from datasets import load_dataset

    # 데이터셋 경로와 이름 분리
    dataset_option = getattr(options, 'test_dataset', None)
    if dataset_option:
        parts = dataset_option.split(':')
        dataset_path = parts[0]
        dataset_name = parts[1] if len(parts) > 1 else None
        dataset_split = parts[2] if len(parts) > 2 else None
    else:
        dataset_path = "shangrilar/ko_text2sql"
        dataset_name = "origin"
        dataset_split = "test"

    # 데이터셋 로드 및 사용 가능한 스플릿 확인
    try:
        logging.info(f"데이터셋 로드 중: {dataset_path}, 설정: {dataset_name}, 스플릿: {dataset_split}")

        # 데이터셋 로드
        if dataset_name:
            dataset = load_dataset(path=dataset_path, name=dataset_name, split=dataset_split)
        else:
            dataset = load_dataset(path=dataset_path)

        # 사용 가능한 스플릿 목록
        logging.info(f'dataset: [{type(dataset)}]{dataset}')

        import datasets
        if isinstance(dataset, datasets.arrow_dataset.Dataset):
            is_dataset = True
            available_splits = [dataset_split]
        else:
            is_dataset = False
            available_splits = list(dataset.keys())
            logging.info(f"사용 가능한 스플릿: {available_splits}")

        # 결과 저장 딕셔너리
        results_all_splits = {}

        # 각 스플릿 처리
        for split_name in available_splits:
            logging.info(f"=== 스플릿 '{split_name}' 처리 시작 ===")

            # 결과 파일 경로
            result_file = path.join(result_prefix, f"{safe_model_name}_{split_name}.jsonl")

            # 스플릿별 데이터셋 로드
            split_df = dataset.to_pandas() if is_dataset else dataset[split_name].to_pandas()

            # 테스트 크기 제한 적용 (옵션으로만 제한)
            if hasattr(options, 'test_size') and options.test_size is not None and isinstance(options.test_size, int):
                split_df = split_df[:options.test_size]
                logging.info(f"테스트 크기 제한 적용: {options.test_size}개 샘플")

            logging.info(f"스플릿 '{split_name}' 데이터 크기: {len(split_df)}개")

            # 병렬 배치 처리 실행
            logging.info(f"스플릿 '{split_name}' 배치 처리 시작: 모델={options.base_model}, 모드={batch_mode.value}")
            logging.info(f"배치 크기={options.batch_size}, 동시 요청={options.max_concurrent}, 최대 재시도={options.max_retries}")

            try:
                # 배치 처리 시작
                responses = llm_invoke_parallel(
                    options.base_model,
                    split_df,
                    url=api_url,
                    log_dir=options.prefix,
                    warmup=False,  # 첫 번째 스플릿에서만 예열
                    response_processor=batch_response_processor,
                    options=batch_options,
                )

                # 결과 처리
                results_df = process_response_by_mode(
                    responses,
                    split_df,
                    options=batch_options
                )

                # 결과 저장
                results_df.to_json(result_file, orient='records', lines=True)
                logging.info(f"스플릿 '{split_name}' 처리 결과 저장 완료: {result_file}")

                # CSV 형식으로도 저장
                csv_file = result_file.replace('.jsonl', '.csv')
                results_df.to_csv(csv_file, index=False)
                logging.info(f"스플릿 '{split_name}' CSV 결과 저장 완료: {csv_file}")

                # 결과 딕셔너리에 저장
                results_all_splits[split_name] = results_df

                # 스플릿 처리 완료 로그
                success_count = getattr(results_df, 'attrs', {}).get('success_count', 0)
                logging.info(f"스플릿 '{split_name}' 처리 완료: 총 {len(split_df)}개 항목, 성공 {success_count}개")

            except Exception as e:
                logging.error(f"스플릿 '{split_name}' 배치 처리 중 오류 발생: {str(e)}")
                results_all_splits[split_name] = None

        # Hugging Face에 업로드 (모든 스플릿 업로드)
        if options.upload_to_hf is not None:
            logging.info(f"Hugging Face에 모든 스플릿 데이터셋 업로드 시작: {options.upload_to_hf}")

            # 수정된 upload_to_huggingface 함수 사용
            upload_to_huggingface(
                results_all_splits,  # 스플릿별 데이터프레임 딕셔너리
                options.upload_to_hf,  # 업로드할 데이터셋 경로
                get_hf_token()  # Hugging Face API 토큰
            )

            logging.info(f"Hugging Face 업로드 완료: {options.upload_to_hf}")

    except Exception as e:
        logging.error(f"데이터셋 처리 중 오류 발생: {str(e)}")
        raise

    # 실행 시간 계산
    execution_time = time.time() - start_time

    # 요약 정보 출력
    logging.info(f"\n===== 배치 처리 요약 =====")
    logging.info(f"모델: {options.base_model}")
    logging.info(f"모드: {batch_mode.value}")
    logging.info(f"처리된 스플릿: {list(results_all_splits.keys())}")
    logging.info(f"총 실행 시간: {execution_time:.2f}초")
    logging.info("============================\n")

    return results_all_splits


def process_upload(input_file, hf_dataset_path):
    """
    이미 생성된 결과 파일들을 Hugging Face에 업로드하는 간단한 함수

    Args:
        input_file: 결과 파일이 저장된 파일 경로 (문자열)
        hf_dataset_path: Hugging Face에 업로드할 데이터셋 경로 (예: 'username/dataset-name')

    Returns:
        dict: 업로드된 스플릿별 데이터프레임
    """
    # 시작 시간 측정
    start_time = time.time()
    input_file = Path(input_file)

    # 입력 디렉토리 경로 정규화
    logging.info(f"입력 파일: {input_file}")

    # 경로가 존재하는지 확인
    if not Path(input_file).exists():
        raise ValueError(f"입력 디렉토리가 존재하지 않습니다: {input_file}")

    # 결과 저장 딕셔너리
    results_all_splits = {}

    try:
        file_ext = input_file.suffix.lower()
        # 파일 확장자에 따라 로드 방식 결정
        if file_ext.lower() == '.jsonl':
            df = pd.read_json(input_file, lines=True)
        elif file_ext.lower() == '.csv':
            df = pd.read_csv(input_file)
        else:
            logging.warning(f"지원되지 않는 파일 형식: {file_ext}")
            raise

        # 결과 딕셔너리에 추가
        results_all_splits['test'] = df

    except Exception as e:
        logging.error(f"파일 '{input_file}' 로드 중 오류 발생: {str(e)}")
        results_all_splits['test'] = None

    # 업로드할 데이터가 있는지 확인
    if not results_all_splits:
        raise ValueError("업로드할 데이터가 없습니다")

    # Hugging Face에 업로드
    try:
        logging.info(f"Hugging Face에 스플릿 데이터셋 업로드 시작: {hf_dataset_path}")

        # 업로드 함수 호출
        upload_to_huggingface(
            results_all_splits,  # 스플릿별 데이터프레임 딕셔너리
            hf_dataset_path,  # 업로드할 데이터셋 경로
            get_hf_token()  # Hugging Face API 토큰
        )

        logging.info(f"Hugging Face 업로드 완료: {hf_dataset_path}")
    except Exception as e:
        logging.error(f"Hugging Face 업로드 중 오류 발생: {str(e)}")
        raise

    # 실행 시간 계산
    execution_time = time.time() - start_time

    # 요약 정보 출력
    logging.info(f"\n===== 업로드 처리 요약 =====")
    logging.info(f"입력 파일: {input_file}")
    logging.info(f"업로드 대상: {hf_dataset_path}")
    logging.info(f"총 실행 시간: {execution_time:.2f}초")
    logging.info("============================\n")
