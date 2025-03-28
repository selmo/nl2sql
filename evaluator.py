import logging
import time
import json
import psutil
import pandas as pd
import torch
from os import path
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import api_request_parallel_processor
from llms.ollama_api import llm_invoke_parallel, llm_invoke_jobs_parallel
from llms.response_processor import make_result
from util import util_common
from os.path import join
from pathlib import Path
from util.config import get_apikey
from llms.prompt_generator import make_request_jobs, make_prompt
from util.util_common import check_and_create_directory, change_jsonl_to_csv, get_api_url
from util.eval_results_logger import EvalResultsLogger


def prepare_evaluation(option):
    """병렬 처리를 사용한 테스트 데이터셋 준비 (진행률 로깅 기능 추가)"""

    # 시작 시간 측정
    start_time = time.time()

    model_prefix = join(option.prefix, "test")
    check_and_create_directory(model_prefix)

    # 모델명을 안전한 파일명으로 변환
    safe_model_name = util_common.sanitize_filename(option.base_model)
    filepath = join(model_prefix, f"{safe_model_name}.jsonl")

    if not Path(filepath).exists():
        logging.info(f"파일이 존재하지 않습니다. 데이터 파일 생성 중: {filepath}")

        # 데이터셋 불러오기
        df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
        df = df.to_pandas()
        if option.test_size is not None:
            df = df[:option.test_size]

        # NL2SQL 변환 시작 시간
        translation_start_time = time.time()

        # 프롬프트 목록 생성
        dataset = []
        for _, row in df.iterrows():
            dataset.append(row)

        # 병렬 호출 실행
        logging.info(f"총 {len(dataset)}개 데이터셋에 대한 병렬 처리를 시작합니다.")
        logging.info(
            f"병렬 처리 설정: 배치 크기 {option.batch_size}, 최대 동시 요청 {option.max_concurrent}, 최대 재시도 {option.max_retries}")

        # log_dir에 절대 경로 전달
        abs_log_dir = path.abspath(join(option.prefix, "logs"))

        # 모델 예열 옵션 사용 (기본값: True)
        warmup_enabled = getattr(option, 'warmup_model', True)

        responses = llm_invoke_parallel(
            option.base_model,
            dataset,
            batch_size=option.batch_size,
            max_retries=option.max_retries,
            max_concurrent=option.max_concurrent,
            url=util_common.get_api_url(option.ollama_url, option.base_model),
            log_dir=abs_log_dir,  # 절대 경로 사용
            warmup=warmup_enabled,  # 모델 예열 옵션 전달
        )

        # NL2SQL 변환 종료 시간
        translation_end_time = time.time()
        translation_time = translation_end_time - translation_start_time

        # 결과 처리
        results, success_count, error_count = make_result(responses, df)

        # NL2SQL 변환 성능 측정 결과 기록
        nl2sql_stats = {
            'nl2sql_model': option.base_model,
            'evaluator_model': '',  # 변환 단계에서는 평가자 모델 없음
            'test_dataset': getattr(option, 'test_dataset', 'Spider'),
            'test_size': len(df),
            'successful_count': success_count,  # 성공 수 추가
            'accuracy': (success_count / len(df)) * 100 if len(df) > 0 else 0,
            'avg_processing_time': translation_time / len(df) if len(df) > 0 else 0,
            'batch_throughput': len(df) / translation_time if translation_time > 0 else 0,
            # 'batch_size': option.batch_size,
            # 'max_concurrent': option.max_concurrent,
            # 'max_retries': option.max_retries,
            'comments': f"성공: {success_count}, 실패: {error_count}",
            'phase': 'translation',
            'success_rate': (success_count / len(df)) * 100 if len(df) > 0 else 0,
            'error_rate': (error_count / len(df)) * 100 if len(df) > 0 else 0,
            'avg_translation_time_s': translation_time / len(df) if len(df) > 0 else 0,  # 변경: ms -> s 단위
            'throughput': len(df) / translation_time if translation_time > 0 else 0
        }

        # 모델 크기 추정 (가능한 경우)
        model_name = option.base_model.lower()
        if '7b' in model_name:
            nl2sql_stats['model_size'] = '7B'
        elif '8b' in model_name:
            nl2sql_stats['model_size'] = '8B'
        elif '13b' in model_name or '14b' in model_name:
            nl2sql_stats['model_size'] = '13-14B'
        elif '27b' in model_name or '30b' in model_name:
            nl2sql_stats['model_size'] = '27-30B'
        elif '70b' in model_name:
            nl2sql_stats['model_size'] = '70B'

        # 직접 로그 출력을 위한 상세 정보 출력
        logging.info("\n===== NL2SQL 변환 결과 요약 =====")
        logging.info(f"모델: {option.base_model}")
        logging.info(f"성공률: {nl2sql_stats['success_rate']:.2f}% ({success_count}/{len(df)})")
        logging.info(f"평균 처리 시간: {(translation_time / len(df)):.3f}초/쿼리")
        logging.info(f"총 소요 시간: {translation_time:.2f}초")
        logging.info("===============================\n")

        # NL2SQL 로거 초기화 및 결과 기록
        stats_dir = path.join(option.prefix, 'stats')
        check_and_create_directory(stats_dir)
        translation_logger = EvalResultsLogger(output_dir=stats_dir,
                                               filename='nl2sql_translation_stats.csv')
        translation_logger.log_evaluation_result(nl2sql_stats)

        logging.info(f"결과 처리 완료: 성공 {success_count}개, 오류 {error_count}개")

        # 결과 저장
        results.to_json(filepath, orient='records', lines=True)
        logging.info(f"파일 저장 완료: {filepath}")
    else:
        logging.info(f"파일이 존재합니다. 데이터 파일 로딩 중: {filepath}")
        results = pd.read_json(filepath, lines=True)
        logging.info(f"데이터 컬럼: {results.keys()}")
        logging.info("파일 로딩 완료.")

    prepare_time = time.time() - start_time
    logging.info(f"테스트 데이터셋 준비 완료: {prepare_time:.2f}초 소요")

    return results


def process_response(response, metadata=None):
    """
    API 응답을 즉시 처리하는 함수
    """
    try:
        # 응답에서 JSON 추출 및 처리
        if isinstance(response, dict):
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                # JSON 파싱
                resolve_yn_json = json.loads(content)
                if 'resolve_yn' in resolve_yn_json:
                    # 검증 결과 처리
                    return resolve_yn_json
            elif 'response' in response:
                # Ollama 응답 처리
                content = response['response']
                # JSON 파싱
                resolve_yn_json = json.loads(content)
                if 'resolve_yn' in resolve_yn_json:
                    # 검증 결과 처리
                    return resolve_yn_json

        # 처리할 수 없는 형식이면 오류 발생
        raise ValueError(f"응답 형식이 올바르지 않습니다: {response}")
    except json.JSONDecodeError as e:
        # JSON 파싱 오류
        raise ValueError(f"JSON 파싱 오류: {str(e)}, 응답: {response}")
    except KeyError as e:
        # 필요한 키가 없음
        raise ValueError(f"응답에 필요한 키가 없습니다: {str(e)}, 응답: {response}")
    except Exception as e:
        # 기타 오류
        raise ValueError(f"응답 처리 중 오류 발생: {str(e)}, 응답: {response}")


def perform_evaluation(option, dataset):
    # 검증 시작 시간 측정
    start_time = time.time()

    base_model = option.base_model
    model = option.verifying_model

    # 모델명을 안전한 파일명으로 변환
    safe_base_model = util_common.sanitize_filename(base_model)
    safe_model = util_common.sanitize_filename(model)

    result_prefix = join(option.prefix, 'eval')
    if not Path(result_prefix).exists():
        Path(result_prefix).mkdir(parents=True)

    # 안전한 파일명으로 경로 생성
    requests_filepath = join(result_prefix, f"{safe_base_model}_{safe_model}_requests.jsonl")
    save_filepath = join(result_prefix, f"{safe_base_model}_{safe_model}_results.jsonl")
    output_file = join(result_prefix, f"{safe_base_model}-{safe_model}.csv")

    logging.debug("DataFrame:\n%s", dataset)
    logging.info("Evaluation file path: %s", output_file)

    if not Path(requests_filepath).exists():
        # 평가를 위한 requests.jsonl 생성
        jobs = make_request_jobs(model, dataset, evaluation=True)

        with open(requests_filepath, "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")

    verification_start_time = time.time()

    if not Path(save_filepath).exists():
        api_request_parallel_processor.process_by_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=util_common.get_api_url(option.ollama_url, model),
            api_key=get_apikey(),
            max_requests_per_minute=2500,
            max_tokens_per_minute=100000,
            token_encoding_name="cl100k_base",
            max_attempts=option.max_retries if hasattr(option, 'max_retries') else 10,
            logging_level=20,
            # max_concurrent_requests=option.max_concurrent if hasattr(option, 'max_concurrent') else 10,
            # batch_size=option.batch_size if hasattr(option, 'batch_size') else 20,
            # response_processor=process_response,  # 응답 처리 함수 전달
            prefix=option.prefix  # 로그 디렉토리를 option.prefix로 설정
        )

    verification_end_time = time.time()
    verification_time = verification_end_time - verification_start_time

    if not Path(output_file).exists():
        base_eval = change_jsonl_to_csv(
            save_filepath,
            output_file,
            response_column="resolve_yn",
            model=model
        )
    else:
        base_eval = pd.read_csv(output_file)

    base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]
    total_size = len(dataset)
    accuracy = (num_correct_answers / total_size) * 100 if total_size > 0 else 0

    logging.debug("Evaluation CSV:\n%s", base_eval)
    logging.info("Number of correct answers: %s", num_correct_answers)

    eval_time = time.time() - start_time
    logging.info(f"평가 완료: {eval_time:.2f}초 소요")

    # 검증 성능 측정 결과 기록
    verification_stats = {
        'nl2sql_model': option.base_model,
        'evaluator_model': option.verifying_model,
        'test_dataset': getattr(option, 'test_dataset', 'Spider'),
        'test_size': total_size,
        'successful_count': num_correct_answers,  # 성공 수 추가
        'accuracy': accuracy,
        'avg_processing_time': verification_time / total_size if total_size > 0 else 0,
        'batch_throughput': total_size / verification_time if verification_time > 0 else 0,
        # 'batch_size': option.batch_size,
        # 'max_concurrent': option.max_concurrent,
        # 'max_retries': option.max_retries,
        'comments': f"정답 수: {num_correct_answers}/{total_size}",
        'phase': 'verification',
        'avg_verification_time_s': verification_time / total_size if total_size > 0 else 0  # 변경: ms -> s 단위
    }

    # 모델 크기 추정 (가능한 경우)
    model_name = option.base_model.lower()
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
    logging.info(f"모델: {option.base_model}, 평가자: {option.verifying_model}")
    logging.info(f"정확도: {accuracy:.2f}% ({num_correct_answers}/{total_size})")
    logging.info(f"평균 처리 시간: {(verification_time / total_size):.3f}초/쿼리")
    logging.info(f"총 소요 시간: {verification_time:.2f}초")
    logging.info("==========================\n")

    # EvalResultsLogger 객체 생성 및 결과 로깅
    stats_dir = path.join(option.prefix, 'stats')
    check_and_create_directory(stats_dir)
    verification_logger = EvalResultsLogger(output_dir=stats_dir,
                                            filename='nl2sql_verification_stats.csv')

    # 로깅 수행 (콘솔에도, 파일에도 로깅됨)
    verification_logger.log_evaluation_result(verification_stats)

    return base_eval, accuracy, eval_time


def prepare_evaluation_hf(model, prefix='', test_size=0):
    # 모델명을 안전한 파일명으로 변환
    safe_model_name = util_common.sanitize_filename(model)
    df_filepath = path.join(prefix, f"{safe_model_name}.jsonl")
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

        gen_sqls = []
        # tqdm의 total은 전체 prompt 수
        batch_size = 10
        prompts = df['prompt'].tolist()
        from tqdm import tqdm
        for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
            batch = prompts[i: i + batch_size]
            # pipeline에 한 번에 batch_size개씩 넣어주기
            outputs = hf_pipe(
                batch,
                do_sample=False,
                return_full_text=False,
                truncation=True
            )
            # 파이프라인 반환값이 리스트 형태일 것이므로
            if batch_size == 1:
                # 단일 prompt만 줬을 경우도 리스트로 나오므로
                outputs = [outputs]
            gen_sqls.extend(outputs)

        # gen_sqls = hf_pipe(
        #     df['prompt'].tolist(),
        #     do_sample=False,
        #     return_full_text=False,
        #     # max_length=512,
        #     truncation=True
        # )
        gen_sqls = [x[0][('generated_text')] for x in gen_sqls]
        df['gen_sql'] = gen_sqls
        logging.info(f"Data Columns: {df.keys()}")
        logging.info(f"Data: {df}")
        df.to_json(df_filepath, orient='records', lines=True)
        logging.info(f"Data file saved: {df_filepath}")

    return df


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


def evaluation_api(model, dataset, prefix='', batch_size=10, max_concurrent=10, max_retries=3, size=None, api_key=""):
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
    jobs = make_request_jobs(model, dataset, evaluation=True)

    with open(requests_filepath, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

    # # 로그 파일 설정
    # log_filepath = path.join(prefix, "parallel_processing_eval.log")
    # file_handler = logging.FileHandler(log_filepath)
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    #
    # root_logger = logging.getLogger()
    # root_logger.addHandler(file_handler)
    # root_logger.setLevel(logging.INFO)
    #
    # logging.info(f"병렬 처리 로그를 {log_filepath}에 기록합니다.")

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
        safe_base_model = util_common.sanitize_filename(base_model)
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
