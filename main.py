import api_request_parallel_processor
import aux_ollama
import utils
import json
import logging

import os.path as path

from aux_gpt import prepare_train_dataset, prepare_test_dataset, merge_model
from util_common import check_and_create_directory, clean_filepath, make_requests_for_evaluation, autotrain

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def evaluation(ft_model, verifying_model, dataset, prefix):
    eval_filepath = "text2sql.jsonl"

    logging.info("DataFrame:\n%s", dataset)
    logging.info("Evaluation file path: %s", eval_filepath)

    requests_path = path.join(prefix, 'requests')
    results_path = path.join(prefix, 'results')
    requests_filepath = clean_filepath(eval_filepath, prefix=requests_path)
    save_filepath = clean_filepath(eval_filepath, prefix=results_path)
    output_file = clean_filepath(f"{ft_model}.csv", prefix=results_path)
    check_and_create_directory(path.dirname(requests_filepath))
    check_and_create_directory(path.dirname(save_filepath))
    check_and_create_directory(path.dirname(output_file))

    # 평가를 위한 requests.jsonl 생성
    make_requests_for_evaluation(dataset, eval_filepath, dir=requests_path, model=verifying_model)

    url = "https://api.openai.com/v1/chat/completions" if verifying_model.lower().startswith(
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
    base_eval = utils.change_jsonl_to_csv(
        save_filepath,
        output_file,
        # "prompt",
        response_column="resolve_yn",
        model=verifying_model
    )

    base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]

    logging.info("Evaluation CSV:\n%s", base_eval)
    logging.info("Number of correct answers: %s", num_correct_answers)


import sys

# main routine
if __name__ == "__main__":
    prefix = "20250304-ollama"
    datapath = path.join(prefix, 'data')
    check_and_create_directory(datapath)

    # base_model = 'defog/sqlcoder-7b-2'
    # finetuned_model = "sqlcoder-finetuned"
    verifying_model = "deepseek-r1:70b" #"llama3.3:70b"
    base_model = ''
    finetuned_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            prepare_train_dataset(prefix)
            autotrain(
                base_model,
                finetuned_model,
                data_path=path.join(prefix, 'data'),
                text_column='text',
                lr=2e-4,
                batch_size=14,
                gradient_accumulation=5,
                block_size=1024,
                warmup_ratio=0.1,
                epochs=5,
                lora_r=16,
                lora_alpha=32,
                # epochs=1,
                # lora_r=8,
                # lora_alpha=16,
                lora_dropout=0.05,
                weight_decay=0.01,
                mixed_precision='fp16',
                peft=True,
                quantization='int4',
                trainer='sft',
            )

        elif sys.argv[1] == 'test' or sys.argv[1] == 'eval':
            merge_model(base_model, finetuned_model, prefix)
            test_dataset = prepare_test_dataset(finetuned_model, prefix)
            evaluation(finetuned_model, verifying_model, test_dataset, prefix)

        elif sys.argv[1] == 'eval-ollama':
            test_dataset = aux_ollama.prepare_test_dataset(verifying_model, verifying_model, prefix)

        else:
            print('Arg:\n\ttrain: Finetuning model\n\ttest|eval: Evaluation model')
