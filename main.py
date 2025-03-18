import aux_local
import logging
import sys
import os.path as path

from aux_common import prepare_train_dataset, prepare_test_dataset, merge_model, evaluation
from util_common import check_and_create_directory, autotrain
from utils import load_csv

# 로깅 설정 (원하는 포맷과 레벨로 조정 가능)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# main routine
if __name__ == "__main__":
    prefix = "20250318-http"
    datapath = path.join(prefix, 'data')
    check_and_create_directory(datapath)

    # base_model = 'defog/sqlcoder-7b-2'
    # finetuned_model = "sqlcoder-finetuned"
    verifying_model = "deepseek-r1:70b"  # "llama3.3:70b"
    base_model = ''  # 'qwq'
    finetuned_model = "Qwen/QwQ-32B-AWQ"  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

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
            # finetuned_model = "LGAI-EXAONE/EXAONE-Deep-32B-AWQ"
            merge_model(base_model, finetuned_model, prefix)
            test_dataset = prepare_test_dataset(finetuned_model, prefix)
            evaluation(finetuned_model, verifying_model, test_dataset, prefix)

        elif sys.argv[1] == 'ollama-api':
            finetuned_model = "qwq"
            merge_model(base_model, finetuned_model, prefix)
            test_dataset = aux_local.prepare_test_dataset(finetuned_model, prefix)
            evaluation(finetuned_model, verifying_model, test_dataset, prefix)

        elif sys.argv[1] == 'ollama-http':
            finetuned_model = "qwq"
            merge_model(base_model, finetuned_model, prefix)
            test_dataset = aux_local.prepare_test_ollama(finetuned_model, prefix)
            evaluation(finetuned_model, verifying_model, test_dataset, prefix)

        elif sys.argv[1] == 'eval-csv':
            base_eval = load_csv(sys.argv[2])
            num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]

            logging.info("Evaluation CSV:\n%s", base_eval)
            logging.info("Number of correct answers: %s", num_correct_answers)

        else:
            print('Arg:\n\ttrain: Finetuning model\n\ttest|eval: Evaluation model')
