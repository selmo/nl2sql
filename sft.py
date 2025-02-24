import torch
import warnings

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import os
import utils


# from codecarbon import EmissionsTracker
# tracker = EmissionsTracker()

def quick_start(dataset):
    # dataset = load_dataset("stanfordnlp/imdb", split="train")
    # Dataset({
    #     features: ['text', 'label'],
    #     num_rows: 25000
    # })

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

    training_args = SFTConfig(
        output_dir="/tmp",
        per_device_train_batch_size=4,
        num_train_epochs=1
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()


def print_dataset(dataset):
    i = 5
    print(dataset)
    for data in dataset:
        if (data['db_id'] == 1):
            continue
        elif i == 0:
            break
        else:
            i = i - 1
            print(data)
            # print('instruction:', data['instruction'])
            # print('input:', data['input'])
            # print('output:', data['output'])
            print("-" * 50)


def train_on_completions_only():
    dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
    # Dataset({
    #     features: ['instruction', 'input', 'output'],
    #     num_rows: 20022
    # })
    #
    # {
    #     "instruction": "Create a function that takes a specific input and produces a specific output using any
    #                     mathematical operators. Write corresponding code in Python.",
    #     "input": "",
    #     "output": """def f(x):\\n    \"\"\"\
    #     Takes a specific input and produces a specific output using any mathematical operators\\n    \"\"\"\
    #                  return x**2 + 3*x"
    # }
    #
    # instruction: Convert the following block of code into a more concise version using more appropriate looping
    #              mechanisms. Write corresponding code in Python.
    # input: myList = ["one", "two", "three", "four"]
    #
    #        newList = []
    #        for element in myList:
    #             newList.append(element[0])
    # output: myList = ["one", "two", "three", "four"]
    #
    #         newList = [element[0] for element in myList]

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="/tmp",
            per_device_train_batch_size=7,
            gradient_accumulation_steps=5,
            num_train_epochs=10
        ),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model("models-finetuned/opt-350m-alpaca")


def train_on_completions_only_guanaco():
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    # {
    #     "text": """
    # ### Human: Can you write a short introduction about the relevance of the term \"monopsony\" in economics?
    #            Please use examples related to potential monopsonies in the labour market and cite relevant research.
    # ### Assistant: \"Monopsony\" refers to a market structure where there is only one buyer for a particular good or
    #                service. In economics, this term is particularly relevant in the labor market, where a monopsony
    #                employer has significant power over the wages and working conditions of their employees.
    #                The presence of a monopsony can result in lower wages and reduced employment opportunities for
    #                workers, as the employer has little incentive to increase wages or provide better working
    #                conditions.\\n\\nRecent research has identified potential monopsonies in industries such as retail
    #                and fast food, where a few large companies control a significant portion of the market (Bivens &
    #                Mishel, 2013). In these industries, workers often face low wages, limited benefits, and reduced
    #                bargaining power, leading to a situation where they are dependent on the employer for their
    #                livelihood. This dependence can result in further suppression of wages and a decline in working
    #                conditions.\\n\\nOverall, the concept of monopsony is essential to understanding the dynamics of
    #                labor markets and the impact of market power on workers. Further research is needed to understand
    #                the extent and impact of monopsonies on the economy and to develop policies to address this issue.
    #                \\n\\nReferences:\\nBivens, J., & Mishel, L. (2013). The Pay of Corporate Executives and Financial
    #                Professionals as Evidence of Rents in Top 1 Percent Incomes. Journal of Economic Perspectives,
    #                27(3), 57-78.### Human: Now explain it to a dog"""
    # }
    # print_dataset(dataset)

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    instruction_template = "### Human:"
    response_template = "### Assistant:"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                               response_template=response_template, tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="/tmp",
            per_device_train_batch_size=6,
            gradient_accumulation_steps=5,
            num_train_epochs=1
        ),
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model("models-finetuned/opt-350m-guanaco")


def train_nl2sql():
    df_sql = load_dataset("shangrilar/ko_text2sql", "origin")["train"]
    # Dataset({
    #     features: ['db_id', 'context', 'question', 'answer'],
    #     num_rows: 38246
    # })
    # {
    #     "db_id": 1,
    #     "context": "CREATE TABLE players (\\n  player_id INT PRIMARY KEY AUTO_INCREMENT,\\n
    #                 username VARCHAR(255) UNIQUE NOT NULL,\\n  email VARCHAR(255) UNIQUE NOT NULL,\\n
    #                 password_hash VARCHAR(255) NOT NULL,\\n  date_joined DATETIME NOT NULL,\\n
    #                 last_login DATETIME\\n);",
    #     "question": "모든 플레이어 정보를 조회해 줘",
    #     "answer": "SELECT * FROM players;"
    # }

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['context'])):
            text = f"### Context: {example['context'][i]}\n ### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
            output_texts.append(text)
        return output_texts

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="int4",  # int4 quantization
    #     # bnb_4bit_use_double_quant=True,  # 선택사항: double quantization 사용 여부
    #     # bnb_4bit_compute_dtype=torch.float16  # 계산 dtype 설정 (fp16 권장)
    # )
    model = AutoModelForCausalLM.from_pretrained(
        "beomi/Yi-Ko-6B",
        device_map="auto",
        torch_dtype=torch.float16,
        # quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained("beomi/Yi-Ko-6B")

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model,
        train_dataset=df_sql,
        args=SFTConfig(
            output_dir="/tmp",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=True,
        ),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("models-finetuned/Yi-Ko-6B-finetuned")


if __name__ == "__main__":
    os.environ["CODECARBON_LOG_LEVEL"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=UserWarning)

    # dataset = load_dataset("stanfordnlp/imdb", split="train")
    # print(dataset)

    # train_on_completions_only()
    # train_on_completions_only_guanaco()
    train_nl2sql()

# training_args = SFTConfig(
#     max_seq_length=512,
#     output_dir="/tmp",
# )
# trainer = SFTTrainer(
#     "facebook/opt-350m",
#     train_dataset=dataset,
#     args=training_args,
# )
# trainer.train()


# df = load_dataset("shangrilar/ko_text2sql", "origin")
# df = df.to_pandas()
# for idx, row in df.iterrows():
#     prompt = utils.make_prompt(row['context'], row['question'])
#     df.loc[idx, 'prompt'] = prompt
# # sql 생성
# gen_sqls = hf_pipe(
#     df['prompt'].tolist(),
#     do_sample=False,
#     return_full_text=False,
#     max_length=512,
#     truncation=True
# )
# gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
# df['gen_sql'] = gen_sqls

# print(df)
# for d in df:
#     print(d)
#     print("-" * 50)
