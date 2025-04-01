import re
import json
import logging
import argparse
from datasets import load_dataset, Dataset

from util.config import get_hf_token

# Set up logging format and level (INFO for general progress, WARNING for skips)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class PromptHandler:
    """Base class for prompt format handlers."""

    def can_handle(self, prompt: str) -> bool:
        raise NotImplementedError

    def parse(self, prompt: str, completion: str):
        """Parse prompt and completion into (context, question, answer)."""
        raise NotImplementedError


class QuestionMetadataHandler(PromptHandler):
    """Handler for prompts containing 'question:' and 'metadata:' sections."""

    def can_handle(self, prompt: str) -> bool:
        # Check if the prompt contains the expected keywords
        return "question:" in prompt and "metadata:" in prompt

    def parse(self, prompt: str, completion: str):
        # Find the question text between 'question:' and 'metadata:'
        try:
            q_idx = prompt.index("question:")
            m_idx = prompt.index("metadata:")
        except ValueError:
            # If the keywords aren't found as expected, return None to indicate failure
            return None

        # Extract question substring and strip it
        question_text = prompt[q_idx + len("question:"): m_idx].strip()
        # Extract context substring after 'metadata:'
        context_text = prompt[m_idx + len("metadata:"):].strip()
        # Remove enclosing triple backticks (if any) from context_text
        if context_text.startswith("```"):
            # Remove the first triple backticks
            context_text = context_text[3:]
            # If there's a closing triple backtick, remove it as well
            end_idx = context_text.rfind("```")
            if end_idx != -1:
                context_text = context_text[:end_idx]
        context_text = context_text.strip()
        # The completion itself is the answer (SQL query or final answer)
        answer_text = completion.strip()

        return {"context": context_text, "question": question_text, "answer": answer_text}


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Transform an NL2SQL dataset to context-question-answer format.")
    parser.add_argument("--input_dataset", type=str, required=True,
                        help="HuggingFace dataset name or path (e.g. 'defog/wikisql').")
    parser.add_argument("--output_dataset", type=str, required=True,
                        help="HuggingFace Hub dataset name to push the results to (e.g. 'username/dataset').")
    parser.add_argument("--split", type=str, default="train",
                        help="Which split of the dataset to transform (default: 'train').")
    parser.add_argument("--save_json", action="store_true",
                        help="If set, save the transformed dataset to a local JSON file.")
    args = parser.parse_args()

    input_name = args.input_dataset
    output_name = args.output_dataset
    split_name = args.split

    logging.info(f"Loading dataset '{input_name}' (split: '{split_name}')...")
    # Load the specified split of the input dataset
    dataset = load_dataset(input_name, split=split_name)
    logging.info(f"Loaded {len(dataset)} examples from '{input_name}'.")

    # Initialize prompt handlers (additional handlers can be added to this list)
    handlers: list[PromptHandler] = [QuestionMetadataHandler()]

    transformed_data = {"context": [], "question": [], "answer": []}
    skipped_count = 0

    # Process each example in the input dataset
    for idx, example in enumerate(dataset):
        prompt = example.get("prompt") or ""
        completion = example.get("completion") or ""
        # Try each handler to see if it can parse this prompt
        parsed = None
        for handler in handlers:
            if handler.can_handle(prompt):
                result = handler.parse(prompt, completion)
                if result:
                    parsed = result
                break  # Stop at the first matching handler
        if parsed is None:
            # No handler could parse this prompt format
            skipped_count += 1
            logging.warning(f"Skipping example {idx}: Unrecognized prompt format.")
            continue
        # Append the parsed fields to our transformed data
        transformed_data["context"].append(parsed["context"])
        transformed_data["question"].append(parsed["question"])
        transformed_data["answer"].append(parsed["answer"])

    processed_count = len(transformed_data["question"])
    logging.info(f"Parsed {processed_count} examples successfully. Skipped {skipped_count} examples.")

    # Create a HuggingFace Dataset from the transformed data
    new_dataset = Dataset.from_dict(transformed_data)

    # Optionally, save to a local JSON file
    if args.save_json:
        # Sanitize filename by replacing "/" in dataset name with "_"
        file_name = output_name.replace("/", "_") + f"_{split_name}.json"
        from os import path
        filepath = path.join("dataset", file_name)
        logging.info(f"Saving transformed data to {filepath}...")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                [{"context": c, "question": q, "answer": a}
                 for c, q, a in zip(transformed_data["context"],
                                    transformed_data["question"],
                                    transformed_data["answer"])],
                f, indent=2
            )
        logging.info("Local JSON file saved.")

    # Push the new dataset to the Hugging Face Hub
    logging.info(f"Pushing the dataset to Hugging Face Hub at '{output_name}'...")
    new_dataset.push_to_hub(output_name, token=get_hf_token())  # Requires HF authentication
    logging.info("Dataset push completed successfully.")


if __name__ == "__main__":
    main()
