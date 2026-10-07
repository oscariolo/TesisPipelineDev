import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from peft import LoraConfig, TaskType

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


class TokenizedLogDataset(Dataset):
    """Tokenized causal-LM examples with prompt tokens masked from the loss."""

    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in self.examples[index].items()
        }



def train(model, tokenizer, training_args, train_dataset, eval_dataset=None):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=2048),
    )
    trainer.train()


def _get_field(record: dict[str, Any], field: str) -> Any:
    """Read dotted fields while also supporting flat JSON datasets."""
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(field)
        value = value[part]
    return value


def _format_example(record: dict[str, Any], label_column: str) -> tuple[str, str]:
    log_data = record.get("log_data", {})
    service = record.get("service", {})
    prompt = (
        "Classify this log entry.\n"
        f"Service: {service.get('name', 'unknown')}\n"
        f"Environment: {service.get('environment', 'unknown')}\n"
        f"Level: {log_data.get('level', 'unknown')}\n"
        f"Template: {log_data.get('template', '')}\n"
        f"Log: {log_data.get('raw_message', '')}\n"
        f"{label_column}:"
    )
    label = str(_get_field(record, label_column))
    return prompt, f" {label}"


def _tokenize_examples(
    records: list[dict[str, Any]],
    tokenizer: Any,
    label_column: str,
    max_length: int,
) -> TokenizedLogDataset:
    examples: list[dict[str, list[int]]] = []
    for record in records:
        prompt, target = _format_example(record, label_column)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            target_ids.append(tokenizer.eos_token_id)

        if len(target_ids) >= max_length:
            target_ids = target_ids[:max_length]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[: max_length - len(target_ids)]

        input_ids = prompt_ids + target_ids
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + target_ids,
            }
        )
    return TokenizedLogDataset(examples)


def prepare_dataset(
    train_file: str | Path,
    tokenizer: Any,
    label_column: str = "ground_truth_label.is_error",
    test_size: float = 0.2,
    max_length: int = 512,
    random_state: int = 42,
) -> tuple[TokenizedLogDataset, TokenizedLogDataset]:
    """Load, stratify, and tokenize a JSON log dataset for causal-LM training."""
    path = Path(train_file)
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("Training file must contain a non-empty JSON array")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every dataset item must be a JSON object")

    try:
        labels = [_get_field(record, label_column) for record in records]
    except KeyError as error:
        raise ValueError(f"Label field not found: {label_column}") from error

    try:
        train_records, eval_records = train_test_split(
            records,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "The dataset cannot be stratified; every class needs at least two "
            "records and test_size must leave one example per class."
        ) from error

    return (
        _tokenize_examples(train_records, tokenizer, label_column, max_length),
        _tokenize_examples(eval_records, tokenizer, label_column, max_length),
    )

def loraAdapter(model, r=8, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules=None):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model.add_adapter(lora_config, adapter_name="lora_adapter")


def build_run_output_dir(base_dir: str | Path, model_name: str) -> Path:
    """Return <base_dir>/<sanitized model name>_<timestamp> for one fine-tuning run."""
    safe_name = model_name.replace("/", "-").replace("\\", "-").replace(":", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(base_dir) / f"{safe_name}_{timestamp}"

from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN", ""))


def main():
    parser = argparse.ArgumentParser(description="Training script for Hugging Face models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train")
    parser.add_argument("--gguf-file", type=str, default=None, help="Name of the gguf file to use for the model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--label_column", type=str, default="ground_truth_label.is_error", help="Label field; dotted paths support nested JSON fields")
    parser.add_argument("--tunning_strategy", type=str, default="lora", help="Tuning method to use (e.g., lora, full)")
    parser.add_argument("--target_modules", type=str, nargs='+', default=None, help="List of target modules for LoRA tuning (if applicable)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"], help="Biases to train in the LoRA adapter")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction reserved for evaluation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="Enable gradient checkpointing to save memory")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5, help="Number of steps to accumulate gradients before updating model weights")

    args = parser.parse_args()

    #load model and tokenizer
    if args.gguf_file:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, gguf_file=args.gguf_file, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, gguf_file=args.gguf_file)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = prepare_dataset(
        args.train_file,
        tokenizer,
        label_column=args.label_column,
        test_size=args.test_size,
        max_length=args.max_length,
    )
    run_output_dir = build_run_output_dir(args.output_dir, args.model_name)
    print(f"Saving fine-tuned model to {run_output_dir}")

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="epoch",
    )

    if args.tunning_strategy == "lora":
        loraAdapter(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=args.target_modules,
        )

    train(model, tokenizer, training_args, train_dataset, eval_dataset)
    



if __name__ == "__main__":
    main()



