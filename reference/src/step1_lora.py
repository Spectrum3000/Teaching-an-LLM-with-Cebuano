from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from common import count_parameters, environment_record, perplexity, project_root, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilGPT2 with LoRA and record the full training ledger.")
    parser.add_argument("--corpus", type=Path, default=project_root() / "data" / "cebuano_corpus.txt")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "checkpoints" / "step1_lora")
    parser.add_argument("--ledger", type=Path, default=project_root() / "outputs" / "step1_lora.json")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(f"Run prepare_corpus first: {args.corpus}")

    raw = load_dataset("text", data_files={"train": str(args.corpus)})["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = raw.map(lambda batch: tokenizer(batch["text"]), batched=True, remove_columns=["text"])
    block_size = min(args.block_size, tokenizer.model_max_length)

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {key: sum(batch[key], []) for key in batch}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        return {key: [values[index:index + block_size] for index in range(0, total_length, block_size)] for key, values in concatenated.items()}

    grouped = tokenized.map(group_texts, batched=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.eos_token_id
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["c_attn"], bias="none")
    model = get_peft_model(model, lora)
    total, trainable = count_parameters(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=1, save_strategy="epoch", report_to="none", fp16=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=grouped, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    history = [{key: value for key, value in item.items() if isinstance(value, (int, float))} for item in trainer.state.log_history]
    final_loss = next((item["loss"] for item in reversed(history) if "loss" in item), None)
    payload = {"stage": "step1_lora", "environment": environment_record(), "input": {"corpus": str(args.corpus), "raw_rows": len(raw), "training_blocks": len(grouped), "block_size": block_size}, "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["c_attn"]}, "parameters": {"total": total, "trainable": trainable, "trainable_fraction": trainable / total}, "training": {"epochs": args.epochs, "max_steps": args.max_steps, "learning_rate": 2e-4, "gradient_accumulation_steps": 8, "train_runtime_seconds": result.metrics.get("train_runtime"), "final_loss": final_loss, "final_perplexity": perplexity(final_loss) if final_loss is not None else None, "log_history": history}, "checkpoint": str(args.output_dir)}
    write_json(args.ledger, payload)
    print(f"LoRA checkpoint saved to {args.output_dir}; ledger saved to {args.ledger}")


if __name__ == "__main__":
    main()
