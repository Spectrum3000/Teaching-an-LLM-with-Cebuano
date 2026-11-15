from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from common import environment_record, project_root, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DPO on human-authored Cebuano preference pairs.")
    parser.add_argument("--base-model", default="distilgpt2")
    parser.add_argument("--adapter", type=Path, default=project_root() / "checkpoints" / "step1_lora")
    parser.add_argument("--preferences", type=Path, default=project_root() / "data" / "preferences.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "checkpoints" / "step3_dpo")
    parser.add_argument("--ledger", type=Path, default=project_root() / "outputs" / "step3_dpo.json")
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()
    if not args.preferences.exists():
        raise FileNotFoundError(f"Copy data/preferences.template.csv to {args.preferences} and replace every placeholder with original, native-authored pairs.")
    dataset = load_dataset("csv", data_files=str(args.preferences))["train"]
    placeholders = sum("REPLACE_WITH" in str(row) for row in dataset)
    if placeholders:
        raise ValueError(f"Preference file still contains {placeholders} placeholder rows; DPO requires real human-authored Cebuano pairs.")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    config = DPOConfig(output_dir=str(args.output_dir), max_steps=args.max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=5e-6, logging_steps=1, report_to="none", max_length=256, max_prompt_length=128)
    trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=dataset, processing_class=tokenizer)
    result = trainer.train()
    trainer.save_model(args.output_dir)
    logs = [{key: value for key, value in row.items() if isinstance(value, (int, float))} for row in trainer.state.log_history]
    write_json(args.ledger, {"stage": "step3_dpo", "environment": environment_record(), "dataset": {"path": str(args.preferences), "rows": len(dataset), "placeholder_rows": placeholders}, "training": {"max_steps": args.max_steps, "learning_rate": 5e-6, "runtime_seconds": result.metrics.get("train_runtime"), "log_history": logs}, "checkpoint": str(args.output_dir)})
    print(f"DPO checkpoint saved to {args.output_dir}; ledger saved to {args.ledger}")


if __name__ == "__main__":
    main()
