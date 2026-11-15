from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import count_parameters, environment_record, project_root, timed_call, write_json

PROMPTS = ["Kumusta ka?", "Ang Cebuano nga pinulongan", "Karon sa buntag, ang mga tawo", "Ang kinabuhi sa Sugbo"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the untouched DistilGPT2 baseline.")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--output", type=Path, default=project_root() / "outputs" / "step0_baseline.json")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    total, trainable = count_parameters(model)
    token_stats = []
    for word in ["Cebuano", "pinulongan", "kalipay", "language", "beautiful"]:
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        token_stats.append({"text": word, "token_ids": ids, "token_count": len(ids), "characters": len(word), "tokens_per_character": len(ids) / len(word)})
    generations = []
    for prompt in PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output, seconds = timed_call(model.generate, **encoded, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generations.append({"prompt": prompt, "input_tokens": int(encoded["input_ids"].shape[-1]), "seconds": seconds, "text": tokenizer.decode(output[0], skip_special_tokens=True)})
    payload = {"stage": "step0_baseline", "model": args.model, "environment": environment_record(), "parameters": {"total": total, "trainable": trainable}, "tokenizer": {"vocab_size": tokenizer.vocab_size, "token_stats": token_stats}, "generation": {"max_new_tokens": args.max_new_tokens, "strategy": "greedy", "samples": generations}, "interpretation": "Compare Cebuano token counts and language drift against later checkpoints; this is not a benchmark score."}
    write_json(args.output, payload)
    print(f"Baseline saved to {args.output}")


if __name__ == "__main__":
    main()
