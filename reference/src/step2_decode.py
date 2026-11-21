from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import environment_record, project_root, write_json

PROMPTS = ["Kumusta ka?", "Ang Cebuano nga pinulongan", "Karon sa buntag"]


def generate(model: object, tokenizer: object, prompt: str, strategy: str, use_cache: bool, max_new_tokens: int) -> tuple[str, float, int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    kwargs = {"max_new_tokens": max_new_tokens, "use_cache": use_cache, "pad_token_id": tokenizer.eos_token_id}
    if strategy == "greedy":
        kwargs.update(do_sample=False)
    elif strategy == "top_k":
        torch.manual_seed(7)
        kwargs.update(do_sample=True, top_k=40, temperature=0.8)
    elif strategy == "top_p":
        torch.manual_seed(7)
        kwargs.update(do_sample=True, top_p=0.9, temperature=0.8)
    elif strategy == "beam":
        kwargs.update(do_sample=False, num_beams=4, early_stopping=True)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**encoded, **kwargs)
    seconds = time.perf_counter() - start
    return tokenizer.decode(output[0], skip_special_tokens=True), seconds, int(output.shape[-1] - encoded["input_ids"].shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare four decoders and measure KV-cache latency.")
    parser.add_argument("--model", type=Path, default=project_root() / "checkpoints" / "step1_lora")
    parser.add_argument("--output", type=Path, default=project_root() / "outputs" / "step2_decode.json")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = []
    for prompt in PROMPTS:
        for strategy in ("greedy", "top_k", "top_p", "beam"):
            text, seconds, generated_tokens = generate(model, tokenizer, prompt, strategy, True, args.max_new_tokens)
            records.append({"prompt": prompt, "strategy": strategy, "use_cache": True, "seconds": seconds, "generated_tokens": generated_tokens, "tokens_per_second": generated_tokens / seconds if seconds else None, "text": text})
        cache_times = {}
        for use_cache in (True, False):
            _, seconds, generated_tokens = generate(model, tokenizer, prompt, "greedy", use_cache, args.max_new_tokens)
            cache_times[str(use_cache)] = {"seconds": seconds, "generated_tokens": generated_tokens}
        records.append({"prompt": prompt, "strategy": "cache_ablation", "cache_times": cache_times, "speedup_formula": "uncached_seconds / cached_seconds", "speedup": cache_times["False"]["seconds"] / cache_times["True"]["seconds"] if cache_times["True"]["seconds"] else None})
    write_json(args.output, {"stage": "step2_decode", "environment": environment_record(), "model": str(args.model), "max_new_tokens": args.max_new_tokens, "records": records, "analysis_note": "Sampling outputs are stochastic; record exact settings and seed policy before comparing quality."})
    print(f"Decoding and cache ledger saved to {args.output}")


if __name__ == "__main__":
    main()
