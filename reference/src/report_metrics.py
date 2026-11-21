from __future__ import annotations

import argparse
from pathlib import Path

from common import perplexity, project_root, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Print report-ready explanations for every measured number.")
    parser.add_argument("--step1", type=Path, default=project_root() / "outputs" / "step1_lora.json")
    parser.add_argument("--step2", type=Path, default=project_root() / "outputs" / "step2_decode.json")
    args = parser.parse_args()
    step1 = read_json(args.step1)
    print("STEP 1 NUMBER LEDGER")
    print(f"Raw rows -> training blocks: {step1['input']['raw_rows']} -> {step1['input']['training_blocks']}")
    print(f"Trainable fraction = {step1['parameters']['trainable']} / {step1['parameters']['total']} = {step1['parameters']['trainable_fraction']:.6%}")
    if step1["training"]["final_loss"] is not None:
        loss = step1["training"]["final_loss"]
        print(f"Perplexity = exp(loss) = exp({loss:.6f}) = {perplexity(loss):.4f}")
    if args.step2.exists():
        step2 = read_json(args.step2)
        cache_rows = [row for row in step2["records"] if row["strategy"] == "cache_ablation"]
        speedups = [row["speedup"] for row in cache_rows if row["speedup"] is not None]
        if speedups:
            print(f"KV speedups = {speedups}; mean = {sum(speedups) / len(speedups):.4f}")


if __name__ == "__main__":
    main()
