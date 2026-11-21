from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import project_root, read_json


COLORS = {"news": "#0f766e", "literary": "#d97706", "greedy": "#2563eb", "top_k": "#7c3aed", "top_p": "#db2777", "beam": "#059669"}


def save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def require(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the corresponding pipeline step first.")
    return read_json(path)


def plot_corpus(audit: dict[str, Any], out: Path) -> str:
    sources = audit["sources"]
    labels = ["CebuaNER news", "Readability prose"]
    rows = [sources["josephimperial/CebuaNER"]["rows_seen"], sources["filbench/cebuano-readability"]["rows_seen"]]
    kept = [sources["josephimperial/CebuaNER"].get("sentences_reconstructed", 0), sources["filbench/cebuano-readability"].get("passages_kept", 0)]
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = range(len(labels))
    axis.bar([position - 0.18 for position in positions], rows, width=0.36, label="Rows observed", color="#94a3b8")
    axis.bar([position + 0.18 for position in positions], kept, width=0.36, label="Texts kept", color=[COLORS["news"], COLORS["literary"]])
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("Count")
    axis.set_title("Corpus audit: observed rows versus usable text")
    axis.legend()
    for position, value in enumerate(rows):
        axis.text(position - 0.18, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    for position, value in enumerate(kept):
        axis.text(position + 0.18, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    save_figure(figure, out / "01_corpus_audit.png")
    total = audit["output"]
    return f"""### 1. Corpus audit\n\n![Corpus audit](01_corpus_audit.png)\n\nThis chart separates **rows observed** from **usable texts kept**. The first bar is the dataset row count read from the Hub. The second is the count that survived field extraction: CebuaNER rows become reconstructed token sequences, while readability rows become non-empty passages. The output ledger reports {total['documents']:,} documents, {total['characters']:,} characters, approximately {total['words_approx']:,} whitespace-delimited words, and {total['bytes_utf8']:,} UTF-8 bytes. Those are corpus measurements, not estimates of model quality.\n"""


def plot_tokenization(baseline: dict[str, Any], out: Path) -> str:
    stats = baseline["tokenizer"]["token_stats"]
    labels = [item["text"] for item in stats]
    counts = [item["token_count"] for item in stats]
    densities = [item["tokens_per_character"] for item in stats]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, counts, color=["#0f766e", "#0f766e", "#0f766e", "#64748b", "#64748b"])
    axes[0].set_title("BPE token count by word")
    axes[0].set_ylabel("Tokens")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(labels, densities, color=["#0f766e", "#0f766e", "#0f766e", "#64748b", "#64748b"])
    axes[1].set_title("Token density")
    axes[1].set_ylabel("Tokens / character")
    axes[1].tick_params(axis="x", rotation=35)
    save_figure(figure, out / "02_tokenization.png")
    highest = max(stats, key=lambda item: item["tokens_per_character"])
    return f"""### 2. Tokenizer fragmentation\n\n![Tokenizer fragmentation](02_tokenization.png)\n\nThe left panel is `len(tokenizer(word)[\"input_ids\"])`. The right panel divides that count by the Unicode character count, so words of different lengths can be compared. The highest density in this run is **{highest['text']}** at **{highest['tokens_per_character']:.3f} tokens/character**. Fragmentation is evidence about segmentation efficiency; it does not by itself prove that the language model understands a word.\n"""


def plot_training(step1: dict[str, Any], out: Path) -> str:
    history = step1["training"]["log_history"]
    loss_rows = [row for row in history if "loss" in row]
    if not loss_rows:
        raise ValueError("Step 1 ledger has no loss records yet.")
    steps = [row.get("step", index + 1) for index, row in enumerate(loss_rows)]
    losses = [row["loss"] for row in loss_rows]
    perplexities = [__import__("math").exp(value) if value < 20 else float("inf") for value in losses]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(steps, losses, marker="o", color="#2563eb")
    axes[0].set_title("LoRA training loss")
    axes[0].set_xlabel("Trainer step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(steps, perplexities, marker="o", color="#d97706")
    axes[1].set_title("Derived perplexity")
    axes[1].set_xlabel("Trainer step")
    axes[1].set_ylabel("exp(loss)")
    save_figure(figure, out / "03_training_curve.png")
    return f"""### 3. LoRA training curve\n\n![Training curve](03_training_curve.png)\n\nThe left panel plots the trainer's recorded loss values at steps {steps[0]} through {steps[-1]}. The right panel applies the exact transformation `perplexity = exp(loss)` to each recorded loss; it is not an independently measured metric. The final logged loss is **{losses[-1]:.6f}**, producing derived perplexity **{perplexities[-1]:.4f}**. A falling curve means the training objective is improving on the training blocks; it does not alone establish generalization or fluent Cebuano.\n"""


def plot_decoding(step2: dict[str, Any], out: Path) -> str:
    records = [row for row in step2["records"] if row["strategy"] != "cache_ablation"]
    strategies = ["greedy", "top_k", "top_p", "beam"]
    means = []
    throughputs = []
    for strategy in strategies:
        rows = [row for row in records if row["strategy"] == strategy]
        means.append(sum(row["seconds"] for row in rows) / len(rows))
        throughputs.append(sum(row["tokens_per_second"] for row in rows) / len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(strategies, means, color=[COLORS[strategy] for strategy in strategies])
    axes[0].set_title("Mean generation latency")
    axes[0].set_ylabel("Seconds per prompt")
    axes[1].bar(strategies, throughputs, color=[COLORS[strategy] for strategy in strategies])
    axes[1].set_title("Mean generation throughput")
    axes[1].set_ylabel("Generated tokens / second")
    save_figure(figure, out / "04_decoding_latency.png")
    cache = [row for row in step2["records"] if row["strategy"] == "cache_ablation"]
    prompts = [row["prompt"] for row in cache]
    speedups = [row["speedup"] for row in cache]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(prompts, speedups, color="#0f766e")
    axis.axhline(1, color="#64748b", linestyle="--", linewidth=1)
    axis.set_title("KV-cache speedup")
    axis.set_ylabel("Uncached seconds / cached seconds")
    axis.tick_params(axis="x", rotation=20)
    save_figure(figure, out / "05_kv_cache_speedup.png")
    return f"""### 4. Decoding latency and throughput\n\n![Decoding latency](04_decoding_latency.png)\n\nLatency is measured with `time.perf_counter()` around `model.generate()`. The prompt, model checkpoint, and maximum new-token budget are held constant while the decoding strategy changes. Throughput is calculated as `generated_tokens / seconds`. Beam search can cost more because it tracks multiple hypotheses; sampling changes selection behavior, not just output appearance.\n\n### 5. KV-cache ablation\n\n![KV-cache speedup](05_kv_cache_speedup.png)\n\nEach bar is calculated as `uncached_seconds / cached_seconds`. A value above 1 means caching was faster for that prompt. This is a wall-clock experiment, so repeat runs if CPU contention is high. The comparison is meaningful only because `use_cache` is the intended changed variable.\n"""


def plot_dpo(step3: dict[str, Any], out: Path) -> str:
    history = step3["training"]["log_history"]
    loss_rows = [row for row in history if "loss" in row]
    if not loss_rows:
        return "### 6. DPO training\n\nNo DPO loss rows were found in the ledger. Run DPO with a completed, human-authored preference CSV first.\n"
    steps = [row.get("step", index + 1) for index, row in enumerate(loss_rows)]
    losses = [row["loss"] for row in loss_rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, losses, marker="o", color="#7c3aed")
    axis.set_title("DPO training loss")
    axis.set_xlabel("Trainer step")
    axis.set_ylabel("Loss")
    save_figure(figure, out / "06_dpo_curve.png")
    return f"""### 6. DPO training\n\n![DPO curve](06_dpo_curve.png)\n\nThis curve uses the DPO trainer's logged loss values, with **{step3['dataset']['rows']}** preference rows. It is valid only after the placeholder check passes. The curve should be interpreted alongside held-out before/after generations and human preference counts, not as proof of alignment by itself.\n"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detailed figures and explanations from pipeline ledgers.")
    parser.add_argument("--outputs", type=Path, default=project_root() / "outputs")
    args = parser.parse_args()
    out = args.outputs / "figures"
    sections = []
    sections.append(plot_corpus(require(args.outputs / "corpus_audit.json"), out))
    sections.append(plot_tokenization(require(args.outputs / "step0_baseline.json"), out))
    sections.append(plot_training(require(args.outputs / "step1_lora.json"), out))
    sections.append(plot_decoding(require(args.outputs / "step2_decode.json"), out))
    dpo_path = args.outputs / "step3_dpo.json"
    if dpo_path.exists():
        sections.append(plot_dpo(read_json(dpo_path), out))
    else:
        sections.append("### 6. DPO training\n\nThe DPO ledger does not exist yet. Complete Step 3 before plotting its curve.\n")
    (args.outputs / "FIGURES.md").write_text("# Generated Figures\n\n" + "\n".join(sections), encoding="utf-8")
    print(f"Figures written to {out}")
    print(f"Explanations written to {args.outputs / 'FIGURES.md'}")


if __name__ == "__main__":
    main()
