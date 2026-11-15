# Teaching DistilGPT2 Cebuano

This repository is a reproducible, CPU-friendly implementation of the midterm project. It is deliberately instrumented: every step writes a JSON ledger explaining where the reported numbers came from.

## What is measured

- **Step 0:** tokenizer vocabulary size, token IDs, token count per word, tokens per character, parameter count, prompt length, generation time, and baseline generations.
- **Step 1:** source rows, tokenized blocks, block size, LoRA rank/alpha/dropout, total and trainable parameters, trainable fraction, optimizer settings, logged loss, and perplexity using $PPL = e^{loss}$.
- **Step 2:** greedy, top-k, top-p, and beam outputs with exact settings, generated token count, wall-clock latency, throughput, and a cache ablation using `uncached_seconds / cached_seconds`.
- **Step 3:** preference-pair row count, placeholder validation, DPO hyperparameters, runtime, and training logs.
- **Step 4:** the report template records the architectural claims and sources that must be cited.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands from the repository root. The first model or dataset download can be large and requires internet access.

## Reproducible run order

### Step 0: baseline

```powershell
python src/step0_baseline.py
```

This produces `outputs/step0_baseline.json`. Preserve it before training.

### Step 1: build the corpus

```powershell
python src/prepare_corpus.py
```

This downloads `josephimperial/CebuaNER` and `filbench/cebuano-readability`, records the observed columns, parses CebuaNER's CoNLL-style `token -X- surface_form NER_tag` rows, discards the labels, and reconstructs text units at the observed blank-line boundaries (with `-DOCSTART-` handled explicitly). It writes:

- `data/cebuano_corpus.txt`
- `outputs/corpus_audit.json`

The audit is important because dataset schemas can change. It exposes the actual row count, fields, documents, characters, approximate words, and UTF-8 bytes used in the run.

### Step 1: LoRA training

```powershell
python src/step1_lora.py --epochs 1 --max-steps 200
```

The default is intentionally conservative for a laptop. Increase `--max-steps` only after a short smoke run works. The checkpoint is written under `checkpoints/step1_lora`; the ledger is `outputs/step1_lora.json`.

The causal-LM block calculation is:

1. tokenize each text row;
2. concatenate token IDs;
3. discard the remainder shorter than `block_size`;
4. split the remaining stream into fixed blocks;
5. train on next-token labels produced by the data collator.

Therefore `training_blocks` is an observed count, not a guess.

### Step 2: decoding and KV cache

```powershell
python src/step2_decode.py
```

The script uses the same prompts across strategies. Greedy has no sampling; top-k uses `k=40`, top-p uses `p=0.9`, both sampling modes use `temperature=0.8`, and beam search uses four beams. The cache comparison holds the prompt, decoder, and token budget constant while changing only `use_cache`.

### Step 3: DPO

```powershell
Copy-Item data/preferences.template.csv data/preferences.csv
# Replace every placeholder with original Cebuano pairs authored and reviewed by your group.
python src/step3_dpo.py --max-steps 30
```

The script refuses to run while placeholders remain. Each row must have `prompt`, `chosen`, and `rejected`, and the chosen/rejected distinction must represent one narrow behavior such as formal versus casual register. Keep a second copy of your annotation rationale in the report.

### Report metrics

```powershell
python src/report_metrics.py
```

This turns the saved ledgers into report-ready arithmetic, including trainable fraction, perplexity, and mean cache speedup.

### Generate graphs and explanations

```powershell
python src/make_plots.py
```

This writes PNG figures into `outputs/figures/` and a narrated figure sheet to `outputs/FIGURES.md`. The figure sheet explains the source field, transformation, and interpretation for each chart:

1. `01_corpus_audit.png`: rows observed versus texts retained for news and readability sources.
2. `02_tokenization.png`: BPE token counts and tokens-per-character for Cebuano and English comparison words.
3. `03_training_curve.png`: logged LoRA loss and derived perplexity, where `PPL = exp(loss)`.
4. `04_decoding_latency.png`: mean latency and throughput for greedy, top-k, top-p, and beam decoding.
5. `05_kv_cache_speedup.png`: `uncached_seconds / cached_seconds` for each prompt.
6. `06_dpo_curve.png`: DPO loss when a completed preference ledger is available.

Do not run the graph step until the relevant pipeline outputs exist; it intentionally stops at the first missing prerequisite rather than drawing fabricated or empty charts.

### Open the Excel calculation workbook

```powershell
python src/make_excel.py
```

This creates `outputs/cebuano_project_calculations.xlsx`. It contains the `Guide`, `Corpus Data`, `Architecture`, `Weight Calculations`, `Forward Pass`, `Training Metrics`, `Decoding Metrics`, `Graphs`, and `Sources` sheets. The weight sheet calculates embedding, attention, feed-forward, normalization, and LoRA matrix sizes; the graph sheet contains parameter-distribution, frozen-versus-trainable, corpus-composition, and KV-cache charts. Orange cells are reserved for measured loss and latency values from completed training runs.

## Artifact map

| Assignment item | Artifact |
| --- | --- |
| Baseline evidence | `outputs/step0_baseline.json` |
| Corpus reconstruction | `src/prepare_corpus.py`, `outputs/corpus_audit.json` |
| LoRA checkpoint and loss | `checkpoints/step1_lora`, `outputs/step1_lora.json` |
| Four decoders and KV cache | `outputs/step2_decode.json` |
| Original preference data | `data/preferences.csv` |
| DPO checkpoint | `checkpoints/step3_dpo`, `outputs/step3_dpo.json` |
| Written analysis | `REPORT.md` |

## Important interpretation limits

This is continued-text causal-language-model adaptation, not instruction tuning. A question prompt is not evidence that the model learned to answer questions. Compare identical prompts before and after each stage, report settings beside every sample, and have a Cebuano speaker review qualitative claims. Do not claim that a lower training loss alone proves fluent Cebuano generation; use the baseline, held-out checks if added, and human review together.
