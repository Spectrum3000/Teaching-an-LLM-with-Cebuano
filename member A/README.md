# Member A: ML Engineer Workstream

This folder implements the Member A responsibilities from the shared group guide:

- Step 0: baseline generation quality using untouched `distilgpt2`
- Step 1: LoRA fine-tuning, loss curve, and before/after generation
- Step 3: DPO fine-tuning and before/after behavior comparison

## Structure

```text
member A/
├── notebooks/
│   ├── step0_baseline.ipynb
│   ├── step1_lora.ipynb
│   └── step3_dpo.ipynb
├── preference_pairs/
│   └── dpo_pairs.template.csv
├── checkpoints/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

The notebooks expect the shared corpus at `../data/cebuano_corpus.txt` relative to this folder. If the Member B corpus is stored somewhere else, change the `CORPUS_PATH` cell in `step1_lora.ipynb`.

## Run order

1. Run every cell in `step0_baseline.ipynb`. Save `outputs/step0_generations.txt` and the tokenizer observations.
2. Run `step1_lora.ipynb` after the corpus is available. It creates `checkpoints/lora_step1`, `outputs/loss_curve.png`, `outputs/step1_training_log.json`, and before/after generations.
3. Copy `preference_pairs/dpo_pairs.template.csv` to `dpo_pairs.csv`. Replace all placeholders with 30 original Cebuano preference pairs. Each group member contributes 10 pairs and all members review every pair.
4. Run `step3_dpo.ipynb`. It creates `checkpoints/dpo_step3` and `outputs/step3_before_after.txt`.

Do not commit model checkpoint binaries unless the group has explicitly chosen external storage. Commit the notebooks, plots, JSON/text logs, and a README link to the checkpoint location.

## CPU settings

The notebooks use batch size 1 or 2, gradient accumulation, `fp16=False`, and short smoke-run controls. Run a small smoke test first, then increase `MAX_STEPS` or epochs for the final experiment. Record hardware, runtime, loss, and exact settings in the report.
