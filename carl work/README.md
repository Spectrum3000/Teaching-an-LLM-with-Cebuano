# Carl Work - Week 1

This folder is a local, unpushed work area based on the historical `file_data_prep.ipynb.ipynb` notebook and the shared Claude Week 1 Member A guide.

## Week 1 scope

- Data-preparation reference: verify the two live Hugging Face dataset schemas and the CebuaNER reconstruction rule
- Step 0: baseline DistilGPT2 generation quality
- Step 1: LoRA fine-tuning, loss curve, and before/after generation

## Structure

```text
carl work/
├── notebooks/
│   ├── file_data_prep_reference.ipynb
│   ├── step0_baseline.ipynb
│   └── step1_lora.ipynb
├── checkpoints/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

The notebooks locate the shared corpus at `../data/cebuano_corpus.txt` from the repository root. They do not push, commit, or modify GitHub. Run the data-prep reference cells first if you need to verify the source datasets, then run Step 0 and Step 1.

Use a short smoke run before a full CPU run. Record the model version, dataset split, chunk size, seed, runtime, loss, and checkpoint path in the report.
