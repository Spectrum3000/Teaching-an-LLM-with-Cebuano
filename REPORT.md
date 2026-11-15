# DistilGPT2 Cebuano Midterm Report

Replace bracketed fields with values copied from the JSON ledgers. Every numerical claim should be traceable to a file and a formula.

## 1. Baseline diagnostic

**Model and environment.** Record the model identifier, Python version, platform, and total parameter count from `outputs/step0_baseline.json`.

**Tokenizer method.** For each comparison word, the tokenizer returns a list of integer BPE IDs. Token count is `len(token_ids)` and token density is:

`tokens_per_character = token_count / number_of_characters`.

Table to complete:

| Text | Characters | Token IDs | Tokens | Tokens/character |
| --- | ---: | --- | ---: | ---: |
| Cebuano word 1 | [ ] | [ ] | [ ] | [ ] |
| Cebuano word 2 | [ ] | [ ] | [ ] | [ ] |
| English comparison 1 | [ ] | [ ] | [ ] | [ ] |
| English comparison 2 | [ ] | [ ] | [ ] | [ ] |

These counts show tokenizer segmentation, not language understanding. Include the exact baseline generations and explain language drift without presenting a subjective judgment as an objective score.

## 2. Corpus and LoRA adaptation

### Corpus reconstruction

Cite both dataset cards and their licenses. From `outputs/corpus_audit.json`, report rows observed, candidate columns, CebuaNER `-DOCSTART-` and blank-line boundary counts, reconstructed CebuaNER documents, readability passages, documents written, characters, approximate words, and UTF-8 bytes. Explain that each CebuaNER row is parsed as `token -X- surface_form NER_tag`, the NER labels are discarded, and the token stream is reconstructed at the blank-line boundaries present in the downloaded file.

### Training arithmetic

From `outputs/step1_lora.json`, report raw rows, block size, training blocks, total parameters, trainable parameters, and trainable fraction.

`trainable_fraction = trainable_parameters / total_parameters`.

For final loss $L$, report perplexity using:

$$PPL = \exp(L)$$

Do not fabricate a loss curve. Copy the `log_history` values into a plot and label the x-axis as the trainer's recorded step. Discuss whether the curve is smooth, noisy, or overfit.

### Scaling-law reasoning

Explain that LoRA updates low-rank adapter matrices while the pretrained weights remain frozen. Contrast this adaptation cost with full pretraining. Cite Kaplan et al. (2020) and Hoffmann et al. (2022/Chinchilla) from the papers you actually read; do not cite them as evidence for a specific local loss value.

## 3. Decoding and inference

Use `outputs/step2_decode.json`. Include one table with the same prompt and token budget under greedy, top-k, top-p, and beam search. Report exact settings, generated token count, seconds, and tokens/second.

Run `python src/make_plots.py` after the completed stages. Insert the generated figures from `outputs/figures/` and use `outputs/FIGURES.md` as the explanation source. The latency figure is an arithmetic summary of each strategy's records, not a quality ranking. The tokenization graph reports segmentation density, not comprehension. The training graph plots the trainer's logs and computes perplexity from the same loss values.

For KV caching, the measured speedup is:

`speedup = uncached_seconds / cached_seconds`.

Report each prompt separately before giving a mean. Explain that wall-clock timing is affected by CPU load and should be repeated if results look unstable. Describe repetition, diversity, and fluency with a documented human-review protocol.

For the Mixture-of-Experts discussion, distinguish parameter count from active parameters per token: routing can reduce active compute, but routing, memory movement, and expert capacity introduce overhead.

## 4. DPO alignment

State exactly one target behavior. Explain how each annotator decided chosen versus rejected, how ambiguity was handled, and whether a native speaker reviewed the data. Report the final row count from `outputs/step3_dpo.json`.

Contrast classic RLHF (reward model plus PPO) with DPO's direct preference objective relative to a reference model. Report pre/post generations using the same prompts and decoding settings. A behavior claim should include a held-out prompt set and a preference count, for example `post-DPO preferred on X/Y prompts`, with the judges and protocol described.

## 5. Architecture comparison

DistilGPT2 inherits GPT-2's decoder-only transformer and learned absolute positional embeddings. Compare those with one researched alternative, RoPE or ALiBi, and distinguish short-context course behavior from long-context extrapolation claims. Compare decoder-only generation with encoder-only models for CebuaNER-style token classification: encoder-only models can use bidirectional context and task-specific classification heads, while decoder-only models are naturally suited to continuation generation.

## 6. Limitations and reproducibility

Record hardware, software versions, random seeds or seed policy, dataset revision if available, training duration, and any interrupted runs. State that the corpus is small and register-mixed, DPO pairs are group-authored, and qualitative Cebuano quality requires native-speaker review. Include the incremental Git commits and links to the Canva deck.
