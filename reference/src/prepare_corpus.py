from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from common import environment_record, first_present, project_root, write_json


def as_words(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return value.split()
    return []


def reconstruct_ner(rows: Any) -> tuple[list[str], dict[str, Any]]:
    documents: list[str] = []
    field_counts: Counter[str] = Counter()
    current_tokens: list[str] = []
    doc_boundaries = 0
    blank_boundaries = 0

    def flush_document() -> None:
        if current_tokens:
            documents.append(" ".join(current_tokens))
            current_tokens.clear()

    for row in rows:
        row_dict = dict(row)
        field_counts.update(row_dict.keys())
        raw_line = first_present(row_dict, ("text", "tokens", "words", "word"))
        if not isinstance(raw_line, str):
            continue
        line = raw_line.strip()
        if line.startswith("-DOCSTART-"):
            flush_document()
            doc_boundaries += 1
            continue
        if not line:
            flush_document()
            blank_boundaries += 1
            continue
        token = line.split(" -X- ", 1)[0].strip()
        if token:
            current_tokens.append(token)
    flush_document()
    return documents, {"rows_seen": len(rows), "candidate_fields": dict(field_counts), "documents_reconstructed": len(documents), "docstart_boundaries": doc_boundaries, "blank_boundaries": blank_boundaries, "format": "CoNLL-style token -X- surface_form NER_tag; labels discarded; blank lines delimit documents"}


def readability_text(rows: Any) -> tuple[list[str], dict[str, Any]]:
    passages: list[str] = []
    field_counts: Counter[str] = Counter()
    for row in rows:
        row_dict = dict(row)
        field_counts.update(row_dict.keys())
        text = first_present(row_dict, ("text", "passage", "content"))
        if isinstance(text, str) and text.strip():
            passages.append(text.strip())
    return passages, {"rows_seen": len(rows), "candidate_fields": dict(field_counts), "passages_kept": len(passages)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and audit the two licensed Cebuano datasets.")
    parser.add_argument("--output", type=Path, default=project_root() / "data" / "cebuano_corpus.txt")
    parser.add_argument("--audit", type=Path, default=project_root() / "outputs" / "corpus_audit.json")
    parser.add_argument("--ner-config", default=None, help="Optional CebuaNER dataset config.")
    args = parser.parse_args()

    ner_kwargs = {"name": args.ner_config} if args.ner_config else {}
    ner = load_dataset("josephimperial/CebuaNER", **ner_kwargs)
    readability = load_dataset("filbench/cebuano-readability")
    ner_split = ner["train"] if "train" in ner else next(iter(ner.values()))
    readability_split = readability["train"] if "train" in readability else next(iter(readability.values()))
    ner_documents, ner_audit = reconstruct_ner(ner_split)
    passages, readability_audit = readability_text(readability_split)
    documents = [text for text in ner_documents + passages if text.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(documents) + "\n", encoding="utf-8")
    audit = {"environment": environment_record(), "sources": {"josephimperial/CebuaNER": ner_audit, "filbench/cebuano-readability": readability_audit}, "output": {"path": str(args.output), "documents": len(documents), "characters": sum(len(document) for document in documents), "words_approx": sum(len(document.split()) for document in documents), "bytes_utf8": args.output.stat().st_size}, "reproducibility": {"ner_config": args.ner_config}}
    write_json(args.audit, audit)
    print(f"Wrote {len(documents):,} documents and {audit['output']['characters']:,} characters to {args.output}")
    print(f"Audit ledger: {args.audit}")


if __name__ == "__main__":
    main()
