"""Build the open-ended splits from the Hugging Face releases.

Produces the three JSONL files per corpus that the pipeline reads, with one object per
line and the fields {question, context, answer}. A dataset that ships a single split is
divided 80/10/10 **in its original row order**, not shuffled, so the boundaries are a
property of the release rather than of a seed; rows with an empty question or answer are
dropped after the split is taken.

  MedQuAD        lavita/MedQuAD              -> data/medquad/
  iCliniq-style  lavita/ChatDoctor-iCliniq   -> data/icliniq/

The iCliniq answer column matters: --answer_col selects it, and without that flag the
order of preference is answer_chatdoctor, answer_chatgpt, answer_icliniq, so a model's
answer would be taken as the reference ahead of the physician's. Pass
--answer_col answer_icliniq for the physician answer.

Usage:
  python ingest_open_ended.py --dataset lavita/MedQuAD --outdir ../data/medquad
  python ingest_open_ended.py --dataset lavita/ChatDoctor-iCliniq \
      --answer_col answer_icliniq --outdir ../data/icliniq
"""
import argparse
import json
import os
import re
from typing import Dict, List, Optional

from datasets import load_dataset


def save_jsonl(records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def map_chatdoctor_row(row, answer_col_preferred: Optional[str] = None):
    """
    Heuristic mapper for ChatDoctor-iCliniq rows.
    Known fields: 'input', 'answer_icliniq', 'answer_chatgpt', 'answer_chatdoctor', etc.
    Prefer explicit answer_col_preferred if provided.
    """
    q = row.get("input") or row.get("question") or row.get("query") or ""
    # choose answer column
    if answer_col_preferred and answer_col_preferred in row:
        a = row.get(answer_col_preferred)
    else:
        # priority: answer_chatdoctor (model), answer_chatgpt, answer_icliniq (physician)
        a = row.get("answer_chatdoctor") or row.get("answer_chatgpt") or row.get("answer_icliniq") or row.get("answer")
    # no explicit context in this dataset
    c = ""
    return {"question": normalize_text(q), "context": normalize_text(c), "answer": normalize_text(a or "")}


def map_medquad_row(row):
    """
    Heuristic mapper for MedQuAD rows.
    Typical MedQuAD columns from original MedQuAD: 'question', 'answer', 'doc_id', 'source' etc.
    Fallback heuristics included.
    """
    q = row.get("question") or row.get("query") or row.get("question_text") or ""
    # try common MEDQUAD columns
    a = row.get("answer") or row.get("answer_text") or row.get("final_answer") or row.get("target") or ""
    # sometimes MedQuAD has 'context' or 'snippet'
    c = row.get("context") or row.get("snippet") or row.get("passage") or ""
    meta = {}
    # copy common meta fields if present
    for k in ("source", "doc_id", "source_url"):
        if k in row:
            meta[k] = row[k]
    return {"question": normalize_text(q), "context": normalize_text(c), "answer": normalize_text(a), "meta": meta}


def ingest_dataset(dataset_id: str, outdir: str, answer_col: Optional[str] = None, split_strategy: str = "autopct"):
    """
    dataset_id: HF dataset id e.g. "lavita/ChatDoctor-iCliniq"
    outdir: output directory to store processed jsonl
    answer_col: optional explicit column to take as answer
    split_strategy: "autopct" means if single split present, do 80/10/10 split
    """
    print(f"Loading dataset {dataset_id} from Hugging Face...")
    ds = load_dataset(dataset_id)
    # pick available split(s)
    splits = list(ds.keys())
    print("Detected splits:", splits)

    # flatten rows -> list of dicts
    rows_all = []
    # if dataset already has train/validation/test, prefer those; otherwise split.
    if "train" in ds and ("validation" in ds or "test" in ds):
        # collect by split
        # We'll still standardize each split separately
        for s in ds:
            rows_all.append((s, list(ds[s])))
    elif len(splits) == 1:
        # one split provided (common). We'll create train/val/test splits.
        all_rows = list(ds[splits[0]])
        n = len(all_rows)
        if n < 3:
            raise ValueError("Dataset too small to split automatically.")
        # deterministic split: keep original order
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        rows_all = [
            ("train", all_rows[:train_end]),
            ("validation", all_rows[train_end:val_end]),
            ("test", all_rows[val_end:]),
        ]
    else:
        # multiple splits with names; process each as-is
        for s in splits:
            rows_all.append((s, list(ds[s])))

    # Choose mapping function per dataset name heuristic
    name_lower = dataset_id.lower()
    mapped_splits = {}
    for split_name, rows in rows_all:
        mapped = []
        for r in rows:
            # Hugging Face dataset row may be a dict-like; convert to plain dict
            row = dict(r)
            if "chatdoctor" in name_lower or "icliniq" in name_lower:
                rec = map_chatdoctor_row(row, answer_col_preferred=answer_col)
            elif "medquad" in name_lower:
                rec = map_medquad_row(row)
            else:
                # generic fallback: try to detect fields
                q = row.get("question") or row.get("input") or row.get("query") or ""
                a = row.get("answer") or row.get("output") or row.get("response") or ""
                c = row.get("context") or row.get("passage") or ""
                rec = {"question": normalize_text(q), "context": normalize_text(c), "answer": normalize_text(a)}
            # skip empty answers/questions
            if rec["question"] and rec.get("answer"):
                mapped.append(rec)
        mapped_splits[split_name] = mapped
        print(f"Mapped split {split_name}: {len(mapped)} examples")

    # write out standardized jsonl per dataset under outdir
    os.makedirs(outdir, exist_ok=True)
    for split_name, recs in mapped_splits.items():
        fname = os.path.join(outdir, f"{split_name}.jsonl")
        save_jsonl(recs, fname)
        print(f"Wrote {len(recs)} -> {fname}")

    return True


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id, e.g. lavita/ChatDoctor-iCliniq")
    parser.add_argument("--outdir", required=True, help="Output directory to save processed jsonl")
    parser.add_argument("--answer_col", default=None, help="(optional) explicit answer column to prefer")
    args = parser.parse_args()
    ingest_dataset(args.dataset, args.outdir, answer_col=args.answer_col)


if __name__ == "__main__":
    _cli()
