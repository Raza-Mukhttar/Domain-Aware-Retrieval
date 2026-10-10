# -*- coding: utf-8 -*-
"""Build the biomedical entity vocabulary used for retention scoring and reranking.

scispaCy `en_ner_bc5cdr_md` is run over the training answers (or their summaries, which
is what the executed build used) and every distinct entity string longer than two
characters is kept, lower-cased. The result is the fallback vocabulary in
data/derived/medical_entities.json: when scispaCy is importable the pipeline tags text
directly and does not need it, and which path a run took is recorded in its manifest.

The vocabulary is phrase-level, so 64.4% of its entries contain a space. That matters:
matching it token-by-token makes those entries unreachable, which is why
medlm.entities matches normalised n-grams instead.

Install the model first, as it is not on PyPI:
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

Usage:
  python extract_entity_vocabulary.py --dataset medquad --source summary
"""
import argparse
import json
import os

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))

from medlm import paths
from medlm.io_utils import read_jsonl, strip_echo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--split", default="train")
    ap.add_argument("--source", default="summary", choices=["summary", "answer"],
                    help="'summary' reproduces the executed build, which tagged the "
                         "summaries; 'answer' tags the original answers instead")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import spacy
    nlp = spacy.load("en_ner_bc5cdr_md")

    spec = paths.DATASETS[a.dataset]
    rows = read_jsonl(os.path.join(spec["dir"], f"{a.split}.jsonl"))
    texts = [r.get("answer", "") for r in rows]
    if a.source == "summary":
        if not os.path.exists(spec["summaries"]):
            raise SystemExit(f"no summaries at {spec['summaries']}; run make_answer_summaries.py "
                             "first or pass --source answer")
        with open(spec["summaries"], encoding="utf-8") as f:
            summ = json.load(f)
        texts = [strip_echo(summ[i]) if i < len(summ) else texts[i] for i in range(len(rows))]

    vocab = set()
    for i, t in enumerate(texts):
        if not t:
            continue
        for e in nlp(t).ents:
            s = e.text.strip().lower()
            if len(s) > 2:
                vocab.add(s)
        if i % 2000 == 0:
            print(f"  {i}/{len(texts)} texts, {len(vocab)} entities", flush=True)

    out = a.out or paths.data("derived", "medical_entities.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    entries = sorted(vocab)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    multi = sum(1 for e in entries if " " in e)
    print(f"[done] {len(entries)} entities -> {out}")
    print(f"       {multi} ({100*multi/max(len(entries),1):.1f}%) contain a space, so the "
          f"vocabulary has to be matched as phrases")


if __name__ == "__main__":
    main()
