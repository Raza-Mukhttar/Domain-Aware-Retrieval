#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A - Offline Preparation and Retrieval (Section 3.2 of the paper).

For the training pairs D = {(q_i, a_i)}, this phase produces everything the
online phase needs and nothing that depends on the query:

  s_i                    answer summary               Eq. (3)
  E(q_i), E(a_i)         biomedical entity sets       Eq. (4)
  t_i = T(q_i)           deterministic question type  Eq. (5)
  h_i = g(q_i)           frozen sentence embedding    Eq. (6)
  type-specific FAISS IndexFlatIP index built once    Eqs. (7)-(8)

The answer summaries are produced by a separate generative pass and are shipped
with the repository under data/derived, because regenerating them requires the
summariser backbone. Everything else in this phase is deterministic and is
rebuilt here from the splits.

Usage:
  python phase_a_offline_preparation.py --dataset medquad --n_train 2000 --seed 42
"""
import argparse
import json
import os
import time

from medlm import paths
from medlm.entities import extract_entities, load_vocab
from medlm.io_utils import load_split, set_seed, write_jsonl
from medlm.retrieval import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--n_train", type=int, default=2000,
                    help="size of the retrieval pool; 0 uses the whole training split")
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--first_test_rows", action="store_true",
                    help="evaluate the first n_test rows in file order (sampling rule of the "
                         "submitted runs) instead of a seeded random sample")
    ap.add_argument("--use_question_type", action="store_true", default=True)
    ap.add_argument("--no_question_type", dest="use_question_type", action="store_false")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if _cuda() else "cpu")
    outdir = args.out or paths.output("phase_a", f"{args.dataset}_seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    train, test = load_split(args.dataset, args.n_train, args.n_test, args.seed,
                             args.first_test_rows)
    print(f"[data] {args.dataset}: retrieval pool {len(train)}, evaluation set {len(test)}",
          flush=True)

    vocab = load_vocab(paths.ENTITY_VOCAB)
    print(f"[entities] vocabulary {len(vocab)} entries", flush=True)

    from collections import Counter
    types = Counter(t["qtype"] for t in train)
    print(f"[types] {len(types)} buckets: "
          + ", ".join(f"{k}={v}" for k, v in types.most_common()), flush=True)

    for row in train:
        row["entities_question"] = sorted(extract_entities(row["question"], vocab))
        row["entities_answer"] = sorted(extract_entities(row["answer"], vocab))
    for row in test:
        row["entities_question"] = sorted(extract_entities(row["question"], vocab))

    print(f"[index] encoding {len(train)} pool questions on {device}", flush=True)
    retriever = Retriever(train, device, args.use_question_type)
    retriever.save(os.path.join(outdir, "index"))

    write_jsonl(os.path.join(outdir, "pool.jsonl"), train)
    write_jsonl(os.path.join(outdir, "evaluation.jsonl"), test)
    manifest = dict(
        phase="A", dataset=args.dataset, seed=args.seed,
        n_pool=len(train), n_eval=len(test),
        use_question_type=args.use_question_type,
        test_sampling="first_rows" if args.first_test_rows else "random_seed",
        embedder=paths.EMBEDDER, index="FAISS IndexFlatIP, L2-normalised",
        question_types={k: v for k, v in types.most_common()},
        entity_vocabulary_size=len(vocab),
        summaries_available=os.path.exists(paths.DATASETS[args.dataset]["summaries"]),
        seconds=round(time.time() - t0, 1))
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] phase A -> {outdir} in {time.time()-t0:.0f}s", flush=True)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
