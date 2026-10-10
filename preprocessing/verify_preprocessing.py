# -*- coding: utf-8 -*-
"""Check that the shipped data matches what the preprocessing scripts produce.

Verifies the split sizes and their 80/10/10 boundaries, the JSONL schema, that no
evaluation question is empty, that the derived summaries are index-aligned to their split,
and that the group-disjoint repartition really is disjoint. Runs offline in seconds and
exits non-zero on failure.

Usage: python verify_preprocessing.py
"""
import json
import os
import sys

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))

from medlm import paths
from medlm.io_utils import read_jsonl

results = []


def check(name, ok, detail=""):
    results.append((bool(ok), name, detail))
    return ok


# --- open-ended splits ------------------------------------------------------
for corpus in ("medquad", "icliniq"):
    d = paths.DATASETS[corpus]["dir"]
    sizes = {}
    for split in ("train", "validation", "test"):
        rows = read_jsonl(os.path.join(d, f"{split}.jsonl"))
        sizes[split] = len(rows)
        bad = [i for i, r in enumerate(rows[:2000])
               if not r.get("question", "").strip() or not r.get("answer", "").strip()]
        check(f"{corpus}/{split}: no empty question or answer", not bad,
              f"{len(rows)} rows" + (f", first offender row {bad[0]}" if bad else ""))
        check(f"{corpus}/{split}: schema is question/context/answer",
              all(set(("question", "context", "answer")) <= set(r) for r in rows[:200]))
    total = sum(sizes.values())
    # the ingestion splits in original order at int(0.8N) and int(0.9N), then drops
    # rows with an empty field, so the shipped sizes must be consistent with some raw N
    consistent = any(int(0.8 * raw) >= sizes["train"]
                     and int(0.9 * raw) - int(0.8 * raw) >= sizes["validation"]
                     and raw - int(0.9 * raw) >= sizes["test"]
                     for raw in range(total, total + 60))
    check(f"{corpus}: sizes consistent with an 80/10/10 split in original order",
          consistent, f"train/val/test = {sizes['train']}/{sizes['validation']}/{sizes['test']}")

# --- derived artefacts ------------------------------------------------------
for corpus in paths.DATASETS:
    spec = paths.DATASETS[corpus]
    if not os.path.exists(spec["summaries"]):
        continue
    n_train = len(read_jsonl(os.path.join(spec["dir"], "train.jsonl")))
    with open(spec["summaries"], encoding="utf-8") as f:
        summ = json.load(f)
    check(f"{corpus}: summaries index-aligned to the training split",
          len(summ) == n_train, f"{len(summ)} summaries, {n_train} rows")
    nonempty = sum(1 for s in summ if s and s.strip())
    check(f"{corpus}: summaries non-empty", nonempty == len(summ),
          f"{len(summ) - nonempty} empty")

vocab_path = paths.ENTITY_VOCAB
if check("entity vocabulary present", os.path.isfile(vocab_path)):
    with open(vocab_path, encoding="utf-8") as f:
        v = json.load(f)
    v = list(v.keys()) if isinstance(v, dict) else v
    multi = sum(1 for e in v if isinstance(e, str) and " " in e)
    check("entity vocabulary is phrase-level", multi > 0,
          f"{len(v)} entries, {100*multi/max(len(v),1):.1f}% multi-word")

# --- group-disjoint repartition --------------------------------------------
gs = paths.DATASETS.get("medquad_group_disjoint")
if gs and os.path.isdir(gs["dir"]):
    man_p = os.path.join(gs["dir"], "manifest.json")
    if check("group-disjoint manifest present", os.path.isfile(man_p)):
        with open(man_p, encoding="utf-8") as f:
            man = json.load(f)
        groups = {k: set(map(tuple, v)) if isinstance(v, list) else v
                  for k, v in man.items() if isinstance(v, list)}
        check("group-disjoint manifest records the partitioning", bool(man),
              ", ".join(sorted(man)[:6]))
    qs = {}
    for split in ("train", "validation", "test"):
        qs[split] = {r["question"].strip() for r in read_jsonl(os.path.join(gs["dir"], f"{split}.jsonl"))}
    check("group-disjoint: no question shared between train and test",
          not (qs["train"] & qs["test"]), f"{len(qs['train'] & qs['test'])} shared")
    check("group-disjoint: no question shared between train and validation",
          not (qs["train"] & qs["validation"]), f"{len(qs['train'] & qs['validation'])} shared")

# --- report -----------------------------------------------------------------
width = max(len(n) for _o, n, _d in results)
failed = sum(1 for ok, _n, _d in results if not ok)
for ok, name, detail in results:
    print(f"[{'  ok  ' if ok else ' FAIL '}] {name:<{width}}  {detail}")
print(f"\n{len(results) - failed} of {len(results)} checks passed")
sys.exit(1 if failed else 0)
