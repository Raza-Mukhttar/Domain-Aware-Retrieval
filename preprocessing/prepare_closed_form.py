# -*- coding: utf-8 -*-
"""Materialise the four closed-form benchmarks, with the answer key validated.

The multiple-choice benchmarks are pulled from the Hub at run time rather than stored,
so this script exists to make that preprocessing inspectable and to write the splits out
as JSONL if you want them on disk. Each loader fixes a specific way the key can be
unusable:

  MedQA     GBaker/MedQA-USMLE-4-options   gold from answer_idx; 4 options
  MMLU-Med  cais/mmlu                      six medical subjects concatenated in a fixed
                                           order; dev+validation form the pool (153 rows),
                                           test is 1,089
  MedMCQA   openlifescienceai/medmcqa      evaluated on VALIDATION: the test split ships
                                           cop = -1 for all 6,150 rows, so a loader that
                                           clamps it to 0 silently makes every gold "A"
  PubMedQA  qiaojin/PubMedQA pqa_labeled    yes/no/maybe from final_decision; the abstract
                                           is the context; split 80/20 by a seeded
                                           permutation

`assert_labels_usable` then aborts if a key has fewer than two classes or a majority class
above 95%, which is what the earlier runs lacked: their MedMCQA accuracies were the rate
at which a model answered "A", and their PubMedQA accuracies the rate of "yes".

Usage:
  python prepare_closed_form.py --check                 # validate every key, write nothing
  python prepare_closed_form.py --dataset medqa --out ../data/medqa
"""
import argparse
import collections
import json
import os

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))

import closedform_phase_c_inference_and_selection as CF


def report(name, rows):
    c = collections.Counter(r["gold"] for r in rows)
    n = sum(c.values())
    maj = max(c.values()) / n
    prev = {k: round(100 * v / n, 1) for k, v in sorted(c.items())}
    print(f"  {name:12s} n={n:6d} classes={len(c)} majority={100*maj:5.1f}%  prevalence={prev}")
    return maj, len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CF.LOADERS))
    ap.add_argument("--check", action="store_true", help="validate the keys and exit")
    ap.add_argument("--out", default=None, help="directory to write train/test JSONL into")
    a = ap.parse_args()

    names = [a.dataset] if a.dataset else list(CF.LOADERS)
    bad = []
    for name in names:
        train, test, n_ch = CF.LOADERS[name]()
        print(f"[{name}] {n_ch} options; train {len(train)}, evaluation split {len(test)}")
        for part, rows in (("train", train), ("test", test)):
            maj, k = report(f"{part}", rows)
            if k < 2 or maj > 0.95:
                bad.append(f"{name}/{part}")
        # the pipeline's own guard, so this script fails the same way a run would
        CF.assert_labels_usable(test, f"{name}/test")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            for part, rows in (("train", train), ("test", test)):
                p = os.path.join(a.out, f"{part}.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                print(f"  wrote {len(rows)} -> {p}")
    if bad:
        raise SystemExit("degenerate answer key in: " + ", ".join(bad))
    print("\nevery key usable: at least two classes and no majority above 95%")


if __name__ == "__main__":
    main()
