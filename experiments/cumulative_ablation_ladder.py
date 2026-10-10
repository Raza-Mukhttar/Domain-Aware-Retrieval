# -*- coding: utf-8 -*-
"""Cumulative ablation ladder, in the order the specification fixes.

generator only (N=1) -> +candidate budget (N=5, first beam) -> +retrieval -> +question-type
retrieval -> +unsummarised exemplars -> +verifier selection -> +entity reranking -> +entity decoding,
with the contrastive stage as a separate branch because it changes the checkpoint rather than
flipping an inference switch.

Each rung's difference is paired against the PRECEDING rung, not the baseline: that is what makes it
a ladder rather than a second copy of the independent grid. All five metrics are reported.
Inference-only, over the seed-42 checkpoints with corrected (left) truncation.

Writes results/ladder_spec.json.
"""
import json, os, sys
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws
M = (("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L"),
     ("meteor", "METEOR"), ("bertscore_f1", "BERTScore-F1"))
RUNGS = [
    ("generator only ($N=1$, no retrieval)",              "",        "norag_n1_first"),
    ("+ candidate budget ($N=5$, first beam)",            "",        "norag_n5_first"),
    ("+ exemplar retrieval ($k=3$, summarised)",          "",        "rag_n5_first"),
    ("+ question-type stratified retrieval",              "",        "rag_qtype_n5_first"),
    ("+ unsummarised exemplars",                          "_ladder", "rag_qtype_fullans_n5_first"),
    ("+ verifier selection",                              "_ladder", "rag_qtype_fullans_n5_verifier"),
    ("+ entity reranking (combined rule)",                "_ladder", "rag_qtype_fullans_n5_combined"),
    ("+ decoding-time entity bonus ($\\gamma=1$)",        "_ladder", "rag_qtype_fullans_edec_n5_combined"),
]
BRANCH = ("contrastive stage on the retrieval rung (separate branch: it changes the checkpoint)",
          "llama31cT42", "", "rag_n5_combined", "llama31T42", "", "rag_n5_combined")
RNG = np.random.default_rng(42)
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def load(ds, lab, tag, cond):
    f = ws("experiments", f"openended_{ds}_{lab}{tag}", f"preds_{cond}.jsonl")
    return {r["test_idx"]: r for r in rd(f)} if os.path.exists(f) else None
def paired(a, b, keys, m):
    d = np.array([a[k][m] - b[k][m] for k in sorted(keys)])
    bs = d[RNG.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(diff=round(100 * float(d.mean()), 2),
                ci=[round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)])
OUT = {}
for ds in ("medquad", "icliniq"):
    loaded, missing = [], []
    for label, tag, cond in RUNGS:
        r = load(ds, "llama31T42", tag, cond)
        (loaded if r else missing).append((label, cond, r))
    present = [(l, c, r) for l, c, r in loaded if r]
    if len(present) < 2: continue
    keys = set.intersection(*[set(r) for _l, _c, r in present])
    rows = []
    for i, (label, cond, r) in enumerate(present):
        lvl = {k: round(100 * float(np.mean([r[x][k] for x in keys])), 2) for k, _n in M}
        step = None if i == 0 else {k: paired(r, present[i-1][2], keys, k) for k, _n in M}
        rows.append(dict(rung=i + 1, label=label, condition=cond, n=len(keys), level=lvl, step_vs_previous=step))
    rec = dict(rungs=rows, n=len(keys),
               missing=[dict(label=l, condition=c) for l, c, r in loaded + missing if r is None])
    # contrastive branch
    lab_c, tag_c, cond_c, lab_b, tag_b, cond_b = BRANCH[1:]
    A, B = load(ds, lab_c, tag_c, cond_c), load(ds, lab_b, tag_b, cond_b)
    if A and B:
        k2 = set(A) & set(B)
        rec["contrastive_branch"] = dict(label=BRANCH[0], n=len(k2),
                                         step={k: paired(A, B, k2, k) for k, _n in M})
    OUT[ds] = rec
json.dump(OUT, open(ws("results", "ladder_spec.json"), "w"), indent=1)
for ds, rec in OUT.items():
    print(f"\n=== {ds} (n={rec['n']}) ===")
    for r in rec["rungs"]:
        st = r["step_vs_previous"]
        s = "" if not st else "  step R-1 %+.2f [%+.2f, %+.2f]" % (
            st["rouge1"]["diff"], st["rouge1"]["ci"][0], st["rouge1"]["ci"][1])
        print(f"  {r['rung']}. {r['label'][:46]:46s} R1={r['level']['rouge1']:6.2f}{s}")
    if rec["missing"]:
        print("  pending rungs:", ", ".join(m["condition"] for m in rec["missing"]))
    if "contrastive_branch" in rec:
        b = rec["contrastive_branch"]["step"]["rouge1"]
        print("  branch: contrastive stage %+.2f [%+.2f, %+.2f]" % (b["diff"], b["ci"][0], b["ci"][1]))
