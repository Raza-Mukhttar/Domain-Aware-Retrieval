"""Effect of the test-sampling rule: the seed-42 generators evaluated on
(a) the 300 randomly sampled test questions of the matched experiments and (b) the first
300 test rows in file order (rule of the submitted experiments), same 2,000-row pool.
Reports overlap of the two subsets, metrics per condition, and the paired retrieval effect
with a 10,000-resample bootstrap under each rule.

Both prompt-truncation policies are checked, because the primary result of the paper now comes
from the corrected policy: labels llama31s42 (submitted, right truncation) and llama31T42
(corrected, left truncation). Keys "random"/"first" are the submitted policy and
"random_corrected"/"first_corrected" the corrected one."""

import os, sys
import json, os, numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
M = ["rouge1", "meteor", "bertscore_f1"]; rng = np.random.default_rng(42); out = {}
CONDS = ("norag_n1_first", "norag_n5_combined", "rag_n1_first", "rag_n5_combined")
POLICIES = (("", "llama31s42", "random", "first"), ("_corrected", "llama31T42", "random_corrected", "first_corrected"))
for ds in ("medquad", "icliniq"):
    rec = {}; ids = {}
    for _sfx, lab, k_rand, k_first in POLICIES:
        A = ws("experiments", f"openended_{ds}_{lab}"); B = ws("experiments", f"openended_{ds}_{lab}_first300")
        if not all(os.path.exists(f"{d}/preds_{c}.jsonl") for d in (A, B) for c in CONDS):
            print(ds, lab, "first300 not finished"); continue
        for tag, D in ((k_rand, A), (k_first, B)):
            rows = {c: {r["test_idx"]: r for r in rd(f"{D}/preds_{c}.jsonl")} for c in CONDS}
            ids[tag] = set(rows["rag_n5_combined"]); rec[tag] = {}
            for c, R in rows.items():
                rec[tag][c] = {m: round(100 * float(np.mean([r[m] for r in R.values()])), 2) for m in M}
            for m in M:
                dd = np.array([rows["rag_n5_combined"][i][m] - rows["norag_n5_combined"][i][m] for i in sorted(rows["rag_n5_combined"])])
                bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
                rec[tag][f"retrieval_effect_{m}"] = dict(diff=round(100 * dd.mean(), 2), ci=[round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)], n=len(dd))
    if not rec: continue
    if "random" in ids and "first" in ids: rec["overlap_questions"] = len(ids["random"] & ids["first"])
    rec["n_each"] = 300
    out[ds] = rec
    print(ds, "overlap", rec.get("overlap_questions"), {t: rec[t]["retrieval_effect_rouge1"]["diff"] for t in ("random", "first", "random_corrected", "first_corrected") if t in rec})
json.dump(out, open(ws("results", "sampling_check.json"), "w"), indent=1)
