# -*- coding: utf-8 -*-
"""Source-disjoint retrieval evaluation.

Compares the retrieval effect when exemplars may come from anywhere in the pool against the effect
when every exemplar sharing the query's MedQuAD source document or topic entity is barred. The
no-retrieval control is taken from the same run in each case, so both effects are paired within a
single generation run. Writes results/srcdisj_check.json.
"""
import os, sys, json
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
RNG = np.random.default_rng(42)
def eff(a, b, keys, m="rouge1"):
    d = np.array([a[k][m] - b[k][m] for k in sorted(keys)])
    bs = d[RNG.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(n=len(d), norag=round(100 * float(np.mean([b[k][m] for k in keys])), 2),
                rag=round(100 * float(np.mean([a[k][m] for k in keys])), 2),
                diff=round(100 * float(d.mean()), 2),
                ci=[round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)])
OUT = {}
RUNS = [("Unrestricted pool (primary corrected run)", ws("experiments", "openended_medquad_llama31T42")),
        ("Source- and topic-disjoint pool", ws("experiments", "openended_medquad_llama31T42_srcdisj"))]
for name, d in RUNS:
    fa, fb = f"{d}/preds_rag_n5_combined.jsonl", f"{d}/preds_norag_n5_combined.jsonl"
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print("missing:", name); continue
    A = {r["test_idx"]: r for r in rd(fa)}; B = {r["test_idx"]: r for r in rd(fb)}
    keys = set(A) & set(B)
    OUT[name] = eff(A, B, keys)
    OUT[name]["mean_exemplars"] = round(float(np.mean([len(A[k].get("exemplar_ids") or []) for k in keys])), 2)
# how much the restriction removes
meta_p = ws("results", "medquad_recovered_meta.json")
sd = ws("experiments", "openended_medquad_llama31T42_srcdisj", "manifest.json")
if os.path.exists(meta_p) and os.path.exists(sd):
    meta = json.load(open(meta_p)); man = json.load(open(sd))
    trr = (meta.get("train") or {}).get("records") or []; ter = (meta.get("test") or {}).get("records") or []
    def key(r):
        r = r or {}
        return ((r.get("doc") or "").strip() or None, (r.get("focus") or "").strip().lower() or None)
    pool = [key(trr[i]) if 0 <= i < len(trr) else (None, None) for i in man["train_ids"]]
    bans = []
    for t in man["test_ids"]:
        qd, qf = key(ter[t]) if 0 <= t < len(ter) else (None, None)
        bans.append(sum(1 for pd_, pf in pool if (qd and pd_ == qd) or (qf and pf == qf)))
    OUT["_meta"] = dict(pool_items=len(pool), mean_banned=round(float(np.mean(bans)), 2),
                        max_banned=int(max(bans)) if bans else 0,
                        pct_questions_with_any_ban=round(100 * float(np.mean([b > 0 for b in bans])), 1),
                        exclude_same=man.get("exclude_same"))
json.dump(OUT, open(ws("results", "srcdisj_check.json"), "w"), indent=1)
print(json.dumps(OUT, indent=1))
