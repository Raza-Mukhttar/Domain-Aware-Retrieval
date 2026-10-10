# -*- coding: utf-8 -*-
"""The retrieval effect on the source/topic-disjoint repartition.

Three seeds trained on the disjoint training split, each evaluated on the disjoint test split with
the FAISS index rebuilt from that split (the Retriever always builds from the pool it is given, so
this follows from the dataset change). Compares against the two estimates the paper already has: the
stratified subset analysis and the source-disjoint retrieval restriction.
Writes results/group_disjoint_repartition.json.
"""
import json, os, sys
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
RNG = np.random.default_rng(42)
M = ("rouge1", "meteor", "bertscore_f1")
def eff(A, B, keys, m="rouge1"):
    d = np.array([A[k][m] - B[k][m] for k in sorted(keys)])
    bs = d[RNG.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(diff=round(100 * float(d.mean()), 2),
                ci=[round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)],
                n=len(d))
OUT = {"per_seed": {}}
for seed in (42, 1, 2):
    d = ws("experiments", f"openended_medquad_gs_llama31T{seed}")
    fa, fb = f"{d}/preds_rag_n5_combined.jsonl", f"{d}/preds_norag_n5_combined.jsonl"
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    A = {r["test_idx"]: r for r in rd(fa)}; B = {r["test_idx"]: r for r in rd(fb)}
    keys = set(A) & set(B)
    OUT["per_seed"][str(seed)] = dict(
        n=len(keys),
        norag={m: round(100 * float(np.mean([B[k][m] for k in keys])), 2) for m in M},
        rag={m: round(100 * float(np.mean([A[k][m] for k in keys])), 2) for m in M},
        retrieval_effect={m: eff(A, B, keys, m) for m in M})
if OUT["per_seed"]:
    r1 = [v["retrieval_effect"]["rouge1"]["diff"] for v in OUT["per_seed"].values()]
    OUT["summary"] = dict(seeds=len(r1), mean=round(float(np.mean(r1)), 2),
                          sd=round(float(np.std(r1, ddof=1)), 2) if len(r1) > 1 else None,
                          same_sign_in_all_seeds=all(x > 0 for x in r1) or all(x < 0 for x in r1))
    # the comparators the specification asks for
    gc = (json.load(open(ws("results", "grouped_corrected.json"))) or {}).get("corrected") or {} \
        if os.path.exists(ws("results", "grouped_corrected.json")) else {}
    sd = json.load(open(ws("results", "srcdisj_check.json"))) if os.path.exists(ws("results", "srcdisj_check.json")) else {}
    OUT["comparators"] = dict(
        random_split_all=(gc.get("all") or {}).get("diff"),
        random_split_topic_held_out=(gc.get("focus_held_out") or {}).get("diff"),
        random_split_topic_in_pool=(gc.get("focus_in_pool") or {}).get("diff"),
        source_disjoint_retrieval_only=(sd.get("Source- and topic-disjoint pool") or {}).get("diff"))
json.dump(OUT, open(ws("results", "group_disjoint_repartition.json"), "w"), indent=1)
print(json.dumps(OUT, indent=1)[:1400])
