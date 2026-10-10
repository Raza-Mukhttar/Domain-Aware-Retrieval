# -*- coding: utf-8 -*-
"""Closed-form seed variability.

The submitted closed-form results are single-seed. Each benchmark x backbone cell was re-trained
with seeds 1 and 2 and re-evaluated on the identical items, giving three seeds per cell. Reports
the per-seed accuracy, the mean and standard deviation, and — the point of the exercise — whether
any effect the paper reports is smaller than the seed spread. Writes results/seed_variability.json.
"""
import json, os, sys
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws
BB = {"llama31": "LLaMA-3.1-8B", "qwen3": "Qwen3-8B", "deepseek": "DeepSeek-R1-8B"}
DS = {"medqa": "MedQA", "medmcqa": "MedMCQA", "mmlu_med": "MMLU-Med", "pubmedqa": "PubMedQA"}
COND = ("adapted", "adapted_verifier")
def acc(d, cond):
    f = os.path.join(d, "metrics.json")
    if not os.path.exists(f): return None
    m = json.load(open(f))
    for k, v in m.items():
        if k == cond or k.endswith("::" + cond):
            return v.get("accuracy_pct") if isinstance(v, dict) else None
    return None
OUT = {}
for bb in BB:
    for ds in DS:
        rec = {}
        for cond in COND:
            vals = []
            for seed, d in ((42, ws("experiments", f"closed_{ds}_{bb}ft")),
                            (1, ws("experiments", f"closed_{ds}_{bb}ft_s1")),
                            (2, ws("experiments", f"closed_{ds}_{bb}ft_s2"))):
                a = acc(d, cond)
                if a is not None: vals.append((seed, a))
            if len(vals) < 2: continue
            xs = [v for _s, v in vals]
            rec[cond] = dict(per_seed={str(s): v for s, v in vals}, n_seeds=len(xs),
                             mean=round(float(np.mean(xs)), 2), sd=round(float(np.std(xs, ddof=1)), 2),
                             spread=round(max(xs) - min(xs), 2))
        if rec:
            # the effect the paper reports for this cell: verifier fusion minus task adaptation
            if all(c in rec for c in COND):
                d_seed = [rec["adapted_verifier"]["per_seed"][s] - rec["adapted"]["per_seed"][s]
                          for s in rec["adapted"]["per_seed"] if s in rec["adapted_verifier"]["per_seed"]]
                rec["verifier_effect"] = dict(per_seed=[round(x, 2) for x in d_seed],
                                              mean=round(float(np.mean(d_seed)), 2),
                                              sd=round(float(np.std(d_seed, ddof=1)), 2) if len(d_seed) > 1 else None,
                                              negative_in_all_seeds=all(x < 0 for x in d_seed))
            OUT[f"{bb}::{ds}"] = rec
sds = [v[c]["sd"] for v in OUT.values() for c in COND if c in v]
OUT["_summary"] = dict(cells=len([k for k in OUT if not k.startswith("_")]),
                       mean_sd_accuracy_pts=round(float(np.mean(sds)), 2) if sds else None,
                       max_sd_accuracy_pts=round(float(max(sds)), 2) if sds else None,
                       verifier_effect_negative_in_all_seeds=sum(
                           1 for k, v in OUT.items() if not k.startswith("_") and v.get("verifier_effect", {}).get("negative_in_all_seeds")),
                       cells_with_verifier_effect=sum(1 for k, v in OUT.items() if not k.startswith("_") and "verifier_effect" in v))
# ---------------------------------------------------------------- open-ended side (the two backbones
# whose open-ended runs were single-seed). Same three seeds, same 300 questions, corrected prompting.
def rdj(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
OPEN = {}
M = ("rouge1", "meteor", "bertscore_f1")
RNG = np.random.default_rng(42)
for bb in ("qwen3", "deepseek"):
    for ds in ("medquad", "icliniq"):
        per_seed = {}
        eff = {}
        for seed in (42, 1, 2):
            d = ws("experiments", f"openended_{ds}_{bb}T{seed}")
            fa, fb = f"{d}/preds_rag_n5_combined.jsonl", f"{d}/preds_norag_n5_combined.jsonl"
            if not (os.path.exists(fa) and os.path.exists(fb)): continue
            A = {r["test_idx"]: r for r in rdj(fa)}; B = {r["test_idx"]: r for r in rdj(fb)}
            keys = sorted(set(A) & set(B))
            per_seed[str(seed)] = {m: round(100 * float(np.mean([A[k][m] for k in keys])), 2) for m in M}
            per_seed[str(seed)]["norag_rouge1"] = round(100 * float(np.mean([B[k][m] for k in keys for m in ["rouge1"]])), 2)
            dd = np.array([A[k]["rouge1"] - B[k]["rouge1"] for k in keys])
            bs = dd[RNG.integers(0, len(dd), (10000, len(dd)))].mean(1)
            eff[str(seed)] = dict(diff=round(100 * float(dd.mean()), 2),
                                  ci=[round(100 * float(np.percentile(bs, 2.5)), 2),
                                      round(100 * float(np.percentile(bs, 97.5)), 2)], n=len(dd))
        if len(per_seed) < 2: continue
        r1 = [per_seed[s]["rouge1"] for s in per_seed]
        de = [eff[s]["diff"] for s in eff]
        OPEN[f"{bb}::{ds}"] = dict(per_seed=per_seed, retrieval_effect_per_seed=eff,
                                   rouge1_mean=round(float(np.mean(r1)), 2), rouge1_sd=round(float(np.std(r1, ddof=1)), 2),
                                   retrieval_effect_mean=round(float(np.mean(de)), 2),
                                   retrieval_effect_sd=round(float(np.std(de, ddof=1)), 2),
                                   same_sign_in_all_seeds=all(x > 0 for x in de) or all(x < 0 for x in de),
                                   n_seeds=len(per_seed))
if OPEN:
    OUT["_open_ended"] = OPEN
    sds = [v["rouge1_sd"] for v in OPEN.values()]
    OUT["_summary"]["open_ended_cells"] = len(OPEN)
    OUT["_summary"]["open_ended_mean_sd_rouge1"] = round(float(np.mean(sds)), 2)
    OUT["_summary"]["open_ended_retrieval_effect_same_sign"] = sum(1 for v in OPEN.values() if v["same_sign_in_all_seeds"])
json.dump(OUT, open(ws("results", "seed_variability.json"), "w"), indent=1)
print("%-24s %-18s %-22s %6s %6s" % ("cell", "condition", "per-seed (42/1/2)", "mean", "SD"))
for k, v in OUT.items():
    if k.startswith("_"): continue
    for c in COND:
        if c in v:
            ps = v[c]["per_seed"]
            print("%-24s %-18s %-22s %6.1f %6.2f" % (k, c, "/".join(f"{ps.get(s, float('nan')):.1f}" for s in ("42","1","2")), v[c]["mean"], v[c]["sd"]))
print("\n%-22s %-26s %8s %7s %-26s %8s" % ("open-ended cell","ROUGE-1 seeds 42/1/2","mean","SD","retr. effect 42/1/2","mean"))
for k, v in (OUT.get("_open_ended") or {}).items():
    ps = v["per_seed"]; ef = v["retrieval_effect_per_seed"]
    print("%-22s %-26s %8.2f %7.2f %-26s %+8.2f" % (
        k, " / ".join(f"{ps[s]['rouge1']:.2f}" for s in ("42","1","2") if s in ps), v["rouge1_mean"], v["rouge1_sd"],
        " / ".join(f"{ef[s]['diff']:+.2f}" for s in ("42","1","2") if s in ef), v["retrieval_effect_mean"]))
print("\nsummary:", json.dumps(OUT["_summary"]))
