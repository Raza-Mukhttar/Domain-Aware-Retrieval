# -*- coding: utf-8 -*-
"""The two hyper-parameters that could only be swept by generating: the exemplar count k and the
decoding-bonus strength gamma. Each sweep value was generated in its own run with its own paired
control, so no comparison crosses a run boundary. Writes results/hparam_sweeps.json.
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
def paired(a, b, keys, m="rouge1"):
    d = np.array([a[k][m] - b[k][m] for k in sorted(keys)])
    bs = d[RNG.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(diff=round(100 * float(d.mean()), 2),
                ci=[round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)], n=len(d))
def lvl(rows, keys):
    return {m: round(100 * float(np.mean([rows[k][m] for k in keys])), 2) for m in M}
def load(tag, cond):
    d = ws("experiments", f"openended_medquad_llama31T42{tag}", f"preds_{cond}.jsonl")
    return {r["test_idx"]: r for r in rd(d)} if os.path.exists(d) else None
OUT = {}
# ---- k sweep: retrieval effect at k=1,3,5, each against the no-retrieval control of its own run
krows = []
for tag, k in (("_k1", 1), ("", 3), ("_k5", 5)):
    rag, norag = load(tag, "rag_n5_combined"), load(tag, "norag_n5_combined")
    if not (rag and norag): print("k: missing", k); continue
    keys = set(rag) & set(norag)
    e = paired(rag, norag, keys)
    krows.append(dict(k=k, run="primary corrected run" if tag == "" else f"sweep run {tag[1:]}",
                      n=len(keys), **{f"rag_{m}": lvl(rag, keys)[m] for m in M},
                      norag_rouge1=lvl(norag, keys)["rouge1"], retrieval_effect=e,
                      mean_exemplars=round(float(np.mean([len(rag[x].get("exemplar_ids") or []) for x in keys])), 2)))
if krows: OUT["k_sweep"] = krows
# ---- gamma sweep: entity-bonus effect at each gamma, against the no-bonus retrieval condition
#      generated in the SAME run. All five reported metrics plus the question-entity retention the
#      bonus targets, with 10,000-resample paired bootstraps and Wilcoxon tests Holm-corrected over
#      the five metrics within each gamma -- the same protocol the manuscript uses elsewhere.
from scipy.stats import wilcoxon
M5 = ("rouge1", "rouge2", "rougeL", "meteor", "bertscore_f1")
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [None] * len(ps); running = 0.0
    for rank, i in enumerate(order):
        adj = ps[i] * (len(ps) - rank)
        running = max(running, min(adj, 1.0))
        out[i] = running
    return out
grows = []
for tag, g in (("_g05", 0.5), ("", 1.0), ("_g20", 2.0), ("_g40", 4.0), ("_g50", 5.0), ("_g100", 10.0)):
    edec, base = load(tag, "rag_edec_n5_combined"), load(tag, "rag_n5_combined")
    if not (edec and base): print("gamma: missing", g); continue
    keys = sorted(set(edec) & set(base))
    rec = dict(gamma=g, run="primary corrected run" if tag == "" else f"sweep run {tag[1:]}", n=len(keys),
               levels={}, effects={})
    raw = []
    for m in M5:
        a = np.array([edec[k][m] for k in keys]); b = np.array([base[k][m] for k in keys])
        rec["levels"][m] = dict(without_bonus=round(100 * float(b.mean()), 2),
                                with_bonus=round(100 * float(a.mean()), 2))
        d = a - b
        bs = d[RNG.integers(0, len(d), (10000, len(d)))].mean(1)
        try:
            pv = float(wilcoxon(a, b, zero_method="wilcox").pvalue) if np.any(d) else 1.0
        except Exception:
            pv = 1.0
        raw.append(pv)
        rec["effects"][m] = dict(diff=round(100 * float(d.mean()), 2),
                                 ci=[round(100 * float(np.percentile(bs, 2.5)), 2),
                                     round(100 * float(np.percentile(bs, 97.5)), 2)],
                                 p_raw=pv)
    for m, ph in zip(M5, holm(raw)):
        rec["effects"][m]["p_holm"] = ph
        rec["effects"][m]["significant"] = bool(ph < 0.05)
    # question-entity retention: the quantity the bonus is designed to raise
    def ret(src):
        v = [src[k].get("cov_q") for k in keys]
        v = [x for x in v if x is not None]
        return round(100 * float(np.mean(v)), 2) if v else None
    rb, ra = ret(base), ret(edec)
    rec["question_entity_retention"] = dict(without_bonus=rb, with_bonus=ra,
                                            change=None if (rb is None or ra is None) else round(ra - rb, 2))
    grows.append(rec)
if grows: OUT["gamma_sweep"] = grows
json.dump(OUT, open(ws("results", "hparam_sweeps.json"), "w"), indent=1)
for name, rows in OUT.items():
    print(f"== {name}")
    for r in rows:
        key = r.get("k", r.get("gamma"))
        e = r.get("retrieval_effect") or r.get("bonus_effect") or (r.get("effects") or {}).get("rouge1")
        star = "*" if e.get("significant") else " "
        print(f"   {key:>4} n={r['n']} R1={e['diff']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]{star}"
              + (f"  retention {r['question_entity_retention']['without_bonus']}->{r['question_entity_retention']['with_bonus']}"
                 if r.get("question_entity_retention") else ""))
