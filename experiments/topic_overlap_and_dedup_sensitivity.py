"""Topic-overlap and deduplicated-subset sensitivity analysis.

Computed on the corrected-prompt predictions and reported as two panels: the
submitted truncation policy, which is diagnostic, and the corrected policy, which
is primary."""

import os, sys
import json, os, re, numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
from sklearn.feature_extraction.text import TfidfVectorizer
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def norm(q): return re.sub(r"\s+", " ", q.strip().lower())
META = json.load(open(ws("results", "medquad_recovered_meta.json")))
rng = np.random.default_rng(42)
out = {}
for pol, lab in (("submitted", "llama31s42"), ("corrected", "llama31T42")):
    exp = ws("experiments", f"openended_medquad_{lab}")
    if not os.path.exists(f"{exp}/manifest.json"): continue
    man = json.load(open(f"{exp}/manifest.json"))
    trrec = META["train"]["records"]; terec = META["test"]["records"]
    pool_foci = {trrec[i].get("focus", "").lower() for i in man["train_ids"] if i < len(trrec)}
    rows = {c: {r["test_idx"]: r for r in rd(f"{exp}/preds_{c}.jsonl")} for c in ("norag_n5_combined", "rag_n5_combined")}
    ids = sorted(set(rows["rag_n5_combined"]) & set(rows["norag_n5_combined"]))
    held = [i for i in ids if terec[i].get("focus", "").lower() not in pool_foci]
    seen = [i for i in ids if i not in held]
    # deduplicated subset: test questions with no exact or near-duplicate training question
    train_q = [norm(r["question"]) for r in rd(src("data", "medquad", "train.jsonl"))]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(train_q); Xt = vec.transform(train_q)
    qs = [norm(rows["rag_n5_combined"][i]["question"]) for i in ids]
    mx = (vec.transform(qs) @ Xt.T).max(axis=1).toarray().ravel()
    ts = set(train_q)
    nondup = [i for i, q, m in zip(ids, qs, mx) if m < 0.90 and q not in ts]
    rec = {"n": len(ids)}
    for name, S in (("all", ids), ("focus_held_out", held), ("focus_in_pool", seen), ("nonduplicate", nondup)):
        if not S: continue
        d = np.array([rows["rag_n5_combined"][i]["rouge1"] - rows["norag_n5_combined"][i]["rouge1"] for i in S])
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        rec[name] = dict(n=len(S),
                         norag=round(100 * float(np.mean([rows["norag_n5_combined"][i]["rouge1"] for i in S])), 2),
                         rag=round(100 * float(np.mean([rows["rag_n5_combined"][i]["rouge1"] for i in S])), 2),
                         diff=round(100 * float(d.mean()), 2),
                         ci=[round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)])
    out[pol] = rec
    print(pol, {k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("n", "diff", "ci")}) for k, v in rec.items()})
json.dump(out, open(ws("results", "grouped_corrected.json"), "w"), indent=1)
