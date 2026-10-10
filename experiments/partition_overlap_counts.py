"""Deduplicated partition counts: exact-duplicate questions within and across
partitions and TF-IDF char 3-5-gram near-duplicates (cos >= 0.90) of validation/test
questions against the training partition, with the resulting 'deduplicated' sizes."""

import os, sys
import json, re, numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
from sklearn.feature_extraction.text import TfidfVectorizer
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def norm(q): return re.sub(r"\s+", " ", q.strip().lower())
out = {}
for name, d in (("MedQuAD", "medquad"), ("iCliniq-style", "iCliniq")):
    P = {s: rd(src("data", d, f"{s}.jsonl")) for s in ("train", "validation", "test")}
    Q = {s: [norm(r["question"]) for r in P[s]] for s in P}
    trainset = set(Q["train"]); res = {"sizes": {s: len(Q[s]) for s in Q}}
    res["train_unique_questions"] = len(trainset)
    res["train_internal_duplicates"] = len(Q["train"]) - len(trainset)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(Q["train"])
    Xt = vec.transform(Q["train"])
    for s in ("validation", "test"):
        exact = [i for i, q in enumerate(Q[s]) if q in trainset]
        Xs = vec.transform(Q[s]); sims = (Xs @ Xt.T).toarray() if len(Q[s]) * len(Q["train"]) < 4e7 else None
        if sims is None:
            mx = np.zeros(len(Q[s]))
            for i in range(0, len(Q[s]), 200): mx[i:i+200] = (Xs[i:i+200] @ Xt.T).max(axis=1).toarray().ravel()
        else: mx = sims.max(axis=1)
        near = [i for i in range(len(Q[s])) if mx[i] >= 0.90]
        keep_exact = len(Q[s]) - len(exact); keep_near = len(Q[s]) - len(set(exact) | set(near))
        res[s] = dict(n=len(Q[s]), exact_in_train=len(exact), near90_in_train=len(set(near) - set(exact)), after_exact_dedup=keep_exact, after_near_dedup=keep_near,
                      exact_pct=round(100 * len(exact) / len(Q[s]), 2), near_or_exact_pct=round(100 * len(set(exact) | set(near)) / len(Q[s]), 2))
    # evaluated 300-question test subset (seed 42) after the same rules
    out[name] = res; print(name, json.dumps(res))
json.dump(out, open(ws("results", "dedup_counts.json"), "w"), indent=1)
