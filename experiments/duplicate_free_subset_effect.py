"""Retrieval effect on the evaluated MedQuAD/iCliniq test subsets after removing test
questions that duplicate (exact or TF-IDF char 3-5-gram cos>=0.90) a TRAINING question: same seed-42 predictions, restricted to non-duplicate questions."""

import os, sys
import json, re, numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
from sklearn.feature_extraction.text import TfidfVectorizer
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def norm(q): return re.sub(r"\s+", " ", q.strip().lower())
out={}
for ds, d in (("medquad","medquad"),("icliniq","iCliniq")):
    train=[norm(r["question"]) for r in rd(src("data",d,"train.jsonl"))]; ts=set(train)
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5)).fit(train); Xt=vec.transform(train)
    exp=ws("experiments",f"openended_{ds}_llama31s42")
    rows={c:{r["test_idx"]:r for r in rd(f"{exp}/preds_{c}.jsonl")} for c in ("norag_n5_combined","rag_n5_combined","rag_n5_first","rag_edec_n5_first")}
    qs={i:norm(r["question"]) for i,r in rows["rag_n5_combined"].items()}
    ids=sorted(qs); Xs=vec.transform([qs[i] for i in ids]); mx=(Xs@Xt.T).max(axis=1).toarray().ravel()
    dup={i for i,m in zip(ids,mx) if m>=0.90 or qs[i] in ts}; keep=[i for i in ids if i not in dup]
    rec=dict(n_evaluated=len(ids),n_duplicate_of_train=len(dup),n_kept=len(keep))
    for c in rows:
        for sub,S in (("all",ids),("nondup",keep),("dup",sorted(dup))):
            if not S: continue
            rec[f"{c}::{sub}"]={m:round(100*float(np.mean([rows[c][i][m] for i in S])),2) for m in ("rouge1","meteor","bertscore_f1")}
    # paired bootstrap of retrieval effect on the non-duplicate subset
    rng=np.random.default_rng(42)
    for sub,S in (("nondup",keep),("dup",sorted(dup))):
        if len(S)<5: continue
        dd=np.array([rows["rag_n5_combined"][i]["rouge1"]-rows["norag_n5_combined"][i]["rouge1"] for i in S]); bs=dd[rng.integers(0,len(dd),(10000,len(dd)))].mean(1)
        rec[f"retrieval_effect_rouge1::{sub}"]=dict(diff=round(100*dd.mean(),2),ci=[round(100*float(np.percentile(bs,2.5)),2),round(100*float(np.percentile(bs,97.5)),2)],n=len(S))
    out[ds]=rec; print(ds, json.dumps({k:v for k,v in rec.items() if "effect" in k or k.startswith("n_")}))
json.dump(out,open(ws("results","dedup_subset.json"),"w"),indent=1)
