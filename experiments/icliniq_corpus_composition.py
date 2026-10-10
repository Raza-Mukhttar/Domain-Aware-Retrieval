"""Recover per-platform composition of the iCliniq-style corpus by matching our
JSONL questions against the source files of LasseRegin/medical-question-answer-data."""

import os, sys
import glob, json, os
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
norm=lambda t:" ".join((t or "").lower().split())
srcq={}
for f in glob.glob(ws("data","mqa_src","*.json")):
    name=os.path.basename(f).replace("QAs.json","").replace(".json","")
    try: rows=json.load(open(f))
    except Exception: continue
    for r in rows:
        q=norm(r.get("question"));
        if q and q not in srcq: srcq[q]=name
print("source questions:",len(srcq),"| platforms:",sorted(set(srcq.values())))
out={}
from collections import Counter
for split in ("train","validation","test"):
    rows=[json.loads(l) for l in open(src("data","iCliniq",f"{split}.jsonl"))]
    c=Counter(srcq.get(norm(r["question"]),"unmatched") for r in rows); out[split]=dict(n=len(rows),by_platform=dict(c))
    print(split,dict(c))
json.dump(out,open(ws("results","icliniq_composition.json"),"w"),indent=1)
