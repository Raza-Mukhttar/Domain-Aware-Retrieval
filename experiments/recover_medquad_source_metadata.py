"""Recover MedQuAD focus/source/qtype metadata by matching our JSONL questions to
the original MedQuAD XML release (grouped-split analysis, )."""

import os, sys
import glob, json, os, re, xml.etree.ElementTree as ET
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
norm=lambda t:" ".join((t or "").lower().split())
meta={}
for f in glob.glob(ws("data","medquad_src","**","*.xml"),recursive=True):
    try: root=ET.parse(f).getroot()
    except Exception: continue
    focus=(root.findtext("Focus") or "").strip(); source=root.get("source") or ""; url=root.get("url") or ""
    for q in root.iter("Question"):
        qt=q.get("qtype") or ""; txt=norm(q.text)
        if txt and txt not in meta: meta[txt]={"focus":focus,"source":source,"url":url,"qtype":qt,"doc":os.path.basename(f)}
print("xml questions:",len(meta))
out={}
for split in ("train","validation","test"):
    rows=[json.loads(l) for l in open(src("data","medquad",f"{split}.jsonl"))]
    hit=[meta.get(norm(r["question"])) for r in rows]
    out[split]=dict(n=len(rows),matched=sum(h is not None for h in hit),records=[(h or {}) for h in hit])
    print(split,out[split]["n"],"matched",out[split]["matched"])
json.dump(out,open(ws("results","medquad_recovered_meta.json"),"w"))
# grouped-split diagnostic: test items whose focus entity does NOT occur in the first-2000-row retrieval pool / training split
tr=out["train"]["records"]; te=out["test"]["records"]
foci_train=set(h.get("focus","").lower() for h in tr if h); docs_train=set(h.get("doc","") for h in tr if h)
g=dict(test_matched=sum(1 for h in te if h),test_focus_in_train=sum(1 for h in te if h and h.get("focus","").lower() in foci_train),
       test_doc_in_train=sum(1 for h in te if h and h.get("doc","") in docs_train),
       qtype_dist_test={},source_dist_test={})
from collections import Counter
g["qtype_dist_test"]=dict(Counter(h.get("qtype","?") for h in te if h).most_common(12)); g["source_dist_test"]=dict(Counter(h.get("source","?") for h in te if h).most_common(12))
json.dump(g,open(ws("results","medquad_grouped_diag.json"),"w"),indent=1); print(g)
