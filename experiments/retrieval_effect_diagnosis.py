"""Diagnostic for the open-ended result: why does the retrieval condition score lower
than the same fine-tuned generator without retrieval? Uses the saved seed-42 predictions."""

import os, sys
import json, os, numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws
from rouge_score import rouge_scorer
rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
out = {}
for ds_lab in ("medquad::llama31s42", "icliniq::llama31s42", "medquad::llama31T42", "icliniq::llama31T42"):
    ds, lab = ds_lab.split("::")
    D = ws("experiments", f"openended_{ds}_{lab}")
    if not os.path.exists(f"{D}/metrics.json"): continue
    rows = {c: {r["test_idx"]: r for r in rd(f"{D}/preds_{c}.jsonl")} for c in ("norag_n5_combined", "rag_n5_combined")}
    ids = sorted(set(rows["norag_n5_combined"]) & set(rows["rag_n5_combined"]))
    rec = {}
    ref_len = np.mean([len(rows["rag_n5_combined"][i]["reference"].split()) for i in ids])
    for c in rows:
        R = rows[c]
        pl = np.array([len(R[i]["prediction"].split()) for i in ids], float)
        sc = [rs.score(R[i]["reference"], R[i]["prediction"])["rouge1"] for i in ids]
        rec[c] = dict(pred_words=round(float(pl.mean()), 1), pred_words_median=round(float(np.median(pl)), 1),
                      len_ratio_to_reference=round(float(pl.mean() / ref_len), 2),
                      rouge1_precision=round(100 * float(np.mean([s.precision for s in sc])), 2),
                      rouge1_recall=round(100 * float(np.mean([s.recall for s in sc])), 2),
                      rouge1_f=round(100 * float(np.mean([s.fmeasure for s in sc])), 2),
                      empty_or_short_pct=round(100 * float((pl < 20).mean()), 1))
    rec["reference_words"] = round(float(ref_len), 1); rec["n"] = len(ids)
    # per-question: does retrieval lose because of length or because of content?
    dn = np.array([rs.score(rows["rag_n5_combined"][i]["reference"], rows["rag_n5_combined"][i]["prediction"])["rouge1"].recall -
                   rs.score(rows["norag_n5_combined"][i]["reference"], rows["norag_n5_combined"][i]["prediction"])["rouge1"].recall for i in ids])
    dp = np.array([rs.score(rows["rag_n5_combined"][i]["reference"], rows["rag_n5_combined"][i]["prediction"])["rouge1"].precision -
                   rs.score(rows["norag_n5_combined"][i]["reference"], rows["norag_n5_combined"][i]["prediction"])["rouge1"].precision for i in ids])
    rec["delta_recall_pts"] = round(100 * float(dn.mean()), 2); rec["delta_precision_pts"] = round(100 * float(dp.mean()), 2)
    rec["questions_where_retrieval_worse_pct"] = round(100 * float(np.mean([
        rs.score(rows["rag_n5_combined"][i]["reference"], rows["rag_n5_combined"][i]["prediction"])["rouge1"].fmeasure <
        rs.score(rows["norag_n5_combined"][i]["reference"], rows["norag_n5_combined"][i]["prediction"])["rouge1"].fmeasure for i in ids])), 1)
    out[ds_lab] = rec
    print(ds_lab, json.dumps({k: v for k, v in rec.items() if not isinstance(v, dict)}))
    for c in ("norag_n5_combined", "rag_n5_combined"): print("   ", c, rec[c])
json.dump(out, open(ws("results", "why_retrieval_hurts.json"), "w"), indent=1)
