"""Why is the zero-shot backbone better than the closed-form framework in most cells?
Diagnostics from the saved per-item predictions: (a) McNemar for zero-shot vs the full
framework, (b) accuracy of the verifier used alone as an option selector (it is fused at
weight 0.35 into option scoring), (c) skew of the predicted-option distribution, which
detects fine-tuning collapse onto a single letter."""

import os, sys
import json, glob, os, collections, numpy as np
from scipy.stats import binomtest
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws
CFN = {"medqa": "MedQA", "mmlu_med": "MMLU-Med", "medmcqa": "MedMCQA", "pubmedqa": "PubMedQA"}
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
out = {}
for lab in ("llama31", "qwen3", "deepseek"):
    for ds in ("medqa", "mmlu_med", "medmcqa", "pubmedqa"):
        B = ws("experiments", f"closed_{ds}_{lab}"); F = ws("experiments", f"closed_{ds}_{lab}ft")
        if not (os.path.exists(f"{B}/metrics.json") and os.path.exists(f"{F}/metrics.json")): continue
        z = {r["test_index"]: r for r in rd(f"{B}/preds_vanilla.jsonl")}
        full = {r["test_index"]: r for r in rd(f"{F}/preds_adapted_verifier.jsonl")}
        ad = {r["test_index"]: r for r in rd(f"{F}/preds_adapted.jsonl")}
        ids = sorted(set(z) & set(full))
        n10 = sum(z[i]["correct"] == 1 and full[i]["correct"] == 0 for i in ids)
        n01 = sum(z[i]["correct"] == 0 and full[i]["correct"] == 1 for i in ids)
        p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
        rec = dict(n=len(ids), zero_only_correct=n10, full_only_correct=n01, mcnemar_p=round(p, 5),
                   acc_zero=round(100 * np.mean([z[i]["correct"] for i in ids]), 2),
                   acc_adapted=round(100 * np.mean([ad[i]["correct"] for i in ids]), 2),
                   acc_full=round(100 * np.mean([full[i]["correct"] for i in ids]), 2))
        # verifier used alone as the selector (its scores are stored per option)
        vr = [r for r in full.values() if r.get("verifier")]
        if vr:
            rec["verifier_only_acc"] = round(100 * np.mean([int(np.argmax(r["verifier"])) == r["gold"] for r in vr]), 2)
            rec["chance"] = round(100 / len(vr[0]["verifier"]), 1)
        # predicted-option distribution: fine-tuning collapse shows up as a spike
        for name, R in (("zero", z), ("adapted", ad), ("full", full)):
            c = collections.Counter(r["pred"] for r in R.values()); tot = sum(c.values())
            rec[f"pred_dist_{name}"] = [round(100 * c[k] / tot, 1) for k in range(len(vr[0]["verifier"]) if vr else 4)]
            rec[f"max_option_share_{name}"] = max(rec[f"pred_dist_{name}"])
        out[f"{lab}::{ds}"] = rec
        print(f"{lab:9s} {CFN[ds]:9s} zero={rec['acc_zero']:5.1f} adapted={rec['acc_adapted']:5.1f} full={rec['acc_full']:5.1f} "
              f"| McNemar p={rec['mcnemar_p']:.2e} ({n10}/{n01}) | verifier-only={rec.get('verifier_only_acc')} (chance {rec.get('chance')}) "
              f"| max option share zero/adapted={rec['max_option_share_zero']}/{rec['max_option_share_adapted']}")
json.dump(out, open(ws("results", "why_closedform.json"), "w"), indent=1)
