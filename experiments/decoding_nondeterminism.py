"""The no-retrieval prompts sit below the input cap, so the
truncation policy is a no-op for them; this quantifies what the residual differences between
the submitted-policy and corrected-policy runs of the same checkpoint actually are."""

import os, sys
import json, os, numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws
from transformers import AutoTokenizer
import medlm.openended_compat as R
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(ws("experiments", "ckpt_medquad_llama31_seed42"))
out = {}
for ds in ("medquad", "icliniq"):
    A = ws("experiments", f"openended_{ds}_llama31s42"); B = ws("experiments", f"openended_{ds}_llama31T42")
    if not (os.path.exists(f"{A}/metrics.json") and os.path.exists(f"{B}/metrics.json")): continue
    rec = {}
    # 1. are the no-retrieval prompts actually below the cap?
    man = json.load(open(f"{A}/manifest.json"))
    train, test = R.load_split(ds, man["n_train_pool"], man["n_test"], man["seed"])
    lens = [len(tok(R.build_prompt(t["question"], [], None), add_special_tokens=False)["input_ids"]) for t in test]
    rec["norag_prompt_tokens_max"] = int(max(lens)); rec["norag_over_cap"] = int(sum(l > 512 for l in lens)); rec["cap"] = 512
    # 2. how do the no-retrieval outputs differ between the two runs?
    for cond in ("norag_n1_first", "norag_n5_combined"):
        a = {r["test_idx"]: r for r in rd(f"{A}/preds_{cond}.jsonl")}; b = {r["test_idx"]: r for r in rd(f"{B}/preds_{cond}.jsonl")}
        ids = sorted(set(a) & set(b))
        same = sum(a[i]["prediction"] == b[i]["prediction"] for i in ids)
        dd = np.array([b[i]["rouge1"] - a[i]["rouge1"] for i in ids])
        rec[cond] = dict(n=len(ids), identical_answers=same, identical_pct=round(100 * same / len(ids), 1),
                         mean_rouge1_diff=round(100 * float(dd.mean()), 3),
                         max_abs_item_diff=round(100 * float(np.abs(dd).max()), 2))
    out[ds] = rec
    print(ds, json.dumps(rec))
json.dump(out, open(ws("results", "nondeterminism_check.json"), "w"), indent=1)
