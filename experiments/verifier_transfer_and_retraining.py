# -*- coding: utf-8 -*-
"""Does per-benchmark verifier training fix the closed-form decrease?

The submitted closed-form path fuses option log-probabilities with a verifier trained on MedQA
question/answer pairs and applied unchanged to all four benchmarks. Here the saved option
log-probabilities are re-fused with (a) that transferred verifier and (b) a verifier trained on the
benchmark's own training split, at the executed weight and across the whole weight range, so the
diagnosis can be tested instead of asserted.

Runs on CPU by default: ClinicalBERT is small and the GPUs are occupied by the seed runs.
"""
import argparse, json, os, sys
import numpy as np, torch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import src, ws
import closedform_phase_c_inference_and_selection as CF
from transformers import AutoTokenizer, AutoModelForSequenceClassification
LETTERS = "ABCDEFGH"
WEIGHTS = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def minmax(v):
    v = np.asarray(v, dtype=float)
    return np.zeros_like(v) if v.max() - v.min() < 1e-12 else (v - v.min()) / (v.max() - v.min())
def scorer(path, dev):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    def f(pairs):
        out = []
        for i in range(0, len(pairs), 64):
            enc = tok([f"Q: {q} [SEP] A: {a}" for q, a in pairs[i:i + 64]], return_tensors="pt",
                      padding=True, truncation=True, max_length=256).to(dev)
            with torch.no_grad():
                out += torch.softmax(m(**enc).logits, -1)[:, 1].float().cpu().tolist()
        return out
    return f
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    OUT = {}
    for ds in ("medqa", "medmcqa", "mmlu_med", "pubmedqa"):
        own = ws("experiments", f"verifier_cf_{ds}")
        if not os.path.exists(own): print("no own verifier for", ds); continue
        rows_by_bb = {}
        for bb in ("llama31", "qwen3", "deepseek"):
            f = ws("experiments", f"closed_{ds}_{bb}ft", "preds_adapted_verifier.jsonl")
            if os.path.exists(f): rows_by_bb[bb] = rd(f)
        if not rows_by_bb: continue
        # the evaluated items are identical across backbones, so build the option texts once
        _tr, te, _n = CF.LOADERS[ds]()
        any_rows = list(rows_by_bb.values())[0]
        idx = [int(r["test_index"]) for r in any_rows]
        pairs, spans = [], []
        for i in idx:
            opts = te[i]["options"]; q = te[i]["question"]
            spans.append((len(pairs), len(opts)))
            pairs += [(q, f"{LETTERS[j]}. {o}") for j, o in enumerate(opts)]
        sc = {}
        for name, path in (("transferred (MedQA-trained)", src("outputs_llm", "verifier")), ("own benchmark", own)):
            s = scorer(path, a.device)(pairs)
            sc[name] = [s[o:o + n] for o, n in spans]
            print(f"[{ds}] scored {len(pairs)} option pairs with the {name} verifier", flush=True)
        for bb, rows in rows_by_bb.items():
            rec = {}
            for name in sc:
                accs = []
                for w in WEIGHTS:
                    ok = 0
                    for t, r in enumerate(rows):
                        comb = (1 - w) * minmax(r["logprobs"]) + w * minmax(sc[name][t])
                        ok += int(int(np.argmax(comb)) == int(r["gold"]))
                    accs.append(round(100 * ok / len(rows), 2))
                rec[name] = dict(by_weight=dict(zip(map(str, WEIGHTS), accs)),
                                 no_verifier=accs[0], at_executed=accs[WEIGHTS.index(0.35)],
                                 best=max(accs), best_weight=WEIGHTS[int(np.argmax(accs))])
            OUT[f"{bb}::{ds}"] = dict(n=len(rows), **rec)
    json.dump(OUT, open(ws("results", "verifier_transfer.json"), "w"), indent=1)
    print("\n%-22s %8s %10s %10s %10s %10s" % ("cell", "n", "no verif.", "transf@.35", "own@.35", "own best"))
    for k, v in OUT.items():
        t = v["transferred (MedQA-trained)"]; o = v["own benchmark"]
        print("%-22s %8d %10.1f %10.1f %10.1f %10.1f" % (k, v["n"], t["no_verifier"], t["at_executed"],
                                                         o["at_executed"], o["best"]))
if __name__ == "__main__":
    main()
