"""Consistency checks over the reported numbers.
C05 exact weight comparison of the contrastive stage; C13 input-truncation audit and
source-vs-summary qualifier audit; C08 uncertainty and tie handling for the query-level
verifier-selection statistic."""

import os, sys
import json, os, re, glob, numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
OUT = {}

# ---------------------------------------------------------------- C05: did the contrastive stage change any weight?
import torch
from safetensors.torch import load_file
def embed_tensor(ckpt):
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        m = json.load(open(idx))["weight_map"]
        key = next(k for k in m if "embed_tokens.weight" in k)
        return load_file(os.path.join(ckpt, m[key]))[key], key
    f = os.path.join(ckpt, "model.safetensors")
    d = load_file(f); key = next(k for k in d if "embed_tokens.weight" in k); return d[key], key
c05 = {}
for ds in ("medquad", "icliniq"):
    a, b = ws("experiments", f"ckpt_{ds}_llama31_seed42"), ws("experiments", f"ckpt_{ds}_llama31c_seed42")
    if not (os.path.exists(b)): continue
    ta, k = embed_tensor(a); tb, _ = embed_tensor(b)
    same = bool(torch.equal(ta, tb))
    diff = (ta.float() - tb.float()).abs()
    nz = int((diff > 0).sum()); tot = diff.numel()
    c05[ds] = dict(tensor=k, bitwise_equal=same, max_abs_change=float(diff.max()), mean_abs_change=float(diff.mean()),
                   changed_weights=nz, total_weights=tot, changed_fraction=round(nz / tot, 6),
                   dtype=str(ta.dtype), max_change_relative_to_eps=round(float(diff.max()) / float(torch.finfo(ta.dtype).eps), 4))
    print("C05", ds, c05[ds])
OUT["c05_contrastive_weights"] = c05

# ---------------------------------------------------------------- C13: does the target question survive the 512-token cap?
from transformers import AutoTokenizer
import medlm.openended_compat as R
tok = AutoTokenizer.from_pretrained(ws("experiments", "ckpt_medquad_llama31_seed42"))
c13 = {}
for ds in ("medquad", "icliniq"):
    man = json.load(open(ws("experiments", f"openended_{ds}_llama31s42", "manifest.json")))
    train, test = R.load_split(ds, man["n_train_pool"], man["n_test"], man["seed"])
    retr = R.Retriever(train, "cpu", False)
    qe = retr.encode([t["question"] for t in test])
    kept, lost, tot_tok = 0, 0, []
    for i, t in enumerate(test):
        ids = retr.search(qe[i], "general", 3)
        pr = R.build_prompt(t["question"], [train[j] for j in ids], "summary")
        full = tok(pr, add_special_tokens=False)["input_ids"]
        trunc = tok(pr, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        tot_tok.append(len(full))
        # the question survives if its text is still present in the decoded truncated prompt
        q_head = " ".join(t["question"].split()[:8]).lower()
        (kept, lost) = (kept + 1, lost) if q_head in tok.decode(trunc).lower() else (kept, lost + 1)
    c13[ds] = dict(n=len(test), mean_prompt_tokens=round(float(np.mean(tot_tok)), 1),
                   median_prompt_tokens=float(np.median(tot_tok)), pct_over_512=round(100 * float(np.mean(np.array(tot_tok) > 512)), 1),
                   question_survives_truncation=kept, question_lost=lost,
                   truncation_side=tok.truncation_side)
    print("C13-truncation", ds, c13[ds])
OUT["c13_truncation"] = c13

# ---------------------------------------------------------------- C13: do the summaries drop qualifiers?
QUAL = re.compile(r"\b(not|no|never|unless|only if|except|avoid|contraindicat\w*|should not|must not|do not|without|rare(ly)?|may|can|risk)\b", re.I)
NUM = re.compile(r"\b\d+(\.\d+)?\s*(mg|ml|g|mcg|%|percent|hours?|days?|weeks?|months?|years?)\b", re.I)
c13b = {}
for ds, fn in (("medquad", "train_summaries_medquad.json"), ("icliniq", "train_summaries_icliniq.json")):
    p = src("outputs_llm", fn)
    if not os.path.exists(p): continue
    summ = json.load(open(p, encoding="utf-8"))
    rows = [json.loads(l) for l in open(src("data", "medquad" if ds == "medquad" else "iCliniq", "train.jsonl"), encoding="utf-8")]
    drops, nums, ex = 0, 0, None
    n = min(len(summ), len(rows), 2000)
    for i in range(n):
        a = rows[i]["answer"]; s = R.strip_echo(summ[i]) if hasattr(R, "strip_echo") else summ[i]
        qa, qs = set(x.group(0).lower() for x in QUAL.finditer(a)), set(x.group(0).lower() for x in QUAL.finditer(s))
        na, ns = set(x.group(0).lower() for x in NUM.finditer(a)), set(x.group(0).lower() for x in NUM.finditer(s))
        if qa - qs: drops += 1
        if na - ns: nums += 1
        if ex is None and (na - ns) and len(a) < 900 and len(s) < 600 and ("not" in a.lower()):
            ex = dict(index=i, answer=a[:600], summary=s[:400], lost_numeric=sorted(na - ns)[:4], lost_qualifiers=sorted(qa - qs)[:6])
    c13b[ds] = dict(n_checked=n, answers_losing_a_qualifier_pct=round(100 * drops / n, 1),
                    answers_losing_a_dose_or_duration_pct=round(100 * nums / n, 1), example=ex)
    print("C13-qualifiers", ds, {k: v for k, v in c13b[ds].items() if k != "example"})
OUT["c13_qualifiers"] = c13b

# ---------------------------------------------------------------- C08: query-level selection uncertainty and ties
from rouge_score import rouge_scorer
rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
cand = ws("experiments", "openended_medquad_llama31s42", "cands_rag_n5.jsonl")
if os.path.exists(cand):
    rows = [json.loads(l) for l in open(cand, encoding="utf-8")]
    rng = np.random.default_rng(42); hit, reg, ties = [], [], 0
    for r in rows:
        r1 = np.array([rs.score(r["reference"], c)["rouge1"].fmeasure for c in r["candidates"]])
        v = np.array(r["verifier"], float)
        best = np.flatnonzero(r1 >= r1.max() - 1e-12); pick = int(np.argmax(v))
        if len(np.flatnonzero(v >= v.max() - 1e-12)) > 1: ties += 1
        hit.append(pick in best); reg.append(r1.max() - r1[pick])
    hit = np.array(hit, float); reg = np.array(reg)
    bh = hit[rng.integers(0, len(hit), (10000, len(hit)))].mean(1); br = reg[rng.integers(0, len(reg), (10000, len(reg)))].mean(1)
    OUT["c08_query_selection"] = dict(n=len(rows), oracle_match_pct=round(100 * float(hit.mean()), 1),
        oracle_match_ci=[round(100 * float(np.percentile(bh, 2.5)), 1), round(100 * float(np.percentile(bh, 97.5)), 1)],
        mean_regret_pts=round(100 * float(reg.mean()), 2),
        regret_ci=[round(100 * float(np.percentile(br, 2.5)), 2), round(100 * float(np.percentile(br, 97.5)), 2)],
        verifier_score_ties=ties, tie_rule="ties in the ROUGE-1 oracle count as a match; verifier ties broken by first index (numpy argmax)")
    print("C08", OUT["c08_query_selection"])
json.dump(OUT, open(ws("results", "audit_checks.json"), "w"), indent=1, default=str)
print("wrote results/audit_checks.json")
