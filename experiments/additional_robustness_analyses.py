# -*- coding: utf-8 -*-
"""Additional robustness analyses reported in Section 4.6.

All five analyses here reuse saved per-example outputs and need no GPU:

  A  retrieval_focus_agreement  -- a retrieval-relevance metric built from the recovered MedQuAD
                                   focus metadata (the labels that did not exist before)
  B  cumulative_ladder          -- the cumulative build-up, generator only -> ... -> full pipeline
  C  alpha_sweep                -- sensitivity of the combined selection rule to alpha, by
                                   re-selecting from the saved candidate pools
  D  fusion_sweep               -- sensitivity of closed-form accuracy to the 0.35 verifier fusion
                                   weight, by re-fusing the saved option log-probabilities
  E  narrowing_audit            -- whether the withdrawal of factuality/completeness/safety claims
                                   is consistent across highlights, abstract, results and conclusions

Writes results/extra_analyses.json.
"""
import os, sys, json, re, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def J(p, d=None):
    q = ws("results", p)
    return json.load(open(q)) if os.path.exists(q) else d
RNG = np.random.default_rng(42)
def boot(d, n=10000):
    d = np.asarray(d, dtype=float)
    if len(d) == 0: return dict(diff=None, ci=[None, None], n=0)
    bs = d[RNG.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(diff=round(100 * float(d.mean()), 2),
                ci=[round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)],
                n=int(len(d)))
OUT = {}

# ---------------------------------------------------------------- A. retrieval relevance from recovered focus labels
def retrieval_focus_agreement():
    """The exemplar pool carries no relevance judgements, so precision@k against a relevance label is
    not definable. The recovered MedQuAD focus (topic) metadata does give one checkable notion of
    relevance: does a retrieved exemplar share the query's topic entity? Reported as agreement@k for
    k=1..3, against the rate expected if the k exemplars were drawn from the pool at random.

    Indices: preds rows carry test_idx (an index into the test split file) and exemplar_ids (indices
    into the 2,000-row pool); the manifest's train_ids maps a pool index to its train-split row. The
    recovered metadata is stored index-aligned per split, so both sides resolve by position."""
    meta = J("medquad_recovered_meta.json")
    if not meta: return None
    tr_rec = (meta.get("train") or {}).get("records") or []
    te_rec = (meta.get("test") or {}).get("records") or []
    if not (tr_rec and te_rec): return None
    from collections import Counter
    res = {}
    for lab, pol in (("llama31s42", "submitted"), ("llama31T42", "corrected")):
        d = ws("experiments", f"openended_medquad_{lab}")
        f = f"{d}/preds_rag_n5_combined.jsonl"
        if not (os.path.exists(f) and os.path.exists(f"{d}/manifest.json")): continue
        man = json.load(open(f"{d}/manifest.json")); tids = man.get("train_ids") or []
        if not tids: continue
        def foc(rec): return (rec or {}).get("focus", "").strip().lower() or None
        pool_focus = [foc(tr_rec[i]) if 0 <= i < len(tr_rec) else None for i in tids]
        cnt = Counter(x for x in pool_focus if x); n_lab = sum(cnt.values())
        at_k = {1: [], 2: [], 3: []}; any_k = []; chance = []
        for r in rd(f):
            ti = r.get("test_idx"); ids = r.get("exemplar_ids") or []
            qf = foc(te_rec[ti]) if isinstance(ti, int) and 0 <= ti < len(te_rec) else None
            if not qf or not ids: continue
            hits = [1.0 if (0 <= i < len(pool_focus) and pool_focus[i] == qf) else 0.0 for i in ids[:3]]
            for k in (1, 2, 3):
                if len(hits) >= k: at_k[k].append(float(np.mean(hits[:k])))
            any_k.append(1.0 if any(hits) else 0.0)
            chance.append(cnt.get(qf, 0) / max(n_lab, 1))
        if not any_k: continue
        ch = float(np.mean(chance))
        res[pol] = dict(n_questions=len(any_k), pool_items=len(pool_focus), pool_items_with_focus=n_lab,
                        distinct_pool_foci=len(cnt),
                        focus_agreement_at_1=round(100 * float(np.mean(at_k[1])), 2),
                        focus_agreement_at_2=round(100 * float(np.mean(at_k[2])), 2),
                        focus_agreement_at_3=round(100 * float(np.mean(at_k[3])), 2),
                        at_least_one_same_focus_pct=round(100 * float(np.mean(any_k)), 2),
                        chance_agreement_pct=round(100 * ch, 3),
                        lift_over_chance=round(float(np.mean(at_k[3])) / ch, 1) if ch > 0 else None)
    return res or None

# ---------------------------------------------------------------- B. cumulative ablation ladder
LADDER = [("norag_n1_first", "Generator only (N=1, no retrieval)"),
          ("norag_n5_combined", "+ candidate budget N=5 with combined selection"),
          ("rag_n5_combined", "+ dense exemplar retrieval (k=3)"),
          ("rag_qtype_n5_combined", "+ question-type stratified retrieval"),
          ("rag_edec_n5_combined", "+ decoding-time entity bonus")]
def cumulative_ladder():
    """Each rung adds one component to the rung above it, on one fixed generator, so the cumulative
    build-up can be read off alongside the independent grid."""
    res = {}
    for lab, pol in (("llama31T42", "corrected"), ("llama31s42", "submitted")):
        for ds in ("medquad", "icliniq"):
            d = ws("experiments", f"openended_{ds}_{lab}")
            have = [(c, n) for c, n in LADDER if os.path.exists(f"{d}/preds_{c}.jsonl")]
            if len(have) < 2: continue
            per = {c: {r["test_idx"]: r for r in rd(f"{d}/preds_{c}.jsonl")} for c, _ in have}
            keys = set.intersection(*[set(v) for v in per.values()])
            rungs = []
            for i, (c, name) in enumerate(have):
                lvl = {m: round(100 * float(np.mean([per[c][k][m] for k in keys])), 2)
                       for m in ("rouge1", "meteor", "bertscore_f1")}
                step = None
                if i:
                    prev = have[i - 1][0]
                    step = boot([per[c][k]["rouge1"] - per[prev][k]["rouge1"] for k in keys])
                cum = None
                if i:
                    base = have[0][0]
                    cum = boot([per[c][k]["rouge1"] - per[base][k]["rouge1"] for k in keys])
                rungs.append(dict(condition=c, label=name, n=len(keys), **lvl,
                                  step_rouge1=step, cumulative_rouge1=cum))
            res[f"{ds}::{pol}"] = rungs
    return res or None

# ---------------------------------------------------------------- C. alpha sweep of the selection rule
def alpha_sweep(alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Re-selects among the saved N=5 candidate pools at several alpha values and rescores, so the
    near-zero selection effect can be checked against the weight rather than assumed independent of it.
    ROUGE and METEOR only: they are deterministic text metrics and need no model."""
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    res = {}
    for lab, pol in (("llama31T42", "corrected"), ("llama31s42", "submitted")):
        for ds in ("medquad", "icliniq"):
            f = ws("experiments", f"openended_{ds}_{lab}", "cands_rag_n5.jsonl")
            if not os.path.exists(f): continue
            items = rd(f)
            rows = []
            for a in alphas:
                r1, mt, picks = [], [], []
                for it in items:
                    ver, cov = it["verifier"], it["cov_q"]
                    sc = [a * ver[j] + (1 - a) * (0.0 if cov[j] is None else cov[j])
                          for j in range(len(it["candidates"]))]
                    j = int(np.argmax(sc)); picks.append(j)
                    p, ref = it["candidates"][j], it["reference"]
                    r1.append(rs.score(ref, p)["rouge1"].fmeasure)
                    try: mt.append(meteor_score([word_tokenize(ref)], word_tokenize(p)))
                    except Exception: pass
                rows.append(dict(alpha=a, rouge1=round(100 * float(np.mean(r1)), 2),
                                 meteor=round(100 * float(np.mean(mt)), 2) if mt else None,
                                 first_beam_picked_pct=round(100 * float(np.mean([p == 0 for p in picks])), 1)))
            # the executed rule is alpha=0.75; state the spread over the sweep
            vals = [x["rouge1"] for x in rows]
            res[f"{ds}::{pol}"] = dict(n=len(items), rows=rows,
                                       rouge1_spread=round(max(vals) - min(vals), 2),
                                       best_alpha=rows[int(np.argmax(vals))]["alpha"],
                                       executed_alpha=0.75)
    return res or None

# ---------------------------------------------------------------- D. closed-form fusion-weight sweep
def fusion_sweep(weights=(0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)):
    """Re-fuses the saved per-option log-probabilities with the saved verifier scores at several
    weights. w=0 is pure log-probability (the task-adapted system) and w=1 is the verifier alone."""
    def minmax(v):
        v = np.asarray(v, dtype=float)
        return np.zeros_like(v) if v.max() - v.min() < 1e-12 else (v - v.min()) / (v.max() - v.min())
    res = {}
    for bb in ("llama31", "qwen3", "deepseek"):
        for ds in ("medqa", "mmlu_med", "medmcqa", "pubmedqa"):
            f = ws("experiments", f"closed_{ds}_{bb}ft", "preds_adapted_verifier.jsonl")
            if not os.path.exists(f): continue
            items = [r for r in rd(f) if r.get("verifier") and r.get("logprobs")]
            if not items: continue
            rows = []
            for w in weights:
                ok = 0
                for r in items:
                    comb = (1 - w) * minmax(r["logprobs"]) + w * minmax(r["verifier"])
                    ok += int(int(np.argmax(comb)) == int(r["gold"]))
                rows.append(dict(weight=w, accuracy_pct=round(100 * ok / len(items), 2)))
            accs = [x["accuracy_pct"] for x in rows]
            res[f"{bb}::{ds}"] = dict(n=len(items), rows=rows, executed_weight=0.35,
                                      best_weight=rows[int(np.argmax(accs))]["weight"],
                                      accuracy_at_0=rows[0]["accuracy_pct"],
                                      accuracy_at_executed=next(x["accuracy_pct"] for x in rows if x["weight"] == 0.35),
                                      best_accuracy=max(accs))
    return res or None

# ---------------------------------------------------------------- E. consistency of the claim narrowing
def narrowing_audit():
    """asked for the narrowing to be consistent throughout, not only in the limitations. This
    checks each surface of the paper for (a) any surviving unqualified factuality / completeness /
    safety claim and (b) the presence of an explicit disclaimer."""
    tex = os.path.join(os.path.dirname(WS), "final", "main_revised_highlighted.tex")
    if not os.path.exists(tex): return None
    t = open(tex, encoding="utf-8").read()
    def span(a, b):
        i = t.find(a)
        if i < 0: return ""
        j = t.find(b, i + len(a))
        return t[i:j if j > 0 else len(t)]
    SURF = {
        "Highlights": span(r"\begin{highlights}", r"\end{highlights}") or span("Highlights", r"\begin{abstract}"),
        "Abstract": span(r"\begin{abstract}", r"\end{abstract}"),
        "Introduction": span(r"\section{Introduction}", r"\section{Method"),
        "Methods": span(r"\section{Method", r"\section{Results"),
        "Results": span(r"\section{Results", r"\section{Discussion"),
        "Discussion": span(r"\section{Discussion", r"\section{Conclusion"),
        "Conclusions": span(r"\section{Conclusion", r"\section*{Data and code"),
    }
    # an unqualified claim: the word appears without a negation or hedge nearby
    CLAIM = re.compile(r"(factual(?:ly)? (?:correct|accura\w+|plausib\w+|reliab\w+)|clinical(?:ly)? "
                       r"(?:complete\w*|safe\w*|valid\w*)|guarantee\w*|ensures? (?:correctness|safety)|"
                       r"establish\w* (?:factual|clinical))", re.I)
    NEG = re.compile(r"\b(not|no|nor|never|without|cannot|does not|do not|neither|rather than|"
                     r"withdraw\w*|removed|instead of|is not a measure)\b", re.I)
    # a mention is not a claim about this system when it reports what other work says, states what
    # future work should do, or defines what the metrics would need to capture
    CTX = re.compile(r"(future work|should focus|literature|studies also show|prior work|require[sd]?\b|"
                     r"insufficient|\\cite|remains necessary|is required before|clinician-led)", re.I)
    DISC = re.compile(r"(do(?:es)? not establish|not a measure of|no clinician|not evidence of|"
                      r"remains necessary before|prospective clinician)", re.I)
    res = {}
    for name, seg in SURF.items():
        seg = re.sub(r"\s+", " ", seg)
        unq, ctx = [], []
        for m in CLAIM.finditer(seg):
            w = seg[max(0, m.start() - 330):m.end() + 200]
            if NEG.search(w): continue
            (ctx if CTX.search(w) else unq).append(m.group(0))
        res[name] = dict(chars=len(seg), claim_mentions=len(CLAIM.findall(seg)),
                         unqualified=unq, not_a_claim_about_this_system=ctx,
                         has_explicit_disclaimer=bool(DISC.search(seg)),
                         makes_no_such_claim=len(CLAIM.findall(seg)) == 0)
    body = {k: v for k, v in res.items() if not k.startswith("_")}
    res["_summary"] = dict(
        surfaces=len(body),
        surfaces_with_unqualified_claim=sum(1 for v in body.values() if v["unqualified"]),
        surfaces_with_disclaimer=sum(1 for v in body.values() if v["has_explicit_disclaimer"]),
        surfaces_making_no_such_claim=sum(1 for v in body.values() if v["makes_no_such_claim"]),
        consistent=all(v["unqualified"] == [] for v in body.values()))
    return res

if __name__ == "__main__":
    for name, fn in (("retrieval_focus_agreement", retrieval_focus_agreement),
                     ("cumulative_ladder", cumulative_ladder),
                     ("alpha_sweep", alpha_sweep),
                     ("fusion_sweep", fusion_sweep),
                     ("narrowing_audit", narrowing_audit)):
        try:
            OUT[name] = fn(); print(f"[ok] {name}")
        except Exception as ex:
            OUT[name] = None; print(f"[FAIL] {name}: {type(ex).__name__}: {ex}")
    json.dump(OUT, open(ws("results", "extra_analyses.json"), "w"), indent=1)
    print("wrote", ws("results", "extra_analyses.json"))
