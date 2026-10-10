# -*- coding: utf-8 -*-
"""Retrieval precision and recall under a non-circular relevance criterion.

Focus agreement was rejected as a relevance proxy because focus is a near-synonym of the retrieval
criterion. Here relevance is judged by answer utility instead: a pool exemplar is relevant to a test
question if its UNSUMMARISED answer reaches ROUGE-1 F >= tau against that question's reference
answer. Retrieval ranks by question-embedding similarity, so the ranking signal and the relevance
signal are different quantities and precision@k is not measuring the retriever against itself.

Reports precision@3, recall@{1,3,5,10} and MRR at tau in {0.25, 0.35, 0.45}, for dense and for
question-type retrieval. CPU only: encodings are small and the relevance matrix is computed from
stemmed unigram counts, which is exactly ROUGE-1 F (verified against rouge_score below).

Writes results/retrieval_prf.json.
"""
import json, os, sys
from collections import Counter
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
import medlm.openended_compat as R
KS = (1, 3, 5, 10)
TAUS = (0.25, 0.35, 0.45)
TOPK = max(KS)

from rouge_score import tokenizers as _rs_tok
_TOK = _rs_tok.DefaultTokenizer(use_stemmer=True)
def bag(t): return Counter(_TOK.tokenize(t or ""))
def r1f(a, b):
    """ROUGE-1 F between two pre-computed stemmed unigram bags."""
    if not a or not b: return 0.0
    ov = sum((a & b).values())
    if ov == 0: return 0.0
    p, r = ov / sum(a.values()), ov / sum(b.values())
    return 2 * p * r / (p + r)

def verify_against_rouge_score(pairs):
    """Prove the fast path equals rouge_score's rouge1 fmeasure before relying on it."""
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    worst = 0.0
    for ref, cand in pairs:
        mine = r1f(bag(cand), bag(ref))
        theirs = rs.score(ref, cand)["rouge1"].fmeasure
        worst = max(worst, abs(mine - theirs))
    return worst

def main():
    man_path = ws("experiments", "openended_medquad_llama31T42", "manifest.json")
    man = json.load(open(man_path))
    train, test = R.load_split("medquad", man["n_train_pool"], man["n_test"], man["seed"])
    print(f"pool {len(train)} items, {len(test)} evaluated questions", flush=True)

    # --- verify the fast ROUGE-1 F before using it on 600k pairs
    sample = [(test[i]["answer"], train[j]["answer"]) for i in range(0, 40, 4) for j in range(0, 200, 40)]
    worst = verify_against_rouge_score(sample)
    print(f"fast ROUGE-1 F vs rouge_score on {len(sample)} pairs: max abs difference {worst:.2e}", flush=True)
    assert worst < 1e-9, "fast path does not reproduce rouge_score"

    # --- relevance matrix: unsummarised pool answer vs test reference
    pb = [bag(t["answer"]) for t in train]
    qb = [bag(t["answer"]) for t in test]
    Rel = np.zeros((len(test), len(train)), dtype=np.float32)
    for i, b in enumerate(qb):
        Rel[i] = [r1f(p, b) for p in pb]
    print(f"relevance matrix {Rel.shape}; mean {Rel.mean():.3f}, max {Rel.max():.3f}", flush=True)

    # --- retrieval rankings (top-10), dense and question-type, on CPU
    rank = {}
    for name, use_qt in (("dense", False), ("question_type", True)):
        retr = R.Retriever(train, "cpu", use_qt)
        qe = retr.encode([t["question"] for t in test])
        rows = []
        for i in range(len(test)):
            if use_qt:
                ids = retr.buckets.get(test[i]["qtype"], [])
                if len(ids) >= TOPK:
                    _s, I = retr.bucket_index[test[i]["qtype"]].search(qe[i][None, :], TOPK)
                    rows.append([ids[j] for j in I[0] if j >= 0]); continue
            _s, I = retr.global_index.search(qe[i][None, :], TOPK)
            rows.append([int(j) for j in I[0] if j >= 0])
        rank[name] = rows
        print(f"{name}: ranked top-{TOPK} for {len(rows)} questions (fallbacks {retr.fallbacks})", flush=True)

    # --- metrics
    OUT = dict(pool_items=len(train), questions=len(test), top_k=TOPK,
               relevance="unsummarised pool answer vs test reference answer, ROUGE-1 F >= tau",
               fast_rouge_max_abs_diff=float(worst), by_tau={})
    for tau in TAUS:
        rel = Rel >= tau
        n_rel = rel.sum(1)
        entry = dict(mean_relevant_per_question=round(float(n_rel.mean()), 2),
                     questions_with_no_relevant_exemplar=int((n_rel == 0).sum()),
                     prevalence_pct=round(100 * float(rel.mean()), 3), strategies={})
        for name, rows in rank.items():
            prec3, mrr, rec = [], [], {k: [] for k in KS}
            for i, ids in enumerate(rows):
                hits = [bool(rel[i, j]) for j in ids]
                if len(hits) >= 3: prec3.append(sum(hits[:3]) / 3)
                first = next((p + 1 for p, h in enumerate(hits) if h), None)
                mrr.append(1.0 / first if first else 0.0)
                if n_rel[i] > 0:
                    for k in KS:
                        rec[k].append(sum(hits[:k]) / n_rel[i])
            entry["strategies"][name] = dict(
                precision_at_3=round(100 * float(np.mean(prec3)), 2),
                mrr=round(float(np.mean(mrr)), 4),
                recall_at={str(k): round(100 * float(np.mean(rec[k])), 3) for k in KS},
                # a random-ranking reference point for precision@3
                random_precision_at_3=round(100 * float(rel.mean()), 3))
        OUT["by_tau"][str(tau)] = entry
    json.dump(OUT, open(ws("results", "retrieval_prf.json"), "w"), indent=1)
    print("\n%-6s %-14s %11s %10s %9s %9s %9s %9s" % ("tau", "strategy", "precision@3", "random@3", "R@1", "R@3", "R@5", "MRR"))
    for tau, e in OUT["by_tau"].items():
        for name, v in e["strategies"].items():
            print("%-6s %-14s %10.2f%% %9.3f%% %8.2f%% %8.2f%% %8.2f%% %9.4f" % (
                tau, name, v["precision_at_3"], v["random_precision_at_3"],
                v["recall_at"]["1"], v["recall_at"]["3"], v["recall_at"]["5"], v["mrr"]))
        print("%-6s %-14s relevant per question: %.2f; questions with none: %d" % (
            tau, "", e["mean_relevant_per_question"], e["questions_with_no_relevant_exemplar"]))
if __name__ == "__main__":
    main()
