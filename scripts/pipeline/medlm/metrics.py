# -*- coding: utf-8 -*-
"""Evaluation metrics and the paired bootstrap used throughout the paper."""
import numpy as np

_BERTSCORER = None


def score_pairs(predictions, references):
    """Per-example ROUGE-1/2/L F1 and METEOR."""
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    out = {k: [] for k in ("rouge1", "rouge2", "rougeL", "meteor")}
    for p, r in zip(predictions, references):
        s = scorer.score(r, p)
        out["rouge1"].append(s["rouge1"].fmeasure)
        out["rouge2"].append(s["rouge2"].fmeasure)
        out["rougeL"].append(s["rougeL"].fmeasure)
        out["meteor"].append(meteor_score([word_tokenize(r)], word_tokenize(p)))
    return out


def bertscore(predictions, references, device):
    """BERTScore F1 with lang="en" (roberta-large, layer 17, no baseline
    rescaling). The scorer is instantiated once and reused, since it is identical
    across conditions."""
    global _BERTSCORER
    if _BERTSCORER is None:
        import bert_score
        _BERTSCORER = bert_score.BERTScorer(lang="en", rescale_with_baseline=False,
                                            device=device, batch_size=64)
    _, _, f1 = _BERTSCORER.score(predictions, references)
    return f1.float().numpy().tolist()


def boot_ci(x, n_boot=10000, seed=42, level=95.0):
    """Percentile bootstrap over question resamples."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_boot_ci(a, b, n_boot=10000, seed=42, level=95.0):
    """Interval for the difference a - b on the SAME questions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(d, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)
