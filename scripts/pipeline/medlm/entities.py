"""Biomedical entity extraction and retention scoring.

Matching is phrase-aware: multi-word vocabulary entries are matched as normalised
n-grams, so an entry containing a space is reachable.

Two target sets are computed for every candidate, the question-only set E(q) and
the whole-prompt set E(prompt), which includes the retrieved exemplars. Eq. (17)
of the paper uses E(q); both are reported so the difference between them is
visible rather than assumed.

Entities come from scispaCy `en_ner_bc5cdr_md` (BC5CDR: chemicals and diseases)
when it is importable. Otherwise the stored BC5CDR vocabulary is phrase-matched,
which is slower and slightly narrower; which path was taken is recorded in the
run manifest.
"""
import json
import re

_WS = re.compile(r"\s+")
_STRIP = ".,;:()[]{}\"'?!"

_NLP = None
_NLP_TRIED = False


def _try_scispacy():
    global _NLP, _NLP_TRIED
    if _NLP_TRIED:
        return _NLP
    _NLP_TRIED = True
    try:
        import spacy
        _NLP = spacy.load("en_ner_bc5cdr_md")
    except Exception:
        _NLP = None
    return _NLP


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        v = json.load(f)
    if isinstance(v, dict):
        v = list(v.keys())
    return {_norm(x) for x in v if isinstance(x, str) and len(x.strip()) > 2}


def _norm(t):
    return _WS.sub(" ", (t or "").strip().lower())


def _tokens(text):
    return [w.strip(_STRIP).lower() for w in _WS.split(text or "") if w.strip(_STRIP)]


def extract_entities(text, vocab, max_ngram=4, use_ner=True):
    """Return the set of biomedical entities in `text`.

    Uses scispaCy when available; otherwise phrase-matches against `vocab`.
    """
    if use_ner:
        nlp = _try_scispacy()
        if nlp is not None:
            doc = nlp(text or "")
            return {_norm(e.text) for e in doc.ents if len(e.text.strip()) > 2}
    toks = _tokens(text)
    found = set()
    n = len(toks)
    for size in range(min(max_ngram, n), 0, -1):
        for i in range(n - size + 1):
            cand = " ".join(toks[i:i + size])
            if cand in vocab:
                found.add(cand)
    return found


def retention(candidate_ents, target_ents):
    """Question-entity retention: |E(a) ∩ E(target)| / |E(target)|.

    Returns None when the target set is empty, so the caller can exclude the
    example rather than scoring it as 0 (the manuscript states exclusion; the
    original code scored 0 and still weighted it).
    """
    if not target_ents:
        return None
    return len(candidate_ents & target_ents) / len(target_ents)
