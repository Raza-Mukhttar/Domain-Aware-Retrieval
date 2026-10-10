"""Question-type assignment for MedQuAD / iCliniq-style questions.

MedQuAD questions are generated from a small set of templated stems, so the type
is recoverable from the stem by regular expression. This replaces the previous
behaviour, where `qtype` was read from a dataset metadata field that is empty in
every row and therefore always fell back to the single bucket "general".

Taxonomy follows the MedQuAD question-type set used by Ben Abacha & Demner-Fushman
(2019), restricted to the types that actually occur in these splits.
"""
import re

# Ordered: the first matching pattern wins, so specific stems precede general ones.
RULES = [
    ("treatment",   r"\b(treatment|treatments|therapy|therapies|how (?:is|are) .* treated|manage(?:d|ment)|cure)\b"),
    ("diagnosis",   r"\b(diagnos\w*|how to diagnose|tests? for|screening)\b"),
    ("prevention",  r"\b(prevent\w*|how can .* be avoided)\b"),
    ("causes",      r"\b(cause[sd]?|causes? of|what causes|etiolog\w*|risk factors?)\b"),
    ("symptoms",    r"\b(symptom\w*|sign[s]? and symptoms?|what are the signs)\b"),
    ("prognosis",   r"\b(outlook|prognos\w*|survival|life expectancy)\b"),
    ("inheritance", r"\b(inherit\w*|genetic\w* (?:change|cause)|passed down)\b"),
    ("frequency",   r"\b(how many people|frequency|prevalence|how common)\b"),
    ("side_effects", r"\b(side effects?|adverse|interactions?)\b"),
    ("dosage",      r"\b(dosage|dose|how much .* take|how should .* use)\b"),
    ("information", r"\b(what (?:is|are)|do you have information|definition)\b"),
]
COMPILED = [(name, re.compile(pat, re.I)) for name, pat in RULES]
FALLBACK = "general"


def assign_qtype(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return FALLBACK
    for name, rx in COMPILED:
        if rx.search(q):
            return name
    return FALLBACK


def qtype_distribution(questions):
    from collections import Counter
    return Counter(assign_qtype(q) for q in questions)
