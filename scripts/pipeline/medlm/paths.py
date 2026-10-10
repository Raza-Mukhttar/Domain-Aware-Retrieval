# -*- coding: utf-8 -*-
"""Path resolution for the whole repository.

Every path is derived from the repository root, which is located relative to this
file, so the tree can be cloned anywhere. Two environment variables override the
defaults:

  MEDLM_DATA_DIR     where the dataset splits live      (default: <root>/data)
  MEDLM_OUTPUT_DIR   where runs write their artefacts   (default: <root>/outputs)

Model weights are never stored in the repository. MEDLM_MODEL_CACHE points at the
Hugging Face cache used for the backbones; if it is unset, the Hugging Face
default applies.
"""
import os

# medlm/paths.py -> medlm -> pipeline -> scripts -> repository root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA = os.environ.get("MEDLM_DATA_DIR", os.path.join(ROOT, "data"))
OUTPUT = os.environ.get("MEDLM_OUTPUT_DIR", os.path.join(ROOT, "outputs"))
CONFIG = os.path.join(ROOT, "config")
LOGS = os.path.join(ROOT, "logs")

# Sentence encoder for exemplar retrieval. A local directory is used when one is
# given, otherwise the Hub identifier is resolved by sentence-transformers.
EMBEDDER = os.environ.get("MEDLM_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")

_cache = os.environ.get("MEDLM_MODEL_CACHE")
if _cache:
    os.environ.setdefault("HF_HOME", _cache)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(_cache, "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _cache)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The three open-ended corpora held in this repository. Closed-form benchmarks
# (MedQA, MMLU-Med, MedMCQA, PubMedQA) are pulled from the Hub at run time and are
# therefore not stored here.
DATASETS = {
    "medquad": dict(
        dir=os.path.join(DATA, "medquad"),
        summaries=os.path.join(DATA, "derived", "train_summaries_medquad.json")),
    "icliniq": dict(
        dir=os.path.join(DATA, "icliniq"),
        summaries=os.path.join(DATA, "derived", "train_summaries_icliniq.json")),
    # Source/topic-disjoint repartition of MedQuAD: no (source document, focus
    # entity) group appears in more than one partition, so the retrieval pool
    # shares no topic or source document with the evaluated questions.
    "medquad_group_disjoint": dict(
        dir=os.path.join(DATA, "medquad_group_disjoint"),
        summaries=os.path.join(DATA, "derived",
                               "train_summaries_medquad_group_disjoint.json")),
}

ENTITY_VOCAB = os.path.join(DATA, "derived", "medical_entities.json")


def data(*parts):
    return os.path.join(DATA, *parts)


def output(*parts):
    p = os.path.join(OUTPUT, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def config(*parts):
    return os.path.join(CONFIG, *parts)
