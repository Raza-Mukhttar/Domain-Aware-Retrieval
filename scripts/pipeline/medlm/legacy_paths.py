# -*- coding: utf-8 -*-
"""Path mapping for the scripts carried over from the executed runs.

Those scripts address artefacts by their position in the original working tree,
through two helpers: src() for inputs that lived beside the code and ws() for a
run's own outputs. Rewriting every call site would mean touching code whose
behaviour produced the published numbers, so the helpers are kept and the two
mappings are made explicit here instead. Each entry records where an artefact
lives in this repository.
"""
import os

from . import paths

_VERIFIER_ENV = "MEDLM_VERIFIER"

# Inputs that the executed scripts addressed as src(...).
_SRC = {
    ("all-MiniLM-L6-v2",): None,                       # resolved to paths.EMBEDDER
    ("data", "medquad"): ("medquad",),
    ("data", "iCliniq"): ("icliniq",),
    ("outputs_llm", "medical_entities.json"): ("derived", "medical_entities.json"),
    ("outputs_llm", "train_summaries_medquad.json"): ("derived", "train_summaries_medquad.json"),
    ("outputs_llm", "train_summaries_icliniq.json"): ("derived", "train_summaries_icliniq.json"),
}

# Outputs that the executed scripts addressed as ws(...). The first element of
# the original path selects the destination.
_WS_ROOTS = {
    "experiments": ("runs",),
    "results": ("analysis",),
    "figures": ("figures",),
    "logs": None,                                      # resolved to paths.LOGS
    "data_groupsplit": None,                           # resolved under paths.DATA
}


_RENAME = {"iCliniq": "icliniq", "medquad_gs": "medquad_group_disjoint",
           "outputs_llm": "derived"}


def src(*parts):
    """Resolve an input path of the original working tree."""
    key = tuple(parts)
    if len(key) > 2 and key[0] == "data":
        # data/<corpus>/<file>: the corpus directories moved to the top of data/
        return paths.data(*(_RENAME.get(p, p) for p in key[1:]))
    if len(key) > 2 and key[0] == "outputs_llm":
        return paths.data("derived", *key[1:])
    if key == ("all-MiniLM-L6-v2",):
        return paths.EMBEDDER
    if key == ("outputs_llm", "verifier"):
        v = os.environ.get(_VERIFIER_ENV)
        if not v:
            raise RuntimeError(
                f"the ClinicalBERT verifier is not stored in this repository; set {_VERIFIER_ENV} "
                "to its directory or hub id, or pass --verifier explicitly")
        return v
    if key in _SRC:
        return paths.data(*_SRC[key])
    # anything else is data that travels with the repository
    return paths.data(*parts)


def ws(*parts):
    """Resolve an output path of the original working tree."""
    if not parts:
        return paths.OUTPUT
    head, rest = parts[0], parts[1:]
    if head == "logs":
        return os.path.join(paths.LOGS, *rest)
    if head == "data_groupsplit":
        # the group-disjoint repartition is shipped as a dataset, not as an output
        rest = tuple("medquad_group_disjoint" if r == "medquad_gs" else r for r in rest)
        return paths.data(*rest)
    mapped = _WS_ROOTS.get(head)
    target = os.path.join(paths.OUTPUT, *(mapped or (head,)), *rest)
    parent = os.path.dirname(target) if os.path.splitext(target)[1] else target
    os.makedirs(parent, exist_ok=True)
    return target


# Kept so carried-over scripts that import these names continue to resolve.
NEW = paths.DATA
WS = paths.OUTPUT
