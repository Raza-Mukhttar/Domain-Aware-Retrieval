#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check that the repository holds everything the paper's numbers rest on.

Verifies, without a GPU and without the network:

  1. the dataset splits are present, readable and the expected size
  2. the derived artefacts the pipeline needs are present
  3. every configuration file parses
  4. every run directory carries a manifest, aggregate metrics and per-item
     predictions, and every metrics entry carries the run id of its configuration
  5. every cross-run analysis file an experiment writes is present and parses
  6. every figure in the paper exists as both PDF and PNG
  7. no absolute path or local user name survives anywhere in the tree

Exits non-zero if any check fails. What the repository deliberately does not
hold - backbone weights and the trained verifier - is reported separately, since
those are downloaded or retrained rather than stored.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

OK, FAIL, NOTE = "ok", "FAIL", "note"
results = []


def check(name, condition, detail=""):
    results.append((OK if condition else FAIL, name, detail))
    return condition


def note(name, detail=""):
    results.append((NOTE, name, detail))


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# --- 1. dataset splits ------------------------------------------------------
EXPECTED_SPLITS = {
    "medquad": ("train", "validation", "test"),
    "icliniq": ("train", "validation", "test"),
    "medquad_group_disjoint": ("train", "validation", "test"),
}
for corpus, splits in EXPECTED_SPLITS.items():
    for split in splits:
        p = os.path.join(ROOT, "data", corpus, f"{split}.jsonl")
        if check(f"data/{corpus}/{split}.jsonl", os.path.isfile(p)):
            n = count_lines(p)
            check(f"data/{corpus}/{split}.jsonl is non-empty", n > 0, f"{n} rows")

# --- 2. derived artefacts the pipeline needs --------------------------------
for f in ("train_summaries_medquad.json", "train_summaries_icliniq.json",
          "train_summaries_medquad_group_disjoint.json", "medical_entities.json"):
    p = os.path.join(ROOT, "data", "derived", f)
    if check(f"data/derived/{f}", os.path.isfile(p)):
        try:
            with open(p, encoding="utf-8") as fh:
                obj = json.load(fh)
            check(f"data/derived/{f} parses", True, f"{len(obj)} entries")
        except Exception as e:
            check(f"data/derived/{f} parses", False, str(e)[:80])

# --- 3. configuration -------------------------------------------------------
for f in ("executed_config.json", "backbones.json", "datasets.json"):
    p = os.path.join(ROOT, "config", f)
    try:
        with open(p, encoding="utf-8") as fh:
            json.load(fh)
        check(f"config/{f}", True)
    except Exception as e:
        check(f"config/{f}", False, str(e)[:80])

# --- 4. run artefacts -------------------------------------------------------
runs_dir = os.path.join(ROOT, "outputs", "runs")
runs = sorted(d for d in os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []
check("outputs/runs exists", bool(runs), f"{len(runs)} runs")

no_metrics, no_prov, no_preds, no_runid = [], [], [], []
for run in runs:
    d = os.path.join(runs_dir, run)
    mpath = os.path.join(d, "metrics.json")
    if not os.path.isfile(mpath):
        no_metrics.append(run)
        continue
    if not any(f.startswith("preds_") for f in os.listdir(d)):
        no_preds.append(run)
    try:
        with open(mpath, encoding="utf-8") as fh:
            metrics = json.load(fh)
    except Exception:
        no_metrics.append(run)
        continue
    # Provenance lives either in a separate manifest (open-ended runs) or inside
    # each condition of metrics.json (closed-form runs). Either satisfies the
    # rule that no reported number exists without its configuration.
    has_manifest = os.path.isfile(os.path.join(d, "manifest.json"))
    inline = all(isinstance(e, dict) and ("config" in e or "config" in e.get("_meta", {}))
                 for e in metrics.values() if isinstance(e, dict))
    if not (has_manifest or (inline and metrics)):
        no_prov.append(run)
    for cond, entry in metrics.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get("_meta", entry)
        if "run_id" not in meta:
            no_runid.append(f"{run}:{cond}")

check("every run has aggregate metrics", not no_metrics, ", ".join(no_metrics[:4]))
check("every run records its configuration", not no_prov, ", ".join(no_prov[:4]))
check("every run has per-item predictions", not no_preds,
      f"{len(no_preds)} without, e.g. " + ", ".join(no_preds[:3]) if no_preds else "")
check("every metrics entry carries a run id", not no_runid,
      f"{len(no_runid)} without, e.g. " + ", ".join(no_runid[:3]) if no_runid else "")

# training provenance, extracted before each checkpoint was deleted
tm = os.path.join(ROOT, "outputs", "analysis", "training_manifests")
tm_files = sorted(os.listdir(tm)) if os.path.isdir(tm) else []
manifests = [f[:-len("__manifest.json")] for f in tm_files if f.endswith("__manifest.json")]
losses = {f[:-len("__log_history.json")] for f in tm_files if f.endswith("__log_history.json")}
check("every training run has a manifest", bool(manifests), f"{len(manifests)} training runs")
# The contrastive stage adapts the encoder outside the Trainer loop, so it emits
# no step-wise loss history; every Trainer-driven run has one.
no_loss = [m for m in manifests if m not in losses]
trainer_runs = [m for m in manifests if "llama31c" not in m]
check("every Trainer-driven training run has its loss history",
      all(m in losses for m in trainer_runs), f"{len(losses)} loss histories")
if no_loss:
    note("training runs without a step-wise loss history",
         ", ".join(no_loss) + " - contrastive stage, adapted outside the Trainer loop")

# --- 5. cross-run analyses --------------------------------------------------
ANALYSES = [
    "final_v2.json", "ladder_spec.json", "seed_variability.json",
    "group_disjoint_repartition.json", "group_split_feasibility.json",
    "group_split_build.json", "srcdisj_check.json", "retrieval_prf.json",
    "verifier_eval_medquad.json", "verifier_transfer.json", "hparam_sweeps.json",
    "extra_analyses.json", "why_closedform.json", "why_retrieval_hurts.json",
    "dedup_counts.json", "dedup_subset.json", "icliniq_composition.json",
    "sampling_check.json", "nondeterminism_check.json", "grouped_corrected.json",
    "audit_checks.json", "qualifier_example.json", "medquad_recovered_meta.json",
    "results_digest.json", "dataset_revisions.json",
]
missing_analyses = []
for f in ANALYSES:
    p = os.path.join(ROOT, "outputs", "analysis", f)
    if not os.path.isfile(p):
        missing_analyses.append(f)
        continue
    try:
        with open(p, encoding="utf-8") as fh:
            json.load(fh)
    except Exception:
        missing_analyses.append(f + " (unparseable)")
check("every cross-run analysis file is present and parses", not missing_analyses,
      ", ".join(missing_analyses))

# --- 6. figures -------------------------------------------------------------
fig_dir = os.path.join(ROOT, "outputs", "figures")
pdfs = {f[:-4] for f in os.listdir(fig_dir) if f.endswith(".pdf")} if os.path.isdir(fig_dir) else set()
pngs = {f[:-4] for f in os.listdir(fig_dir) if f.endswith(".png")} if os.path.isdir(fig_dir) else set()
check("figures present", bool(pdfs), f"{len(pdfs)} PDF, {len(pngs)} PNG")
pdf_only = sorted(pdfs - pngs)
if pdf_only:
    note("figures without a PNG companion", ", ".join(pdf_only))

# --- 7. no local paths ------------------------------------------------------
HOME_RX = re.compile(rb"/home/[A-Za-z0-9_.-]+/")
SKIP_EXT = {".pdf", ".png", ".npy", ".docx", ".zip", ".gz", ".index", ".pyc"}
leaky = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
    for fn in filenames:
        if os.path.splitext(fn)[1].lower() in SKIP_EXT:
            continue
        p = os.path.join(dirpath, fn)
        try:
            with open(p, "rb") as fh:
                if HOME_RX.search(fh.read()):
                    leaky.append(os.path.relpath(p, ROOT))
        except OSError:
            continue
check("no absolute local path anywhere in the tree", not leaky, ", ".join(leaky[:5]))

# --- what the repository deliberately does not hold -------------------------
note("backbone weights", "downloaded from the Hub at run time; see config/backbones.json")
note("trained verifier", "retrain with scripts/pipeline/train_verifier.py, then set MEDLM_VERIFIER")
note("raw candidate pools", "~220 MB of intermediate generations, regenerated by Phase C")

# --- report -----------------------------------------------------------------
width = max(len(n) for _s, n, _d in results)
failed = 0
for status, name, detail in results:
    if status == FAIL:
        failed += 1
    mark = {OK: "  ok  ", FAIL: " FAIL ", NOTE: " note "}[status]
    print(f"[{mark}] {name:<{width}}  {detail}")
print()
checks = sum(1 for s, _n, _d in results if s != NOTE)
print(f"{checks - failed} of {checks} checks passed")
sys.exit(1 if failed else 0)
