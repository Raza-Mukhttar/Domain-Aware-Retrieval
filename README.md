# Domain-Aware Retrieval, Verification and Entity-Aware Decoding in Medical Question Answering

Code, data splits, configuration and results for *Domain-Aware Retrieval,
Verification and Entity-Aware Decoding in Medical Question Answering: A
Controlled Empirical Evaluation*.

## What the paper does

The paper evaluates a three-phase medical question-answering pipeline under
matched conditions, and reports mostly null results for its own components.

**Phase A — offline preparation and retrieval.** Training answers are
summarised, biomedical entities are extracted, each question is assigned a
deterministic type, and a frozen all-MiniLM-L6-v2 encoder feeds a type-specific
FAISS `IndexFlatIP` index built once from the training split.

**Phase B — prompt construction and adaptation.** The three nearest exemplars
are assembled into a prompt and the generator is fully fine-tuned on those
prompts, with the loss taken on the answer only.

**Phase C — online inference and selection.** Five candidates are generated,
optionally under an entity-aware decoding bonus, then scored by a ClinicalBERT
verifier and by question-entity retention and selected under four rules applied
to the *same* candidate pool.

Everything is evaluated with 10,000-resample paired bootstrap intervals, exact
McNemar tests on identical items for the closed-form benchmarks, Holm correction
across metric families, and a three-seed replication that bounds how much of any
effect is training-seed noise.

## Layout

```
config/        the configuration behind every reported number
data/          the local dataset splits (Hub datasets are not stored here)
experiments/   one script per experiment in the paper
logs/          stdout/stderr of the executed runs
outputs/       every result file: metrics, per-item predictions, figures
scripts/
  pipeline/    the proposed methodology, phase by phase
  analysis/    figure and digest generation from the result files
requirements.txt
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# entity model (not on PyPI)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# NLTK data for METEOR
python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','wordnet','omw-1.4')]"
```

Three environment variables control where things live. All are optional except
`MEDLM_VERIFIER`, which Phase C needs:

| Variable | Meaning | Default |
|---|---|---|
| `MEDLM_VERIFIER` | directory or hub id of the trained ClinicalBERT verifier | none — must be set |
| `MEDLM_MODEL_CACHE` | where backbone weights are cached | Hugging Face default |
| `MEDLM_DATA_DIR` | dataset location | `data/` |
| `MEDLM_OUTPUT_DIR` | where runs write | `outputs/` |

No model weights are stored in this repository. Backbones are pulled from the
Hub at run time; `config/backbones.json` lists the exact identifiers. Train the
verifier once with `scripts/pipeline/train_verifier.py` and point
`MEDLM_VERIFIER` at the result.

## Hardware

The reported runs used three 48 GB GPUs per training lane. Full fine-tuning of
an 8B model with AdamW does not fit on one card: bf16 weights are about 16 GB and
the fp32 optimiser state about four times the parameter bytes again, so a single
lane peaks near 81 GB across its devices. `--max_memory_map` takes a
*per-device* cap in GiB. A uniform cap is wrong on a shared node, because cards
differ in how much they actually have free and a cap above a card's real headroom
only relocates the failure; measure free memory and allow roughly `free / 5.5`
per card.

Inference needs one card. Every training lane deletes its checkpoint as soon as
the evaluation is on disk, so peak disk stays at about one checkpoint rather than
the ~360 GB the full set would need.

## Running the whole pipeline

```bash
export MEDLM_VERIFIER=/path/to/verifier

scripts/pipeline/run_pipeline.sh medquad NousResearch/Meta-Llama-3.1-8B-Instruct llama31 42
```

That runs Phase A, Phase B and Phase C in order and deletes the checkpoint at the
end. The closed-form track is the same three phases on multiple-choice data:

```bash
scripts/pipeline/run_pipeline_closedform.sh medqa NousResearch/Meta-Llama-3.1-8B-Instruct llama31 42
```

Each phase also runs on its own:

```bash
cd scripts/pipeline

python phase_a_offline_preparation.py --dataset medquad --seed 42 --n_train 2000 --n_test 300

python phase_b_prompt_and_adaptation.py --dataset medquad --label llama31 --seed 42 \
    --base NousResearch/Meta-Llama-3.1-8B-Instruct --n_train 500 --n_val 200 --epochs 3

python phase_c_inference_and_selection.py --dataset medquad --generator <checkpoint> \
    --generator_label llama31 --verifier "$MEDLM_VERIFIER" \
    --prompt_sets norag,rag,rag_qtype,rag_fullans --n_cand 5 --also_n1 \
    --n_test 300 --n_train 2000 --seed 42 --truncation_side left
```

Two switches change results and are worth knowing before comparing numbers:

- `--truncation_side left` keeps the end of the prompt, so the target question
  survives the 512-token input cap. `right` reproduces the submitted runs, in
  which the question was dropped from 270 of 300 MedQuAD retrieval prompts and
  all 300 iCliniq ones. Both are reported in the paper; `left` is primary.
- `--first_test_rows` evaluates the first *n* test rows in file order, the
  sampling rule of the submitted runs, instead of a seeded random sample.

## Running one experiment

Every experiment in the paper has its own script, named for what it measures. Scripts read the result files already in `outputs/` where an analysis
only needs the saved per-item predictions, so most of them run on CPU in seconds
and need no GPU.

### Exemplar contamination and failure modes (Section 4.1)

| Script | What it produces |
|---|---|
| `partition_overlap_counts.py` | exact and near-duplicate overlap between partitions |
| `duplicate_free_subset_effect.py` | the retrieval effect after removing test questions that duplicate a training question |
| `icliniq_corpus_composition.py` | per-platform composition of the iCliniq-style corpus |
| `retrieval_effect_diagnosis.py` | diagnosis of the lower retrieval score: prompt truncation, not retrieval |

### Repeated-seed stability and component attribution (Section 4.2)

| Script | What it produces |
|---|---|
| `run_seed_variability.sh` | trains and evaluates the extra seeds (needs GPUs) |
| `closedform_seed_variability.py` | closed-form spread across training seeds |
| `cumulative_ablation_ladder.py` | the cumulative ladder and each rung's own contribution |
| `decoding_nondeterminism.py` | how much of the residual difference is bf16 beam-search noise |
| `aggregate_openended_results.py` | aggregates every run into one summary file |

```bash
# three devices per lane, one lane per backbone
experiments/run_seed_variability.sh 0,1,2 open qwen3:medquad:1 qwen3:medquad:2
experiments/run_seed_variability.sh 3,4,5 closed llama31:medqa:1 llama31:medqa:2
python experiments/cumulative_ablation_ladder.py
```

### Partition overlap and group-disjoint controls (Section 4.3)

| Script | What it produces |
|---|---|
| `recover_medquad_source_metadata.py` | recovers source document and focus entity per question |
| `group_split_feasibility.py` | whether a group-disjoint repartition is viable at all |
| `build_group_disjoint_split.py` | builds the repartition (already shipped in `data/`) |
| `run_group_disjoint_training.sh` | trains and evaluates on it at three seeds (needs GPUs) |
| `group_disjoint_retrieval_effect.py` | the retrieval effect on the disjoint split |
| `source_disjoint_retrieval.py` | retrieval with exemplars sharing the query's source or focus barred |

```bash
experiments/run_group_disjoint_training.sh 4,5,6 8,8,8 42 1 2
python experiments/group_disjoint_retrieval_effect.py
```

### Retrieval precision and recall (Section 4.4)

```bash
python experiments/retrieval_precision_recall.py
```

Precision and recall under a relevance criterion that does not reuse the
retrieval score itself, so the measurement is not circular.

### Verifier uncertainty and selection quality (Section 4.5)

| Script | What it produces |
|---|---|
| `verifier_discrimination.py` | ROC-AUC and PR-AUC with bootstrap intervals |
| `verifier_transfer_and_retraining.py` | whether training the verifier per benchmark recovers the closed-form loss |

### Robustness checks and historical comparisons (Section 4.6)

| Script | What it produces |
|---|---|
| `hyperparameter_sweeps_k_and_gamma.py` | sweeps over exemplar count *k* and entity-bonus strength γ |
| `additional_robustness_analyses.py` | the remaining robustness analyses |
| `closedform_failure_diagnostics.py` | why the zero-shot backbone beats the framework in most closed-form cells |
| `test_sampling_equivalence.py` | effect of the test-sampling rule |
| `topic_overlap_and_dedup_sensitivity.py` | topic-overlap and deduplicated-subset analysis on the corrected prompts |
| `reported_value_consistency_checks.py` | consistency checks over the reported numbers |
| `summarisation_qualifier_example.py` | a real case of a clinically weighty qualifier dropped by the summariser |

### Retained from the submitted version

`experiments/legacy_submitted_version/` holds the scripts behind the four
comparisons the paper retains in Appendix F because they could not be re-run
under the revised protocol. They are kept for inspection and are not part of the
reproduction path; see the README in that directory.

## Rebuilding the figures and the results digest

First, a self-check that the repository holds everything the numbers rest on:

```bash
python scripts/analysis/verify_repository.py
```

It checks the splits, the derived artefacts, the configuration, every run's
metrics, provenance and per-item predictions, the training manifests and loss
histories, the cross-run analyses, the figures, and that no absolute local path
survives anywhere. It needs no GPU and no network, and exits non-zero on failure.

```bash
python scripts/analysis/make_table_figures.py                # the figures in the paper
python scripts/analysis/make_table_figures_full_coverage.py  # full-coverage variants
python scripts/analysis/verify_figure_values.py              # audits every drawn value against its table
python scripts/analysis/results_digest.py                    # one-line headline per experiment
```

`verify_figure_values.py` re-reads each figure's source table and checks that
every number the figure draws appears there.

The figure scripts parse their values out of the manuscript source in
`outputs/manuscript/`, so a figure and its table cannot drift apart. A figure
whose source table has since been renamed or folded into another table is skipped
with a note naming the missing label, rather than aborting the run. Set
`MEDLM_MANUSCRIPT` to point at a different manuscript file.

## What is in `outputs/`

| Path | Contents |
|---|---|
| `outputs/runs/<run>/` | one directory per evaluation run (107 of them): `manifest.json`, `metrics.json` and the per-item `preds_*.jsonl` |
| `outputs/analysis/` | the cross-run analyses each experiment script writes |
| `outputs/analysis/training_manifests/` | each training run's configuration and step-wise loss history, extracted before its checkpoint was deleted, so training convergence stays reproducible without shipping weights |
| `outputs/figures/` | every figure in the paper, as PDF and PNG |
| `outputs/manuscript/` | the manuscript source, from which the figure scripts read their table values |
| `outputs/Experiments_and_Results.md` | every experiment with its result tables |
| `outputs/Experiments_EXP1-EXP6.md` | the six specified experiments in detail |

The per-item prediction files carry each example's own metric values, so the
paired bootstrap and the McNemar tests can be recomputed without regenerating any
text.

Each run directory carries a `manifest.json` with the configuration, the seed,
the split row ids and the library versions, and every metrics entry carries a
`run_id` derived from its configuration, so no reported number exists without the
configuration that produced it.

The raw N=5 candidate pools are not included — about 220 MB of intermediate
generations. They are regenerated by Phase C, which writes
`candidates_<prompt_set>_n<N>.jsonl` alongside the predictions.

## Reproducibility notes

- **Determinism.** Beam search in bf16 is not bit-reproducible across runs. Only
  about 10% of answers are identical between two runs of the same checkpoint on
  prompts that sit well below the input cap, so small differences are expected
  and the paper reports a seed-to-seed floor rather than treating any difference
  as an effect.
- **Interpretation rule.** A difference counts as an effect only when its
  interval excludes zero *and* it exceeds the seed-to-seed standard deviation.
  Sub-point differences are reported as no difference.
- **Tokenizers.** Under transformers 5.x, `DeepSeek-R1-Distill-Llama-8B` resolves
  to the legacy slow tokenizer despite its config requesting the fast one, and
  that path drops every space on decode. `medlm/tokenizer_util.py` round-trips a
  probe string at load and rebuilds the tokenizer from `tokenizer.json` when the
  spaces are lost. Keep it — without it that backbone's open-ended scores are
  meaningless.
- **Entity extraction.** scispaCy is used when importable; otherwise the stored
  BC5CDR vocabulary is phrase-matched. Which path was taken is recorded in each
  run manifest.
