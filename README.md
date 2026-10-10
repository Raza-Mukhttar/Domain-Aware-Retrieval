# Domain-Aware Retrieval, Verification and Entity-Aware Decoding in Medical Question Answering

Code, data splits, configuration and results for *Domain-Aware Retrieval,
Verification and Entity-Aware Decoding in Medical Question Answering: A
Controlled Empirical Evaluation*.

**Phase A — offline preparation and retrieval.** Training answers are
summarised, biomedical entities are extracted, each question is assigned a
deterministic type built once from the training split.

**Phase B — prompt construction and adaptation.** The three nearest exemplars
are assembled into a prompt and the generator is fully fine-tuned on those
prompts, with the loss taken on the answer only.

**Phase C — online inference and selection.** Five candidates are generated,
optionally under an entity-aware decoding bonus, then scored by a verifier and by question-entity retention and selected under four rules applied
to the *same* candidate pool.



## Layout

```
config/        the configuration behind every reported number
experiments/   one script per experiment in the paper
preprocessing/ including scripts for data preprocessing
scripts/
  pipeline/    the proposed methodology, phase by phase
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


## Running the whole pipeline

```bash
export MEDLM_VERIFIER=/path/to/verifier

scripts/pipeline/run_pipeline.sh medquad NousResearch/Meta-Llama-3.1-8B-Instruct llama31 42
```

That runs Phase A, Phase B and Phase C in order. The closed-form track is the same three phases on multiple-choice data:

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
