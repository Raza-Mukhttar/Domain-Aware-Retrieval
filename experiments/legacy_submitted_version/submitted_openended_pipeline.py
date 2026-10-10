# file: medlm_llm_pipeline.py
"""
Full MedLM-LLaMA3 Pipeline — Extended Multi-Dataset Evaluation
===============================================================
Original pipeline (MedQuAD / iCliniq) is 100% unchanged.
Added support for the following benchmark datasets:

  OPEN-ENDED / LONG-FORM (ROUGE + METEOR + BERTScore + BLEU)
  ├── medquad     Local JSONL, 47K NIH Q-A pairs         [original]
  └── icliniq     Local JSONL, patient-expert pairs      [original]

  MULTIPLE-CHOICE / MCQ (Accuracy + ROUGE + METEOR)
  ├── medqa       HuggingFace: bigbio/med_qa              US USMLE 4-choice
  ├── medmcqa     HuggingFace: openlifescienceai/medmcqa  4-choice
  ├── mmlu_med    HuggingFace: cais/mmlu  (med subjects)  4-choice
  └── pubmedqa    HuggingFace: bigbio/pubmed_qa           Yes/No/Maybe 3-choice

Usage
-----
  # Original datasets (unchanged behaviour)
  python medlm_llm_pipeline.py --dataset medquad
  python medlm_llm_pipeline.py --dataset icliniq

  # New benchmark datasets (eval only — no fine-tuning needed)
  python medlm_llm_pipeline.py --dataset medqa
  python medlm_llm_pipeline.py --dataset medmcqa
  python medlm_llm_pipeline.py --dataset mmlu_med
  python medlm_llm_pipeline.py --dataset pubmedqa

  # Custom subset sizes
  python medlm_llm_pipeline.py --dataset medqa \
      --train_samples 500 --val_samples 100 --test_samples 500

  # Use pre-trained checkpoint (skip training, just evaluate on new dataset)
  python medlm_llm_pipeline.py --dataset medqa \
      --eval_only \
      --generator_ckpt SET_THIS_PATH \
      --verifier_ckpt  SET_THIS_PATH

  # Run ALL datasets sequentially
  python medlm_llm_pipeline.py --run_all_datasets

Notes
-----
- MCQ datasets use accuracy as the primary metric in addition to
  text-generation metrics (ROUGE / METEOR / BERTScore) computed on
  the generated answer vs. the gold answer text.
- For MCQ datasets the pipeline generates a free-text answer and ALSO
  extracts the predicted option letter (A/B/C/D or Yes/No/Maybe) by
  searching the generated text — this gives both accuracy and generation
  quality in a single forward pass.
- HuggingFace datasets are downloaded automatically on first run and
  cached locally in ~/.cache/huggingface/datasets.
"""

import os
import json
import random
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from evaluate import load as load_metric

try:
    from scipy.stats import ttest_rel
except Exception:  # scipy is optional; bootstrap CIs still work without it
    ttest_rel = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import spacy
except Exception:
    raise RuntimeError(
        "Install scispacy: pip install scispacy spacy\n"
        "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
        "releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz"
    )


# =============================================================================
# DATASET REGISTRY
# =============================================================================
# Each entry describes one supported dataset.
# Keys
# ----
# source        "local" | "huggingface"
# task          "open_ended" | "mcq"
# hf_id         HuggingFace dataset ID (source=huggingface only)
# hf_config     HuggingFace config name (optional)
# hf_split_map  mapping of canonical splits to HF split names
# answer_col    column name of the answer / gold text
# options_col   column name of the options list (MCQ only)
# label_col     column name of the correct option index/letter (MCQ only)
# label_type    "index" (int 0-3) | "letter" (A/B/C/D) | "text" (Yes/No/Maybe)
# description   human-readable name

DATASET_REGISTRY: Dict[str, Dict] = {

    # ── Original local datasets ───────────────────────────────────────────
    "medquad": {
        "source":       "local",
        "task":         "open_ended",
        "description":  "MedQuAD — NIH medical Q-A (47K pairs)",
        "metrics":      ["ROUGE1","ROUGE2","ROUGEL","METEOR","BERTScore","BLEU1","BLEU4",
                         "FactualAccuracy","HallucinationRate"],
    },
    "icliniq": {
        "source":       "local",
        "task":         "open_ended",
        "description":  "iCliniq — patient-expert Q-A",
        "metrics":      ["ROUGE1","ROUGE2","ROUGEL","METEOR","BERTScore","BLEU1","BLEU4",
                         "FactualAccuracy","HallucinationRate"],
    },

    # ── HuggingFace MCQ datasets ──────────────────────────────────────────
    # "medqa": {
    #     "source":       "huggingface",
    #     "task":         "mcq",
    #     "hf_id":        "bigbio/med_qa",
    #     "hf_config":    "med_qa_en_bigbio_qa",
    #     "hf_split_map": {"train": "train", "validation": "validation", "test": "test"},
    #     "question_col": "question",
    #     "answer_col":   "answer",          # gold free-text answer
    #     "options_col":  "choices",         # list of option dicts {"text":..., "label":...}
    #     "label_col":    "answer",          # the correct option text is in "answer"
    #     "label_type":   "text_match",      # match generated text against options
    #     "description":  "MedQA-USMLE — 4-choice US medical licensing exam",
    #     "metrics":      ["Accuracy","ROUGE1","METEOR","BERTScore"],
    #     "num_choices":  4,
    # },

    "medqa": {
    "source":       "huggingface",
    "task":         "mcq",
    "hf_id":        "GBaker/MedQA-USMLE-4-options",
    "hf_split_map": {"train": "train", "validation": "validation", "test": "test"},
    "question_col": "question",
    "options_col":  "options",
    "label_col":    "answer_idx",
    "label_type":   "index",
    "description":  "MedQA-USMLE — 4-choice US medical licensing exam",
    "metrics":      ["Accuracy","ROUGE1","METEOR","BERTScore"],
    "num_choices":  4,
    },
    "medmcqa": {
        "source":       "huggingface",
        "task":         "mcq",
        "hf_id":        "openlifescienceai/medmcqa",
        "hf_config":    None,
        "hf_split_map": {"train": "train", "validation": "validation", "test": "test"},
        "question_col": "question",
        "answer_col":   "exp",             # explanation / rationale text
        "options_col":  None,              # opa, opb, opc, opd columns
        "label_col":    "cop",             # 0=A, 1=B, 2=C, 3=D
        "label_type":   "index",
        "description":  "MedMCQA — 4-choice Indian medical PG entrance exam",
        "metrics":      ["Accuracy","ROUGE1","METEOR","BERTScore"],
        "num_choices":  4,
    },
    "mmlu_med": {
        "source":       "huggingface",
        "task":         "mcq",
        "hf_id":        "cais/mmlu",
        "hf_config":    None,              # loaded per subject
        "hf_split_map": {"train": "auxiliary_train", "validation": "validation",
                         "test": "test"},
        "question_col": "question",
        "answer_col":   "answer",          # int 0-3
        "options_col":  "choices",         # list of 4 strings
        "label_col":    "answer",          # int 0-3
        "label_type":   "index",
        "description":  "MMLU Medical — 6 medical subjects from MMLU",
        "metrics":      ["Accuracy","ROUGE1","METEOR","BERTScore"],
        "num_choices":  4,
        "mmlu_subjects": [
            "clinical_knowledge", "medical_genetics",
            "anatomy", "professional_medicine",
            "college_biology", "college_medicine",
        ],
    },


    "pubmedqa": {
    "source":       "huggingface",
    "task":         "mcq",
    "hf_id":        "reginaboateng/cleaned_pubmedqa",
    "hf_config":    None,  # no config needed
    "hf_split_map": {"train": "train", "validation": "validation", "test": "test"},
    "question_col": "question",
    "answer_col":   "final_decision",   # updated
    "options_col":  None,               # will construct manually
    "label_col":    "final_decision",
    "label_type":   "text_match",
    "description":  "Cleaned PubMedQA — Yes/No/Maybe QA",
    "metrics":      ["Accuracy","ROUGE1","METEOR","BERTScore"],
    "num_choices":  3,
    },
    # "pubmedqa": {
    #     "source":       "huggingface",
    #     "task":         "mcq",
    #     "hf_id":        "bigbio/pubmed_qa",
    #     "hf_config":    "pubmed_qa_labeled_fold0_bigbio_qa",
    #     "hf_split_map": {"train": "train", "validation": "validation", "test": "test"},
    #     "question_col": "question",
    #     "answer_col":   "answer",          # list: [answer_text]
    #     "options_col":  "choices",         # ["yes","no","maybe"]
    #     "label_col":    "answer",
    #     "label_type":   "text_match",
    #     "description":  "PubMedQA — biomedical Yes/No/Maybe QA",
    #     "metrics":      ["Accuracy","ROUGE1","METEOR","BERTScore"],
    #     "num_choices":  3,
    # },
}

# MMLU medical subject list (reused in loader)
MMLU_MED_SUBJECTS = DATASET_REGISTRY["mmlu_med"]["mmlu_subjects"]


# =============================================================================
# CONFIG  (original fields preserved; new fields added at bottom)
# =============================================================================

@dataclass
class Config:
    # ── Original fields ──────────────────────────────────────────────────
    dataset_name:         str   = "medquad"
    medquad_dir:          str   = "SET_THIS_PATH"
    icliniq_dir:          str   = "SET_THIS_PATH"
    out_dir:              str   = "SET_THIS_PATH"
    summarizer_model:     str   = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    generator_model:      str   = "Qwen/Qwen3-8B"
    verifier_model:       str   = "medicalai/ClinicalBERT"
    embedder_model:       str   = "SET_THIS_PATH"
    per_type_index_dir:   str   = "SET_THIS_PATH"
    max_summary_tokens:   int   = 200
    max_input_tokens:     int   = 512
    n_exemplars:          int   = 3
    n_best:               int   = 5
    device:               str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed:                 int   = 42
    batch_size:           int   = 4
    epochs:               int   = 3
    lr:                   float = 3e-5
    contrastive_epochs:   int   = 2
    contrastive_batch_size: int = 8
    contrastive_temp:     float = 0.07
    verifier_threshold:   float = 0.5
    use_subset:           bool  = False
    subset_train_size:    int   = 500
    subset_val_size:      int   = 200
    subset_test_size:     int   = 300

    # ── Checkpoints / sweeps ──────────────────────────────────────────────
    eval_only:            bool  = False
    generator_ckpt:       str   = ""
    verifier_ckpt:        str   = ""
    run_all_datasets:     bool  = False
    hf_cache_dir:         Optional[str] = None

    # ── Multi-GPU / training control ─────────────────────────────────────
    gradient_accumulation_steps: int = 4
    eval_steps:           int   = 100
    logging_steps:        int   = 20
    num_workers:          int   = 2
    ddp_find_unused_parameters: bool = False
    gradient_checkpointing: bool = True
    trust_remote_code:    bool  = True
    use_flash_attention_2: bool = False

    # Optional PEFT/QLoRA support. Disabled by default to preserve full FT.
    use_lora:             bool  = False
    load_in_4bit:         bool  = False
    lora_r:               int   = 16
    lora_alpha:           int   = 32
    lora_dropout:         float = 0.05
    lora_target_modules:  str   = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # ── MCQ / PubMedQA improvements ──────────────────────────────────────
    mcq_decoding:         str   = "logprob"  # logprob | generate | hybrid
    mcq_max_new_tokens:   int   = 16
    train_verifier_for_mcq: bool = True
    skip_contrastive_for_mcq: bool = True

    # ── Statistical reporting ────────────────────────────────────────────
    bootstrap_samples:    int   = 1000
    ci_level:             float = 95.0

# =============================================================================
# CLI  (extends original with new flags)
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MedLM pipeline — multi-dataset evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="medquad",
                   choices=list(DATASET_REGISTRY.keys()), help="Dataset to use.")
    p.add_argument("--train_samples", type=int, default=500,
                   help="Training samples (-1 = full split).")
    p.add_argument("--val_samples", type=int, default=200,
                   help="Validation samples (-1 = full split).")
    p.add_argument("--test_samples", type=int, default=300,
                   help="Test samples (-1 = full split).")

    # Paths and models — useful when fine-tuning different LLMs.
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--medquad_dir", type=str, default=None)
    p.add_argument("--icliniq_dir", type=str, default=None)
    p.add_argument("--generator_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--summarizer_model", type=str, default=None)
    p.add_argument("--verifier_model", type=str, default=None)
    p.add_argument("--embedder_model", type=str, default=None)
    p.add_argument("--per_type_index_dir", type=str, default=None)

    # Checkpoint/eval flags.
    p.add_argument("--eval_only", action="store_true", default=False,
                   help="Skip training; load generator/verifier from checkpoints if provided/found.")
    p.add_argument("--generator_ckpt", type=str, default="",
                   help="Generator checkpoint path for --eval_only.")
    p.add_argument("--verifier_ckpt", type=str, default="",
                   help="Verifier checkpoint path for --eval_only.")
    p.add_argument("--run_all_datasets", action="store_true", default=False,
                   help="Run evaluation sequentially over ALL datasets.")
    p.add_argument("--hf_cache_dir", type=str, default=None,
                   help="Custom HuggingFace dataset cache directory.")

    # Training controls.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Per-device batch size. With 5 GPUs, effective batch = batch_size*5*grad_accum.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_input_tokens", type=int, default=None)
    p.add_argument("--max_summary_tokens", type=int, default=None)
    p.add_argument("--n_best", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--no_gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--use_flash_attention_2", action="store_true", default=False)

    # Optional LoRA/QLoRA.
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    p.add_argument("--lora_target_modules", type=str, default=None)

    # MCQ/PubMedQA controls.
    p.add_argument("--mcq_decoding", type=str, default=None,
                   choices=["logprob", "generate", "hybrid"],
                   help="MCQ decision method. logprob is recommended for PubMedQA.")
    p.add_argument("--mcq_max_new_tokens", type=int, default=None)
    p.add_argument("--train_verifier_for_mcq", action="store_true", default=False)
    p.add_argument("--run_contrastive_for_mcq", action="store_true", default=False)

    # Statistical reporting.
    p.add_argument("--bootstrap_samples", type=int, default=None)
    p.add_argument("--ci_level", type=float, default=None)
    return p.parse_args()

# =============================================================================
# ORIGINAL UTILITIES (UNCHANGED)
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


# =============================================================================
# ORIGINAL LOCAL DATASET LOADER (UNCHANGED)
# =============================================================================



def is_distributed() -> bool:
    """True when launched with torchrun/accelerate using more than one process."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_rank_zero() -> bool:
    return get_rank() == 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def first_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_tokenizer_defaults(tok, left_truncate: bool = False):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left" if left_truncate else "right"
    return tok


def truncate_token_ids(ids: List[int], max_len: int, keep: str = "right") -> List[int]:
    if max_len <= 0 or len(ids) <= max_len:
        return ids
    if keep == "left":
        return ids[:max_len]
    # keep the right/suffix of the prompt; this preserves options + "Answer:" for MCQs
    return ids[-max_len:]


def model_load_kwargs(cfg: Config, for_training: bool = True) -> Dict:
    """Centralised model loading kwargs for single-GPU, model-parallel, and DDP runs."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype, "trust_remote_code": cfg.trust_remote_code}
    if cfg.use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"

    # DDP/Accelerate: each process owns one GPU, so do NOT shard with device_map="auto".
    if is_distributed() and for_training and not cfg.load_in_4bit:
        return kwargs

    if cfg.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": local_rank()} if is_distributed() else "auto"
        except Exception as e:
            raise RuntimeError(
                "--load_in_4bit requires bitsandbytes and a compatible transformers install."
            ) from e
    elif torch.cuda.is_available() and not is_distributed():
        # Single process: allow HF to shard a large model across visible GPUs.
        kwargs["device_map"] = "auto"
    return kwargs


def maybe_apply_lora(model, cfg: Config):
    if not cfg.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise RuntimeError(
            "--use_lora requires peft. Install with: pip install peft"
        ) from e

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [x.strip() for x in cfg.lora_target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def make_training_arguments(**kwargs):
    """Compatibility wrapper for Transformers versions using eval_strategy vs evaluation_strategy."""
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    # Drop None-valued args that older versions may not accept.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return TrainingArguments(**kwargs)

def load_dataset_from_dir(data_dir: str):
    splits = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(p):
            recs    = read_jsonl(p)
            cleaned = []
            for r in recs:
                q  = r.get("question", "").strip()
                a  = r.get("answer",   "").strip()
                if q and a:
                    qt = r.get("qtype", r.get("type", "general"))
                    cleaned.append({"question": q, "answer": a, "qtype": qt})
            splits[split] = Dataset.from_list(cleaned)
    return DatasetDict(splits)


def load_local_datasets(cfg: Config) -> DatasetDict:
    """Original local loader for medquad / icliniq (UNCHANGED logic)."""
    if cfg.dataset_name.lower() == "medquad":
        data = load_dataset_from_dir(cfg.medquad_dir)
    elif cfg.dataset_name.lower() == "icliniq":
        data = load_dataset_from_dir(cfg.icliniq_dir)
    else:
        raise ValueError("dataset_name must be 'medquad' or 'icliniq'")

    merged = {}
    for s in ["train", "validation", "test"]:
        if s in data:
            merged[s] = data[s]
            if cfg.use_subset:
                full  = len(merged[s])
                subsz = {
                    "train":      min(cfg.subset_train_size, full),
                    "validation": min(cfg.subset_val_size,   full),
                    "test":       min(cfg.subset_test_size,  full),
                }[s]
                merged[s] = merged[s].select(range(subsz))
                print(f"  [{s}] Using {subsz}/{full} samples")
    print("Dataset sizes:", {k: len(v) for k, v in merged.items()})
    return DatasetDict(merged)


# =============================================================================
# NEW HuggingFace DATASET LOADERS
# =============================================================================

def _apply_subset(ds: Dataset, max_n: int) -> Dataset:
    if max_n > 0:
        n = min(max_n, len(ds))
        return ds.select(range(n))
    return ds


def _norm(text) -> str:
    """Normalise any value to a clean string."""
    if text is None:
        return ""
    if isinstance(text, list):
        return " ".join(str(t) for t in text).strip()
    return str(text).strip()


# ── MedQA ────────────────────────────────────────────────────────────────────
def _parse_medqa_row(row: Dict) -> Dict:
    question = _norm(row.get("question", ""))

    raw_options = row.get("options", {})
    raw_answer  = row.get("answer_idx", 0)

    # -----------------------------
    # 1. Normalize options → list
    # -----------------------------
    if isinstance(raw_options, dict):
        # Sort keys like A,B,C,D safely
        ordered_keys = sorted(raw_options.keys())
        options = [str(raw_options[k]).strip() for k in ordered_keys]
    elif isinstance(raw_options, list):
        options = [str(x).strip() for x in raw_options]
    else:
        options = []

    # -----------------------------
    # 2. Normalize answer index
    # -----------------------------
    answer_idx = 0  # default fallback

    if isinstance(raw_answer, int):
        answer_idx = raw_answer

    elif isinstance(raw_answer, str):
        raw_answer = raw_answer.strip()

        # Case 1: numeric string ("0", "1", ...)
        if raw_answer.isdigit():
            answer_idx = int(raw_answer)

        # Case 2: letter ("A", "B", ...)
        elif len(raw_answer) == 1 and raw_answer.isalpha():
            answer_idx = ord(raw_answer.upper()) - 65

        # Case 3: sometimes answer text itself is given
        else:
            if raw_answer in options:
                answer_idx = options.index(raw_answer)

    # -----------------------------
    # 3. Safety checks
    # -----------------------------
    if not options:
        answer_idx = 0
    elif answer_idx < 0 or answer_idx >= len(options):
        answer_idx = 0

    gold_letter = chr(65 + answer_idx) if options else ""

    opt_str = "\n".join(
        f"{chr(65+i)}. {c}" for i, c in enumerate(options)
    )

    return {
        "question":     question,
        "answer":       options[answer_idx] if options else "",
        "gold_letter":  gold_letter,
        "options_text": opt_str,
        "options_list": options,
        "qtype":        "general",
    }
# def _parse_medqa_row(row: Dict) -> Dict:
#     """
#     bigbio/med_qa row fields:
#       question: str
#       choices:  list of {"text": str, "label": "A"/"B"/"C"/"D"}
#       answer:   the correct choice text (not label letter)
#     """
#     question = _norm(row.get("question", ""))
#     choices  = row.get("choices", [])
#     options  = [c["text"] for c in choices] if choices else []
#     labels   = [c.get("label", chr(65 + i)) for i, c in enumerate(choices)]
#     gold_txt = _norm(row.get("answer", ""))
#     # Find matching label letter
#     gold_letter = "A"
#     for c in choices:
#         if _norm(c["text"]).lower() == gold_txt.lower():
#             gold_letter = c.get("label", "A")
#             break
#     # Build option display string for prompt
#     opt_str = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, options))
#     return {
#         "question":     question,
#         "answer":       gold_txt,
#         "gold_letter":  gold_letter,
#         "options_text": opt_str,
#         "options_list": options,
#         "qtype":        "general",
#     }


def load_medqa(cfg: Config) -> DatasetDict:
    entry = DATASET_REGISTRY["medqa"]
    print(f"  Downloading {entry['description']} from HuggingFace...")

    # safe loader (no hf_config)
    raw = hf_load_dataset(
        entry["hf_id"],
        cache_dir=cfg.hf_cache_dir,
        trust_remote_code=cfg.trust_remote_code
    )

    splits = {}

    # -------------------------
    # 1. TRAIN → split into train + validation (90:10)
    # -------------------------
    train_rows = [_parse_medqa_row(r) for r in raw["train"]]
    train_ds   = Dataset.from_list(train_rows)

    split_ds = train_ds.train_test_split(
        test_size=0.1,
        seed=42,
        shuffle=True
    )

    splits["train"]      = split_ds["train"]
    splits["validation"] = split_ds["test"]

    # -------------------------
    # 2. TEST → keep original
    # -------------------------
    if "test" in raw:
        test_rows = [_parse_medqa_row(r) for r in raw["test"]]
        splits["test"] = Dataset.from_list(test_rows)

    # -------------------------
    # 3. Apply subset
    # -------------------------
    size_map = {
        "train": cfg.subset_train_size,
        "validation": cfg.subset_val_size,
        "test": cfg.subset_test_size,
    }

    for split_name in splits:
        if cfg.use_subset:
            splits[split_name] = _apply_subset(
                splits[split_name],
                size_map[split_name]
            )

        print(f"  [{split_name}] {len(splits[split_name])} samples")

    return DatasetDict(splits)


# ── MedMCQA ──────────────────────────────────────────────────────────────────

def _parse_medmcqa_row(row: Dict) -> Dict:
    """
    openlifescienceai/medmcqa fields:
      question: str
      opa, opb, opc, opd: option strings
      cop: int 0-3  (correct option index)
      exp: explanation text
    """
    options = [
        _norm(row.get("opa", "")),
        _norm(row.get("opb", "")),
        _norm(row.get("opc", "")),
        _norm(row.get("opd", "")),
    ]
    cop = int(row.get("cop", 0))
    cop = max(0, min(3, cop))
    labels   = ["A", "B", "C", "D"]
    opt_str  = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, options))
    return {
        "question":     _norm(row.get("question", "")),
        "answer":       options[cop],             # gold answer text
        "gold_letter":  labels[cop],              # A/B/C/D
        "gold_index":   cop,
        "options_text": opt_str,
        "options_list": options,
        "explanation":  _norm(row.get("exp", "")),
        "qtype":        _norm(row.get("subject_name", "general")),
    }


def load_medmcqa(cfg: Config) -> DatasetDict:
    entry = DATASET_REGISTRY["medmcqa"]
    print(f"  Downloading {entry['description']} from HuggingFace...")
    raw = hf_load_dataset(
        entry["hf_id"], cache_dir=cfg.hf_cache_dir, trust_remote_code=True
    )
    size_map = {
        "train": cfg.subset_train_size,
        "validation": cfg.subset_val_size,
        "test": cfg.subset_test_size,
    }
    splits = {}
    for canon in ["train", "validation", "test"]:
        hf_split = entry["hf_split_map"].get(canon, canon)
        if hf_split not in raw:
            continue
        rows = [_parse_medmcqa_row(r) for r in raw[hf_split]]
        ds   = Dataset.from_list(rows)
        if cfg.use_subset:
            ds = _apply_subset(ds, size_map[canon])
        splits[canon] = ds
        print(f"  [{canon}] {len(ds)} samples")
    return DatasetDict(splits)


# ── MMLU Medical ─────────────────────────────────────────────────────────────

def _parse_mmlu_row(row: Dict, subject: str) -> Dict:
    """
    cais/mmlu fields:
      question: str
      choices:  list of 4 strings
      answer:   int 0-3
    """
    options    = list(row.get("choices", []))
    gold_idx   = int(row.get("answer", 0))
    gold_idx   = max(0, min(3, gold_idx))
    labels     = ["A", "B", "C", "D"]
    opt_str    = "\n".join(
        f"{lbl}. {txt}" for lbl, txt in zip(labels, options)
    )
    return {
        "question":     _norm(row.get("question", "")),
        "answer":       options[gold_idx] if options else "",
        "gold_letter":  labels[gold_idx],
        "gold_index":   gold_idx,
        "options_text": opt_str,
        "options_list": options,
        "qtype":        subject,
    }


def load_mmlu_med(cfg: Config) -> DatasetDict:
    entry    = DATASET_REGISTRY["mmlu_med"]
    subjects = entry["mmlu_subjects"]
    print(f"  Downloading {entry['description']} ({len(subjects)} subjects)...")

    all_rows: Dict[str, List] = {"train": [], "validation": [], "test": []}

    for subj in subjects:
        try:
            raw = hf_load_dataset(
                entry["hf_id"], subj,
                cache_dir=cfg.hf_cache_dir, trust_remote_code=True
            )
        except Exception as e:
            print(f"  WARNING: could not load subject '{subj}': {e}")
            continue

        # MMLU has no train → build it manually
        dev_rows = [_parse_mmlu_row(r, subj) for r in raw.get("dev", [])]
        val_rows = [_parse_mmlu_row(r, subj) for r in raw.get("validation", [])]
        test_rows = [_parse_mmlu_row(r, subj) for r in raw.get("test", [])]

        # Build splits
        all_rows["train"].extend(dev_rows + val_rows)
        all_rows["validation"].extend(val_rows)
        all_rows["test"].extend(test_rows)

    size_map = {
        "train": cfg.subset_train_size,
        "validation": cfg.subset_val_size,
        "test": cfg.subset_test_size,
    }
    splits = {}
    for canon, rows in all_rows.items():
        if not rows:
            continue
        random.shuffle(rows)   # mix subjects
        ds = Dataset.from_list(rows)
        if cfg.use_subset:
            ds = _apply_subset(ds, size_map[canon])
        splits[canon] = ds
        print(f"  [{canon}] {len(ds)} samples across {len(subjects)} subjects")
    return DatasetDict(splits)


# ── PubMedQA ─────────────────────────────────────────────────────────────────

# def _parse_pubmedqa_row(row: Dict) -> Dict:
#     """
#     bigbio/pubmed_qa row fields (bigbio_qa schema):
#       question:  str
#       choices:   list of strings (["yes","no","maybe"])
#       answer:    list with one string (the correct choice)
#     """
#     question = _norm(row.get("question", ""))
#     choices  = row.get("choices", ["yes", "no", "maybe"])
#     answer_l = row.get("answer", ["yes"])
#     gold_txt = _norm(answer_l[0] if isinstance(answer_l, list) else answer_l)
#     # map yes/no/maybe to A/B/C
#     ymap     = {"yes": "A", "no": "B", "maybe": "C"}
#     gold_letter = ymap.get(gold_txt.lower(), "A")
#     opt_str  = "\n".join(
#         f"{chr(65+i)}. {c}" for i, c in enumerate(choices)
#     )
#     # context may be available as "context" in some configs
#     context  = _norm(row.get("context", ""))
#     combined_q = f"{question}\n\nContext: {context}" if context else question
#     return {
#         "question":     combined_q,
#         "answer":       gold_txt,
#         "gold_letter":  gold_letter,
#         "options_text": opt_str,
#         "options_list": list(choices),
#         "qtype":        "general",
#     }


def _extract_pubmedqa_context(row: Dict) -> str:
    """Robust context extraction for cleaned_pubmedqa and BigBio-style rows."""
    for key in ["context", "contexts", "abstract", "long_answer", "context_text"]:
        if key not in row or row.get(key) is None:
            continue
        val = row.get(key)
        if isinstance(val, dict):
            if "contexts" in val:
                return _norm(val.get("contexts", ""))
            if "context" in val:
                return _norm(val.get("context", ""))
            return " ".join(_norm(v) for v in val.values() if _norm(v)).strip()
        return _norm(val)
    return ""


def _extract_pubmedqa_answer(row: Dict) -> str:
    for key in ["final_decision", "answer", "label", "decision"]:
        if key not in row or row.get(key) is None:
            continue
        val = row.get(key)
        if isinstance(val, list) and val:
            val = val[0]
        ans = _norm(val).lower().strip(" .;:[]()")
        if ans in {"yes", "no", "maybe"}:
            return ans
        # Some rows contain a sentence; scan for an explicit decision word.
        for y in ["yes", "no", "maybe"]:
            if re.search(r"\b" + y + r"\b", ans):
                return y
    return "yes"


def _parse_pubmedqa_row(row: Dict) -> Dict:
    question = _norm(row.get("question", ""))
    context  = _extract_pubmedqa_context(row)
    gold_txt = _extract_pubmedqa_answer(row)

    choices = ["yes", "no", "maybe"]
    ymap = {"yes": "A", "no": "B", "maybe": "C"}
    gold_letter = ymap.get(gold_txt.lower(), "A")

    opt_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

    # Put context first and the question immediately before options. With left-side
    # prompt truncation this preserves the question + options + Answer suffix.
    combined_q = (
        f"Context:\n{context}\n\nQuestion:\n{question}"
        if context else question
    )

    return {
        "question":     combined_q,
        "answer":       gold_txt,
        "gold_letter":  gold_letter,
        "gold_index":   choices.index(gold_txt) if gold_txt in choices else 0,
        "options_text": opt_str,
        "options_list": choices,
        "qtype":        "pubmedqa",
    }

def load_pubmedqa(cfg: Config) -> DatasetDict:
    entry = DATASET_REGISTRY["pubmedqa"]
    print(f"  Downloading {entry['description']} from HuggingFace...")

    raw = hf_load_dataset(
        entry["hf_id"],
        cache_dir=cfg.hf_cache_dir,
        trust_remote_code=cfg.trust_remote_code,
    )

    splits: Dict[str, Dataset] = {}

    if all(s in raw for s in ["train", "validation", "test"]):
        for canon in ["train", "validation", "test"]:
            splits[canon] = Dataset.from_list([_parse_pubmedqa_row(r) for r in raw[canon]])
    elif "train" in raw:
        rows = [_parse_pubmedqa_row(r) for r in raw["train"]]
        base = Dataset.from_list(rows)
        # Robust fallback: train/validation/test = 80/10/10 when the cleaned
        # dataset exposes only a train split. This avoids validating on train.
        split_1 = base.train_test_split(test_size=0.20, seed=cfg.seed, shuffle=True)
        split_2 = split_1["test"].train_test_split(test_size=0.50, seed=cfg.seed, shuffle=True)
        splits["train"] = split_1["train"]
        splits["validation"] = split_2["train"]
        splits["test"] = split_2["test"]
    else:
        # Last-resort path for datasets exposing only validation/test.
        first_split = list(raw.keys())[0]
        rows = [_parse_pubmedqa_row(r) for r in raw[first_split]]
        base = Dataset.from_list(rows)
        split_1 = base.train_test_split(test_size=0.20, seed=cfg.seed, shuffle=True)
        split_2 = split_1["test"].train_test_split(test_size=0.50, seed=cfg.seed, shuffle=True)
        splits["train"] = split_1["train"]
        splits["validation"] = split_2["train"]
        splits["test"] = split_2["test"]

    size_map = {
        "train": cfg.subset_train_size,
        "validation": cfg.subset_val_size,
        "test": cfg.subset_test_size,
    }
    for split_name in list(splits.keys()):
        if cfg.use_subset:
            splits[split_name] = _apply_subset(splits[split_name], size_map[split_name])
        print(f"  [{split_name}] {len(splits[split_name])} samples")

    return DatasetDict(splits)

# =============================================================================
# UNIFIED DATASET LOADER
# =============================================================================

def load_all_datasets(cfg: Config) -> DatasetDict:
    """
    Single entry point — routes to local or HuggingFace loader based on
    dataset_name. Original local datasets behave exactly as before.
    """
    name = cfg.dataset_name.lower()
    print(f"\n  Loading dataset: {name}")

    if name in ("medquad", "icliniq"):
        return load_local_datasets(cfg)
    elif name == "medqa":
        return load_medqa(cfg)
    elif name == "medmcqa":
        return load_medmcqa(cfg)
    elif name == "mmlu_med":
        return load_mmlu_med(cfg)
    elif name == "pubmedqa":
        return load_pubmedqa(cfg)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}"
        )


# =============================================================================
# ORIGINAL SUMMARIZATION (UNCHANGED)
# =============================================================================

def summarize_dataset(ds: Dataset, cfg: Config, split_name: str = "train"):
    os.makedirs(cfg.out_dir, exist_ok=True)
    entry_for_cache = DATASET_REGISTRY.get(cfg.dataset_name, {})
    cache_tag = cfg.dataset_name.replace("/", "_")
    if entry_for_cache.get("task", "open_ended") == "mcq":
        # Versioned cache so old MCQ summaries that included question/options are
        # not reused as generator targets.
        cache_tag += "_mcqtarget_v2"
    save_path = os.path.join(
        cfg.out_dir, f"{split_name}_summaries_{cache_tag}.json"
    )

    def _clean_summary(s: str) -> str:
        if s is None:
            return ""
        s  = s.strip()
        sl = s.lower()
        for p in ["summary:", "summary -", "summary :", "summarize the following",
                  "summarize:", "summarize :", "answer:", "summary\n:"]:
            if sl.startswith(p):
                s  = s[len(p):].strip()
                sl = s.lower()
        tokens    = s.split()
        out_toks  = []
        prev, rep = None, 0
        for t in tokens:
            rep = rep + 1 if t == prev else 0
            if rep < 3:
                out_toks.append(t)
            prev = t
        return " ".join(" ".join(out_toks).split())

    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) == len(ds):
                print(f"  Loaded cached summaries for {split_name} ({len(cached)}).")
                return ds.add_column("summary", cached)
            print(f"  Cache length mismatch — regenerating ...")
        except Exception as e:
            print(f"  Cache load failed ({e}) — regenerating ...")

    print(f"  Generating summaries for {split_name} ...")
    entry  = DATASET_REGISTRY.get(cfg.dataset_name, {})
    is_mcq = (entry.get("task", "open_ended") == "mcq")

    tok   = None
    model = None
    if not is_mcq:
        tok = AutoTokenizer.from_pretrained(cfg.summarizer_model, trust_remote_code=cfg.trust_remote_code)
        set_tokenizer_defaults(tok)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.summarizer_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=cfg.trust_remote_code,
        )

    summaries = []
    for ex in tqdm(ds, desc=f"Summarising {split_name}"):
        if is_mcq:
            # MCQ target is now exactly what the model should learn to emit.
            # This fixes the earlier target leakage where summary contained
            # question + all options + "Correct answer".
            answer_text = _norm(ex.get("answer", ""))
            gold_letter = _norm(ex.get("gold_letter", ""))
            if gold_letter:
                summary = f"{gold_letter}. {answer_text}"
            else:
                summary = answer_text or _norm(ex.get("explanation", "No answer available"))
            summaries.append(summary.strip())
            continue

        # Open-ended path (original logic preserved)
        answer_text = ex.get("answer", ex.get("explanation", ""))
        prompt = (
            "Summarize the following medical answer factually and concisely "
            "(limit ~100 tokens):\n\n"
            f"{answer_text}\n\nSummary:"
        )
        inputs = tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=cfg.max_input_tokens, padding=True
        ).to(cfg.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=cfg.max_summary_tokens,
                pad_token_id=tok.eos_token_id, do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        raw     = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        cleaned = _clean_summary(raw)
        summaries.append(cleaned)

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"  Saved {len(summaries)} summaries → {save_path}")
    except Exception as e:
        print(f"  WARNING: could not save summaries: {e}")

    return ds.add_column("summary", summaries)

# =============================================================================
# ORIGINAL ENTITY EXTRACTION (UNCHANGED)
# =============================================================================

try:
    NLP_MED = spacy.load("en_ner_bc5cdr_md")
except Exception:
    raise RuntimeError(
        "scispaCy model not found. Install:\n"
        "  pip install scispacy\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
        "releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz"
    )

MED_ENT_VOCAB: set = set()


def extract_medical_entities(text: str):
    if not text or not isinstance(text, str):
        return []
    doc  = NLP_MED(text)
    ents = set()
    for ent in doc.ents:
        clean = ent.text.strip().lower()
        if len(clean) > 2:
            ents.add(clean)
    return sorted(list(ents))


def build_entity_vocabulary(dataset: Dataset, cfg: Config, path=None):
    if path is None:
        path = os.path.join(cfg.out_dir, "medical_entities.json")
    all_ents = set()
    print("  Extracting medical entity vocabulary ...")
    for ex in tqdm(dataset, desc="Extracting"):
        text = ex.get("summary") or ex.get("answer") or ""
        if text:
            all_ents.update(extract_medical_entities(text))
    all_ents = sorted(list(all_ents))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_ents, f, indent=2)
    with open(path.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        for e in all_ents:
            f.write(e + "\n")
    print(f"  {len(all_ents)} entities → {path}")
    return all_ents


def load_entity_vocabulary(path):
    if not os.path.exists(path):
        return set()
    return set(json.load(open(path, "r", encoding="utf-8")))


def extract_entities(text: str):
    if not text:
        return []
    tokens = [t.strip(".,;:()").lower() for t in re.split(r"\s+", text)]
    return sorted({t for t in tokens if t in MED_ENT_VOCAB})


def add_entities_column(ds: Dataset):
    ents = [extract_entities(x["summary"]) for x in ds]
    return ds.add_column("entities", ents)


# =============================================================================
# ORIGINAL FAISS INDEX BUILDER (UNCHANGED)
# =============================================================================

def build_per_type_index(ds: Dataset, cfg: Config):
    os.makedirs(cfg.per_type_index_dir, exist_ok=True)
    embedder = SentenceTransformer(cfg.embedder_model, device=cfg.device)
    grouped: Dict[str, List] = {}
    for i, row in enumerate(ds):
        qt = row.get("qtype", "general")
        grouped.setdefault(qt, []).append((i, row["question"], row["summary"]))

    for qt, items in grouped.items():
        questions = [q for (_, q, _) in items]
        embs      = embedder.encode(
            questions, convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        faiss.write_index(
            index,
            os.path.join(cfg.per_type_index_dir, f"idx_{qt}.faiss")
        )
        meta = {
            "ids":       [it[0] for it in items],
            "questions": questions,
            "summaries": [s for (_, _, s) in items],
        }
        with open(
            os.path.join(cfg.per_type_index_dir, f"meta_{qt}.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f)


def retrieve_exemplars(question: str, qtype: str, cfg: Config, topk: int = None):
    topk     = topk or cfg.n_exemplars
    idx_path = os.path.join(cfg.per_type_index_dir, f"idx_{qtype}.faiss")
    meta_path = os.path.join(cfg.per_type_index_dir, f"meta_{qtype}.json")
    if not os.path.exists(idx_path):
        files = [f for f in os.listdir(cfg.per_type_index_dir)
                 if f.startswith("idx_")]
        if not files:
            return []
        idx_path  = os.path.join(cfg.per_type_index_dir, files[0])
        meta_path = os.path.join(
            cfg.per_type_index_dir,
            files[0].replace("idx_", "meta_").replace(".faiss", ".json")
        )
    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    embedder = SentenceTransformer(cfg.embedder_model, device=cfg.device)
    q_emb    = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    _, I     = index.search(q_emb, min(topk, index.ntotal))
    exemplars = []
    for pos in I[0].tolist():
        if pos < len(meta["summaries"]):
            exemplars.append({
                "question": meta["questions"][pos],
                "summary":  meta["summaries"][pos],
            })
    return exemplars


# =============================================================================
# ORIGINAL PROMPT ASSEMBLY (UNCHANGED)
# =============================================================================

def _format_choice_letters(num_choices: int) -> str:
    letters = [chr(65 + i) for i in range(num_choices)]
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def assemble_prompt(question: str, exemplars: List[Dict],
                    options_text: str = "",
                    task: str = "open_ended",
                    num_choices: int = 4) -> str:
    """
    Build prompt. MCQ prompts are answer-only and dynamically support 3-choice
    PubMedQA as well as 4-choice medical benchmarks.
    """
    if task == "mcq" and options_text:
        valid_letters = _format_choice_letters(num_choices)
        return (
            "You are a medical expert taking a multiple-choice medical exam.\n"
            "Choose the single best answer from the options.\n"
            f"Valid choices are {valid_letters}.\n"
            "Return ONLY the option in this exact format: '<letter>. <answer>'.\n"
            "Do not add extra explanation.\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}\n\n"
            "Answer:"
        )

    # Open-ended format (original, unchanged)
    parts = [
        "You are a medical expert.\n"
        "Use clinical reasoning to answer concisely.\n"
        "Answer must be related to the question.\n"
    ]
    for ex in exemplars:
        parts.append(f"Q: {ex['question']}\nA: {ex['summary']}\n---\n")
    parts.append(f"Q: {question}\nA:")
    return "\n".join(parts)

def prepare_prompted_datasets(ds: Dataset, cfg: Config) -> Dataset:
    entry = DATASET_REGISTRY.get(cfg.dataset_name, {})
    task  = entry.get("task", "open_ended")
    prompts = []
    for ex in ds:
        opts = ex.get("options_text", "")
        num_choices = len(ex.get("options_list", []) or []) or entry.get("num_choices", 4)
        if task == "mcq":
            prompt = assemble_prompt(ex["question"], [], opts, task=task, num_choices=num_choices)
        else:
            exs = retrieve_exemplars(ex["question"], ex.get("qtype", "general"), cfg)
            prompt = assemble_prompt(ex["question"], exs, opts, task=task, num_choices=num_choices)
        prompts.append(prompt)
    return ds.add_column("prompt", prompts)

# =============================================================================
# ORIGINAL TRAIN_GENERATOR (UNCHANGED)
# =============================================================================

from transformers import DataCollatorForSeq2Seq

def train_generator(train_ds: Dataset, val_ds: Dataset, cfg: Config):
    entry = DATASET_REGISTRY.get(cfg.dataset_name, {})
    is_mcq = entry.get("task", "open_ended") == "mcq"

    tok = AutoTokenizer.from_pretrained(cfg.generator_model, trust_remote_code=cfg.trust_remote_code)
    set_tokenizer_defaults(tok, left_truncate=is_mcq)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.generator_model,
        **model_load_kwargs(cfg, for_training=True),
    )
    model = maybe_apply_lora(model, cfg)

    if cfg.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    def preprocess(batch):
        prompts = list(batch["prompt"])
        targets = list(batch["summary"])
        model_inputs = []
        labels_list  = []
        attention_masks = []
        max_total = cfg.max_input_tokens + cfg.max_summary_tokens + 1

        for p, t in zip(prompts, targets):
            p_ids = tok.encode(p, add_special_tokens=False)
            p_ids = truncate_token_ids(
                p_ids, cfg.max_input_tokens,
                keep="right" if is_mcq else "left"
            )
            t_ids = tok.encode(
                t, add_special_tokens=False,
                truncation=True,
                max_length=cfg.max_summary_tokens,
            )

            full = p_ids + [tok.eos_token_id] + t_ids + [tok.eos_token_id]
            if len(full) > max_total:
                full = full[:max_total]
            labels = [-100] * (len(p_ids) + 1) + t_ids + [tok.eos_token_id]
            labels = labels[:len(full)]

            model_inputs.append(full)
            labels_list.append(labels)
            attention_masks.append([1] * len(full))

        return {"input_ids": model_inputs, "labels": labels_list,
                "attention_mask": attention_masks}

    print("  Tokenising train/val ...")
    t = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    v = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    args = make_training_arguments(
        output_dir=os.path.join(cfg.out_dir, "generator"),
        per_device_train_batch_size=max(1, cfg.batch_size),
        per_device_eval_batch_size=max(1, cfg.batch_size),
        gradient_accumulation_steps=max(1, cfg.gradient_accumulation_steps),
        num_train_epochs=max(1, cfg.epochs),
        learning_rate=cfg.lr,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        report_to="none",
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        dataloader_num_workers=cfg.num_workers,
        ddp_find_unused_parameters=cfg.ddp_find_unused_parameters if is_distributed() else None,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=t,
        eval_dataset=v,
        processing_class=tok,
        data_collator=data_collator,
    )

    print("  Fine-tuning generator ...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    plot_training_curves(trainer, args.output_dir, "generator")

    if hasattr(model, "config"):
        model.config.use_cache = True
    print("  Generator fine-tuning complete.")
    return model, tok

# =============================================================================
# ORIGINAL TRAIN_VERIFIER (UNCHANGED)
# =============================================================================

def build_verifier_dataset(ds: Dataset):
    texts, labels = [], []
    rng       = random.Random(42)
    summaries = list(ds["summary"])
    for ex in ds:
        q, pos = ex["question"], ex["summary"]
        texts.append(f"Q: {q} [SEP] A: {pos}"); labels.append(1)
        neg = summaries[rng.randrange(len(summaries))]
        texts.append(f"Q: {q} [SEP] A: {neg}"); labels.append(0)
    return Dataset.from_list([{"text": t, "label": l}
                               for t, l in zip(texts, labels)])


def train_verifier(train_v: Dataset, val_v: Dataset, cfg: Config):
    tok   = AutoTokenizer.from_pretrained(cfg.verifier_model, trust_remote_code=cfg.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.verifier_model, num_labels=2, trust_remote_code=cfg.trust_remote_code
    )
    if not is_distributed():
        model = model.to(cfg.device)

    def preprocess(ex):
        d = tok(ex["text"], truncation=True, padding="max_length", max_length=256)
        d["labels"] = ex["label"]
        return d

    t = train_v.map(preprocess, batched=True, remove_columns=train_v.column_names)
    v = val_v.map(preprocess, batched=True, remove_columns=val_v.column_names)

    args = make_training_arguments(
        output_dir=os.path.join(cfg.out_dir, "verifier"),
        per_device_train_batch_size=max(1, min(cfg.batch_size, 8)),
        per_device_eval_batch_size=max(1, min(cfg.batch_size, 8)),
        gradient_accumulation_steps=max(1, cfg.gradient_accumulation_steps),
        num_train_epochs=max(1, min(cfg.epochs, 3)),
        learning_rate=cfg.lr,
        evaluation_strategy="steps",
        eval_steps=max(50, cfg.eval_steps),
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=cfg.num_workers,
        ddp_find_unused_parameters=cfg.ddp_find_unused_parameters if is_distributed() else None,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=t,
        eval_dataset=v,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    plot_training_curves(trainer, args.output_dir, "verifier")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.output_dir, num_labels=2, trust_remote_code=cfg.trust_remote_code
    )
    if not is_distributed():
        model = model.to(cfg.device)
    return model, tok

# =============================================================================
# ORIGINAL CONTRASTIVE FINE-TUNING (UNCHANGED)
# =============================================================================

def contrastive_finetune_generator_encoder(
    model, tok, train_ds: Dataset, cfg: Config
):
    embedder   = SentenceTransformer(cfg.embedder_model, device=cfg.device)
    llama_hid  = model.config.hidden_size
    st_hid     = embedder.get_sentence_embedding_dimension()
    proj_dtype = next(model.parameters()).dtype
    projection = nn.Linear(llama_hid, st_hid, dtype=proj_dtype).to(cfg.device)
    encoder    = model.get_input_embeddings()
    optimizer  = optim.Adam(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=cfg.lr * 0.5,
    )

    for ep in range(cfg.contrastive_epochs):
        perm = list(range(len(train_ds)))
        random.shuffle(perm)
        pbar = tqdm(
            range(0, len(train_ds), cfg.contrastive_batch_size),
            desc=f"Contrastive epoch {ep+1}",
        )
        for i in pbar:
            idxs    = perm[i:i + cfg.contrastive_batch_size]
            batch   = [train_ds[int(ii)] for ii in idxs]
            prompts = [x["prompt"]  for x in batch]
            pos     = [x["summary"] for x in batch]
            neg     = [train_ds[random.randrange(len(train_ds))]["summary"]
                       for _ in batch]

            enc = tok(
                prompts, return_tensors="pt",
                truncation=True, padding=True,
                max_length=cfg.max_input_tokens
            ).to(cfg.device)
            q = model.model.embed_tokens(enc["input_ids"]).mean(dim=1).to(proj_dtype)
            q = projection(q)
            with torch.no_grad():
                p = embedder.encode(pos, convert_to_tensor=True, device=cfg.device)
                n = embedder.encode(neg, convert_to_tensor=True, device=cfg.device)
            q = nn.functional.normalize(q, dim=-1)
            p = nn.functional.normalize(p, dim=-1)
            n = nn.functional.normalize(n, dim=-1)
            sim_pos = torch.cosine_similarity(q, p) / cfg.contrastive_temp
            sim_neg = torch.cosine_similarity(q, n) / cfg.contrastive_temp
            loss = -(
                torch.log(
                    torch.exp(sim_pos) /
                    (torch.exp(sim_pos) + torch.exp(sim_neg))
                )
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"loss": float(loss.detach().cpu())})


# =============================================================================
# ORIGINAL clean_generated (UNCHANGED)
# =============================================================================

def clean_generated(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s     = text.strip()
    lower = s.lower()
    if "\nA:" in lower:
        idx = max(lower.rfind("\nA:"), lower.rfind("\na:"))
        s   = s[idx + 3:].strip()
    elif " a:" in lower and lower.count("q:") > 0:
        idx = lower.rfind(" a:")
        s   = s[idx + 3:].strip()
    low = s.lower()
    for p in ["summary:", "summarize:", "summarize the following",
              "medical answer", "answer the following",
              "you are a medical expert",
              "medical answer factually", "medical answer concisely"]:
        if low.startswith(p):
            s   = s[len(p):].strip()
            low = s.lower()
    s = re.sub(r"\b(\w+)( \1){3,}\b", r"\1", s)
    s = re.sub(r"[\uFEFF\uFFFD]+", "", s)
    s = s.replace("://", " ").replace("  ", " ")
    s = " ".join(s.split())
    if len(s) > 2000:
        s = s[:2000].rsplit(".", 1)[0]
    return s.strip()


# =============================================================================
# NEW: MCQ ANSWER EXTRACTION
# =============================================================================

def extract_mcq_prediction(generated_text: str,
                            options_list: List[str],
                            num_choices: int = 4) -> str:
    """
    Extract the predicted option letter from generated text. Supports A-D and
    PubMedQA A-C yes/no/maybe.
    """
    text = (generated_text or "").strip()
    valid = [chr(65 + i) for i in range(num_choices)]
    if not text:
        return valid[0] if valid else "A"

    letter_class = "".join(valid)
    escaped = re.escape(letter_class)

    # 1) Letter at very start: "A", "A.", "A)".
    m = re.match(rf"^([{escaped}])(?:\s|[\.\)\:\-]|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2) Explicit answer phrase.
    for pat in [
        rf"(?:the\s+)?answer\s+is\s+([{escaped}])",
        rf"answer:\s*([{escaped}])",
        rf"correct\s+(?:answer|option|choice)\s+is\s+([{escaped}])",
        rf"option\s+([{escaped}])\s+is\s+correct",
        rf"\bselect\s+([{escaped}])\b",
    ]:
        m = re.search(pat, text[:240], re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 3) Yes/No/Maybe verbalizer for PubMedQA.
    text_lower = text.lower()
    ynm_map = {"yes": "A", "no": "B", "maybe": "C"}
    for word, letter in ynm_map.items():
        if letter in valid and re.search(r"\b" + word + r"\b", text_lower[:120]):
            return letter

    # 4) Match option text.
    best_match_idx, best_match_len = -1, 0
    for i, opt in enumerate(options_list or []):
        opt_l = _norm(opt).lower()
        if opt_l and opt_l in text_lower and len(opt_l) > best_match_len:
            best_match_idx = i
            best_match_len = len(opt_l)
    if best_match_idx >= 0 and best_match_idx < len(valid):
        return valid[best_match_idx]

    # 5) Character/word overlap fallback.
    if options_list:
        gen_words = set(re.findall(r"[a-zA-Z0-9]+", text_lower))
        scores = []
        for opt in options_list:
            opt_words = set(re.findall(r"[a-zA-Z0-9]+", _norm(opt).lower()))
            scores.append(len(opt_words & gen_words))
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if best_idx < len(valid):
            return valid[best_idx]

    return valid[0] if valid else "A"

def letter_to_text(letter: str, options_list: List[str]) -> str:
    """Convert a letter (A/B/C/D) to the corresponding option text."""
    idx = ord(letter.upper()) - 65
    if 0 <= idx < len(options_list):
        return options_list[idx]
    return ""



# =============================================================================
# STATISTICAL REPORTING — Bootstrap CI + paired tests
# =============================================================================

def bootstrap_ci_and_p(
    scores_a: list,
    scores_b: list = None,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for scores_a. If scores_b is provided,
    compute a paired bootstrap CI for (a-b), paired bootstrap P-value, paired
    t-test P-value, and Cohen's d.
    """
    arr_a = np.array(scores_a, dtype=float)
    if arr_a.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    n = len(arr_a)
    lo = (100.0 - ci) / 2.0
    hi = 100.0 - lo

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means_a = arr_a[idx].mean(axis=1)
    result = {
        "mean": float(arr_a.mean()),
        "ci_low": float(np.percentile(boot_means_a, lo)),
        "ci_high": float(np.percentile(boot_means_a, hi)),
    }

    if scores_b is not None:
        arr_b = np.array(scores_b, dtype=float)
        if len(arr_b) != n:
            raise ValueError("scores_a and scores_b must have the same length")
        diffs = arr_a - arr_b
        boot_diffs = diffs[idx].mean(axis=1)
        observed_diff = float(diffs.mean())
        shifted = boot_diffs - boot_diffs.mean()
        p_boot = float(np.mean(np.abs(shifted) >= abs(observed_diff)))
        p_boot = max(p_boot, 1.0 / max(1, n_boot))
        p_ttest = float("nan")
        if ttest_rel is not None:
            _, p_ttest = ttest_rel(arr_a, arr_b)
        cohens_d = float(diffs.mean() / (diffs.std(ddof=1) + 1e-12)) if n > 1 else float("nan")
        result.update({
            "mean_b": float(arr_b.mean()),
            "diff_mean": observed_diff,
            "diff_ci_low": float(np.percentile(boot_diffs, lo)),
            "diff_ci_high": float(np.percentile(boot_diffs, hi)),
            "p_bootstrap": p_boot,
            "p_ttest": p_ttest,
            "cohens_d": cohens_d,
        })
    return result


def format_ci(mean: float, ci_low: float, ci_high: float,
              scale: float = 1.0, decimals: int = 4) -> str:
    m  = round(mean * scale, decimals)
    lo = round(ci_low * scale, decimals)
    hi = round(ci_high * scale, decimals)
    return f"{m} (95% CI {lo}–{hi})"


def format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "P=NA"
    if p < 0.001:
        return "P<.001"
    if p < 0.01:
        return f"P={p:.3f}"
    return f"P={p:.2f}"


# =============================================================================
# MCQ LOG-LIKELIHOOD SCORING — recommended for PubMedQA
# =============================================================================

def _mcq_candidate_variants(letter: str, option: str, num_choices: int) -> List[str]:
    opt = _norm(option)
    variants = [f" {letter}", f" {letter}. {opt}", f"{letter}. {opt}"]
    if opt.lower() in {"yes", "no", "maybe"}:
        variants.extend([f" {opt.lower()}", f" {letter}. {opt.lower()}"])
    return list(dict.fromkeys(variants))


def _score_continuation_logprob(gen_model, gen_tok, prompt: str,
                                continuation: str, cfg: Config) -> float:
    """Average log probability of a continuation under a causal LM."""
    device = first_model_device(gen_model)
    prompt_ids = gen_tok.encode(prompt, add_special_tokens=False)
    prompt_ids = truncate_token_ids(prompt_ids, cfg.max_input_tokens, keep="right")
    cont_ids = gen_tok.encode(continuation, add_special_tokens=False)
    if not cont_ids:
        return -1e9

    input_ids = prompt_ids + cont_ids
    labels = [-100] * len(prompt_ids) + cont_ids
    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)
    attn_t = torch.ones_like(input_t, device=device)

    with torch.no_grad():
        out = gen_model(input_ids=input_t, attention_mask=attn_t, labels=labels_t)
    # HF returns mean CE loss over non-ignored continuation tokens.
    return -float(out.loss.detach().cpu().item())


def predict_mcq_by_logprob(question: str,
                           gen_model,
                           gen_tok,
                           cfg: Config,
                           options_text: str,
                           options_list: List[str],
                           num_choices: int) -> Tuple[str, str, Dict[str, float]]:
    prompt = assemble_prompt(
        question, [], options_text, task="mcq", num_choices=num_choices
    )
    set_tokenizer_defaults(gen_tok, left_truncate=True)
    gen_model.eval()
    if hasattr(gen_model, "config"):
        gen_model.config.use_cache = True

    valid = [chr(65 + i) for i in range(num_choices)]
    scores: Dict[str, float] = {}
    for i, opt in enumerate(options_list[:num_choices]):
        letter = valid[i]
        var_scores = [
            _score_continuation_logprob(gen_model, gen_tok, prompt, v, cfg)
            for v in _mcq_candidate_variants(letter, opt, num_choices)
        ]
        scores[letter] = max(var_scores) if var_scores else -1e9

    pred_letter = max(scores, key=scores.get) if scores else valid[0]
    pred_text = f"{pred_letter}. {letter_to_text(pred_letter, options_list)}".strip()
    return pred_text, pred_letter, scores


# =============================================================================
# ORIGINAL generate_and_rerank EXTENDED for MCQ
# =============================================================================

def generate_and_rerank(
    question:      str,
    qtype:         str,
    gen_model,
    gen_tok,
    ver_model,
    ver_tok,
    cfg:           Config,
    options_text:  str = "",
    options_list:  List[str] = None,
    num_choices:   int = 4,
    task:          str = "open_ended",
) -> Tuple[str, str]:
    """
    Generate/rank an answer. For MCQ, log-likelihood scoring is the default,
    because it is more stable than free-text extraction and improves PubMedQA.
    """
    options_list = options_list or []

    if task == "mcq" and options_list and cfg.mcq_decoding in {"logprob", "hybrid"}:
        pred_text, pred_letter, _ = predict_mcq_by_logprob(
            question, gen_model, gen_tok, cfg,
            options_text=options_text,
            options_list=options_list,
            num_choices=num_choices,
        )
        if cfg.mcq_decoding == "logprob":
            return pred_text, pred_letter
        # Hybrid mode falls through to generation only if requested; logprob is
        # retained if generation is empty or unparseable.
        logprob_fallback = (pred_text, pred_letter)
    else:
        logprob_fallback = None

    exemplars = [] if task == "mcq" else retrieve_exemplars(question, qtype, cfg)
    prompt = assemble_prompt(question, exemplars, options_text, task=task, num_choices=num_choices)

    set_tokenizer_defaults(gen_tok, left_truncate=(task == "mcq"))
    prompt_ids = gen_tok.encode(prompt, add_special_tokens=False)
    prompt_ids = truncate_token_ids(
        prompt_ids, cfg.max_input_tokens,
        keep="right" if task == "mcq" else "left",
    )
    inputs = {
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=first_model_device(gen_model)),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=first_model_device(gen_model)),
    }

    num_beams = max(1, min(6, cfg.n_best))
    num_ret   = min(cfg.n_best, num_beams) if num_beams > 1 else 1
    max_new = cfg.mcq_max_new_tokens if task == "mcq" else cfg.max_summary_tokens

    gen_model.eval()
    if hasattr(gen_model, "config"):
        gen_model.config.use_cache = True
    if hasattr(gen_model, "gradient_checkpointing_disable"):
        gen_model.gradient_checkpointing_disable()

    with torch.no_grad():
        outs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new,
            num_beams=num_beams,
            num_return_sequences=num_ret,
            do_sample=False,
            use_cache=True,
            pad_token_id=gen_tok.eos_token_id,
            eos_token_id=gen_tok.eos_token_id,
            early_stopping=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    candidates = [
        clean_generated(gen_tok.decode(o[prompt_len:], skip_special_tokens=True))
        for o in outs
    ]
    candidates = [c for c in candidates if c.strip()]
    if not candidates:
        return logprob_fallback if logprob_fallback is not None else ("", "A")

    if task == "mcq":
        # For answer-only MCQ prompts, first beam is usually best. Extraction is
        # used only for generate/hybrid modes.
        best_text = candidates[0]
        pred_letter = extract_mcq_prediction(best_text, options_list, num_choices)
        if pred_letter:
            return f"{pred_letter}. {letter_to_text(pred_letter, options_list)}".strip(), pred_letter
        return logprob_fallback if logprob_fallback is not None else (best_text, "A")

    best_text, best_score = None, -1e9
    prompt_ents = set(extract_entities(prompt))

    for c in candidates:
        verif_prob = 0.0
        if ver_model is not None and ver_tok is not None:
            v_in = ver_tok(
                f"Q: {question} [SEP] A: {c}",
                return_tensors="pt", truncation=True,
                max_length=256,
            ).to(first_model_device(ver_model))
            with torch.no_grad():
                verif_prob = float(
                    torch.softmax(ver_model(**v_in).logits, dim=-1)[0, 1].item()
                )
        cand_ents = set(extract_entities(c))
        ent_cov = len(cand_ents & prompt_ents) / len(prompt_ents) if prompt_ents else 0.0
        score = 0.75 * verif_prob + 0.25 * ent_cov
        if score > best_score:
            best_score = score
            best_text = c

    best_text = best_text or candidates[0]
    return best_text, ""

# =============================================================================
# ORIGINAL EVALUATE EXTENDED — adds accuracy for MCQ datasets
# =============================================================================

def evaluate(
    ds:        Dataset,
    gen_model,
    gen_tok,
    ver_model,
    ver_tok,
    cfg:       Config,
    val_ds:    Dataset = None,
) -> Dict:
    """
    Unified evaluation function with bootstrap 95% CIs.
    """
    entry    = DATASET_REGISTRY.get(cfg.dataset_name, {})
    task     = entry.get("task", "open_ended")
    is_mcq   = (task == "mcq")
    num_choices = entry.get("num_choices", 4)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    meteor_m    = load_metric("meteor")
    bertscore_m = load_metric("bertscore")
    bleu_m      = load_metric("bleu")

    hyps, refs, factuals, halls = [], [], [], []
    correct_count = 0
    correct_binary: List[float] = []

    if val_ds is not None and len(val_ds) > 0:
        ex = val_ds[random.randrange(len(val_ds))]
        print("\n======== RANDOM VALIDATION SAMPLE ========")
        print("Q:", ex["question"][:400])
        print("GT:", ex.get("summary", ex.get("answer", ""))[:200])
        pred_val, letter_val = generate_and_rerank(
            ex["question"], ex.get("qtype", "general"),
            gen_model, gen_tok, ver_model, ver_tok, cfg,
            options_text=ex.get("options_text", ""),
            options_list=ex.get("options_list", []),
            num_choices=len(ex.get("options_list", []) or []) or num_choices,
            task=task,
        )
        print("Pred:", pred_val[:200])
        if is_mcq:
            print(f"Predicted letter: {letter_val}  |  Gold letter: {ex.get('gold_letter','?')}")
        print("==========================================\n")

    print("\n  Computing metrics ...")
    for ex in tqdm(ds, desc="Evaluating"):
        q   = ex["question"]
        ref = ex.get("summary", ex.get("answer", ""))

        opts_text = ex.get("options_text", "")
        opts_list = ex.get("options_list", [])
        ex_num_choices = len(opts_list or []) or num_choices
        gold_ltr  = ex.get("gold_letter", "")

        pred, pred_ltr = generate_and_rerank(
            q, ex.get("qtype", "general"),
            gen_model, gen_tok, ver_model, ver_tok, cfg,
            options_text=opts_text,
            options_list=opts_list,
            num_choices=ex_num_choices,
            task=task,
        )

        pred = pred if pred.strip() else " "
        ref = ref if _norm(ref) else " "
        hyps.append(pred)
        refs.append(ref)

        hit = 0.0
        if is_mcq:
            hit = 1.0 if (gold_ltr and pred_ltr and pred_ltr.upper() == gold_ltr.upper()) else 0.0
            correct_count += int(hit)
            correct_binary.append(hit)

        if ver_model is not None and ver_tok is not None:
            v_in = ver_tok(
                f"Q: {q} [SEP] A: {pred}",
                return_tensors="pt", truncation=True,
                max_length=256,
            ).to(first_model_device(ver_model))
            with torch.no_grad():
                prob = float(torch.softmax(ver_model(**v_in).logits, dim=-1)[0, 1].item())
            factuals.append(prob)
        elif is_mcq:
            # When verifier training is intentionally skipped for MCQ, use the
            # correctness indicator as a conservative factuality proxy.
            factuals.append(hit)
        else:
            factuals.append(0.0)

        pred_t = set(pred.split())
        ref_t  = set(ref.split())
        halls.append(1 - len(pred_t & ref_t) / max(1, len(pred_t | ref_t)))

    # ── Per-sample metrics ───────────────────────────────────────────────
    r1_scores = [rouge.score(r, p)["rouge1"].fmeasure for r, p in zip(refs, hyps)]
    r2_scores = [rouge.score(r, p)["rouge2"].fmeasure for r, p in zip(refs, hyps)]
    rl_scores = [rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, hyps)]
    # METEOR is computed per sample to enable CI; corpus METEOR alone cannot be bootstrapped.
    meteor_scores = [
        meteor_m.compute(predictions=[p], references=[r])["meteor"]
        for r, p in zip(refs, hyps)
    ]
    bs = bertscore_m.compute(predictions=hyps, references=refs, lang="en")
    bs_scores = list(bs["f1"])

    r1  = float(np.mean(r1_scores))
    r2  = float(np.mean(r2_scores))
    rl  = float(np.mean(rl_scores))
    met = float(np.mean(meteor_scores))
    bsf = float(np.mean(bs_scores))

    b1 = bleu_m.compute(predictions=hyps, references=[[r] for r in refs], max_order=1)["bleu"]
    b4 = bleu_m.compute(predictions=hyps, references=[[r] for r in refs], max_order=4)["bleu"]

    n_boot = max(100, int(cfg.bootstrap_samples))
    ci = float(cfg.ci_level)
    ci_r1  = bootstrap_ci_and_p(r1_scores, n_boot=n_boot, ci=ci, seed=cfg.seed)
    ci_r2  = bootstrap_ci_and_p(r2_scores, n_boot=n_boot, ci=ci, seed=cfg.seed)
    ci_rl  = bootstrap_ci_and_p(rl_scores, n_boot=n_boot, ci=ci, seed=cfg.seed)
    ci_met = bootstrap_ci_and_p(meteor_scores, n_boot=n_boot, ci=ci, seed=cfg.seed)
    ci_bs  = bootstrap_ci_and_p(bs_scores, n_boot=n_boot, ci=ci, seed=cfg.seed)
    ci_fa  = bootstrap_ci_and_p(factuals, n_boot=n_boot, ci=ci, seed=cfg.seed)
    ci_hal = bootstrap_ci_and_p(halls, n_boot=n_boot, ci=ci, seed=cfg.seed)

    results: Dict = {
        "ROUGE1":    round(r1, 4),
        "ROUGE2":    round(r2, 4),
        "ROUGEL":    round(rl, 4),
        "METEOR":    round(met, 4),
        "BERTScore": round(bsf, 4),
        "FactualAccuracy":   round(float(np.mean(factuals)), 4),
        "HallucinationRate": round(float(np.mean(halls)), 4),

        "ROUGE1_CI":    format_ci(ci_r1["mean"], ci_r1["ci_low"], ci_r1["ci_high"]),
        "ROUGE2_CI":    format_ci(ci_r2["mean"], ci_r2["ci_low"], ci_r2["ci_high"]),
        "ROUGEL_CI":    format_ci(ci_rl["mean"], ci_rl["ci_low"], ci_rl["ci_high"]),
        "METEOR_CI":    format_ci(ci_met["mean"], ci_met["ci_low"], ci_met["ci_high"]),
        "BERTScore_CI": format_ci(ci_bs["mean"], ci_bs["ci_low"], ci_bs["ci_high"]),
        "FactualAccuracy_CI": format_ci(ci_fa["mean"], ci_fa["ci_low"], ci_fa["ci_high"]),
        "HallucinationRate_CI": format_ci(ci_hal["mean"], ci_hal["ci_low"], ci_hal["ci_high"]),

        "ROUGE1_ci_low": ci_r1["ci_low"], "ROUGE1_ci_high": ci_r1["ci_high"],
        "ROUGE2_ci_low": ci_r2["ci_low"], "ROUGE2_ci_high": ci_r2["ci_high"],
        "ROUGEL_ci_low": ci_rl["ci_low"], "ROUGEL_ci_high": ci_rl["ci_high"],
        "METEOR_ci_low": ci_met["ci_low"], "METEOR_ci_high": ci_met["ci_high"],
        "BERTScore_ci_low": ci_bs["ci_low"], "BERTScore_ci_high": ci_bs["ci_high"],
    }

    if not is_mcq:
        results["BLEU1"] = round(float(b1), 4)
        results["BLEU4"] = round(float(b4), 4)

    if is_mcq:
        accuracy = correct_count / max(1, len(ds))
        ci_acc = bootstrap_ci_and_p(correct_binary, n_boot=n_boot, ci=ci, seed=cfg.seed)
        results["Accuracy"] = round(accuracy, 4)
        results["Accuracy_CI"] = format_ci(ci_acc["mean"], ci_acc["ci_low"], ci_acc["ci_high"],
                                            scale=100.0, decimals=2)
        results["Accuracy_ci_low"] = round(ci_acc["ci_low"] * 100, 2)
        results["Accuracy_ci_high"] = round(ci_acc["ci_high"] * 100, 2)

    print(f"\n======== RESULTS — {cfg.dataset_name.upper()} ========")
    for k, v in results.items():
        marker = " ← primary" if k == "Accuracy" else ""
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            print(f"  {k:24s}: {float(v):.4f}{marker}")
        else:
            print(f"  {k:24s}: {v}{marker}")
    print("=" * 50)

    return results

# =============================================================================
# ORIGINAL TRAINING CURVES (UNCHANGED)
# =============================================================================

def plot_training_curves(trainer, save_dir: str, name: str = "model"):
    logs = trainer.state.log_history
    df   = pd.DataFrame(logs)
    if df.empty:
        return
    os.makedirs(save_dir, exist_ok=True)

    if "loss" in df.columns:
        plt.figure()
        ld = df.dropna(subset=["loss"])
        plt.plot(ld["step"], ld["loss"])
        plt.xlabel("Step"); plt.ylabel("Loss")
        plt.title(f"{name} Training Loss")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}_loss_curve.png"))
        plt.close()

    if "learning_rate" in df.columns:
        plt.figure()
        lr = df.dropna(subset=["learning_rate"])
        plt.plot(lr["step"], lr["learning_rate"])
        plt.xlabel("Step"); plt.ylabel("LR")
        plt.title(f"{name} Learning Rate Schedule")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}_lr_curve.png"))
        plt.close()

    print(f"  Training curves saved: {save_dir}")


# =============================================================================
# NEW: COMPARISON TABLE ACROSS DATASETS
# =============================================================================

def save_multi_dataset_comparison(
    all_results: Dict[str, Dict], cfg: Config
):
    """Save CSV + Markdown + bar chart comparing results across datasets."""
    comp_dir = os.path.join(cfg.out_dir, "multi_dataset_comparison")
    os.makedirs(comp_dir, exist_ok=True)

    rows = []
    for dset, res in all_results.items():
        row = {"Dataset": dset,
               "Task": DATASET_REGISTRY.get(dset, {}).get("task", "?")}
        row.update(res)
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(comp_dir, "multi_dataset_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Multi-dataset CSV → {csv_path}")

    md_path = os.path.join(comp_dir, "multi_dataset_results.md")
    with open(md_path, "w") as f:
        f.write("# Multi-Dataset Evaluation Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"  Markdown → {md_path}")

    # Figure: ROUGE1 + METEOR + BERTScore per dataset
    common_metrics = ["ROUGE1", "METEOR", "BERTScore"]
    mcq_datasets   = [d for d, e in DATASET_REGISTRY.items()
                      if e.get("task") == "mcq"]

    n_d    = len(all_results)
    colors = plt.cm.Set2(np.linspace(0, 1, n_d))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, met in zip(axes, common_metrics):
        dsets = [d for d in all_results if met in all_results[d]]
        vals  = [all_results[d][met] for d in dsets]
        bars  = ax.bar(dsets, vals, color=colors[:len(dsets)])
        ax.set_title(met, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(vals) * 1.15 if vals else 1.0)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.suptitle("Multi-Dataset Comparison — ROUGE1 / METEOR / BERTScore",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "generation_metrics.png"), dpi=150)
    plt.close()

    # Accuracy bar for MCQ datasets
    acc_data = {d: r["Accuracy"] for d, r in all_results.items()
                if "Accuracy" in r}
    if acc_data:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        bars = ax2.bar(list(acc_data.keys()), list(acc_data.values()),
                       color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(acc_data))))
        ax2.set_title("MCQ Accuracy per Dataset", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.1)
        ax2.axhline(0.25, color="gray", linestyle="--",
                    linewidth=0.8, label="Random chance (4-choice)")
        ax2.legend(fontsize=9)
        for bar, val in zip(bars, acc_data.values()):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01, f"{val:.3f}",
                     ha="center", va="bottom", fontsize=9)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, "accuracy_mcq.png"), dpi=150)
        plt.close()
        print(f"  MCQ accuracy chart → {comp_dir}")

    print(f"  Multi-dataset figures → {comp_dir}")


# =============================================================================
# CHECKPOINT LOADERS (for --eval_only mode)
# =============================================================================

def load_generator_from_ckpt(ckpt_path: str, cfg: Config):
    print(f"  Loading generator from: {ckpt_path}")
    tok = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=cfg.trust_remote_code)
    set_tokenizer_defaults(tok, left_truncate=(DATASET_REGISTRY.get(cfg.dataset_name, {}).get("task") == "mcq"))
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        **model_load_kwargs(cfg, for_training=False),
    )
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tok

def load_verifier_from_ckpt(ckpt_path: str, cfg: Config):
    print(f"  Loading verifier from: {ckpt_path}")
    tok = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=cfg.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, num_labels=2, trust_remote_code=cfg.trust_remote_code
    )
    if not is_distributed():
        model = model.to(cfg.device)
    return model, tok

# =============================================================================
# SINGLE DATASET RUN
# =============================================================================

def run_one_dataset(dataset_name: str, cfg: Config) -> Dict:
    """Full pipeline for one dataset. Returns metrics dict."""
    cfg.dataset_name = dataset_name
    entry = DATASET_REGISTRY[dataset_name]
    is_mcq = entry.get("task") == "mcq"
    needs_verifier = (not is_mcq) or cfg.train_verifier_for_mcq

    root_out_dir = cfg.out_dir
    dataset_out_dir = os.path.join(root_out_dir, dataset_name.replace("/", "_"))
    cfg.out_dir = dataset_out_dir

    print(f"\n{'='*60}")
    print(f"  DATASET : {dataset_name}")
    print(f"  {entry['description']}")
    print(f"  Task    : {entry['task']}")
    print(f"  Output  : {cfg.out_dir}")
    print(f"{'='*60}")

    try:
        res_path = os.path.join(cfg.out_dir, f"metrics_{dataset_name}.json")
        if os.path.exists(res_path) and cfg.eval_only:
            with open(res_path) as f:
                cached = json.load(f)
            print(f"  Cached results loaded — {res_path}")
            return cached

        os.makedirs(cfg.out_dir, exist_ok=True)

        # 1) Load data
        data  = load_all_datasets(cfg)
        train = data.get("train")
        val   = data.get("validation")
        test  = data.get("test")

        # 2) Summaries/targets
        print("  Summarising ...")
        train = summarize_dataset(train, cfg, "train")
        if val   is not None: val  = summarize_dataset(val,  cfg, "validation")
        if test  is not None: test = summarize_dataset(test, cfg, "test")

        # 3) Entity vocabulary
        ent_path = os.path.join(cfg.out_dir, f"entities_{dataset_name}.json")
        global MED_ENT_VOCAB
        if os.path.exists(ent_path):
            MED_ENT_VOCAB = load_entity_vocabulary(ent_path)
            print(f"  Entity vocab loaded: {len(MED_ENT_VOCAB)} terms")
        else:
            ents = build_entity_vocabulary(train, cfg, ent_path)
            MED_ENT_VOCAB = set(ents)

        # 4) Add entity columns
        train = add_entities_column(train)
        if val  is not None: val  = add_entities_column(val)
        if test is not None: test = add_entities_column(test)

        # 5) Build per-type FAISS index
        idx_dir_orig = cfg.per_type_index_dir
        cfg.per_type_index_dir = os.path.join(idx_dir_orig, dataset_name.replace("/", "_"))
        os.makedirs(cfg.per_type_index_dir, exist_ok=True)
        print("  Building FAISS index ...")
        build_per_type_index(train, cfg)

        # 6) Build prompts
        print("  Assembling prompts ...")
        train = prepare_prompted_datasets(train, cfg)
        if val  is not None: val  = prepare_prompted_datasets(val,  cfg)
        if test is not None: test = prepare_prompted_datasets(test, cfg)

        # 7) Train/load generator and verifier
        generator_save_dir = os.path.join(cfg.out_dir, "generator")
        verifier_save_dir  = os.path.join(cfg.out_dir, "verifier")
        generator_exists = os.path.exists(os.path.join(generator_save_dir, "config.json"))
        verifier_exists  = os.path.exists(os.path.join(verifier_save_dir, "config.json"))

        gen_model = gen_tok = ver_model = ver_tok = None

        if cfg.eval_only:
            gen_path = cfg.generator_ckpt or (generator_save_dir if generator_exists else cfg.generator_model)
            gen_model, gen_tok = load_generator_from_ckpt(gen_path, cfg)
            if needs_verifier:
                ver_path = cfg.verifier_ckpt or (verifier_save_dir if verifier_exists else "")
                if ver_path:
                    ver_model, ver_tok = load_verifier_from_ckpt(ver_path, cfg)
                else:
                    print("  No verifier checkpoint supplied/found; factual verifier metrics will be skipped.")
        else:
            if generator_exists:
                print(f"  Existing generator found → {generator_save_dir} (skipping training)")
                gen_model, gen_tok = load_generator_from_ckpt(generator_save_dir, cfg)
            else:
                print("  Training generator ...")
                gen_model, gen_tok = train_generator(train, val if val is not None else train, cfg)

            if needs_verifier:
                if verifier_exists:
                    print(f"  Existing verifier found → {verifier_save_dir} (skipping training)")
                    ver_model, ver_tok = load_verifier_from_ckpt(verifier_save_dir, cfg)
                else:
                    if gen_model is not None and torch.cuda.is_available() and not is_distributed():
                        print("  Freeing generator GPU memory before verifier training ...")
                        try:
                            gen_model = gen_model.to("cpu")
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                    print("  Training verifier ...")
                    vt = build_verifier_dataset(train)
                    vv = (build_verifier_dataset(val)
                          if val is not None
                          else vt.train_test_split(test_size=0.1, seed=cfg.seed)["test"])
                    ver_model, ver_tok = train_verifier(vt, vv, cfg)
                    # Reload generator after verifier training for evaluation.
                    if gen_model is None or first_model_device(gen_model).type == "cpu":
                        gen_model, gen_tok = load_generator_from_ckpt(generator_save_dir, cfg)
            else:
                print("  MCQ mode: verifier training skipped (use --train_verifier_for_mcq to enable).")

            run_contrastive = not (is_mcq and cfg.skip_contrastive_for_mcq)
            if run_contrastive:
                print("  Contrastive fine-tuning ...")
                contrastive_finetune_generator_encoder(gen_model, gen_tok, train, cfg)
            else:
                print("  MCQ mode: contrastive fine-tuning skipped (use --run_contrastive_for_mcq to enable).")

        # 8) Evaluate
        eval_split = test if test is not None else val
        if eval_split is None:
            print("  WARNING: no test/validation split found. Skipping evaluation.")
            cfg.per_type_index_dir = idx_dir_orig
            return {}

        print(f"  Evaluating on {len(eval_split)} samples ...")
        results = evaluate(eval_split, gen_model, gen_tok, ver_model, ver_tok, cfg, val_ds=val)

        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved → {res_path}")

        cfg.per_type_index_dir = idx_dir_orig
        return results

    finally:
        cfg.out_dir = root_out_dir

# =============================================================================
# MAIN
# =============================================================================

def main():
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()
    cfg  = Config()

    # Override cfg from CLI
    cfg.dataset_name      = args.dataset
    cfg.eval_only         = args.eval_only
    cfg.generator_ckpt    = args.generator_ckpt
    cfg.verifier_ckpt     = args.verifier_ckpt
    cfg.run_all_datasets  = args.run_all_datasets
    cfg.hf_cache_dir      = args.hf_cache_dir

    for attr in [
        "out_dir", "medquad_dir", "icliniq_dir", "generator_model", "summarizer_model",
        "verifier_model", "embedder_model", "per_type_index_dir", "epochs", "batch_size",
        "gradient_accumulation_steps", "lr", "max_input_tokens", "max_summary_tokens",
        "n_best", "num_workers", "eval_steps", "logging_steps", "lora_r", "lora_alpha",
        "lora_dropout", "lora_target_modules", "mcq_decoding", "mcq_max_new_tokens",
        "bootstrap_samples", "ci_level",
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)

    cfg.gradient_checkpointing = not args.no_gradient_checkpointing
    cfg.use_flash_attention_2  = args.use_flash_attention_2
    cfg.use_lora               = args.use_lora
    cfg.load_in_4bit           = args.load_in_4bit
    cfg.train_verifier_for_mcq = args.train_verifier_for_mcq
    cfg.skip_contrastive_for_mcq = not args.run_contrastive_for_mcq

    if hasattr(args, "train_samples"):
        cfg.subset_train_size = args.train_samples if args.train_samples > 0 else 999999
        cfg.subset_val_size   = args.val_samples   if args.val_samples   > 0 else 999999
        cfg.subset_test_size  = args.test_samples  if args.test_samples  > 0 else 999999
        cfg.use_subset        = True

    if torch.cuda.is_available() and is_distributed():
        cfg.device = f"cuda:{local_rank()}"
    elif torch.cuda.is_available():
        cfg.device = "cuda"
    else:
        cfg.device = "cpu"

    set_seed(cfg.seed)

    print("\n" + "="*60)
    print("  MedLM Pipeline — Multi-Dataset Evaluation")
    print("="*60)
    print(f"  Mode       : {'eval_only' if cfg.eval_only else 'train+eval'}")
    print(f"  Datasets   : {list(DATASET_REGISTRY.keys())}")
    print(f"  Device     : {cfg.device}")
    print(f"  Distributed: {is_distributed()} (WORLD_SIZE={os.environ.get('WORLD_SIZE','1')})")
    print(f"  MCQ decode : {cfg.mcq_decoding}")
    print("="*60)

    all_results: Dict[str, Dict] = {}

    if cfg.run_all_datasets:
        print("\n  Running ALL datasets sequentially ...")
        for dset in DATASET_REGISTRY:
            res = run_one_dataset(dset, cfg)
            if res:
                all_results[dset] = res
    else:
        res = run_one_dataset(cfg.dataset_name, cfg)
        if res:
            all_results[cfg.dataset_name] = res

    if len(all_results) > 1:
        save_multi_dataset_comparison(all_results, cfg)

    if all_results:
        print("\n\n" + "="*60)
        print("  FINAL MULTI-DATASET SUMMARY")
        print("="*60)
        cols = ["ROUGE1", "METEOR", "BERTScore", "Accuracy"]
        hdr  = f"  {'Dataset':<16}" + "".join(f"{c:>14}" for c in cols)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for dset, res in all_results.items():
            vals = ""
            for c in cols:
                v = res.get(c, None)
                vals += f"{v:>14.4f}" if isinstance(v, (int, float, np.floating)) else f"{'N/A':>14}"
            print(f"  {dset:<16}{vals}")

        summary_path = os.path.join(cfg.out_dir, "summary_all_datasets.json")
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Master summary → {summary_path}")

if __name__ == "__main__":
    main()