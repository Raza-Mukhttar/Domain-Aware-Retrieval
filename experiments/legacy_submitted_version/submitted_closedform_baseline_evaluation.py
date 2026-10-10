# file: baseline_mcq_experiments.py
"""
MCQ Baseline Experiment Suite — Closed Datasets
=================================================
Fine-tunes and evaluates medical LLM baselines on multiple-choice benchmarks.
Primary metric: Accuracy (letter prediction A/B/C/D).
Secondary metrics: ROUGE-1, METEOR, BERTScore on generated text vs. gold answer.

Datasets supported (select with --dataset):
--------------------------------------------
  medqa       GBaker/MedQA-USMLE-4-options        4-choice USMLE
  medmcqa     openlifescienceai/medmcqa            4-choice Indian PG entrance
  mmlu_med    cais/mmlu  (6 medical subjects)      4-choice MMLU medical
  pubmedqa    reginaboateng/cleaned_pubmedqa        3-choice Yes/No/Maybe

Baselines supported (select with --baseline_model):
----------------------------------------------------
  GENERATIVE MEDICAL LLMs (causal)
  meditron7b       epfl-llm/meditron-7b
  medalpaca7b      medalpaca/medalpaca-7b
  pmc_llama13b     axiong/PMC_LLaMA_13B
  biomedgpt        PharMolix/BioMedGPT-LM-7B
  llama3_med       meta-llama/Meta-Llama-3-8B-Instruct
  biogpt           microsoft/biogpt
  biomedlm         stanford-crfm/BioMedLM

  RECENT 2024-2025 BASELINES
  openbiollm8b     aaditya/Llama3-OpenBioLLM-8B
  med42v2_8b       m42-health/Llama3-Med42-8B
  biomistral7b     BioMistral/BioMistral-7B
  meditron3_7b     OpenMeditron/Meditron3-Qwen2.5-7B

Usage examples
--------------
  # Single model, single dataset
  python baseline_mcq_experiments.py --baseline_model meditron7b --dataset medqa
  python baseline_mcq_experiments.py --baseline_model biomistral7b --dataset medmcqa
  python baseline_mcq_experiments.py --baseline_model openbiollm8b --dataset mmlu_med
  python baseline_mcq_experiments.py --baseline_model llama3_med --dataset pubmedqa

  # Zero-shot only (skip fine-tuning — useful for large models)
  python baseline_mcq_experiments.py --baseline_model meditron7b --dataset medqa --zero_shot

  # Custom sample sizes
  python baseline_mcq_experiments.py --baseline_model meditron7b --dataset medqa \
      --train_samples 500 --val_samples 200 --test_samples 300

  # Full dataset
  python baseline_mcq_experiments.py --baseline_model biogpt --dataset medmcqa \
      --train_samples -1 --val_samples -1 --test_samples -1

  # Local/offline model path
  python baseline_mcq_experiments.py --baseline_model llama3_med --dataset medqa \
      --local_model_path SET_THIS_PATH

  # Run ALL baselines on one dataset (saves comparison CSV + plots)
  python baseline_mcq_experiments.py --run_all --dataset medqa

  # Run ALL baselines across ALL datasets
  python baseline_mcq_experiments.py --run_all --run_all_datasets

  # Force offline (no HuggingFace downloads)
  TRANSFORMERS_OFFLINE=1 python baseline_mcq_experiments.py \
      --baseline_model biogpt --dataset medqa --offline

Notes
-----
- Fine-tuning uses the same hyperparameters as baseline_experiments.py
  (open-dataset script) for fair comparison.
- For MCQ fine-tuning, the model is trained to predict the gold option letter
  followed by the answer text. At evaluation, only the first predicted letter
  is extracted for accuracy computation.
- Models with use_slow_tok=True (medalpaca7b, pmc_llama13b) use
  LlamaTokenizer(legacy=True) to avoid SentencePiece conversion errors.
- All BioGPT/BioMedLM fixes from baseline_experiments.py are preserved.
"""

import os
import json
import random
import re
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from evaluate import load as load_metric

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# BASELINE MODEL REGISTRY  (same as baseline_experiments.py + 2024-2025 models)
# =============================================================================

BASELINE_REGISTRY: Dict[str, Dict] = {

    # ── Generative causal LLMs (carried from open-dataset script) ─────────
    "meditron7b": {
        "hf_model":             "epfl-llm/meditron-7b",
        "family":               "causal_lm",
        "description":          "Meditron-7B (EPFL, LLaMA-2 base)",
        "paper":                "Chen et al., 2023 - arxiv:2311.16079",
        "max_new_tokens":       8,     # MCQ: only need one letter
        "max_input_tokens":     512,
        "dtype":                "float16",
        "large_model":          True,
        "use_slow_tok":         False,
    },
    "medalpaca7b": {
        "hf_model":             "medalpaca/medalpaca-7b",
        "family":               "causal_lm",
        "description":          "MedAlpaca-7B (medical instruction tuning)",
        "paper":                "Han et al., 2023 - arxiv:2304.08247",
        "max_new_tokens":       8,
        "max_input_tokens":     256,
        "dtype":                "float16",
        "large_model":          True,
        "use_slow_tok":         True,   # SentencePiece-only
    },
    "pmc_llama13b": {
        "hf_model":             "axiong/PMC_LLaMA_13B",
        "family":               "causal_lm",
        "description":          "PMC-LLaMA-13B (biomedical literature)",
        "paper":                "Wu et al., 2023 - arxiv:2304.14454",
        "max_new_tokens":       8,
        "max_input_tokens":     256,
        "dtype":                "float16",
        "large_model":          True,
        "use_slow_tok":         True,   # SentencePiece-only
    },
    "biomedgpt": {
        "hf_model":             "PharMolix/BioMedGPT-LM-7B",
        "family":               "causal_lm",
        "description":          "BioMedGPT-LM-7B (pharma/bio domain)",
        "paper":                "Luo et al., 2023 - arxiv:2308.09442",
        "max_new_tokens":       8,
        "max_input_tokens":     512,
        "dtype":                "float16",
        "large_model":          True,
        "use_slow_tok":         False,
    },
    "llama3_med": {
        "hf_model":             "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "family":               "causal_lm",
        "description":          "LLaMA-3-8B-Instruct (vanilla reference)",
        "paper":                "Meta AI, 2024",
        "max_new_tokens":       8,
        "max_input_tokens":     512,
        "dtype":                "bfloat16",
        "large_model":          True,
        "use_slow_tok":         False,
        "use_chat_template":    True,   # LLaMA-3 chat format
    },
    "flan_t5_med": {
        "hf_model":             "google/flan-t5-large",
        "family":               "seq2seq",
        "description":          "BioGPT (Microsoft, PubMed abstracts)",
        "paper":                "Luo et al., 2022 - Briefings in Bioinformatics",
        "max_new_tokens":       8,
        "max_input_tokens":     300,    # BioGPT 1024 pos limit
        "dtype":                "float32",
        "large_model":          False,  # no grad ckpt
        "use_slow_tok":         False,
    },
    "biomedlm": {
        "hf_model":             "stanford-crfm/BioMedLM",
        "family":               "causal_lm",
        "description":          "BioMedLM-2.7B (Stanford CRFM)",
        "paper":                "Venigalla et al., 2022 - Stanford HAI",
        "max_new_tokens":       8,
        "max_input_tokens":     256,    # GPT-2 arch, 1024 pos limit
        "dtype":                "float32",
        "large_model":          False,
        "use_slow_tok":         False,
        "batch_override":       2,
        "grad_accum_override":  16,
    },

    # ── Recent 2024-2025 medical LLM baselines ────────────────────────────
    "openbiollm8b": {
        "hf_model":             "aaditya/Llama3-OpenBioLLM-8B",
        "family":               "causal_lm",
        "description":          "OpenBioLLM-8B (LLaMA-3, DPO medical)",
        "paper":                "Pal & Sankarasubbu, 2024 - HuggingFace",
        "max_new_tokens":       8,
        "max_input_tokens":     512,
        "dtype":                "bfloat16",
        "large_model":          True,
        "use_slow_tok":         False,
        "use_chat_template":    True,
    },
    "med42v2_8b": {
        "hf_model":             "m42-health/Llama3-Med42-8B",
        "family":               "causal_lm",
        "description":          "Med42-v2-8B (M42 Health, LLaMA-3)",
        "paper":                "Christophe et al., 2024 - arxiv:2408.06142",
        "max_new_tokens":       8,
        "max_input_tokens":     512,
        "dtype":                "bfloat16",
        "large_model":          True,
        "use_slow_tok":         False,
        "use_chat_template":    True,
    },
    "biomistral7b": {
        "hf_model":             "BioMistral/BioMistral-7B",
        "family":               "causal_lm",
        "description":          "BioMistral-7B (Mistral base, PubMed Central)",
        "paper":                "Labrak et al., 2024 - arxiv:2402.10373",
        "max_new_tokens":       8,
        "max_input_tokens":     512,
        "dtype":                "float16",
        "large_model":          True,
        "use_slow_tok":         False,
    },
    "meditron3_7b": {
        "hf_model":             "OpenMeditron/Meditron3-Qwen2.5-7B",
        "family":               "causal_lm",
        "description":          "Meditron3-7B (EPFL, Qwen2.5 base)",
        "paper":                "EPFL OpenMeditron, 2024",
        "max_new_tokens":       8,
        "max_input_tokens":     512,
        "dtype":                "bfloat16",
        "large_model":          True,
        "use_slow_tok":         False,
        "use_chat_template":    True,
    },
}


# =============================================================================
# MCQ DATASET REGISTRY
# =============================================================================

MCQ_DATASET_REGISTRY: Dict[str, Dict] = {
    "medqa": {
        "hf_id":         "GBaker/MedQA-USMLE-4-options",
        "hf_config":     None,
        "description":   "MedQA-USMLE — 4-choice US medical licensing exam",
        "paper":         "Jin et al., 2021 - Applied Sciences",
        "num_choices":   4,
        "choice_labels": ["A", "B", "C", "D"],
    },
    "medmcqa": {
        "hf_id":         "openlifescienceai/medmcqa",
        "hf_config":     None,
        "description":   "MedMCQA — 4-choice Indian medical PG entrance",
        "paper":         "Pal et al., 2022 - CHIL",
        "num_choices":   4,
        "choice_labels": ["A", "B", "C", "D"],
    },
    "mmlu_med": {
        "hf_id":         "cais/mmlu",
        "hf_config":     None,
        "description":   "MMLU Medical — 6 medical subjects",
        "paper":         "Hendrycks et al., 2020 - arxiv:2009.03300",
        "num_choices":   4,
        "choice_labels": ["A", "B", "C", "D"],
        "subjects": [
            "clinical_knowledge", "medical_genetics",
            "anatomy", "professional_medicine",
            "college_biology", "college_medicine",
        ],
    },
    "pubmedqa": {
        "hf_id":         "reginaboateng/cleaned_pubmedqa",
        "hf_config":     None,
        "description":   "PubMedQA — Yes/No/Maybe biomedical QA",
        "paper":         "Jin et al., 2019 - EMNLP",
        "num_choices":   3,
        "choice_labels": ["A", "B", "C"],
    },
}


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MCQConfig:
    # Dataset
    dataset_name:      str   = "medqa"
    out_dir:           str   = "SET_THIS_PATH"
    hf_cache_dir:      Optional[str] = None

    # Model
    baseline_model:    str   = "meditron7b"
    local_model_path:  str   = ""
    run_all:           bool  = False
    run_all_datasets:  bool  = False
    zero_shot:         bool  = False   # skip fine-tuning, evaluate directly

    # Sample sizes
    train_samples:     int   = 500     # -1 = full split
    val_samples:       int   = 200
    test_samples:      int   = 300

    # Training hyperparameters (same as open-dataset baseline script)
    max_input_tokens:  int   = 512
    batch_size:        int   = 8
    grad_accum:        int   = 4
    epochs:            int   = 3
    lr:                float = 5e-6
    warmup_ratio:      float = 0.03
    max_grad_norm:     float = 1.0
    seed:              int   = 42

    device:            str   = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCQ baseline experiment suite — MedQA, MedMCQA, MMLU-Med, PubMedQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline_model", type=str, default="meditron7b",
                   choices=list(BASELINE_REGISTRY.keys()),
                   help="Baseline model to fine-tune and evaluate.")
    p.add_argument("--dataset", type=str, default="medqa",
                   choices=list(MCQ_DATASET_REGISTRY.keys()),
                   help="MCQ dataset to evaluate on.")
    p.add_argument("--train_samples", type=int, default=500,
                   help="Training samples. -1 = full split.")
    p.add_argument("--val_samples",   type=int, default=200,
                   help="Validation samples. -1 = full split.")
    p.add_argument("--test_samples",  type=int, default=300,
                   help="Test samples. -1 = full split.")
    p.add_argument("--epochs",    type=int,   default=3)
    p.add_argument("--batch_size",type=int,   default=8)
    p.add_argument("--lr",        type=float, default=5e-6)
    p.add_argument("--zero_shot", action="store_true", default=False,
                   help="Skip fine-tuning; evaluate zero-shot directly.")
    p.add_argument("--local_model_path", type=str, default="",
                   help="Local path overriding HuggingFace ID.")
    p.add_argument("--run_all", action="store_true", default=False,
                   help="Run ALL baselines on chosen dataset.")
    p.add_argument("--run_all_datasets", action="store_true", default=False,
                   help="Run ALL baselines across ALL datasets.")
    p.add_argument("--offline", action="store_true", default=False,
                   help="Set TRANSFORMERS_OFFLINE=1 and HF_DATASETS_OFFLINE=1.")
    p.add_argument("--out_dir", type=str,
                   default="SET_THIS_PATH")
    p.add_argument("--hf_cache_dir", type=str, default=None,
                   help="Custom HuggingFace dataset cache directory.")
    return p.parse_args()


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _norm(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(str(x) for x in v).strip()
    return str(v).strip()


def _apply_subset(ds: Dataset, max_n: int) -> Dataset:
    if max_n > 0 and len(ds) > max_n:
        return ds.select(range(max_n))
    return ds


def resolve_model_path(cfg: MCQConfig, key: str) -> str:
    if cfg.local_model_path:
        print(f"  Using local path: {cfg.local_model_path}")
        return cfg.local_model_path
    return BASELINE_REGISTRY[key]["hf_model"]


def get_dtype(key: str) -> torch.dtype:
    d = BASELINE_REGISTRY[key].get("dtype", "float32")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(d, torch.float32)


def get_model_max_input(key: str, cfg: MCQConfig) -> int:
    return BASELINE_REGISTRY[key].get("max_input_tokens") or cfg.max_input_tokens


# =============================================================================
# MCQ DATASET LOADERS
# =============================================================================

def _parse_medqa_row(row: Dict) -> Dict:
    """GBaker/MedQA-USMLE-4-options format."""
    question    = _norm(row.get("question", ""))
    raw_options = row.get("options", {})
    raw_answer  = row.get("answer_idx", 0)

    if isinstance(raw_options, dict):
        ordered_keys = sorted(raw_options.keys())
        options = [str(raw_options[k]).strip() for k in ordered_keys]
    elif isinstance(raw_options, list):
        options = [str(x).strip() for x in raw_options]
    else:
        options = []

    if isinstance(raw_answer, int):
        answer_idx = raw_answer
    elif isinstance(raw_answer, str) and raw_answer.strip().isdigit():
        answer_idx = int(raw_answer.strip())
    elif isinstance(raw_answer, str) and len(raw_answer.strip()) == 1:
        answer_idx = ord(raw_answer.strip().upper()) - 65
    else:
        answer_idx = 0

    answer_idx  = max(0, min(len(options) - 1, answer_idx)) if options else 0
    gold_letter = chr(65 + answer_idx)
    opt_str     = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(options))

    return {
        "question":     question,
        "answer":       options[answer_idx] if options else "",
        "gold_letter":  gold_letter,
        "options_text": opt_str,
        "options_list": options,
    }


def _parse_medmcqa_row(row: Dict) -> Dict:
    """openlifescienceai/medmcqa format."""
    options = [
        _norm(row.get("opa", "")),
        _norm(row.get("opb", "")),
        _norm(row.get("opc", "")),
        _norm(row.get("opd", "")),
    ]
    cop        = max(0, min(3, int(row.get("cop", 0))))
    gold_letter = chr(65 + cop)
    opt_str    = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(options))
    return {
        "question":     _norm(row.get("question", "")),
        "answer":       options[cop],
        "gold_letter":  gold_letter,
        "options_text": opt_str,
        "options_list": options,
        "explanation":  _norm(row.get("exp", "")),
    }


def _parse_mmlu_row(row: Dict) -> Dict:
    """cais/mmlu format."""
    options  = list(row.get("choices", []))
    gold_idx = max(0, min(3, int(row.get("answer", 0))))
    opt_str  = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(options))
    return {
        "question":     _norm(row.get("question", "")),
        "answer":       options[gold_idx] if options else "",
        "gold_letter":  chr(65 + gold_idx),
        "options_text": opt_str,
        "options_list": options,
    }


def _parse_pubmedqa_row(row: Dict) -> Dict:
    """reginaboateng/cleaned_pubmedqa format."""
    question = _norm(row.get("question", ""))
    context  = _norm(row.get("context", ""))
    gold_txt = _norm(row.get("final_decision", "yes")).lower()
    ymap     = {"yes": "A", "no": "B", "maybe": "C"}
    gold_letter = ymap.get(gold_txt, "A")
    choices  = ["yes", "no", "maybe"]
    opt_str  = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    combined = f"{question}\n\nContext: {context}" if context else question
    return {
        "question":     combined,
        "answer":       gold_txt,
        "gold_letter":  gold_letter,
        "options_text": opt_str,
        "options_list": choices,
    }


def load_mcq_dataset(cfg: MCQConfig) -> DatasetDict:
    """Load and parse the selected MCQ dataset from HuggingFace."""
    name  = cfg.dataset_name
    entry = MCQ_DATASET_REGISTRY[name]
    print(f"\n  Loading {entry['description']} ...")

    size_map = {
        "train":      cfg.train_samples,
        "validation": cfg.val_samples,
        "test":       cfg.test_samples,
    }
    splits: Dict[str, Dataset] = {}

    # ── MedQA ─────────────────────────────────────────────────────────────
    if name == "medqa":
        raw = hf_load_dataset(
            entry["hf_id"], cache_dir=cfg.hf_cache_dir
        )
        # Build train/val from raw["train"] (90:10 split)
        all_rows  = [_parse_medqa_row(r) for r in raw["train"]]
        full_ds   = Dataset.from_list(all_rows)
        tv_split  = full_ds.train_test_split(test_size=0.1, seed=cfg.seed)
        splits["train"]      = tv_split["train"]
        splits["validation"] = tv_split["test"]
        if "test" in raw:
            splits["test"] = Dataset.from_list(
                [_parse_medqa_row(r) for r in raw["test"]]
            )

    # ── MedMCQA ───────────────────────────────────────────────────────────
    elif name == "medmcqa":
        raw = hf_load_dataset(
            entry["hf_id"], cache_dir=cfg.hf_cache_dir, trust_remote_code=True
        )
        for split in ["train", "validation", "test"]:
            if split in raw:
                splits[split] = Dataset.from_list(
                    [_parse_medmcqa_row(r) for r in raw[split]]
                )

    # ── MMLU Medical ──────────────────────────────────────────────────────
    elif name == "mmlu_med":
        subjects = entry["subjects"]
        all_rows: Dict[str, List] = {"train": [], "validation": [], "test": []}
        for subj in subjects:
            try:
                raw = hf_load_dataset(
                    entry["hf_id"], subj,
                    cache_dir=cfg.hf_cache_dir, trust_remote_code=True
                )
            except Exception as e:
                print(f"  WARNING: skipping subject '{subj}': {e}")
                continue
            # MMLU splits: dev → train proxy, validation, test
            for hf_split, canon in [
                ("dev",        "train"),
                ("validation", "validation"),
                ("test",       "test"),
            ]:
                if hf_split in raw:
                    all_rows[canon].extend(
                        [_parse_mmlu_row(r) for r in raw[hf_split]]
                    )
        for canon, rows in all_rows.items():
            if rows:
                random.shuffle(rows)
                splits[canon] = Dataset.from_list(rows)

    # ── PubMedQA ──────────────────────────────────────────────────────────
    elif name == "pubmedqa":
        raw = hf_load_dataset(
            entry["hf_id"], cache_dir=cfg.hf_cache_dir, trust_remote_code=True
        )
        all_rows = [_parse_pubmedqa_row(r) for r in raw["train"]]
        full_ds  = Dataset.from_list(all_rows)
        tv_split = full_ds.train_test_split(test_size=0.2, seed=cfg.seed)
        splits["train"] = tv_split["train"]
        splits["test"]  = tv_split["test"]
        if "validation" in raw:
            splits["validation"] = Dataset.from_list(
                [_parse_pubmedqa_row(r) for r in raw["validation"]]
            )

    # Apply subset
    for split_name in list(splits.keys()):
        max_n = size_map.get(split_name, -1)
        if max_n > 0:
            splits[split_name] = _apply_subset(splits[split_name], max_n)
        print(f"  [{split_name}] {len(splits[split_name])} samples")

    return DatasetDict(splits)


# =============================================================================
# MCQ PROMPT BUILDERS
# =============================================================================

def build_mcq_prompt_zero_shot(question: str, options_text: str,
                                key: str) -> str:
    """
    Build a zero-shot MCQ prompt. Uses chat template format for models
    that have use_chat_template=True (LLaMA-3, OpenBioLLM, Med42, Meditron3).
    Uses a plain instruction for all others.
    """
    use_chat = BASELINE_REGISTRY[key].get("use_chat_template", False)

    if use_chat:
        return (
            "<|im_start|>system\n"
            "You are a medical expert taking a multiple-choice examination. "
            "Respond with ONLY the single letter (A, B, C, or D) of the correct "
            "answer. Do not explain. Output exactly one letter.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            "Answer (single letter only):\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        return (
            "You are a medical expert taking a multiple-choice examination.\n"
            "Read the question and options carefully.\n"
            "Respond with ONLY the single letter of the correct answer (A, B, C, or D).\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            "Answer:"
        )


def build_mcq_train_target(gold_letter: str, answer_text: str) -> str:
    """Build the fine-tuning target: letter + answer text."""
    return f"{gold_letter}. {answer_text}"


# =============================================================================
# MODEL LOADING  (same pattern as baseline_experiments.py)
# =============================================================================

def load_model_and_tokenizer(key: str, cfg: MCQConfig) -> Tuple:
    """Load model and tokenizer. Returns (model, tokenizer, family)."""
    entry      = BASELINE_REGISTRY[key]
    family     = entry["family"]
    model_path = resolve_model_path(cfg, key)
    dtype      = get_dtype(key)

    print(f"  Loading tokenizer: {model_path}")

    # checkpoints that ship only tokenizer.model, with no tokenizer.json
    if entry.get("use_slow_tok", False):
        from transformers import LlamaTokenizer
        print("  INFO: using LlamaTokenizer(legacy=True) for SentencePiece model")
        tok = LlamaTokenizer.from_pretrained(
            model_path, legacy=True, trust_remote_code=True
        )
    else:
        tok = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"
    if tok.eos_token is None:
        tok.eos_token = tok.pad_token

    print(f"  Loading model ({family}): {model_path}")

    if family == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    elif family == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown family: {family}")

    print("  Model loaded OK")
    return model, tok, family


# =============================================================================
# TOKENISATION FOR FINE-TUNING
# =============================================================================

def tokenize_mcq_causal(examples: Dict, tok, cfg: MCQConfig, key: str) -> Dict:
    """
    Causal LM fine-tuning: concatenate prompt + target, mask prompt in labels.
    Target = gold_letter + ". " + answer_text (e.g. "A. Cricoid cartilage")
    """
    model_max_in = get_model_max_input(key, cfg)

    prompts = examples["prompt"]
    targets = examples["target"]

    p_enc = tok(list(prompts), truncation=True, padding=False, add_special_tokens=False)
    t_enc = tok(list(targets), truncation=True, padding=False, add_special_tokens=False)

    all_ids, all_labels, all_masks = [], [], []
    max_total = model_max_in + 32   # target is short (letter + answer phrase)

    for p_ids, t_ids in zip(p_enc["input_ids"], t_enc["input_ids"]):
        full = p_ids + [tok.eos_token_id] + t_ids
        if len(full) > max_total:
            full  = p_ids[:model_max_in] + [tok.eos_token_id] + t_ids[:32]
        label = ([-100] * (len(p_ids) + 1) + t_ids)[: len(full)]
        all_ids.append(full)
        all_labels.append(label)
        all_masks.append([1] * len(full))

    bmax   = max(len(x) for x in all_ids)
    pad_id = tok.pad_token_id or 0

    padded_ids, padded_lbl, padded_msk = [], [], []
    for ids, lbl, msk in zip(all_ids, all_labels, all_masks):
        plen = bmax - len(ids)
        padded_ids.append(ids + [pad_id] * plen)
        padded_lbl.append(lbl + [-100]   * plen)
        padded_msk.append(msk + [0]      * plen)


    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else 50257
    clamped_lbl = []
    for lbl in padded_lbl:
        clamped_lbl.append([
            t if (t == -100 or 0 <= t < vocab_size) else -100
            for t in lbl
        ])
    return {
        "input_ids":      padded_ids,
        "labels":         clamped_lbl,
        "attention_mask": padded_msk,
    }


def tokenize_mcq_seq2seq(examples: Dict, tok, cfg: MCQConfig, key: str) -> Dict:
    """Seq2Seq fine-tuning: source = prompt, target = gold_letter + answer."""
    model_max_in = get_model_max_input(key, cfg)

    # Encode source prompts
    model_inputs = tok(
        list(examples["prompt"]),
        max_length=model_max_in,
        truncation=True,
        padding="max_length",
    )

    # Encode targets using modern HF API
    labels = tok(
        text_target=list(examples["target"]),
        max_length=32,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token ids with -100 for loss masking
    model_inputs["labels"] = [
        [
            (t if t != tok.pad_token_id else -100)
            for t in l
        ]
        for l in labels["input_ids"]
    ]

    return model_inputs
    # model_inputs = tok(
    #     list(examples["prompt"]),
    #     max_length=model_max_in, truncation=True, padding="max_length",
    # )
    # with tok.as_target_tokenizer():
    #     labels = tok(
    #         list(examples["target"]),
    #         max_length=32, truncation=True, padding="max_length",
    #     )
    # model_inputs["labels"] = [
    #     [(t if t != tok.pad_token_id else -100) for t in l]
    #     for l in labels["input_ids"]
    # ]
    # return model_inputs


def prepare_mcq_datasets(ds: Dataset, key: str) -> Dataset:
    """Add prompt and target columns for fine-tuning."""
    prompts = [
        build_mcq_prompt_zero_shot(ex["question"], ex["options_text"], key)
        for ex in ds
    ]
    targets = [
        build_mcq_train_target(ex["gold_letter"], ex["answer"])
        for ex in ds
    ]
    ds = ds.add_column("prompt", prompts)
    ds = ds.add_column("target", targets)
    return ds


# =============================================================================
# TRAINING
# =============================================================================

def train_mcq_baseline(
    model, tok, train_ds: Dataset, val_ds: Dataset,
    cfg: MCQConfig, key: str, family: str
) -> "Trainer":
    """Fine-tune model on MCQ data using HuggingFace Trainer."""
    entry    = BASELINE_REGISTRY[key]
    save_dir = os.path.join(cfg.out_dir, key, cfg.dataset_name, "checkpoint")
    os.makedirs(save_dir, exist_ok=True)

    eff_batch = entry.get("batch_override",    cfg.batch_size)
    eff_accum = entry.get("grad_accum_override", cfg.grad_accum)
    use_grad_ckpt = (family == "causal_lm") and entry.get("large_model", False)

    use_bf16 = torch.cuda.is_available() and entry.get("dtype") == "bfloat16"
    use_fp16 = (torch.cuda.is_available() and not use_bf16
                and entry.get("dtype") == "float16")

    print(f"\n  Fine-tuning MCQ: {key} | {cfg.dataset_name} | "
          f"batch={eff_batch} accum={eff_accum} grad_ckpt={use_grad_ckpt}")

    if family == "causal_lm":
        fn = lambda ex: tokenize_mcq_causal(ex, tok, cfg, key)
    else:
        fn = lambda ex: tokenize_mcq_seq2seq(ex, tok, cfg, key)

    remove_cols = [c for c in train_ds.column_names
                   if c not in ["input_ids", "labels", "attention_mask"]]
    train_tok = train_ds.map(fn, batched=True, remove_columns=remove_cols)
    val_tok   = val_ds.map(fn,   batched=True, remove_columns=remove_cols)
    train_tok.set_format("torch")
    val_tok.set_format("torch")

    args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=max(1, eff_batch // 2),
        per_device_eval_batch_size=max(1,  eff_batch // 2),
        gradient_accumulation_steps=eff_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=1e-6,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=0.3,
        fp16=False,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=use_grad_ckpt,
        seed=cfg.seed,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        processing_class=tok,
    )
    trainer.train()

    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p       = None
        model.generation_config.do_sample   = False

    trainer.save_model(save_dir)
    print(f"  Checkpoint saved: {save_dir}")
    return trainer


# =============================================================================
# MCQ LETTER EXTRACTION  (robust, 6-strategy approach)
# =============================================================================

def extract_mcq_letter(text: str, options_list: List[str],
                        num_choices: int = 4) -> str:
    """
    Extract predicted option letter from generated text.
    Uses 6 strategies in priority order — same approach as main pipeline.
    """
    text  = text.strip() if text else ""
    valid = [chr(65 + i) for i in range(num_choices)]

    if not text:
        return valid[0]

    # Strategy 0: bare single letter
    stripped = text.strip().rstrip(".)\n\r ")
    if len(stripped) == 1 and stripped.upper() in valid:
        return stripped.upper()

    # Strategy 1: letter at very start
    m = re.match(r"^([A-D])\s*[\.\)\:\-\s]", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Strategy 2: explicit answer phrase
    for pat in [
        r"(?:the\s+)?answer\s+is\s+([A-D])",
        r"answer:\s*([A-D])",
        r"correct\s+(?:answer|option|choice)\s+is\s+([A-D])",
        r"option\s+([A-D])\s+is\s+correct",
        r"\bselect\s+([A-D])\b",
    ]:
        m = re.search(pat, text[:200], re.IGNORECASE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    # Strategy 3: letter + punctuation anywhere in first 120 chars
    for pat in [r"\b([A-D])[\.\)\:\-]", r"\(([A-D])\)"]:
        m = re.search(pat, text[:120], re.IGNORECASE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    # Strategy 4: substring match of option text (longest match wins)
    text_lower = text.lower()
    best_idx, best_len = -1, 0
    for i, opt in enumerate(options_list):
        if opt and opt.strip():
            opt_l = opt.lower().strip()
            if opt_l in text_lower and len(opt_l) > best_len:
                best_idx, best_len = i, len(opt_l)
    if best_idx >= 0:
        return valid[best_idx]

    # Strategy 5: Yes/No/Maybe (PubMedQA)
    ynm = {"yes": "A", "no": "B", "maybe": "C"}
    for word, letter in ynm.items():
        if re.search(r"\b" + word + r"\b", text_lower[:80]) and letter in valid:
            return letter

    # Strategy 6: word-overlap fallback (never random)
    if options_list:
        gen_words = set(text_lower.split())
        scores    = [
            len(set(opt.lower().split()) & gen_words) if opt else 0
            for opt in options_list
        ]
        return valid[max(range(len(scores)), key=lambda i: scores[i])]

    return valid[0]


# =============================================================================
# INFERENCE
# =============================================================================

def predict_mcq(
    question: str,
    options_text: str,
    options_list: List[str],
    model, tok, cfg: MCQConfig, key: str, family: str,
    num_choices: int = 4,
) -> Tuple[str, str]:
    """
    Generate answer for one MCQ question.
    Returns (generated_text, predicted_letter).
    """
    prompt       = build_mcq_prompt_zero_shot(question, options_text, key)
    model_max_in = get_model_max_input(key, cfg)
    max_new      = BASELINE_REGISTRY[key].get("max_new_tokens", 8)

    enc = tok(
        prompt, return_tensors="pt",
        truncation=True, max_length=model_max_in,
    ).to(cfg.device)

    # FIX: disable grad checkpointing + ensure use_cache for inference
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    with torch.no_grad():
        if family == "causal_lm":
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                num_beams=1,        # greedy — fastest and most reliable for MCQ
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
            prompt_len = enc["input_ids"].shape[1]
            generated  = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        else:
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                num_beams=1,
                do_sample=False,
            )
            generated = tok.decode(out[0], skip_special_tokens=True)

    generated  = generated.strip()
    pred_letter = extract_mcq_letter(generated, options_list, num_choices)
    return generated, pred_letter


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_mcq(
    test_ds: Dataset, model, tok, cfg: MCQConfig,
    key: str, family: str,
) -> Dict:
    """
    Evaluate on MCQ test split.
    Primary metric: Accuracy.
    Secondary: ROUGE-1, METEOR, BERTScore (generated text vs. gold answer text).
    """
    entry       = MCQ_DATASET_REGISTRY[cfg.dataset_name]
    num_choices = entry["num_choices"]

    rouge_m     = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    meteor_m    = load_metric("meteor")
    bertscore_m = load_metric("bertscore")

    hyps, refs  = [], []
    correct     = 0
    total       = len(test_ds)

    print(f"\n  Evaluating {key} on {cfg.dataset_name} ({total} samples) ...")

    for ex in tqdm(test_ds, desc=f"  Eval {key}"):
        generated, pred_letter = predict_mcq(
            ex["question"], ex["options_text"], ex["options_list"],
            model, tok, cfg, key, family, num_choices,
        )
        gold_letter = ex.get("gold_letter", "A")

        if pred_letter.upper() == gold_letter.upper():
            correct += 1

        hyps.append(generated)
        refs.append(ex.get("answer", ""))

    accuracy = correct / max(1, total)

    r1  = np.mean([
        rouge_m.score(r, p)["rouge1"].fmeasure
        for r, p in zip(refs, hyps)
    ])
    met = meteor_m.compute(predictions=hyps, references=refs)["meteor"]
    # bs  = bertscore_m.compute(predictions=hyps, references=refs, lang="en")
    # bsf = float(np.mean(bs["f1"]))

    results = {
        "Accuracy":  round(accuracy, 4),
        "ROUGE1":    round(r1,       4),
        "METEOR":    round(met,      4),
        # "BERTScore": round(bsf,      4),
        "Correct":   correct,
        "Total":     total,
    }

    print(f"\n  ===== {key} | {cfg.dataset_name} =====")
    for k, v in results.items():
        marker = " ← primary" if k == "Accuracy" else ""
        print(f"    {k:12s}: {v}{marker}")

    return results


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_curves(trainer, save_dir: str, label: str):
    logs = trainer.state.log_history
    df   = pd.DataFrame(logs)
    if df.empty:
        return
    os.makedirs(save_dir, exist_ok=True)

    if "loss" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ld = df.dropna(subset=["loss"])
        ax.plot(ld["step"], ld["loss"], label="train loss", color="#2563EB")
        if "eval_loss" in df.columns:
            ev = df.dropna(subset=["eval_loss"])
            ax.plot(ev["step"], ev["eval_loss"],
                    label="val loss", color="#DC2626", linestyle="--")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title(f"{label} — Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{label}_loss.png"), dpi=150)
        plt.close()
    print(f"  Training curves saved: {save_dir}")


# =============================================================================
# COMPARISON PLOTS AND CSV
# =============================================================================

def save_comparison(all_results: Dict[str, Dict], cfg: MCQConfig):
    """Save CSV, Markdown table, and accuracy bar chart."""
    comp_dir = os.path.join(cfg.out_dir, "comparison", cfg.dataset_name)
    os.makedirs(comp_dir, exist_ok=True)

    metrics = ["Accuracy", "ROUGE1", "METEOR", "BERTScore"]
    rows = []
    for bkey, res in all_results.items():
        row = {
            "Baseline":    bkey,
            "Description": BASELINE_REGISTRY.get(bkey, {}).get("description", bkey),
            "Dataset":     cfg.dataset_name,
            "ZeroShot":    cfg.zero_shot,
        }
        row.update({m: res.get(m, 0.0) for m in metrics})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)

    csv_path = os.path.join(comp_dir, "mcq_baseline_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV → {csv_path}")

    md_path = os.path.join(comp_dir, "mcq_baseline_comparison.md")
    with open(md_path, "w") as f:
        f.write(f"# MCQ Baseline Comparison — {cfg.dataset_name.upper()}\n\n")
        f.write(f"Zero-shot: {cfg.zero_shot}\n\n")
        f.write(df[["Baseline", "Description"] + metrics].to_markdown(index=False))
        f.write("\n")
    print(f"  Markdown → {md_path}")

    n_b    = len(all_results)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, n_b))
    sorted_items = sorted(all_results.items(),
                          key=lambda x: x[1].get("Accuracy", 0))

    # Figure 1: accuracy horizontal bar (primary metric)
    fig, ax = plt.subplots(figsize=(9, max(4, n_b * 0.55)))
    y_pos = np.arange(n_b)
    acc_vals = [v.get("Accuracy", 0) for _, v in sorted_items]
    ax.barh(y_pos, acc_vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([k for k, _ in sorted_items], fontsize=9)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"MCQ Accuracy — {cfg.dataset_name.upper()}"
                 + (" (Zero-shot)" if cfg.zero_shot else " (Fine-tuned)"))
    # Random chance line
    num_choices = MCQ_DATASET_REGISTRY[cfg.dataset_name]["num_choices"]
    ax.axvline(1.0 / num_choices, color="gray", linestyle="--",
               linewidth=0.8, label=f"Random chance (1/{num_choices})")
    ax.legend(fontsize=9)
    for i, (_, v) in enumerate(sorted_items):
        acc = v.get("Accuracy", 0)
        ax.text(acc + 0.005, i, f"{acc:.3f}", va="center", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "accuracy_ranked.png"), dpi=150)
    plt.close()

    # Figure 2: grouped bar — Accuracy + ROUGE1 + METEOR + BERTScore
    met4   = ["Accuracy", "ROUGE1", "METEOR", "BERTScore"]
    x      = np.arange(len(met4))
    w      = 0.8 / max(n_b, 1)
    colors2 = plt.cm.tab10(np.linspace(0, 1, n_b))
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    for i, (bkey, res) in enumerate(all_results.items()):
        vals   = [res.get(m, 0.0) for m in met4]
        offset = (i - n_b / 2 + 0.5) * w
        desc   = BASELINE_REGISTRY.get(bkey, {}).get("description", bkey)
        ax2.bar(x + offset, vals, width=w * 0.9, label=desc, color=colors2[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels(met4)
    ax2.set_ylabel("Score")
    ax2.set_title(f"All Metrics — {cfg.dataset_name.upper()}")
    ax2.legend(loc="upper right", fontsize=7, ncol=max(1, n_b // 4))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "all_metrics.png"), dpi=150)
    plt.close()

    print(f"  Figures → {comp_dir}")


# =============================================================================
# PER-BASELINE RUNNER
# =============================================================================

def run_one_baseline(key: str, cfg: MCQConfig) -> Optional[Dict]:
    """Full pipeline: load data → (fine-tune) → evaluate → save."""
    entry = BASELINE_REGISTRY[key]
    mode  = "zero-shot" if cfg.zero_shot else "fine-tuned"

    print(f"\n{'='*65}")
    print(f"  BASELINE : {key}  [{mode}]")
    print(f"  {entry['description']}")
    print(f"  {entry['paper']}")
    print(f"  Dataset  : {cfg.dataset_name}  |  "
          f"train={cfg.train_samples}  val={cfg.val_samples}  "
          f"test={cfg.test_samples}")
    print(f"{'='*65}")

    # Result cache
    tag      = "zeroshot" if cfg.zero_shot else "finetuned"
    res_dir  = os.path.join(cfg.out_dir, key, cfg.dataset_name)
    res_path = os.path.join(res_dir, f"results_{tag}.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            cached = json.load(f)
        r = cached["results"]
        print(f"  Cached results found — skipping.")
        print(f"  Accuracy={r.get('Accuracy',0):.4f}  "
              f"ROUGE1={r.get('ROUGE1',0):.4f}  "
              f"METEOR={r.get('METEOR',0):.4f}")
        return r
    os.makedirs(res_dir, exist_ok=True)

    # Free GPU memory from previous run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load dataset
    print("\n  Loading dataset ...")
    splits = load_mcq_dataset(cfg)
    if "test" not in splits:
        print(f"  ERROR: no test split available. Skipping {key}.")
        return None

    train_ds = splits.get("train")
    val_ds   = splits.get("validation", splits.get("train"))
    test_ds  = splits["test"]

    # Load model
    try:
        model, tok, family = load_model_and_tokenizer(key, cfg)
    except Exception as e:
        print(f"  ERROR loading model '{key}': {e}")
        return None

    # Fine-tune (unless zero_shot)
    if not cfg.zero_shot and train_ds is not None and val_ds is not None:
        print("  Preparing MCQ datasets ...")
        train_p = prepare_mcq_datasets(train_ds, key)
        val_p   = prepare_mcq_datasets(val_ds,   key)

        trainer = train_mcq_baseline(model, tok, train_p, val_p, cfg, key, family)
        curves_dir = os.path.join(res_dir, "curves")
        plot_training_curves(trainer, curves_dir, f"{key}_{cfg.dataset_name}_{tag}")
    elif cfg.zero_shot:
        print("  Zero-shot mode — skipping fine-tuning.")
    else:
        print("  WARNING: no train/val split — evaluating without fine-tuning.")

    # Evaluate
    results = evaluate_mcq(test_ds, model, tok, cfg, key, family)

    # Save
    payload = {
        "baseline":     key,
        "dataset":      cfg.dataset_name,
        "mode":         tag,
        "hf_model":     entry["hf_model"],
        "paper":        entry["paper"],
        "train_samples": len(train_ds) if train_ds else 0,
        "test_samples":  len(test_ds),
        "config": {
            "epochs":    cfg.epochs if not cfg.zero_shot else 0,
            "lr":        cfg.lr,
            "batch_size": entry.get("batch_override", cfg.batch_size),
            "zero_shot": cfg.zero_shot,
        },
        "results": results,
    }
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {res_path}")

    # Free GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    args = parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"]  = "1"
        print("  Offline mode enabled.")

    cfg = MCQConfig(
        dataset_name     = args.dataset,
        baseline_model   = args.baseline_model,
        local_model_path = args.local_model_path,
        run_all          = args.run_all,
        run_all_datasets = args.run_all_datasets,
        zero_shot        = args.zero_shot,
        train_samples    = args.train_samples,
        val_samples      = args.val_samples,
        test_samples     = args.test_samples,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        out_dir          = args.out_dir,
        hf_cache_dir     = args.hf_cache_dir,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    mode = "ZERO-SHOT" if cfg.zero_shot else "FINE-TUNED"

    print("\n" + "="*65)
    print(f"  MCQ Baseline Experiment Suite [{mode}]")
    print("="*65)
    print(f"  Dataset  : {cfg.dataset_name.upper()}")
    print(f"  Samples  : train={cfg.train_samples}  "
          f"val={cfg.val_samples}  test={cfg.test_samples}")
    print(f"  Epochs   : {cfg.epochs}   LR: {cfg.lr}   "
          f"Batch: {cfg.batch_size}")
    print(f"  Device   : {cfg.device}")
    print(f"  Out dir  : {cfg.out_dir}")
    print("="*65)

    # Determine which datasets to run
    datasets_to_run = (
        list(MCQ_DATASET_REGISTRY.keys())
        if cfg.run_all_datasets
        else [cfg.dataset_name]
    )
    # Determine which baselines to run
    baselines_to_run = (
        list(BASELINE_REGISTRY.keys())
        if cfg.run_all
        else [cfg.baseline_model]
    )

    grand_results: Dict[str, Dict[str, Dict]] = {}

    for dataset_name in datasets_to_run:
        cfg.dataset_name = dataset_name
        all_results: Dict[str, Dict] = {}

        print(f"\n\n{'#'*65}")
        print(f"  DATASET: {dataset_name.upper()}")
        print(f"{'#'*65}")

        for key in baselines_to_run:
            cfg.baseline_model   = key
            cfg.local_model_path = ""   # reset per model in sweep
            res = run_one_baseline(key, cfg)
            if res:
                all_results[key] = res

        if len(all_results) > 1:
            save_comparison(all_results, cfg)

        if all_results:
            grand_results[dataset_name] = all_results

    # Final summary table
    if grand_results:
        print("\n\n" + "="*65)
        print(f"  FINAL SUMMARY [{mode}]")
        print("="*65)

        for dset, res_dict in grand_results.items():
            print(f"\n  Dataset: {dset.upper()}")
            print(f"  {'Baseline':<20} {'Accuracy':>10} {'ROUGE1':>10} "
                  f"{'METEOR':>10} {'BERTScore':>11}")
            print("  " + "-" * 63)
            ranked = sorted(
                res_dict.items(),
                key=lambda x: x[1].get("Accuracy", 0),
                reverse=True,
            )
            for bkey, res in ranked:
                print(
                    f"  {bkey:<20} "
                    f"{res.get('Accuracy',0):>10.4f} "
                    f"{res.get('ROUGE1',0):>10.4f} "
                    f"{res.get('METEOR',0):>10.4f} "
                    f"{res.get('BERTScore',0):>11.4f}"
                )

        # Save master summary JSON
        tag = "zeroshot" if cfg.zero_shot else "finetuned"
        summary_path = os.path.join(cfg.out_dir, f"summary_mcq_{tag}.json")
        with open(summary_path, "w") as f:
            json.dump(grand_results, f, indent=2)
        print(f"\n  Master summary → {summary_path}")


if __name__ == "__main__":
    main()