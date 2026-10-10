"""
Baseline Experiment Suite for Medical QA
=========================================
Trains and evaluates recent domain-specific medical/biomedical LLM baselines
on MedQuAD and iCliniq datasets using the SAME configuration as the main
MedLM-LLaMA3 pipeline (same splits, same metrics, same preprocessing).

Baselines covered (select with --baseline_model):
--------------------------------------------------
  GENERATIVE MEDICAL LLMs
  meditron7b       epfl-llm/meditron-7b
  medalpaca7b      medalpaca/medalpaca-7b
  pmc_llama13b     axiong/PMC_LLaMA_13B
  biomedgpt        PharMolix/BioMedGPT-LM-7B
  llama3_med       meta-llama/Meta-Llama-3-8B-Instruct

  ENCODER-DECODER / SEQ2SEQ
  flan_t5_med      google/flan-t5-large
  biogpt           microsoft/biogpt
  biomedlm         stanford-crfm/BioMedLM

  ENCODER-ONLY (generative head)
  pubmedbert_qa    ncbi/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
  biolinkbert_qa   michiyasunaga/BioLinkBERT-large
  clinicalbert_qa  medicalai/ClinicalBERT

Usage examples
--------------
  python submitted_additional_medical_baselines.py --baseline_model meditron7b    --dataset medquad
  python submitted_additional_medical_baselines.py --baseline_model biogpt        --dataset icliniq
  python submitted_additional_medical_baselines.py --baseline_model biomedlm      --dataset icliniq
  python submitted_additional_medical_baselines.py --baseline_model medalpaca7b   --dataset icliniq
  python submitted_additional_medical_baselines.py --baseline_model pmc_llama13b  --dataset icliniq
  python submitted_additional_medical_baselines.py --baseline_model flan_t5_med   --dataset medquad

  Custom sample sizes:
  python submitted_additional_medical_baselines.py --baseline_model meditron7b    --dataset medquad \
      --train_samples 500 --val_samples 200 --test_samples 300

  Full dataset:
  python submitted_additional_medical_baselines.py --baseline_model biogpt        --dataset medquad \
      --train_samples -1 --val_samples -1 --test_samples -1

  Offline / local model:
  python submitted_additional_medical_baselines.py --baseline_model llama3_med    --dataset medquad \
      --local_model_path SET_THIS_PATH

  Full sweep:
  python submitted_additional_medical_baselines.py --run_all --dataset medquad
  python submitted_additional_medical_baselines.py --run_all --dataset icliniq
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
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
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
# BASELINE MODEL REGISTRY
# =============================================================================
# Registry fields
# ---------------
# hf_model            HuggingFace model ID (overridden by --local_model_path)
# family              causal_lm | seq2seq | encoder_gen
# description         human-readable label for plots/tables
# paper               citation string
# max_new_tokens      generation length cap
# max_input_tokens    per-model context limit (overrides cfg.max_input_tokens)
# dtype               float16 | bfloat16 | float32
# large_model         True  -> gradient checkpointing ON during training
#                     False -> gradient checkpointing OFF (small/GPT-2 arch)
# use_slow_tok        True  -> LlamaTokenizer(legacy=True) for checkpoints that
#                              ship only tokenizer.model, with no tokenizer.json
# batch_override      overrides cfg.batch_size for OOM-prone models
# grad_accum_override overrides cfg.grad_accum to compensate

BASELINE_REGISTRY: Dict[str, Dict] = {

    # ── Generative causal LLMs ────────────────────────────────────────────
    "meditron7b": {
        "hf_model":             "epfl-llm/meditron-7b",
        "family":               "causal_lm",
        "description":          "Meditron-7B (EPFL medical LLM, LLaMA-2 base)",
        "paper":                "Chen et al., 2023 - arxiv:2311.16079",
        "max_new_tokens":       200,
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
        "max_new_tokens":       150,
        "max_input_tokens":     256,   # LLaMA-1 max pos=2048; headroom
        "dtype":                "bfloat16",
        "large_model":          True,
        "use_slow_tok":         True,  # ships only tokenizer.model
    },
    "pmc_llama13b": {
        "hf_model":             "axiong/PMC_LLaMA_13B",
        "family":               "causal_lm",
        "description":          "PMC-LLaMA-13B (biomedical literature fine-tune)",
        "paper":                "Wu et al., 2023 - arxiv:2304.14454",
        "max_new_tokens":       150,
        "max_input_tokens":     256,   # LLaMA-1 limit
        "dtype":                "bfloat16",
        "batch_override": 1,
        "grad_accum_override": 4,
        "large_model":          True,
        "use_slow_tok":         True,  # ships only tokenizer.model
    },
    "biomedgpt": {
        "hf_model":             "PharMolix/BioMedGPT-LM-7B",
        "family":               "causal_lm",
        "description":          "BioMedGPT-LM-7B (pharma/bio domain GPT)",
        "paper":                "Luo et al., 2023 - arxiv:2308.09442",
        "max_new_tokens":       200,
        "max_input_tokens":     512,
        "dtype":                "float16",
        "large_model":          True,
        "use_slow_tok":         False,
    },
    "llama3_med": {
        "hf_model":             "meta-llama/Meta-Llama-3-8B-Instruct",
        "family":               "causal_lm",
        "description":          "LLaMA-3-8B-Instruct (vanilla reference baseline)",
        "paper":                "Meta AI, 2024",
        "max_new_tokens":       200,
        "max_input_tokens":     512,
        "dtype":                "bfloat16",
        "large_model":          True,
        "use_slow_tok":         False,
    },

    # ── Seq2Seq / encoder-decoder ─────────────────────────────────────────
    "flan_t5_med": {
        "hf_model":             "google/flan-t5-large",
        "family":               "seq2seq",
        "description":          "Flan-T5-Large (instruction-tuned, medical QA FT)",
        "paper":                "Chung et al., 2022 - arxiv:2210.11610",
        "max_new_tokens":       200,
        "max_input_tokens":     512,
        "dtype":                "float32",
        "large_model":          False,
        "use_slow_tok":         False,
    },
    "biogpt": {
        "hf_model":             "microsoft/biogpt",
        "family":               "causal_lm",
        "description":          "BioGPT (Microsoft, PubMed abstracts)",
        "paper":                "Luo et al., 2022 - Briefings in Bioinformatics",
        "max_new_tokens":       150,
        "max_input_tokens":     300,   # BioGPT max position is 1024; stay well under it
        "dtype":                "float32",
        "large_model":          False, # no gradient checkpointing on this architecture
        "use_slow_tok":         False,
    },
    "biomedlm": {
        "hf_model":             "stanford-crfm/BioMedLM",
        "family":               "causal_lm",
        "description":          "BioMedLM-2.7B (Stanford CRFM, PubMed)",
        "paper":                "Venigalla et al., 2022 - Stanford HAI Blog",
        "max_new_tokens":       150,
        "max_input_tokens":     256,   # GPT-2 architecture; short context keeps it in memory
        "dtype":                "float32",
        "large_model":          False, # no gradient checkpointing on this architecture
        "use_slow_tok":         False,
        "batch_override":       2,     # small batch to fit a 47 GiB card
        "grad_accum_override":  16,    # keeps the effective batch size
    },

    # ── Encoder-only adapted for generative QA ────────────────────────────
    "pubmedbert_qa": {
        "hf_model":             "ncbi/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "family":               "encoder_gen",
        "description":          "PubMedBERT + generative head (encoder for QA)",
        "paper":                "Gu et al., 2021 - Nature Methods",
        "max_new_tokens":       200,
        "max_input_tokens":     512,
        "dtype":                "float32",
        "large_model":          False,
        "use_slow_tok":         False,
    },
    "biolinkbert_qa": {
        "hf_model":             "michiyasunaga/BioLinkBERT-large",
        "family":               "encoder_gen",
        "description":          "BioLinkBERT-large + generative head",
        "paper":                "Yasunaga et al., 2022 - ACL",
        "max_new_tokens":       200,
        "max_input_tokens":     512,
        "dtype":                "float32",
        "large_model":          False,
        "use_slow_tok":         False,
    },
    "clinicalbert_qa": {
        "hf_model":             "medicalai/ClinicalBERT",
        "family":               "encoder_gen",
        "description":          "ClinicalBERT + generative head",
        "paper":                "Alsentzer et al., 2019 - ACL Workshop",
        "max_new_tokens":       200,
        "max_input_tokens":     512,
        "dtype":                "float32",
        "large_model":          False,
        "use_slow_tok":         False,
    },
}


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class BaselineConfig:
    # Dataset paths (same as main pipeline)
    dataset_name:      str   = "icliniq"
    medquad_dir:       str   = "SET_THIS_PATH"
    icliniq_dir:       str   = "SET_THIS_PATH"
    out_dir:           str   = "SET_THIS_PATH"

    # Model selection
    baseline_model:    str   = "meditron7b"
    local_model_path:  str   = ""
    run_all:           bool  = False

    # Sample sizes (same defaults as main pipeline)
    train_samples:     int   = 500    # -1 = full split
    val_samples:       int   = 200
    test_samples:      int   = 300

    # Training hyperparameters (identical to main pipeline)
    max_input_tokens:  int   = 512
    max_target_tokens: int   = 200
    batch_size:        int   = 8
    grad_accum:        int   = 4
    epochs:            int   = 3
    lr:                float = 5e-6
    warmup_ratio:      float = 0.03
    max_grad_norm:     float = 1.0
    seed:              int   = 42

    # Inference
    n_beams: int = 4
    device:  str = "cuda" if torch.cuda.is_available() else "cpu"



# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Medical QA baseline experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline_model", type=str, default="meditron7b",
                   choices=list(BASELINE_REGISTRY.keys()),
                   help="Which baseline to fine-tune and evaluate.")
    p.add_argument("--dataset", type=str, default="icliniq",
                   choices=["medquad", "icliniq"],
                   help="Dataset: 'medquad' or 'icliniq'.")
    p.add_argument("--train_samples", type=int, default=500,
                   help="Training samples. -1 = full split.")
    p.add_argument("--val_samples",   type=int, default=200,
                   help="Validation samples. -1 = full split.")
    p.add_argument("--test_samples",  type=int, default=300,
                   help="Test samples. -1 = full split.")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch_size",    type=int,   default=8,
                   help="Per-device batch size (may be overridden per-model).")
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--local_model_path", type=str, default="",
                   help="Local path overriding HuggingFace ID (for gated/offline models).")
    p.add_argument("--run_all",  action="store_true", default=False,
                   help="Run ALL baselines sequentially.")
    p.add_argument("--offline",  action="store_true", default=False,
                   help="Set TRANSFORMERS_OFFLINE=1 and HF_DATASETS_OFFLINE=1.")
    p.add_argument("--out_dir",  type=str,
                   default="SET_THIS_PATH")
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


def read_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def load_split(data_dir: str, split: str) -> Dataset:
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    records = read_jsonl(path)
    cleaned = []
    for r in records:
        q  = r.get("question", "").strip()
        a  = r.get("answer",   "").strip()
        if q and a:
            qt = r.get("qtype", r.get("type", "general"))
            cleaned.append({"question": q, "answer": a, "qtype": qt})
    return Dataset.from_list(cleaned)


def load_dataset_splits(cfg: BaselineConfig) -> DatasetDict:
    data_dir = cfg.medquad_dir if cfg.dataset_name == "medquad" else cfg.icliniq_dir
    splits: Dict[str, Dataset] = {}
    size_map = {
        "train":      cfg.train_samples,
        "validation": cfg.val_samples,
        "test":       cfg.test_samples,
    }
    for split_name, max_n in size_map.items():
        try:
            ds = load_split(data_dir, split_name)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} - skipping '{split_name}'.")
            continue
        if max_n > 0:
            ds = ds.select(range(min(max_n, len(ds))))
        print(f"  [{split_name}] {len(ds)} samples")
        splits[split_name] = ds
    return DatasetDict(splits)


def clean_output(text: str) -> str:
    """Strip prompt leakage and generation artefacts."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    for marker in ["\nAnswer:", "\nA:", "Answer:", "A:"]:
        idx = s.rfind(marker)
        if idx != -1:
            s = s[idx + len(marker):].strip()
    s = re.sub(r"\b(\w+)( \1){3,}\b", r"\1", s)
    s = re.sub(r"[\uFEFF\uFFFD]+", "", s)
    s = " ".join(s.split())
    if len(s) > 2000:
        s = s[:2000].rsplit(".", 1)[0]
    return s.strip()


def resolve_model_path(cfg: BaselineConfig, key: str) -> str:
    if cfg.local_model_path:
        print(f"  Using local path: {cfg.local_model_path}")
        return cfg.local_model_path
    return BASELINE_REGISTRY[key]["hf_model"]


def get_dtype(key: str) -> torch.dtype:
    d = BASELINE_REGISTRY[key].get("dtype", "float32")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(d, torch.float32)


def get_model_max_input(key: str, cfg: BaselineConfig) -> int:
    """Per-model token limit, falling back to cfg default."""
    return BASELINE_REGISTRY[key].get("max_input_tokens") or cfg.max_input_tokens


# =============================================================================
# PROMPT BUILDER  (same style as main pipeline)
# =============================================================================

def build_prompt(question: str, family: str) -> str:
    if family in ("causal_lm", "encoder_gen"):
        return (
            "You are a medical expert. "
            "Answer the following clinical question concisely and factually.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    return f"Answer the medical question: {question}"


def build_train_text(question: str, answer: str, family: str) -> Dict:
    if family in ("causal_lm", "encoder_gen"):
        return {"input": build_prompt(question, family), "target": answer}
    return {"input": f"Answer the medical question: {question}", "target": answer}


# =============================================================================
# MODEL LOADING
# =============================================================================

class EncoderGenModel(nn.Module):
    """Encoder-only BERT model + lightweight LM head for short-form generation."""

    def __init__(self, encoder, hidden_size: int, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(out.last_hidden_state)
        loss   = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(loss=loss, logits=logits)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, **kwargs):
        cur_ids  = input_ids
        cur_mask = attention_mask
        for _ in range(max_new_tokens):
            out     = self.encoder(input_ids=cur_ids, attention_mask=cur_mask)
            logits  = self.lm_head(out.last_hidden_state[:, -1, :])
            next_id = logits.argmax(dim=-1, keepdim=True)
            cur_ids  = torch.cat([cur_ids, next_id], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
        return cur_ids


def load_model_and_tokenizer(key: str, cfg: BaselineConfig) -> Tuple:
    """Load model and tokenizer.

    Checkpoints marked use_slow_tok carry only a SentencePiece tokenizer.model
    and are loaded through LlamaTokenizer(legacy=True); AutoTokenizer cannot
    build a fast tokenizer from them.
    """
    entry      = BASELINE_REGISTRY[key]
    family     = entry["family"]
    model_path = resolve_model_path(cfg, key)
    dtype      = get_dtype(key)

    print(f"  Loading tokenizer: {model_path}")

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
    elif family == "encoder_gen":
        from transformers import AutoModel
        encoder = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
        model = EncoderGenModel(
            encoder,
            encoder.config.hidden_size,
            tok.vocab_size or encoder.config.vocab_size,
        ).to(cfg.device)
    else:
        raise ValueError(f"Unknown family: {family}")

    print("  Model loaded OK")
    return model, tok, family

def fix_generation_config(model):
    if hasattr(model, "generation_config") and model.generation_config is not None:
        gc = model.generation_config

        # Force deterministic mode
        gc.do_sample = False

        # Remove invalid sampling params
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None

        # Optional: avoid future warnings
        if hasattr(gc, "typical_p"):
            gc.typical_p = None

    return model
# =============================================================================
# TOKENISATION
# =============================================================================

def tokenize_causal(examples: Dict, tok, cfg: BaselineConfig, key: str = "") -> Dict:
    """
    Concatenate prompt + answer; mask prompt tokens in labels.
    Identical logic to main pipeline train_generator preprocess().
    """
    model_max_in = get_model_max_input(key, cfg) if key else cfg.max_input_tokens

    p_enc = tok(list(examples["input"]),  truncation=True, padding=False,
                add_special_tokens=False)
    t_enc = tok(list(examples["target"]), truncation=True, padding=False,
                add_special_tokens=False)

    all_ids, all_labels, all_masks = [], [], []
    max_total = model_max_in + cfg.max_target_tokens

    for p_ids, t_ids in zip(p_enc["input_ids"], t_enc["input_ids"]):
        full = p_ids + [tok.eos_token_id] + t_ids
        if len(full) > max_total:
            keep = max_total - len(p_ids) - 1
            if keep <= 0:
                p_ids = p_ids[:model_max_in]
                full  = p_ids + [tok.eos_token_id] + t_ids[:cfg.max_target_tokens]
            else:
                full = p_ids + [tok.eos_token_id] + t_ids[-keep:]
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

    return {"input_ids": padded_ids, "labels": padded_lbl,
            "attention_mask": padded_msk}


def tokenize_seq2seq(examples: Dict, tok, cfg: BaselineConfig, key: str = "") -> Dict:
    """T5-family: encode source and target separately."""
    model_max_in = get_model_max_input(key, cfg) if key else cfg.max_input_tokens
    model_inputs = tok(
        list(examples["input"]),
        max_length=model_max_in,
        truncation=True, padding="max_length",
    )
    with tok.as_target_tokenizer():
        labels = tok(
            list(examples["target"]),
            max_length=cfg.max_target_tokens,
            truncation=True, padding="max_length",
        )
    model_inputs["labels"] = [
        [(t if t != tok.pad_token_id else -100) for t in l]
        for l in labels["input_ids"]
    ]
    return model_inputs


def tokenize_encoder_gen(examples: Dict, tok, cfg: BaselineConfig,
                         key: str = "") -> Dict:
    return tokenize_causal(examples, tok, cfg, key)


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_dataset(ds: Dataset, tok, cfg: BaselineConfig,
                    family: str, split_name: str, key: str = "") -> Dataset:
    inputs  = [build_train_text(ex["question"], ex["answer"], family)["input"]
               for ex in ds]
    targets = [build_train_text(ex["question"], ex["answer"], family)["target"]
               for ex in ds]
    ds = ds.add_column("input",  inputs)
    ds = ds.add_column("target", targets)

    print(f"  Tokenising {split_name} ({len(ds)} samples, family={family}) ...")

    if family == "causal_lm":
        fn = lambda ex: tokenize_causal(ex, tok, cfg, key)
    elif family == "seq2seq":
        fn = lambda ex: tokenize_seq2seq(ex, tok, cfg, key)
    else:
        fn = lambda ex: tokenize_encoder_gen(ex, tok, cfg, key)

    remove_cols = [c for c in ds.column_names
                   if c not in ["input_ids", "labels", "attention_mask"]]
    tokenised = ds.map(fn, batched=True, remove_columns=remove_cols)
    tokenised.set_format("torch")
    return tokenised


# =============================================================================
# TRAINING
# =============================================================================

def train_baseline(model, tok, train_tok: Dataset, val_tok: Dataset,
                   cfg: BaselineConfig, key: str, family: str) -> Trainer:
    """Fine-tune with the HuggingFace Trainer.

    Gradient checkpointing is applied only to causal models marked large_model;
    the smaller GPT-2-style architectures train without it. Accumulation steps
    come from the registry; the executed runs fixed the per-device batch at 1 for
    every baseline and did not apply the registry's batch_override, so --batch_size
    had no effect on them.
    """
    entry    = BASELINE_REGISTRY[key]
    save_dir = os.path.join(cfg.out_dir, key, cfg.dataset_name, "checkpoint")
    os.makedirs(save_dir, exist_ok=True)

    # per-model overrides from the registry
    eff_batch = entry.get("batch_override",    cfg.batch_size)
    eff_accum = entry.get("grad_accum_override", cfg.grad_accum)
    # gradient checkpointing only for the large causal models
    use_grad_ckpt = (family == "causal_lm") and entry.get("large_model", False)

    print(f"\n  Fine-tuning: {key} | dataset={cfg.dataset_name} | "
          f"per-device batch=1 accum={eff_accum} "
          f"(registry batch_override={eff_batch}, not applied) "
          f"grad_ckpt={use_grad_ckpt}")

    args = TrainingArguments(
        output_dir=save_dir,
        # held at 1 for every baseline in the executed runs
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=eff_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        # The executed runs forced bf16 for every baseline, rather than taking
        # the per-model dtype from the registry.
        fp16=False,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=use_grad_ckpt,
        seed=cfg.seed,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tok,   # avoids FutureWarning in transformers >= 5
    )

    trainer.train()
    trainer.save_model(save_dir)
    print(f"  Checkpoint saved: {save_dir}")
    return trainer


# =============================================================================
# INFERENCE
# =============================================================================

def generate_answer(question: str, model, tok, cfg: BaselineConfig,
                    family: str, key: str = "",
                    max_new_tokens: int = 200) -> str:
    """Generate a single answer.

    Gradient checkpointing is disabled and use_cache enabled before generation,
    so past_key_values caching is active; without it the GPT-2-style backbones
    decode incoherently.
    """
    prompt       = build_prompt(question, family)
    model_max_in = get_model_max_input(key, cfg) if key else cfg.max_input_tokens

    if family in ("causal_lm", "encoder_gen"):
        enc = tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=model_max_in,
        ).to(cfg.device)

        # inference state: caching on, checkpointing off
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=cfg.n_beams,
                early_stopping=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
                do_sample=False,    # greedy beam: stable across models
            )
        # Decode ONLY new tokens (exclude prompt prefix)
        prompt_len = enc["input_ids"].shape[1]
        decoded    = tok.decode(out[0][prompt_len:], skip_special_tokens=True)

    else:   # seq2seq
        enc = tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=model_max_in, padding=True,
        ).to(cfg.device)
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                num_beams=cfg.n_beams, early_stopping=True,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True)

    return clean_output(decoded)


# =============================================================================
# EVALUATION  (same metrics as main pipeline)
# =============================================================================

def evaluate_baseline(test_ds: Dataset, model, tok, cfg: BaselineConfig,
                      family: str, key: str,
                      max_new_tokens: int = 200) -> Dict:
    rouge       = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    meteor_m    = load_metric("meteor")
    bertscore_m = load_metric("bertscore")
    bleu_m      = load_metric("bleu")

    hyps, refs = [], []
    print(f"\n  Evaluating {key} on {cfg.dataset_name} "
          f"({len(test_ds)} samples) ...")

    for ex in tqdm(test_ds, desc=f"  Eval {key}"):
        pred = generate_answer(
            ex["question"], model, tok, cfg, family, key, max_new_tokens
        )
        hyps.append(pred)
        refs.append(ex["answer"])

    r1  = np.mean([rouge.score(r, p)["rouge1"].fmeasure  for r, p in zip(refs, hyps)])
    r2  = np.mean([rouge.score(r, p)["rouge2"].fmeasure  for r, p in zip(refs, hyps)])
    rl  = np.mean([rouge.score(r, p)["rougeL"].fmeasure  for r, p in zip(refs, hyps)])
    met = meteor_m.compute(predictions=hyps, references=refs)["meteor"]
    bs  = bertscore_m.compute(predictions=hyps, references=refs, lang="en")
    bsf = float(np.mean(bs["f1"]))
    b1  = bleu_m.compute(
        predictions=hyps, references=[[r] for r in refs], max_order=1)["bleu"]
    b4  = bleu_m.compute(
        predictions=hyps, references=[[r] for r in refs], max_order=4)["bleu"]

    results = {
        "ROUGE1":    round(r1,  4),
        "ROUGE2":    round(r2,  4),
        "ROUGEL":    round(rl,  4),
        "METEOR":    round(met, 4),
        "BERTScore": round(bsf, 4),
        "BLEU1":     round(b1,  4),
        "BLEU4":     round(b4,  4),
    }

    print(f"\n  ===== {key} | {cfg.dataset_name} =====")
    for k, v in results.items():
        print(f"    {k:12s}: {v:.4f}")
    return results


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_curves(trainer: Trainer, save_dir: str, label: str):
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
        ax.set_title(f"{label} - Training & Validation Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{label}_loss.png"), dpi=150)
        plt.close()

    if "learning_rate" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        lr = df.dropna(subset=["learning_rate"])
        ax.plot(lr["step"], lr["learning_rate"], color="#16A34A")
        ax.set_xlabel("Step"); ax.set_ylabel("LR")
        ax.set_title(f"{label} - Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{label}_lr.png"), dpi=150)
        plt.close()

    print(f"  Training curves saved: {save_dir}")


# =============================================================================
# COMPARISON PLOTS AND CSV
# =============================================================================

def save_comparison(all_results: Dict[str, Dict], cfg: BaselineConfig):
    comp_dir = os.path.join(cfg.out_dir, "comparison", cfg.dataset_name)
    os.makedirs(comp_dir, exist_ok=True)

    metrics = ["ROUGE1", "ROUGE2", "ROUGEL", "METEOR",
               "BERTScore", "BLEU1", "BLEU4"]

    rows = []
    for bkey, res in all_results.items():
        row = {
            "Baseline":    bkey,
            "Description": BASELINE_REGISTRY.get(bkey, {}).get("description", bkey),
            "Family":      BASELINE_REGISTRY.get(bkey, {}).get("family", "?"),
            "Dataset":     cfg.dataset_name,
        }
        row.update({m: res.get(m, 0.0) for m in metrics})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("METEOR", ascending=False)

    csv_path = os.path.join(comp_dir, "baseline_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    md_path = os.path.join(comp_dir, "baseline_comparison.md")
    with open(md_path, "w") as f:
        f.write(f"# Baseline Comparison - {cfg.dataset_name.upper()}\n\n")
        f.write(df[["Baseline", "Description", "Family"] + metrics].to_markdown(
            index=False))
        f.write("\n")
    print(f"  Markdown saved: {md_path}")

    n_b    = len(all_results)
    x      = np.arange(len(metrics))
    w      = 0.8 / max(n_b, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_b))

    # Figure 1: all metrics grouped bar
    fig, ax = plt.subplots(figsize=(max(14, len(metrics) * 1.6), 6))
    for i, (bkey, res) in enumerate(all_results.items()):
        vals   = [res.get(m, 0.0) for m in metrics]
        offset = (i - n_b / 2 + 0.5) * w
        desc   = BASELINE_REGISTRY.get(bkey, {}).get("description", bkey)
        ax.bar(x + offset, vals, width=w * 0.9, label=desc, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Baseline Comparison - {cfg.dataset_name.upper()} - All Metrics")
    ax.legend(loc="upper right", fontsize=8, ncol=max(1, n_b // 3))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "all_metrics.png"), dpi=150)
    plt.close()

    # Figure 2: ROUGE1 / METEOR / BERTScore
    key3 = ["ROUGE1", "METEOR", "BERTScore"]
    x3   = np.arange(len(key3))
    w3   = 0.8 / max(n_b, 1)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (bkey, res) in enumerate(all_results.items()):
        vals   = [res.get(m, 0.0) for m in key3]
        offset = (i - n_b / 2 + 0.5) * w3
        desc   = BASELINE_REGISTRY.get(bkey, {}).get("description", bkey)
        ax2.bar(x3 + offset, vals, width=w3 * 0.9, label=desc, color=colors[i])
    ax2.set_xticks(x3)
    ax2.set_xticklabels(key3)
    ax2.set_ylabel("Score")
    ax2.set_title(f"Key Metrics - {cfg.dataset_name.upper()}")
    ax2.legend(loc="upper right", fontsize=8, ncol=max(1, n_b // 3))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "key_metrics.png"), dpi=150)
    plt.close()

    # Figure 3: horizontal bar ranked by METEOR
    df_s  = df.sort_values("METEOR")
    fig3, ax3 = plt.subplots(figsize=(9, max(4, len(df_s) * 0.55)))
    y_pos = np.arange(len(df_s))
    ax3.barh(y_pos, df_s["METEOR"],
             color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df_s))))
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(df_s["Baseline"], fontsize=9)
    ax3.set_xlabel("METEOR")
    ax3.set_title(f"Baselines Ranked by METEOR - {cfg.dataset_name.upper()}")
    ax3.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "ranked_meteor.png"), dpi=150)
    plt.close()

    print(f"  3 comparison figures saved: {comp_dir}")


# =============================================================================
# PER-BASELINE RUNNER
# =============================================================================

def run_one_baseline(key: str, cfg: BaselineConfig) -> Optional[Dict]:
    """Full pipeline: load -> prepare -> fine-tune -> evaluate -> save."""
    entry = BASELINE_REGISTRY[key]
    print(f"\n{'='*65}")
    print(f"  BASELINE : {key}")
    print(f"  {entry['description']}")
    print(f"  {entry['paper']}")
    print(f"  Dataset  : {cfg.dataset_name} | "
          f"train={cfg.train_samples} val={cfg.val_samples} "
          f"test={cfg.test_samples}")
    print(f"{'='*65}")

    # Result cache
    res_dir  = os.path.join(cfg.out_dir, key, cfg.dataset_name)
    res_path = os.path.join(res_dir, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            cached = json.load(f)
        r = cached["results"]
        print(f"  Cached results found - skipping.")
        print(f"  ROUGE1={r.get('ROUGE1',0):.4f}  "
              f"METEOR={r.get('METEOR',0):.4f}  "
              f"BERTScore={r.get('BERTScore',0):.4f}")
        return r

    os.makedirs(res_dir, exist_ok=True)

    # Free GPU memory before loading next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load data
    print("\n  Loading dataset ...")
    splits = load_dataset_splits(cfg)
    if "train" not in splits or "test" not in splits:
        print(f"  ERROR: missing required splits. Skipping {key}.")
        return None
    train_ds = splits["train"]
    val_ds   = splits.get("validation", splits["train"])
    test_ds  = splits["test"]

    # Load model
    try:
        model, tok, family = load_model_and_tokenizer(key, cfg)
        model = fix_generation_config(model)
    except Exception as e:
        print(f"  ERROR loading model '{key}': {e}")
        return None

    # Tokenise
    train_tok = prepare_dataset(train_ds, tok, cfg, family, "train", key)
    val_tok   = prepare_dataset(val_ds,   tok, cfg, family, "val",   key)

    # Train
    trainer = train_baseline(model, tok, train_tok, val_tok, cfg, key, family)

    # Curves
    plot_training_curves(
        trainer,
        os.path.join(res_dir, "curves"),
        label=f"{key}_{cfg.dataset_name}",
    )

    # Evaluate
    max_new = entry.get("max_new_tokens", 200)
    results = evaluate_baseline(test_ds, model, tok, cfg, family, key, max_new)

    # Save
    payload = {
        "baseline":  key,
        "dataset":   cfg.dataset_name,
        "family":    family,
        "hf_model":  entry["hf_model"],
        "paper":     entry["paper"],
        "train_samples": len(train_ds),
        "val_samples":   len(val_ds),
        "test_samples":  len(test_ds),
        "config": {
            "epochs":            cfg.epochs,
            "lr":                cfg.lr,
            "batch_size":        entry.get("batch_override",    cfg.batch_size),
            "grad_accum":        entry.get("grad_accum_override", cfg.grad_accum),
            "max_input_tokens":  entry.get("max_input_tokens",  cfg.max_input_tokens),
            "max_target_tokens": cfg.max_target_tokens,
        },
        "results": results,
    }
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved: {res_path}")

    # Free GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Both must be set before any CUDA allocation. expandable_segments limits
    # the fragmentation that otherwise exhausts memory on the backward pass, and
    # a single visible device avoids the DataParallel reduce, which would double
    # the backward-pass requirement.
    os.environ.setdefault("PYTORCH_ALLOC_CONF",  "expandable_segments:True")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"]  = "1"
        print("  Offline mode enabled.")

    cfg = BaselineConfig(
        dataset_name     = args.dataset,
        baseline_model   = args.baseline_model,
        local_model_path = args.local_model_path,
        run_all          = args.run_all,
        train_samples    = args.train_samples,
        val_samples      = args.val_samples,
        test_samples     = args.test_samples,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        out_dir          = args.out_dir,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("\n" + "="*65)
    print("  Medical QA Baseline Experiment Suite")
    print("="*65)
    print(f"  Dataset  : {cfg.dataset_name.upper()}")
    print(f"  Samples  : train={cfg.train_samples}  "
          f"val={cfg.val_samples}  test={cfg.test_samples}")
    print(f"  Epochs   : {cfg.epochs}   LR: {cfg.lr}   "
          f"Batch (default): {cfg.batch_size}")
    print(f"  Device   : {cfg.device}")
    print(f"  Out dir  : {cfg.out_dir}")
    print("="*65)

    all_results: Dict[str, Dict] = {}

    if cfg.run_all:
        print(f"\n  Running ALL {len(BASELINE_REGISTRY)} baselines ...")
        for key in BASELINE_REGISTRY:
            cfg.baseline_model   = key
            cfg.local_model_path = ""
            res = run_one_baseline(key, cfg)
            if res:
                all_results[key] = res
    else:
        key = cfg.baseline_model
        res = run_one_baseline(key, cfg)
        if res:
            all_results[key] = res

    if len(all_results) > 1:
        save_comparison(all_results, cfg)

    if all_results:
        print("\n\n" + "="*65)
        print(f"  FINAL SUMMARY - {cfg.dataset_name.upper()}")
        print("="*65)
        metrics = ["ROUGE1", "METEOR", "BERTScore", "BLEU1", "BLEU4"]
        header  = f"  {'Baseline':<20} {'Family':<14}" + "".join(
            f"{m:>12}" for m in metrics)
        print(header)
        print("  " + "-" * (len(header) - 2))

        ranked = sorted(
            all_results.items(),
            key=lambda x: x[1].get("METEOR", 0), reverse=True,
        )
        for bkey, res in ranked:
            fam  = BASELINE_REGISTRY.get(bkey, {}).get("family", "?")
            vals = "".join(f"{res.get(m, 0):>12.4f}" for m in metrics)
            print(f"  {bkey:<20} {fam:<14}{vals}")

        summary = {
            "dataset":   cfg.dataset_name,
            "baselines": all_results,
            "config": {
                "train_samples": cfg.train_samples,
                "val_samples":   cfg.val_samples,
                "test_samples":  cfg.test_samples,
                "epochs":        cfg.epochs,
                "lr":            cfg.lr,
                "batch_size":    cfg.batch_size,
            },
        }
        summary_path = os.path.join(
            cfg.out_dir, f"summary_{cfg.dataset_name}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Master summary saved: {summary_path}")


if __name__ == "__main__":
    main()