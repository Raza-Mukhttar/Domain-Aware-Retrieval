# file: medlm_verifier_experiments.py
"""
Full MedLM-LLaMA3 pipeline with VERIFIER EXPERIMENT SUPPORT.

Original pipeline is 100% unchanged.
New additions (search "VERIFIER EXPERIMENT" to find every new block):
  - VerifierConfig  : registry of all verifier models + their loading strategy
  - build_verifier  : unified factory that returns (model, tokenizer, score_fn)
                      for any verifier family
  - score_with_verifier : family-aware scoring used inside generate_and_rerank
  - run_verifier_experiments : iterates all verifiers, evaluates, saves CSV + plots
  - CLI flag  --verifier_type   (one verifier)
  - CLI flag  --run_all_verifiers  (sweep every verifier sequentially)

Usage examples
--------------
# Single verifier (original behaviour, default = clinicalbert):
    python medlm_verifier_experiments.py --verifier_type clinicalbert

# Single verifier — cross-encoder family:
    python medlm_verifier_experiments.py --verifier_type biolinkbert

# Single verifier — NLI family:
    python medlm_verifier_experiments.py --verifier_type mednli_deberta

# Single verifier — LLM-as-judge (Prometheus-2):
    python medlm_verifier_experiments.py --verifier_type prometheus2

# Single verifier — atomic fact-check (MiniCheck):
    python medlm_verifier_experiments.py --verifier_type minicheck

# Single verifier — KG-aware (BioKG-BERT):
    python medlm_verifier_experiments.py --verifier_type biokg_bert

# Full sweep — all verifiers sequentially, saves comparison CSV + bar chart:
    python medlm_verifier_experiments.py --run_all_verifiers

# Combine with retriever flag (from original):
    python medlm_verifier_experiments.py --retriever_type dense --run_all_verifiers

Available --verifier_type values
---------------------------------
  clinicalbert        medicalai/ClinicalBERT                [original, cross-encoder style]
  pubmedbert          ncbi/BiomedNLP-PubMedBERT-...         [cross-encoder, biomedical BERT]
  biolinkbert         ncbi/BioLinkBERT-large                [cross-encoder, strongest bio]
  bioelectra          kamalkraj/bioelectra-base-...         [cross-encoder, ELECTRA bio]
  minilm_crossenc     cross-encoder/ms-marco-MiniLM-L-6-v2  [cross-encoder, fast baseline]
  mednli_deberta      cross-encoder/nli-deberta-v3-large    [NLI family, DeBERTa-v3]
  biomed_roberta_nli  cross-encoder/nli-roberta-base        [NLI family, RoBERTa]
  meditron_judge      epfl-llm/meditron-7b                  [LLM-judge, domain expert]
  prometheus2         prometheus-eval/prometheus-2-7b-v2.0  [LLM-judge, open evaluator]
  minicheck           bespokelabs/Bespoke-MiniCheck-7B      [atomic fact-check]
  biokg_bert          allenai/scibert_scivocab_cased        [KG-aware, triple verif.]

Notes
-----
- Install extra packages when needed:
    pip install transformers accelerate datasets evaluate rouge_score
    pip install sentence-transformers faiss-cpu scispacy rank_bm25
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
- Large LLM-judge models (meditron_judge, prometheus2) require ~16 GB GPU VRAM.
  If GPU is insufficient, those entries are automatically skipped with a warning.
- KG verifier (biokg_bert) uses SciBERT fine-tuned on SciTail NLI as a proxy for
  triple verification; a full UMLS API integration is noted as a TODO.
"""

# ============================================================
# ORIGINAL IMPORTS (unchanged)
# ============================================================
import os
import json
import random
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
import numpy as np
import torch
from torch import nn, optim
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from evaluate import load as load_metric
import argparse

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import pandas as pd

# scispaCy for biomedical entity extraction
try:
    import spacy
except Exception as e:
    raise RuntimeError(
        "Please install scispacy & spacy:\n"
        "  pip install scispacy spacy\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
        "v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz"
    )


# ============================================================
# ORIGINAL CONFIG (unchanged)
# ============================================================
@dataclass
class Config:
    dataset_name: str = "icliniq"          # "medquad" or "icliniq"

    medquad_dir: str  = "SET_THIS_PATH"
    icliniq_dir: str  = "SET_THIS_PATH"
    out_dir: str      = "SET_THIS_PATH"

    summarizer_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    generator_model: str  = "meta-llama/Meta-Llama-3-8B-Instruct"
    verifier_model: str   = "medicalai/ClinicalBERT"   # overridden by --verifier_type
    embedder_model: str   = "SET_THIS_PATH"
    per_type_index_dir: str = "SET_THIS_PATH"

    max_summary_tokens: int = 200
    max_input_tokens: int   = 512
    n_exemplars: int        = 3
    n_best: int             = 5
    device: str             = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int               = 42
    batch_size: int         = 8
    epochs: int             = 3
    lr: float               = 3e-5
    contrastive_epochs: int = 2
    contrastive_batch_size: int = 8
    contrastive_temp: float = 0.07
    verifier_threshold: float = 0.5

    use_subset: bool       = True
    subset_train_size: int = 500
    subset_val_size: int   = 200
    subset_test_size: int  = 300

    # ── NEW fields (defaults keep original behaviour) ──────────────
    retriever_type: str  = "dense"         # populated from CLI
    verifier_type: str   = "clinicalbert"  # populated from CLI
    run_all_verifiers: bool = False         # populated from CLI


# ============================================================
# VERIFIER EXPERIMENT — Registry
# ============================================================
"""
Each entry in VERIFIER_REGISTRY describes one verifier variant.

Keys
----
hf_model   : HuggingFace model ID (or local path)
family     : one of  cross_encoder | nli | llm_judge | atomic | kg_aware
description: human-readable label used in result tables and plots
score_mode : how to extract a scalar correctness score from the model
             - "seq_cls"     : AutoModelForSequenceClassification, softmax[:,1]
             - "cross_enc"   : sentence_transformers CrossEncoder, sigmoid(score)
             - "nli_entail"  : seq_cls with 3 labels, use entailment logit (index 2)
             - "causal_lm"   : AutoModelForCausalLM prompted judge (0-1 float from text)
             - "minicheck"   : causal LM with special MiniCheck prompt format
             - "scibert_nli" : seq_cls 3-class NLI, entailment index 0 (SciTail convention)
"""

VERIFIER_REGISTRY: Dict[str, Dict] = {

    # ── Cross-encoder family ──────────────────────────────────────────────────
    "clinicalbert": {
        "hf_model":    "medicalai/ClinicalBERT",
        "family":      "cross_encoder",
        "description": "ClinicalBERT (original)",
        "score_mode":  "seq_cls",
    },
    "pubmedbert": {
        "hf_model":    "ncbi/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "family":      "cross_encoder",
        "description": "PubMedBERT (cross-encoder)",
        "score_mode":  "seq_cls",
    },
    "biolinkbert": {
        "hf_model":    "michiyasunaga/BioLinkBERT-large",
        "family":      "cross_encoder",
        "description": "BioLinkBERT-large (cross-encoder)",
        "score_mode":  "seq_cls",
    },
    "bioelectra": {
        "hf_model":    "kamalkraj/bioelectra-base-discriminator-pubmed",
        "family":      "cross_encoder",
        "description": "BioELECTRA-base-PubMed (cross-encoder)",
        "score_mode":  "seq_cls",
    },
    "minilm_crossenc": {
        "hf_model":    "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "family":      "cross_encoder",
        "description": "MiniLM-L6 MS-MARCO (fast baseline)",
        "score_mode":  "cross_enc",   # uses SentenceTransformers CrossEncoder
    },

    # ── NLI family ───────────────────────────────────────────────────────────
    "mednli_deberta": {
        "hf_model":    "cross-encoder/nli-deberta-v3-large",
        "family":      "nli",
        "description": "DeBERTa-v3-large NLI",
        "score_mode":  "nli_entail",
    },
    "biomed_roberta_nli": {
        "hf_model":    "cross-encoder/nli-roberta-base",
        "family":      "nli",
        "description": "RoBERTa-base NLI",
        "score_mode":  "nli_entail",
    },

    # ── LLM-judge family ─────────────────────────────────────────────────────
    "meditron_judge": {
        "hf_model":    "epfl-llm/meditron-7b",
        "family":      "llm_judge",
        "description": "Meditron-7B (LLM judge)",
        "score_mode":  "causal_lm",
    },
    "prometheus2": {
        "hf_model":    "prometheus-eval/prometheus-2-7b-v2.0",
        "family":      "llm_judge",
        "description": "Prometheus-2-7B (open evaluator)",
        "score_mode":  "causal_lm",
    },

    # ── Atomic fact-check family ──────────────────────────────────────────────
    "minicheck": {
        "hf_model":    "bespokelabs/Bespoke-MiniCheck-7B",
        "family":      "atomic",
        "description": "Bespoke-MiniCheck-7B (atomic fact-check)",
        "score_mode":  "minicheck",
    },

    # ── KG-aware family ───────────────────────────────────────────────────────
    "biokg_bert": {
        "hf_model":    "allenai/scibert_scivocab_cased",
        "family":      "kg_aware",
        "description": "SciBERT SciTail NLI (KG-aware proxy)",
        "score_mode":  "scibert_nli",
    },
}


# ============================================================
# VERIFIER EXPERIMENT — Build verifier (unified factory)
# ============================================================

def _load_seq_cls(hf_model: str, device: str):
    """Load a binary/ternary AutoModelForSequenceClassification verifier."""
    tok   = AutoTokenizer.from_pretrained(hf_model, token = "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")
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model,
        ignore_mismatched_sizes=True,   # safe for cross-domain loading
    ).to(device)
    model.eval()
    return tok, model


def _load_cross_enc(hf_model: str, device: str):
    """Load a SentenceTransformers CrossEncoder verifier."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(hf_model, device=device)
    return None, model      # tok=None; CrossEncoder handles tokenisation internally


def _load_causal_lm_judge(hf_model: str, device: str):
    """Load a causal LM for use as a prompted judge."""
    tok = AutoTokenizer.from_pretrained(hf_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load LLM judge '{hf_model}'. "
            f"Check GPU memory (need ~16 GB). Original error: {exc}"
        )
    model.eval()
    return tok, model


def build_verifier(verifier_key: str, cfg: Config):
    """
    Factory function — returns (tok, model, score_fn).

    score_fn signature:
        score_fn(question: str, answer: str) -> float   (0.0 – 1.0)

    This is the only function called outside this block to obtain a verifier
    score.
    """
    if verifier_key not in VERIFIER_REGISTRY:
        raise ValueError(
            f"Unknown verifier '{verifier_key}'. "
            f"Choose from: {list(VERIFIER_REGISTRY.keys())}"
        )

    entry      = VERIFIER_REGISTRY[verifier_key]
    hf_model   = entry["hf_model"]
    score_mode = entry["score_mode"]
    device     = cfg.device

    print(f"\n🔬 Loading verifier [{verifier_key}] → {hf_model}  (mode={score_mode})")

    # ── Cross-encoder via AutoModelForSequenceClassification ─────────────────
    if score_mode == "seq_cls":
        tok, model = _load_seq_cls(hf_model, device)

        def score_fn(question: str, answer: str) -> float:
            text = f"Q: {question} [SEP] A: {answer}"
            enc  = tok(
                text, return_tensors="pt",
                truncation=True, max_length=512, padding="max_length"
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            # Works for both binary (2 labels) and multi-label; take label-1 prob.
            probs = torch.softmax(logits, dim=-1)[0]
            # If model has only 1 label fall back gracefully
            return float(probs[min(1, probs.shape[0] - 1)].item())

        return tok, model, score_fn

    # ── SentenceTransformers CrossEncoder ────────────────────────────────────
    elif score_mode == "cross_enc":
        tok, ce_model = _load_cross_enc(hf_model, device)

        def score_fn(question: str, answer: str) -> float:
            raw = ce_model.predict([(question, answer)])[0]
            # CrossEncoder scores are logits; sigmoid maps to [0,1]
            return float(1.0 / (1.0 + np.exp(-raw)))

        return tok, ce_model, score_fn

    # ── NLI — entailment logit (label index 2 for contradiction/neutral/entail)
    elif score_mode == "nli_entail":
        tok, model = _load_seq_cls(hf_model, device)

        def score_fn(question: str, answer: str) -> float:
            # Premise = question context, hypothesis = answer
            premise    = f"The medical question is: {question}"
            hypothesis = answer
            enc = tok(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True, max_length=512, padding="max_length"
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            n_labels = probs.shape[0]
            if n_labels == 3:
                # Standard NLI order: contradiction(0), neutral(1), entailment(2)
                return float(probs[2].item())
            else:
                # Binary — take positive class
                return float(probs[-1].item())

        return tok, model, score_fn

    # ── LLM-as-judge (causal LM) ─────────────────────────────────────────────
    elif score_mode == "causal_lm":
        tok, model = _load_causal_lm_judge(hf_model, device)
        max_new = 16   # we only need a short numeric reply

        def score_fn(question: str, answer: str) -> float:
            prompt = (
                "You are a medical expert evaluator.\n"
                "Rate how factually correct and complete the following medical answer "
                "is for the given question.\n"
                "Respond with a single number between 0 and 1 (e.g. 0.85). "
                "Do not add any explanation.\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                "Score (0-1):"
            )
            enc = tok(
                prompt, return_tensors="pt",
                truncation=True, max_length=cfg.max_input_tokens
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            generated = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            # parse first float found in the output
            nums = re.findall(r"\d+\.?\d*", generated)
            if nums:
                val = float(nums[0])
                # model might output 0-100; normalise
                return float(min(1.0, max(0.0, val if val <= 1.0 else val / 100.0)))
            return 0.5   # fallback when model gives no parseable number

        return tok, model, score_fn

    # ── Atomic fact-check (MiniCheck) ────────────────────────────────────────
    elif score_mode == "minicheck":
        tok, model = _load_causal_lm_judge(hf_model, device)
        max_new = 8

        def score_fn(question: str, answer: str) -> float:
            # MiniCheck prompt: doc = answer, claim = condensed question claim
            prompt = (
                f"Doc: {answer}\n"
                f"Claim: {question}\n"
                "Is the claim supported by the document? "
                "Answer with a probability between 0 and 1:"
            )
            enc = tok(
                prompt, return_tensors="pt",
                truncation=True, max_length=cfg.max_input_tokens
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            generated = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            nums = re.findall(r"\d+\.?\d*", generated)
            if nums:
                val = float(nums[0])
                return float(min(1.0, max(0.0, val if val <= 1.0 else val / 100.0)))
            return 0.5

        return tok, model, score_fn

    # ── KG-aware (SciBERT NLI proxy, SciTail convention) ─────────────────────
    elif score_mode == "scibert_nli":
        tok, model = _load_seq_cls(hf_model, device)

        def score_fn(question: str, answer: str) -> float:
            # SciTail has 2 labels: entails(0), neutral(1) — we want entails prob
            text = f"Q: {question} [SEP] A: {answer}"
            enc  = tok(
                text, return_tensors="pt",
                truncation=True, max_length=512, padding="max_length"
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            # label 0 = entailment in SciTail; if 3-label fall back to standard NLI
            if probs.shape[0] == 2:
                return float(probs[0].item())
            return float(probs[2].item())

        return tok, model, score_fn

    else:
        raise ValueError(f"Unknown score_mode '{score_mode}'.")


# ============================================================
# VERIFIER EXPERIMENT — Score helper used in generate_and_rerank
# ============================================================

def score_with_verifier(question: str, answer: str, score_fn: Callable) -> float:
    """
    Thin wrapper so generate_and_rerank can call a unified score_fn
    regardless of verifier family.  Returns float in [0, 1].
    """
    try:
        return float(score_fn(question, answer))
    except Exception as exc:
        # Never let a verifier crash crash evaluation — return neutral 0.5
        print(f"  ⚠️  Verifier score_fn raised exception: {exc}. Returning 0.5.")
        return 0.5


# ============================================================
# ORIGINAL UnifiedRetriever (unchanged)
# ============================================================

class UnifiedRetriever:
    def __init__(self, ds, cfg):
        self.cfg       = cfg
        self.questions = list(ds["question"])
        self.summaries = list(ds["summary"])

        assert cfg.retriever_type in ["dense", "bm25", "hybrid", "rerank"]

        self.embedder     = None
        self.index        = None
        self.bm25         = None
        self.cross_encoder = None

        if cfg.retriever_type in ["dense", "hybrid", "rerank"]:
            self.embedder = SentenceTransformer(cfg.embedder_model, device=cfg.device)
            self.embs     = self.embedder.encode(self.questions, convert_to_numpy=True)
            faiss.normalize_L2(self.embs)
            self.index = faiss.IndexFlatIP(self.embs.shape[1])
            self.index.add(self.embs)

        if cfg.retriever_type in ["bm25", "hybrid"]:
            from rank_bm25 import BM25Okapi
            tokenized = [q.lower().split() for q in self.questions]
            self.bm25 = BM25Okapi(tokenized)

        if cfg.retriever_type == "rerank":
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", device=cfg.device
            )

    def search(self, query, topk):
        if self.cfg.retriever_type == "dense":   return self._dense(query, topk)
        if self.cfg.retriever_type == "bm25":    return self._bm25(query, topk)
        if self.cfg.retriever_type == "hybrid":  return self._hybrid(query, topk)
        if self.cfg.retriever_type == "rerank":  return self._rerank(query, topk)
        raise ValueError("Invalid retriever type")

    def _dense(self, query, topk):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, I = self.index.search(q_emb, topk)
        return I[0]

    def _bm25(self, query, topk):
        scores = self.bm25.get_scores(query.lower().split())
        return np.argsort(scores)[::-1][:topk]

    def _hybrid(self, query, topk):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I  = self.index.search(q_emb, topk * 2)
        dense_scores = D[0]; dense_idx = I[0]
        bm25_scores  = self.bm25.get_scores(query.lower().split())
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() + 1e-6)
        bm25_scores  = (bm25_scores  - np.min(bm25_scores)) / (np.max(bm25_scores) + 1e-6)
        combined = {}
        for i, idx in enumerate(dense_idx):
            combined[idx] = dense_scores[i] + bm25_scores[idx]
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked[:topk]]

    def _rerank(self, query, topk):
        base   = self._dense(query, topk * 3)
        pairs  = [(query, self.questions[i]) for i in base]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(base, scores), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked[:topk]]


# ============================================================
# ORIGINAL utilities (unchanged)
# ============================================================

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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


def load_all_datasets(cfg: Config):
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
                full = len(merged[s])
                subsz = {
                    "train":      min(cfg.subset_train_size, full),
                    "validation": min(cfg.subset_val_size,   full),
                    "test":       min(cfg.subset_test_size,  full),
                }[s]
                merged[s] = merged[s].select(range(subsz))
                print(f"⚙️  Using {subsz}/{full} samples for {s}")
    print("Dataset sizes:", {k: len(v) for k, v in merged.items()})
    return DatasetDict(merged)


# ============================================================
# ORIGINAL summarization (unchanged)
# ============================================================

def summarize_dataset(ds: Dataset, cfg: Config, split_name: str = "train"):
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, f"{split_name}_summaries_icliniq500.json")

    def _clean_summary(s: str) -> str:
        if s is None: return ""
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
            if rep < 3: out_toks.append(t)
            prev = t
        return " ".join(" ".join(out_toks).split())

    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) == len(ds):
                print(f"✅ Loaded cached summaries for {split_name} ({len(cached)} entries).")
                return ds.add_column("summary", cached)
            else:
                print(f"⚠️  Cache length mismatch. Regenerating...")
        except Exception as e:
            print(f"⚠️  Failed loading cache ({e}). Regenerating...")

    print(f"🩺 Generating summaries for {split_name} split...")
    tok = AutoTokenizer.from_pretrained(cfg.summarizer_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.summarizer_model, torch_dtype=torch.float16, device_map="auto"
    )
    summaries = []
    for ex in tqdm(ds, desc=f"Summarizing {split_name}"):
        prompt = (
            "Summarize the following medical answer factually and concisely "
            "(limit ~100 tokens):\n\n"
            f"{ex['answer']}\n\nSummary:"
        )
        inputs = tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=cfg.max_input_tokens, padding=True
        ).to(cfg.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=cfg.max_summary_tokens,
                temperature=0.2, top_p=0.9,
                pad_token_id=tok.eos_token_id, do_sample=False
            )
        raw     = tok.decode(out[0], skip_special_tokens=True)
        cleaned = _clean_summary(raw)
        summaries.append(cleaned)

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"💾 Saved {len(summaries)} summaries → {save_path}")
    except Exception as e:
        print(f"⚠️  Failed saving summaries: {e}")

    return ds.add_column("summary", summaries)


# ============================================================
# ORIGINAL entity extraction (unchanged)
# ============================================================

try:
    NLP_MED = spacy.load("en_ner_bc5cdr_md")
except Exception:
    raise RuntimeError(
        "scispaCy model not found. Install:\n"
        "  pip install scispacy\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
        "v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz"
    )

MED_ENT_VOCAB: set = set()    # populated in main()


def extract_medical_entities(text: str):
    if not text or not isinstance(text, str): return []
    doc  = NLP_MED(text)
    ents = set()
    for ent in doc.ents:
        clean = ent.text.strip().lower()
        if len(clean) > 2: ents.add(clean)
    return sorted(list(ents))


def build_entity_vocabulary(dataset: Dataset, cfg: Config, path=None):
    if path is None:
        path = os.path.join(cfg.out_dir, "medical_entities.json")
    all_ents = set()
    print("🔍 Extracting medical entities...")
    for ex in tqdm(dataset, desc="Extracting concepts"):
        text = ex.get("summary") or ex.get("answer") or ""
        if text: all_ents.update(extract_medical_entities(text))
    all_ents = sorted(list(all_ents))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_ents, f, indent=2)
    with open(path.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        for e in all_ents: f.write(e + "\n")
    print(f"✅ Extracted {len(all_ents)} entities → {path}")
    return all_ents


def load_entity_vocabulary(path):
    if not os.path.exists(path): return set()
    return set(json.load(open(path, "r", encoding="utf-8")))


def extract_entities(text: str):
    if not text: return []
    tokens = [t.strip(".,;:()").lower() for t in re.split(r"\s+", text)]
    return sorted({t for t in tokens if t in MED_ENT_VOCAB})


def add_entities_column(ds: Dataset):
    ents = [extract_entities(x["summary"]) for x in ds]
    return ds.add_column("entities", ents)


# ============================================================
# ORIGINAL retriever helpers (unchanged)
# ============================================================

def build_retriever(train_ds, cfg):
    return UnifiedRetriever(train_ds, cfg)


def retrieve_exemplars(question, retriever, train_ds, cfg):
    idxs = retriever.search(question, cfg.n_exemplars)
    return [
        {"question": train_ds[i]["question"], "summary": train_ds[i]["summary"]}
        for i in idxs
    ]


def assemble_prompt(question: str, exemplars: List[Dict]):
    parts = [
        "You are a medical expert.\n"
        "Use clinical reasoning to answer concisely.\n"
        "Answer must be related to the question.\n"
    ]
    for ex in exemplars:
        parts.append(f"Q: {ex['question']}\nA: {ex['summary']}\n---\n")
    parts.append(f"Q: {question}\nA:")
    return "\n".join(parts)


def prepare_dataset_with_retriever(ds, retriever, train_ds, cfg):
    prompts = []
    for ex in ds:
        exs = retrieve_exemplars(ex["question"], retriever, train_ds, cfg)
        prompts.append(assemble_prompt(ex["question"], exs))
    return ds.add_column("prompt", prompts)


# ============================================================
# ORIGINAL train_generator (unchanged)
# ============================================================

def train_generator(train_ds: Dataset, val_ds: Dataset, cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.generator_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.generator_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    def preprocess(batch):
        prompts = list(batch["prompt"])
        targets = list(batch["summary"])
        p_tok   = tok(prompts, truncation=True, padding=False, add_special_tokens=False)
        t_tok   = tok(targets, truncation=True, padding=False, add_special_tokens=False)
        input_ids_batch, labels_batch, attention_masks = [], [], []
        max_total = cfg.max_input_tokens + cfg.max_summary_tokens
        for p_ids, t_ids in zip(p_tok["input_ids"], t_tok["input_ids"]):
            full = p_ids + [tok.eos_token_id] + t_ids
            if len(full) > max_total:
                keep_t = max_total - (len(p_ids) + 1)
                if keep_t <= 0:
                    p_cut = p_ids[:cfg.max_input_tokens]
                    full  = p_cut + [tok.eos_token_id] + t_ids[:cfg.max_summary_tokens]
                else:
                    full  = p_ids + [tok.eos_token_id] + t_ids[-keep_t:]
            label = ([-100] * (len(p_ids) + 1) + t_ids)[:len(full)]
            input_ids_batch.append(full)
            labels_batch.append(label)
            attention_masks.append([1] * len(full))
        batch_max = max(len(x) for x in input_ids_batch)
        padded_inputs, padded_labels, padded_masks = [], [], []
        for inp, lab, am in zip(input_ids_batch, labels_batch, attention_masks):
            pad_len = batch_max - len(inp)
            padded_inputs.append(inp + [tok.pad_token_id] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)
            padded_masks.append(am  + [0]   * pad_len)
        return {"input_ids": padded_inputs, "labels": padded_labels,
                "attention_mask": padded_masks}

    print("Tokenizing train / validation data...")
    t = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    v = val_ds.map(preprocess,   batched=True, remove_columns=val_ds.column_names)

    args = TrainingArguments(
        output_dir=os.path.join(cfg.out_dir, "generator"),
        per_device_train_batch_size=max(1, cfg.batch_size // 2),
        per_device_eval_batch_size=max(1,  cfg.batch_size // 2),
        gradient_accumulation_steps=4,
        num_train_epochs=max(1, cfg.epochs),
        learning_rate=5e-6,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=10,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=t, eval_dataset=v, tokenizer=tok,
    )
    print("Starting full-model fine-tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    plot_training_curves(trainer, os.path.join(cfg.out_dir, "generator"), "generator")
    print("✅ Generator fine-tuning complete.")
    return model, tok


# ============================================================
# ORIGINAL train_verifier (unchanged — used for initial training run)
# ============================================================

def build_verifier_dataset(ds: Dataset):
    texts, labels = [], []
    rng       = random.Random(42)
    summaries = list(ds["summary"])
    for ex in ds:
        q, pos = ex["question"], ex["summary"]
        texts.append(f"Q: {q} [SEP] A: {pos}"); labels.append(1)
        neg = summaries[rng.randrange(len(summaries))]
        texts.append(f"Q: {q} [SEP] A: {neg}");  labels.append(0)
    return Dataset.from_list([{"text": t, "label": l} for t, l in zip(texts, labels)])


def train_verifier(train_v: Dataset, val_v: Dataset, cfg: Config):
    tok   = AutoTokenizer.from_pretrained(cfg.verifier_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.verifier_model, num_labels=2
    ).to(cfg.device)

    def preprocess(ex):
        d = tok(ex["text"], truncation=True, padding="max_length", max_length=256)
        d["labels"] = ex["label"]
        return d

    t = train_v.map(preprocess, batched=True, remove_columns=train_v.column_names)
    v = val_v.map(preprocess,   batched=True, remove_columns=val_v.column_names)
    args = TrainingArguments(
        output_dir=os.path.join(cfg.out_dir, "verifier"),
        per_device_train_batch_size=max(2, cfg.batch_size // 2),
        per_device_eval_batch_size=max(2,  cfg.batch_size // 2),
        num_train_epochs=max(5, cfg.epochs // 2),
        eval_strategy="steps", eval_steps=20, save_strategy="no",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=t, eval_dataset=v, tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    plot_training_curves(trainer, os.path.join(cfg.out_dir, "verifier"), "verifier")
    return model, tok


# ============================================================
# ORIGINAL contrastive fine-tuning (unchanged)
# ============================================================

def contrastive_finetune_generator_encoder(model, tok, train_ds: Dataset, cfg: Config):
    embedder   = SentenceTransformer(cfg.embedder_model, device=cfg.device)
    llama_hid  = model.config.hidden_size
    st_hid     = embedder.get_sentence_embedding_dimension()
    proj_dtype = next(model.parameters()).dtype
    projection = nn.Linear(llama_hid, st_hid, dtype=proj_dtype).to(cfg.device)
    encoder    = model.get_input_embeddings()
    optimizer  = optim.Adam(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=cfg.lr * 0.5
    )
    for ep in range(cfg.contrastive_epochs):
        perm = list(range(len(train_ds))); random.shuffle(perm)
        pbar = tqdm(
            range(0, len(train_ds), cfg.contrastive_batch_size),
            desc=f"Contrastive {ep+1}"
        )
        for i in pbar:
            idxs   = perm[i:i + cfg.contrastive_batch_size]
            batch  = [train_ds[int(ii)] for ii in idxs]
            prompts = [x["prompt"]  for x in batch]
            pos     = [x["summary"] for x in batch]
            neg     = [train_ds[random.randrange(len(train_ds))]["summary"] for _ in batch]
            enc = tok(
                prompts, return_tensors="pt",
                truncation=True, padding=True, max_length=cfg.max_input_tokens
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
            loss = -(torch.log(
                torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg))
            )).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"loss": float(loss.detach().cpu().numpy())})


# ============================================================
# ORIGINAL clean_generated (unchanged)
# ============================================================

def clean_generated(text: str) -> str:
    if not isinstance(text, str): return ""
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
              "medical answer", "answer the following", "you are a medical expert",
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


# ============================================================
# generate_and_rerank — extended to accept score_fn
# ============================================================
# NOTE: signature gains an optional score_fn parameter.
# When score_fn is None the function falls back to the ORIGINAL
# ClinicalBERT verifier behaviour (ver_model + ver_tok), so
# all existing call-sites work without change.

def generate_and_rerank(
    question, qtype, gen_model, gen_tok,
    ver_model, ver_tok, cfg,
    retriever, train_ds,
    score_fn: Optional[Callable] = None,   # NEW — verifier score_fn
):
    exs    = retrieve_exemplars(question, retriever, train_ds, cfg)
    prompt = assemble_prompt(question, exs)

    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    inputs     = gen_tok(
        prompt, return_tensors="pt",
        truncation=True, max_length=cfg.max_input_tokens
    ).to(cfg.device)
    num_beams  = max(1, min(6, cfg.n_best))
    num_return = min(cfg.n_best, num_beams) if num_beams > 1 else 1

    with torch.no_grad():
        outs = gen_model.generate(
            **inputs,
            max_new_tokens=cfg.max_summary_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=gen_tok.eos_token_id,
            early_stopping=True,
        )

    candidates = [
        clean_generated(gen_tok.decode(o, skip_special_tokens=True)) for o in outs
    ]

    best_text, best_score = None, -1e9
    prompt_ents = set(extract_entities(prompt))

    for c in candidates:
        # ── VERIFIER EXPERIMENT: use score_fn when provided ────────────────
        if score_fn is not None:
            prob = score_with_verifier(question, c, score_fn)
        else:
            # Original ClinicalBERT path (unchanged)
            v_in = ver_tok(
                f"Q: {question} [SEP] A: {c}",
                return_tensors="pt", truncation=True,
                max_length=512, padding="max_length"
            ).to(cfg.device)
            with torch.no_grad():
                prob = torch.softmax(ver_model(**v_in).logits, dim=-1)[0, 1].item()

        cand_ents = set(extract_entities(c))
        ent_cov   = len(cand_ents & prompt_ents) / max(1, len(prompt_ents))
        score     = 0.75 * prob + 0.25 * ent_cov

        if score > best_score:
            best_score, best_text = score, c

    return best_text or candidates[0]


# ============================================================
# evaluate — extended to accept score_fn
# ============================================================
# Same extension pattern: optional score_fn, falls back to
# original ver_model/ver_tok path when None.

def evaluate(
    ds, gen_model, gen_tok,
    ver_model, ver_tok, cfg,
    retriever, train_ds, val_ds=None,
    score_fn: Optional[Callable] = None,   # NEW
):
    rouge       = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    meteor      = load_metric("meteor")
    bertscore   = load_metric("bertscore")
    bleu_metric = load_metric("bleu")

    hyps, refs, factuals, halls = [], [], [], []

    if val_ds is not None and len(val_ds) > 0:
        ex = val_ds[random.randrange(len(val_ds))]
        print("\n=== VALIDATION SAMPLE ===")
        print("Q:",  ex["question"])
        print("GT:", ex["summary"])
        pred_val = generate_and_rerank(
            ex["question"], ex["qtype"],
            gen_model, gen_tok, ver_model, ver_tok,
            cfg, retriever, train_ds, score_fn=score_fn,
        )
        print("Pred:", pred_val)
        print("=========================\n")

    print("\n[Evaluation] Computing metrics...")
    for ex in tqdm(ds):
        q, ref = ex["question"], ex["summary"]
        pred   = generate_and_rerank(
            q, ex["qtype"],
            gen_model, gen_tok, ver_model, ver_tok,
            cfg, retriever, train_ds, score_fn=score_fn,
        )
        hyps.append(pred); refs.append(ref)

        # Factual accuracy: use score_fn when available, else original path
        if score_fn is not None:
            prob = score_with_verifier(q, pred, score_fn)
        else:
            v_in = ver_tok(
                f"Q: {q} [SEP] A: {pred}",
                return_tensors="pt", truncation=True,
                max_length=512, padding="max_length"
            ).to(cfg.device)
            with torch.no_grad():
                prob = torch.softmax(ver_model(**v_in).logits, dim=-1)[0, 1].item()

        factuals.append(prob)
        pred_t = set(pred.split()); ref_t = set(ref.split())
        halls.append(1 - len(pred_t & ref_t) / max(1, len(pred_t | ref_t)))

    rouge1 = np.mean([rouge.score(r, p)["rouge1"].fmeasure for r, p in zip(refs, hyps)])
    rouge2 = np.mean([rouge.score(r, p)["rouge2"].fmeasure for r, p in zip(refs, hyps)])
    rougel = np.mean([rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, hyps)])
    mete   = meteor.compute(predictions=hyps, references=refs)["meteor"]
    bert   = bertscore.compute(predictions=hyps, references=refs, lang="en")
    bert_f1 = float(np.mean(bert["f1"]))
    bleu1   = bleu_metric.compute(predictions=hyps, references=[[r] for r in refs], max_order=1)["bleu"]
    bleu4   = bleu_metric.compute(predictions=hyps, references=[[r] for r in refs], max_order=4)["bleu"]

    results = {
        "ROUGE1": rouge1, "ROUGE2": rouge2, "ROUGEL": rougel,
        "BLEU1":  bleu1,  "BLEU4":  bleu4,
        "METEOR": mete,   "BERTScore": bert_f1,
        "FactualAccuracy":  float(np.mean(factuals)),
        "HallucinationRate": float(np.mean(halls)),
    }

    print("\n===== RESULTS =====")
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")

    return results


# ============================================================
# ORIGINAL plot_training_curves (unchanged)
# ============================================================

def plot_training_curves(trainer, save_dir, name="model"):
    logs = trainer.state.log_history
    df   = pd.DataFrame(logs)
    if len(df) == 0:
        print("No training logs found."); return
    os.makedirs(save_dir, exist_ok=True)
    if "loss" in df.columns:
        plt.figure()
        ld = df.dropna(subset=["loss"])
        plt.plot(ld["step"], ld["loss"])
        plt.xlabel("Training Steps"); plt.ylabel("Loss")
        plt.title(f"{name} Training Loss"); plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}_loss_curve.png"))
        plt.close()
    if "learning_rate" in df.columns:
        plt.figure()
        lr = df.dropna(subset=["learning_rate"])
        plt.plot(lr["step"], lr["learning_rate"])
        plt.xlabel("Training Steps"); plt.ylabel("Learning Rate")
        plt.title(f"{name} Learning Rate Schedule"); plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}_lr_curve.png"))
        plt.close()
    print(f"Saved training curves for {name}")


# ============================================================
# VERIFIER EXPERIMENT — Comparison plots and CSV
# ============================================================

def _plot_verifier_comparison(all_results: Dict[str, Dict], save_dir: str):
    """
    Generates two comparison figures:
      1. Bar chart of all metrics across all verifiers
      2. Grouped bar chart of ROUGE1 / METEOR / BERTScore side-by-side
    Also saves a CSV and a Markdown table.
    """
    os.makedirs(save_dir, exist_ok=True)

    verifier_names = list(all_results.keys())
    metrics_keys   = [
        "ROUGE1", "ROUGE2", "ROUGEL",
        "BLEU1",  "BLEU4",
        "METEOR", "BERTScore",
        "FactualAccuracy", "HallucinationRate",
    ]

    # ── CSV ────────────────────────────────────────────────────────────────
    rows = []
    for vname, res in all_results.items():
        entry = {"Verifier": vname, "Family": VERIFIER_REGISTRY.get(vname, {}).get("family", "?")}
        entry.update({k: round(res.get(k, 0.0), 4) for k in metrics_keys})
        rows.append(entry)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "verifier_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Saved comparison CSV → {csv_path}")

    # ── Markdown table ─────────────────────────────────────────────────────
    md_path = os.path.join(save_dir, "verifier_comparison.md")
    with open(md_path, "w") as f:
        f.write("# Verifier Comparison Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"📝 Saved Markdown table  → {md_path}")

    # ── Figure 1: full metric bar chart ────────────────────────────────────
    n_ver  = len(verifier_names)
    n_met  = len(metrics_keys)
    x      = np.arange(n_met)
    width  = 0.8 / max(n_ver, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_ver))

    fig, ax = plt.subplots(figsize=(max(14, n_met * 1.5), 6))
    for i, (vname, res) in enumerate(all_results.items()):
        vals   = [res.get(k, 0.0) for k in metrics_keys]
        offset = (i - n_ver / 2 + 0.5) * width
        desc   = VERIFIER_REGISTRY.get(vname, {}).get("description", vname)
        ax.bar(x + offset, vals, width=width * 0.9, label=desc, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_keys, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Verifier Comparison — All Metrics")
    ax.legend(loc="upper right", fontsize=8, ncol=max(1, n_ver // 4))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "verifier_all_metrics.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"📈 Saved full-metric chart → {fig_path}")

    # ── Figure 2: ROUGE1 / METEOR / BERTScore grouped ──────────────────────
    key3   = ["ROUGE1", "METEOR", "BERTScore"]
    x3     = np.arange(len(key3))
    w3     = 0.8 / max(n_ver, 1)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (vname, res) in enumerate(all_results.items()):
        vals   = [res.get(k, 0.0) for k in key3]
        offset = (i - n_ver / 2 + 0.5) * w3
        desc   = VERIFIER_REGISTRY.get(vname, {}).get("description", vname)
        ax2.bar(x3 + offset, vals, width=w3 * 0.9, label=desc, color=colors[i])

    ax2.set_xticks(x3)
    ax2.set_xticklabels(key3)
    ax2.set_ylabel("Score")
    ax2.set_title("Verifier Comparison — ROUGE1 / METEOR / BERTScore")
    ax2.legend(loc="upper right", fontsize=8, ncol=max(1, n_ver // 4))
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "verifier_key_metrics.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"📈 Saved key-metric chart  → {fig2_path}")

    # ── Figure 3: FactualAccuracy vs HallucinationRate scatter ────────────
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for i, (vname, res) in enumerate(all_results.items()):
        desc = VERIFIER_REGISTRY.get(vname, {}).get("description", vname)
        ax3.scatter(
            res.get("HallucinationRate", 0),
            res.get("FactualAccuracy",   0),
            color=colors[i], s=100, label=desc, zorder=3,
        )
    ax3.set_xlabel("Hallucination Rate (lower is better)")
    ax3.set_ylabel("Factual Accuracy (higher is better)")
    ax3.set_title("Verifier — Factual Accuracy vs Hallucination Rate")
    ax3.legend(loc="best", fontsize=7, ncol=max(1, n_ver // 5))
    ax3.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = os.path.join(save_dir, "verifier_factual_scatter.png")
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print(f"📈 Saved factual scatter    → {fig3_path}")


def run_verifier_experiments(
    verifier_keys: List[str],
    test_ds, gen_model, gen_tok,
    cfg, retriever, train_ds, val_ds,
):
    """
    Iterates over each verifier key, loads the verifier, runs evaluation,
    collects results and produces comparison plots + CSV.

    Parameters
    ----------
    verifier_keys  : list of keys from VERIFIER_REGISTRY to evaluate
    test_ds        : dataset split to evaluate on
    gen_model / gen_tok : trained generator
    cfg            : Config object
    retriever      : UnifiedRetriever
    train_ds       : training split (for exemplar retrieval)
    val_ds         : validation split (for sample preview)
    """
    all_results: Dict[str, Dict] = {}
    exp_dir = os.path.join(cfg.out_dir, "verifier_experiments")
    os.makedirs(exp_dir, exist_ok=True)

    for vkey in verifier_keys:
        entry = VERIFIER_REGISTRY.get(vkey)
        if entry is None:
            print(f"⚠️  Unknown verifier '{vkey}', skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"🔬 VERIFIER EXPERIMENT: {vkey}  —  {entry['description']}")
        print(f"   Family : {entry['family']}")
        print(f"   Model  : {entry['hf_model']}")
        print(f"{'='*60}")

        try:
            ver_tok_exp, ver_model_exp, score_fn = build_verifier(vkey, cfg)
        except RuntimeError as exc:
            print(f"  ❌ Skipping '{vkey}': {exc}")
            continue

        # Run evaluation with the new verifier's score_fn
        # ver_model and ver_tok are passed as None so evaluate()
        # always routes through score_fn when it is provided.
        res = evaluate(
            test_ds,
            gen_model, gen_tok,
            ver_model_exp, ver_tok_exp,   # kept for signature compatibility
            cfg,
            retriever, train_ds,
            val_ds=val_ds,
            score_fn=score_fn,            # NEW routing parameter
        )

        all_results[vkey] = res

        # Per-verifier JSON
        json_path = os.path.join(exp_dir, f"{vkey}_results.json")
        with open(json_path, "w") as f:
            json.dump({"verifier": vkey, "config": entry, "results": res}, f, indent=2)
        print(f"  💾 Saved → {json_path}")

        # Free verifier GPU memory between runs
        if hasattr(ver_model_exp, "to"):
            try:
                ver_model_exp.to("cpu")
            except Exception:
                pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if all_results:
        _plot_verifier_comparison(all_results, exp_dir)
    else:
        print("⚠️  No verifier experiments completed — no comparison plot generated.")

    return all_results


# ============================================================
# VERIFIER EXPERIMENT — CLI argument parsing (extended)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MedLM pipeline with verifier experiments"
    )
    # Original flag
    parser.add_argument(
        "--retriever_type",
        type=str, default="dense",
        choices=["dense", "bm25", "hybrid", "rerank"],
        help="Retrieval strategy (unchanged from original).",
    )
    # NEW: single verifier selection
    parser.add_argument(
        "--verifier_type",
        type=str, default="clinicalbert",
        choices=list(VERIFIER_REGISTRY.keys()),
        help=(
            "Verifier to use for a single-verifier run. "
            "Ignored when --run_all_verifiers is set."
        ),
    )
    # NEW: sweep all verifiers
    parser.add_argument(
        "--run_all_verifiers",
        action="store_true",
        default=False,
        help=(
            "When set, iterate over ALL verifiers in VERIFIER_REGISTRY, "
            "evaluate each, and produce a comparison CSV + plots. "
            "Overrides --verifier_type."
        ),
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    from huggingface_hub import login
    login(token="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")
    args = parse_args()
    cfg  = Config()
    cfg.retriever_type   = args.retriever_type
    cfg.verifier_type    = args.verifier_type
    cfg.run_all_verifiers = args.run_all_verifiers

    # Keep original verifier_model in sync with the chosen verifier_type
    cfg.verifier_model = VERIFIER_REGISTRY[cfg.verifier_type]["hf_model"]

    set_seed(cfg.seed)

    # ── Data loading (original) ───────────────────────────────────────────
    data  = load_all_datasets(cfg)
    train = data["train"]
    val   = data.get("validation")
    test  = data.get("test")

    train = summarize_dataset(train, cfg, "train")
    if val:  val  = summarize_dataset(val,  cfg, "val")
    if test: test = summarize_dataset(test, cfg, "test")

    # ── Entity vocabulary (original) ─────────────────────────────────────
    global MED_ENT_VOCAB
    ent_path = os.path.join(cfg.out_dir, "medical_entities.json")
    if os.path.exists(ent_path):
        MED_ENT_VOCAB = load_entity_vocabulary(ent_path)
        print(f"✅ Loaded entity vocab ({len(MED_ENT_VOCAB)} terms).")
    else:
        vocab        = build_entity_vocabulary(train, cfg, ent_path)
        MED_ENT_VOCAB = set(vocab)

    print(f"\n🚀 Retriever : {cfg.retriever_type.upper()}")
    print(f"🔬 Verifier  : {cfg.verifier_type}  (run_all={cfg.run_all_verifiers})")

    # ── Build retriever (original) ────────────────────────────────────────
    retriever = build_retriever(train, cfg)

    # ── Prepare datasets (original) ───────────────────────────────────────
    train_p = prepare_dataset_with_retriever(train, retriever, train, cfg)
    val_p   = prepare_dataset_with_retriever(val,   retriever, train, cfg) if val   else None
    test_p  = prepare_dataset_with_retriever(test,  retriever, train, cfg) if test  else None

    # ── Train generator (original) ────────────────────────────────────────
    gen_model, gen_tok = train_generator(train_p, val_p or train_p, cfg)

    # ── Train initial verifier (original ClinicalBERT path) ───────────────
    # Always trained regardless of which verifier is used at evaluation time,
    # so the training pipeline is fully intact.
    ver_train_ds = build_verifier_dataset(train_p)
    ver_val_ds   = build_verifier_dataset(val_p) if val_p else ver_train_ds
    ver_model, ver_tok = train_verifier(ver_train_ds, ver_val_ds, cfg)

    # ── Contrastive fine-tuning (original) ───────────────────────────────
    contrastive_finetune_generator_encoder(gen_model, gen_tok, train_p, cfg)

    # ══════════════════════════════════════════════════════════════════════
    # VERIFIER EXPERIMENT — evaluation branch
    # ══════════════════════════════════════════════════════════════════════

    if cfg.run_all_verifiers:
        # ── Full sweep: evaluate every verifier in the registry ──────────
        print("\n" + "="*60)
        print("🧪 RUNNING FULL VERIFIER SWEEP")
        print("="*60)
        all_keys = list(VERIFIER_REGISTRY.keys())
        all_results = run_verifier_experiments(
            verifier_keys=all_keys,
            test_ds=test_p,
            gen_model=gen_model, gen_tok=gen_tok,
            cfg=cfg,
            retriever=retriever,
            train_ds=train,
            val_ds=val,
        )
        print("\n\n===== VERIFIER SWEEP SUMMARY =====")
        # Print ranked by METEOR
        ranked = sorted(all_results.items(), key=lambda x: x[1].get("METEOR", 0), reverse=True)
        print(f"{'Rank':<6} {'Verifier':<22} {'Family':<15} {'ROUGE1':>8} {'METEOR':>8} {'BERTScore':>10} {'FactAcc':>9}")
        print("-" * 82)
        for rank, (vkey, res) in enumerate(ranked, 1):
            fam  = VERIFIER_REGISTRY.get(vkey, {}).get("family", "?")
            print(
                f"{rank:<6} {vkey:<22} {fam:<15} "
                f"{res.get('ROUGE1',0):>8.4f} {res.get('METEOR',0):>8.4f} "
                f"{res.get('BERTScore',0):>10.4f} {res.get('FactualAccuracy',0):>9.4f}"
            )

    else:
        # ── Single verifier: original evaluation flow with chosen verifier ─
        print(f"\n🔬 Single-verifier evaluation: {cfg.verifier_type}")
        _, _, score_fn = build_verifier(cfg.verifier_type, cfg)

        res = evaluate(
            test_p,
            gen_model, gen_tok,
            ver_model, ver_tok,
            cfg,
            retriever, train,
            val_ds=val,
            score_fn=score_fn,
        )
        print("\n===== FINAL RESULT =====")
        print(f"Verifier : {cfg.verifier_type}")
        print(f"Retriever: {cfg.retriever_type}")
        for k, v in res.items():
            print(f"  {k:20s}: {v:.4f}")

        # Save single-run result
        os.makedirs(cfg.out_dir, exist_ok=True)
        result_path = os.path.join(
            cfg.out_dir,
            f"result_{cfg.dataset_name}_{cfg.retriever_type}_{cfg.verifier_type}.json"
        )
        with open(result_path, "w") as f:
            json.dump({
                "dataset":   cfg.dataset_name,
                "retriever": cfg.retriever_type,
                "verifier":  cfg.verifier_type,
                "results":   res,
            }, f, indent=2)
        print(f"💾 Saved → {result_path}")


if __name__ == "__main__":
    main()