# file: medlm_llm_pipeline.py
"""
Full MedLM-LLaMA3 pipeline (ready-to-run)
- Summarizes answers via LLaMA3/Mixtral (cached)
- Builds per-type FAISS indexes
- Uses exemplars in medical reasoning prompts
- Generator: LLaMA3-8B-Instruct (fine-tuned via Trainer)
- Verifier: BioMed-RoBERTa/ClinicalBERT (fine-tuned)
- Contrastive encoder fine-tuning (projection -> ST dims)
- Final eval: ROUGE, METEOR, BERTScore, factual accuracy, hallucination rate

Notes:
- Install required packages if missing:
    pip install transformers accelerate datasets evaluate rouge_score sentence-transformers faiss-cpu scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
- Models used (large LLMs) require GPU and adequate memory. If you can't load LLaMA3, switch to smaller generator_model in Config.
"""
import os
import json
import random
import math
import re
from dataclasses import dataclass
from typing import List, Dict
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

import matplotlib.pyplot as plt
import pandas as pd

# scispaCy for biomedical entity extraction
try:
    import spacy
except Exception as e:
    raise RuntimeError("Please install scispacy & spacy: pip install scispacy spacy\n"
                       "and the scispacy model: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz")

# --------------------
# Config
# --------------------
@dataclass
class Config:
    dataset_name: str = "icliniq" # "medquad" or "icliniq"

    # # add this inside Config
    # retriever_type: str = "dense"
    # retriever_types: list = ("dense", "bm25", "hybrid", "rerank")

    medquad_dir: str = "SET_THIS_PATH"
    icliniq_dir: str = "SET_THIS_PATH"
    out_dir: str = "SET_THIS_PATH"
    summarizer_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   #mistralai/Mixtral-8x7B-Instruct-v0.1
    generator_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"   #deepseek-ai/DeepSeek-R1-Distill-Llama-8B     meta-llama/Meta-Llama-3-8B-Instruct     Qwen/Qwen2-7B-Instruct
    verifier_model: str = "medicalai/ClinicalBERT"   # change if you prefer other biomedical verifier
    embedder_model: str = "SET_THIS_PATH"
    per_type_index_dir: str = "SET_THIS_PATH"

    max_summary_tokens: int = 200
    max_input_tokens: int = 512
    n_exemplars: int = 3
    n_best: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    batch_size: int = 8
    epochs: int = 3
    lr: float = 3e-5
    contrastive_epochs: int = 2
    contrastive_batch_size: int = 8
    contrastive_temp: float = 0.07
    verifier_threshold: float = 0.5

    use_subset: bool = True
    subset_train_size: int = 500
    subset_val_size: int = 200
    subset_test_size: int = 300

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever_type",
        type=str,
        default="dense",
        choices=["dense", "bm25", "hybrid", "rerank"],
    )
    return parser.parse_args()

# --------------------

# --------------------
class UnifiedRetriever:
    def __init__(self, ds, cfg):
        self.cfg = cfg
        self.questions = list(ds["question"])
        self.summaries = list(ds["summary"])

        assert cfg.retriever_type in ["dense", "bm25", "hybrid", "rerank"]

        self.embedder = None
        self.index = None
        self.bm25 = None
        self.cross_encoder = None

        if cfg.retriever_type in ["dense", "hybrid", "rerank"]:
            self.embedder = SentenceTransformer(
                cfg.embedder_model,
                device=cfg.device
            )
            self.embs = self.embedder.encode(
                self.questions,
                convert_to_numpy=True
            )
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
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=cfg.device
            )

    def search(self, query, topk):
        if self.cfg.retriever_type == "dense":
            return self._dense(query, topk)

        if self.cfg.retriever_type == "bm25":
            return self._bm25(query, topk)

        if self.cfg.retriever_type == "hybrid":
            return self._hybrid(query, topk)

        if self.cfg.retriever_type == "rerank":
            return self._rerank(query, topk)

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

        D, I = self.index.search(q_emb, topk * 2)
        dense_scores = D[0]
        dense_idx = I[0]

        bm25_scores = self.bm25.get_scores(query.lower().split())

        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() + 1e-6)
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) + 1e-6)

        combined = {}
        for i, idx in enumerate(dense_idx):
            combined[idx] = dense_scores[i] + bm25_scores[idx]

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked[:topk]]

    def _rerank(self, query, topk):
        base = self._dense(query, topk * 3)

        pairs = [(query, self.questions[i]) for i in base]
        scores = self.cross_encoder.predict(pairs)

        ranked = sorted(zip(base, scores), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked[:topk]]

# --------------------
# Utilities
# --------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except:
                continue
    return out

# --------------------
# Dataset loading
# --------------------
def load_dataset_from_dir(data_dir: str):
    splits = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(p):
            recs = read_jsonl(p)
            cleaned = []
            for r in recs:
                q = r.get("question","").strip()
                a = r.get("answer","").strip()
                if q and a:
                    qt = r.get("qtype", r.get("type", "general"))
                    cleaned.append({"question": q, "answer": a, "qtype": qt})
            splits[split] = Dataset.from_list(cleaned)
    return DatasetDict(splits)

# def load_all_datasets(cfg: Config):
#     med = load_dataset_from_dir(cfg.medquad_dir)
#     ic = load_dataset_from_dir(cfg.icliniq_dir)
#     merged = {}
#     for s in ["train","validation","test"]:
#         parts = []
#         if s in med: parts.append(med[s])
#         if s in ic: parts.append(ic[s])
#         if parts:
#             merged[s] = Dataset.from_list(sum([list(d) for d in parts], []))
#             if cfg.use_subset:
#                 full = len(merged[s])
#                 if s == "train":
#                     subsz = min(cfg.subset_train_size, full)
#                 elif s == "validation":
#                     subsz = min(cfg.subset_val_size, full)
#                 else:
#                     subsz = min(cfg.subset_test_size, full)
#                 merged[s] = merged[s].select(range(subsz))
#                 print(f"⚙️ Using {subsz}/{full} samples for {s}")
#     print("Dataset sizes:", {k: len(v) for k,v in merged.items()})
#     return DatasetDict(merged)

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
                if s == "train":
                    subsz = min(cfg.subset_train_size, full)
                elif s == "validation":
                    subsz = min(cfg.subset_val_size, full)
                else:
                    subsz = min(cfg.subset_test_size, full)

                merged[s] = merged[s].select(range(subsz))
                print(f"⚙️ Using {subsz}/{full} samples for {s}")

    print("Dataset sizes:", {k: len(v) for k, v in merged.items()})
    return DatasetDict(merged)

# --------------------
# Summarization with caching
# --------------------
def summarize_dataset(ds: Dataset, cfg: Config, split_name: str = "train"):
    """
    Summarize dataset answers using LLaMA-style summarizer and cache cleaned summaries.
    If cached file exists and matches dataset length, it is reused.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, f"{split_name}_summaries_icliniq500.json")

    def _clean_summary(s: str) -> str:
        if s is None: return ""
        s = s.strip()
        prefixes = [
            "summary:", "summary -", "summary :", "summarize the following",
            "summarize:", "summarize :", "answer:", "summary\n:"
        ]
        sl = s.lower()
        for p in prefixes:
            if sl.startswith(p):
                s = s[len(p):].strip()
                sl = s.lower()
        # collapse repeated tokens
        tokens = s.split()
        out_tokens = []
        prev = None
        repeat = 0
        for t in tokens:
            if t == prev:
                repeat += 1
            else:
                repeat = 0
            if repeat < 3:
                out_tokens.append(t)
            prev = t
        s = " ".join(out_tokens)
        s = " ".join(s.split())
        return s

    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) == len(ds):
                print(f"✅ Loaded cached cleaned summaries for {split_name} ({len(cached)} entries).")
                return ds.add_column("summary", cached)
            else:
                print(f"⚠️ Cache length mismatch ({len(cached)} vs {len(ds)}). Regenerating...")
        except Exception as e:
            print(f"⚠️ Failed loading cache ({e}). Regenerating...")

    print(f"🩺 Generating new summaries for {split_name} split...")
    tok = AutoTokenizer.from_pretrained(cfg.summarizer_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.summarizer_model, torch_dtype=torch.float16, device_map="auto")

    summaries = []
    for ex in tqdm(ds, desc=f"Summarizing {split_name}"):
        prompt = (
            "Summarize the following medical answer factually and concisely (limit ~100 tokens):\n\n"
            f"{ex['answer']}\n\nSummary:"
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_tokens, padding=True).to(cfg.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_summary_tokens,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
                do_sample=False
            )
        raw = tok.decode(out[0], skip_special_tokens=True)
        cleaned = _clean_summary(raw)
        summaries.append(cleaned)

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"💾 Saved {len(summaries)} cleaned summaries → {save_path}")
    except Exception as e:
        print(f"⚠️ Failed saving summaries: {e}")

    return ds.add_column("summary", summaries)

# --------------------
# scispaCy biomedical entity extraction + vocabulary caching
# --------------------
# You must install the scispacy NER model:
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
try:
    NLP_MED = spacy.load("en_ner_bc5cdr_md")
except Exception as e:
    raise RuntimeError("scispaCy model not found. Install with:\n"
                       "pip install scispacy\n"
                       "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz")

def extract_medical_entities(text: str):
    if not text or not isinstance(text, str):
        return []
    doc = NLP_MED(text)
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
    print("🔍 Extracting medical entities from dataset (this may take some time)...")
    for ex in tqdm(dataset, desc="Extracting concepts"):
        text = ex.get("summary") or ex.get("answer") or ""
        if text:
            ents = extract_medical_entities(text)
            all_ents.update(ents)
    all_ents = sorted(list(all_ents))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_ents, f, indent=2)
    with open(path.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        for e in all_ents: f.write(e + "\n")
    print(f"✅ Extracted {len(all_ents)} unique entities → {path}")
    return all_ents

def load_entity_vocabulary(path):
    if not os.path.exists(path):
        return set()
    return set(json.load(open(path, "r", encoding="utf-8")))

# fast matching extractor using prebuilt vocabulary (used in scoring)
MED_ENT_VOCAB = set()  # will be populated in main
def extract_entities(text: str):
    if not text:
        return []
    tokens = [t.strip(".,;:()").lower() for t in re.split(r"\s+", text)]
    found = set()
    for t in tokens:
        if t in MED_ENT_VOCAB:
            found.add(t)
    return sorted(list(found))

def add_entities_column(ds: Dataset):
    ents = [extract_entities(x["summary"]) for x in ds]
    return ds.add_column("entities", ents)

# --------------------
# Build per-type FAISS indexes
# --------------------
def build_retriever(train_ds, cfg):
    return UnifiedRetriever(train_ds, cfg)


def retrieve_exemplars(question, retriever, train_ds, cfg):
    idxs = retriever.search(question, cfg.n_exemplars)

    return [
        {
            "question": train_ds[i]["question"],
            "summary": train_ds[i]["summary"]
        }
        for i in idxs
    ]

# --------------------
# Prompt assembly
# --------------------
def assemble_prompt(question: str, exemplars: List[Dict]):
    parts = ["You are a medical expert.\nUse clinical reasoning to answer concisely.\nAnswer must be related to the question.\n"]
    for ex in exemplars:
        parts.append(f"Q: {ex['question']}\nA: {ex['summary']}\n---\n")
    parts.append(f"Q: {question}\nA:")
    return "\n".join(parts)

def prepare_dataset_with_retriever(ds, retriever, train_ds, cfg):
    prompts = []
    for ex in ds:
        exs = retrieve_exemplars(
            ex["question"],
            retriever,
            train_ds,
            cfg
        )
        prompts.append(
            assemble_prompt(ex["question"], exs)
        )
    return ds.add_column("prompt", prompts)

# --------------------
# Train generator (causal LM Trainer)
# --------------------
def train_generator(train_ds: Dataset, val_ds: Dataset, cfg: Config):
    """
    Full model fine-tuning for LLaMA-style causal LM.
    Correctly builds concatenated input sequences and masks prompt tokens with -100.
    """

    tok = AutoTokenizer.from_pretrained(cfg.generator_model)
    # ensure pad token exists
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model (full fine-tuning)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.generator_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Preprocess: build "prompt || answer" input_ids and labels (mask prompt)
    def preprocess(batch):
        prompts = list(batch["prompt"])
        targets = list(batch["summary"])

        # tokenize prompts and targets separately so we can mask prompts
        p_tok = tok(prompts, truncation=True, padding=False, add_special_tokens=False)
        t_tok = tok(targets, truncation=True, padding=False, add_special_tokens=False)

        input_ids_batch = []
        labels_batch = []
        attention_masks = []

        max_total = cfg.max_input_tokens + cfg.max_summary_tokens

        for p_ids, t_ids in zip(p_tok["input_ids"], t_tok["input_ids"]):
            # include a separator if tokenizer doesn't add it automatically
            # many LLMs don't use special separator tokens; we'll just concatenate with eos
            full = p_ids + [tok.eos_token_id] + t_ids
            if len(full) > max_total:
                # truncate target first (we want to keep prompt), so keep last tokens of target
                keep_t = max_total - (len(p_ids) + 1)
                if keep_t <= 0:
                    # prompt itself too long: truncate prompt to max_input_tokens and discard target prefix
                    p_cut = p_ids[:cfg.max_input_tokens]
                    # recompute full
                    full = p_cut + [tok.eos_token_id] + t_ids[: cfg.max_summary_tokens]
                else:
                    full = p_ids + [tok.eos_token_id] + t_ids[-keep_t:]
            # create labels: mask prompt + eos with -100, keep target tokens
            label = [-100] * (len(p_ids) + 1) + t_ids
            # trim label to full length (in case full was truncated)
            label = label[: len(full)]

            input_ids_batch.append(full)
            labels_batch.append(label)
            attention_masks.append([1] * len(full))

        # Now pad to max length in batch
        batch_max = max(len(x) for x in input_ids_batch)
        padded_inputs = []
        padded_labels = []
        padded_masks = []
        for inp, lab, am in zip(input_ids_batch, labels_batch, attention_masks):
            pad_len = batch_max - len(inp)
            padded_inputs.append(inp + [tok.pad_token_id] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)
            padded_masks.append(am + [0] * pad_len)

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": padded_masks
        }

    print("Tokenizing train and validation data...")
    t = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    v = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    # TrainingArguments: conservative for full FT
    args = TrainingArguments(
        output_dir=os.path.join(cfg.out_dir, "generator"),
        per_device_train_batch_size=max(1, cfg.batch_size // 2),  # keep safe memory
        per_device_eval_batch_size=max(1, cfg.batch_size // 2),
        gradient_accumulation_steps=4,
        num_train_epochs=max(1, cfg.epochs),
        learning_rate=5e-6,  # conservative LR for full FT
        bf16=torch.cuda.is_available(),  # use bf16 where possible
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
        model=model,
        args=args,
        train_dataset=t,
        eval_dataset=v,
        tokenizer=tok,
    )

    print("Starting full-model fine-tuning (this can take a long time)...")
    trainer.train()
    trainer.save_model(args.output_dir)

    # 🔵 Plot curves
    plot_training_curves(
        trainer,
        os.path.join(cfg.out_dir, "generator"),
        "generator"
    )

    print("✅ Generator fine-tuning complete.")
    return model, tok

# --------------------
# Train verifier
# --------------------
def build_verifier_dataset(ds: Dataset):
    texts, labels = [], []
    rng = random.Random(42)
    summaries = list(ds["summary"])
    for ex in ds:
        q, pos = ex["question"], ex["summary"]
        texts.append(f"Q: {q} [SEP] A: {pos}"); labels.append(1)
        neg = summaries[rng.randrange(len(summaries))]
        texts.append(f"Q: {q} [SEP] A: {neg}"); labels.append(0)
    return Dataset.from_list([{"text":t,"label":l} for t,l in zip(texts,labels)])

def train_verifier(train_v: Dataset, val_v: Dataset, cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.verifier_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.verifier_model, num_labels=2).to(cfg.device)

    def preprocess(ex):
        d = tok(ex["text"], truncation=True, padding="max_length", max_length=256)
        d["labels"] = ex["label"]
        return d

    t = train_v.map(preprocess, batched=True, remove_columns=train_v.column_names)
    v = val_v.map(preprocess, batched=True, remove_columns=val_v.column_names)
    args = TrainingArguments(output_dir=os.path.join(cfg.out_dir, "verifier"), per_device_train_batch_size=max(2, cfg.batch_size//2), per_device_eval_batch_size=max(2, cfg.batch_size//2), num_train_epochs=max(5, cfg.epochs//2), eval_strategy="steps", eval_steps=20, save_strategy="no")
    trainer = Trainer(model=model, args=args, train_dataset=t, eval_dataset=v, tokenizer=tok)
    trainer.train()
    trainer.save_model(args.output_dir)

    # 🔵 Plot curves
    plot_training_curves(
        trainer,
        os.path.join(cfg.out_dir, "verifier"),
        "verifier"
    )

    return model, tok

# --------------------
# Contrastive fine-tuning (with projection)
# --------------------
def contrastive_finetune_generator_encoder(model, tok, train_ds: Dataset, cfg: Config):
    embedder = SentenceTransformer(cfg.embedder_model, device=cfg.device)
    llama_hidden = model.config.hidden_size
    st_hidden = embedder.get_sentence_embedding_dimension()

    proj_dtype = next(model.parameters()).dtype
    projection = nn.Linear(llama_hidden, st_hidden, dtype=proj_dtype).to(cfg.device)
    encoder = model.get_input_embeddings()
    optimizer = optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=cfg.lr * 0.5)

    for ep in range(cfg.contrastive_epochs):
        perm = list(range(len(train_ds)))
        random.shuffle(perm)
        pbar = tqdm(range(0, len(train_ds), cfg.contrastive_batch_size), desc=f"Contrastive {ep+1}")
        for i in pbar:
            idxs = perm[i:i+cfg.contrastive_batch_size]
            batch = [train_ds[int(ii)] for ii in idxs]
            prompts = [x["prompt"] for x in batch]
            pos = [x["summary"] for x in batch]
            neg = [train_ds[random.randrange(len(train_ds))]["summary"] for _ in batch]

            enc = tok(prompts, return_tensors="pt", truncation=True, padding=True, max_length=cfg.max_input_tokens).to(cfg.device)
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

            loss = -torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg))).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"loss": float(loss.detach().cpu().numpy())})

# --------------------
# Clean generated text
# --------------------
def clean_generated(text: str) -> str:
    """
    Cleans LLaMA/Mixtral generation artifacts:
    - Removes prompt leakage (the model echoing the prompt)
    - Removes repeated patterns / loops (heuristic)
    - Removes obvious broken-unicode noise (but preserves medical characters like β, µ)
    - Ensures answer starts cleanly
    """
    if not isinstance(text, str):
        return ""

    s = text.strip()

    # If the model echoed the full prompt with "Q:" / "A:", try to keep only the answer.
    # Keep last occurrence of "\nA:" or "A:" marker.
    lower = s.lower()
    if "\nA:" in lower:
        # find last case-insensitive "\nA:" and slice after it
        idx = max(lower.rfind("\nA:"), lower.rfind("\na:"))
        s = s[idx + 3 :].strip()
    elif " a:" in lower and lower.count("q:") > 0:
        # fallback: cut after the last " A:" if there is a Q:
        idx = lower.rfind(" a:")
        s = s[idx + 3 :].strip()

    # Remove leading instruction fragments that often leak
    leak_prefixes = [
        "summary:", "summarize:", "summarize the following",
        "medical answer", "answer the following", "you are a medical expert",
        "medical answer factually", "medical answer concisely"
    ]
    low = s.lower()
    for p in leak_prefixes:
        if low.startswith(p):
            s = s[len(p):].strip()
            low = s.lower()

    # Remove long runs of repeated tokens ("more more more", "States States ...")
    import re
    s = re.sub(r"\b(\w+)( \1){3,}\b", r"\1", s)

    # Remove very long lines of nonsensical repeated punctuation/unicode junk
    s = re.sub(r"[\uFEFF\uFFFD]+", "", s)  # remove replacement characters
    s = re.sub(r"[^ -~\u00A0-\u024F\u1E00-\u1EFF\n\r]+", lambda m: m.group(0), s)  # keep unicode (no-op but explicit)

    # Remove isolated weird sequences like '://', leftover html-ish fragments
    s = s.replace("://", " ")
    s = s.replace("  ", " ")
    s = " ".join(s.split())  # normalize whitespace

    # Truncate obviously corrupted tails (very long repeated single-char tails)
    if len(s) > 2000:
        s = s[:2000].rsplit(".", 1)[0]  # cut at last full sentence if possible

    # Final strip and return
    return s.strip()

# =========================
# generate_and_rerank
# =========================
def generate_and_rerank(question, qtype, gen_model, gen_tok, ver_model, ver_tok, cfg, retriever, train_ds):
    exs = retrieve_exemplars(question, retriever, train_ds, cfg)

    prompt = assemble_prompt(question, exs)

    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_tokens).to(cfg.device)

    num_beams = max(1, min(6, cfg.n_best))
    num_return = min(cfg.n_best, num_beams) if num_beams > 1 else 1

    with torch.no_grad():
        outs = gen_model.generate(
            **inputs,
            max_new_tokens=cfg.max_summary_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=gen_tok.eos_token_id,
            early_stopping=True,
        )

    candidates = [clean_generated(gen_tok.decode(o, skip_special_tokens=True)) for o in outs]

    best_text, best_score = None, -1e9
    prompt_ents = set(extract_entities(prompt))

    for c in candidates:
        v_in = ver_tok(
                f"Q: {question} [SEP] A: {c}",
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            ).to(cfg.device)
        with torch.no_grad():
            prob = torch.softmax(ver_model(**v_in).logits, dim=-1)[0, 1].item()

        cand_ents = set(extract_entities(c))
        ent_cov = len(cand_ents & prompt_ents) / max(1, len(prompt_ents))

        score = 0.75 * prob + 0.25 * ent_cov

        if score > best_score:
            best_score, best_text = score, c

    return best_text or candidates[0]

# =========================
# evaluate
# =========================
def evaluate(ds, gen_model, gen_tok, ver_model, ver_tok, cfg, retriever, train_ds, val_ds=None):
    rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    meteor = load_metric("meteor")
    bertscore = load_metric("bertscore")
    bleu_metric = load_metric("bleu")

    hyps, refs = [], []
    factuals, halls = [], []


    if val_ds is not None and len(val_ds) > 0:
        ex = val_ds[random.randrange(len(val_ds))]
        print("\n=== VALIDATION SAMPLE ===")
        print("Q:", ex["question"])
        print("GT:", ex["summary"])

        pred_val = generate_and_rerank(
            ex["question"],
            ex["qtype"],
            gen_model,
            gen_tok,
            ver_model,
            ver_tok,
            cfg,
            retriever,
            train_ds
        )

        print("Pred:", pred_val)
        print("========================\n")

    print("\n[Evaluation] Computing metrics...")

    for ex in tqdm(ds):
        q, ref = ex["question"], ex["summary"]

        pred = generate_and_rerank(
            q,
            ex["qtype"],
            gen_model,
            gen_tok,
            ver_model,
            ver_tok,
            cfg,
            retriever,
            train_ds
        )

        hyps.append(pred)
        refs.append(ref)

        v_in = ver_tok(
            f"Q: {q} [SEP] A: {pred}",
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(cfg.device)
        with torch.no_grad():
            prob = torch.softmax(ver_model(**v_in).logits, dim=-1)[0,1].item()

        factuals.append(prob)

        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())

        halls.append(1 - len(pred_tokens & ref_tokens) / max(1, len(pred_tokens | ref_tokens)))

    rouge1 = np.mean([rouge.score(r,p)["rouge1"].fmeasure for r,p in zip(refs,hyps)])
    rouge2 = np.mean([rouge.score(r,p)["rouge2"].fmeasure for r,p in zip(refs,hyps)])
    rougel = np.mean([rouge.score(r,p)["rougeL"].fmeasure for r,p in zip(refs,hyps)])

    mete = meteor.compute(predictions=hyps, references=refs)["meteor"]
    bert = bertscore.compute(predictions=hyps, references=refs, lang="en")
    bert_f1 = float(np.mean(bert["f1"]))

    bleu1 = bleu_metric.compute(predictions=hyps, references=[[r] for r in refs], max_order=1)["bleu"]
    bleu4 = bleu_metric.compute(predictions=hyps, references=[[r] for r in refs], max_order=4)["bleu"]

    print("\n===== RESULTS =====")
    print(f"ROUGE1: {rouge1:.4f}")
    print(f"ROUGE2: {rouge2:.4f}")
    print(f"ROUGEL: {rougel:.4f}")
    print(f"BLEU1 : {bleu1:.4f}")
    print(f"BLEU4 : {bleu4:.4f}")
    print(f"METEOR: {mete:.4f}")
    print(f"BERT  : {bert_f1:.4f}")
    print(f"FACT  : {np.mean(factuals):.4f}")
    print(f"HALL  : {np.mean(halls):.4f}")

    return {
        "ROUGE1": rouge1,
        "ROUGE2": rouge2,
        "ROUGEL": rougel,
        "BLEU1": bleu1,
        "BLEU4": bleu4,
        "METEOR": mete,
        "BERTScore": bert_f1,
        "FactualAccuracy": float(np.mean(factuals)),
        "HallucinationRate": float(np.mean(halls)),
    }



# -------------------
# GRAPHS
# -------------------

def plot_training_curves(trainer, save_dir, name="model"):
    """
    Plot loss and learning rate curves from HuggingFace Trainer logs
    """

    logs = trainer.state.log_history
    df = pd.DataFrame(logs)

    if len(df) == 0:
        print("No training logs found.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # -------------------
    # LOSS CURVE
    # -------------------
    if "loss" in df:
        plt.figure()
        loss_df = df.dropna(subset=["loss"])
        plt.plot(loss_df["step"], loss_df["loss"])
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title(f"{name} Training Loss")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}_loss_curve.png"))
        plt.close()

    # -------------------
    # LEARNING RATE CURVE
    # -------------------
    if "learning_rate" in df:
        plt.figure()
        lr_df = df.dropna(subset=["learning_rate"])
        plt.plot(lr_df["step"], lr_df["learning_rate"])
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title(f"{name} Learning Rate Schedule")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}_lr_curve.png"))
        plt.close()

    print(f"Saved training curves for {name}")
# --------------------
# Main
# --------------------
def main():
    args = parse_args()
    cfg = Config()
    cfg.retriever_type = args.retriever_type

    set_seed(cfg.seed)

    data = load_all_datasets(cfg)
    train, val, test = data["train"], data.get("validation"), data.get("test")

    train = summarize_dataset(train, cfg, "train")
    if val:
        val = summarize_dataset(val, cfg, "val")
    if test:
        test = summarize_dataset(test, cfg, "test")

    print(f"\n🚀 Using Retriever: {cfg.retriever_type.upper()}")

    # BUILD RETRIEVER ONCE
    retriever = build_retriever(train, cfg)

    # PREPARE DATASETS (STRICT)
    train_p = prepare_dataset_with_retriever(train, retriever, train, cfg)
    val_p = prepare_dataset_with_retriever(val, retriever, train, cfg) if val else None
    test_p = prepare_dataset_with_retriever(test, retriever, train, cfg) if test else None

    # TRAIN MODELS
    gen_model, gen_tok = train_generator(train_p, val_p or train_p, cfg)

    ver_model, ver_tok = train_verifier(
        build_verifier_dataset(train_p),
        build_verifier_dataset(val_p) if val_p else None,
        cfg
    )

    contrastive_finetune_generator_encoder(gen_model, gen_tok, train_p, cfg)

    # EVALUATE (USES SAME RETRIEVER)
    res = evaluate(
        test_p,
        gen_model,
        gen_tok,
        ver_model,
        ver_tok,
        cfg,
        retriever,
        train,
        val_ds=val
    )

    print("\n===== FINAL RESULT =====")
    print(f"{cfg.retriever_type}: {res}")


if __name__ == "__main__":
    main()