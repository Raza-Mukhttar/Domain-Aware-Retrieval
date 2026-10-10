#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase B - Prompt Construction and Adaptation (Section 3.3 of the paper).

Two jobs:

  1. Prompt construction, Eq. (9). For every row the k nearest exemplars are
     retrieved from the Phase A index and assembled into
     Prompt(q) = {(q_i1, s_i1), ..., (q_ik, s_ik), q}. An exemplar is never
     allowed to be the row itself.

  2. Generator adaptation. Full fine-tuning on those prompts with the
     autoregressive objective of Eq. (10): prompt tokens are masked to -100, so
     the loss is taken on the answer only. bf16, AdamW, linear schedule, warm-up
     0.03, 3 epochs, effective batch 16 - the executed configuration.

The prompt cap keeps the RIGHT side of the prompt (--truncation_side left), so a
prompt longer than the cap loses exemplars rather than the target question. The
submitted runs used the opposite side; --truncation_side right reproduces them.

Usage:
  python phase_b_prompt_and_adaptation.py --dataset medquad \
      --base <hub-id-or-path> --label llama31 --seed 42
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

from medlm import paths
from medlm.io_utils import load_split, read_jsonl, set_seed
from medlm.prompts import build_prompt
from medlm.retrieval import Retriever
from medlm.tokenizer_util import load_tokenizer


def build_training_prompts(dataset, n_train, n_val, seed, k, pool_size, device,
                           exemplar_field, use_question_type):
    """Training rows drawn from the pool, each with its own retrieved exemplars."""
    pool, _ = load_split(dataset, pool_size, 10, seed)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pool), min(n_train, len(pool)), replace=False))

    val_path = os.path.join(paths.DATASETS[dataset]["dir"], "validation.jsonl")
    val_rows = read_jsonl(val_path)[:n_val]

    retriever = Retriever(pool, device, use_question_type)

    def assemble(rows, is_train):
        questions = [r["question"] for r in rows]
        emb = retriever.encode(questions)
        out = []
        for i, r in enumerate(rows):
            ids = retriever.search(emb[i], r.get("qtype", "general"), k + 1)
            # never retrieve the row itself as its own exemplar
            ids = [j for j in ids if not (is_train and pool[j]["idx"] == r.get("idx", -1))][:k]
            exemplars = [pool[j] for j in ids]
            out.append(dict(prompt=build_prompt(r["question"], exemplars, exemplar_field),
                            target=r.get("summary") or r["answer"]))
        return out

    train = assemble([pool[int(i)] for i in idx], True)
    val = assemble(val_rows, False)
    del retriever
    torch.cuda.empty_cache()
    return train, val, [int(pool[int(i)]["idx"]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--base", required=True, help="hub id or local path of the backbone")
    ap.add_argument("--label", required=True, help="short name used in output paths")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--exemplar_field", default="summary", choices=["summary", "answer"])
    ap.add_argument("--use_question_type", action="store_true")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--truncation_side", default="left", choices=["left", "right"])
    ap.add_argument("--max_memory_map", default=None,
                    help="comma-separated per-visible-device cap in GiB, e.g. 44,24,24. On a shared "
                         "node a uniform cap is wrong: cards differ in free memory, and a cap above "
                         "a card's real headroom only relocates the failure.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = args.out or paths.output("phase_b", f"{args.dataset}_{args.label}_seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    train_rows, val_rows, train_ids = build_training_prompts(
        args.dataset, args.n_train, args.n_val, args.seed, args.k, args.pool, device,
        args.exemplar_field, args.use_question_type)
    print(f"[prompts] {len(train_rows)} training, {len(val_rows)} validation", flush=True)

    tokenizer = load_tokenizer(args.base)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    def encode(example):
        prompt = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        target = tokenizer(" " + example["target"] + tokenizer.eos_token,
                           add_special_tokens=False)["input_ids"][:200]
        keep = slice(-512, None) if args.truncation_side == "left" else slice(0, 512)
        prompt = prompt[keep]
        ids = (prompt + target)[:args.max_len]
        labels = ([-100] * len(prompt) + target)[:args.max_len]
        return dict(input_ids=ids, attention_mask=[1] * len(ids), labels=labels)

    dtrain = Dataset.from_list(train_rows).map(encode, remove_columns=["prompt", "target"])
    dval = Dataset.from_list(val_rows).map(encode, remove_columns=["prompt", "target"])

    max_memory = None
    if args.max_memory_map:
        caps = [float(x) for x in args.max_memory_map.split(",")]
        max_memory = {i: f"{c}GiB" for i, c in enumerate(caps)}
        max_memory["cpu"] = "200GiB"
        print(f"[placement] per-device caps {caps} GiB", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    targs = TrainingArguments(
        output_dir=outdir, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="linear", warmup_steps=0.03,
        max_grad_norm=1.0, bf16=True, logging_steps=5, eval_strategy="steps",
        eval_steps=10, save_strategy="no", report_to="none", seed=args.seed,
        data_seed=args.seed, remove_unused_columns=False, dataloader_num_workers=1)
    trainer = Trainer(model=model, args=targs, train_dataset=dtrain, eval_dataset=dval,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                                           label_pad_token_id=-100))
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    trainer.train()
    train_seconds = time.time() - t1
    peak = max(torch.cuda.max_memory_allocated(i)
               for i in range(torch.cuda.device_count())) / 2 ** 30

    trainer.save_model(outdir)
    tokenizer.save_pretrained(outdir)
    with open(os.path.join(outdir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            phase="B", dataset=args.dataset, base=args.base, seed=args.seed,
            n_train=args.n_train, n_val=args.n_val, train_row_ids=train_ids,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            grad_accum=args.grad_accum, effective_batch=args.batch_size * args.grad_accum,
            max_len=args.max_len, k=args.k, pool=args.pool,
            exemplar_field=args.exemplar_field,
            truncation_side=args.truncation_side,
            adaptation="full fine-tuning (all parameters)",
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            train_seconds=round(train_seconds, 1),
            total_seconds=round(time.time() - t0, 1),
            peak_gpu_mem_gb=round(peak, 2), n_gpus=torch.cuda.device_count(),
            torch=torch.__version__,
            transformers=__import__("transformers").__version__), f, indent=2)
    print(f"[done] phase B -> {outdir}, train {train_seconds/60:.1f} min, peak {peak:.1f} GB",
          flush=True)


if __name__ == "__main__":
    main()
