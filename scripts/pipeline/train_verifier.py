# -*- coding: utf-8 -*-
"""Per-benchmark verifier training.

The submitted closed-form path scores options with a verifier trained on MedQA question/answer
pairs and applied unchanged to the other three benchmarks. This trains one verifier per benchmark
on that benchmark's own training split, using the same architecture, input format and proxy-label
construction as the released checkpoint, so the transfer can be tested rather than only diagnosed.

Positive = (question, correct option text). Negatives = (question, each incorrect option), which is
the closed-form analogue of the released checkpoint's "answer to a different question" negatives.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlm.legacy_paths import src, ws
import closedform_phase_c_inference_and_selection as CF
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding)
from datasets import Dataset
LETTERS = "ABCDEFGH"
BASE = "medicalai/ClinicalBERT"

def pairs_from(rows, max_neg=3, rng=None):
    ex = []
    for r in rows:
        opts, g = r["options"], int(r["gold"])
        if not (0 <= g < len(opts)): continue
        ex.append((f"Q: {r['question']} [SEP] A: {LETTERS[g]}. {opts[g]}", 1))
        negs = [i for i in range(len(opts)) if i != g]
        if rng is not None and len(negs) > max_neg:
            negs = list(rng.choice(negs, max_neg, replace=False))
        for i in negs:
            ex.append((f"Q: {r['question']} [SEP] A: {LETTERS[i]}. {opts[i]}", 0))
    return ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CF.LOADERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr_rows, _te, _nch = CF.LOADERS[a.dataset]()
    rng.shuffle(tr_rows)
    take = min(a.n_train + a.n_val, len(tr_rows))
    n_val = a.n_val if take >= a.n_train + a.n_val else max(1, take // 5)
    tr, va = tr_rows[:take - n_val], tr_rows[take - n_val:take]
    ptr, pva = pairs_from(tr, rng=rng), pairs_from(va, rng=rng)
    print(f"[{a.dataset}] train rows {len(tr)} -> {len(ptr)} pairs ({sum(y for _, y in ptr)} positive); "
          f"val rows {len(va)} -> {len(pva)} pairs", flush=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=2)
    def mk(p):
        d = Dataset.from_dict(dict(text=[t for t, _ in p], labels=[y for _, y in p]))
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len), batched=True,
                     remove_columns=["text"])
    dtr, dva = mk(ptr), mk(pva)
    args = TrainingArguments(output_dir=a.out + "_tmp", per_device_train_batch_size=a.bs,
                             per_device_eval_batch_size=a.bs * 2, num_train_epochs=a.epochs,
                             learning_rate=a.lr, eval_strategy="epoch", save_strategy="no",
                             logging_steps=50, report_to=[], seed=a.seed, fp16=torch.cuda.is_available())
    def metrics(p):
        pred = np.argmax(p.predictions, -1)
        return dict(accuracy=float((pred == p.label_ids).mean()))
    t = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    t.train()
    ev = t.evaluate()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump(dict(dataset=a.dataset, base=a.base, n_train_rows=len(tr), n_train_pairs=len(ptr),
                   n_val_pairs=len(pva), epochs=a.epochs, lr=a.lr, seed=a.seed,
                   eval_accuracy=ev.get("eval_accuracy"), eval_loss=ev.get("eval_loss")),
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(f"[done] {a.out} val_acc={ev.get('eval_accuracy')}", flush=True)
if __name__ == "__main__":
    main()
