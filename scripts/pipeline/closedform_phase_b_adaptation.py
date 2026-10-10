"""Fine-tune a backbone for closed-form option selection with the recipe of the submitted
study.

Target and prompt reproduce the released closed-form code: prompt = template C.1 (no
exemplars), target = "<letter>. <option text>". Recipe from Table 3 of the manuscript
(closed-form generator): 3 epochs, lr 3e-5, per-device batch 4 x grad-accum 4, linear
schedule with 3% warm-up, bf16, prompt tokens masked to -100, seed 42.
"""

import argparse, json, os, time
from medlm.legacy_paths import ws, src
import numpy as np, torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)
from medlm.tokenizer_util import load_tokenizer
import closedform_phase_c_inference_and_selection as CF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CF.LOADERS))
    ap.add_argument("--base", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=500); ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=640)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = ws("experiments", f"ckptcf_{a.dataset}_{a.label}_seed{a.seed}"); os.makedirs(out, exist_ok=True)
    t0 = time.time()

    train, _test, _n_ch = CF.LOADERS[a.dataset]()   # training pool (test split unused here)
    rng = np.random.default_rng(a.seed)
    take = min(a.n_train + a.n_val, len(train))
    idx = np.sort(rng.choice(len(train), take, replace=False))
    rows = [train[int(i)] for i in idx]
    # small pools (MMLU-Med has only ~150 dev+validation items) keep a proportional validation share
    n_val = a.n_val if take >= a.n_train + a.n_val else max(1, take // 5)
    n_tr = take - n_val
    tr_rows, va_rows = rows[:n_tr], rows[n_tr:take]
    assert tr_rows and va_rows, (len(train), take, n_tr, n_val)

    def build(rs):
        out_rows = []
        for r in rs:
            n_ch = len(r["options"])
            assert 0 <= int(r["gold"]) < n_ch
            p = CF.build_prompt(r["question"], r["options"], r.get("context", ""), [], n_ch)
            g = int(r["gold"])
            out_rows.append(dict(prompt=p, target=f"{CF.LETTERS[g]}. {r['options'][g]}"))
        return out_rows
    tr, va = build(tr_rows), build(va_rows)

    tok = load_tokenizer(a.base); tok.pad_token = tok.pad_token or tok.eos_token
    def enc(ex):
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(" " + ex["target"] + tok.eos_token, add_special_tokens=False)["input_ids"][:64]
        p = p[-(a.max_len - len(t)):]                       # keep the right side: options must survive
        ids = (p + t)[:a.max_len]; lab = ([-100] * len(p) + t)[:a.max_len]
        return dict(input_ids=ids, attention_mask=[1] * len(ids), labels=lab)
    dtr = Dataset.from_list(tr).map(enc, remove_columns=["prompt", "target"])
    dva = Dataset.from_list(va).map(enc, remove_columns=["prompt", "target"])

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False; model.gradient_checkpointing_enable()
    args = TrainingArguments(output_dir=out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
                             gradient_accumulation_steps=a.ga, num_train_epochs=a.epochs, learning_rate=a.lr,
                             lr_scheduler_type="linear", warmup_steps=0.03, max_grad_norm=1.0, bf16=True,
                             logging_steps=5, eval_strategy="steps", eval_steps=10, save_strategy="no",
                             report_to="none", seed=a.seed, data_seed=a.seed, remove_unused_columns=False,
                             dataloader_num_workers=1)
    trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    torch.cuda.reset_peak_memory_stats(); t1 = time.time(); trainer.train(); tsec = time.time() - t1
    peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2 ** 30
    trainer.save_model(out); tok.save_pretrained(out)
    json.dump(trainer.state.log_history, open(f"{out}/log_history.json", "w"), indent=1)
    json.dump(dict(dataset=a.dataset, base=a.base, label=a.label, seed=a.seed, n_train=len(tr), n_val=len(va),
                   train_row_ids=[int(i) for i in idx[:len(tr_rows)]], pool_size=len(train), epochs=a.epochs, lr=a.lr,
                   effective_batch=a.bs * a.ga, max_len=a.max_len, train_seconds=round(tsec, 1),
                   peak_gpu_mem_gb=round(peak, 2), n_gpus=torch.cuda.device_count(),
                   trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   final_eval_loss=next((h["eval_loss"] for h in reversed(trainer.state.log_history) if "eval_loss" in h), None),
                   sampling="random n_train+n_val from the training split, seed 42"),
              open(f"{out}/manifest.json", "w"), indent=1)
    print(f"[done] {out} train {tsec/60:.1f} min peak {peak:.1f} GB total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
