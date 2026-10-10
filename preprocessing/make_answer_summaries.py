# -*- coding: utf-8 -*-
"""Generate the answer summaries that serve as retrieval exemplars.

Every training answer is summarised once, offline, and the summaries are what the
retrieved exemplars carry in the prompt. They are never used as an evaluation
reference: all reported metrics score against the original dataset answer.

Model and decoding, as executed: DeepSeek-R1-Distill-Llama-8B, greedy, 200 new tokens,
fp16, with the prompt of Figure C.3:

    Summarize the following medical answer factually and concisely (limit ~100 tokens):

    <answer>

    Summary:

The ~100-token budget was not met. On MedQuAD the executed summaries have a median
length ratio of 1.81 to the answers they replaced, so a "summary" is typically longer
than its source; the paper reports this rather than assuming the instruction held.

Post-processing strips a leading "Summary:"-style echo and collapses any token repeated
more than twice in a row, which is the degeneration the greedy decode produces on long
answers. `strip_echo` in medlm.io_utils removes a further echo of the instruction that
survives in some rows, at prompt-assembly time.

Output is one JSON list per split, aligned by index to the split's rows, written to
data/derived/train_summaries_<corpus>.json.

Usage:
  python make_answer_summaries.py --dataset medquad --split train
  python make_answer_summaries.py --dataset icliniq --split train --model <hub-id>
"""
import argparse
import json
import os
import re
import sys

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))

import torch
from transformers import AutoModelForCausalLM

from medlm import paths
from medlm.io_utils import read_jsonl
from medlm.tokenizer_util import load_tokenizer

PROMPT = ("Summarize the following medical answer factually and concisely "
          "(limit ~100 tokens):\n\n{answer}\n\nSummary:")
LEADERS = ("summary:", "summary -", "summary :", "summarize the following",
           "summarize:", "summarize :", "answer:", "summary\n:")


def clean_summary(s):
    """Strip a leading instruction echo, then collapse runs of a repeated token."""
    if not s:
        return ""
    s = s.strip()
    low = s.lower()
    for p in LEADERS:
        if low.startswith(p):
            s = s[len(p):].strip()
            low = s.lower()
    out, prev, run = [], None, 0
    for t in s.split():
        run = run + 1 if t == prev else 0
        if run < 3:
            out.append(t)
        prev = t
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--max_input_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = read_jsonl(os.path.join(paths.DATASETS[a.dataset]["dir"], f"{a.split}.jsonl"))
    out_path = a.out or paths.data("derived", f"{a.split}_summaries_{a.dataset}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[data] {a.dataset}/{a.split}: {len(rows)} answers to summarise", flush=True)

    tok = load_tokenizer(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None).eval()

    summaries = []
    with torch.no_grad():
        for s in range(0, len(rows), a.batch_size):
            chunk = [PROMPT.format(answer=r.get("answer", "")) for r in rows[s:s + a.batch_size]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_input_tokens).to(model.device)
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            summaries += [clean_summary(t) for t in text]
            if s % (a.batch_size * 50) == 0:
                print(f"  {len(summaries)}/{len(rows)}", flush=True)

    assert len(summaries) == len(rows), "a summary must exist for every row, aligned by index"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    ratio = sorted(len(s) / max(len(r.get("answer", "")), 1) for s, r in zip(summaries, rows))
    print(f"[done] {len(summaries)} summaries -> {out_path}")
    print(f"       median length ratio to the source answer: {ratio[len(ratio)//2]:.2f}")


if __name__ == "__main__":
    main()
