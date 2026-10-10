# -*- coding: utf-8 -*-
"""ClinicalBERT verifier score, Eq. (15).

v(q, a) = softmax(f(q, a))_1 is an uncalibrated question-answer relatedness
score. It is not interpreted as a probability of factual correctness.
"""
import torch


@torch.no_grad()
def verifier_scores(model, tokenizer, question, candidates, batch_size=16):
    texts = [f"Q: {question} [SEP] A: {c}" for c in candidates]
    probs = []
    for s in range(0, len(texts), batch_size):
        enc = tokenizer(texts[s:s + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(model.device)
        p = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
        probs.extend(p.float().cpu().tolist())
    return probs
