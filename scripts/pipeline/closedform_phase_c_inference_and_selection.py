"""Closed-form inference and selection (MedQA / MedMCQA / PubMedQA / MMLU-Med).

Three properties the reported numbers depend on:

* Answer keys are validated before scoring. `assert_labels_usable` aborts the run
  if the gold key is degenerate, so a split whose label column is absent cannot
  silently collapse every gold answer onto one option.
* Retrieval is a real switch. When `use_retrieval` is set, exemplars enter the
  multiple-choice prompt; when it is not, none are added.
* Per-item predictions are written, so paired McNemar tests between systems can
  be computed on identical items, and macro-F1 and class prevalence are reported
  alongside accuracy.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import time

from medlm.legacy_paths import src, ws

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)
from medlm.tokenizer_util import load_tokenizer

SEED = 42
EMBEDDER = src("all-MiniLM-L6-v2")
Z = 1.959963984540054


# ------------------------------------------------------------------ dataset specs
def load_medqa():
    from datasets import load_dataset
    d = load_dataset("GBaker/MedQA-USMLE-4-options")
    def conv(r):
        opts = [r["options"][k] for k in ["A", "B", "C", "D"]]
        return dict(question=r["question"], options=opts,
                    gold=ord(r["answer_idx"].strip().upper()) - 65, context="")
    return [conv(r) for r in d["train"]], [conv(r) for r in d["test"]], 4


def load_medmcqa():
    # the TEST split carries no usable `cop`; the VALIDATION split does.
    from datasets import load_dataset
    d = load_dataset("openlifescienceai/medmcqa")
    def conv(r):
        return dict(question=r["question"],
                    options=[r["opa"], r["opb"], r["opc"], r["opd"]],
                    gold=int(r["cop"]), context="")
    return [conv(r) for r in d["train"]], [conv(r) for r in d["validation"]], 4


def load_pubmedqa():
    from datasets import load_dataset
    d = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    ymap = {"yes": 0, "no": 1, "maybe": 2}
    rows = []
    for r in d:
        ctx = r.get("context", {})
        txt = " ".join(ctx.get("contexts", [])) if isinstance(ctx, dict) else str(ctx)
        rows.append(dict(question=r["question"], options=["yes", "no", "maybe"],
                         gold=ymap[r["final_decision"].strip().lower()], context=txt))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(rows))
    cut = int(0.8 * len(rows))
    return [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]], 3


MMLU_SUBJECTS = ["clinical_knowledge", "medical_genetics", "anatomy",
                 "professional_medicine", "college_biology", "college_medicine"]


def load_mmlu_med():
    from datasets import load_dataset
    tr, te = [], []
    for s in MMLU_SUBJECTS:
        d = load_dataset("cais/mmlu", s)
        for split, sink in (("dev", tr), ("validation", tr), ("test", te)):
            if split in d:
                for r in d[split]:
                    sink.append(dict(question=r["question"], options=list(r["choices"]),
                                     gold=int(r["answer"]), context="", subject=s))
    return tr, te, 4


LOADERS = {"medqa": load_medqa, "medmcqa": load_medmcqa,
           "pubmedqa": load_pubmedqa, "mmlu_med": load_mmlu_med}


def assert_labels_usable(rows, name):
    c = collections.Counter(r["gold"] for r in rows)
    n = sum(c.values())
    maj = max(c.values()) / n
    if len(c) < 2:
        raise SystemExit(f"ABORT {name}: gold labels collapsed to one class {dict(c)}")
    if maj > 0.95:
        raise SystemExit(f"ABORT {name}: majority class {maj:.3f} exceeds 0.95 {dict(c)}")
    print(f"[labels] {name}: n={n} classes={len(c)} "
          f"prevalence={ {k: round(100*v/n,2) for k,v in sorted(c.items())} } "
          f"majority={100*maj:.2f}%", flush=True)
    return {str(k): round(100 * v / n, 4) for k, v in sorted(c.items())}


# ----------------------------------------------------------------------- scoring
def wilson(k, n, z=Z):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def macro_f1(gold, pred, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)), f1s


LETTERS = "ABCDEFGH"


def build_prompt(q, options, context, exemplars, n_choices, template="c2"):
    """template 'c2' = matched-control head; 'c1' = the head of the submitted closed-form runs,
    which adds the exact-output and no-explanation instructions (Appendix C)."""
    opt = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    valid = ", ".join(LETTERS[:n_choices - 1]) + f", or {LETTERS[n_choices-1]}"
    head = ("You are a medical expert taking a multiple-choice medical exam.\n"
            "Choose the single best answer from the options.\n"
            f"Valid choices are {valid}.\n"
            + ("Return ONLY the option in this exact format: '<letter>. <answer>'.\n"
               "Do not add extra explanation.\n\n" if template == "c1" else "\n"))
    ex = ""
    for e in exemplars:
        eo = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(e["options"]))
        ex += (f"Question:\n{e['question']}\n\nOptions:\n{eo}\n\n"
               f"Answer: {LETTERS[e['gold']]}. {e['options'][e['gold']]}\n---\n\n")
    body = (f"Context:\n{context}\n\nQuestion:\n{q}" if context else f"Question:\n{q}")
    return f"{head}{ex}{body}\n\nOptions:\n{opt}\n\nAnswer:"


@torch.no_grad()
def option_logprobs(model, tok, prompt, options, max_input):
    """Mean per-token continuation log-probability of each option under the model."""
    scores = []
    p_ids = tok(prompt, return_tensors="pt", truncation=True,
                max_length=max_input).input_ids[0]
    for i, o in enumerate(options):
        cont = f" {LETTERS[i]}. {o}"
        c_ids = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids[0]
        ids = torch.cat([p_ids, c_ids])[None, :].to(model.device)
        logits = model(ids).logits[0, :-1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tgt = ids[0, 1:]
        sel = lp[range(len(tgt)), tgt][-len(c_ids):]
        scores.append(float(sel.mean()))
    return scores


def minmax(v):
    v = np.asarray(v, float)
    if v.max() - v.min() < 1e-9:
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min())


@torch.no_grad()
def verifier_option_scores(vm, vt, q, options):
    texts = [f"Q: {q} [SEP] A: {LETTERS[i]}. {o}" for i, o in enumerate(options)]
    enc = vt(texts, return_tensors="pt", padding=True, truncation=True,
             max_length=256).to(vm.device)
    return torch.softmax(vm(**enc).logits, dim=-1)[:, 1].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--generator", required=True)
    ap.add_argument("--generator_label", required=True)
    ap.add_argument("--verifier", default=src("outputs_llm", "verifier"))
    ap.add_argument("--conditions", default="vanilla,dense_rag,full")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--max_input", type=int, default=1024)
    ap.add_argument("--verifier_weight", type=float, default=0.35)
    ap.add_argument("--template", default="c2", choices=["c1", "c2"],
                    help="prompt head: c2 (matched controls) or c1 (head used by the submitted closed-form runs)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    outdir = ws("experiments", f"closed_{args.dataset}_{args.generator_label}")
    os.makedirs(outdir, exist_ok=True)

    train, test, n_ch = LOADERS[args.dataset]()
    rng = np.random.default_rng(args.seed)
    if args.n_train > 0 and args.n_train < len(train):
        train = [train[i] for i in np.sort(rng.choice(len(train), args.n_train, False))]
    if args.n_test > 0 and args.n_test < len(test):
        test = [test[i] for i in np.sort(rng.choice(len(test), args.n_test, False))]
    prev = assert_labels_usable(test, f"{args.dataset}/test")
    assert_labels_usable(train, f"{args.dataset}/train")

    tok = load_tokenizer(args.generator)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.generator, dtype=torch.bfloat16, device_map="auto").eval()
    vt = load_tokenizer(args.verifier)
    vm = AutoModelForSequenceClassification.from_pretrained(args.verifier).to(dev).eval()
    print(f"[model] loaded in {time.time()-t0:.0f}s", flush=True)

    enc = SentenceTransformer(EMBEDDER, device=dev)
    temb = enc.encode([t["question"] for t in train], convert_to_numpy=True,
                      batch_size=256, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(temb)
    index = faiss.IndexFlatIP(temb.shape[1])
    index.add(temb)
    qemb = enc.encode([t["question"] for t in test], convert_to_numpy=True,
                      batch_size=256, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(qemb)

    gold = [t["gold"] for t in test]
    results = json.load(open(f"{outdir}/metrics.json")) if os.path.exists(f"{outdir}/metrics.json") else {}
    frng = np.random.default_rng(args.seed + 1)   # few-shot control: k RANDOM training exemplars (not retrieved)
    for cond in args.conditions.split(","):
        use_rag = cond in ("dense_rag", "full", "adapted_rag", "adapted_rag_verifier")
        use_few = cond == "fewshot"
        use_ver = cond in ("full", "adapted_verifier", "adapted_rag_verifier")
        tg = time.time(); torch.cuda.reset_peak_memory_stats()
        preds, rows = [], []
        for i, t in enumerate(test):
            exs = []
            if use_rag:
                _, I = index.search(qemb[i][None, :], args.k)
                exs = [train[int(j)] for j in I[0] if j >= 0]
            elif use_few:
                exs = [train[int(j)] for j in frng.choice(len(train), args.k, replace=False)]
            pr = build_prompt(t["question"], t["options"], t.get("context", ""),
                              exs, n_ch, args.template)
            lp = option_logprobs(model, tok, pr, t["options"], args.max_input)
            comb = minmax(lp)
            vsc = None
            if use_ver:
                vsc = verifier_option_scores(vm, vt, t["question"], t["options"])
                comb = (1 - args.verifier_weight) * minmax(lp) + \
                    args.verifier_weight * minmax(vsc)
            p = int(np.argmax(comb))
            preds.append(p)
            rows.append(dict(test_index=i, question=t["question"][:400],
                             gold=t["gold"], pred=p, correct=int(p == t["gold"]),
                             logprobs=[round(x, 5) for x in lp],
                             verifier=(None if vsc is None else
                                       [round(float(x), 5) for x in vsc])))
            if (i + 1) % 100 == 0:
                print(f"  [{cond}] {i+1}/{len(test)} "
                      f"acc so far {np.mean([r['correct'] for r in rows]):.3f}", flush=True)

        k = int(sum(1 for g, p in zip(gold, preds) if g == p))
        acc, lo, hi = wilson(k, len(gold))
        mf1, perc = macro_f1(gold, preds, n_ch)
        cm = [[int(sum(1 for g, p in zip(gold, preds) if g == a and p == b))
               for b in range(n_ch)] for a in range(n_ch)]
        cfg = dict(dataset=args.dataset, generator=args.generator_label,
                   condition=cond, k=args.k if (use_rag or use_few) else 0,
                   use_retrieval=use_rag, few_shot_random=use_few, use_verifier=use_ver,
                   task_adapted=cond.startswith("adapted"), generator_path=args.generator, template=args.template,
                   verifier_weight=args.verifier_weight if use_ver else None,
                   n_test=len(test), seed=args.seed)
        rid = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        results[cond] = dict(
            run_id=rid, config=cfg, n=len(gold), correct=k,
            accuracy_pct=round(100 * acc, 2),
            accuracy_ci95=[round(100 * lo, 2), round(100 * hi, 2)],
            macro_f1_pct=round(100 * mf1, 2),
            per_class_f1_pct=[round(100 * x, 2) for x in perc],
            majority_class_baseline_pct=round(100 * max(
                collections.Counter(gold).values()) / len(gold), 2),
            gold_prevalence_pct=prev, confusion_matrix=cm,
            seconds=round(time.time() - tg, 1), seconds_per_item=round((time.time() - tg) / len(test), 3),
            peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        with open(f"{outdir}/preds_{cond}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                r["run_id"] = rid
                f.write(json.dumps(r) + "\n")
        print(f"[res] {args.dataset}/{cond:10s} run={rid} acc={100*acc:.2f} "
              f"[{100*lo:.2f},{100*hi:.2f}] macroF1={100*mf1:.2f} "
              f"majority={results[cond]['majority_class_baseline_pct']:.2f}", flush=True)
        json.dump(results, open(f"{outdir}/metrics.json", "w"), indent=2)

    json.dump(results, open(f"{outdir}/metrics.json", "w"), indent=2)
    print(f"[done] {outdir} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
