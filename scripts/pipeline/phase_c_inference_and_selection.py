#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase C - Online Inference and Selection (Section 3.4 of the paper).

For each evaluation question the pipeline retrieves exemplars, assembles the
prompt, generates N=5 candidates, scores them and selects one:

  A(q) = {a^(1), ..., a^(N)}                       Eq. (13)
  entity-aware decoding bonus, strength gamma      Eq. (14)
  v(q, a) = softmax(f(q, a))_1                     Eq. (15)
  Ret(a) = |E(q) n E(a)| / |E(q)|                  Eq. (17)
  score = alpha * v(q, a) + (1 - alpha) * Ret(a)   Eq. (18)

Two properties make the ablations in the paper exactly matched:

  * Generation happens ONCE per distinct prompt set. The four selection rules
    (first beam / verifier only / entity only / combined) are then applied to the
    same candidate pool, so they differ only in the scoring function and never in
    the generated text.
  * The candidate budget is a switch, so N=1 and N=5 conditions are directly
    comparable.

Every run writes its configuration, seed, split ids, per-example predictions,
per-example metric values and aggregate metrics under a run id, so no reported
number exists without its provenance.

Usage:
  python phase_c_inference_and_selection.py --dataset medquad \
      --generator <path-or-hub-id> --generator_label llama31 \
      --verifier <path-or-hub-id> --prompt_sets norag,rag,rag_qtype,rag_fullans
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from medlm import paths
from medlm.decoding import PROMPT_SETS, SELECTORS, generate
from medlm.entities import extract_entities, load_vocab, retention
from medlm.io_utils import load_split, run_id, set_seed
from medlm.metrics import bertscore, boot_ci, score_pairs
from medlm.prompts import build_prompt
from medlm.retrieval import Retriever
from medlm.tokenizer_util import load_tokenizer


def banned_pool_items(train, test, mode, meta_path):
    """Source-disjoint retrieval: bar exemplars sharing the query's MedQuAD source
    document and/or focus entity. Returns one banned-index set per test question."""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    train_meta = (meta.get("train") or {}).get("records") or []
    test_meta = (meta.get("test") or {}).get("records") or []

    def key(rec):
        rec = rec or {}
        return ((rec.get("doc") or "").strip() or None,
                (rec.get("focus") or "").strip().lower() or None)

    pool = [key(train_meta[t["idx"]]) if 0 <= t["idx"] < len(train_meta) else (None, None)
            for t in train]
    out = []
    for t in test:
        qd, qf = key(test_meta[t["idx"]]) if 0 <= t["idx"] < len(test_meta) else (None, None)
        ban = set()
        for j, (pd, pf) in enumerate(pool):
            if mode in ("doc", "both") and qd and pd == qd:
                ban.add(j)
            if mode in ("focus", "both") and qf and pf == qf:
                ban.add(j)
        out.append(ban)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--generator", required=True, help="path or hub id of the generator")
    ap.add_argument("--generator_label", required=True)
    ap.add_argument("--verifier", required=True, help="path or hub id of the ClinicalBERT verifier")
    ap.add_argument("--prompt_sets", default="norag,rag,rag_qtype,rag_fullans")
    ap.add_argument("--n_cand", type=int, default=5)
    ap.add_argument("--also_n1", action="store_true",
                    help="additionally generate a matched N=1 pass for each prompt set")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n_train", type=int, default=2000, help="retrieval pool size")
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--max_input", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--truncation_side", default="left", choices=["left", "right"],
                    help="'left' keeps the end of the prompt so the target question survives the "
                         "input cap; 'right' reproduces the submitted runs, which dropped it")
    ap.add_argument("--gamma_override", type=float, default=None,
                    help="override the entity-decoding bonus strength of the edec prompt sets; "
                         "the executed value is 1.0")
    ap.add_argument("--exclude_same", default="off", choices=["off", "doc", "focus", "both"],
                    help="source-disjoint retrieval; needs --meta")
    ap.add_argument("--meta", default=None,
                    help="MedQuAD source/focus metadata for --exclude_same")
    ap.add_argument("--first_test_rows", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    outdir = args.out or paths.output(
        "phase_c", f"{args.dataset}_{args.generator_label}{args.tag}")
    os.makedirs(outdir, exist_ok=True)

    train, test = load_split(args.dataset, args.n_train, args.n_test, args.seed,
                             args.first_test_rows)
    print(f"[data] {args.dataset}: pool {len(train)}, evaluation {len(test)}", flush=True)

    vocab = load_vocab(paths.ENTITY_VOCAB)
    tokenizer = load_tokenizer(args.generator)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = args.truncation_side
    model = AutoModelForCausalLM.from_pretrained(
        args.generator, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = True

    vtokenizer = load_tokenizer(args.verifier)
    vmodel = AutoModelForSequenceClassification.from_pretrained(args.verifier).to(device).eval()
    from medlm.verifier import verifier_scores

    question_entities = [extract_entities(t["question"], vocab) for t in test]
    gold = [t["answer"] for t in test]

    banned = None
    if args.exclude_same != "off":
        if not args.meta:
            raise SystemExit("--exclude_same needs --meta (MedQuAD source/focus metadata)")
        banned = banned_pool_items(train, test, args.exclude_same, args.meta)
        mean_banned = sum(len(b) for b in banned) / max(len(banned), 1)
        print(f"[source-disjoint] mode={args.exclude_same}, mean banned pool items per "
              f"question {mean_banned:.1f} of {len(train)}", flush=True)

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            phase="C", dataset=args.dataset, generator=args.generator,
            generator_label=args.generator_label, verifier=args.verifier,
            truncation_side=args.truncation_side, exclude_same=args.exclude_same,
            test_sampling="first_rows" if args.first_test_rows else "random_seed",
            seed=args.seed, n_train_pool=len(train), n_test=len(test), k=args.k,
            n_cand=args.n_cand, max_new=args.max_new, max_input=args.max_input,
            train_ids=[t["idx"] for t in train], test_ids=[t["idx"] for t in test],
            reference="original dataset answer", torch=torch.__version__,
            transformers=__import__("transformers").__version__), f, indent=2)

    results = {}
    for ps_name in args.prompt_sets.split(","):
        ps = dict(PROMPT_SETS[ps_name])
        if args.gamma_override is not None and ps.get("gamma"):
            ps["gamma"] = args.gamma_override

        retriever = None
        exemplar_ids = [[] for _ in test]
        retrieval_seconds = 0.0
        if ps["use_retrieval"]:
            retriever = Retriever(train, device, ps["use_question_type"])
            tr0 = time.time()
            emb = retriever.encode([t["question"] for t in test])
            if banned is not None:
                exemplar_ids = [retriever.search_excluding(emb[i], test[i]["qtype"], args.k,
                                                           banned[i]) for i in range(len(test))]
            else:
                exemplar_ids = [retriever.search(emb[i], test[i]["qtype"], args.k)
                                for i in range(len(test))]
            retrieval_seconds = time.time() - tr0

        prompts, prompt_entities = [], []
        for i, t in enumerate(test):
            exemplars = [train[j] for j in exemplar_ids[i]] if ps["use_retrieval"] else []
            prompt = build_prompt(t["question"], exemplars,
                                  ps["exemplar_field"] if ps["use_retrieval"] else None)
            prompts.append(prompt)
            prompt_entities.append(extract_entities(prompt, vocab))

        gamma = ps.get("gamma", 0.0)
        entity_seqs = None
        if gamma > 0:
            # target entities for decoding control are the entities of the QUESTION
            entity_seqs = [[tokenizer(" " + e, add_special_tokens=False)["input_ids"]
                            for e in sorted(question_entities[i])][:12] for i in range(len(test))]

        for n_cand in [args.n_cand] + ([1] if args.also_n1 else []):
            tag = f"{ps_name}_n{n_cand}"
            tg = time.time()
            torch.cuda.reset_peak_memory_stats()
            print(f"[gen] {tag}: {len(prompts)} prompts, N={n_cand}, gamma={gamma}", flush=True)
            candidates = generate(model, tokenizer, prompts, n_cand, args.max_new,
                                  args.max_input, args.batch_size, entity_seqs, gamma)
            gen_seconds = time.time() - tg
            gen_peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30

            tv = time.time()
            per_item = []
            for i, cl in enumerate(candidates):
                cl = [c for c in cl if c.strip()] or [""]
                scores = verifier_scores(vmodel, vtokenizer, test[i]["question"], cl)
                cov_q, cov_prompt = [], []
                for c in cl:
                    ce = extract_entities(c, vocab)
                    cov_q.append(retention(ce, question_entities[i]))
                    cov_prompt.append(retention(ce, prompt_entities[i]))
                per_item.append(dict(candidates=cl, verifier=scores, cov_q=cov_q,
                                     cov_prompt=cov_prompt, exemplar_ids=exemplar_ids[i]))
            verifier_seconds = time.time() - tv

            with open(os.path.join(outdir, f"candidates_{tag}.jsonl"), "w", encoding="utf-8") as f:
                for i, it in enumerate(per_item):
                    f.write(json.dumps(dict(
                        test_idx=test[i]["idx"], question=test[i]["question"],
                        reference=gold[i], candidates=it["candidates"],
                        verifier=it["verifier"], cov_q=it["cov_q"])) + "\n")

            for selector in (list(SELECTORS) if n_cand > 1 else ["first"]):
                rule = SELECTORS[selector]
                picks = []
                for it in per_item:
                    cl = it["candidates"]
                    if selector == "first" or len(cl) == 1:
                        picks.append(0)
                        continue
                    best, best_i = -1e9, 0
                    for j in range(len(cl)):
                        cq = it["cov_q"][j]
                        cq = 0.0 if cq is None else cq
                        if rule["use_verifier"] and rule["use_entity"]:
                            s = rule["alpha"] * it["verifier"][j] + (1 - rule["alpha"]) * cq
                        elif rule["use_verifier"]:
                            s = it["verifier"][j]
                        else:
                            s = cq
                        if s > best:
                            best, best_i = s, j
                    picks.append(best_i)

                predictions = [per_item[i]["candidates"][picks[i]] for i in range(len(test))]
                condition = f"{tag}_{selector}"
                cfg = dict(dataset=args.dataset, generator=args.generator_label,
                           prompt_set=ps_name, n_cand=n_cand, selector=selector,
                           k=args.k if ps["use_retrieval"] else 0,
                           use_retrieval=ps["use_retrieval"],
                           use_question_type=ps["use_question_type"],
                           exemplar_field=ps["exemplar_field"], gamma=gamma,
                           seed=args.seed, n_test=len(test), generator_path=args.generator)
                rid = run_id(cfg)

                per_example = score_pairs(predictions, gold)
                per_example["bertscore_f1"] = bertscore(predictions, gold, device)
                agg = {}
                for metric, values in per_example.items():
                    mean, lo, hi = boot_ci(values, seed=args.seed)
                    agg[metric] = dict(mean_pct=round(100 * mean, 2),
                                       ci_low_pct=round(100 * lo, 2),
                                       ci_high_pct=round(100 * hi, 2))
                cov_q = [it["cov_q"][p] for it, p in zip(per_item, picks)
                         if it["cov_q"][p] is not None]
                cov_p = [it["cov_prompt"][p] for it, p in zip(per_item, picks)
                         if it["cov_prompt"][p] is not None]
                agg["entity_retention_Eq"] = dict(
                    mean_pct=round(100 * float(np.mean(cov_q)), 2) if cov_q else None,
                    n_scored=len(cov_q), n_excluded_empty_Eq=len(test) - len(cov_q))
                agg["entity_retention_Eprompt"] = dict(
                    mean_pct=round(100 * float(np.mean(cov_p)), 2) if cov_p else None,
                    n_scored=len(cov_p))
                agg["_meta"] = dict(
                    run_id=rid, config=cfg,
                    generation_seconds=round(gen_seconds, 1),
                    generation_seconds_per_item=round(gen_seconds / len(test), 3),
                    verifier_entity_seconds=round(verifier_seconds, 1),
                    retrieval_seconds=round(retrieval_seconds, 2),
                    generation_peak_gpu_mem_gb=round(gen_peak_gb, 2),
                    retrieval_type_fallbacks=(retriever.fallbacks if retriever else 0))
                results[condition] = agg

                with open(os.path.join(outdir, f"predictions_{condition}.jsonl"), "w",
                          encoding="utf-8") as f:
                    for i in range(len(test)):
                        f.write(json.dumps(dict(
                            run_id=rid, test_idx=test[i]["idx"], question=test[i]["question"],
                            qtype=test[i]["qtype"], reference=gold[i], prediction=predictions[i],
                            n_candidates=len(per_item[i]["candidates"]), chosen=picks[i],
                            verifier=per_item[i]["verifier"][picks[i]],
                            cov_q=per_item[i]["cov_q"][picks[i]],
                            cov_prompt=per_item[i]["cov_prompt"][picks[i]],
                            rouge1=per_example["rouge1"][i], rouge2=per_example["rouge2"][i],
                            rougeL=per_example["rougeL"][i], meteor=per_example["meteor"][i],
                            bertscore_f1=per_example["bertscore_f1"][i],
                            exemplar_ids=per_item[i]["exemplar_ids"])) + "\n")

                print(f"[res] {condition:28s} run={rid}  "
                      f"R1={agg['rouge1']['mean_pct']:.2f} "
                      f"MET={agg['meteor']['mean_pct']:.2f} "
                      f"BS={agg['bertscore_f1']['mean_pct']:.2f}", flush=True)
                with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

    print(f"[done] phase C -> {outdir} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
