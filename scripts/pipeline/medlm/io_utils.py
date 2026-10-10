# -*- coding: utf-8 -*-
"""Split loading, seeding and run identity.

All metrics in the paper are computed against the ORIGINAL dataset answer. The
machine summaries are used only as retrieval exemplars, never as the reference.
"""
import hashlib
import json
import os
import random
import re

import numpy as np

from . import paths
from .qtype import assign_qtype

# The summariser was instructed with a token limit; a few summaries echo that
# instruction back before the summary itself. The echo is stripped here so it
# never enters a prompt.
ECHO_RX = re.compile(r"^.*?limit\s*~?100\s*tokens\)\s*:\s*", re.I | re.M)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def strip_echo(s):
    return ECHO_RX.sub("", s or "").strip()


def run_id(cfg):
    """Stable identifier for a configuration, so no reported number exists
    without the configuration that produced it."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def load_split(name, n_train, n_test, seed, first_test_rows=False):
    """Retrieval pool and evaluation set for one dataset.

    first_test_rows=True reproduces the sampling rule of the submitted
    experiments for the test side (the first n_test rows in file order); the pool
    stays a seeded random sample in both cases.
    """
    spec = paths.DATASETS[name]
    train_rows = read_jsonl(os.path.join(spec["dir"], "train.jsonl"))
    test_rows = read_jsonl(os.path.join(spec["dir"], "test.jsonl"))

    rng = np.random.default_rng(seed)
    if n_train <= 0 or n_train >= len(train_rows):
        tr_idx = np.arange(len(train_rows))
    else:
        tr_idx = np.sort(rng.choice(len(train_rows), n_train, replace=False))
    if first_test_rows:
        te_idx = np.arange(min(n_test, len(test_rows)))
    elif n_test <= 0 or n_test >= len(test_rows):
        te_idx = np.arange(len(test_rows))
    else:
        te_idx = np.sort(rng.choice(len(test_rows), n_test, replace=False))

    summaries = None
    if os.path.exists(spec["summaries"]):
        with open(spec["summaries"], encoding="utf-8") as f:
            summaries = json.load(f)

    train = []
    for i in tr_idx:
        i = int(i)
        row = train_rows[i]
        summary = strip_echo(summaries[i]) if summaries and i < len(summaries) else row["answer"]
        train.append(dict(idx=i, question=row["question"], answer=row["answer"],
                          summary=summary, qtype=assign_qtype(row["question"])))
    test = [dict(idx=int(i), question=test_rows[int(i)]["question"],
                 answer=test_rows[int(i)]["answer"],
                 qtype=assign_qtype(test_rows[int(i)]["question"]))
            for i in te_idx]
    return train, test
