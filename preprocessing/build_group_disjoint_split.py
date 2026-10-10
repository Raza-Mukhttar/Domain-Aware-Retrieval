# -*- coding: utf-8 -*-
"""Build the source/topic-disjoint repartition of MedQuAD.

Design decision worth stating: the split is built from the ORIGINAL TRAINING PARTITION only, not
from the pooled corpus. Exemplar summaries exist only for those rows (they are a list aligned to
train.jsonl), so pooling all three partitions would mean part of the new retrieval pool carried
summarised exemplars and part carried raw answers. That would confound group-disjointness with a
change in exemplar content, and the comparison against the stratified estimate would no longer be
about disjointness. Restricting to the training partition keeps exemplar preprocessing identical on
both sides of the manipulation.

Whole (source document, focus entity) groups are packed 80/10/10, so no group appears in more than
one partition. Writes work2/data_groupsplit/medquad_gs/{train,validation,test}.jsonl, a summary list
aligned to the new train file, and a manifest that verifies disjointness.
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
OUTDIR = ws("data_groupsplit", "medquad_gs")
os.makedirs(OUTDIR, exist_ok=True)
rows = rd(src("data", "medquad", "train.jsonl"))
summ = json.load(open(src("outputs_llm", "train_summaries_medquad.json")))
meta = json.load(open(ws("results", "medquad_recovered_meta.json")))
recs = (meta.get("train") or {}).get("records") or []
assert len(summ) == len(rows), (len(summ), len(rows))
def key(i):
    m = recs[i] if i < len(recs) else {}
    return ((m or {}).get("doc") or "", ((m or {}).get("focus") or "").strip().lower())
groups = defaultdict(list)
ungrouped = []
for i in range(len(rows)):
    k = key(i)
    (groups[k] if any(k) else ungrouped).append(i)
# greedy largest-first packing, whole groups only
targets = {"train": 0.8, "validation": 0.1, "test": 0.1}
tot = sum(len(v) for v in groups.values())
cap = {k: v * tot for k, v in targets.items()}
have = {k: 0 for k in targets}
parts = {k: [] for k in targets}
for k, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    p = min(have, key=lambda x: have[x] / max(cap[x], 1e-9))
    parts[p] += idxs; have[p] += len(idxs)
# verification: no group key may appear in two partitions
keysets = {p: {key(i) for i in idxs} for p, idxs in parts.items()}
overlap = {f"{a}&{b}": len(keysets[a] & keysets[b])
           for a in keysets for b in keysets if a < b}
assert all(v == 0 for v in overlap.values()), overlap
for p, idxs in parts.items():
    idxs.sort()
    with open(os.path.join(OUTDIR, f"{p}.jsonl"), "w", encoding="utf-8") as f:
        for i in idxs:
            f.write(json.dumps(rows[i]) + "\n")
json.dump([summ[i] for i in sorted(parts["train"])],
          open(os.path.join(OUTDIR, "train_summaries.json"), "w"))
man = dict(source="data/medquad/train.jsonl (training partition only, so exemplar summaries exist for every pool item)",
           grouping="(source document, focus entity), whole groups packed 80/10/10",
           corpus_items=len(rows), grouped_items=tot, ungrouped_items_dropped=len(ungrouped),
           groups=len(groups), split_sizes={p: len(v) for p, v in parts.items()},
           group_overlap_between_partitions=overlap,
           disjoint=all(v == 0 for v in overlap.values()))
json.dump(man, open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=1)
json.dump(man, open(ws("results", "group_split_build.json"), "w"), indent=1)
print(json.dumps(man, indent=1))
