# -*- coding: utf-8 -*-
"""Is a source/topic-disjoint repartition of MedQuAD viable at all?

Groups the whole corpus by (source document, focus entity), assigns WHOLE groups to an 80/10/10
split, and reports how many items land in a test split with zero cross-partition overlap. The spec
is explicit that the number itself is the result: if the disjoint test split is too small to support
paired tests, that is a publishable statement about the corpus and the rest of Section 4.3 should not
start. Writes results/group_split_feasibility.json.
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
meta = json.load(open(ws("results", "medquad_recovered_meta.json")))
OUT = {}
items = []      # (split, idx, doc, focus)
for split in ("train", "validation", "test"):
    rows = rd(src("data", "medquad", f"{split}.jsonl"))
    recs = (meta.get(split) or {}).get("records") or []
    for i, r in enumerate(rows):
        m = recs[i] if i < len(recs) else {}
        items.append((split, i, (m or {}).get("doc") or "", ((m or {}).get("focus") or "").strip().lower()))
n = len(items)
matched = sum(1 for x in items if x[2] or x[3])
OUT["corpus_items"] = n
OUT["items_with_metadata"] = matched
OUT["items_without_metadata"] = n - matched
for keyname, keyfn in (("source_document", lambda x: x[2]),
                       ("focus_entity", lambda x: x[3]),
                       ("source_and_focus", lambda x: (x[2], x[3]))):
    groups = defaultdict(list)
    for x in items:
        k = keyfn(x)
        if (isinstance(k, tuple) and not any(k)) or (not isinstance(k, tuple) and not k):
            continue                                   # unmatched items cannot be grouped
        groups[k].append(x)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    # greedy largest-first packing into 80/10/10 by item count, whole groups only
    targets = {"train": 0.8, "validation": 0.1, "test": 0.1}
    tot = sum(sizes)
    cap = {k: v * tot for k, v in targets.items()}
    have = {k: 0 for k in targets}
    assign = {}
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        part = min(have, key=lambda p: have[p] / max(cap[p], 1e-9))
        assign[k] = part; have[part] += len(v)
    OUT[keyname] = dict(
        groups=len(groups), grouped_items=tot,
        largest_group=sizes[0] if sizes else 0,
        median_group=int(np.median(sizes)) if sizes else 0,
        singleton_groups=sum(1 for s in sizes if s == 1),
        split_items={p: have[p] for p in targets},
        disjoint_test_items=have["test"],
        disjoint_test_pct_of_corpus=round(100 * have["test"] / n, 2))
# what the current (random) partition looks like for comparison
cur = Counter(x[0] for x in items)
OUT["current_random_partition"] = dict(cur)
g = OUT["source_and_focus"]
OUT["verdict"] = (
    f"A whole-group 80/10/10 split on (source document, focus entity) yields {g['disjoint_test_items']} "
    f"disjoint test items from {g['groups']:,} groups. "
    + ("That is large enough to support the paired tests used elsewhere in the paper (n>=250), so the repartition is viable."
       if g["disjoint_test_items"] >= 250 else
       f"That is below the ~300 questions the paper's paired tests use, so a repartition would change the "
       f"estimand as well as the split, and the stratified analysis plus the source-disjoint retrieval test remain "
       f"the informative evidence."))
json.dump(OUT, open(ws("results", "group_split_feasibility.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != "verdict"}, indent=1))
print("\nVERDICT:", OUT["verdict"])
