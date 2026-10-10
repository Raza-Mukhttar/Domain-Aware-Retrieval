# -*- coding: utf-8 -*-
"""-- find a REAL example of a clinically weighty qualifier dropped by the summariser.

The audit already quantifies the loss (6.7 % of MedQuAD answers drop at least one qualifier). What
was missing was a quotable instance. The earlier selector required a lost dose AND a short answer AND
the token "not" simultaneously, which nothing satisfied, so it returned nothing.

This searches for the case that best illustrates the failure: a source answer containing a hedge or
restriction that carries clinical weight, whose summary drops it, scored so that negations and
contraindications rank above weak hedges, and shorter texts rank above long ones so the example is
quotable. Candidates are printed for inspection rather than auto-inserted.
"""
import json, os, re, sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import src, ws
import medlm.openended_compat as R
# graded by how much the loss changes clinical meaning
STRONG = re.compile(r"\b(should not|must not|do not|never|contraindicat\w+|not recommended|avoid|"
                    r"only if|unless|except|no evidence|does not|cannot|is not|are not)\b", re.I)
WEAK = re.compile(r"\b(may|can|rare(?:ly)?|risk|possible|sometimes|often|usually|might)\b", re.I)
DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|%|percent|hours?|days?|weeks?|months?|years?)\b", re.I)
def sents(t): return [s.strip() for s in re.split(r"(?<=[.;])\s+", t or "") if s.strip()]
rows = [json.loads(l) for l in open(src("data", "medquad", "train.jsonl"), encoding="utf-8")]
summ = json.load(open(src("outputs_llm", "train_summaries_medquad.json"), encoding="utf-8"))
n = min(len(rows), len(summ))
cands = []
for i in range(n):
    a = rows[i]["answer"] or ""
    s = R.strip_echo(summ[i]) or ""
    if not a or not s: continue
    sl = s.lower()
    # a strong qualifier present in the source and absent from the summary
    lost_strong = sorted({m.group(0).lower() for m in STRONG.finditer(a)} - {m.group(0).lower() for m in STRONG.finditer(s)})
    lost_dose = sorted({m.group(0).lower() for m in DOSE.finditer(a)} - {m.group(0).lower() for m in DOSE.finditer(s)})
    if not lost_strong: continue
    # the sentence that carries the lost qualifier, and a check that its content is really gone
    carrier = next((x for x in sents(a) if STRONG.search(x) and
                    any(q in x.lower() for q in lost_strong)), None)
    if not carrier: continue
    # require the carrier's distinctive content words to be absent from the summary too, so we are not
    # flagging a paraphrase that preserved the meaning
    content = [w for w in re.findall(r"[a-z]{5,}", carrier.lower())
               if w not in ("should", "would", "could", "there", "these", "those", "which", "while")]
    kept = sum(1 for w in set(content) if w in sl)
    frac_kept = kept / max(len(set(content)), 1)
    score = (3 * len(lost_strong) + 2 * len(lost_dose) + (1 - frac_kept) * 4
             - 0.002 * len(a) - 0.001 * len(carrier))
    cands.append(dict(index=i, score=round(score, 2), question=rows[i]["question"],
                      answer_len=len(a), summary_len=len(s), lost_strong=lost_strong,
                      lost_dose=lost_dose, carrier=carrier, frac_content_kept=round(frac_kept, 2),
                      answer=a, summary=s))
cands.sort(key=lambda c: -c["score"])
print(f"scanned {n} pairs; {len(cands)} candidates where a strong qualifier is lost\n")
for c in cands[:6]:
    print("=" * 110)
    print(f"[{c['index']}] score {c['score']} | lost {c['lost_strong']} {c['lost_dose']} | "
          f"content words of the carrier retained in the summary: {c['frac_content_kept']:.0%}")
    print(f"Q: {c['question'][:110]}")
    print(f"CARRIER SENTENCE (source): {c['carrier'][:300]}")
    print(f"SUMMARY ({c['summary_len']} chars): {c['summary'][:300]}")
json.dump(cands[:12], open(ws("results", "qualifier_candidates.json"), "w"), indent=1)
print(f"\nwrote {min(12,len(cands))} candidates to results/qualifier_candidates.json")
