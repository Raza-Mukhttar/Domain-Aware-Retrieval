"""Verifier evaluation with uncertainty: bootstrap 95% CIs for
ROC-AUC/PR-AUC under BOTH constructions of the original Table 6 (hard-negative
classifier set; random-negative reranker set), on the SAME 300 test questions, plus
a query-level top-1 selection metric computed on the saved candidate pools of the
matched experiment (does the verifier pick the candidate with the best reference
score?)."""

import os, sys
import json, glob, os, sys, numpy as np, torch, faiss
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import src, ws
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--verifier",default=None,help="verifier directory or hub id; defaults to $MEDLM_VERIFIER"); ap.add_argument("--exp",required=True,help="openended experiment dir with preds_*.jsonl")
ap.add_argument("--dataset",default="medquad"); a=ap.parse_args()
if a.verifier is None: a.verifier = src("outputs_llm", "verifier")
dev="cuda"; tok=AutoTokenizer.from_pretrained(a.verifier); vm=AutoModelForSequenceClassification.from_pretrained(a.verifier).to(dev).eval()
def score(pairs):
    out=[]
    for i in range(0,len(pairs),32):
        enc=tok([f"Q: {q} [SEP] A: {x}" for q,x in pairs[i:i+32]],return_tensors="pt",padding=True,truncation=True,max_length=256).to(dev)
        with torch.no_grad(): out+=torch.softmax(vm(**enc).logits,-1)[:,1].float().cpu().tolist()
    return np.array(out)
rd=lambda p:[json.loads(l) for l in open(p)]
d=src("data","medquad" if a.dataset=="medquad" else "iCliniq"); train=rd(f"{d}/train.jsonl")[:2000]; test=rd(f"{d}/test.jsonl")
man=json.load(open(os.path.join(a.exp,"manifest.json"))); tidx=man["test_ids"]; T=[test[i] for i in tidx]
enc=SentenceTransformer(src("all-MiniLM-L6-v2"),device=dev)
E=enc.encode([r["question"] for r in train],convert_to_numpy=True,batch_size=256).astype("float32"); faiss.normalize_L2(E); ix=faiss.IndexFlatIP(E.shape[1]); ix.add(E)
Q=enc.encode([r["question"] for r in T],convert_to_numpy=True,batch_size=256).astype("float32"); faiss.normalize_L2(Q); _,I=ix.search(Q,3)
rng=np.random.default_rng(42)
sets={}
# construction A (original 'classifier'): 1 positive + 2 retrieved hard negatives
pa,la,qa=[],[],[]
for i,r in enumerate(T):
    pa.append((r["question"],r["answer"])); la.append(1); qa.append(i)
    for j in I[i][:2]: pa.append((r["question"],train[int(j)]["answer"])); la.append(0); qa.append(i)
sets["hard_negatives_1to2"]=(pa,np.array(la),np.array(qa))
# construction B (original 'reranker'): 1 positive + 1 random negative
pb,lb,qb=[],[],[]
for i,r in enumerate(T):
    pb.append((r["question"],r["answer"])); lb.append(1); qb.append(i)
    pb.append((r["question"],train[int(rng.integers(len(train)))]["answer"])); lb.append(0); qb.append(i)
sets["random_negatives_1to1"]=(pb,np.array(lb),np.array(qb))
res={}
for name,(pairs,lab,qid) in sets.items():
    s=score(pairs); roc=roc_auc_score(lab,s); pr,rc,_=precision_recall_curve(lab,s); pra=auc(rc,pr)
    rocs,pras=[],[]; uq=np.unique(qid)
    for _ in range(2000):   # question-level bootstrap
        take=rng.choice(uq,len(uq),replace=True); m=np.concatenate([np.where(qid==q)[0] for q in take])
        rocs.append(roc_auc_score(lab[m],s[m])); p_,r_,_=precision_recall_curve(lab[m],s[m]); pras.append(auc(r_,p_))
    res[name]=dict(n=len(lab),n_questions=len(uq),positive_prevalence=round(float(lab.mean()),3),roc_auc=round(roc,4),roc_auc_ci95=[round(np.percentile(rocs,2.5),4),round(np.percentile(rocs,97.5),4)],
                   pr_auc=round(pra,4),pr_auc_ci95=[round(np.percentile(pras,2.5),4),round(np.percentile(pras,97.5),4)],no_skill_pr_auc=round(float(lab.mean()),3))
    print(name,res[name],flush=True)
# query-level selection: on the saved N=5 candidate pools, how often does verifier-argmax equal ROUGE-1-best candidate? and mean regret
import re
from rouge_score import rouge_scorer
rs=rouge_scorer.RougeScorer(["rouge1"],use_stemmer=True)
# rebuild candidate pools from the saved per-selector predictions (first/verifier/entity/combined share the pool) -> use preds files' 'candidates' if present
cand_file=os.path.join(a.exp,"cands_rag_n5.jsonl")
if os.path.exists(cand_file):
    rows=rd(cand_file); hit=0; regret=[]; oracle=[]; first=[]; chosen=[]
    for r in rows:
        r1=[rs.score(r["reference"],c)["rouge1"].fmeasure for c in r["candidates"]]
        vi=int(np.argmax(r["verifier"])); bi=int(np.argmax(r1)); hit+=vi==bi; regret.append(r1[bi]-r1[vi]); oracle.append(r1[bi]); first.append(r1[0]); chosen.append(r1[vi])
    res["query_level_selection"]=dict(n=len(rows),top1_agreement_with_oracle_pct=round(100*hit/len(rows),1),mean_rouge1_regret_pct=round(100*float(np.mean(regret)),2),
        oracle_rouge1=round(100*float(np.mean(oracle)),2),first_beam_rouge1=round(100*float(np.mean(first)),2),verifier_pick_rouge1=round(100*float(np.mean(chosen)),2))
    print("query-level:",res["query_level_selection"])
json.dump(res,open(ws("results",f"verifier_eval_{a.dataset}.json"),"w"),indent=1)
