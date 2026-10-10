# -*- coding: utf-8 -*-
"""Aggregate every rerun into results/final_v2.json:
 open-ended per-condition summaries (per seed), multi-seed mean+/-SD, paired contrasts
 (seed 42, incl. decoding-time entity control), closed-form summaries for three
 backbones with McNemar, latency/memory tables, training curves (train + validation),
 grouped-split subset analysis (MedQuAD focus/document held out of the pool),
 exemplar contamination, and success/failure examples."""

import os, sys
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon, binomtest
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "scripts", "pipeline"))
from medlm.legacy_paths import ws, src
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M=["rouge1","rouge2","rougeL","meteor","bertscore_f1"]; NB=10000; rng=np.random.default_rng(42)
def rd(p): return [json.loads(l) for l in open(p,encoding="utf-8")]
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*p[i]); adj[i]=min(1,run)
    return adj
OUT={}
# ---------------- open-ended
oe={}
for d in sorted(glob.glob(ws("experiments","openended_*"))):
    if not os.path.exists(f"{d}/metrics.json"): continue
    name=os.path.basename(d)[len("openended_"):]; ds,lab=name.split("_",1)
    agg=json.load(open(f"{d}/metrics.json"))
    for f in glob.glob(f"{d}/preds_*.jsonl"):
        cond=os.path.basename(f)[6:-6]; oe[(ds,lab,cond)]=dict(rows=rd(f),agg=agg.get(cond,{}),dir=d)
SEEDLAB={"llama31s42":42,"llama31s1":1,"llama31s2":2}; MAIN="llama31s42"; CONTR="llama31cs42"   # CONTR = seed-42 generator + contrastive stage
OUT["open_ended_summary"]={f"{ds}::{lab}::{c}":{m:v["agg"].get(m) for m in M}|{"entity_retention_Eq":v["agg"].get("entity_retention_Eq"),"meta":v["agg"].get("_meta",{})} for (ds,lab,c),v in oe.items()}
# multi-seed mean+/-SD over seeds 42,1,2 for conditions present in all
ms={}; TSEED={"llama31T42":42,"llama31t1":1,"llama31t2":2}   # seed 42 corrected numbers all come from the six-prompt-set run
for ds in ("medquad","icliniq"):
    for c in sorted({c for (d_,s_,c) in oe if d_==ds}):
        for labs,suffix in ((SEEDLAB,""),(TSEED,"::fixed_truncation")):
            vals={m:[oe[(ds,s,c)]["agg"][m]["mean_pct"] for s in labs if (ds,s,c) in oe and m in oe[(ds,s,c)]["agg"]] for m in M}
            if len(vals["rouge1"])==3: ms[f"{ds}::{c}{suffix}"]={m:dict(mean=round(float(np.mean(v)),2),sd=round(float(np.std(v,ddof=1)),2),values=v) for m,v in vals.items()}
OUT["multi_seed"]=ms
# paired contrasts (seed 42)
PAIRS=[("rag_n5_combined","norag_n5_combined","retrieval"),("norag_n5_first","norag_n1_first","budget_norag"),("rag_n5_first","rag_n1_first","budget_rag"),
       ("rag_n5_combined","rag_n5_first","rerank"),("rag_n5_verifier","rag_n5_first","verifier"),("rag_n5_entity","rag_n5_first","entity_rerank"),
       ("rag_qtype_n5_combined","rag_n5_combined","qtype"),("rag_fullans_n5_combined","rag_n5_combined","unsummarised"),
       ("rag_edec_n5_first","rag_n5_first","entity_decoding"),("rag_edec_n5_combined","rag_n5_combined","entity_decoding_plus_rerank"),("norag_edec_n5_first","norag_n5_first","entity_decoding_norag"),
       ((CONTR,"rag_n5_combined"),(MAIN,"rag_n5_combined"),"contrastive_rag"),((CONTR,"norag_n5_combined"),(MAIN,"norag_n5_combined"),"contrastive_norag"),((CONTR,"rag_n5_first"),(MAIN,"rag_n5_first"),"contrastive_rag_first"),
       # C13 of the consistency audit: the submitted inference truncated prompts from the RIGHT at 512 tokens,
       # which removes the target question from retrieval prompts. "t" labels repeat the runs with left truncation.
       (("llama31T42","rag_n5_combined"),("llama31T42","norag_n5_combined"),"retrieval_fixed_truncation"),
       (("llama31T42","rag_n5_combined"),(MAIN,"rag_n5_combined"),"truncation_policy_rag"),
       (("llama31T42","norag_n5_combined"),(MAIN,"norag_n5_combined"),"truncation_policy_norag"),
       (("llama31T42","rag_n5_first"),("llama31T42","norag_n5_first"),"retrieval_fixed_truncation_first"),
       # full component grid under the corrected policy (label llama31T42; contrastive checkpoint llama31cT42)
       (("llama31T42","rag_n5_combined"),("llama31T42","norag_n5_combined"),"C_retrieval"),
       (("llama31T42","norag_n5_first"),("llama31T42","norag_n1_first"),"C_budget_norag"),
       (("llama31T42","rag_n5_first"),("llama31T42","rag_n1_first"),"C_budget_rag"),
       (("llama31T42","rag_n5_combined"),("llama31T42","rag_n5_first"),"C_rerank"),
       (("llama31T42","rag_n5_verifier"),("llama31T42","rag_n5_first"),"C_verifier"),
       (("llama31T42","rag_n5_entity"),("llama31T42","rag_n5_first"),"C_entity_rerank"),
       (("llama31T42","rag_qtype_n5_combined"),("llama31T42","rag_n5_combined"),"C_qtype"),
       (("llama31T42","rag_fullans_n5_combined"),("llama31T42","rag_n5_combined"),"C_unsummarised"),
       (("llama31T42","rag_edec_n5_first"),("llama31T42","rag_n5_first"),"C_entity_decoding"),
       (("llama31T42","rag_edec_n5_combined"),("llama31T42","rag_n5_combined"),"C_entity_decoding_plus_rerank"),
       (("llama31T42","norag_edec_n5_first"),("llama31T42","norag_n5_first"),"C_entity_decoding_norag"),
       (("llama31cT42","rag_n5_combined"),("llama31T42","rag_n5_combined"),"C_contrastive_rag"),
       (("llama31cT42","norag_n5_combined"),("llama31T42","norag_n5_combined"),"C_contrastive_norag")]
pc=[]
def _key(ds,x): return (ds,x[0],x[1]) if isinstance(x,tuple) else (ds,MAIN,x)
for ds in ("medquad","icliniq"):
    for a,b,lab in PAIRS:
        ka,kb=_key(ds,a),_key(ds,b)
        if ka not in oe or kb not in oe: continue
        a,b=f"{ka[1]}::{ka[2]}",f"{kb[1]}::{kb[2]}"
        ra={r["test_idx"]:r for r in oe[ka]["rows"]}; rb={r["test_idx"]:r for r in oe[kb]["rows"]}; ids=sorted(set(ra)&set(rb))
        rec=dict(dataset=ds,contrast=lab,a=a,b=b,n=len(ids),metrics={}); pr=[]
        for m in M:
            xa=np.array([ra[i][m] for i in ids]); xb=np.array([rb[i][m] for i in ids]); dd=xa-xb
            bs=dd[rng.integers(0,len(dd),(NB,len(dd)))].mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
            try: p=wilcoxon(xa,xb,zero_method="wilcox").pvalue
            except ValueError: p=1.0
            rec["metrics"][m]=dict(a=round(100*xa.mean(),2),b=round(100*xb.mean(),2),diff=round(100*dd.mean(),2),ci=[round(100*lo,2),round(100*hi,2)],p_raw=float(p)); pr.append(p)
        for m,pa in zip(M,holm(pr)): rec["metrics"][m]["p_holm"]=float(pa); rec["metrics"][m]["sig"]=bool(pa<0.05)
        pc.append(rec)
OUT["paired"]=pc
# seed-level paired contrasts (retrieval effect per seed) -> shows sign stability
OUT["retrieval_effect_by_seed_fixed_truncation"]={f"{ds}::{'llama31T42' if s_==42 else f'llama31t{s_}'}":{m:round(oe[(ds,('llama31T42' if s_==42 else f"llama31t{s_}"),"rag_n5_combined")]["agg"][m]["mean_pct"]-oe[(ds,('llama31T42' if s_==42 else f"llama31t{s_}"),"norag_n5_combined")]["agg"][m]["mean_pct"],2) for m in M}
                                 for ds in ("medquad","icliniq") for s_ in (42,1,2) if (ds,('llama31T42' if s_==42 else f"llama31t{s_}"),"rag_n5_combined") in oe and (ds,('llama31T42' if s_==42 else f"llama31t{s_}"),"norag_n5_combined") in oe}
OUT["retrieval_effect_by_seed"]={f"{ds}::{s}":{m:round(oe[(ds,s,"rag_n5_combined")]["agg"][m]["mean_pct"]-oe[(ds,s,"norag_n5_combined")]["agg"][m]["mean_pct"],2) for m in M}
                                 for ds in ("medquad","icliniq") for s in SEEDLAB if (ds,s,"rag_n5_combined") in oe and (ds,s,"norag_n5_combined") in oe}
# ---------------- closed-form (three backbones)
cf={};
for d in sorted(glob.glob(ws("experiments","closed_*"))):
    if not os.path.exists(f"{d}/metrics.json"): continue
    base=os.path.basename(d)[len("closed_"):]; ds=next(k for k in ("mmlu_med","medmcqa","pubmedqa","medqa") if base.startswith(k)); lab=base[len(ds)+1:]
    agg=json.load(open(f"{d}/metrics.json"))
    for f in glob.glob(f"{d}/preds_*.jsonl"): cond=os.path.basename(f)[6:-6]; cf[(lab,ds,cond)]=dict(rows=rd(f),agg=agg.get(cond,{}))
OUT["closed_form"]={f"{l}::{d}::{c}":v["agg"] for (l,d,c),v in cf.items()}
mc=[]
def _mc(la,ca_,lb,cb_,d):
    """paired McNemar between two (label, condition) runs on the same test items"""
    if (la,d,ca_) not in cf or (lb,d,cb_) not in cf: return
    ca={r["test_index"]:r["correct"] for r in cf[(la,d,ca_)]["rows"]}; cb={r["test_index"]:r["correct"] for r in cf[(lb,d,cb_)]["rows"]}
    ids=sorted(set(ca)&set(cb))
    if not ids: return
    n10=sum(ca[i]==1 and cb[i]==0 for i in ids); n01=sum(ca[i]==0 and cb[i]==1 for i in ids); n=n10+n01
    mc.append(dict(backbone=la,dataset=d,a=ca_,b=(cb_ if la==lb else f"{lb}::{cb_}"),n=len(ids),b_only=n10,c_only=n01,
                   p=float(binomtest(n10,n,0.5).pvalue) if n else 1.0,
                   acc_a=cf[(la,d,ca_)]["agg"]["accuracy_pct"],acc_b=cf[(lb,d,cb_)]["agg"]["accuracy_pct"]))
for (l,d,c) in list(cf):
    for a,b in (("dense_rag","vanilla"),("full","vanilla"),("full","dense_rag"),("fewshot","vanilla"),("dense_rag","fewshot"),
                ("adapted_verifier","adapted"),("adapted_rag","adapted"),("adapted_rag_verifier","adapted_rag"),("adapted_verifier","adapted_rag")):
        if c==a: _mc(l,a,l,b,d)
# task-adapted (full framework) versus the untrained backbone: same items, different label
for l in sorted({l for (l,_,_) in cf if l.endswith("ft")}):
    base=l[:-2]
    for d in sorted({d for (l2,d,_) in cf if l2==l}):
        for a,b in (("adapted","vanilla"),("adapted_verifier","vanilla"),("adapted_verifier","dense_rag")):
            _mc(l,a,base,b,d)
        # C03 of the consistency audit: same comparison against controls run under the SAME prompt head (C.1)
        for a,b in (("adapted","vanilla"),("adapted_verifier","vanilla"),("adapted_verifier","dense_rag"),("adapted_rag","dense_rag")):
            _mc(l,a,base+"c1",b,d)
OUT["closed_mcnemar"]=mc
# ---------------- latency / memory / training cost
def _hist(name, d):
    """training log of a run, from the live checkpoint or from the preserved copy"""
    for p_ in (os.path.join(d,"log_history.json"), ws("results","training_manifests",f"{name}__log_history.json")):
        if os.path.exists(p_): return json.load(open(p_))
    return None
def _final_eval(name, d):
    h=_hist(name,d)
    return next((x["eval_loss"] for x in reversed(h) if "eval_loss" in x), None) if h else None
lat={}
for (ds,s,c),v in oe.items():
    if s!=MAIN or not c.endswith(("_n5_combined","_n1_first")): continue
    m=v["agg"].get("_meta",{}); lat[f"{ds}::{c}"]=dict(gen_s_per_item=m.get("generation_seconds_per_item"),verify_s_per_item=m.get("verifier_entity_seconds_per_item"),
        retrieval_s_total=m.get("retrieval_seconds"),peak_gpu_gb=m.get("generation_peak_gpu_mem_gb"))
for (l,d,c),v in cf.items(): lat[f"closed::{l}::{d}::{c}"]=dict(s_per_item=v["agg"].get("seconds_per_item"),peak_gpu_gb=v["agg"].get("peak_gpu_mem_gb"))
tr={}
_live={os.path.basename(d):f"{d}/manifest.json" for d in glob.glob(ws("experiments","ckpt_*"))+glob.glob(ws("experiments","ckptcf_*")) if os.path.exists(f"{d}/manifest.json")}
# checkpoints deleted to free disk: their manifests and loss histories were preserved
for f in sorted(glob.glob(ws("results","training_manifests","*__manifest.json"))):
    _live.setdefault(os.path.basename(f).split("__")[0], f)
for name,mp in sorted(_live.items()):
    d=ws("experiments",name); mm=json.load(open(mp))
    if "stage_seconds" in mm: tr[name]=dict(contrastive=True,stage_min=round(mm["stage_seconds"]/60,2),peak_gpu_gb=mm["peak_gpu_mem_gb"],steps=mm["steps"],loss_first=round(mm["loss_first"],4),loss_last=round(mm["loss_last"],4),mean_abs_embedding_change=mm["mean_abs_embedding_change"],n_train=mm["n_train"])
    else: tr[name]=dict(train_min=round(mm["train_seconds"]/60,1),peak_gpu_gb=mm["peak_gpu_mem_gb"],n_gpus=mm["n_gpus"],trainable_params=mm["trainable_params"],effective_batch=mm["effective_batch"],epochs=mm["epochs"],lr=mm["lr"],seed=mm["seed"],final_eval_loss=_final_eval(name,d))
OUT["latency"]=lat; OUT["training"]=tr
# ---------------- curves (train + validation loss)
os.makedirs(ws("figures"),exist_ok=True)
for ds in ("medquad","icliniq"):
    fig,ax=plt.subplots(figsize=(3.6,2.6)); any_=False
    for s,col in ((42,"#0B3D91"),(1,"#c05621"),(2,"#2f855a")):
        H=_hist(f"ckpt_{ds}_llama31_seed{s}", ws("experiments",f"ckpt_{ds}_llama31_seed{s}"))
        if not H: continue
        trn=[(h["step"],h["loss"]) for h in H if "loss" in h]; ev=[(h["step"],h["eval_loss"]) for h in H if "eval_loss" in h]
        if trn: ax.plot(*zip(*trn),color=col,lw=0.9,alpha=0.6,label=f"train, seed {s}")
        if ev: ax.plot(*zip(*ev),color=col,lw=1.6,marker="o",ms=2.5,label=f"validation, seed {s}"); any_=True
    ax.set_xlabel("optimiser step (effective batch 16)"); ax.set_ylabel("cross-entropy loss"); ax.set_title(ds,fontsize=9); ax.legend(fontsize=6,frameon=False); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(ws("figures",f"loss_{ds}.pdf")); plt.close(fig)
# ---------------- grouped-split subset (MedQuAD): test items whose focus / source document is absent from the retrieval pool
try:
    meta=json.load(open(ws("results","medquad_recovered_meta.json")))
    man=json.load(open(ws("experiments","openended_medquad_llama31s42","manifest.json")))
    pool_ids=set(man["train_ids"]); trrec=meta["train"]["records"]; terec=meta["test"]["records"]
    pool_foci={trrec[i].get("focus","").lower() for i in pool_ids if i<len(trrec)}; pool_docs={trrec[i].get("doc","") for i in pool_ids if i<len(trrec)}
    grp={}
    for cond in ("norag_n5_combined","rag_n5_combined","rag_qtype_n5_combined"):
        if ("medquad",MAIN,cond) not in oe: continue
        rows=oe[("medquad",MAIN,cond)]["rows"]; held=[r for r in rows if terec[r["test_idx"]].get("focus","").lower() not in pool_foci]; seen=[r for r in rows if r not in held]
        grp[cond]={k:dict(n=len(v),rouge1=round(100*np.mean([r["rouge1"] for r in v]),2),meteor=round(100*np.mean([r["meteor"] for r in v]),2),bertscore=round(100*np.mean([r["bertscore_f1"] for r in v]),2)) for k,v in (("focus_held_out",held),("focus_in_pool",seen))}
    OUT["grouped_split"]=dict(pool_size=len(pool_ids),test_focus_in_pool_pct=round(100*sum(1 for r in terec[:len(terec)] if r.get("focus","").lower() in pool_foci)/len(terec),1),by_condition=grp)
except Exception as e: OUT["grouped_split"]={"error":str(e)}
# ---------------- contamination + success/failure examples (seed 42)
from medlm.entities import extract_entities, load_vocab
V=load_vocab(src("outputs_llm","medical_entities.json"))
import medlm.openended_compat as R
con={}; ex={}
for ds_lab in ("medquad::llama31s42","icliniq::llama31s42","medquad::llama31T42","icliniq::llama31T42"):
    ds,_lab=ds_lab.split("::")
    d=ws("experiments",f"openended_{ds}_{_lab}")
    if not os.path.exists(f"{d}/manifest.json"): continue
    man=json.load(open(f"{d}/manifest.json")); train,_=R.load_split(ds,man["n_train_pool"],man["n_test"],man["seed"])
    for cond in ("norag_n5_combined","rag_n5_combined","rag_qtype_n5_combined","rag_edec_n5_combined"):
        f=f"{d}/preds_{cond}.jsonl"
        if not os.path.exists(f): continue
        rows=rd(f); imp=[]; tot=[]; nw=0
        for r in rows:
            a=extract_entities(r["prediction"],V); q=extract_entities(r["question"],V); g=extract_entities(r["reference"],V)
            e=extract_entities(" ".join(train[i]["summary"] for i in r.get("exemplar_ids",[]) if 0<=i<len(train)),V) if r.get("exemplar_ids") else set()
            im=(a&e)-q-g; imp.append(len(im)); tot.append(len(a)); nw+=bool(im)
        con[f"{ds_lab}::{cond}"]=dict(n=len(rows),entities_per_answer=round(float(np.mean(tot)),2),imported_per_answer=round(float(np.mean(imp)),3),answers_with_import_pct=round(100*nw/len(rows),1))
    # success / failure examples: largest ROUGE-1 change rag vs norag
    if _lab==MAIN and (ds,MAIN,"rag_n5_combined") in oe and (ds,MAIN,"norag_n5_combined") in oe:
        ra={r["test_idx"]:r for r in oe[(ds,MAIN,"rag_n5_combined")]["rows"]}; rb={r["test_idx"]:r for r in oe[(ds,MAIN,"norag_n5_combined")]["rows"]}
        diffs=sorted(((ra[i]["rouge1"]-rb[i]["rouge1"],i) for i in ra if i in rb))
        pick=[("retrieval_hurt",diffs[0][1]),("retrieval_helped",diffs[-1][1])]
        ex[ds]=[dict(kind=k,question=ra[i]["question"][:300],reference=ra[i]["reference"][:400],with_retrieval=ra[i]["prediction"][:400],without_retrieval=rb[i]["prediction"][:400],rouge1_with=round(100*ra[i]["rouge1"],1),rouge1_without=round(100*rb[i]["rouge1"],1)) for k,i in pick]
OUT["contamination"]=con; OUT["examples"]=ex
json.dump(OUT,open(ws("results","final_v2.json"),"w"),indent=1)
print("open-ended conditions:",len(oe),"| multi-seed:",len(ms),"| paired:",len(pc),"| closed:",len(cf),"| mcnemar:",len(mc),"| training runs:",len(tr))
for r in pc:
    if r["dataset"]=="medquad": print(f"  {r['contrast']:28s} dR1={r['metrics']['rouge1']['diff']:+6.2f} {r['metrics']['rouge1']['ci']} p={r['metrics']['rouge1']['p_holm']:.3g}")
print("retrieval effect by seed:",OUT["retrieval_effect_by_seed"]); print("grouped:",{k:v for k,v in OUT.get("grouped_split",{}).items() if k!="by_condition"})
