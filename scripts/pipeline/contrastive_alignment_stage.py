"""Contrastive representation-alignment stage, re-implemented exactly as in the
submitted code (medlm_llm_pipeline_fixed.py::contrastive_finetune_generator_encoder):
trains the generator's input-embedding table and a hidden->384 linear projection
with an InfoNCE-style loss (one random negative, tau=0.07, Adam at half the
generator learning rate, 2 epochs, batch 8) on the SAME 500 training prompts used
for the generator fine-tuning of the given checkpoint. The FAISS index is not
rebuilt (as in the original). Output: a new checkpoint directory so that the
stage can be ablated at inference (with vs without)."""

import argparse, json, os, random, time
from medlm.legacy_paths import ws
import numpy as np, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from medlm.openended_compat import load_split, Retriever, build_prompt, set_seed, EMBEDDER
ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--ckpt",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--bs",type=int,default=8)
ap.add_argument("--lr",type=float,default=5e-6*0.5); ap.add_argument("--tau",type=float,default=0.07); ap.add_argument("--max_input",type=int,default=512)
a=ap.parse_args(); set_seed(a.seed); random.seed(a.seed); os.makedirs(a.out,exist_ok=True); t0=time.time()
man=json.load(open(os.path.join(a.ckpt,"manifest.json")))
pool,_=load_split(a.dataset,man.get("pool",2000),10,a.seed); byidx={r["idx"]:r for r in pool}
rows=[byidx[i] for i in man["train_row_ids"] if i in byidx]
retr=Retriever(pool,"cuda",False); qe=retr.encode([r["question"] for r in rows]); train=[]
for i,r in enumerate(rows):
    ids=[j for j in retr.search(qe[i],"general",4) if pool[j]["idx"]!=r["idx"]][:3]
    train.append(dict(prompt=build_prompt(r["question"],[pool[j] for j in ids],"summary"),summary=r["summary"]))
del retr; torch.cuda.empty_cache()
tok=AutoTokenizer.from_pretrained(a.ckpt); tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(a.ckpt,dtype=torch.bfloat16,device_map={"":0}); model.config.use_cache=False
for p in model.parameters(): p.requires_grad_(False)
encoder=model.get_input_embeddings(); encoder.weight.requires_grad_(True)
embedder=SentenceTransformer(EMBEDDER,device="cuda"); dt=encoder.weight.dtype
projection=nn.Linear(model.config.hidden_size,embedder.get_sentence_embedding_dimension(),dtype=dt).cuda()
opt=torch.optim.Adam(list(encoder.parameters())+list(projection.parameters()),lr=a.lr)
hist=[]; torch.cuda.reset_peak_memory_stats(); t1=time.time()
for ep in range(a.epochs):
    perm=list(range(len(train))); random.shuffle(perm)
    for i in range(0,len(train),a.bs):
        batch=[train[j] for j in perm[i:i+a.bs]]; neg=[train[random.randrange(len(train))]["summary"] for _ in batch]
        enc=tok([x["prompt"] for x in batch],return_tensors="pt",truncation=True,padding=True,max_length=a.max_input).to("cuda")
        q=projection(encoder(enc["input_ids"]).mean(1).to(dt))
        with torch.no_grad():
            p=embedder.encode([x["summary"] for x in batch],convert_to_tensor=True,device="cuda").to(dt); n=embedder.encode(neg,convert_to_tensor=True,device="cuda").to(dt)
        q,p,n=[nn.functional.normalize(x,dim=-1) for x in (q,p,n)]
        sp=torch.cosine_similarity(q,p)/a.tau; sn=torch.cosine_similarity(q,n)/a.tau
        loss=-(torch.log(torch.exp(sp)/(torch.exp(sp)+torch.exp(sn)))).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(projection.parameters(),1.0); nn.utils.clip_grad_norm_(encoder.parameters(),1.0); opt.step()
        hist.append(dict(epoch=ep,step=len(hist),loss=float(loss)))
    print(f"epoch {ep+1} mean loss {np.mean([h['loss'] for h in hist if h['epoch']==ep]):.4f}",flush=True)
delta=float((encoder.weight.detach().float().cpu()-AutoModelForCausalLM.from_pretrained(a.ckpt,dtype=torch.bfloat16,device_map={"":"cpu"}).get_input_embeddings().weight.float()).abs().mean())
model.save_pretrained(a.out,safe_serialization=True); tok.save_pretrained(a.out); torch.save(projection.state_dict(),os.path.join(a.out,"projection.pt"))
json.dump(dict(base_ckpt=a.ckpt,dataset=a.dataset,seed=a.seed,epochs=a.epochs,batch=a.bs,lr=a.lr,tau=a.tau,n_train=len(train),steps=len(hist),
    stage_seconds=round(time.time()-t1,1),peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated()/2**30,2),mean_abs_embedding_change=delta,
    loss_first=hist[0]["loss"],loss_last=hist[-1]["loss"],history=hist,train_row_ids=man["train_row_ids"]),open(os.path.join(a.out,"manifest.json"),"w"),indent=1)
print("done",a.out,f"{time.time()-t0:.0f}s embed-delta={delta:.3e}",flush=True)
