#!/usr/bin/env python3
"""Ten-seed, consumer-grouped PI-DLNN reproduction study.

Data:
  * SGCC: real consumer-level theft labels, 26 biweekly bins from the last year.
  * UCI Electricity Load Diagrams: 13 weekly windows/client with six deterministic
    synthetic attacks, downsampled to 42 four-hour bins.

Every run saves raw test probabilities, validation-locked thresholds, split
manifests, training logs, checkpoint hashes, timing, and classification metrics.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 149, 173]
MAIN = ["LSTM", "Attention-LSTM", "CNN-LSTM", "CNN-TCN", "Transformer", "PI-DLNN"]
ABLATIONS = ["PI-DLNN-NoPhysics", "Smoothness-LSTM", "PI-DLNN-Shuffled"]

@dataclass
class Config:
    dataset: str
    model: str
    seed: int
    epochs: int = 20
    patience: int = 4
    batch: int = 512
    hidden: int = 24
    lr: float = 1e-3
    weight_decay: float = 1e-4
    physics_lambda: float = 1e-4
    smoothness_lambda: float = 1e-3

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_sgcc(path, cache):
    if cache.exists():
        d=np.load(cache,allow_pickle=True); return d["x"],d["y"],d["g"],d["attack"]
    df=pd.read_csv(path,low_memory=False)
    cols=sorted([c for c in df if c not in ("CONS_NO","FLAG")],
                key=lambda c: pd.to_datetime(c,format="%Y/%m/%d"))[-364:]
    v=df[cols].to_numpy(np.float32); v[v<0]=np.nan
    med=np.nanmedian(v,axis=1).astype(np.float32); med[~np.isfinite(med)]=0
    bad=~np.isfinite(v); v[bad]=np.take(med,np.nonzero(bad)[0])
    x=np.log1p(np.clip(v.reshape(len(v),26,14).mean(2),0,None)).astype(np.float32)
    y=df.FLAG.to_numpy(np.int64); g=df.CONS_NO.astype(str).to_numpy()
    attack=np.where(y==1,"utility_labelled_theft","normal")
    np.savez_compressed(cache,x=x,y=y,g=g,attack=attack); return x,y,g,attack

def attack_window(w,rng,k):
    a=w.copy()
    if k==0: a*=rng.uniform(.15,.75)
    elif k==1:
        n=int(rng.integers(21,57)); s=int(rng.integers(0,len(a)-n+1)); a[s:s+n]=0
    elif k==2: a=a[::-1].copy()
    elif k==3: a=np.clip(a+rng.normal(0,max(float(a.std()),1e-3)*rng.uniform(.25,.65),len(a)),0,None)
    elif k==4: a=np.minimum(a,np.quantile(a,rng.uniform(.45,.70)))
    else:
        a=a.reshape(7,24); a[:]=a.mean(1,keepdims=True); a=a.ravel()
    return a.astype(np.float32)

def load_uci(path,cache):
    if cache.exists():
        d=np.load(cache,allow_pickle=True); return d["x"],d["y"],d["g"],d["attack"]
    with open(path,encoding="utf-8",errors="replace") as f:
        names=next(csv.reader([f.readline()],delimiter=";"))
    with open(path,"rb") as f: lines=sum(1 for _ in f)
    n=13*7*96
    df=pd.read_csv(path,sep=";",decimal=",",header=None,names=names,skiprows=lines-n,low_memory=False)
    raw=df.iloc[:,1:].to_numpy(np.float32)
    weekly=raw.reshape(13*7*24,4,len(names)-1).mean(1).T.reshape(-1,13,168)
    clients=np.asarray(names[1:]); keep=weekly.mean((1,2))>0; weekly=weekly[keep]; clients=clients[keep]
    normal=weekly.reshape(-1,168); groups=np.repeat(clients,13)
    rng=np.random.default_rng(20260826); kinds=np.arange(len(normal))%6; rng.shuffle(kinds)
    attacked=np.stack([attack_window(w,rng,int(k)) for w,k in zip(normal,kinds)])
    normal=normal.reshape(-1,42,4).mean(2); attacked=attacked.reshape(-1,42,4).mean(2)
    x=np.log1p(np.clip(np.concatenate([normal,attacked]),0,None)).astype(np.float32)
    y=np.r_[np.zeros(len(normal),np.int64),np.ones(len(attacked),np.int64)]
    g=np.r_[groups,groups]
    labels=np.array(["scaled_reduction","partial_zeroing","reversal","noise","peak_clipping","daily_flattening"])
    attack=np.r_[np.full(len(normal),"normal",object),labels[kinds]]
    np.savez_compressed(cache,x=x,y=y,g=g,attack=attack); return x,y,g,attack

def split(y,g,seed,dataset):
    if dataset=="SGCC":
        s=StratifiedShuffleSplit(1,test_size=.3,random_state=seed); tr,rest=next(s.split(y,y))
        s=StratifiedShuffleSplit(1,test_size=.5,random_state=seed+1000); va,te=next(s.split(y[rest],y[rest]))
        return tr,rest[va],rest[te]
    u=np.unique(g); np.random.default_rng(seed).shuffle(u); a=round(.7*len(u)); b=a+round(.15*len(u))
    return np.flatnonzero(np.isin(g,u[:a])),np.flatnonzero(np.isin(g,u[a:b])),np.flatnonzero(np.isin(g,u[b:]))

def preprocess(x,tr):
    hi=float(np.quantile(x[tr],.995)); z=np.minimum(x,hi); mu=float(z[tr].mean()); sd=max(float(z[tr].std()),1e-6)
    return ((z-mu)/sd).astype(np.float32),{"p995":hi,"mean":mu,"std":sd}

class LSTM(nn.Module):
    def __init__(self,h): super().__init__(); self.r=nn.LSTM(1,h,batch_first=True); self.c=nn.Linear(h,1)
    def forward(self,x): z,_=self.r(x); return self.c(z[:,-1]).squeeze(-1),z
class AttLSTM(nn.Module):
    def __init__(self,h): super().__init__(); self.r=nn.LSTM(1,h,batch_first=True); self.a=nn.Linear(h,1); self.c=nn.Linear(h,1)
    def forward(self,x):
        z,_=self.r(x); a=torch.softmax(self.a(z).squeeze(-1),1); p=(z*a.unsqueeze(-1)).sum(1)
        return self.c(p).squeeze(-1),z
class CNNLSTM(nn.Module):
    def __init__(self,h): super().__init__(); self.v=nn.Sequential(nn.Conv1d(1,8,5,padding=2),nn.ReLU(),nn.MaxPool1d(2)); self.r=nn.LSTM(8,h,batch_first=True); self.c=nn.Linear(h,1)
    def forward(self,x): z,_=self.r(self.v(x.transpose(1,2)).transpose(1,2)); return self.c(z[:,-1]).squeeze(-1),z
class TCN(nn.Module):
    def __init__(self,h):
        super().__init__(); self.v=nn.Sequential(nn.Conv1d(1,h,3,padding=1),nn.ReLU(),nn.Conv1d(h,h,3,padding=2,dilation=2),nn.ReLU(),nn.Conv1d(h,h,3,padding=4,dilation=4),nn.ReLU()); self.c=nn.Linear(h,1)
    def forward(self,x): z=self.v(x.transpose(1,2)).transpose(1,2); return self.c(z.mean(1)).squeeze(-1),z
class Trans(nn.Module):
    def __init__(self,T):
        super().__init__(); h=16; self.i=nn.Linear(1,h); self.p=nn.Parameter(torch.zeros(1,T,h)); l=nn.TransformerEncoderLayer(h,4,32,.1,batch_first=True); self.e=nn.TransformerEncoder(l,1); self.c=nn.Linear(h,1)
    def forward(self,x): z=self.e(self.i(x)+self.p[:,:x.shape[1]]); return self.c(z.mean(1)).squeeze(-1),z
class PI(nn.Module):
    def __init__(self,h,shuffled=False):
        super().__init__(); self.r=nn.LSTM(1,h,batch_first=True); self.a=nn.Linear(h,1); self.v=nn.Sequential(nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1)); self.c=nn.Linear(h,1); self.shuffled=shuffled
    def forward(self,x,regularize=False):
        z,_=self.r(x); a=torch.softmax(self.a(z).squeeze(-1),1); logit=self.c((z*a.unsqueeze(-1)).sum(1)).squeeze(-1)
        if not regularize:return logit,z
        q=z[:,torch.randperm(z.shape[1])] if self.shuffled else z
        grad=torch.autograd.grad(self.v(q).sum(),q,create_graph=True)[0]
        acc=q[:,2:]-2*q[:,1:-1]+q[:,:-2]; return logit,z,(acc+grad[:,1:-1]).square().mean()

def make(name,h,T):
    if name=="LSTM":return LSTM(h)
    if name in ("Attention-LSTM","Smoothness-LSTM"):return AttLSTM(h)
    if name=="CNN-LSTM":return CNNLSTM(h)
    if name=="CNN-TCN":return TCN(h)
    if name=="Transformer":return Trans(T)
    if name in ("PI-DLNN","PI-DLNN-NoPhysics"):return PI(h)
    if name=="PI-DLNN-Shuffled":return PI(h,True)
    raise ValueError(name)

def ece(y,p):
    ids=np.clip(np.digitize(p,np.linspace(0,1,11))-1,0,9); out=0
    for i in range(10):
        m=ids==i
        if m.any():out+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)
def threshold(y,p):
    q=np.linspace(.05,.95,181); s=[matthews_corrcoef(y,p>=t) for t in q]; return float(q[int(np.argmax(s))])
def metrics(y,p,t):
    z=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return dict(threshold=t,precision=precision_score(y,z,zero_division=0),recall=recall_score(y,z,zero_division=0),specificity=tn/max(tn+fp,1),f1=f1_score(y,z,zero_division=0),balanced_accuracy=balanced_accuracy_score(y,z),mcc=matthews_corrcoef(y,z),auroc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),brier=brier_score_loss(y,p),ece=ece(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
def predict(model,x,batch):
    model.eval(); out=[]; start=time.perf_counter()
    with torch.no_grad():
        for (a,) in DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(-1)),batch_size=batch): out.append(torch.sigmoid(model(a)[0]).numpy())
    return np.concatenate(out),time.perf_counter()-start

def train(cfg,x,y,g,attack,out):
    seed_all(cfg.seed); tr,va,te=split(y,g,cfg.seed,cfg.dataset); x,prep=preprocess(x,tr); model=make(cfg.model,cfg.hidden,x.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); pos=int(y[tr].sum()); neg=len(tr)-pos; lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/max(pos,1)))
    loader=DataLoader(TensorDataset(torch.from_numpy(x[tr]).unsqueeze(-1),torch.from_numpy(y[tr].astype(np.float32))),batch_size=cfg.batch,shuffle=True,generator=torch.Generator().manual_seed(cfg.seed))
    best=-9; state=None; best_epoch=0; stale=0; hist=[]; start=time.perf_counter()
    for epoch in range(1,cfg.epochs+1):
        model.train(); losses=[]
        for bx,by in loader:
            opt.zero_grad(set_to_none=True)
            if cfg.model in ("PI-DLNN","PI-DLNN-Shuffled"):
                logit,z,reg=model(bx,True); loss=lossfn(logit,by)+cfg.physics_lambda*reg
            else:
                logit,z=model(bx); loss=lossfn(logit,by)
                if cfg.model=="Smoothness-LSTM":loss+=cfg.smoothness_lambda*(z[:,2:]-2*z[:,1:-1]+z[:,:-2]).square().mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(float(loss.detach()))
        pv,_=predict(model,x[va],cfg.batch); t=threshold(y[va],pv); score=matthews_corrcoef(y[va],pv>=t); hist.append((epoch,np.mean(losses),score,t))
        if score>best+1e-5:best=score;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=cfg.patience:break
    train_seconds=time.perf_counter()-start; model.load_state_dict(state); pv,_=predict(model,x[va],cfg.batch); t=threshold(y[va],pv); pt,inf=predict(model,x[te],cfg.batch); m=metrics(y[te],pt,t)
    d=out/cfg.dataset/cfg.model/f"seed_{cfg.seed}";d.mkdir(parents=True,exist_ok=True); ck=d/"checkpoint.pt";torch.save(state,ck); h=sha(ck)
    pd.DataFrame(dict(dataset=cfg.dataset,model=cfg.model,seed=cfg.seed,consumer_id=g[te],sample_index=te,attack_type=attack[te],y_true=y[te],probability=pt,threshold=t,y_pred=(pt>=t).astype(int),checkpoint_sha256=h)).to_csv(d/"predictions.csv.gz",index=False,compression="gzip")
    pd.DataFrame(hist,columns=["epoch","train_loss","validation_mcc","threshold"]).to_csv(d/"history.csv",index=False)
    pd.DataFrame({"consumer_id":np.r_[g[tr],g[va],g[te]],"split":np.r_[np.full(len(tr),"train"),np.full(len(va),"validation"),np.full(len(te),"test")]}).drop_duplicates().to_csv(d/"split_manifest.csv",index=False)
    result={**asdict(cfg),**m,"n_train":len(tr),"n_validation":len(va),"n_test":len(te),"best_epoch":best_epoch,"validation_mcc":best,"training_seconds":train_seconds,"inference_ms_per_sample":inf/len(te)*1000,"parameter_count":sum(q.numel() for q in model.parameters()),"checkpoint_sha256":h,"preprocessing":json.dumps(prep,sort_keys=True)}
    (d/"metrics.json").write_text(json.dumps(result,indent=2));return result

def holm(ps):
    order=np.argsort(ps);out=np.empty(len(ps));run=0
    for rank,i in enumerate(order):run=max(run,(len(ps)-rank)*ps[i]);out[i]=min(run,1)
    return out
def summarize(path,out):
    df=pd.read_csv(path); nums=["precision","recall","specificity","f1","balanced_accuracy","mcc","auroc","average_precision","brier","ece","training_seconds","inference_ms_per_sample","parameter_count"]
    df.groupby(["dataset","model"])[nums].agg(["mean","std","median"]).to_csv(out/"summary_by_model.csv")
    rows=[]; pending=[]
    for ds in sorted(df.dataset.unique()):
        main=df[(df.dataset==ds)&df.model.isin(MAIN)]; comp=main[main.model!="PI-DLNN"].groupby("model").mcc.mean().idxmax()
        if "PI-DLNN" not in set(main.model) or main.model.nunique()<2: continue
        for metric in ["mcc","f1","balanced_accuracy","auroc","average_precision","brier"]:
            p=main.pivot(index="seed",columns="model",values=metric).dropna();diff=p["PI-DLNN"].to_numpy()-p[comp].to_numpy();diff=-diff if metric=="brier" else diff
            rng=np.random.default_rng(20260826);boot=rng.choice(diff,(2000,len(diff)),replace=True).mean(1);lo,hi=np.quantile(boot,[.025,.975]);pv=float(wilcoxon(diff,alternative="greater").pvalue) if np.any(diff) else 1
            pending.append((ds,metric,comp,diff,lo,hi,pv))
    adj=holm([z[-1] for z in pending])
    for z,a in zip(pending,adj):ds,metric,comp,diff,lo,hi,pv=z;rows.append(dict(dataset=ds,metric=metric,comparator=comp,paired_mean_improvement=diff.mean(),ci95_low=lo,ci95_high=hi,wilcoxon_one_sided_p=pv,holm_adjusted_p=a,n_seeds=len(diff)))
    pd.DataFrame(rows).to_csv(out/"paired_comparisons.csv",index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--sgcc",type=Path,required=True);p.add_argument("--uci",type=Path,required=True);p.add_argument("--out",type=Path,default=Path("experiment_outputs"));p.add_argument("--datasets",nargs="+",default=["SGCC","UCI"]);p.add_argument("--models",nargs="+",default=MAIN+ABLATIONS);p.add_argument("--seeds",nargs="+",type=int,default=SEEDS);p.add_argument("--epochs",type=int,default=20);p.add_argument("--patience",type=int,default=4);p.add_argument("--threads",type=int,default=8);p.add_argument("--summary-only",action="store_true");p.add_argument("--force",action="store_true",help="Replace matching completed runs instead of skipping them.");a=p.parse_args();a.out.mkdir(exist_ok=True);torch.set_num_threads(a.threads);mp=a.out/"run_metrics.csv"
    if a.summary_only:summarize(mp,a.out);return
    data={}
    if "SGCC" in a.datasets:data["SGCC"]=load_sgcc(a.sgcc,a.out/"sgcc.npz")
    if "UCI" in a.datasets:data["UCI"]=load_uci(a.uci,a.out/"uci.npz")
    old=pd.read_csv(mp) if mp.exists() else pd.DataFrame()
    if a.force and not old.empty:
        selected=(old.dataset.isin(a.datasets)&old.model.isin(a.models)&old.seed.isin(a.seeds))
        old=old.loc[~selected].copy()
    rows=old.to_dict("records");done=set(zip(old.get("dataset",[]),old.get("model",[]),old.get("seed",[])))
    for ds in a.datasets:
        x,y,g,attack=data[ds]
        for model in a.models:
            for seed in a.seeds:
                if (ds,model,seed) in done:print("SKIP",ds,model,seed,flush=True);continue
                print("START",ds,model,seed,flush=True);r=train(Config(ds,model,seed,epochs=a.epochs,patience=a.patience),x,y,g,attack,a.out);rows.append(r);pd.DataFrame(rows).to_csv(mp,index=False);print(f"DONE {ds} {model} {seed} MCC={r['mcc']:.4f} AUROC={r['auroc']:.4f} SEC={r['training_seconds']:.1f}",flush=True)
    summarize(mp,a.out)
if __name__=="__main__":main()
