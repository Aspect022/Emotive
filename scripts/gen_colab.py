import json, textwrap
from pathlib import Path

OUT = Path(r"D:\Projects\Major-Project\CogProfile_Net_v2_Colab.ipynb")

def c(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[src]}
def m(src): return {"cell_type":"markdown","metadata":{},"source":[src]}

cells = []

cells.append(m("""# 🧠 CogProfile-Net v2 — EEG Cognitive Profiler (Google Colab)
**CWT Scalogram Stacking · Riemannian SPD · Deep Embedded Clustering · Evidential DL · Digital Twin**

---

### ✅ Before running — 3 steps:
1. **Runtime → Change runtime type → T4 GPU** (free)
2. Upload 3 files via the 📁 Files panel on the left → Upload to session storage:
   - `X_raw.npy` (~83 MB) from your local `data/processed/dl_dataset/`
   - `y_task.npy` — same folder
   - `participant_ids.npy` — same folder
3. **Runtime → Run all**

*Estimated total time on T4 GPU: ~35 minutes*
"""))

cells.append(m("## 📦 Step 1 — Install & Imports"))
cells.append(c("""%%capture
!pip install pyriemann netcal --quiet
print("Dependencies installed")
"""))

cells.append(c("""import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("Warning: No GPU — go to Runtime > Change runtime type > T4 GPU")
"""))

cells.append(m("## 📂 Step 2 — Load Dataset"))
cells.append(c("""DATA = Path("/content")
X_raw = np.load(DATA / "X_raw.npy")
y     = np.load(DATA / "y_task.npy")
pids  = np.load(DATA / "participant_ids.npy", allow_pickle=True)
print(f"X_raw: {X_raw.shape}  y: {y.shape}  subjects: {len(np.unique(pids))}")
cls_counts = dict(zip(*np.unique(y, return_counts=True)))
print(f"Class distribution: {cls_counts}")
"""))

cells.append(m("## ⚡ Step 3 — Sub-window + CWT Scalogram Precomputation\n*~30 seconds on GPU*"))
cells.append(c("""# Config
SFREQ=128; WIN=128; STEP=32; N_FREQ=64; PAD=32; T_PAD=WIN+2*PAD; W0=5.0
FREQ_MIN=1.0; FREQ_MAX=45.0

def make_subwindows(X, y, pids):
    wins,labs,tids,sids=[],[],[],[]
    for i,(trial,lab,pid) in enumerate(zip(X,y,pids)):
        for s in range(0, trial.shape[1]-WIN+1, STEP):
            wins.append(trial[:,s:s+WIN]); labs.append(lab)
            tids.append(i); sids.append(pid)
    return (np.stack(wins,0).astype(np.float32), np.array(labs,np.int64),
            np.array(tids,np.int64), np.array(sids))

t0=time.time()
X_win,y_win,trial_ids,subj_ids = make_subwindows(X_raw,y,pids)
N=len(X_win)
print(f"Sub-windows: {N}  shape={X_win.shape}  elapsed={time.time()-t0:.1f}s")

def build_filters():
    freqs=np.logspace(np.log10(FREQ_MIN),np.log10(FREQ_MAX),N_FREQ)
    filters=[]
    for f in freqs:
        s=W0/(2.0*np.pi*f); M=min(int(6.0*s*SFREQ)|1, T_PAD)
        if M<3: M=3
        t=(np.arange(M)-(M-1)/2.0)/SFREQ; ts=t/s
        psi=(np.pi**-0.25)*np.exp(1j*W0*ts)*np.exp(-0.5*ts**2)/np.sqrt(s)
        if len(psi)<T_PAD:
            pl=(T_PAD-len(psi))//2; pr=T_PAD-len(psi)-pl; psi=np.pad(psi,(pl,pr))
        else:
            st=(len(psi)-T_PAD)//2; psi=psi[st:st+T_PAD]
        filters.append(psi)
    ft=torch.tensor(np.array(filters),dtype=torch.complex64).to(DEVICE)
    return torch.fft.fft(ft,dim=-1)

filters_fft=build_filters()
print(f"Filter bank shape: {filters_fft.shape}")

def compute_all_scalograms(X_win, filters_fft, batch=512):
    N=len(X_win); out=np.empty((N,14,N_FREQ,WIN),dtype=np.float16)
    t0b=time.time()
    for i in range(0,N,batch):
        b=torch.tensor(X_win[i:i+batch],dtype=torch.float32,device=DEVICE)
        padded=F.pad(b,(PAD,PAD),mode="reflect")
        sig_fft=torch.fft.fft(padded,dim=-1)
        prod=sig_fft.unsqueeze(2)*filters_fft.unsqueeze(0).unsqueeze(0).conj()
        conv=torch.fft.ifft(prod,dim=-1)
        power=conv[:,:,:,PAD:PAD+WIN].abs().pow(2)
        power=torch.log1p(power)
        mn=power.amin(-1,keepdim=True); mx=power.amax(-1,keepdim=True)
        power=(power-mn)/torch.clamp(mx-mn,min=1e-8)
        out[i:i+len(b)]=power.to(torch.float16).cpu().numpy()
        if i%(batch*5)==0:
            pct=100*(i+len(b))/N; el=time.time()-t0b
            eta=el/max(i+len(b),1)*(N-i-len(b))
            print(f"  [{pct:5.1f}%] {i+len(b)}/{N}  ETA={eta:.0f}s")
    return out

print("Computing CWT scalograms on GPU...")
SCALOGRAMS=compute_all_scalograms(X_win,filters_fft)
print(f"Done! Shape: {SCALOGRAMS.shape}  total={time.time()-t0:.1f}s")
"""))

cells.append(m("## ✂️ Step 4 — Train/Val/Test Splits"))
cells.append(c("""def tier2_split(trial_ids, y_window, seed=42):
    unique_t=np.unique(trial_ids)
    y_t=np.array([y_window[trial_ids==t][0] for t in unique_t])
    sss=StratifiedShuffleSplit(1,test_size=0.1,random_state=seed)
    trv,te=next(sss.split(unique_t,y_t))
    sss2=StratifiedShuffleSplit(1,test_size=0.111,random_state=seed)
    tr,val=next(sss2.split(unique_t[trv],y_t[trv]))
    def wins_of(ts): return np.where(np.isin(trial_ids,unique_t[ts]))[0]
    return {"train_idx":wins_of(trv[tr]), "val_idx":wins_of(trv[val]), "test_idx":wins_of(te)}

def tier3_split(subj_ids, y_window, n_test=5, seed=42):
    gss=GroupShuffleSplit(1,test_size=n_test/len(np.unique(subj_ids)),random_state=seed)
    tr_i,te_i=next(gss.split(np.arange(len(y_window)),y_window,groups=subj_ids))
    np.random.seed(seed)
    val_i=np.random.choice(tr_i,int(0.15*len(tr_i)),replace=False)
    tr_i=np.setdiff1d(tr_i,val_i)
    return {"train_idx":tr_i, "val_idx":val_i, "test_idx":te_i}

SPLITS = {"tier2":tier2_split(trial_ids,y_win), "tier3":tier3_split(subj_ids,y_win)}
sp=SPLITS["tier2"]
print(f"Tier2 split > Train: {len(sp['train_idx'])}  Val: {len(sp['val_idx'])}  Test: {len(sp['test_idx'])}")
"""))

cells.append(m("## 🏗️ Step 5 — CogProfile-Net v2 Model"))
cells.append(c("""# ── Branch A: CWT Scalogram CNN ───────────────────────────────
class ScalogramCNN(nn.Module):
    def __init__(self, C=14, F=64, T=128, emb=128, do=0.4):
        super().__init__()
        self.t=nn.Sequential(
            nn.Conv2d(C,32,(1,8),(1,1),(0,4),bias=False),nn.BatchNorm2d(32),nn.ELU())
        self.f=nn.Sequential(
            nn.Conv2d(32,64,(8,1),(1,1),(4,0),groups=32,bias=False),nn.BatchNorm2d(64),
            nn.ELU(),nn.AvgPool2d((2,4)),nn.Dropout(do*0.5))
        self.s=nn.Sequential(
            nn.Conv2d(64,128,(4,4),(1,1),(2,2),bias=False),nn.BatchNorm2d(128),
            nn.ELU(),nn.AvgPool2d((4,4)),nn.Dropout(do))
        with torch.no_grad():
            d=torch.zeros(1,C,F,T); flat=self.s(self.f(self.t(d))).flatten(1).shape[1]
        self.p=nn.Sequential(nn.Flatten(),nn.Linear(flat,emb),nn.LayerNorm(emb))
    def forward(self,x): return self.p(self.s(self.f(self.t(x))))

# ── Branch B: Riemannian SPD Encoder ──────────────────────────
class CovLayer(nn.Module):
    def forward(self,x):
        x=x-x.mean(-1,keepdim=True)
        return torch.bmm(x,x.transpose(1,2))/(x.shape[-1]-1)+torch.eye(x.shape[1],device=x.device)*1e-5

class BiMap(nn.Module):
    def __init__(self,i,o):
        super().__init__(); W=torch.empty(i,o); nn.init.orthogonal_(W); self.W=nn.Parameter(W)
    def forward(self,X): return self.W.T@X@self.W

class ReEig(nn.Module):
    def forward(self,X): L,V=torch.linalg.eigh(X); return V@torch.diag_embed(L.clamp(1e-4))@V.mT

class LogEig(nn.Module):
    def forward(self,X): L,V=torch.linalg.eigh(X); return V@torch.diag_embed(L.clamp(1e-6).log())@V.mT

class RiemannianSPD(nn.Module):
    def __init__(self,C=14,emb=64):
        super().__init__()
        self.cov=CovLayer(); self.b1=BiMap(C,8); self.r1=ReEig()
        self.b2=BiMap(8,6); self.r2=ReEig(); self.log=LogEig()
        self.p=nn.Sequential(nn.Linear(36,emb),nn.LayerNorm(emb))
    def forward(self,x):
        S=self.log(self.r2(self.b2(self.r1(self.b1(self.cov(x))))))
        return self.p(S.flatten(1))

# ── Evidential Deep Learning Head ─────────────────────────────
class EDLHead(nn.Module):
    def __init__(self,d=128,K=5):
        super().__init__(); self.K=K; self.fc=nn.Linear(d,K)
    def forward(self,z):
        e=F.softplus(self.fc(z)); a=e+1; S=a.sum(1,keepdim=True)
        return {"e":e,"a":a,"prob":a/S,"u":self.K/S,"S":S}

def edl_loss(o,y,ep,K=5):
    a=o["a"]; S=o["S"]; yoh=F.one_hot(y,K).float()
    mse=((yoh-a/S).pow(2)+a*(S-a)/(S.pow(2)*(S+1))).sum(1).mean()
    ka=(yoh+(1-yoh)*a)
    lc=torch.lgamma(ka.sum(1))-torch.lgamma(ka).sum(1)
    kl=(lc+((ka-1)*(torch.digamma(ka)-torch.digamma(ka.sum(1,keepdim=True)))).sum(1)).mean()
    return mse+min(1.0,ep/10.0)*kl

# ── Deep Embedded Clustering ───────────────────────────────────
class DEC(nn.Module):
    def __init__(self,K=8,d=128):
        super().__init__(); self.mu=nn.Parameter(torch.randn(K,d))
    @torch.no_grad()
    def init_km(self,embs,seed=42):
        from sklearn.cluster import KMeans
        km=KMeans(n_clusters=self.mu.shape[0],n_init=10,random_state=seed).fit(embs)
        self.mu.data=torch.tensor(km.cluster_centers_,dtype=torch.float32,device=self.mu.device)
        print(f"  DEC centroids initialized (inertia={km.inertia_:.1f})")
    def forward(self,z):
        d2=((z.unsqueeze(1)-self.mu.unsqueeze(0))**2).sum(-1)
        q=(1+d2).pow(-1); return q/q.sum(1,keepdim=True)

def dec_kl(q):
    f=q.sum(0,keepdim=True); p=(q**2/f); p=(p/p.sum(1,keepdim=True)).detach()
    return F.kl_div(q.log(),p,reduction="batchmean")

# ── Full CogProfile-Net v2 ─────────────────────────────────────
class CogProfileNet(nn.Module):
    def __init__(self,K=5,emb=128,n_clust=8,do=0.4):
        super().__init__(); self.K=K
        self.a=ScalogramCNN(emb=emb,do=do)
        self.b=RiemannianSPD(emb=emb//2)
        self.pa=nn.Linear(emb,emb); self.pb=nn.Linear(emb//2,emb)
        self.gate=nn.Sequential(nn.Linear(emb*2,emb),nn.Sigmoid())
        self.ln=nn.LayerNorm(emb)
        self.bmlp=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,emb),nn.Sigmoid())
        self.aux=nn.Linear(emb,K)
        self.dec=DEC(n_clust,emb); self.edl=EDLHead(emb,K)

    def forward(self,sc,rw,bh):
        za=self.a(sc); zb=self.b(rw)
        a=self.pa(za); b_=self.pb(zb)
        g=self.gate(torch.cat([a,b_],1))
        z=self.ln(g*a+(1-g)*b_)
        z=z+self.bmlp(bh)*z
        return {"z":z,"za":za,"zb":zb,"q":self.dec(z),"edl":self.edl(z),"aux":self.aux(z)}

    def compute_loss(self,o,y,ep,ld=0.1,lo=0.05,la=0.3):
        el=edl_loss(o["edl"],y,ep); dl=dec_kl(o["q"])
        d=min(o["za"].shape[1],o["zb"].shape[1])
        ol=(F.normalize(o["za"][:,:d],1)*F.normalize(o["zb"][:,:d],1)).sum(1).pow(2).mean()
        al=F.cross_entropy(o["aux"],y)
        return el+ld*dl+lo*ol+la*al

model=CogProfileNet().to(DEVICE)
n_p=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"CogProfile-Net v2: {n_p:,} trainable parameters")
with torch.no_grad():
    sc=torch.randn(4,14,64,128).to(DEVICE); rw=torch.randn(4,14,128).to(DEVICE)
    bh=torch.randn(4,3).to(DEVICE); o=model(sc,rw,bh)
    print(f"z:{o['z'].shape}  prob:{o['edl']['prob'].shape}  q:{o['q'].shape}")
print("Model architecture OK")
"""))

cells.append(m("## 📦 Step 6 — Dataset & DataLoaders"))
cells.append(c("""class EEGDataset(Dataset):
    def __init__(self,S,X,y,idx,aug=False):
        self.S=S; self.X=X; self.y=y; self.idx=idx; self.aug=aug
        self.beh=np.array([0.5,0.5,0.5],dtype=np.float32)
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        r=self.idx[i]
        sc=np.array(self.S[r],dtype=np.float32)
        rw=np.array(self.X[r],dtype=np.float32)
        if self.aug and np.random.rand()<0.3:
            rw=rw[:,::-1].copy(); sc=sc[:,:,::-1].copy()
        return (torch.from_numpy(sc), torch.from_numpy(rw),
                torch.from_numpy(self.beh), torch.tensor(self.y[r],dtype=torch.long))

BATCH=256; sp=SPLITS["tier2"]
train_loader=DataLoader(EEGDataset(SCALOGRAMS,X_win,y_win,sp["train_idx"],True), BATCH,True, num_workers=2,pin_memory=True)
val_loader  =DataLoader(EEGDataset(SCALOGRAMS,X_win,y_win,sp["val_idx"]),        BATCH,False,num_workers=2,pin_memory=True)
test_loader =DataLoader(EEGDataset(SCALOGRAMS,X_win,y_win,sp["test_idx"]),       BATCH,False,num_workers=2,pin_memory=True)
print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")
"""))

cells.append(m("## 🏋️ Step 7 — Three-Phase Training\n*Phase 1: 15 epochs encoder pre-train | Phase 2: K-Means DEC init | Phase 3: 60 epochs joint*"))
cells.append(c("""@torch.no_grad()
def evaluate(model,loader):
    model.eval(); ps,ls,us=[],[],[]
    for sc,rw,bh,lb in loader:
        sc,rw,bh=sc.to(DEVICE),rw.to(DEVICE),bh.to(DEVICE)
        o=model(sc,rw,bh)
        ps.extend(o["edl"]["prob"].argmax(1).cpu().numpy())
        ls.extend(lb.numpy())
        us.extend(o["edl"]["u"].squeeze(1).cpu().numpy())
    ps,ls=np.array(ps),np.array(ls)
    acc=(ps==ls).mean()
    f1s=[]
    for k in range(5):
        tp=((ps==k)&(ls==k)).sum(); fp=((ps==k)&(ls!=k)).sum(); fn=((ps!=k)&(ls==k)).sum()
        p=tp/(tp+fp+1e-8); r=tp/(tp+fn+1e-8); f1s.append(2*p*r/(p+r+1e-8))
    return float(acc), float(np.mean(f1s)), float(np.array(us).mean())

opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,75)
CKPT="/content/cogprofile_best.pt"
best_acc=0.0; hist=[]; T0=time.time(); pat=0

print("="*60)
print("PHASE 1: Pre-train encoder (no DEC, 15 epochs)")
for ep in range(1,16):
    model.train()
    for sc,rw,bh,lb in train_loader:
        sc,rw,bh,lb=sc.to(DEVICE),rw.to(DEVICE),bh.to(DEVICE),lb.to(DEVICE)
        opt.zero_grad()
        o=model(sc,rw,bh)
        (F.cross_entropy(o["edl"]["prob"],lb)+0.3*F.cross_entropy(o["aux"],lb)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    sched.step()
    if ep%5==0:
        va,vf,vu=evaluate(model,val_loader)
        print(f"  Ep {ep:3d} | val_acc={va:.4f} | macro_f1={vf:.4f}")

print()
print("PHASE 2a: Initialising DEC centroids via K-Means...")
model.eval(); embs=[]
with torch.no_grad():
    for sc,rw,bh,_ in train_loader:
        sc,rw,bh=sc.to(DEVICE),rw.to(DEVICE),bh.to(DEVICE)
        embs.append(model(sc,rw,bh)["z"].cpu().numpy())
model.dec.init_km(np.concatenate(embs))

print()
print("PHASE 2b: Joint DEC + EDL training (60 epochs)")
for ep in range(1,61):
    model.train(); tloss=0.0; nb=0
    for sc,rw,bh,lb in train_loader:
        sc,rw,bh,lb=sc.to(DEVICE),rw.to(DEVICE),bh.to(DEVICE),lb.to(DEVICE)
        opt.zero_grad()
        o=model(sc,rw,bh); loss=model.compute_loss(o,lb,ep)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tloss+=loss.item(); nb+=1
    sched.step()
    if ep%5==0 or ep==1:
        va,vf,vu=evaluate(model,val_loader)
        el=time.time()-T0
        print(f"  Ep {ep:3d} | loss={tloss/nb:.4f} | val_acc={va:.4f} | f1={vf:.4f} | u={vu:.3f} | {el:.0f}s")
        hist.append({"epoch":ep,"acc":va,"f1":vf})
        if va>best_acc:
            best_acc=va; pat=0; torch.save(model.state_dict(),CKPT)
            print(f"    ✓ New best saved ({best_acc:.4f})")
        else:
            pat+=1
            if pat>=3: print(f"  Early stopping at epoch {ep}"); break

print()
print(f"Training complete in {(time.time()-T0)/60:.1f} min")
print(f"Best val accuracy: {best_acc:.4f}")
"""))

cells.append(m("## 📊 Step 8 — Final Evaluation"))
cells.append(c("""model.load_state_dict(torch.load(CKPT,map_location=DEVICE,weights_only=True))
NAMES=["Mental Arithmetic","Pattern Recognition","Working Memory","Reading Comprehension","Sustained Attention"]

print("="*60)
print("FINAL RESULTS ACROSS ALL TIERS")
print("="*60)
for tier,sp_t in SPLITS.items():
    ds=EEGDataset(SCALOGRAMS,X_win,y_win,sp_t["test_idx"])
    lo=DataLoader(ds,256,num_workers=2,pin_memory=True)
    a,f,u=evaluate(model,lo)
    print(f"  [{tier.upper():6s}] Accuracy={a:.4f}  Macro-F1={f:.4f}  Uncertainty={u:.3f}")

all_p,all_l=[],[]
with torch.no_grad():
    for sc,rw,bh,lb in test_loader:
        sc,rw,bh=sc.to(DEVICE),rw.to(DEVICE),bh.to(DEVICE)
        o=model(sc,rw,bh)
        all_p.extend(o["edl"]["prob"].argmax(1).cpu().numpy())
        all_l.extend(lb.numpy())

print()
print("Tier-2 Classification Report (test set):")
print(classification_report(np.array(all_l),np.array(all_p),target_names=NAMES))
"""))

cells.append(m("## 📈 Step 9 — Visualizations"))
cells.append(c("""from sklearn.metrics import confusion_matrix

fig,axes=plt.subplots(1,3,figsize=(20,5))
fig.suptitle("CogProfile-Net v2 — Results",fontsize=14,fontweight="bold")

# 1. Confusion Matrix
cm=confusion_matrix(np.array(all_l),np.array(all_p),normalize="true")
sns.heatmap(cm,annot=True,fmt=".2f",cmap="Blues",
            xticklabels=[n[:12] for n in NAMES],yticklabels=[n[:12] for n in NAMES],ax=axes[0])
axes[0].set_title("Confusion Matrix (Tier-2 Test)"); axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")
plt.setp(axes[0].get_xticklabels(),rotation=30,ha="right",fontsize=8)

# 2. Training Curve
if hist:
    accs=[h["acc"] for h in hist]; eps=[h["epoch"] for h in hist]
    axes[1].plot(eps,accs,"o-",color="#0f766e",label="CogProfileNet v2",linewidth=2)
    axes[1].axhline(0.7334,color="#ef4444",ls="--",linewidth=1.5,label="LightGBM baseline 73.34%")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title("Validation Accuracy Curve")

# 3. Digital Twin Trajectory
np.random.seed(42); N_S=60
profile=np.array([0.40,0.25,0.15,0.12,0.08])
traj=[]
for s in range(N_S):
    p=profile.copy(); p[0]+=0.007*s; p[1]+=0.003*s; p[2]+=0.005*s
    p[3]-=0.002*s; p[4]-=0.002*s
    p=np.clip(p,0.05,None); p/=p.sum()
    traj.append(np.random.dirichlet(p*30+0.1))
traj=np.array(traj)
colors=["#0f766e","#3b82f6","#db2777","#f59e0b","#8b5cf6"]
for k,(name,col) in enumerate(zip(NAMES,colors)):
    axes[2].plot(range(1,N_S+1),traj[:,k],label=name[:18],color=col,alpha=0.8,linewidth=1.5)
axes[2].set_xlabel("Session"); axes[2].set_ylabel("Cognitive Profile Weight")
axes[2].set_title("Digital Twin: Cognitive Trajectory Simulation"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/content/cogprofile_results.png",dpi=150,bbox_inches="tight")
plt.show()
print("Figure saved to /content/cogprofile_results.png")
"""))

cells.append(m("## 💾 Step 10 — Save to Google Drive"))
cells.append(c("""# Uncomment to save trained model and results to Google Drive
# from google.colab import drive
# drive.mount("/content/drive")
# import shutil
# shutil.copy("/content/cogprofile_best.pt", "/content/drive/MyDrive/CogProfile_Net_v2.pt")
# shutil.copy("/content/cogprofile_results.png", "/content/drive/MyDrive/CogProfile_results.png")
# print("Saved to Google Drive")

results={"best_val_acc":best_acc,"tier2_test_acc":float((np.array(all_p)==np.array(all_l)).mean()),
         "n_params":int(sum(p.numel() for p in model.parameters()))}
with open("/content/cogprofile_results.json","w") as f: json.dump(results,f,indent=2)
print("Results summary:")
print(json.dumps(results,indent=2))
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
        "colab": {"provenance":[]},
        "accelerator": "GPU",
        "gpuClass": "standard"
    },
    "cells": cells
}

OUT.parent.mkdir(parents=True,exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f:
    json.dump(nb,f,indent=1,ensure_ascii=False)
print(f"Written: {OUT}")
print(f"Cells: {len(cells)}")
size = OUT.stat().st_size
print(f"Size: {size/1024:.0f} KB")
