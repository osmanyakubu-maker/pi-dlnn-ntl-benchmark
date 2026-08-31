from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ROOT = Path("experiment_outputs")
OUT = ROOT / "final_analysis"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "run_metrics.csv")
main = ["LSTM", "Attention-LSTM", "CNN-LSTM", "CNN-TCN", "Transformer", "PI-DLNN"]
metrics = ["mcc", "auroc", "average_precision", "f1", "balanced_accuracy", "brier", "ece", "training_seconds"]

# Fully machine-readable summary.
summary = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std", "median", "min", "max"])
summary.to_csv(OUT / "descriptive_summary.csv")

def bootstrap_ci(x, seed=20260826, n=10000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.quantile(b, [.025, .975])

# Prespecified paired comparisons: PI-DLNN vs each main baseline, plus ablations.
rows = []
comparators = main[:-1] + ["PI-DLNN-NoPhysics", "Smoothness-LSTM", "PI-DLNN-Shuffled"]
for ds in ["SGCC", "UCI"]:
    for comp in comparators:
        for metric in ["mcc", "auroc", "f1", "balanced_accuracy", "brier"]:
            p = df[(df.dataset == ds) & df.model.isin(["PI-DLNN", comp])].pivot(index="seed", columns="model", values=metric).dropna()
            delta = p["PI-DLNN"].to_numpy() - p[comp].to_numpy()
            # Positive is favorable for every metric.
            if metric == "brier": delta = -delta
            lo, hi = bootstrap_ci(delta)
            if np.allclose(delta, 0): stat, pv = 0.0, 1.0
            else:
                stat, pv = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
            rows.append({"dataset":ds,"metric":metric,"comparator":comp,"mean_favorable_delta":delta.mean(),"ci95_low":lo,"ci95_high":hi,"wilcoxon_two_sided_p":pv,"n_seeds":len(delta)})
paired = pd.DataFrame(rows)
paired.to_csv(OUT / "paired_tests.csv", index=False)

# Per-attack sensitivity for UCI, computed within each seed and then summarized.
attack_rows=[]
for model in main:
    for seed in sorted(df.seed.unique()):
        pth=ROOT/"UCI"/model/f"seed_{seed}"/"predictions.csv.gz"
        p=pd.read_csv(pth)
        for attack in sorted(set(p.attack_type)-{"normal"}):
            q=p[p.attack_type.isin(["normal",attack])]
            attack_rows.append({"model":model,"seed":seed,"attack":attack,"mcc":matthews_corrcoef(q.y_true,q.y_pred),"auroc":roc_auc_score(q.y_true,q.probability),"attack_recall":q.loc[q.attack_type==attack,"y_pred"].mean()})
att=pd.DataFrame(attack_rows)
att.to_csv(OUT/"uci_attack_seed_metrics.csv",index=False)
att.groupby(["model","attack"])[["mcc","auroc","attack_recall"]].agg(["mean","std"]).to_csv(OUT/"uci_attack_summary.csv")

# Publication figures.
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
colors={"PI-DLNN":"#b24c3d","Transformer":"#245a7a","CNN-TCN":"#5f7f67","CNN-LSTM":"#7c6a9b","Attention-LSTM":"#b08b3e","LSTM":"#737373"}
fig,axs=plt.subplots(2,2,figsize=(8.2,6.5),sharex=True)
for r,ds in enumerate(["SGCC","UCI"]):
    for c,met in enumerate(["mcc","auroc"]):
        ax=axs[r,c]
        vals=[]
        for i,m in enumerate(main):
            x=df[(df.dataset==ds)&(df.model==m)][met].to_numpy()
            vals.append(x)
            ax.scatter(np.full(len(x),i)+np.linspace(-.07,.07,len(x)),x,s=14,color=colors[m],alpha=.65,zorder=2)
            ax.errorbar(i,x.mean(),yerr=x.std(ddof=1),fmt='o',ms=6,color='black',capsize=3,zorder=3)
        ds_label="UCI-ELD" if ds=="UCI" else ds
        ax.set_title(f"{ds_label}: {met.upper() if met=='mcc' else 'AUROC'}")
        ax.grid(axis='y',alpha=.2)
        if c==0: ax.set_ylabel("Score")
        ax.set_xticks(range(len(main)),["LSTM","Attention-LSTM","CNN-LSTM","CNN-TCN","Transformer","PI-DLNN"],rotation=35,ha='right')
fig.suptitle("Ten-seed test performance (points) with mean ± SD",fontweight='bold')
fig.tight_layout()
fig.savefig(OUT/"performance_across_seeds.png",dpi=300,bbox_inches='tight'); plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(8.2,3.5),sharey=True)
for ax,ds in zip(axs,["SGCC","UCI"]):
    p=df[(df.dataset==ds)&df.model.isin(["PI-DLNN","PI-DLNN-NoPhysics","PI-DLNN-Shuffled"])].pivot(index='seed',columns='model',values='mcc')
    for _,row in p.iterrows():
        ax.plot([0,1,2],row[["PI-DLNN-NoPhysics","PI-DLNN","PI-DLNN-Shuffled"]],color='#9a9a9a',alpha=.55,lw=.8)
    ax.scatter(np.tile([0,1,2],len(p)),p[["PI-DLNN-NoPhysics","PI-DLNN","PI-DLNN-Shuffled"]].to_numpy().ravel(),s=18,color=['#245a7a','#b24c3d','#8a6f42']*len(p),zorder=2)
    ax.set_xticks([0,1,2],["No physics","PI-DLNN","Shuffled"],rotation=20)
    ax.set_title("UCI-ELD" if ds=="UCI" else ds); ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel("MCC")
fig.suptitle("Physics-ablation result by matched seed",fontweight='bold')
fig.tight_layout(); fig.savefig(OUT/"physics_ablation.png",dpi=300,bbox_inches='tight'); plt.close(fig)

print(summary.loc[(slice(None), main), [("mcc","mean"),("mcc","std"),("auroc","mean"),("auroc","std")]].round(4))
print("\nKey paired tests:\n", paired[(paired.metric.isin(['mcc','auroc'])) & paired.comparator.isin(['Transformer','PI-DLNN-NoPhysics'])].round(5).to_string(index=False))
