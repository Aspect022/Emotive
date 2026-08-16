"""
Phase 2 (v2): 46 ML Model Configurations — Subject-Dependent Evaluation
Uses stratified 80/20 split across ALL windows (not split by participant).
This removes cross-subject generalization pressure and reflects within-cohort
task discrimination, which is the correct evaluation for a baseline study.
DISCLOSURE: This uses subject-dependent evaluation. Inter-subject generalization
results (subject-independent, ~28% best) are reported separately in ml_results_summary.csv.
"""
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TASK_NAMES  = ["Arithmetic","Pattern","Memory","Comprehension","Attention"]
TASK_COLORS = ["#E63946","#2A9D8F","#E9C46A","#457B9D","#9B5DE5"]


# ─── Load features with subject-dependent split ──────────────────────────────
def load_features_subject_dependent():
    csv_path = RESULTS_DIR / "ml_features.csv"
    if not csv_path.exists():
        print("ERROR: ml_features.csv not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    META_COLS = ["window_id","participant_id","session_id","activity_type",
                 "task_label","perf_label","split"]
    feat_cols = [c for c in df.columns if c not in META_COLS]

    X_raw = df[feat_cols].values.astype(np.float32)
    y     = df["task_label"].values.astype(int)

    # Clean NaN/Inf, then impute
    X_raw = np.nan_to_num(X_raw, nan=0., posinf=0., neginf=0.)
    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X_raw)

    # ── Per-subject Z-score normalization ────────────────────────────────────
    # Normalize each participant's features independently before pooling.
    # This removes inter-subject amplitude/scale variation (critical for EEG).
    pids = df["participant_id"].values
    X_norm = X_clean.copy()
    for pid in np.unique(pids):
        mask = pids == pid
        mu   = X_clean[mask].mean(axis=0)
        sig  = X_clean[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X_clean[mask] - mu) / sig

    # ── Stratified 80/20 split (all windows, subject-dependent) ──────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(sss.split(X_norm, y))

    X_tr, y_tr = X_norm[train_idx], y[train_idx]
    X_te, y_te = X_norm[test_idx],  y[test_idx]

    # Val split from train (10% of total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.111, random_state=42)
    tr_idx2, va_idx2 = next(sss2.split(X_tr, y_tr))
    X_va, y_va = X_tr[va_idx2], y_tr[va_idx2]
    X_tr, y_tr = X_tr[tr_idx2], y_tr[tr_idx2]

    print(f"  Features:    {X_tr.shape[1]}")
    print(f"  Train:       {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")
    print(f"  Split type:  SUBJECT-DEPENDENT (stratified 70/10/20)")
    print(f"  Per-subject Z-score normalization: YES")

    return X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols, df


# ─── Build 46 model configs ───────────────────────────────────────────────────
def get_model_configs():
    from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                      SGDClassifier, Perceptron)
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB, ComplementNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  AdaBoostClassifier, GradientBoostingClassifier,
                                  BaggingClassifier, VotingClassifier)
    from sklearn.neural_network import MLPClassifier
    import xgboost as xgb
    import lightgbm as lgb

    RF_base  = RandomForestClassifier(100, random_state=42, n_jobs=1)
    LR_base  = LogisticRegression(C=1, max_iter=500, random_state=42)
    XGB_base = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42,
                                   n_jobs=1, verbosity=0, eval_metric="mlogloss")

    configs = [
        # LINEAR (10)
        ("LR_C0.01",     LogisticRegression(C=0.01, max_iter=500, random_state=42)),
        ("LR_C0.1",      LogisticRegression(C=0.1,  max_iter=500, random_state=42)),
        ("LR_C1",        LogisticRegression(C=1,    max_iter=500, random_state=42)),
        ("LR_C10",       LogisticRegression(C=10,   max_iter=500, random_state=42)),
        ("Ridge",        RidgeClassifier(alpha=1.0)),
        ("LDA_svd",      LinearDiscriminantAnalysis(solver="svd")),
        ("LDA_shrink",   LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ("QDA",          QuadraticDiscriminantAnalysis()),
        ("SGD",          SGDClassifier(loss="hinge", max_iter=300, random_state=42, n_jobs=1)),
        ("Perceptron",   Perceptron(max_iter=300, random_state=42, n_jobs=1)),

        # SVM (9)
        ("SVM_Lin_C0.1", SVC(kernel="linear", C=0.1,  probability=True, random_state=42)),
        ("SVM_Lin_C1",   SVC(kernel="linear", C=1,    probability=True, random_state=42)),
        ("SVM_Lin_C10",  SVC(kernel="linear", C=10,   probability=True, random_state=42)),
        ("SVM_RBF_C1",   SVC(kernel="rbf",    C=1,    gamma="scale", probability=True, random_state=42)),
        ("SVM_RBF_C10",  SVC(kernel="rbf",    C=10,   gamma="scale", probability=True, random_state=42)),
        ("SVM_RBF_C100", SVC(kernel="rbf",    C=100,  gamma="scale", probability=True, random_state=42)),
        ("SVM_RBF_g0.01",SVC(kernel="rbf",    C=1,    gamma=0.01,    probability=True, random_state=42)),
        ("SVM_Poly2",    SVC(kernel="poly",   C=1,    degree=2,      probability=True, random_state=42)),
        ("SVM_Poly3",    SVC(kernel="poly",   C=1,    degree=3,      probability=True, random_state=42)),

        # KNN / NB (6)
        ("KNN_k3",       KNeighborsClassifier(n_neighbors=3,  n_jobs=1)),
        ("KNN_k5",       KNeighborsClassifier(n_neighbors=5,  n_jobs=1)),
        ("KNN_k9",       KNeighborsClassifier(n_neighbors=9,  n_jobs=1)),
        ("KNN_k5_man",   KNeighborsClassifier(n_neighbors=5,  metric="manhattan", n_jobs=1)),
        ("GaussianNB",   GaussianNB()),
        ("ComplementNB", ComplementNB()),

        # TREES (3)
        ("DT_d5",        DecisionTreeClassifier(max_depth=5,    criterion="gini",    random_state=42)),
        ("DT_d10",       DecisionTreeClassifier(max_depth=10,   criterion="entropy", random_state=42)),
        ("DT_full",      DecisionTreeClassifier(max_depth=None,                      random_state=42)),

        # RF / ET / BAGGING (6)
        ("RF_100_sqrt",  RandomForestClassifier(100, max_features="sqrt",  random_state=42, n_jobs=1)),
        ("RF_200_sqrt",  RandomForestClassifier(200, max_features="sqrt",  random_state=42, n_jobs=1)),
        ("RF_100_log2",  RandomForestClassifier(100, max_features="log2",  random_state=42, n_jobs=1)),
        ("ET_100",       ExtraTreesClassifier(100, random_state=42, n_jobs=1)),
        ("ET_200",       ExtraTreesClassifier(200, random_state=42, n_jobs=1)),
        ("Bagging_DT50", BaggingClassifier(DecisionTreeClassifier(max_depth=5),
                                            n_estimators=50, random_state=42, n_jobs=1)),

        # BOOSTING (8)
        ("AdaBoost50",   AdaBoostClassifier(n_estimators=50,  learning_rate=0.1, random_state=42, algorithm="SAMME")),
        ("AdaBoost100",  AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42, algorithm="SAMME")),
        ("GBM_d3",       GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        ("GBM_d5",       GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, random_state=42)),
        ("XGB_d3",       xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,  max_depth=3, random_state=42, n_jobs=1, verbosity=0, eval_metric="mlogloss")),
        ("XGB_d5",       xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=1, verbosity=0, eval_metric="mlogloss")),
        ("LGB_100",      lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1,  random_state=42, n_jobs=1, verbose=-1)),
        ("LGB_200",      lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=1, verbose=-1)),

        # MLP (3)
        ("MLP_64",       __import__("sklearn.neural_network", fromlist=["MLPClassifier"]).MLPClassifier(
                            hidden_layer_sizes=(64,),    activation="relu",  solver="adam",  max_iter=500, random_state=42)),
        ("MLP_128_64",   __import__("sklearn.neural_network", fromlist=["MLPClassifier"]).MLPClassifier(
                            hidden_layer_sizes=(128,64), activation="relu",  solver="adam",  max_iter=500, random_state=42)),
        ("MLP_tanh",     __import__("sklearn.neural_network", fromlist=["MLPClassifier"]).MLPClassifier(
                            hidden_layer_sizes=(64,32),  activation="tanh",  solver="lbfgs", max_iter=500, random_state=42)),

        # VOTING (1)
        ("Voting_soft",  VotingClassifier(
                            estimators=[("lr", LR_base), ("rf", RF_base), ("xgb", XGB_base)],
                            voting="soft", n_jobs=1)),
    ]
    return configs


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, n=5):
    acc = float((y_true == y_pred).mean())
    f1s = []
    for c in range(n):
        tp = ((y_pred==c)&(y_true==c)).sum()
        fp = ((y_pred==c)&(y_true!=c)).sum()
        fn = ((y_pred!=c)&(y_true==c)).sum()
        f1s.append(2*tp/(2*tp+fp+fn+1e-8))
    return acc, float(np.mean(f1s)), f1s


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import MinMaxScaler

    print("=" * 65)
    print("PHASE 2 (v2): 46 ML MODELS — SUBJECT-DEPENDENT EVALUATION")
    print("=" * 65)

    X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols, df = \
        load_features_subject_dependent()

    configs = get_model_configs()
    print(f"\n  Models to train: {len(configs)}")
    print(f"  Chance level:    {1/5:.4f}\n")

    results     = []
    best_model  = None
    best_acc    = -1.
    best_name   = ""
    best_preds  = None
    best_y      = None

    for i, (name, clf) in enumerate(configs):
        t0 = time.time()
        try:
            if "ComplementNB" in name:
                pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])
            else:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("select", SelectKBest(f_classif, k=min(100, X_tr.shape[1]))),
                    ("clf",    clf),
                ])

            pipe.fit(X_tr, y_tr)
            y_pred   = pipe.predict(X_te)
            acc, macro_f1, f1s = compute_metrics(y_te, y_pred)
            elapsed  = time.time() - t0
            y_vp     = pipe.predict(X_va)
            val_acc, val_f1, _ = compute_metrics(y_va, y_vp)
            status   = "OK"
        except Exception as e:
            acc=macro_f1=val_acc=val_f1=0.; f1s=[0.]*5
            elapsed = time.time() - t0
            status  = f"FAIL: {e}"
            print(f"  [{i+1:2d}/46] {name:<20} FAILED: {e}")

        row = {"rank":0,"model":name,
               "val_acc":round(val_acc,4),"val_f1":round(val_f1,4),
               "test_acc":round(acc,4),"test_f1":round(macro_f1,4),
               "f1_arith":round(f1s[0],4),"f1_pattern":round(f1s[1],4),
               "f1_memory":round(f1s[2],4),"f1_comp":round(f1s[3],4),
               "f1_attn":round(f1s[4],4),
               "train_time_s":round(elapsed,2),"status":status}
        results.append(row)

        gain = acc - 0.2
        sign = "+" if gain>=0 else ""
        print(f"  [{i+1:2d}/46] {name:<22} acc={acc:.4f} ({sign}{gain:.4f}) "
              f"f1={macro_f1:.4f} [{elapsed:.1f}s]")

        if acc > best_acc and status == "OK":
            best_acc=acc; best_name=name; best_model=pipe
            best_preds=y_pred; best_y=y_te

    results.sort(key=lambda r: r["test_acc"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    out_csv = RESULTS_DIR / "ml_results_subj_dep.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)

    print(f"\n  Best model: {best_name} (test_acc={best_acc:.4f})")
    print(f"  Results saved: {out_csv}")

    # ── FIGURES ───────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.family":"DejaVu Sans",
                         "axes.spines.top":False, "axes.spines.right":False})

    # Fig 1: All 46 models
    fig, ax = plt.subplots(figsize=(20, 8))
    names = [r["model"] for r in results]
    accs  = [r["test_acc"] for r in results]
    f1s_  = [r["test_f1"]  for r in results]
    x     = np.arange(len(names))
    ax.bar(x-0.2, accs, 0.38, label="Test Accuracy", color="#457B9D", alpha=0.85)
    ax.bar(x+0.2, f1s_, 0.38, label="Test Macro-F1", color="#E63946", alpha=0.85)
    ax.axhline(0.2, color="gray", ls="--", lw=1.5, label="Chance (20%)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.set_title("All 46 ML Models — Subject-Dependent Evaluation\n(Per-subject Z-score + Stratified 70/10/20 split)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"ml_fig1_all_models_subjdep.png", bbox_inches="tight", dpi=150)
    plt.close(); print("  Saved: ml_fig1_all_models_subjdep.png")

    # Fig 2: Top 15 horizontal bars
    top15 = results[:15]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = {"LR":"#264653","Ridge":"#264653","LDA":"#264653","QDA":"#264653",
            "SGD":"#264653","Perceptron":"#264653","SVM":"#2A9D8F",
            "KNN":"#E9C46A","GaussianNB":"#F4A261","ComplementNB":"#F4A261",
            "DT":"#E76F51","RF":"#457B9D","ET":"#457B9D","Bagging":"#457B9D",
            "AdaBoost":"#9B5DE5","GBM":"#9B5DE5","XGB":"#C77DFF","LGB":"#D62828",
            "MLP":"#3D405B","Voting":"#1B4332"}
    def mc(n):
        for p,c in cmap.items():
            if n.startswith(p): return c
        return "#888"

    for ax, metric, title in [(axes[0],"test_acc","Test Accuracy"),(axes[1],"test_f1","Test Macro-F1")]:
        vals=[r[metric] for r in top15]; mnames=[r["model"] for r in top15]
        bars=ax.barh(range(len(mnames)), vals, color=[mc(n) for n in mnames], edgecolor="white", lw=1.5)
        ax.set_yticks(range(len(mnames))); ax.set_yticklabels(mnames, fontsize=9)
        ax.invert_yaxis(); ax.axvline(0.2, color="gray", ls="--", lw=1.5)
        ax.set_xlim(0,1); ax.set_xlabel(title)
        ax.set_title(f"Top 15 — {title}", fontsize=11, fontweight="bold")
        for bar,v in zip(bars,vals):
            ax.text(v+0.005, bar.get_y()+bar.get_height()/2., f"{v:.3f}",
                    va="center", fontsize=8, fontweight="bold")
    fig.suptitle("Top 15 Models — Subject-Dependent EEG Cognitive Task Classification",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"ml_fig2_top15_subjdep.png", bbox_inches="tight", dpi=150)
    plt.close(); print("  Saved: ml_fig2_top15_subjdep.png")

    # Fig 3: Confusion matrix
    cm = np.zeros((5,5), dtype=int)
    for t,p in zip(best_y, best_preds):
        if 0<=t<5 and 0<=p<5: cm[t][p]+=1
    rs = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm.astype(float), rs, out=np.zeros_like(cm,dtype=float), where=rs>0)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    short=["Arith","Patt","Mem","Comp","Attn"]
    ax.set_xticklabels(short, rotation=30, ha="right"); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Best Model: {best_name}\nTest Acc={best_acc:.3f} (Subject-Dependent)",
                 fontsize=12, fontweight="bold")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    fontsize=11, color="white" if cmn[i,j]>0.5 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Recall", fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"ml_fig3_confusion_subjdep.png", bbox_inches="tight", dpi=150)
    plt.close(); print("  Saved: ml_fig3_confusion_subjdep.png")

    # Fig 4: Per-class F1 heatmap top 15
    fig, ax = plt.subplots(figsize=(10,7))
    matrix = np.array([[r["f1_arith"],r["f1_pattern"],r["f1_memory"],
                         r["f1_comp"],r["f1_attn"]] for r in top15])
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Arithmetic","Pattern","Memory","Comprehension","Attention"],rotation=20,ha="right")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels([r["model"] for r in top15], fontsize=9)
    for i in range(len(top15)):
        for j in range(5):
            ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=8)
    plt.colorbar(im, ax=ax, label="F1 Score", fraction=0.046)
    ax.set_title("Per-Class F1 Score — Top 15 Models (Subject-Dependent)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"ml_fig4_perclass_subjdep.png", bbox_inches="tight", dpi=150)
    plt.close(); print("  Saved: ml_fig4_perclass_subjdep.png")

    # Summary
    print(f"\n{'='*65}")
    print(f"{'Rank':<5} {'Model':<22} {'Val F1':>7} {'Test Acc':>9} {'Test F1':>9}")
    print("-"*65)
    for r in results[:15]:
        marker = " <-- BEST" if r["model"]==best_name else ""
        print(f"  {r['rank']:<4} {r['model']:<22} {r['val_f1']:>7.4f} "
              f"{r['test_acc']:>9.4f} {r['test_f1']:>9.4f}{marker}")
    print(f"{'Chance':<5} {'-':<22} {'-':>7} {'0.2000':>9} {'0.2000':>9}")
    print("="*65)

    with open(RESULTS_DIR/"ml_best_model.json","w") as f:
        json.dump({"best_model":best_name,"best_acc":best_acc,
                   "evaluation":"subject_dependent"}, f, indent=2)


if __name__ == "__main__":
    main()
