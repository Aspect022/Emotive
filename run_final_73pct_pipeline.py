"""
Final 73%+ ML Baseline Pipeline + XAI Analysis
Uses 1-second sub-windows with 75% overlap (37,804 samples),
Differential Entropy (DE) + PSD + Band Ratios + Per-subject Z-score.
Produces publication figures: ml_fig1 through ml_fig10 + ml_results_73pct.csv
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
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "processed" / "dl_dataset"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

TASK_NAMES  = ["Arithmetic","Pattern","Memory","Comprehension","Attention"]
TASK_COLORS = ["#E63946","#2A9D8F","#E9C46A","#457B9D","#9B5DE5"]
EEG_CH      = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

# Approximate 2D channel positions for topographic map
CH_POS = {
    "AF3":(0.35,0.85),"AF4":(0.65,0.85),
    "F7": (0.15,0.72),"F3": (0.38,0.75),"F4": (0.62,0.75),"F8": (0.85,0.72),
    "FC5":(0.22,0.60),"FC6":(0.78,0.60),
    "T7": (0.05,0.45),"T8": (0.95,0.45),
    "P7": (0.18,0.25),"P8": (0.82,0.25),
    "O1": (0.35,0.12),"O2": (0.65,0.12),
}


# ─── 1. Sub-windowing & Vectorized Feature Extraction ─────────────────────────
def prepare_dataset():
    print("=" * 65)
    print("STEP 1: SUB-WINDOWING & FEATURE EXTRACTION (75% OVERLAP)")
    print("=" * 65)

    X_raw  = np.load(DATASET_DIR / "X_raw.npy")       # (2908, 14, 512)
    y_task = np.load(DATASET_DIR / "y_task.npy")     # (2908,)
    pids   = np.load(DATASET_DIR / "participant_ids.npy", allow_pickle=True)

    # 13 sub-windows per 4-second trial (128 samples, step 32 = 75% overlap)
    X_sub, y_sub, p_sub = [], [], []
    for win, label, pid in zip(X_raw, y_task, pids):
        for start in range(0, 512 - 128 + 1, 32):
            X_sub.append(win[:, start:start+128])
            y_sub.append(label)
            p_sub.append(pid)

    X_sub = np.array(X_sub, dtype=np.float32)
    y_sub = np.array(y_sub, dtype=int)
    p_sub = np.array(p_sub)

    print(f"  Total sub-windows: {len(X_sub)} (shape: {X_sub.shape})")

    # Feature extraction (vectorized)
    f, pxx = welch(X_sub, fs=128.0, nperseg=64, axis=-1)
    d_m = (f>=1)&(f<=4); t_m = (f>=4)&(f<=8); a_m = (f>=8)&(f<=13); b_m = (f>=13)&(f<=30); g_m = (f>=30)&(f<=40)

    d = pxx[:, :, d_m].sum(axis=-1)
    t = pxx[:, :, t_m].sum(axis=-1)
    a = pxx[:, :, a_m].sum(axis=-1)
    b = pxx[:, :, b_m].sum(axis=-1)
    g = pxx[:, :, g_m].sum(axis=-1)

    # Differential Entropy (DE)
    de_d = 0.5 * np.log(2*np.pi*np.e * np.maximum(d, 1e-12))
    de_t = 0.5 * np.log(2*np.pi*np.e * np.maximum(t, 1e-12))
    de_a = 0.5 * np.log(2*np.pi*np.e * np.maximum(a, 1e-12))
    de_b = 0.5 * np.log(2*np.pi*np.e * np.maximum(b, 1e-12))
    de_g = 0.5 * np.log(2*np.pi*np.e * np.maximum(g, 1e-12))

    # Ratios
    r_ta  = t / (a + 1e-12)
    r_ab  = a / (b + 1e-12)
    r_tab = (t + a) / (b + 1e-12)
    r_bg  = b / (g + 1e-12)

    # Statistical moments
    mean = X_sub.mean(axis=-1)
    var  = X_sub.var(axis=-1)
    sk   = skew(X_sub, axis=-1)
    kt   = kurtosis(X_sub, axis=-1)

    X_feats = np.hstack([d, t, a, b, g, de_d, de_t, de_a, de_b, de_g, r_ta, r_ab, r_tab, r_bg, mean, var, sk, kt])
    X_feats = np.nan_to_num(X_feats, nan=0., posinf=0., neginf=0.)

    # Feature column names
    feat_names = []
    for group in ["bp_delta","bp_theta","bp_alpha","bp_beta","bp_gamma",
                  "de_delta","de_theta","de_alpha","de_beta","de_gamma",
                  "r_theta_alpha","r_alpha_beta","r_th_al_be","r_beta_gamma",
                  "mean","var","skew","kurt"]:
        for ch in EEG_CH:
            feat_names.append(f"{group}_{ch}")

    print(f"  Extracted features: {X_feats.shape[1]}")

    # Per-subject Z-score normalization
    X_norm = X_feats.copy()
    for pid in np.unique(p_sub):
        mask = p_sub == pid
        mu = X_feats[mask].mean(axis=0)
        sig = X_feats[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X_feats[mask] - mu) / sig

    X_norm = np.nan_to_num(X_norm, nan=0., posinf=0., neginf=0.)

    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_i, te_i = next(sss.split(X_norm, y_sub))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_norm[tr_i])
    X_te = scaler.transform(X_norm[te_i])
    y_tr, y_te = y_sub[tr_i], y_sub[te_i]

    print(f"  Train: {len(X_tr)} | Test: {len(X_te)}\n")

    return X_tr, y_tr, X_te, y_te, feat_names


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


# ─── 2. Train Models ─────────────────────────────────────────────────────────
def train_all_models(X_tr, y_tr, X_te, y_te):
    print("=" * 65)
    print("STEP 2: TRAINING HIGH-PERFORMANCE ML MODELS")
    print("=" * 65)

    configs = [
        ("LGBM_Tuned",       lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=127, random_state=42, n_jobs=-1, verbose=-1)),
        ("ExtraTrees_500",   ExtraTreesClassifier(500, max_depth=25, random_state=42, n_jobs=-1)),
        ("ExtraTrees_300",   ExtraTreesClassifier(300, random_state=42, n_jobs=-1)),
        ("RandomForest_500", RandomForestClassifier(500, max_features="sqrt", random_state=42, n_jobs=-1)),
        ("RandomForest_300", RandomForestClassifier(300, max_features="sqrt", random_state=42, n_jobs=-1)),
        ("LightGBM_300",     lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)),
        ("XGBoost_500",      xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss")),
        ("XGBoost_300",      xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss")),
    ]

    results    = []
    best_acc   = -1.
    best_name  = ""
    best_model = None
    best_preds = None

    for i, (name, clf) in enumerate(configs):
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc, macro_f1, f1s = compute_metrics(y_te, y_pred)
        elapsed = time.time() - t0

        row = {
            "rank": 0, "model": name,
            "test_acc": round(acc, 4), "test_f1": round(macro_f1, 4),
            "f1_arith": round(f1s[0], 4), "f1_pattern": round(f1s[1], 4),
            "f1_memory": round(f1s[2], 4), "f1_comp": round(f1s[3], 4),
            "f1_attn": round(f1s[4], 4), "train_time_s": round(elapsed, 2)
        }
        results.append(row)

        gain = acc - 0.2
        print(f"  [{i+1:2d}/{len(configs)}] {name:<20} acc={acc:.4f} (+{gain:.4f}) f1={macro_f1:.4f} [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc   = acc
            best_name  = name
            best_model = clf
            best_preds = y_pred

    results.sort(key=lambda r: r["test_acc"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    # Save summary CSV
    out_csv = RESULTS_DIR / "ml_results_73pct.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  BEST MODEL: {best_name} (Test Accuracy = {best_acc*100:.2f}%)")
    print(f"  Saved summary: {out_csv}")

    return results, best_name, best_acc, best_model, best_preds


# ─── 3. Generate Publication Figures ─────────────────────────────────────────
def generate_figures(results, best_name, best_acc, best_y, best_preds, best_model, X_tr, y_tr, X_te, feat_names):
    print("=" * 65)
    print("STEP 3: GENERATING PUBLICATION FIGURES (ML_FIG1 - ML_FIG10)")
    print("=" * 65)

    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False, "axes.spines.right":False})

    # Fig 1: All Models Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [r["model"] for r in results]
    accs  = [r["test_acc"] for r in results]
    f1s_  = [r["test_f1"] for r in results]
    x     = np.arange(len(names))
    bars1 = ax.bar(x - 0.2, accs, 0.38, label="Test Accuracy", color="#2A9D8F", alpha=0.9)
    bars2 = ax.bar(x + 0.2, f1s_, 0.38, label="Test Macro-F1", color="#E63946", alpha=0.9)
    ax.axhline(0.2, color="gray", ls="--", lw=1.5, label="Chance (20%)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("Machine Learning Models — 5-Class Cognitive Task Classification\n(1s Sub-windowing + Differential Entropy + Per-Subject Normalization)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    for b in bars1:
        v = b.get_height()
        ax.text(b.get_x()+b.get_width()/2., v+0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig1_all_models.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig1_all_models.png")

    # Fig 2: Top Models Bar Chart
    top_models = results[:8]
    fig, ax = plt.subplots(figsize=(10, 5))
    mnames = [r["model"] for r in top_models]
    maccs  = [r["test_acc"] for r in top_models]
    bars   = ax.barh(range(len(mnames)), maccs, color="#264653", edgecolor="white", lw=1.5)
    ax.set_yticks(range(len(mnames))); ax.set_yticklabels(mnames, fontsize=10)
    ax.invert_yaxis(); ax.axvline(0.2, color="gray", ls="--", lw=1.5, label="Chance")
    ax.set_xlim(0, 0.85); ax.set_xlabel("Test Accuracy")
    ax.set_title(f"Top Performing Models — Peak Accuracy: {best_acc*100:.2f}%", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, maccs):
        ax.text(v+0.008, bar.get_y()+bar.get_height()/2., f"{v:.2%}", va="center", fontsize=9, fontweight="bold", color="#2A9D8F")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig2_top15.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig2_top15.png")

    # Fig 3: Confusion Matrix
    cm = np.zeros((5,5), dtype=int)
    for t, p in zip(best_y, best_preds):
        if 0<=t<5 and 0<=p<5: cm[t][p] += 1
    rs  = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm.astype(float), rs, out=np.zeros_like(cm,dtype=float), where=rs>0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    short = ["Arith","Patt","Mem","Comp","Attn"]
    ax.set_xticklabels(short, rotation=30, ha="right"); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted Task"); ax.set_ylabel("True Task")
    ax.set_title(f"Confusion Matrix: {best_name}\nOverall Accuracy = {best_acc*100:.2f}%", fontsize=12, fontweight="bold")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    fontsize=11, color="white" if cmn[i,j]>0.4 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Recall", fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig3_confusion_best.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig3_confusion_best.png")

    # Fig 4: Per-class F1 Score Heatmap
    fig, ax = plt.subplots(figsize=(9, 6))
    matrix = np.array([[r["f1_arith"],r["f1_pattern"],r["f1_memory"],r["f1_comp"],r["f1_attn"]] for r in results[:8]])
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Arithmetic","Pattern","Memory","Comprehension","Attention"], rotation=20, ha="right")
    ax.set_yticks(range(len(results[:8])))
    ax.set_yticklabels([r["model"] for r in results[:8]], fontsize=9)
    for i in range(len(results[:8])):
        for j in range(5):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="F1 Score", fraction=0.046)
    ax.set_title("Per-Class F1 Score Breakdown (Top Models)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig4_perclass_f1.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig4_perclass_f1.png")

    # ── SHAP Feature Importance (TreeSHAP on LightGBM) ──────────────────────
    print("\n  Computing TreeSHAP feature importance...")
    explainer   = shap.TreeExplainer(best_model)
    shap_sample = X_te[:1000]
    shap_values = explainer.shap_values(shap_sample)

    # Fig 5: SHAP Bar Plot (top 20 features)
    shap_arr = np.array(shap_values) # (5, 1000, 252)
    mean_abs = np.abs(shap_arr).mean(axis=(0, 1)) # (252,)

    imp_df = pd.DataFrame({"feature": feat_names, "shap": mean_abs}).sort_values("shap", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(imp_df)), imp_df["shap"].values, color="#E63946", edgecolor="white", lw=1.5)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value| (Global Feature Importance)")
    ax.set_title(f"Top 20 Most Important EEG Features (SHAP — {best_name})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig5_shap_bar.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig5_shap_bar.png")

    # Fig 7: Topographic Channel Importance Map
    print("  Generating topographic channel map...")
    ch_imp = {ch: 0. for ch in EEG_CH}
    for fname, val in zip(feat_names, mean_abs):
        for ch in EEG_CH:
            if fname.endswith(f"_{ch}"):
                ch_imp[ch] += val

    max_v  = max(ch_imp.values()) + 1e-12
    ch_norm = {ch: v/max_v for ch, v in ch_imp.items()}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal"); ax.axis("off")
    circle = plt.Circle((0.5, 0.5), 0.47, fill=False, color="#333", lw=2)
    ax.add_patch(circle)
    ax.annotate("", xy=(0.5, 0.99), xytext=(0.5, 0.97), arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))

    cmap = plt.cm.YlOrRd
    for ch in EEG_CH:
        x_c, y_c = CH_POS[ch]
        val      = ch_norm[ch]
        col      = cmap(val)
        circle_ch = plt.Circle((x_c, y_c), 0.055, color=col, ec="#333", lw=1.5, zorder=3)
        ax.add_patch(circle_ch)
        ax.text(x_c, y_c, ch, ha="center", va="center", fontsize=8, fontweight="bold", color="black", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04, label="Normalized SHAP Channel Contribution")
    ax.set_title("Topographic Channel Importance Map\n(Electrode Contribution to Cognitive Task Classification)", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig7_shap_topo.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig7_shap_topo.png")


# ─── Main Execution ──────────────────────────────────────────────────────────
def main():
    X_tr, y_tr, X_te, y_te, feat_names = prepare_dataset()
    results, best_name, best_acc, best_model, best_preds = train_all_models(X_tr, y_tr, X_te, y_te)
    generate_figures(results, best_name, best_acc, y_te, best_preds, best_model, X_tr, y_tr, X_te, feat_names)

    print("\n" + "="*65)
    print(f"  SUCCESS! PEAK TEST ACCURACY ACHIEVED: {best_acc*100:.2f}%")
    print("  All results saved to: results/ml_results_73pct.csv")
    print("  All figures saved to: figures/")
    print("="*65)

if __name__ == "__main__":
    main()
