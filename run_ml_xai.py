"""
Phase 3: XAI Analysis
SHAP (TreeSHAP + KernelSHAP), permutation importance, PDP, LIME,
topographic channel importance map.
Requires ml_features.csv and ml_results_summary.csv.
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

EEG_CH  = ["AF3","F7","F3","FC5","T7","P7","O1",
            "O2","P8","T8","FC6","F4","F8","AF4"]
TASK_NAMES = ["Arithmetic","Pattern","Memory","Comprehension","Attention"]
TASK_COLORS= ["#E63946","#2A9D8F","#E9C46A","#457B9D","#9B5DE5"]

# Approximate 2D positions for 14 EPOC X channels (normalized 0-1)
CH_POS = {
    "AF3":(0.35,0.85),"AF4":(0.65,0.85),
    "F7": (0.15,0.72),"F3": (0.38,0.75),"F4": (0.62,0.75),"F8": (0.85,0.72),
    "FC5":(0.22,0.60),"FC6":(0.78,0.60),
    "T7": (0.05,0.45),"T8": (0.95,0.45),
    "P7": (0.18,0.25),"P8": (0.82,0.25),
    "O1": (0.35,0.12),"O2": (0.65,0.12),
}

def load_data():
    csv_path = RESULTS_DIR / "ml_features.csv"
    df = pd.read_csv(csv_path)
    META = ["window_id","participant_id","session_id","activity_type",
            "task_label","perf_label","split"]
    feat_cols = [c for c in df.columns if c not in META]

    train = df[df["split"].isin(["train","val"])]
    test  = df[df["split"] == "test"]

    X_tr = train[feat_cols].fillna(0).values.astype(np.float64)
    y_tr = train["task_label"].values.astype(int)
    X_te = test[feat_cols].fillna(0).values.astype(np.float64)
    y_te = test["task_label"].values.astype(int)
    return X_tr, y_tr, X_te, y_te, feat_cols

def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.inspection import permutation_importance, PartialDependenceDisplay
    import shap

    print("=" * 60)
    print("PHASE 3: XAI ANALYSIS")
    print("=" * 60)

    X_tr, y_tr, X_te, y_te, feat_cols = load_data()
    feat_cols = [str(f) for f in feat_cols]
    n_feats = min(100, X_tr.shape[1])

    # Scale + select features (same pipeline as training)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    selector = SelectKBest(f_classif, k=n_feats)
    X_tr_sel = selector.fit_transform(X_tr_sc, y_tr)
    X_te_sel = selector.transform(X_te_sc)
    sel_cols = [feat_cols[i] for i in selector.get_support(indices=True)]

    print(f"  Selected {n_feats} features from {len(feat_cols)} total")

    # ── Train RF for SHAP (fastest with TreeSHAP) ────────────────────────────
    print("\n  Training RF for TreeSHAP...")
    rf = RandomForestClassifier(200, random_state=42, n_jobs=-1)
    rf.fit(X_tr_sel, y_tr)
    rf_acc = (rf.predict(X_te_sel) == y_te).mean()
    print(f"  RF test accuracy: {rf_acc:.4f}")

    # ── TreeSHAP ─────────────────────────────────────────────────────────────
    print("  Computing TreeSHAP values...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_te_sel)   # list of 5 arrays: (N, F)

    # Stack: (5, N, F)
    shap_arr = np.array(shap_values)

    # Save SHAP values
    np.savez_compressed(RESULTS_DIR / "ml_shap_values.npz",
                        shap_values=shap_arr,
                        feature_names=np.array(sel_cols))

    # Mean |SHAP| per feature (across all classes)
    mean_abs_shap = np.abs(shap_arr).mean(axis=(0,1))  # (F,)
    feat_imp_df   = pd.DataFrame({"feature": sel_cols,
                                   "mean_abs_shap": mean_abs_shap})
    feat_imp_df   = feat_imp_df.sort_values("mean_abs_shap", ascending=False)

    # ── Fig: SHAP Bar Plot (global top 20) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    top20  = feat_imp_df.head(20)
    colors = []
    for fname in top20["feature"]:
        if "theta" in fname or "delta" in fname: colors.append("#9B5DE5")
        elif "alpha" in fname or "faa" in fname: colors.append("#E63946")
        elif "beta"  in fname or "gamma" in fname: colors.append("#2A9D8F")
        elif "hj_"   in fname: colors.append("#E9C46A")
        elif "se_" in fname or "pe_" in fname: colors.append("#F4A261")
        elif "coh_"  in fname: colors.append("#457B9D")
        else: colors.append("#888888")

    bars = ax.barh(range(len(top20)), top20["mean_abs_shap"].values,
                   color=colors, edgecolor="white", linewidth=1.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value| (feature importance)")
    ax.set_title("Top 20 EEG Features by SHAP Importance (RF Model)\nGlobal Feature Importance across All 5 Tasks",
                 fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#9B5DE5", label="Theta/Delta power"),
        mpatches.Patch(color="#E63946", label="Alpha power / FAA"),
        mpatches.Patch(color="#2A9D8F", label="Beta/Gamma power"),
        mpatches.Patch(color="#E9C46A", label="Hjorth parameters"),
        mpatches.Patch(color="#F4A261", label="Entropy (SE/PE)"),
        mpatches.Patch(color="#457B9D", label="Coherence"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig5_shap_bar.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig5_shap_bar.png")

    # ── Fig: SHAP Beeswarm (class 0: Arithmetic) ─────────────────────────────
    print("  Generating SHAP beeswarm...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    for cls_idx, (ax, task) in enumerate(zip(axes, TASK_NAMES)):
        sv  = shap_arr[cls_idx]                        # (N, F)
        # Sort by mean |SHAP| for this class
        imp = np.abs(sv).mean(axis=0)
        top10_idx = np.argsort(imp)[::-1][:10]
        top10_names = [sel_cols[i] for i in top10_idx]
        top10_sv    = sv[:, top10_idx]

        for fi, (fname, shap_col) in enumerate(zip(top10_names, top10_sv.T)):
            fvals = X_te_sel[:, top10_idx[fi]]
            fvals_norm = (fvals - fvals.min()) / (fvals.ptp() + 1e-12)
            y_jitter = fi + np.random.default_rng(42).uniform(-0.25, 0.25, len(shap_col))
            sc = ax.scatter(shap_col, y_jitter, c=fvals_norm,
                           cmap="coolwarm", alpha=0.5, s=8)

        ax.set_yticks(range(10))
        ax.set_yticklabels(top10_names, fontsize=7)
        ax.set_xlabel("SHAP value")
        ax.set_title(task, fontsize=10, fontweight="bold", color=TASK_COLORS[cls_idx])
        ax.axvline(0, color="gray", ls="--", lw=0.8)

    fig.suptitle("SHAP Beeswarm — Top 10 Features per Task\n(Blue=low feature value, Red=high)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig6_shap_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig6_shap_beeswarm.png")

    # ── Fig: Topographic channel importance ───────────────────────────────────
    print("  Generating topographic SHAP map...")
    ch_importance = {ch: 0. for ch in EEG_CH}
    for fname, imp_val in zip(feat_imp_df["feature"], feat_imp_df["mean_abs_shap"]):
        for ch in EEG_CH:
            if f"_{ch}" in fname or fname.endswith(ch):
                ch_importance[ch] += imp_val

    # Normalize
    max_imp = max(ch_importance.values()) + 1e-12
    ch_norm = {ch: v/max_imp for ch, v in ch_importance.items()}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    # Head outline
    circle = plt.Circle((0.5, 0.5), 0.47, fill=False, color="#333", lw=2)
    ax.add_patch(circle)
    # Nose
    ax.annotate("", xy=(0.5, 0.99), xytext=(0.5, 0.97),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))

    cmap = plt.cm.YlOrRd
    for ch in EEG_CH:
        x, y = CH_POS[ch]
        imp  = ch_norm[ch]
        color = cmap(imp)
        circle_ch = plt.Circle((x, y), 0.055, color=color, ec="#333", lw=1.5, zorder=3)
        ax.add_patch(circle_ch)
        ax.text(x, y, ch, ha="center", va="center", fontsize=7,
                fontweight="bold", color="black", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Normalized SHAP Importance", fontsize=10)
    ax.set_title("Topographic SHAP Importance Map\n(Which electrodes drive classification)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig7_shap_topo.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig7_shap_topo.png")

    # ── Permutation Importance ────────────────────────────────────────────────
    print("  Computing permutation importance...")
    perm_result = permutation_importance(rf, X_te_sel, y_te,
                                          n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature":    sel_cols,
        "importance": perm_result.importances_mean,
        "std":        perm_result.importances_std,
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    top15p = perm_df.head(15)
    ax.barh(range(len(top15p)), top15p["importance"].values,
            xerr=top15p["std"].values, color="#2A9D8F", alpha=0.8,
            edgecolor="white", linewidth=1.5, capsize=4)
    ax.set_yticks(range(len(top15p)))
    ax.set_yticklabels(top15p["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Mean accuracy decrease when feature is permuted")
    ax.set_title("Permutation Feature Importance — Top 15\n(RF model, 10 repeats, error bars = std)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig8_permutation_importance.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig8_permutation_importance.png")

    # ── LIME (5 test examples) ────────────────────────────────────────────────
    print("  Running LIME on 5 test examples...")
    try:
        import lime.lime_tabular
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_tr_sel, feature_names=sel_cols,
            class_names=TASK_NAMES,
            discretize_continuous=True, random_state=42)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for idx, ax in enumerate(axes):
            test_sample = X_te_sel[idx]
            exp = lime_explainer.explain_instance(
                test_sample, rf.predict_proba,
                num_features=8, num_samples=500)
            true_label = TASK_NAMES[y_te[idx]]
            pred_label = TASK_NAMES[rf.predict([test_sample])[0]]
            exp_list   = exp.as_list()
            feats_l    = [e[0][:25] for e in exp_list]
            values_l   = [e[1] for e in exp_list]
            colors_l   = ["#2A9D8F" if v > 0 else "#E63946" for v in values_l]
            ax.barh(range(len(feats_l)), values_l, color=colors_l,
                    edgecolor="white", linewidth=1.2)
            ax.set_yticks(range(len(feats_l)))
            ax.set_yticklabels(feats_l, fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0, color="gray", ls="--", lw=0.8)
            ax.set_title(f"Test #{idx}\nTrue: {true_label}\nPred: {pred_label}",
                         fontsize=9, fontweight="bold",
                         color="#2A9D8F" if true_label==pred_label else "#E63946")

        fig.suptitle("LIME Local Explanations — 5 Test Windows\n(Green=pushes toward predicted class, Red=pushes away)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ml_fig9_lime.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("  Saved: ml_fig9_lime.png")
    except Exception as e:
        print(f"  LIME failed: {e} (skipping)")

    # ── Feature group ablation bar ─────────────────────────────────────────────
    print("  Generating feature group importance breakdown...")
    group_imp = {
        "Statistical\n(mean/var/skew/kurt)": 0.,
        "Hjorth\nParameters": 0.,
        "Band Powers\n(delta-gamma)": 0.,
        "Spectral\nEntropy": 0.,
        "Theta/Alpha\nRatio": 0.,
        "Sample\nEntropy": 0.,
        "Permutation\nEntropy": 0.,
        "Frontal Alpha\nAsymmetry": 0.,
        "Theta\nCoherence": 0.,
    }
    for fname, imp_val in zip(feat_imp_df["feature"], feat_imp_df["mean_abs_shap"]):
        if any(fname.startswith(p) for p in ["mean_","var_","skew_","kurt_"]):
            group_imp["Statistical\n(mean/var/skew/kurt)"] += imp_val
        elif fname.startswith("hj_"):
            group_imp["Hjorth\nParameters"] += imp_val
        elif fname.startswith("bp_"):
            group_imp["Band Powers\n(delta-gamma)"] += imp_val
        elif fname.startswith("sp_ent"):
            group_imp["Spectral\nEntropy"] += imp_val
        elif fname.startswith("theta_alpha"):
            group_imp["Theta/Alpha\nRatio"] += imp_val
        elif fname.startswith("se_"):
            group_imp["Sample\nEntropy"] += imp_val
        elif fname.startswith("pe_"):
            group_imp["Permutation\nEntropy"] += imp_val
        elif fname.startswith("faa_"):
            group_imp["Frontal Alpha\nAsymmetry"] += imp_val
        elif fname.startswith("coh_"):
            group_imp["Theta\nCoherence"] += imp_val

    grp_names = list(group_imp.keys())
    grp_vals  = list(group_imp.values())
    grp_cols  = ["#264653","#E9C46A","#9B5DE5","#2A9D8F","#F4A261",
                 "#E76F51","#457B9D","#E63946","#1B4332"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(grp_names, grp_vals, color=grp_cols, edgecolor="white", linewidth=2)
    for bar, v in zip(bars, grp_vals):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.0005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Total SHAP Importance (sum of mean |SHAP|)")
    ax.set_title("EEG Feature Group Importance\n(Which feature categories contribute most to classification)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig10_feature_groups.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: ml_fig10_feature_groups.png")

    print(f"\n  Top 10 most important features overall:")
    for _, row in feat_imp_df.head(10).iterrows():
        print(f"    {row['feature']:<40} SHAP={row['mean_abs_shap']:.5f}")

    print(f"\n  All XAI figures saved to: {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
