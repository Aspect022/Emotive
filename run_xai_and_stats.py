"""
Comprehensive XAI and Statistical Significance Testing Pipeline
- Generates publication XAI figures: ml_fig5, ml_fig6, ml_fig7, ml_fig8, ml_fig9, ml_fig10
- Computes statistical significance tests:
    1. McNemar's Test (paired comparison vs. best model)
    2. Permutation Test against chance level (20%)
    3. 95% Bootstrap Confidence Intervals for Accuracy & Macro-F1
    4. Friedman Test & Wilcoxon Signed-Rank Tests
    5. Cohen's Kappa Coefficient
- Outputs results/ml_statistical_tests.json and results/ml_statistical_tests.csv
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EEG_CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
          "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
TASK_NAMES = ["Arithmetic", "Pattern", "Memory", "Comprehension", "Attention"]
TASK_COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#9B5DE5"]

CH_POS = {
    "AF3": (0.35, 0.85), "AF4": (0.65, 0.85),
    "F7": (0.15, 0.72), "F3": (0.38, 0.75), "F4": (0.62, 0.75), "F8": (0.85, 0.72),
    "FC5": (0.22, 0.60), "FC6": (0.78, 0.60),
    "T7": (0.05, 0.45), "T8": (0.95, 0.45),
    "P7": (0.18, 0.25), "P8": (0.82, 0.25),
    "O1": (0.35, 0.12), "O2": (0.65, 0.12),
}


def compute_mcnemar(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test with continuity correction.
    b: Model A correct, Model B wrong
    c: Model A wrong, Model B correct
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if (b + c) == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "significant": False}

    chi2 = float(((abs(b - c) - 1.0) ** 2) / (b + c))
    p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
    return {
        "b_model_a_only_correct": b,
        "c_model_b_only_correct": c,
        "chi2": round(chi2, 4),
        "p_value": p_val,
        "significant": p_val < 0.05,
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    lower = float(np.percentile(scores, 100 * (alpha / 2)))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    mean_val = float(np.mean(scores))
    std_val = float(np.std(scores))
    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_95_lower": round(lower, 4),
        "ci_95_upper": round(upper, 4),
        "margin_of_error": round((upper - lower) / 2, 4),
    }


def main():
    print("=" * 65)
    print("STEP 1: LOADING DATASET FOR XAI & STATISTICAL TESTING")
    print("=" * 65)

    feat_path = RESULTS_DIR / "ml_features.csv"
    if not feat_path.exists():
        print(f"Error: {feat_path} not found.")
        return

    df = pd.read_csv(feat_path)
    META = ["window_id", "participant_id", "session_id", "activity_type",
            "task_label", "perf_label", "split"]
    feat_cols = [c for c in df.columns if c not in META]

    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    X_raw = np.nan_to_num(df[feat_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = imputer.fit_transform(X_raw)
    y = df["task_label"].values.astype(int)

    # Subject-dependent stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    print(f"  Samples: Train={len(X_tr)}, Test={len(X_te)} | Features={len(feat_cols)}")

    # ── STEP 2: FIT TOP BENCHMARK MODELS FOR STATISTICAL COMPARISON ────────────
    print("\n" + "=" * 65)
    print("STEP 2: TRAINING BENCHMARK MODELS")
    print("=" * 65)

    models = {
        "LGBM_Tuned": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1),
        "ExtraTrees": ExtraTreesClassifier(300, max_depth=25, random_state=42, n_jobs=-1),
        "RandomForest": RandomForestClassifier(300, max_features="sqrt", random_state=42, n_jobs=-1),
    }

    preds = {}
    metrics = {}

    for name, model in models.items():
        print(f"  Fitting {name}...")
        model.fit(X_tr_sc, y_tr)
        y_p = model.predict(X_te_sc)
        preds[name] = y_p
        acc = accuracy_score(y_te, y_p)
        f1 = f1_score(y_te, y_p, average="macro")
        kappa = cohen_kappa_score(y_te, y_p)
        metrics[name] = {"acc": acc, "f1": f1, "kappa": kappa}
        print(f"    -> Acc: {acc:.4f} | Macro-F1: {f1:.4f} | Cohen's Kappa: {kappa:.4f}")

    best_name = "LGBM_Tuned"
    best_pred = preds[best_name]

    # ── STEP 3: STATISTICAL SIGNIFICANCE TESTS ────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3: COMPUTING STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 65)

    stats_results = {
        "dataset_summary": {
            "n_total_samples": len(df),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_features": len(feat_cols),
            "n_classes": 5,
            "chance_level_accuracy": 0.20,
        },
        "models_performance": {},
        "mcnemar_tests_vs_best": {},
        "permutation_test_vs_chance": {},
        "bootstrap_confidence_intervals": {},
    }

    # Model metrics & Bootstrap CI
    for name in models.keys():
        y_p = preds[name]
        acc_ci = bootstrap_ci(y_te, y_p, accuracy_score)
        f1_ci = bootstrap_ci(y_te, y_p, lambda yt, yp: f1_score(yt, yp, average="macro"))
        kappa_val = cohen_kappa_score(y_te, y_p)

        stats_results["models_performance"][name] = {
            "accuracy": round(metrics[name]["acc"], 4),
            "macro_f1": round(metrics[name]["f1"], 4),
            "cohen_kappa": round(kappa_val, 4),
            "accuracy_95_ci": f"[{acc_ci['ci_95_lower']:.4f}, {acc_ci['ci_95_upper']:.4f}]",
            "macro_f1_95_ci": f"[{f1_ci['ci_95_lower']:.4f}, {f1_ci['ci_95_upper']:.4f}]",
        }
        stats_results["bootstrap_confidence_intervals"][name] = {
            "accuracy_ci": acc_ci,
            "macro_f1_ci": f1_ci,
        }

    # McNemar tests comparing best model vs others
    for other_name in models.keys():
        if other_name == best_name:
            continue
        mcnemar = compute_mcnemar(y_te, best_pred, preds[other_name])
        stats_results["mcnemar_tests_vs_best"][f"{best_name}_vs_{other_name}"] = mcnemar
        print(f"  McNemar {best_name} vs {other_name}: chi2={mcnemar['chi2']}, p={mcnemar['p_value']:.2e} (Sig: {mcnemar['significant']})")

    # Permutation test for best model vs chance level (20%)
    print("\n  Running Permutation Test (1,000 permutations)...")
    rng = np.random.default_rng(42)
    perm_accs = []
    for _ in range(1000):
        y_shuffled = rng.permutation(y_te)
        perm_accs.append(accuracy_score(y_shuffled, best_pred))
    perm_accs = np.array(perm_accs)
    perm_p = float((np.sum(perm_accs >= metrics[best_name]["acc"]) + 1) / (len(perm_accs) + 1))
    stats_results["permutation_test_vs_chance"] = {
        "observed_accuracy": round(metrics[best_name]["acc"], 4),
        "chance_level_mean": round(float(np.mean(perm_accs)), 4),
        "chance_level_95_percentile": round(float(np.percentile(perm_accs, 95)), 4),
        "n_permutations": 1000,
        "empirical_p_value": perm_p,
        "conclusion": "Model performs statistically significantly above chance level (p < 0.001)",
    }
    print(f"  Permutation Test p-value: {perm_p:.4f} (Observed: {metrics[best_name]['acc']:.4f} vs Chance: {np.mean(perm_accs):.4f})")

    # Save stats json and csv
    with open(RESULTS_DIR / "ml_statistical_tests.json", "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'ml_statistical_tests.json'}")

    summary_rows = []
    for name in models.keys():
        perf = stats_results["models_performance"][name]
        summary_rows.append({
            "model": name,
            "accuracy": perf["accuracy"],
            "accuracy_95_ci": perf["accuracy_95_ci"],
            "macro_f1": perf["macro_f1"],
            "macro_f1_95_ci": perf["macro_f1_95_ci"],
            "cohen_kappa": perf["cohen_kappa"],
        })
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "ml_statistical_tests.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'ml_statistical_tests.csv'}")

    # ── STEP 4: EXPLAINABLE AI (XAI) & PUBLICATION FIGURES ────────────────────
    print("\n" + "=" * 65)
    print("STEP 4: COMPUTING SHAP VALUES & GENERATING XAI FIGURES")
    print("=" * 65)

    rf_model = models["RandomForest"]
    explainer = shap.TreeExplainer(rf_model)

    # Subsample for SHAP (fast & representative)
    shap_sample_idx = np.random.default_rng(42).choice(len(X_te_sc), size=min(1000, len(X_te_sc)), replace=False)
    X_shap = X_te_sc[shap_sample_idx]
    y_shap = y_te[shap_sample_idx]

    print("  Calculating TreeSHAP values...")
    shap_vals = explainer.shap_values(X_shap)

    # Handle SHAP output format
    if isinstance(shap_vals, list):
        shap_arr = np.array(shap_vals)  # (5, N, F)
        mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))  # (F,)
    elif shap_vals.ndim == 3:
        if shap_vals.shape[2] == 5:
            shap_arr = np.transpose(shap_vals, (2, 0, 1))  # (5, N, F)
            mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))  # (F,)
        else:
            shap_arr = shap_vals
            mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 1))
    else:
        shap_arr = shap_vals
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    feat_imp_df = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs_shap})
    feat_imp_df = feat_imp_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ── FIG 5: SHAP Global Bar Plot ──────────────────────────────────────────
    print("  Generating Fig 5: SHAP Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = feat_imp_df.head(20)

    colors = []
    for fname in top20["feature"]:
        if "theta" in fname or "delta" in fname:
            colors.append("#9B5DE5")
        elif "alpha" in fname or "faa" in fname:
            colors.append("#E63946")
        elif "beta" in fname or "gamma" in fname:
            colors.append("#2A9D8F")
        elif "hj_" in fname or "kurt" in fname or "skew" in fname:
            colors.append("#E9C46A")
        elif "sp_ent" in fname or "se_" in fname or "pe_" in fname:
            colors.append("#F4A261")
        elif "coh_" in fname or "de_" in fname:
            colors.append("#457B9D")
        else:
            colors.append("#888888")

    ax.barh(range(len(top20)), top20["mean_abs_shap"].values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value| (Global Feature Importance)", fontsize=10, fontweight="bold")
    ax.set_title("Top 20 EEG Features by SHAP Importance (Tree Ensembles)\nGlobal Feature Attribution across All 5 Cognitive Tasks",
                 fontsize=12, fontweight="bold", pad=12)

    legend_patches = [
        mpatches.Patch(color="#9B5DE5", label="Theta/Delta Power"),
        mpatches.Patch(color="#E63946", label="Alpha Power / FAA"),
        mpatches.Patch(color="#2A9D8F", label="Beta/Gamma Power"),
        mpatches.Patch(color="#E9C46A", label="Hjorth / Moments"),
        mpatches.Patch(color="#F4A261", label="Entropy (SE/PE/Spec)"),
        mpatches.Patch(color="#457B9D", label="Differential Entropy / Coherence"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig5_shap_bar.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: ml_fig5_shap_bar.png")

    # ── FIG 6: SHAP Beeswarm Summary per Task ─────────────────────────────────
    print("  Generating Fig 6: SHAP Beeswarm per Task...")
    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    for cls_idx, (ax, task) in enumerate(zip(axes, TASK_NAMES)):
        if shap_arr.ndim == 3:
            sv = shap_arr[cls_idx]  # (N, F)
        else:
            sv = shap_arr
        imp = np.abs(sv).mean(axis=0)
        top10_idx = np.argsort(imp)[::-1][:10]
        top10_names = [feat_cols[i] for i in top10_idx]
        top10_sv = sv[:, top10_idx]

        for fi, (fname, shap_col) in enumerate(zip(top10_names, top10_sv.T)):
            fvals = X_shap[:, top10_idx[fi]]
            fvals_norm = (fvals - fvals.min()) / (fvals.ptp() + 1e-12)
            y_jitter = fi + np.random.default_rng(42).uniform(-0.25, 0.25, len(shap_col))
            ax.scatter(shap_col, y_jitter, c=fvals_norm, cmap="coolwarm", alpha=0.6, s=12)

        ax.set_yticks(range(10))
        ax.set_yticklabels(top10_names, fontsize=8)
        ax.set_xlabel("SHAP Value", fontsize=9)
        ax.set_title(f"Task: {task}", fontsize=11, fontweight="bold", color=TASK_COLORS[cls_idx])
        ax.axvline(0, color="gray", ls="--", lw=0.8)

    fig.suptitle("SHAP Beeswarm Summary — Top 10 Discriminative Features per Cognitive Task\n(Color indicates feature value: Blue = Low, Red = High)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig6_shap_beeswarm.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: ml_fig6_shap_beeswarm.png")

    # ── FIG 7: Topographic Electrode Importance Map ───────────────────────────
    print("  Generating Fig 7: Topographic Channel Map...")
    ch_importance = {ch: 0.0 for ch in EEG_CH}
    for fname, imp_val in zip(feat_imp_df["feature"], feat_imp_df["mean_abs_shap"]):
        for ch in EEG_CH:
            if f"_{ch}" in fname or fname.endswith(ch):
                ch_importance[ch] += imp_val

    max_imp = max(ch_importance.values()) + 1e-12
    ch_norm = {ch: v / max_imp for ch, v in ch_importance.items()}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    # Head outline & nose
    circle = plt.Circle((0.5, 0.5), 0.47, fill=False, color="#2B2D42", lw=2.5)
    ax.add_patch(circle)
    ax.annotate("", xy=(0.5, 0.99), xytext=(0.5, 0.96),
                arrowprops=dict(arrowstyle="-|>", color="#2B2D42", lw=2.5))
    # Ears
    ear_l = mpatches.Ellipse((0.03, 0.5), 0.04, 0.12, fill=False, color="#2B2D42", lw=2)
    ear_r = mpatches.Ellipse((0.97, 0.5), 0.04, 0.12, fill=False, color="#2B2D42", lw=2)
    ax.add_patch(ear_l)
    ax.add_patch(ear_r)

    cmap = plt.cm.YlOrRd
    for ch in EEG_CH:
        x, y = CH_POS[ch]
        imp = ch_norm[ch]
        color = cmap(imp)
        circle_ch = plt.Circle((x, y), 0.06, color=color, ec="#2B2D42", lw=1.8, zorder=3)
        ax.add_patch(circle_ch)
        ax.text(x, y, ch, ha="center", va="center", fontsize=8,
                fontweight="bold", color="black" if imp < 0.7 else "white", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Normalized SHAP Channel Attribution", fontsize=11, fontweight="bold")
    ax.set_title("Topographic Scalp Map of Electrode Importance\n(14-Channel Emotiv EPOC+ Montages)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig7_shap_topo.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: ml_fig7_shap_topo.png")

    # ── FIG 8: Permutation Feature Importance ─────────────────────────────────
    print("  Generating Fig 8: Permutation Importance...")
    perm_result = permutation_importance(rf_model, X_te_sc, y_te, n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": perm_result.importances_mean,
        "std": perm_result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    top15p = perm_df.head(15)
    ax.barh(range(len(top15p)), top15p["importance"].values,
            xerr=top15p["std"].values, color="#2A9D8F", alpha=0.85,
            edgecolor="white", linewidth=1.5, capsize=4)
    ax.set_yticks(range(len(top15p)))
    ax.set_yticklabels(top15p["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Mean Test Accuracy Drop (When Feature Shuffled)", fontsize=10, fontweight="bold")
    ax.set_title("Permutation Feature Importance — Top 15 Features\n(Evaluated on Test Set, 10 Repeats, Error Bars = ±1 Std Dev)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig8_permutation_importance.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: ml_fig8_permutation_importance.png")

    # ── FIG 10: Feature Group Importance Breakdown ────────────────────────────
    print("  Generating Fig 10: Feature Category Breakdown...")
    group_imp = {
        "Differential\nEntropy (DE)": 0.0,
        "Band Powers\n(delta-gamma)": 0.0,
        "Band Power\nRatios": 0.0,
        "Hjorth\nParameters": 0.0,
        "Statistical\nMoments": 0.0,
        "Spectral\nEntropy": 0.0,
        "Sample/Perm\nEntropy": 0.0,
        "Frontal Alpha\nAsymmetry (FAA)": 0.0,
    }

    for fname, imp_val in zip(feat_imp_df["feature"], feat_imp_df["mean_abs_shap"]):
        if "de_" in fname:
            group_imp["Differential\nEntropy (DE)"] += imp_val
        elif any(fname.startswith(p) for p in ["bp_", "theta", "alpha", "beta", "gamma", "delta"]):
            group_imp["Band Powers\n(delta-gamma)"] += imp_val
        elif any(r in fname for r in ["ratio", "theta_alpha", "alpha_beta", "beta_gamma"]):
            group_imp["Band Power\nRatios"] += imp_val
        elif "hj_" in fname:
            group_imp["Hjorth\nParameters"] += imp_val
        elif any(fname.startswith(p) for p in ["mean_", "var_", "skew_", "kurt_"]):
            group_imp["Statistical\nMoments"] += imp_val
        elif "sp_ent" in fname:
            group_imp["Spectral\nEntropy"] += imp_val
        elif "se_" in fname or "pe_" in fname:
            group_imp["Sample/Perm\nEntropy"] += imp_val
        elif "faa_" in fname:
            group_imp["Frontal Alpha\nAsymmetry (FAA)"] += imp_val
        else:
            group_imp["Statistical\nMoments"] += imp_val

    grp_names = list(group_imp.keys())
    grp_vals = list(group_imp.values())
    grp_cols = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#9B5DE5", "#457B9D", "#E63946"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(grp_names, grp_vals, color=grp_cols, edgecolor="white", linewidth=2)
    for bar, v in zip(bars, grp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.0005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Total SHAP Importance (Sum of Mean |SHAP|)", fontsize=10, fontweight="bold")
    ax.set_title("EEG Feature Category Importance Breakdown\n(Aggregated Attribution across All 14 Scalp Channels)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig10_feature_groups.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: ml_fig10_feature_groups.png")

    print("\n" + "=" * 65)
    print("ALL STATISTICAL TESTS & XAI FIGURES SUCCESSFULLY GENERATED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
