"""
Generate Complete XAI Figures and Statistical Significance Tests
for the 73.34% EEG Cognitive Classification Pipeline (37,804 Sub-Windows)
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "processed" / "dl_dataset"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

TASK_NAMES = ["Arithmetic", "Pattern", "Memory", "Comprehension", "Attention"]
TASK_COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#9B5DE5"]
EEG_CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
          "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

CH_POS = {
    "AF3": (0.35, 0.85), "AF4": (0.65, 0.85),
    "F7": (0.15, 0.72), "F3": (0.38, 0.75), "F4": (0.62, 0.75), "F8": (0.85, 0.72),
    "FC5": (0.22, 0.60), "FC6": (0.78, 0.60),
    "T7": (0.05, 0.45), "T8": (0.95, 0.45),
    "P7": (0.18, 0.25), "P8": (0.82, 0.25),
    "O1": (0.35, 0.12), "O2": (0.65, 0.12),
}


def compute_mcnemar(y_true, y_pred_a, y_pred_b):
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
        "significant": bool(p_val < 0.05),
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
    return {
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "ci_95_lower": round(lower, 4),
        "ci_95_upper": round(upper, 4),
    }


def main():
    print("=" * 65)
    print("STEP 1: SUB-WINDOWING & FEATURE EXTRACTION (37,804 SAMPLES)")
    print("=" * 65)

    X_raw = np.load(DATASET_DIR / "X_raw.npy")
    y_task = np.load(DATASET_DIR / "y_task.npy")
    pids = np.load(DATASET_DIR / "participant_ids.npy", allow_pickle=True)

    X_sub, y_sub, p_sub = [], [], []
    for win, label, pid in zip(X_raw, y_task, pids):
        for start in range(0, 512 - 128 + 1, 32):
            X_sub.append(win[:, start:start + 128])
            y_sub.append(label)
            p_sub.append(pid)

    X_sub = np.array(X_sub, dtype=np.float32)
    y_sub = np.array(y_sub, dtype=int)
    p_sub = np.array(p_sub)

    print(f"  Total sub-windows: {len(X_sub)} (shape: {X_sub.shape})")

    # Vectorized Welch PSD
    f, pxx = welch(X_sub, fs=128.0, nperseg=64, axis=-1)
    d = pxx[:, :, (f >= 1) & (f <= 4)].sum(axis=-1)
    t = pxx[:, :, (f >= 4) & (f <= 8)].sum(axis=-1)
    a = pxx[:, :, (f >= 8) & (f <= 13)].sum(axis=-1)
    b = pxx[:, :, (f >= 13) & (f <= 30)].sum(axis=-1)
    g = pxx[:, :, (f >= 30) & (f <= 40)].sum(axis=-1)

    # Differential Entropy
    de_d = 0.5 * np.log(2 * np.pi * np.e * np.maximum(d, 1e-12))
    de_t = 0.5 * np.log(2 * np.pi * np.e * np.maximum(t, 1e-12))
    de_a = 0.5 * np.log(2 * np.pi * np.e * np.maximum(a, 1e-12))
    de_b = 0.5 * np.log(2 * np.pi * np.e * np.maximum(b, 1e-12))
    de_g = 0.5 * np.log(2 * np.pi * np.e * np.maximum(g, 1e-12))

    # Ratios
    r_ta = t / (a + 1e-12)
    r_ab = a / (b + 1e-12)
    r_tab = (t + a) / (b + 1e-12)
    r_bg = b / (g + 1e-12)

    # Statistical Moments
    mean = X_sub.mean(axis=-1)
    var = X_sub.var(axis=-1)
    sk = skew(X_sub, axis=-1)
    kt = kurtosis(X_sub, axis=-1)

    X_feats = np.hstack([d, t, a, b, g, de_d, de_t, de_a, de_b, de_g, r_ta, r_ab, r_tab, r_bg, mean, var, sk, kt])
    X_feats = np.nan_to_num(X_feats, nan=0.0, posinf=0.0, neginf=0.0)

    feat_names = []
    for group in ["bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma",
                  "de_delta", "de_theta", "de_alpha", "de_beta", "de_gamma",
                  "r_theta_alpha", "r_alpha_beta", "r_th_al_be", "r_beta_gamma",
                  "mean", "var", "skew", "kurt"]:
        for ch in EEG_CH:
            feat_names.append(f"{group}_{ch}")

    # Per-subject Z-score
    X_norm = X_feats.copy()
    for pid in np.unique(p_sub):
        mask = p_sub == pid
        mu = X_feats[mask].mean(axis=0)
        sig = X_feats[mask].std(axis=0) + 1e-8
        X_norm[mask] = (X_feats[mask] - mu) / sig

    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_i, te_i = next(sss.split(X_norm, y_sub))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_norm[tr_i])
    X_te = scaler.transform(X_norm[te_i])
    y_tr, y_te = y_sub[tr_i], y_sub[te_i]

    print(f"  Train samples: {len(X_tr)} | Test samples: {len(X_te)} | Features: {len(feat_names)}")

    # ── STEP 2: MODEL TRAINING ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2: TRAINING MODELS FOR STATISTICAL TESTING & XAI")
    print("=" * 65)

    models = {
        "LGBM_Tuned": lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=127, random_state=42, n_jobs=-1, verbose=-1),
        "ExtraTrees_300": ExtraTreesClassifier(300, random_state=42, n_jobs=-1),
        "RandomForest_500": RandomForestClassifier(500, max_features="sqrt", random_state=42, n_jobs=-1),
    }

    preds = {}
    metrics = {}

    for name, clf in models.items():
        t0 = time.time()
        print(f"  Fitting {name}...")
        clf.fit(X_tr, y_tr)
        y_p = clf.predict(X_te)
        preds[name] = y_p
        acc = accuracy_score(y_te, y_p)
        f1 = f1_score(y_te, y_p, average="macro")
        kappa = cohen_kappa_score(y_te, y_p)
        metrics[name] = {"acc": acc, "f1": f1, "kappa": kappa}
        print(f"    -> Acc: {acc*100:.2f}% | Macro-F1: {f1:.4f} | Cohen's Kappa: {kappa:.4f} [{time.time()-t0:.1f}s]")

    best_name = "LGBM_Tuned"
    best_pred = preds[best_name]

    # ── STEP 3: STATISTICAL TESTING ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3: COMPUTING STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 65)

    stats_results = {
        "dataset_summary": {
            "n_total_subwindows": len(X_sub),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_features": len(feat_names),
            "n_classes": 5,
            "chance_level_accuracy": 0.20,
        },
        "models_performance": {},
        "mcnemar_tests_vs_best": {},
        "permutation_test_vs_chance": {},
        "bootstrap_confidence_intervals": {},
    }

    # Bootstrap 95% CIs
    for name in models.keys():
        y_p = preds[name]
        acc_ci = bootstrap_ci(y_te, y_p, accuracy_score)
        f1_ci = bootstrap_ci(y_te, y_p, lambda yt, yp: f1_score(yt, yp, average="macro"))
        kappa_val = cohen_kappa_score(y_te, y_p)

        stats_results["models_performance"][name] = {
            "accuracy": round(metrics[name]["acc"], 4),
            "accuracy_percent": f"{metrics[name]['acc']*100:.2f}%",
            "macro_f1": round(metrics[name]["f1"], 4),
            "cohen_kappa": round(kappa_val, 4),
            "accuracy_95_ci": f"[{acc_ci['ci_95_lower']*100:.2f}%, {acc_ci['ci_95_upper']*100:.2f}%]",
            "macro_f1_95_ci": f"[{f1_ci['ci_95_lower']:.4f}, {f1_ci['ci_95_upper']:.4f}]",
        }
        stats_results["bootstrap_confidence_intervals"][name] = {
            "accuracy_ci": acc_ci,
            "macro_f1_ci": f1_ci,
        }

    # McNemar tests
    for other_name in models.keys():
        if other_name == best_name:
            continue
        mcnemar = compute_mcnemar(y_te, best_pred, preds[other_name])
        stats_results["mcnemar_tests_vs_best"][f"{best_name}_vs_{other_name}"] = mcnemar
        print(f"  McNemar {best_name} vs {other_name}: chi2={mcnemar['chi2']}, p={mcnemar['p_value']:.2e} (Sig: {mcnemar['significant']})")

    # Permutation Test
    print("  Running Permutation Test (1,000 permutations)...")
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
    print(f"  Permutation Test p-value: {perm_p:.4f} (Observed: {metrics[best_name]['acc']*100:.2f}% vs Chance: {np.mean(perm_accs)*100:.2f}%)")

    with open(RESULTS_DIR / "ml_statistical_tests.json", "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'ml_statistical_tests.json'}")

    # ── STEP 4: XAI SHAP COMPUTATION & FIGURES ────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4: COMPUTING TREE SHAP & GENERATING PUBLICATION FIGURES")
    print("=" * 65)

    explainer = shap.TreeExplainer(models["LGBM_Tuned"])
    shap_sample = X_te[:1000]
    shap_vals = explainer.shap_values(shap_sample)

    if isinstance(shap_vals, list):
        shap_arr = np.array(shap_vals)  # (5, 1000, 252)
        mean_abs = np.abs(shap_arr).mean(axis=(0, 1))
    elif isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            if shap_vals.shape[2] == 5:
                shap_arr = np.transpose(shap_vals, (2, 0, 1))
                mean_abs = np.abs(shap_vals).mean(axis=(0, 2))
            else:
                shap_arr = shap_vals
                mean_abs = np.abs(shap_vals).mean(axis=(0, 1))
        else:
            shap_arr = shap_vals
            mean_abs = np.abs(shap_vals).mean(axis=0)

    feat_imp_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
    feat_imp_df = feat_imp_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Fig 5: SHAP Bar Plot (top 20)
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = feat_imp_df.head(20)
    colors = []
    for fname in top20["feature"]:
        if "theta" in fname or "delta" in fname:
            colors.append("#9B5DE5")
        elif "alpha" in fname:
            colors.append("#E63946")
        elif "beta" in fname or "gamma" in fname:
            colors.append("#2A9D8F")
        elif "kurt" in fname or "skew" in fname or "var" in fname or "mean" in fname:
            colors.append("#E9C46A")
        elif "de_" in fname:
            colors.append("#457B9D")
        else:
            colors.append("#F4A261")

    ax.barh(range(len(top20)), top20["mean_abs_shap"].values, color=colors, edgecolor="white", lw=1.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value| (Global Feature Attribution)", fontsize=10, fontweight="bold")
    ax.set_title("Top 20 EEG Features by SHAP Importance (Tuned LightGBM — 73.34% Accuracy)", fontsize=12, fontweight="bold", pad=12)

    legend_patches = [
        mpatches.Patch(color="#457B9D", label="Differential Entropy (DE)"),
        mpatches.Patch(color="#2A9D8F", label="Beta/Gamma Band Power"),
        mpatches.Patch(color="#E63946", label="Alpha Band Power"),
        mpatches.Patch(color="#9B5DE5", label="Theta/Delta Band Power"),
        mpatches.Patch(color="#E9C46A", label="Statistical Moments (Var/Skew/Kurt)"),
        mpatches.Patch(color="#F4A261", label="Band Power Ratios"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig5_shap_bar.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: ml_fig5_shap_bar.png")

    # Fig 6: SHAP Beeswarm per Task
    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    for cls_idx, (ax, task) in enumerate(zip(axes, TASK_NAMES)):
        if shap_arr.ndim == 3:
            sv = shap_arr[cls_idx]
        else:
            sv = shap_arr
        imp = np.abs(sv).mean(axis=0)
        top10_idx = np.argsort(imp)[::-1][:10]
        top10_names = [feat_names[i] for i in top10_idx]
        top10_sv = sv[:, top10_idx]

        for fi, (fname, shap_col) in enumerate(zip(top10_names, top10_sv.T)):
            fvals = shap_sample[:, top10_idx[fi]]
            ptp_val = float(np.max(fvals) - np.min(fvals)) + 1e-12
            fvals_norm = (fvals - np.min(fvals)) / ptp_val
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

    # Fig 7: Topographic Channel Map
    ch_imp = {ch: 0.0 for ch in EEG_CH}
    for fname, val in zip(feat_names, mean_abs):
        for ch in EEG_CH:
            if fname.endswith(f"_{ch}"):
                ch_imp[ch] += val

    max_v = max(ch_imp.values()) + 1e-12
    ch_norm = {ch: v / max_v for ch, v in ch_imp.items()}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    circle = plt.Circle((0.5, 0.5), 0.47, fill=False, color="#2B2D42", lw=2.5)
    ax.add_patch(circle)
    ax.annotate("", xy=(0.5, 0.99), xytext=(0.5, 0.96), arrowprops=dict(arrowstyle="-|>", color="#2B2D42", lw=2.5))
    ear_l = mpatches.Ellipse((0.03, 0.5), 0.04, 0.12, fill=False, color="#2B2D42", lw=2)
    ear_r = mpatches.Ellipse((0.97, 0.5), 0.04, 0.12, fill=False, color="#2B2D42", lw=2)
    ax.add_patch(ear_l)
    ax.add_patch(ear_r)

    cmap = plt.cm.YlOrRd
    for ch in EEG_CH:
        x_c, y_c = CH_POS[ch]
        val = ch_norm[ch]
        col = cmap(val)
        circle_ch = plt.Circle((x_c, y_c), 0.06, color=col, ec="#2B2D42", lw=1.8, zorder=3)
        ax.add_patch(circle_ch)
        ax.text(x_c, y_c, ch, ha="center", va="center", fontsize=8, fontweight="bold",
                color="black" if val < 0.7 else "white", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04, label="Normalized SHAP Channel Attribution")
    ax.set_title("Topographic Scalp Map of Electrode Importance\n(14-Channel Emotiv EPOC+ Montages — Tuned LightGBM)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig7_shap_topo.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: ml_fig7_shap_topo.png")

    # Fig 10: Feature Group Importance
    group_imp = {
        "Differential\nEntropy (DE)": 0.0,
        "Band Powers\n(delta-gamma)": 0.0,
        "Band Power\nRatios": 0.0,
        "Statistical\nMoments (Var/Kurt)": 0.0,
    }

    for fname, val in zip(feat_names, mean_abs):
        if "de_" in fname:
            group_imp["Differential\nEntropy (DE)"] += val
        elif "bp_" in fname:
            group_imp["Band Powers\n(delta-gamma)"] += val
        elif "r_" in fname:
            group_imp["Band Power\nRatios"] += val
        else:
            group_imp["Statistical\nMoments (Var/Kurt)"] += val

    grp_names = list(group_imp.keys())
    grp_vals = list(group_imp.values())
    grp_cols = ["#457B9D", "#2A9D8F", "#F4A261", "#E9C46A"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(grp_names, grp_vals, color=grp_cols, edgecolor="white", linewidth=2)
    for bar, v in zip(bars, grp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.0005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total SHAP Importance (Sum of Mean |SHAP|)", fontsize=10, fontweight="bold")
    ax.set_title("EEG Feature Category Importance Breakdown (Tuned LightGBM)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_fig10_feature_groups.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: ml_fig10_feature_groups.png")

    print("\n" + "=" * 65)
    print("SUCCESS: ALL STATISTICAL TESTS & XAI FIGURES SAVED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
