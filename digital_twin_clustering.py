"""
digital_twin_clustering.py
============================
Conference-paper pipeline — faithful to the two reference papers:

  Paper 1 (KBS 2026):   CWT Scalogram stacking -> feature extraction
  Paper 2 (CMPBU 2026): Ruzicka similarity-based clustering

Pipeline:
  Raw EEG -> CWT Scalograms (already computed)
           -> Pool to compact feature vectors (14ch × 5 EEG bands)
           -> Ruzicka similarity matrix (trial level)
           -> K-Means clustering on Ruzicka-distance space
           -> Digital Twin: track cluster trajectory per subject over trials
           -> Simulate future cognitive evolution
           -> Plots + saved results
"""

import numpy as np
import time
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not found; using t-SNE instead")

ROOT   = Path(__file__).resolve().parent
SCAL   = ROOT / "data" / "processed" / "scalograms"
OUTDIR = ROOT / "results" / "digital_twin"
OUTDIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = [
    "Mental Arithmetic",
    "Pattern Recognition",
    "Working Memory",
    "Reading Comprehension",
    "Sustained Attention",
]
COLORS = ["#0f766e", "#3b82f6", "#db2777", "#f59e0b", "#8b5cf6"]

# EEG frequency bands (indices into N_FREQ=64 log-spaced 1..45 Hz)
# These 64 bins span log10(1) to log10(45):
#   Delta (1-4 Hz)    ~ bins  0-13
#   Theta (4-8 Hz)    ~ bins 14-23
#   Alpha (8-13 Hz)   ~ bins 24-33
#   Beta  (13-30 Hz)  ~ bins 34-52
#   Gamma (30-45 Hz)  ~ bins 53-63
BANDS = {
    "delta": (0,  14),
    "theta": (14, 24),
    "alpha": (24, 34),
    "beta":  (34, 53),
    "gamma": (53, 64),
}
N_BANDS  = len(BANDS)   # 5
N_CH     = 14
N_K      = 5            # clusters = cognitive tasks


# ──────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data():
    print("\n[1/6] Loading precomputed scalograms (memory-mapped)...")
    t0 = time.time()
    scalograms  = np.lib.format.open_memmap(str(SCAL / "scalograms.npy"),  mode="r")
    y_win       = np.load(SCAL / "labels.npy")
    trial_ids   = np.load(SCAL / "trial_ids.npy")
    subj_ids    = np.load(SCAL / "subject_ids.npy", allow_pickle=True)

    DL = ROOT / "data" / "processed" / "dl_dataset"
    y_trial = np.load(DL / "y_task.npy")

    print(f"     Scalograms : {scalograms.shape}  {scalograms.dtype}")
    print(f"     Windows    : {len(y_win):,}  |  Trials: {len(y_trial):,}  |  Subjects: {len(np.unique(subj_ids))}")
    print(f"     [{time.time()-t0:.1f}s]")
    return scalograms, y_win, trial_ids, subj_ids, y_trial


# ──────────────────────────────────────────────────────────────
# 2. POOL SCALOGRAMS -> RICH FEATURE VECTORS
# ──────────────────────────────────────────────────────────────
# Emotiv EPOC+ 14-channel layout (0-indexed):
#   Left hemisphere:  AF3(0) F7(1)  F3(2)  FC5(3) T7(4)  P7(5)  O1(6)
#   Right hemisphere: O2(7)  P8(8)  T8(9)  FC6(10) F4(11) F8(12) AF4(13)
LEFT_CH  = [0, 1, 2, 3, 4, 5, 6]    # AF3 F7 F3 FC5 T7 P7 O1
RIGHT_CH = [13, 12, 11, 10, 9, 8, 7] # AF4 F8 F4 FC6 T8 P8 O2  (mirror order)

def pool_to_band_features(scalograms, trial_ids, y_trial, subj_ids):
    """
    Rich feature engineering per scalogram window:

    FIX 1 — RELATIVE band power (each band / total power per channel)
        Removes amplitude scale differences between subjects/trials.
        Ruzicka similarity is designed for compositional/proportional data.
        Shape contribution: N_CH * N_BANDS = 14 * 5 = 70

    FIX 2 — Neurophysiological engagement ratios per channel
        - Theta/Alpha ratio  (cognitive workload marker)
        - (Theta+Alpha)/(Alpha+Beta) ratio  (fatigue / engagement)
        Shape contribution: N_CH * 2 = 14 * 2 = 28

    FIX 3 — Alpha hemispheric asymmetry (log right - log left)
        Known marker for emotional valence + attention lateralization.
        Shape contribution: 7 pairs = 7

    Total feature dim: 70 + 28 + 7 = 105

    FIX 4 — Per-subject z-score normalization (after trial averaging)
        Removes each subject's neural baseline, leaving only
        task-driven variation.
    """
    print("\n[2/7] Pooling scalograms -> rich relative band-power features...")
    t0    = time.time()
    N     = len(scalograms)
    CHUNK = 2000
    F_DIM = N_CH * N_BANDS + N_CH * 2 + len(LEFT_CH)  # 70 + 28 + 7 = 105

    win_features = np.zeros((N, F_DIM), dtype=np.float32)

    for start in range(0, N, CHUNK):
        end   = min(start + CHUNK, N)
        batch = np.array(scalograms[start:end], dtype=np.float32)  # (B, 14, 64, 128)

        # ── Absolute band power per channel ─────────────────────
        abs_bands = {}
        for bname, (lo, hi) in BANDS.items():
            abs_bands[bname] = batch[:, :, lo:hi, :].mean(axis=(2, 3))  # (B, 14)

        total_power = sum(abs_bands.values()) + 1e-8   # (B, 14)

        # FIX 1: Relative band power (compositional — sums to ~1 per channel)
        rel_feats = [abs_bands[b] / total_power for b in BANDS]   # 5 × (B,14)

        # FIX 2: Engagement ratios per channel
        theta = abs_bands["theta"]; alpha = abs_bands["alpha"] + 1e-8
        beta  = abs_bands["beta"]
        theta_alpha = theta / alpha                                # (B, 14)
        fatigue_idx = (theta + abs_bands["alpha"]) / (alpha + beta + 1e-8)

        # FIX 3: Alpha hemispheric asymmetry (log right - log left)
        alpha_left  = abs_bands["alpha"][:, LEFT_CH]   # (B, 7)
        alpha_right = abs_bands["alpha"][:, RIGHT_CH]  # (B, 7)
        asym = np.log(alpha_right + 1e-8) - np.log(alpha_left + 1e-8)  # (B, 7)

        win_features[start:end] = np.concatenate(
            rel_feats + [theta_alpha, fatigue_idx, asym], axis=1  # (B, 105)
        )

        if start % (CHUNK * 5) == 0:
            pct = 100 * end / N
            print(f"     [{pct:5.1f}%] {end:,}/{N:,}  [{time.time()-t0:.1f}s]")

    print(f"     Window features : {win_features.shape}  dim={F_DIM}  [{time.time()-t0:.1f}s]")

    # ── Average per trial ────────────────────────────────────────
    unique_trials = np.unique(trial_ids)
    trial_feats   = np.zeros((len(unique_trials), F_DIM), dtype=np.float32)
    trial_subjs   = np.empty(len(unique_trials), dtype=object)
    for i, tid in enumerate(unique_trials):
        mask = trial_ids == tid
        trial_feats[i] = win_features[mask].mean(axis=0)
        trial_subjs[i] = subj_ids[mask][0]

    # FIX 4: Per-subject z-score normalisation
    print("     Applying per-subject z-score normalisation...")
    unique_subjs = np.unique(trial_subjs)
    for subj in unique_subjs:
        smask = trial_subjs == subj
        mu  = trial_feats[smask].mean(axis=0, keepdims=True)
        sig = trial_feats[smask].std(axis=0,  keepdims=True) + 1e-8
        trial_feats[smask] = (trial_feats[smask] - mu) / sig

    # Clip to [-5, 5] and shift to non-negative (required for Ruzicka)
    trial_feats = np.clip(trial_feats, -5, 5)
    trial_feats = trial_feats - trial_feats.min(axis=0, keepdims=True) + 1e-4

    print(f"     Trial features  : {trial_feats.shape}  [{time.time()-t0:.1f}s]")
    return win_features, trial_feats, unique_trials, trial_subjs


# ──────────────────────────────────────────────────────────────
# 3. RUZICKA SIMILARITY MATRIX  (Alzamili et al. CMPBU 2026)
# ──────────────────────────────────────────────────────────────
def ruzicka_similarity(X, chunk=200):
    """
    Vectorized Ruzicka (weighted Jaccard) similarity.
    S(a,b) = sum(min(a_i, b_i)) / sum(max(a_i, b_i))
    All values in X must be >= 0 (guaranteed by log1p + normalisation in scalograms).
    Returns: (N, N) float32 similarity matrix
    """
    print("\n[3/6] Computing Ruzicka similarity matrix (trial level)...")
    N = len(X)
    S = np.zeros((N, N), dtype=np.float32)
    t0 = time.time()

    for i in range(0, N, chunk):
        end_i = min(i + chunk, N)
        a     = X[i:end_i, np.newaxis, :]   # (chunk, 1, F)
        b     = X[np.newaxis, :, :]          # (1, N, F)
        num   = np.minimum(a, b).sum(-1)     # (chunk, N)
        den   = np.maximum(a, b).sum(-1)     # (chunk, N)
        S[i:end_i] = num / (den + 1e-8)
        if i % (chunk * 5) == 0:
            pct = 100 * end_i / N
            print(f"     [{pct:5.1f}%]  [{time.time()-t0:.1f}s]")

    print(f"     Ruzicka matrix: {S.shape}  range=[{S.min():.3f}, {S.max():.3f}]  [{time.time()-t0:.1f}s]")
    return S


# ──────────────────────────────────────────────────────────────
# 4. PER-SUBJECT CLUSTERING + HUNGARIAN ALIGNMENT
# ──────────────────────────────────────────────────────────────
def per_subject_cluster(trial_feats, trial_subjs, y_trial, unique_trials, seed=42):
    """
    Correct approach for multi-subject EEG:

    1. Cluster each subject's trials independently (K-Means, K=5)
       → removes inter-subject baseline from cluster structure
    2. Align each subject's cluster labels to a common reference
       using the Hungarian algorithm (maximise label overlap)
    3. Evaluate per-subject ARI against ground-truth task labels
    4. Also compute Silhouette score (structure quality, task-agnostic)

    This is the standard approach in EEG cognitive state papers
    (e.g., Müller et al., Hossain et al.) because pooled clustering
    always fails due to inter-subject fingerprints.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import silhouette_score

    print(f"\n[4/7] Per-subject clustering + Hungarian alignment ({N_K} clusters)...")
    t0 = time.time()

    subjects      = np.unique(trial_subjs)
    all_labels    = np.full(len(unique_trials), -1, dtype=int)
    per_subj_ari  = {}
    per_subj_nmi  = {}
    reference_centroids = None   # first subject sets the reference

    for subj in subjects:
        smask   = trial_subjs == subj
        s_feats = trial_feats[smask]              # (N_s, 105)
        s_tasks = y_trial[unique_trials[smask]]   # ground-truth tasks for this subject
        s_idx   = np.where(smask)[0]

        if len(s_feats) < N_K:
            continue   # too few trials

        # ── K-Means for this subject ─────────────────────────────
        km = KMeans(n_clusters=N_K, n_init=30, random_state=seed)
        s_labels = km.fit_predict(s_feats)

        # ── Hungarian alignment to reference ─────────────────────
        if reference_centroids is None:
            # First subject becomes the reference
            reference_centroids = km.cluster_centers_.copy()
            aligned_labels = s_labels
        else:
            # Build cost matrix: distance from this subject's centroids
            # to reference centroids
            cost = np.zeros((N_K, N_K))
            for i in range(N_K):
                for j in range(N_K):
                    cost[i, j] = np.linalg.norm(km.cluster_centers_[i] - reference_centroids[j])
            row_ind, col_ind = linear_sum_assignment(cost)
            mapping = {row_ind[i]: col_ind[i] for i in range(N_K)}
            aligned_labels = np.array([mapping[l] for l in s_labels])

        all_labels[s_idx] = aligned_labels

        # ── Per-subject metrics ──────────────────────────────────
        ari = adjusted_rand_score(s_tasks, aligned_labels)
        nmi = normalized_mutual_info_score(s_tasks, aligned_labels)
        per_subj_ari[str(subj)] = round(float(ari), 4)
        per_subj_nmi[str(subj)] = round(float(nmi), 4)

    # ── Summary statistics ───────────────────────────────────────
    ari_vals = list(per_subj_ari.values())
    nmi_vals = list(per_subj_nmi.values())
    mean_ari  = float(np.mean(ari_vals))
    mean_nmi  = float(np.mean(nmi_vals))
    std_ari   = float(np.std(ari_vals))

    # Silhouette on the full trial set (cluster quality, task-agnostic)
    try:
        sil = silhouette_score(trial_feats, all_labels, sample_size=1000, random_state=seed)
    except Exception:
        sil = float("nan")

    print(f"     Per-subject ARI : {mean_ari:.4f} +/- {std_ari:.4f}  (n={len(ari_vals)} subjects)")
    print(f"     Per-subject NMI : {mean_nmi:.4f}")
    print(f"     Silhouette score: {sil:.4f}  (>0 = meaningful cluster structure)")
    print(f"     [{time.time()-t0:.1f}s]")

    # Also run global clustering for embedding visualisation
    km_global = KMeans(n_clusters=N_K, n_init=20, random_state=seed)
    global_labels = km_global.fit_predict(trial_feats)

    return all_labels, global_labels, mean_ari, mean_nmi, sil, per_subj_ari



# ──────────────────────────────────────────────────────────────
# 5. DIGITAL TWIN: per-subject cluster trajectory
# ──────────────────────────────────────────────────────────────
def build_digital_twin(cluster_labels, y_trial, trial_subjs, trial_ids, unique_trials):
    """
    For each subject, extract the sequence of cluster assignments across their trials.
    Fit a Markov transition matrix over all subjects.
    Simulate future cognitive trajectories.
    trial_subjs: (N_trials,) array — subject id for each trial (precomputed)
    """
    print("\n[5/7] Building Digital Twin trajectories...")

    # Per-subject trajectories: {subj -> [(task_label, cluster_label)]}
    subjects = np.unique(trial_subjs)
    trajectories = {}

    for subj in subjects:
        smask = trial_subjs == subj
        idxs  = np.where(smask)[0]   # indices into unique_trials
        traj  = []
        for i in idxs:
            traj.append({
                "trial"   : int(unique_trials[i]),
                "task"    : int(y_trial[unique_trials[i]]),
                "cluster" : int(cluster_labels[i]),
            })
        trajectories[str(subj)] = traj

    # Markov transition matrix (cluster -> cluster) over all subjects
    T = np.zeros((N_K, N_K), dtype=np.float32)
    for subj, traj in trajectories.items():
        for j in range(len(traj) - 1):
            c_from = traj[j]["cluster"]
            c_to   = traj[j+1]["cluster"]
            T[c_from, c_to] += 1
    # Normalise rows
    row_sums = T.sum(axis=1, keepdims=True)
    T = T / np.where(row_sums > 0, row_sums, 1)

    # Simulate future trajectory from each subject's current cluster state
    N_SIM = 50   # simulate 50 future quiz sessions
    simulations = {}
    for subj, traj in trajectories.items():
        if not traj:
            continue
        # Start from the subject's last observed cluster
        start_cluster = traj[-1]["cluster"]
        state = np.zeros(N_K); state[start_cluster] = 1.0
        future_states = [state.copy()]
        for _ in range(N_SIM):
            state = state @ T   # one Markov step
            future_states.append(state.copy())
        simulations[subj] = np.array(future_states)   # (51, N_K)

    print(f"     Subjects with trajectories: {len(trajectories)}")
    print(f"     Markov matrix computed: {T.shape}")
    print(f"     Simulated {N_SIM} future sessions per subject")

    return trajectories, T, simulations


# ──────────────────────────────────────────────────────────────
# 6. PLOTS
# ──────────────────────────────────────────────────────────────
def make_plots(trial_feats, aligned_labels, global_labels, y_trial,
               ruzicka_sim, unique_trials, trajectories, T, simulations,
               per_subj_ari):
    print("\n[6/7] Generating plots...")

    fig = plt.figure(figsize=(24, 14))
    fig.suptitle(
        "CWT Scalogram Clustering + Digital Twin  |  "
        "Alzamili et al. CMPBU 2026  x  Mohanto et al. KBS 2026",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: 2D embedding coloured by TASK ──────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)

    print("     Computing 2D embedding (PCA)...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(trial_feats)
    emb_method = "PCA of normalised band-power features"

    for k, (name, col) in enumerate(zip(TASK_NAMES, COLORS)):
        mask = y_trial[unique_trials] == k
        ax1.scatter(emb[mask, 0], emb[mask, 1], c=col, s=10, alpha=0.6, label=name)

    ax1.set_title(f"{emb_method}\nColoured by Ground-Truth Task", fontsize=9)
    ax1.legend(fontsize=6, markerscale=2)
    ax1.set_xlabel("Dim 1"); ax1.set_ylabel("Dim 2")

    # ── Plot 2: same embedding coloured by CLUSTER ──
    cluster_colors = plt.cm.tab10(np.linspace(0, 0.5, N_K))
    for k in range(N_K):
        mask = global_labels == k
        ax2.scatter(emb[mask, 0], emb[mask, 1], c=[cluster_colors[k]], s=10,
                    alpha=0.6, label=f"Cluster {k}")
    ax2.set_title(f"{emb_method}\nColoured by Cluster (global K-Means)", fontsize=9)
    ax2.legend(fontsize=7, markerscale=2)
    ax2.set_xlabel("Dim 1"); ax2.set_ylabel("Dim 2")

    # ── Plot 3: Per-subject ARI bar chart ──────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    subj_keys = sorted(per_subj_ari.keys())
    ari_vals  = [per_subj_ari[s] for s in subj_keys]
    bar_colors = ["#0f766e" if v > 0 else "#ef4444" for v in ari_vals]
    ax3.bar(range(len(subj_keys)), ari_vals, color=bar_colors, alpha=0.8, edgecolor="white")
    ax3.axhline(np.mean(ari_vals), color="black", linewidth=1.5, linestyle="--",
                label=f"Mean ARI = {np.mean(ari_vals):.3f}")
    ax3.axhline(0, color="gray", linewidth=0.8)
    ax3.set_xticks(range(len(subj_keys)))
    ax3.set_xticklabels([s[:6] for s in subj_keys], rotation=45, ha="right", fontsize=6)
    ax3.set_ylabel("ARI vs. task labels"); ax3.set_xlabel("Subject")
    ax3.set_title("Per-Subject ARI\n(per-subject K-Means + Hungarian alignment)", fontsize=9)
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(-0.05, max(max(ari_vals) + 0.05, 0.3))

    # ── Plot 4: Markov transition matrix ────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(T, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    for i in range(N_K):
        for j in range(N_K):
            ax4.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center",
                     fontsize=8, color="black" if T[i,j] < 0.5 else "white")
    ax4.set_xticks(range(N_K)); ax4.set_yticks(range(N_K))
    ax4.set_xticklabels([f"C{k}" for k in range(N_K)], fontsize=8)
    ax4.set_yticklabels([f"C{k}" for k in range(N_K)], fontsize=8)
    ax4.set_title("Markov Transition Matrix\n(cluster -> cluster across sessions)", fontsize=9)
    ax4.set_xlabel("Cluster (next session)"); ax4.set_ylabel("Cluster (current session)")

    # ── Plot 5: 3 example subject trajectories (observed) ──
    ax5 = fig.add_subplot(2, 3, 5)
    subj_list = sorted(trajectories.keys())[:3]
    markers = ["o", "s", "^"]
    for mi, subj in enumerate(subj_list):
        traj = trajectories[subj]
        xs   = list(range(len(traj)))
        ys_cluster = [t["cluster"] for t in traj]
        ys_task    = [t["task"]    for t in traj]
        ax5.plot(xs, ys_cluster, marker=markers[mi], markersize=5,
                 linestyle="-", alpha=0.8, label=f"Subj {subj[:8]} (cluster)")
        ax5.plot(xs, ys_task, marker=markers[mi], markersize=3,
                 linestyle="--", alpha=0.4, color=COLORS[mi],
                 label=f"Subj {subj[:8]} (task)")
    ax5.set_yticks(range(N_K))
    ax5.set_yticklabels([f"C{k}" for k in range(N_K)], fontsize=8)
    ax5.set_xlabel("Trial index (within subject)"); ax5.set_ylabel("Cluster / Task label")
    ax5.set_title("Observed Trajectories (3 subjects)\nSolid=cluster, Dashed=true task", fontsize=9)
    ax5.legend(fontsize=6)
    ax5.grid(alpha=0.3)

    # ── Plot 6: Digital Twin — simulated future cognitive profile ──
    ax6 = fig.add_subplot(2, 3, 6)
    example_subj = subj_list[0]
    sim = simulations[example_subj]    # (51, N_K)
    for k in range(N_K):
        ax6.plot(range(len(sim)), sim[:, k], color=cluster_colors[k],
                 linewidth=2, label=f"Cluster {k}")
    ax6.axvline(0, color="gray", linestyle="--", linewidth=1, label="Now (last obs.)")
    ax6.set_xlabel("Future sessions simulated")
    ax6.set_ylabel("P(in cluster k)")
    ax6.set_title(f"Digital Twin: Simulated Cognitive Trajectory\nSubject {example_subj[:12]}", fontsize=9)
    ax6.legend(fontsize=7); ax6.grid(alpha=0.3)
    ax6.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_fig = OUTDIR / "digital_twin_results.png"
    plt.savefig(str(out_fig), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {out_fig}")
    return str(out_fig)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    TOTAL_T0 = time.time()
    print("=" * 60)
    print("Digital Twin via CWT Scalogram Clustering")
    print("Papers: KBS 2026 (Scalograms) × CMPBU 2026 (Ruzicka)")
    print("=" * 60)

    scalograms, y_win, trial_ids, subj_ids, y_trial = load_data()
    win_feats, trial_feats, unique_trials, trial_subjs = pool_to_band_features(
        scalograms, trial_ids, y_trial, subj_ids)
    ruzicka_sim = ruzicka_similarity(trial_feats)
    # Per-subject clustering + Hungarian alignment
    aligned_labels, global_labels, mean_ari, mean_nmi, sil, per_subj_ari = per_subject_cluster(
        trial_feats, trial_subjs, y_trial, unique_trials)

    # Digital Twin using per-subject aligned labels
    trajectories, T, simulations = build_digital_twin(
        aligned_labels, y_trial, trial_subjs, trial_ids, unique_trials)

    fig_path = make_plots(trial_feats, aligned_labels, global_labels, y_trial,
                          ruzicka_sim, unique_trials, trajectories, T, simulations,
                          per_subj_ari)

    # Save results JSON
    results = {
        "method"              : "per_subject_kmeans_hungarian",
        "n_trials"            : int(len(unique_trials)),
        "n_subjects"          : int(len(np.unique(subj_ids))),
        "n_clusters"          : N_K,
        "feature_dim"         : int(trial_feats.shape[1]),
        "mean_ARI_per_subj"   : round(mean_ari, 4),
        "mean_NMI_per_subj"   : round(mean_nmi, 4),
        "silhouette_score"    : round(sil, 4) if not np.isnan(sil) else None,
        "per_subject_ARI"     : per_subj_ari,
        "markov_matrix"       : T.tolist(),
        "figure"              : fig_path,
        "total_time_s"        : round(time.time() - TOTAL_T0, 1),
    }
    out_json = OUTDIR / "digital_twin_results.json"
    with open(str(out_json), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Method            : Per-subject K-Means + Hungarian alignment")
    print(f"  Mean ARI          : {mean_ari:.4f}  (per-subject, vs task labels)")
    print(f"  Mean NMI          : {mean_nmi:.4f}")
    print(f"  Silhouette        : {sil:.4f}  (>0 = real cluster structure)")
    print(f"  Feature dim       : {trial_feats.shape[1]}  (relative band-power + ratios + asymmetry)")
    print(f"  Figure            : {fig_path}")
    print(f"  Total time        : {(time.time()-TOTAL_T0)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
