# Master Research Prompt — Step 3
## Topic: Clustering + Scalogram Stacking + Digital Twin Integration for EEG Cognitive Profiling

> **Context:** This prompt is for autonomous AI research agents (Consensus, ChatGPT, Claude, Gemini, Perplexity). Each agent should conduct independent deep research and return a thorough, citation-rich technical report.
> **Do not summarize. Be exhaustive. Cite specific papers, authors, years, results, and formulas.**

---

## Project Background (Read Before Researching)

We are building **CogProfile-Net**, an end-to-end EEG-based neural cognitive profiling system. The pipeline is:

```
Raw EEG (Emotiv EPOC+, 14 channels, 128 Hz)
   → Preprocessing (CAR, 0.5–45 Hz bandpass, CQ gating)
   → Cognitive state classification across 5 tasks:
        Class 0: Mental Arithmetic
        Class 1: Pattern Recognition
        Class 2: Working Memory
        Class 3: Reading Comprehension
        Class 4: Sustained Attention
   → Calibrated cognitive profile (probabilistic, not hard label)
   → Career aptitude mapping via CHC theory + O*NET database
```

**Current baseline:** 73.34% test accuracy (Tuned LightGBM on 252 handcrafted PSD/DE features, subject-dependent evaluation, ~25 subjects, 37,804 sub-windows).

**Dataset constraints:**
- Emotiv EPOC+ (14 electrodes, 128 Hz, consumer-grade, NOT clinical)
- Electrodes: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- ~25 subjects, ~2,908 trials (4 seconds each), 37,804 sliding 1-second sub-windows
- Behavioral metadata per trial: reaction time, difficulty level, score/accuracy

**Two newly confirmed techniques we want to integrate:**
1. **Unsupervised clustering** as a subject stratification / pseudo-label enrichment mechanism before classification
2. **CWT Scalogram Stacking**: Convert each EEG channel to a 2D scalogram via Continuous Wavelet Transform, stack N=14 scalograms → (14, F, T) tensor → feed into 2D CNN

**New idea we want to explore:**
3. **EEG Digital Twin**: A generative model that simulates how a person''s EEG-derived cognitive profile would evolve over repeated testing sessions (e.g., 100–200 quiz sessions), enabling cognitive trajectory prediction and what-if career pathway exploration.

---

## Research Questions to Answer

### BLOCK A — Clustering for EEG Cognitive Profiling

**A1.** What is the current state-of-the-art for unsupervised/semi-supervised clustering applied to EEG signals for cognitive profiling, intelligence assessment, or mental state classification?
- Which clustering algorithms have been validated on small-N EEG datasets (N < 50 subjects)?
- K-Means vs. Gaussian Mixture Models vs. Hierarchical vs. DBSCAN vs. Spectral Clustering — which works best for EEG spectral features and why?
- What features should be clustered: raw PSD vectors, covariance matrices on the Riemannian manifold (SPD clustering), topographic maps, or learned embeddings?

**A2.** How should cluster assignments be used in a downstream deep learning pipeline?
- Method 1: Pseudo-label enrichment — append cluster ID as additional class-conditional feature
- Method 2: Subject stratification — train separate classifiers per cluster
- Method 3: Contrastive clustering — use cluster assignments as auxiliary self-supervised signal during DL training
- Method 4: Prototypical networks — use cluster centroids as class prototypes for few-shot-style classification
- Which of these methods has been empirically validated for EEG? Cite papers.

**A3.** Riemannian SPD Clustering — is there established literature on clustering covariance matrices on the SPD manifold for EEG?
- What distance metrics are used (Riemannian, log-Euclidean, affine-invariant)?
- Has K-Means on the Riemannian manifold (Karcher mean-based K-Means) been applied to EEG classification?
- Can clustering on the SPD manifold serve as a better subject grouping mechanism than Euclidean K-Means on PSD vectors?

**A4.** What are the optimal cluster validation metrics for small EEG datasets?
- Silhouette coefficient, Davies-Bouldin index, Calinski-Harabász index, Gap Statistic
- How do you determine the optimal number of clusters k when N=25 subjects?
- Can cluster stability across sessions be used as a measure of "cognitive trait" consistency?

---

### BLOCK B — Scalogram Stacking for EEG Deep Learning

**B1.** What is the current best practice for CWT scalogram generation from EEG signals?
- Wavelet choice: Morlet (most common), Paul, DOG, Mexican Hat — which is optimal for EEG rhythms (delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz)?
- Optimal scales/frequencies: How many frequency bins are needed to capture all 5 EEG bands adequately?
- Time resolution vs. frequency resolution trade-off in CWT for 128 Hz EEG: what is the practical minimum epoch length?
- How should the scalogram be normalized before stacking (per-channel Z-score, global min-max, log-power)?

**B2.** Scalogram stacking architectures — what CNN/hybrid architectures have been used with stacked multi-channel scalograms?
- 2D CNN baselines: ResNet-18, EfficientNet, custom lightweight CNNs
- Hybrid approaches: CNN feature extractor + LSTM/GRU temporal head
- Attention mechanisms over the stacked scalogram (channel attention, temporal attention, frequency attention)
- What is the computational cost of processing (14, F, T) scalogram stacks vs. raw time series for 128 Hz EEG?
- Best results reported in literature for scalogram-based EEG classification on similar consumer-grade devices

**B3.** Scalogram stacking + Riemannian geometry — is there a way to combine both approaches?
- Can a CNN process scalogram stacks while a parallel branch processes covariance matrices on the SPD manifold?
- Is there literature on combining time-frequency representations with geometric deep learning for EEG?
- How would features from both branches be fused (concatenation, cross-attention, gated fusion)?

**B4.** Data augmentation for scalogram-based EEG on small datasets:
- Time warping, frequency shifting, electrode noise injection in the scalogram domain
- Mixup / CutMix applied to scalogram images
- What augmentation strategies have been validated for small-N EEG (< 50 subjects) scalogram models?
- Does augmenting in the scalogram domain provide more benefit than augmenting in the time domain?

---

### BLOCK C — EEG Digital Twin for Cognitive Trajectory Modeling

**C1.** What is an "EEG Digital Twin" and what exists in the literature?
- Search specifically for: "EEG digital twin", "brain digital twin", "neural digital twin", "cognitive simulation"
- What generative/simulation approaches have been used to model EEG signal evolution over time?
- Distinguish between:
  - Signal-level twins (simulate raw EEG signal evolution)
  - State-level twins (simulate latent cognitive state transitions)
  - Profile-level twins (simulate how a probabilistic cognitive profile changes over sessions)
- Which level is most feasible for our dataset (25 subjects, 5 cognitive classes, behavioral metadata)?

**C2.** Generative models for cognitive state simulation — what techniques are validated?
- VAE: Learn a latent cognitive state space; sample trajectories through latent space
- State Space Models (SSMs): Kalman filter, Linear Dynamical Systems, neural SSMs (Mamba, S4) for modeling cognitive state evolution as a function of practice/fatigue/time
- Gaussian Processes (GP): Model cognitive profile evolution as a GP with session index as input
- Score-based diffusion models: Generate plausible future EEG scalograms conditioned on current cognitive state and "sessions practiced" variable
- Which of these has been demonstrated for EEG trajectory simulation? Cite specifically.

**C3.** What is the neuroscience/cognitive science grounding for cognitive trajectory modeling?
- Does repeated practice of the 5 cognitive tasks produce measurable EEG changes over time?
- What does the learning curve literature say about brain oscillation changes with practice?
- How should we model the "direction" of cognitive evolution for each of the 5 tasks independently?
- Is there a validated cognitive load theory framework for predicting EEG change over sessions?

**C4.** Practical Digital Twin design for our system — what is the minimal viable approach?
- Input: Current cognitive profile (Dir(alpha) parameters), number of completed sessions, behavioral performance trajectory (accuracy, RT over sessions)
- Output: Predicted cognitive profile at session N+k, with uncertainty bounds
- What model class is most appropriate? Neural ODE, Gaussian Process regression, Recurrent State Space Model, or Bayesian updating over Dirichlet posteriors?
- How would the Digital Twin output connect to the career mapping layer?
- What are the key failure modes and limitations?

**C5.** Validation of the Digital Twin:
- How do you validate a simulation model without longitudinal ground truth?
- Methods: held-out temporal splits, leave-one-session-out cross-validation, calibration of uncertainty bounds
- Can the Digital Twin be validated against known neuroscience findings as face validity?
- What metrics should be reported: MSE of predicted profile, calibration of uncertainty, trajectory shape correlation?

---

### BLOCK D — Integration Architecture

**D1.** Propose a concrete, justified pipeline integrating:
1. CWT Scalogram Stacking → 2D CNN encoder
2. Riemannian SPD encoder (parallel branch from covariance matrices)
3. Clustering-based subject stratification (pre-training OR auxiliary contrastive loss)
4. Evidential Deep Learning (EDL) Dirichlet head
5. Digital Twin for cognitive trajectory simulation

**D2.** Multi-branch fusion design for scalogram CNN + Riemannian SPD dual-encoder:
- What fusion strategies are optimal (concatenation, cross-modal attention, gated unit)?
- How do you prevent one branch from dominating during training (gradient balancing)?

**D3.** Training strategy:
- Stage 1: Pre-train clustering module on all subjects for stable cluster assignments
- Stage 2: Train CNN (scalogram) + SPD (Riemannian) dual encoder with cluster-aware contrastive loss
- Stage 3: Fine-tune EDL head
- Stage 4: Post-training: fit Digital Twin on session-level aggregated profiles

**D4.** Novelty assessment:
- Does this integrated system constitute a sufficient novel contribution for IEEE T-NSRE, JNE, or Neural Networks?
- Is there any prior work combining all four components? How does ours differ?
- What is the strongest single-sentence novel claim?

---

### BLOCK E — Practical Constraints & Feasibility

**E1.** Feasibility on our hardware (AMD Ryzen 5000, 12 GB RAM, no dedicated GPU):
- CWT scalogram generation for 37,804 windows × 14 channels — memory footprint?
- Can scalogram-based CNN be trained on CPU in reasonable time?
- Lightweight CNN alternatives for 14-channel EEG scalograms

**E2.** Open-source implementations to look for:
- CWT: PyWavelets, torch-wavelets, ssqueezepy
- Riemannian geometry: geoopt, geomstats, pyriemann
- Digital Twin / Neural ODE: torchdiffeq, diffrax (JAX)
- EDL: evidentorch, custom pytorch
- Any complete EEG + scalogram + CNN pipelines on GitHub?

**E3.** Biggest risks:
- Overfitting risk: 37,804 samples, small N=25 subjects
- Scalogram dimensionality explosion: (14, F, T) tensor size for 128 Hz, 1-second windows
- Digital Twin data requirements: Is N=25 subjects enough for a generative model?
- Cluster instability with N=25

---

## Expected Deliverable Format

```
## [Your Agent Name] Research Report — Step 3

### A. Clustering for EEG Cognitive Profiling
[Answers to A1–A4 with citations]

### B. Scalogram Stacking for EEG Deep Learning
[Answers to B1–B4 with citations]

### C. EEG Digital Twin for Cognitive Trajectory
[Answers to C1–C5 with citations]

### D. Integration Architecture
[Answers to D1–D4 with citations]

### E. Practical Constraints
[Answers to E1–E3 with citations]

### Summary: Top 3 Recommended Directions
[Agent''s top 3 specific, actionable recommendations for our exact dataset and constraints]

### Key Citations
[All papers cited with title, authors, venue, year]
```

**Quality bar:** Do NOT hallucinate citations. Say explicitly if a specific paper cannot be found. Prefer 2020–2026. Prioritize papers with code available.

---

## Grounding Notes for the Agent

- 14 electrodes, 128 Hz, N=25 subjects — solutions for 128-channel clinical arrays may not scale down. Always flag montage mismatch.
- Subject-dependent evaluation is current paradigm. Cross-subject is desirable but secondary.
- The career mapping layer (CHC → O*NET) is already designed. Research needed is on signal processing and modeling side.
- The Digital Twin is a NEW idea. A validated framework design + feasibility analysis is sufficient — full implementation is not required for the paper.
- Priority order of novelty: Digital Twin > Scalogram + Riemannian dual-branch > Clustering integration > EDL
