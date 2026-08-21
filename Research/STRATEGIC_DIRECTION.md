# Strategic Research Direction: From EEG Classification to Neural Cognitive Profiling

> **Document Purpose:** This is the master strategic and technical direction document for the Major Project. It synthesizes all Step 1 and Step 2 research findings, critically evaluates proposed architectures, identifies the true novel contribution, and lays out a concrete, grounded path forward.
>
> **Audience:** Mentor, teammates, and implementation reference.
>
> **Last Updated:** August 2026

---

## Table of Contents

1. [Where We Stand Today](#1-where-we-stand-today)
2. [The Honest Problem: What's Missing](#2-the-honest-problem-whats-missing)
3. [The End-to-End Vision](#3-the-end-to-end-vision)
4. [Critical Evaluation of Proposed Architectures](#4-critical-evaluation-of-proposed-architectures)
5. [The Real Novel Contribution](#5-the-real-novel-contribution)
6. [Proposed Architecture: CogProfile-Net](#6-proposed-architecture-cogprofile-net)
7. [The Cognitive-to-Career Mapping Layer](#7-the-cognitive-to-career-mapping-layer)
8. [Validation Strategy](#8-validation-strategy)
9. [Ethical and Legal Constraints](#9-ethical-and-legal-constraints)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [References](#11-references)

---

## 1. Where We Stand Today

### What We Have Built

| Component | Status | Details |
|:---|:---:|:---|
| Data collection pipeline | ✅ Done | Emotiv EPOC+ (14ch, 128 Hz) + synchronized web-based cognitive tasks |
| Preprocessing | ✅ Done | CAR, bandpass (0.5–45 Hz), CQ gating |
| Sub-windowing augmentation | ✅ Done | 2,908 trials → 37,804 × 1-second windows (75% overlap) |
| Feature engineering | ✅ Done | 252 features: PSD, DE, band ratios, statistical moments |
| ML baseline | ✅ Done | 73.34% test accuracy (Tuned LightGBM, 5-class, subject-dependent) |
| XAI & statistical validation | ✅ Done | TreeSHAP, permutation test (p<0.001), bootstrap CIs, McNemar's |
| Deep learning architecture | ❌ Not started | This document defines the direction |
| Career/aptitude mapping | ❌ Not started | Requires both DL output design and psychometric grounding |

### The Dataset at a Glance

```
Hardware:           Emotiv EPOC+ (14 channels, 128 Hz, consumer-grade)
Subjects:           ~25 participants
Raw trials:         2,908 synchronized 4-second epochs
Augmented samples:  37,804 sliding 1-second sub-windows
Classes:            5 cognitive tasks (chance = 20%)
Behavioral data:    Reaction time, difficulty level, accuracy per trial
```

### The 5 Cognitive Classes and Their Neural Signatures

| Class | Cognitive Task | Primary Brain Region | Key EEG Signature |
|:---:|:---|:---|:---|
| 0 | Mental Arithmetic | DLPFC, Parietal | Frontal θ sync, β elevation |
| 1 | Pattern Recognition | Occipital, Parietal | Occipital α desync, γ bursts |
| 2 | Working Memory | Frontal Midline | Frontal Midline Theta (FMT) |
| 3 | Reading Comprehension | Left Temporal, Broca's | Left θ/β modulation |
| 4 | Sustained Attention | ACC, Frontal | α suppression, β/θ ratio |

---

## 2. The Honest Problem: What's Missing

### The Mentor's Feedback (Verbatim Summary)

> *"The project is highly innovative in concept. The whole idea is very good. But it's missing that one pinnacle or peak important thing that makes it stand out. Nothing is innovative or novel in the technique itself — it's very plain. We need to make something out of this."*

### Why the Current Work Doesn't Stand Out Yet

**What we did (ML baseline):**
- Standard PSD + Differential Entropy features → LightGBM → hard 5-class label
- This is a well-trodden path. Dozens of papers do exactly this.

**What reviewers at IEEE T-NSRE or JNE would ask:**
1. *"Your features are standard. What is architecturally novel?"*
2. *"You used subject-dependent splits. Does this generalize?"*
3. *"You output a hard class label. How is that useful beyond a research benchmark?"*
4. *"What can someone actually DO with this classification result?"*

That last question is the critical one. And it leads directly to the real innovation.

---

## 3. The End-to-End Vision

### The Actual Goal (Revealed by User)

This project is **not** just a classifier. It is a **Neural Cognitive Profiling System** with a downstream application:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         THE FULL PIPELINE                              │
│                                                                        │
│  STAGE 1: Data Acquisition                                            │
│  ┌──────────────┐    ┌──────────────┐                                 │
│  │ Emotiv EPOC+ │    │  Web-Based   │                                 │
│  │  14ch EEG    │───▶│ Cognitive    │──▶ Synchronized trial dataset    │
│  │  128 Hz      │    │ Test Battery │                                 │
│  └──────────────┘    └──────────────┘                                 │
│                                                                        │
│  STAGE 2: Neural Cognitive Profiling (THE CORE INNOVATION)            │
│  ┌──────────────────────────────────────────────────┐                 │
│  │  Raw EEG (14, 128) per sub-window                │                 │
│  │        ↓                                         │                 │
│  │  Deep Learning Encoder                           │                 │
│  │  (Riemannian geometry-aware, subject-invariant)  │                 │
│  │        ↓                                         │                 │
│  │  Evidential Output Head (Dirichlet distribution) │                 │
│  │        ↓                                         │                 │
│  │  Calibrated Cognitive Profile:                   │                 │
│  │  {Math: 15% ± 3%, Pattern: 30% ± 5%,           │                 │
│  │   Memory: 35% ± 4%, Reading: 10% ± 2%,         │                 │
│  │   Attention: 10% ± 2%}                          │                 │
│  │  + Uncertainty mass u = 0.12 (confident)         │                 │
│  └──────────────────────────────────────────────────┘                 │
│                                                                        │
│  STAGE 3: Career Aptitude Mapping (APPLICATION LAYER)                 │
│  ┌──────────────────────────────────────────────────┐                 │
│  │  Cognitive Profile (Dirichlet α parameters)      │                 │
│  │        ↓                                         │                 │
│  │  CHC Cognitive Ability Alignment                  │                 │
│  │  (Psychometrically grounded mapping)             │                 │
│  │        ↓                                         │                 │
│  │  O*NET Occupational Requirements Matrix           │                 │
│  │  (900+ occupations × ability requirements)       │                 │
│  │        ↓                                         │                 │
│  │  Bayesian Career Fit Scores with CIs:            │                 │
│  │  "STEM Engineering: 78% ± 8% fit"               │                 │
│  │  "Data Analysis: 72% ± 6% fit"                  │                 │
│  │  "Creative Design: 41% ± 12% fit"               │                 │
│  └──────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This End-to-End System IS the Innovation

No existing published system does all three stages together:
- **Traditional aptitude tests** (Holland RIASEC, ASVAB, GATB) rely on self-reported questionnaires — not objective neural measurement.
- **Commercial cognitive assessment** (Pymetrics, Cognify) uses gamified behavioral tasks but has no neural signal component and no published validation.
- **EEG classification papers** stop at accuracy benchmarks and never connect to real-world career utility.

**Our contribution spans the gap:** Direct neural signal → Calibrated probabilistic cognitive profile → Grounded career recommendation with uncertainty.

---

## 4. Critical Evaluation of Proposed Architectures

### The "Tri-S²Former" Problem

The Tri-S²Former (proposed earlier) had three branches: Dynamic Graph, SPDNet, and Temporal Conformer. Here is an honest evaluation:

| Branch | Claim | Reality Check | Verdict |
|:---|:---|:---|:---:|
| **Dynamic Graph Attention** | "Captures functional connectivity between 14 electrode nodes" | With only 14 nodes, the adjacency matrix is 14×14 = 196 learnable parameters. This is too small for meaningful graph learning and risks overfitting the edge weights. Graph Neural Networks are designed for hundreds-to-thousands of nodes. At 14 nodes, a simple attention mechanism achieves the same spatial modeling without the graph abstraction overhead. | ⚠️ Unjustified complexity |
| **Riemannian SPD Manifold** | "Covariance matrices lie on SPD manifold; geodesic metrics strip inter-subject noise" | This is mathematically correct and well-supported by 2024–2026 literature (Tibermacine et al., Jin et al., Gao & Deng). 14×14 covariance matrices are the *right* dimensionality — stable to estimate, small enough for efficient manifold operations. The BiMap/ReEig/LogEig pipeline is computationally feasible. | ✅ **Genuinely justified** |
| **Temporal Conformer + Wavelet** | "Multi-scale temporal modeling of neuro-oscillations" | Every EEG deep learning paper does temporal modeling. Calling it a "wavelet filterbank + Conformer" adds naming novelty but not conceptual novelty. No specific argument for *why* multi-scale helps for these 5 cognitive tasks vs. a standard temporal convolution. | ❌ Generic |

**The fundamental issue:** Three branches multiplied the parameter count by ~3×, drastically increasing overfitting risk on 37,804 samples, without a clear ablation argument that each branch contributes independently. The architecture was engineer-optimized (stack everything), not researcher-optimized (justify everything).

### What the Three Research Reports Actually Agree On

After reading all 8 research documents (4 from Step 1, 4 from Step 2), here is what the literature converges on for our exact constraint set:

| Finding | Consensus Strength | Sources |
|:---|:---:|:---|
| **Riemannian/SPD methods provide genuine inter-subject robustness** | Strong (all 4 reports agree) | Tibermacine 2026, Jin 2026, Gao & Deng 2026, Kim 2023 |
| **Evidential Deep Learning enables calibrated uncertainty** | Strong (all Step 2 reports) | Sensoy 2018, Wu (TRUEE) 2025, Pandey 2025 |
| **Deep Ensembles are the strongest practical UQ benchmark** | Strong | Tveter 2026, Manivannan 2024, Lakshminarayanan 2017 |
| **Graph learning at 14 nodes has limited benefit** | Moderate (implicitly acknowledged) | Tang 2024 uses 62+ channels; Ding 2025 uses 32+ |
| **CHC theory is the validated psychometric ground truth** | Strong (all Step 2 reports) | McGrew 2009, Keith & Reynolds 2010, Flanagan & Dixon 2014 |
| **O*NET provides the occupational mapping database** | Strong | Rahim & Basheer 2025, Alonso et al. 2025 |
| **Direct "brain-to-career" claims lack empirical validity** | Strong caution | GATB meta-analyses, Gottfredson 2003, Nye 2022 |
| **System must be positioned as advisory decision-support** | Strong (legal/ethical) | EU AI Act 2024, EEOC 2024, Muhl & Andorno 2023 |

---

## 5. The Real Novel Contribution

### The Three-Layer Novelty Stack

Instead of claiming novelty through architectural complexity, we claim novelty through **a coherent, end-to-end system that no prior work has demonstrated:**

#### Layer 1: Riemannian Geometry-Aware EEG Encoding (Methodological Novelty)
- **What:** Replace handcrafted PSD/DE features with learned spatial covariance representations on the SPD manifold.
- **Why it's justified:** The 14×14 covariance matrix is naturally SPD. Riemannian metrics are mathematically invariant to the linear transformations caused by skull thickness, electrode impedance, and cortical geometry differences — our primary noise source.
- **What it improves:** Eliminates the manual feature engineering bottleneck and provides fundamentally more robust representations than Euclidean neural networks.

#### Layer 2: Evidential Uncertainty-Calibrated Classification (Architectural Novelty)
- **What:** Replace standard softmax with a Dirichlet distribution head that outputs calibrated probabilities AND explicit uncertainty mass.
- **Why it's justified:** When the output feeds into a career recommendation, the difference between "confident: 80% working memory" and "uncertain: maybe 50-80% working memory" is the difference between a useful system and a dangerous one. No EEG cognitive classification paper has used EDL for aptitude profiling.
- **What it enables:** The system can say "I don't know" (high vacuity $u$) instead of giving a spurious confident answer.

#### Layer 3: Psychometrically-Grounded Career Mapping (System Novelty)
- **What:** Map the Dirichlet cognitive profile through CHC cognitive ability theory into O\*NET occupational requirements, producing career fit scores with Bayesian credible intervals.
- **Why it's justified:** CHC theory is the most validated cognitive ability taxonomy in psychometrics (70+ years of research). O\*NET is the US Department of Labor's standardized occupational database with ability requirements for 900+ occupations.
- **What makes it novel:** No prior system connects real-time EEG classification → CHC abilities → O\*NET career mapping with uncertainty propagation.

### The Paper Title

**"CogProfile-Net: Uncertainty-Aware Neural Cognitive Profiling via Riemannian EEG Encoding with Psychometrically-Grounded Career Aptitude Mapping"**

This title signals: (1) a complete system, (2) uncertainty awareness, (3) geometric deep learning, (4) psychometric grounding — all genuinely novel in combination.

---

## 6. Proposed Architecture: CogProfile-Net

### Design Philosophy

Every component must have a **clear, defensible justification** tied to a specific problem in our constraint set. No component is included for impressiveness alone.

```
┌─────────────────────────────────────────────────────────────────┐
│                     CogProfile-Net Architecture                  │
│                                                                  │
│  INPUT: Raw EEG sub-window X ∈ ℝ^{14 × 128}                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  COMPONENT 1: Learnable Temporal Filterbank             │    │
│  │  • SincNet / depthwise temporal convolutions             │    │
│  │  • Extracts band-specific temporal features              │    │
│  │  • WHY: Replaces manual Welch PSD computation            │    │
│  │  • Output: X_filtered ∈ ℝ^{14 × F × T'}                │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  COMPONENT 2: Riemannian SPD Encoder                    │    │
│  │  • Per-band spatial covariance: Σ_b = X_b X_b^T ∈ S++^14│    │
│  │  • BiMap layers: W Σ W^T (Stiefel manifold projection)  │    │
│  │  • ReEig nonlinearity (eigenvalue thresholding)          │    │
│  │  • LogEig projection to tangent space                    │    │
│  │  • WHY: Mathematically invariant to inter-subject       │    │
│  │    skull/impedance scaling. Proven for 14-channel EEG.   │    │
│  │  • Output: v ∈ ℝ^d (tangent space vector)               │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  COMPONENT 3: Contextual Temporal Encoder                │    │
│  │  • Lightweight 1D-CNN or BiGRU on temporal features      │    │
│  │  • WHY: Captures within-window temporal dynamics that    │    │
│  │    the static covariance matrix does not encode           │    │
│  │  • Output: h_t ∈ ℝ^d                                    │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  COMPONENT 4: Feature Fusion + Behavioral Gating         │    │
│  │  • Concatenate [v; h_t]                                  │    │
│  │  • Optional: gated cross-attention with behavioral       │    │
│  │    metadata (RT, difficulty, accuracy)                    │    │
│  │  • Anti-shortcut: auxiliary EEG-only classification head │    │
│  │  • WHY: Behavioral data is informative but dangerous     │    │
│  │    if the model learns to ignore EEG                     │    │
│  │  • Output: h_fused ∈ ℝ^d                                │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  COMPONENT 5: Evidential Classification Head (EDL)       │    │
│  │  • Dense layer → Softplus activation → evidence e ∈ ℝ^5+│    │
│  │  • Dirichlet parameters: α_k = e_k + 1                  │    │
│  │  • Expected probability: p_k = α_k / S                  │    │
│  │  • Epistemic uncertainty: u = K / S                      │    │
│  │  • WHY: Produces calibrated probabilities + explicit     │    │
│  │    "I don't know" signal. Critical for career profiling. │    │
│  │  • Output: Dir(α), p_k, u                                │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  COMPONENT 6: Career Mapping Layer (Post-hoc)            │    │
│  │  • CHC ability alignment matrix                          │    │
│  │  • O*NET occupational requirements lookup                │    │
│  │  • Monte Carlo sampling from Dir(α) → career posteriors  │    │
│  │  • Abstention when u > threshold                         │    │
│  │  • Output: Career fit scores + 95% credible intervals    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{EDL}}(\boldsymbol{\alpha}, \mathbf{y}) + \lambda_1 \mathcal{L}_{\text{aux}}^{\text{EEG-only}} + \lambda_2 \mathcal{L}_{\text{center}}$$

| Term | Formula | Purpose |
|:---|:---|:---|
| $\mathcal{L}_{\text{EDL}}$ | $\sum_k y_k (\psi(S) - \psi(\alpha_k)) + \lambda_t \text{KL}[\text{Dir}(\tilde{\alpha}) \|\| \text{Dir}(\mathbf{1})]$ | Evidential classification with KL regularizer. Penalizes overconfidence and drives uncertain samples toward uniform prior. |
| $\mathcal{L}_{\text{aux}}$ | $-\sum_k y_k \log \hat{p}_k^{\text{EEG}}$ | Auxiliary EEG-only branch prevents behavioral metadata shortcut learning. |
| $\mathcal{L}_{\text{center}}$ | $\sum_i \|v_i - c_{y_i}\|_2^2$ | Pulls tangent-space representations toward class centroids for tighter manifold clustering. |

### Why Each Component Exists (The Justification Table)

| Component | Problem It Solves | What Happens If Removed | Literature Support |
|:---|:---|:---|:---|
| Learnable Filterbank | Manual PSD computation is a bottleneck | Must hand-engineer 252 features | EE(G)-SPDNet (2024); SincNet |
| Riemannian SPD Encoder | Inter-subject amplitude/impedance noise | Accuracy degrades by ~5-10% on unseen subjects | Tibermacine 2026, Jin 2026 |
| Temporal Encoder | Static covariance misses temporal dynamics | Loses within-window transient information | Standard temporal modeling |
| Behavioral Gating | Model ignores EEG, overfits on reaction time | Classification accuracy inflated but useless | Kessler 2024 (preprocessing bias) |
| Evidential Head (EDL) | Softmax overconfidence on small data | Career recommendations are dangerously confident | Sensoy 2018, TRUEE (Wu 2025) |
| Career Mapper | No actionable output from classification | System is just another accuracy benchmark paper | Novel contribution |

### What Is NOT Included (And Why)

| Omitted Component | Why It Was Excluded |
|:---|:---|
| **Dynamic Graph Neural Network** | 14 nodes is too few for meaningful graph learning. The spatial relationships are better captured by the covariance matrix directly. |
| **Foundation Model Adaptation (LaBraM, BIOT)** | Montage mismatch (14→64 channels) requires uncertain interpolation. The adaptation overhead is not justified when training from scratch on 37K samples is feasible. |
| **Self-Supervised Pretraining** | 37K samples is borderline sufficient for supervised learning with proper regularization. SSL adds training complexity without proven benefit at this data scale for supervised cognitive tasks. |
| **Subject-Adversarial Disentanglement (GRL)** | We use subject-dependent evaluation. GRL is for cross-subject generalization, which is a separate paper. Adding it here without LOSO evaluation creates an unjustifiable claim. |
| **Triple-branch architecture** | Multiplies parameters ~3× with no ablation proof that each branch independently contributes. Reviewer would reject without proof. |

---

## 7. The Cognitive-to-Career Mapping Layer

### Step 1: EEG Tasks → CHC Cognitive Abilities

The mapping is grounded in Cattell-Horn-Carroll (CHC) theory, the most validated cognitive ability taxonomy (McGrew 2009, Keith & Reynolds 2010):

| EEG Task | CHC Broad Ability | CHC Code | O\*NET Ability Descriptor |
|:---|:---|:---:|:---|
| Mental Arithmetic | Quantitative Knowledge / Fluid Reasoning | $G_q$ / $G_f$ | Mathematical Reasoning, Number Facility |
| Pattern Recognition | Visual Processing / Fluid Reasoning | $G_v$ / $G_f$ | Spatial Orientation, Visualization |
| Working Memory | Short-Term Working Memory | $G_{wm}$ | Memorization, Information Ordering |
| Reading Comprehension | Reading & Writing / Comprehension-Knowledge | $G_{rw}$ / $G_c$ | Written Comprehension |
| Sustained Attention | Processing Speed (nearest mapping) | $G_s$ | Selective Attention, Time Sharing |

> **Important caveat:** Sustained Attention is NOT a clean CHC broad ability factor. It maps most closely to Processing Speed ($G_s$) via the WJ-III Pair Cancellation task, but it is more accurately described as an executive function / arousal construct. This must be explicitly acknowledged in the paper.

### Step 2: CHC Abilities → O\*NET Occupational Families

O\*NET provides importance and level ratings for 52 cognitive abilities across 900+ occupations. Our 5 EEG-derived abilities map to 5 of these 52 O\*NET descriptors.

**The mapping mechanism:**

Let $\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})$ be the EEG-derived cognitive profile (a probability distribution over 5 abilities).

Let $\mathbf{W} \in \mathbb{R}^{5 \times J}$ be the normalized O\*NET requirement matrix, where $W_{ij}$ is the required level of ability $i$ for occupation $j$.

**Career fit score (point estimate):**
$$\text{Fit}_j = \mathbf{p}^\top \mathbf{w}_j = \sum_{k=1}^5 p_k \cdot W_{kj}$$

**With uncertainty propagation (Monte Carlo):**
1. Draw $N = 1000$ samples: $\mathbf{p}^{(n)} \sim \text{Dir}(\boldsymbol{\alpha})$
2. Compute $\text{Fit}_j^{(n)} = \mathbf{p}^{(n)\top} \mathbf{w}_j$ for each sample
3. Report: $\text{Fit}_j = \mathbb{E}[\text{Fit}_j^{(n)}]$ and 95% CI = $[Q_{2.5\%}, Q_{97.5\%}]$

**Abstention rule:** When epistemic uncertainty $u > u_{\text{threshold}}$ (e.g., $u > 0.5$), the system outputs:
> *"Insufficient confidence in cognitive profile. Consider additional testing sessions."*

### Example Output

```
╔══════════════════════════════════════════════════════════╗
║              NEURAL COGNITIVE PROFILE REPORT             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Cognitive Profile (Dirichlet-calibrated):               ║
║  ┌──────────────────────┬────────┬──────────┐            ║
║  │ Ability              │ Score  │ 95% CI   │            ║
║  ├──────────────────────┼────────┼──────────┤            ║
║  │ Mathematical Reason. │ 0.15   │ ±0.03    │            ║
║  │ Pattern Recognition  │ 0.30   │ ±0.05    │            ║
║  │ Working Memory       │ 0.35   │ ±0.04    │            ║
║  │ Reading Comprehens.  │ 0.10   │ ±0.02    │            ║
║  │ Sustained Attention  │ 0.10   │ ±0.02    │            ║
║  └──────────────────────┴────────┴──────────┘            ║
║                                                          ║
║  System Confidence: HIGH (u = 0.12)                      ║
║                                                          ║
║  Top Career Aptitude Matches:                            ║
║  1. Data Analyst / Business Intelligence    78% ± 6%     ║
║  2. Software Developer / Engineer           72% ± 8%     ║
║  3. UX/UI Designer                          65% ± 10%    ║
║  4. Research Scientist                      61% ± 9%     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

## 8. Validation Strategy

### The Three-Tier Validation Ladder

Following the biomarker translation framework (Clayson 2024, 2025; Parmigiani 2022):

```
┌──────────────────────────────────────────────────┐
│  TIER 1: Analytic & Signal Validity               │
│  • Split-half reliability of EEG features          │
│  • Test-retest ICC across sessions (same subject)  │
│  • Signal quality metrics (SNR, artifact rates)    │
├──────────────────────────────────────────────────┤
│  TIER 2: Classification & Calibration Validity     │
│  • 5-class accuracy, Macro-F1, per-class AUROC     │
│  • Expected Calibration Error (ECE ↓)              │
│  • Brier Score (calibration + sharpness)           │
│  • Reliability diagrams                            │
│  • Uncertainty quality: OOD detection AUROC        │
│  • Permutation test (p < 0.001), bootstrap CIs     │
├──────────────────────────────────────────────────┤
│  TIER 3: Construct & Decision-Support Validity     │
│  • Convergent validity: EEG profile vs. standard   │
│    cognitive tests (WJ-IV, WAIS subtests)          │
│  • Career mapping face validity: expert panel      │
│  • Risk-coverage curves for abstention utility     │
│  • Longitudinal: profile stability over weeks      │
└──────────────────────────────────────────────────┘
```

### Metrics We MUST Report

| Metric | Target | Why Required |
|:---|:---|:---|
| Test Accuracy | >75% (subject-dependent) | Basic classification performance |
| Macro-F1 | >0.73 | Class-balanced performance |
| ECE (Expected Calibration Error) | <0.10 | Probability calibration quality |
| Brier Score | <0.25 | Joint calibration + sharpness |
| Epistemic uncertainty AUROC | >0.80 | Can the model detect OOD subjects? |
| Cohen's Kappa | >0.65 | Beyond-chance agreement |
| Test-retest ICC | >0.70 | Profile stability (trait vs. state) |

### What We Can Validate Now vs. Later

| Validation | Can Do Now? | Notes |
|:---|:---:|:---|
| Classification accuracy, F1, permutation test | ✅ Yes | Already done for ML; repeat for DL |
| ECE, Brier score, reliability diagrams | ✅ Yes | Compute from EDL Dirichlet outputs |
| OOD uncertainty detection | ✅ Yes | Hold out 3-5 subjects, check u rises |
| Test-retest reliability | ⚠️ Partially | Need multi-session data per subject |
| Convergent validity (vs. standardized tests) | ❌ Not yet | Requires administering WJ-IV or similar |
| Career mapping predictive validity | ❌ Not yet | Requires longitudinal tracking |

> **Paper positioning:** We validate Tiers 1 and 2 fully. For Tier 3, we present the framework and acknowledge it requires future longitudinal validation. This is honest and reviewers respect it.

---

## 9. Ethical and Legal Constraints

### The System Must Be Positioned As Advisory

Based on research into EU AI Act (2024), EEOC (2024), and ADA regulations:

| Regulatory Framework | Key Constraint | Our Response |
|:---|:---|:---|
| **EU AI Act** | AI in recruitment/vocational selection = High-Risk. Article 5 prohibits workplace emotion recognition. | Position as voluntary self-discovery and career exploration guidance — never as hiring/screening tool. |
| **US ADA / EEOC** | EEG is a medical examination. Pre-offer medical exams are prohibited for hiring. | Never frame as employment screening. System is for educational/personal career exploration. |
| **GDPR** | EEG data is biometric/health data (special category). | Require explicit informed consent. Minimize data retention. |

### Mandatory Disclaimers in the Paper

The paper must include:
1. *"This system is designed as an exploratory decision-support tool for career self-discovery, not an employment screening instrument."*
2. *"Cognitive ability profiles are influenced by many factors beyond neural measurements, including motivation, prior experience, cultural context, and test-taking conditions."*
3. *"Career success depends on non-cognitive factors (personality, interests, motivation, socioeconomic context) not measured by this system."*

---

## 10. Implementation Roadmap

### Phase 1: Deep Learning Encoder (Weeks 1-3)

| Task | Details | Priority |
|:---|:---|:---:|
| Implement Riemannian SPD encoder | BiMap → ReEig → LogEig in PyTorch (use `geoopt` or `geomstats` library) | P0 |
| Implement learnable temporal filterbank | SincNet or depthwise 1D convolutions (5 frequency bands) | P0 |
| Implement lightweight temporal encoder | BiGRU or 1D-CNN on temporal features | P1 |
| Feature fusion layer | Concatenation + MLP fusion of SPD tangent vector and temporal features | P1 |
| Training infrastructure | Subject-dependent stratified split, early stopping, weight decay | P0 |

### Phase 2: Evidential Classification Head (Week 3-4)

| Task | Details | Priority |
|:---|:---|:---:|
| Replace softmax with EDL head | Dense → Softplus → Dirichlet parameterization | P0 |
| Implement EDL loss | Bayes risk + KL regularizer with annealing schedule | P0 |
| Calibration evaluation | ECE, Brier score, reliability diagrams | P0 |
| Deep Ensemble baseline | Train 5 independently initialized models for UQ comparison | P1 |
| Temperature scaling post-hoc | Single-parameter validation-set calibration | P1 |

### Phase 3: Behavioral Gating (Week 4-5)

| Task | Details | Priority |
|:---|:---|:---:|
| Metadata embedding | Learned embeddings for RT (continuous), difficulty (categorical), accuracy (binary) | P1 |
| Gated cross-attention | EEG queries attend to behavioral keys/values | P1 |
| Anti-shortcut auxiliary head | EEG-only classification branch with auxiliary loss | P1 |
| Ablation: with vs. without behavioral data | Prove EEG alone works; behavioral data improves | P1 |

### Phase 4: Career Mapping Layer (Week 5-6)

| Task | Details | Priority |
|:---|:---|:---:|
| Build CHC alignment matrix | Map 5 EEG classes to CHC broad abilities | P0 |
| Download O\*NET ability ratings | Extract relevant ability dimensions for 900+ occupations | P0 |
| Monte Carlo career scoring | Sample from Dirichlet posterior, compute fit scores + CIs | P0 |
| Abstention logic | High-uncertainty → refuse to recommend | P0 |

### Phase 5: Ablation Studies & Paper Writing (Week 6-8)

| Ablation Experiment | What It Proves |
|:---|:---|
| CogProfile-Net vs. ML baseline (LightGBM 73.34%) | DL encoder learns better representations |
| SPD encoder vs. standard CNN | Riemannian geometry provides subject robustness |
| EDL head vs. standard softmax | Evidential output is better calibrated |
| With behavioral gating vs. without | Metadata improves accuracy without shortcutting |
| Full system vs. no career mapper | End-to-end system produces actionable output |
| EDL vs. Deep Ensemble vs. MC-Dropout | Compare UQ methods for this dataset |

---

## 11. References

### Riemannian Geometry & SPD Networks
- Tibermacine, I. E., Russo, S., & Napoli, C. (2026). Stiefel-SPD Manifold Graph Convolution for End-to-End EEG Learning. *IEEE T-NSRE*, 34, 595-606.
- Jin, J., et al. (2026). RUNet: A Zero-Calibration Framework for Cross-Domain EEG Decoding via Riemannian and Unsupervised Representation Learning. *IEEE TBME*.
- Gao, Y.-Z., & Deng, H. (2026). Cross-subject EEG emotion recognition using Riemannian Graph Transformers with Geodesic Adversarial Adaptation. *Alexandria Engineering Journal*.
- Kim, B. H., et al. (2023). A discriminative SPD feature learning approach on Riemannian manifolds for EEG classification. *Pattern Recognition*, 143, 109751.

### Evidential Deep Learning & Uncertainty
- Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. *NeurIPS*.
- Wu, et al. (2025). TRUEE: Reliable decision making on clinical EEG with multi-view learning and subjective logic. *Expert Systems with Applications*.
- Tveter, M., et al. (2026). Benchmarking UQ Methods on EEG Classification Under Dataset Shifts. *AI in Medicine*.
- Pandey, D. S., et al. (2025). Generalized Regularized EDL. *IEEE TPAMI*.
- Jürgens, M., et al. (2024). Is epistemic uncertainty faithfully represented by EDL methods? *ICML 2024*.
- Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
- Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive UQ using Deep Ensembles. *NeurIPS*.

### Psychometrics & Cognitive Ability Theory
- McGrew, K. S. (2009). CHC theory and the human cognitive abilities project. *Intelligence*, 37(1), 1-10.
- Keith, T. Z., & Reynolds, M. R. (2010). Cattell-Horn-Carroll abilities and cognitive tests. *Psychology in the Schools*, 47(7).
- Flanagan, D. P., & Dixon, S. G. (2014). The CHC Theory of Cognitive Abilities. *Encyclopedia of Special Education*.
- Gottfredson, L. (2003). The Challenge and Promise of Cognitive Career Assessment. *Journal of Career Assessment*.
- Nye, C. D., et al. (2022). Cognitive Ability and Job Performance: Meta-analytic Evidence. *J. Business and Psychology*.
- Kato, K., & Scherbaum, C. A. (2023). Cognitive Ability Tilt and Job Performance. *Journal of Intelligence*.

### Career Mapping & Occupational Databases
- Rahim, M., & Basheer, K. P. M. (2025). Hierarchical Multi-Tiered Personalized Career Recommender. *Indian J. Science and Technology*.
- Alonso, R., et al. (2025). Novel approach for job matching using transformers and O\*NET. *Big Data Research*.
- Wilson, C., & Caliskan, A. (2021). Building and Auditing Fair Algorithms: Pymetrics Case Study. *ACM FAccT*.

### Legal & Ethical
- European Parliament (2024). Artificial Intelligence Act (EU AI Act). *Official Journal of the EU*.
- US EEOC (2024). Wearables in the Workplace: Using Wearable Technology Under Federal Employment Discrimination Laws.
- Muhl, E., & Andorno, R. (2023). Neurosurveillance in the workplace. *Frontiers in Human Dynamics*.

### Validation Frameworks
- Clayson, P. (2024). The Psychometric Upgrade Psychophysiology Needs. *Psychophysiology*.
- Clayson, P. (2025). Translating EEG Biomarkers into Clinical Tools. *American Psychologist*.
- Parmigiani, S., et al. (2022). Reliability and validity of TMS-EEG biomarkers. *Biol. Psychiatry CNNI*.
- Attarha, M., et al. (2026). Construct Validation of a Remote Brain Health Assessment Battery. *JMIR Formative Research*.

---

> **Summary for Mentor Discussion:**
>
> The innovation is not in stacking complex neural network modules. The innovation is in building the **first complete system** that goes from raw consumer-grade EEG → uncertainty-calibrated cognitive profiling via Riemannian deep learning → psychometrically-grounded career aptitude mapping with principled uncertainty propagation. Each component is individually justified by the data constraints, and the end-to-end system is novel in its totality. The Riemannian SPD encoder solves the inter-subject noise problem. The Evidential head solves the calibration problem. The CHC→O\*NET mapping solves the "so what?" problem. Together, they form a coherent research contribution suitable for IEEE T-NSRE, JNE, or TBME.
