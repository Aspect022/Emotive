# EEG Decoding under Low-Data, 14-Channel Constraints (2024–2026)

Recent EEG deep‐learning research has advanced specialized architectures that address both **small-data** and **inter-subject variability**. We structure our review by the six “novelty axes” requested, citing cutting-edge 2023–2026 literature for each.

## Pillar A: Dynamic Functional Connectivity & ST-GNNs

**Dynamic adjacency construction:** 14-channel EEG connectivity can be modeled via *time-varying* graph adjacencies.  Two approaches stand out: using neurophysiological metrics (e.g. phase-locking value (PLV), phase-lag index (PLI), mutual information) computed over short windows, or *learnable* adjacencies via attention. For example, DGAT and related works dynamically learn channel‐to‐channel weights using self-attention: they let each channel’s embedding attend to others to form $A_{ij}(t)$.  Formally, one can define query/key vectors $q_i(t), k_j(t)$ for channel features and set 
$$e_{ij}(t)=\frac{q_i(t)^\top k_j(t)}{\sqrt{d_k}},\quad A_{ij}(t)=\frac{\exp(e_{ij}(t))}{\sum_j\exp(e_{ij}(t))}\,,$$ 
yielding a *learned adjacency* that evolves with time. This is akin to self-attention graph kernels. Biologically motivated adjacencies can also use electrode geometry (e.g. geodesic distance on skull) or frequency‐specific PLV/PLI in sliding windows.  

**Graph convolution & update equations:** With adjacency $A(t)$ (possibly symmetrized and normalized), graph convolution proceeds per time-sample.  A typical spatial graph convolution (following Kipf & Welling’s GCN) is:  
$$H^{(l+1)}(t) = \sigma\!\big(\tilde D^{-1/2}\tilde A(t)\tilde D^{-1/2} H^{(l)}(t)W^{(l)}\big)\,, $$  
where $\tilde A=D+I$ includes self-loops and $H^{(0)}(t)\in\mathbb R^{14\times F_0}$ are the 14 channel features.  Dynamic GNNs (DGCNN) stack such spatial convolutions across time.  Graph Attention Networks (GAT) use learned edge weights: for each node $i$,  
$$h_i' = \sum_{j\in\mathcal{N}(i)} \alpha_{ij}W h_j,\quad \alpha_{ij} = \text{softmax}(\text{LeakyReLU}(a^\top[W h_i\|W h_j]))\,, $$ 
which can be made time-dependent.  Spatial‐temporal graph networks then combine GCN layers for space with 1D convolutions or Transformers for time. For example, one could apply a spatial GCN at each timepoint and then a temporal convolution (or Transformer) across the resulting time‐sequence of graph-pooled features. 

**Spatial-Temporal Graph Transformers:** A recent trend is to replace or augment graph conv layers with Transformer blocks. For instance, after a GCN-based embedding $H(t)$ per time-slice, one can treat time or channels as token sequences and apply multi-head self-attention.  For cross-channel relationships, one variant is *Graph Conformer* (Graph + Conformer): use graph conv to get channel embeddings, then feed the sequence of channel embeddings through a Transformer (with positional encodings based on electrode positions).  Alternatively, an ST Graph Transformer can alternate spatial attention (over channels, possibly masked by adjacency) and temporal attention (over time).  Such modules follow standard Transformer math: 
$$\text{MultiHead}(Q,K,V)=\text{Concat}_{h}( \text{softmax}(Q_h K_h^\top/\sqrt{d_k})V_h)\,, $$ 
with $Q,K,V$ projections of node/time embeddings. E.g. in DeepAttNet (ear-EEG model) bidirectional cross-attention is used between two channels.  

**Regularization of adjacency:**  To avoid overfitting on 14 channels, recent works apply L1/L2 regularization or graph sparsity constraints on $A(t)$, or constrain $A$ to follow known neuroanatomy (e.g. enforcing left/right asymmetries as in).  

**Explainability of connectivity:**  Learned adjacency weights can be visualized to infer biologically plausible networks.  For example, DGAT and DeepAttNet showed that attention weights concentrate along plausible inter-hemispheric links.  Additionally, one can enforce anatomical plausibility by biasing adjacencies to favor nearby electrodes (e.g. distance-based prior via geodesic distance on a sphere). In practice, we encourage interpretable graphs by adding regularization terms (e.g. $\|A\|_1$ sparsity, or $\|A-A_{\text{brain}}\|^2$ to a canonical graph). This yields connectivity that neuroscientists can inspect for, say, fronto-parietal coupling during arithmetic tasks.

## Pillar B: Self-Supervised Learning (SSL) & EEG Foundation Models

**EEG Foundation Models:** Recent efforts have produced large EEG pre-trained “founder” models (analogous to language models).  Notable examples include LaBraM, BrainGPT, BIOT, and REVE (in Stacked LoRA study). These are typically large Transformers trained with masked/self-prediction tasks on massive EEG corpora. For instance, LaBraM segments EEG into “channel patches” and tokenizes them, then pre-trains a Transformer with a masked reconstruction loss. BrainGPT (Aug 2025) is trained autoregressively on a massive 138-channel EEG corpus and handles arbitrary subsets of channels. BIOT (NeurIPS 2023) tokenizes segments of each channel into a “biosignal sentence” with relative position embeddings, enabling cross-dataset pretraining. 

**Adapting to 14 channels:** When using a pretrained model that expects e.g. 64 or 128 channels, we must adapt to the 14-channel Emotiv montage. Strategies include:
- **Channel interpolation or masking:** Pad or mask missing channels and rely on the model to ignore them, or cluster similar channels.  
- **Projection layers:** Learn a linear “head” that maps 14-channel inputs into the pretrained model’s 64-channel embedding space.  
- **Re-training input embeddings:** Use the pretrained Transformer’s architecture but retrain the first layer’s embedding to accept 14 positions. For example, BrainGPT’s electrode-wise strategy suggests treating each electrode as an independent “token”; one could similarly fine-tune with only the 14 needed tokens.  

**Parameter-efficient fine-tuning:**  Low-Rank Adaptation (LoRA) and related methods have been applied to EEG.  *Stacked LoRA* (2026) explicitly decouples **global** (task-shared) and **subject-specific** adaptations by adding separate low-rank adapters. It learns a shared adapter for all subjects and one adapter per subject, thus preserving a common EEG representation while capturing idiosyncrasies. Such methods let us fine-tune a frozen EEG foundation model to our dataset with few parameters. We can apply this to LaBraM or REVE: the input-channels mismatch is handled by position embeddings, and LoRA learns subject-conditioned corrections without overwriting global features. 

**SSL pretext tasks:**  If training from scratch on 37k sliding-window segments, effective self-supervised tasks include:
- **EEG-MAE (Masked Autoencoder):** Randomly mask e.g. 50% of the EEG time-points or channels and train to reconstruct them (as in Zhou et al. 2024).  Apple’s MAEEG (2022) showed that transformer-based masked reconstruction on EEG significantly improves downstream accuracy in low-label regimes. Mathematically, given input $X\in\mathbb R^{14\times T}$ and mask $M$, train encoder $E$ and decoder $D$ via $\min \|D(E(X\odot M)) - X\|^2$.  
- **Temporal/Spatial contrastive:** Methods like TS-TCC treat augmented views of each trial as positives and others as negatives. One can maximize agreement between time-shifted or sensor-shuffled versions of the same trial, e.g. via InfoNCE loss.  
- **Subject-Invariant contrastive (e.g. CLOCS/CLISA):** Align same-task trials across subjects. For example, CLISA contrasts pairs of EEG segments from different subjects but same task label, pulling them together and pushing apart different-task segments. This yields $L_{\text{cont}} = -\log\frac{\exp(s(x_i^a,x_j^b)/\tau)}{\sum_k \exp(s(x_i^a,x_k^b)/\tau)}$ where $s(\cdot,\cdot)$ is similarity and $(i,j)$ are same-task, different-subject.  

**Transfer learning with LoRA:**  With a pretrained model like LaBraM (2500h) or REVE (60,000h), apply LoRA to adapt it.  For example, use a low-rank matrix insertion in each Transformer block: if $W$ is a weight, update $W' = W + BA$ with small matrices $B\in \mathbb{R}^{d\times r},A\in \mathbb{R}^{r\times d}$, learning $B,A$ while freezing $W$.  Stacked LoRA extends this by learning subject‐specific $B_s,A_s$ paths. 

## Pillar C: Riemannian Geometry, SPD Manifolds & SPD Networks

**Covariance as SPD data:**  A classic robust EEG feature is the spatial covariance matrix $C = X X^\top\in\mathbb R^{14\times14}$ (symmetric positive-definite, SPD).  Operating on covariance directly leverages Riemannian geometry.  One can feed $C$ into an SPD-aware network. For instance, the **SPDNet** (Huang & Van Gool 2017) uses layers of the form  
$$C_{l+1} = W_l C_l W_l^\top,$$ 
keeping symmetry and positive-definiteness.  Each layer $W_l\in\mathbb R^{d_{l+1}\times d_l}$ is trainable; dimension can be reduced (e.g. $14\to 8$). Nonlinear activations can be done via eigenvalue corrections (ReEig).  

**Tangent-space mapping:**  A simpler approach maps $C$ to a Euclidean space via the matrix logarithm (Log-Euclidean Riemannian metric).  Compute $V=\log(C)$ (by eigen-decomposition $C=U\Lambda U^\top$, then $V=U\log(\Lambda)U^\top$).  Flatten the upper-triangular part of $V$ into a vector and feed it to an MLP or CNN.  This preserves geodesic distances: for two covariance matrices $C_1,C_2$, $d_R(C_1,C_2)=\|\log(C_1)-\log(C_2)\|_F$.  

**SPD neural nets:**  Recent work uses specialized blocks on SPD manifolds.  For example, the *EE(G)-SPDNet* is an end-to-end Riemannian network: it learns band-pass filters and then SPD layers on covariances.  Likewise, MENDR (2025) uses a *Manifold Transformer* with “Manifold Attention” that respects SPD geometry.  In such models, attention is defined via a log-Euclidean similarity (e.g. $sim(X,Y)=1/(1+\log(1+d_R(X,Y)))$).  

**Robustness to inter-subject noise:**  Covariance/SPD methods are inherently less sensitive to channel-level noise.  The SPD manifold is stable under scaling/impedance variations: two subjects’ covariance matrices can be aligned via Riemannian means.  In practice, projecting to tangent space (log-matrix) often outperforms Euclidean CNNs on cross-subject EEG.  For example, EE(G)-SPDNet showed better generalization by implicitly learning physiologically-plausible filters.  We thus recommend including one SPD-based pipeline (e.g. extract covariance per trial and feed into an SPDNet or Log-Euclidean network) as part of any model ensemble.

## Pillar D: Disentangled Representations & Domain Generalization

**Disentangling task vs subject:**  We want to separate cognitive-task features ($z_{\text{task}}$) from subject-specific ($z_{\text{subj}}$) “neural fingerprint”.  One approach is a **Disentangled VAE**: encode each trial into two latent vectors, $z_t,z_s$.  We use a VAE with two independent outputs: 
$$z_t \sim \mathcal{N}(\mu_t,\sigma_t^2),\quad z_s\sim \mathcal{N}(\mu_s,\sigma_s^2),$$ 
trained with a reconstruction loss and two KL terms $KL(q(z_t)\|p)$ and $KL(q(z_s)\|p)$.  By choosing a larger $\beta$ weight on one KL (factorVAE style), we encourage one latent to carry minimal information.  A discriminator can enforce $z_s$ contains subject ID: e.g. add an adversarial loss $L_{\text{ID}}$ to predict subject from $z_s$ (and gradient-reverse it so $z_t$ is invariant).  Prior work on speech-evoked EEG used a factorial VAE to achieve nearly-perfect subject identification in $z_s$ and poor in $z_t$; we can analogously train $z_s$ to predict (and then remove) subject while $z_t$ predicts task.  

**Adversarial domain adaptation (DANN):**  A classic method is Domain-Adversarial Neural Network.  We append a subject-classifier on a feature layer with a gradient reversal layer (GRL).  The network thus optimizes classification loss $\mathcal{L}_{CE}$ on task *while* maximizing confusion of subjects.  Concretely, let $f(x)$ be features, then add a domain head $d(f)$ (with parameters $\theta_d$) predicting subject identity $s$.  Under GRL, the gradient for $f$ is reversed for the loss $L_s=\text{CE}(d(f(x)),s)$.  The joint objective is 
$$\min_{\theta}\max_{\theta_d}\; \mathcal{L}_{CE}(f(x),y) - \lambda\,\mathcal{L}_{CE}(d(f(x)),s)\,.$$ 
This encourages $f(x)$ to be *subject-invariant*.  The survey confirms DANN effectively learns features agnostic to subject identity.  

**Contrastive disentanglement:**  We can also use contrastive loss to force $z_t$ of the same task (across subjects) to align, while pushing apart $z_t$ of different tasks.  Simultaneously, encourage $z_s$ to cluster by subject (or use an InfoNCE that pushes $z_s$ of the same subject together).  Such dual contrastive objectives are akin to CLISA.  

**Meta-learning for calibration:**  Meta-learning (MAML-style) can enable rapid adaptation to a new subject.  In the MAML paradigm, we treat each subject as a “task” in episodic training.  Each episode samples a subject and a few trials, and optimizes for quick adaptation.  Nguyen & Guan (2024) propose a “subject-independent meta-learning” that reformulates training to minimize cross-subject divergence.  Concretely, one trains model parameters $\theta$ so that after one gradient step on 3–5 trials of an unseen subject, performance on that subject is good.  This yields a model that quickly calibrates with minimal data (3–5 trials) at test-time.  

## Pillar E: Multi-Modal Cross-Attention (EEG + Behavioral Metadata)

We can fuse EEG features with trial metadata ($m$ = difficulty, RT, correctness) via multi-head cross-attention.  One design: encode EEG as a sequence $X\in\mathbb{R}^{14\times d}$ (14 channels/timepoints as tokens) via a CNN or short Transformer, and encode metadata as another sequence $M\in \mathbb{R}^{k\times d}$ (e.g. one token per discrete field with learned embedding, plus perhaps a continuous embedding for RT). Then apply **cross-attention** between them: for example, let queries come from EEG and keys/values from metadata:  
$$A = \text{softmax}\bigl(X W_Q (M W_K)^\top/\sqrt{d_k}\bigr), \quad Z = A (M W_V).$$  
This yields a fused representation $Z$ that injects metadata into EEG features.  We can also do it bidirectionally (as in DeepAttNet), or fuse by concatenation followed by Transformer layers.  The DeepAttNet block is an example of cross-attention across two EEG streams; we extend this idea to EEG+meta.

**Gating to avoid shortcuts:**  To prevent the model from “cheating” by only using easy metadata signals, we propose an **attention gate**. For instance, compute a gate scalar $\alpha=\sigma(w^\top z_m)$ from the metadata embedding $z_m$, and weight EEG features: $h = \alpha \cdot h_{EEG}$.  Alternatively, we penalize reliance on $m$ by a consistency loss: encourage $f(X,m_1)\approx f(X,m_2)$ for different metadata $m_1,m_2$.  Concretely, given a trial $x$ and two meta values $m,m'$, we add 
$$\mathcal{L}_{\text{consist}} = \| \text{softmax}(f(x,m)) - \text{softmax}(f(x,m'))\|_2^2.$$ 
We also inject dropout on metadata tokens during training.  These gating/regularization schemes ensure EEG signals primarily drive decoding rather than metadata shortcuts.  

##_Visualization/Example Image:_ (DeepAttNet architecture).

## Pillar F: Candidate Architectures (Name, Structure, Novelty, Loss, Venues)

We propose **three** distinct architectures addressing the above axes:

1. **ST-GFormer:** *Spatial-Temporal Graph Conformer with Disentangled Latents*.  
   - **Input:** 4-s trial at 128 Hz, shaped $(B,14,512)$ or sliding $(B,14,128)$ windows.  
   - **Layers:** 
     1. **Dynamic Graph Conv (Spatial):** Compute per-trial adjacency $A$ via a small attention subnetwork on channel features. Apply a GCN layer: $H^1=\sigma(\hat D^{-1/2}\hat A\hat D^{-1/2} X W_s)$, $H^1\in\mathbb{R}^{14\times d}$. (Here $X$ is raw EEG $(14\times128)$.)  
     2. **Temporal Conformer:** Reshape $H^1$ to a sequence of length 128 with $d$-dim features (each timestep has 14-d vector from previous layer). Apply a Transformer encoder (multi-head self-attention across time) to get $\widetilde H(t)$.  
     3. **Latent VAE Branch:** From the final features, apply two heads: one predicts task-latent $z_t=\mu_t+\sigma_t\epsilon$, one subject-latent $z_s=\mu_s+\sigma_s\epsilon$.  
     4. **Classifier:** Concatenate $z_t$ with a pooled EEG feature and feed through an MLP to 5-class output.  
   - **Novelty:** Integrates *graph convolution* (capturing dynamic connectivity) with a temporal Transformer, plus a VAE to disentangle $z_{\text{task}}$ vs $z_{\text{subj}}$ (motivated by). Unlike a vanilla CNN, ST-GFormer explicitly models channel interactions and subject factors.  
   - **Loss:** 
     $$\mathcal{L} = \mathcal{L}_{CE}(y,\hat y) + \lambda_1\mathcal{L}_{KL}(q(z_t)\|p)+\lambda_2\mathcal{L}_{KL}(q(z_s)\|p) - \lambda_3\mathcal{L}_{\text{adv}}(z_t)$$ 
     where $\mathcal{L}_{adv}$ is a DANN loss (subject-classifier on $z_t$ with GRL).  I.e. $\mathcal{L}_{\text{total}}=\mathcal{L}_{CE}+\beta(\mathcal{L}_{KL}^t+\mathcal{L}_{KL}^s)-\alpha L_{\text{DANN}}$.  We choose $\beta,\alpha$ to balance task accuracy vs disentanglement.  
   - **Venue:** This hybrid model suits IEEE T-NSRE or TBME (novel architecture for EEG decoding), or a NeurIPS/ICML workshop on neuroscience.  

2. **RG-SPDNet:** *Riemannian Graph SPD Network with Manifold Embedding*.  
   - **Input:** Each 1-s window $X\in\mathbb{R}^{14\times128}$. Compute spatial covariance $C=X\,X^\top\in\mathbb{R}^{14\times14}$ (SPD).  
   - **Layers:** 
     1. **SPDNet Blocks:** Apply two BiMap layers: $C_1=W_1 C W_1^\top$, $C_2=W_2 C_1 W_2^\top$, reducing dimension (e.g. $14\to8\to6$). Between them use ReEig nonlinearity (ReLU on eigenvalues).  
     2. **Tangent Transform:** Compute matrix-log: $V=\log(C_2)$ and vectorize to $v\in\mathbb{R}^{\frac{6(6+1)}{2}}$.  
     3. **Graph Attention:** Treat the 6 SPD nodes as a graph, or simply use a small Transformer on $v$.  
     4. **Classifier:** MLP to 5 classes.  
   - **Novelty:** Works entirely on SPD manifold embeddings, making use of Riemannian layers.  By operating on covariances, it is inherently robust to scale differences (unlike Euclidean CNN).  The manifold attention is drawn from MENDR.  This contrasts a standard CNN on raw EEG.  
   - **Loss:** Standard CE loss $\mathcal{L}_{CE}$ plus **center loss** on tangent vectors: $L_{\text{center}}=\sum\|v_i-\bar v_{y_i}\|^2$ to tighten class clusters on the manifold. Total $L= \mathcal{L}_{CE}+\gamma L_{\text{center}}$.  
   - **Venue:** IEEE TBME or Journal of Neural Eng. emphasize geometry.  

3. **Meta-CrossAttn EEG-Transformer:** *Multimodal Cross-Attentive Transformer with Behavioral Gating*.  
   - **Input:** Raw EEG $(14\times128)$ and metadata: difficulty $d\in\{\text{easy,med,hard}\}$, RT (continuous), correctness $c\in\{0,1\}$.  
   - **Layers:** 
     1. **EEG Embedding:** A small ConvNet or linear patch embedding: map $(14,128)\to (D,T')$ (e.g. flatten to sequence of length $T'$ with embedding dim $D$).  
     2. **Meta Embedding:** Embed difficulty with a learnable vector $e_d\in\mathbb{R}^D$, map RT through a small MLP to $\mathbb{R}^D$, and correctness to $\mathbb{R}^D$. Stack these into a meta sequence $M\in\mathbb{R}^{3\times D}$.  
     3. **Cross-Attention:** Perform multi-head cross-attention: queries from EEG tokens, keys/values from metadata: $Z = \text{softmax}(XW_Q (M W_K)^\top/\sqrt{d_k})\,M W_V$. Concatenate $Z$ with original EEG features.  
     4. **Gated Fusion:** Compute gate $\alpha=\sigma(w^\top [\bar Z;\bar X])$ (sigmoid of pooled fused embedding) and set $H=\alpha Z + (1-\alpha) X$.  
     5. **Transformer Encoder:** One or two standard Transformer layers on sequence $H$.  
     6. **Classifier:** Global pooling + MLP to 5 classes.  
   - **Novelty:** This architecture tightly fuses EEG with trial metadata via cross-attention (inspired by DeepAttNet’s bidirectional cross-attn), plus a gating scalar to modulate influence of metadata.  The gating mechanism (step 4) explicitly prevents the network from ignoring EEG.  Compared to a naive concatenation, cross-attention allows the model to query meta features at each time.  
   - **Loss:** 
     $$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{\text{gate}}\big\|f(X,m)-f(X,m')\big\|^2,$$ 
     where $(X,m)$ and $(X,m')$ are same EEG with altered metadata (e.g. swap difficulty), enforcing consistency.  Here $\mathcal{L}_{CE}$ is cross-entropy, and the gating consistency term penalizes over-reliance on metadata.  
   - **Venue:** A top-tier ML conference (e.g. NeurIPS 2025 Workshop on “Brain and Body Foundation Models”) or ICML; or application journals (e.g. Neural Networks, since it merges modalities).  

**Summary of Novel Mathematical Components:**
- **Graph conv:** $H'= \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}HW)$.  
- **Cross-attention:** $A=\text{softmax}(QK^\top/\sqrt{d_k})$, $Z=A V$.  
- **Disentangle VAE:** $L_{KL}=\mathbb{E}[\,\|z_t\|^2 + \|\log\sigma_t^2\| -1-\mu_t^2 - \sigma_t^2\,]$ etc.  
- **Domain-adversarial:** $\min_{\theta}\max_{\phi} \; \mathcal{L}_{CE}(y,f_\theta(x)) - \lambda \mathcal{L}_{CE}(s,d_\phi(f_\theta(x)))$.  
- **Total losses:** e.g. $\mathcal{L}=\mathcal{L}_{CE}+\lambda_1\mathcal{L}_{KL}+\lambda_2\mathcal{L}_{center}+\lambda_3\mathcal{L}_{consist}$. 

**Target venues:** Given the intersection of methodology and EEG, strong journals include *IEEE Trans. Neural Systems & Rehab Eng.* (T-NSRE), *IEEE Trans. Biomed. Eng.*, *Journal of Neural Eng.*, *Neural Networks*. Workshops/Conferences: NeurIPS/ICML Brain & Body workshops, or domain venues (EMBC, IJCNN). The novelty (combining GNN, Riemannian, SSL, meta-learning) pushes toward a top-tier ML/AI workshop or cross-disciplinary journal.

**Sources:** We have drawn from recent EEG studies (2023–2026) on graph neural networks, SSL and foundation models, Riemannian networks, and domain-generalization. All technical claims above are grounded in these sources.