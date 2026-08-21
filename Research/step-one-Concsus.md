# Low-Density **Cognitive EEG** Architecture Design

For a 14-channel, small-data, 5-class cognitive EEG problem, the strongest 2024–2026 design space is not a larger vanilla CNN or LSTM. The literature instead points to **hybrid geometry-aware, graph-aware, and self-supervised pipelines** that explicitly model dynamic inter-channel structure, preserve covariance geometry, and suppress subject-specific nuisance variation with domain-invariant or disentangled objectives  (Li et al., 2023; Fu et al., 2024; Gao & Deng, 2026; Jin et al., 2026; Zhang et al., 2025). For your exact constraint set, the most publishable direction is a **multi-branch system** that combines a dynamic graph encoder over raw or band-limited windows, an SPD/Riemannian covariance branch for subject-robust statistics, and a lightweight behavioral fusion head with anti-shortcut regularization  (Han et al., 2025; Zhao et al., 2026; Chen et al., 2025).

## Architecture Priorities

The evidence claims table below compares the strongest design bets for your regime.

| Evidence Strength | Claim |
|---|---|
| Evidence strength: Strong (9/10) | **Dynamic graph learning** consistently improves EEG decoding when fixed adjacency is replaced by adaptive or layerwise connectivity that tracks inter-channel dependence over time  (Ding et al., 2025; Li et al., 2023; Cheng et al., 2023; Ye et al., 2022)|
| Evidence strength: Strong (8/10) | **Riemannian/SPD representations** improve cross-subject robustness by preserving covariance geometry instead of flattening it into Euclidean vectors  (Gao & Deng, 2026; Jin et al., 2026; Tibermacine et al., 2026)|
| Evidence strength: Strong (8/10) | **Self-supervised pretraining** is the main route around limited labels, especially masked reconstruction and contrastive objectives adapted to EEG structure  (Kuruppu et al., 2025; Foumani et al., 2024; Fu et al., 2024; Li et al., 2024)|

**Figure 1:** Evidence strength for candidate methodological directions

A purely supervised end-to-end transformer is a weaker choice in this regime because transformer EEG surveys repeatedly frame data scarcity, transfer learning, and augmentation as the key bottlenecks rather than raw model scale alone  (Keutayeva & Abibullaev, 2024; Xiong et al., 2025). Foundation-model reviews also note that current EEG-FM evaluations remain heterogeneous and that off-the-shelf utility is still uncertain, so adaptation strategy matters more than simply importing a large pretrained backbone  (Kuruppu et al., 2025; Portmann & Morishima, 2026).

## Dynamic Graph Designs

A strong 14-channel graph starts with nodes as electrodes and node features as either raw temporal embeddings, bandpower/PSD summaries, or short-window sequence encodings from 1D temporal convolutions or GRUs  (Tang et al., 2024; Zhang et al., 2025; Jin et al., 2026; Rommel et al., 2022). The current literature increasingly uses **hybrid adjacency construction**: initialize with spatial priors or intrinsic scalp topology, then add a learnable similarity term from node embeddings, and finally sparsify or threshold the matrix to suppress noisy edges  (Tang et al., 2024; Fu et al., 2024; Du et al., 2022).

A practical formulation for your setting is
\[
X \in \mathbb{R}^{B \times C \times T}, \quad C=14, \ T=128
\]
for 1 s windows. After a temporal encoder \(h_i=f_\theta(x_i)\in\mathbb{R}^{d}\) for channel \(i\), define
\[
A^{(0)}_{ij}=\alpha A^{\text{phys}}_{ij}+(1-\alpha)A^{\text{func}}_{ij},
\]
where \(A^{\text{phys}}\) is a distance or k-NN scalp graph and \(A^{\text{func}}\) is dPLV, PLI, MI, cosine similarity, or correlation estimated on the same window. Then learn a correction
\[
E_{ij}=\frac{(W_q h_i)^\top (W_k h_j)}{\sqrt d}, \qquad
\tilde A_{ij}=\sigma(A^{(0)}_{ij}+E_{ij}),
\]
followed by top-\(k\) or soft-threshold sparsification. This matches the direction taken by self-attention similarity fusion, dynamic adjacency learning, and learnable sparsification in recent graph EEG systems  (Tang et al., 2024; Ding et al., 2025; Fu et al., 2024; Zhou et al., 2024).

For propagation, the standard choice remains
\[
H^{(\ell+1)}=\phi\!\left(\hat D^{-1/2}\hat A \hat D^{-1/2} H^{(\ell)} W^{(\ell)}\right),
\]
with \(\hat A=\tilde A+I\), but recent EEG models improve on this with graph attention or graph-temporal hybrids that let spatial and temporal updates interact rather than remain sequential  (Jiang et al., 2024; Li et al., 2023; Yan et al., 2025; Cheng et al., 2023). A stronger temporal coupling is
\[
Z_t=\text{GAT}(X_t,\tilde A_t), \qquad
s_t=\text{BiLSTM}(Z_{1:t}) \ \text{or}\  \text{Transformer}(Z_{1:T'}),
\]
which mirrors STGATE, DTS-GAN, and several dynamic graph-temporal designs  (Li et al., 2023; Yan et al., 2025; Luo et al., 2023).

For biological explainability, the most defensible strategy is **constrained adaptivity** rather than fully free edges. Use a prior graph from scalp distance or known lobe partitions, learn only residual edge corrections, and report stable edge saliency aggregated by class and subject  (Tang et al., 2024; Ye et al., 2022; Almohammadi & Wang, 2024; Li et al., 2023). Multi-graph designs are especially attractive for 14 channels because they let you compare topology, causality, and functional coupling without needing dense montages  (Li et al., 2023; Wang et al., 2024).

## SSL and Foundation Adaptation

The center of gravity in EEG foundation modeling is still transformer-based masked pretraining, but the practical problem for your setup is **montage mismatch** and **low-channel adaptation** rather than lack of backbone options  (Kuruppu et al., 2025; Portmann & Morishima, 2026; Jun & Ruotsalo, 2026). Recent montage-agnostic work shows that a lightweight preprocessing or adapter layer can interpolate learned channel embeddings using electrode coordinates while leaving the backbone frozen or mostly frozen  (Jun & Ruotsalo, 2026). That is the cleanest way to adapt a 64-channel pretrained model to Emotiv 14-channel data.

A practical adapter is
\[
e'_i=\sum_{m=1}^{M} w_{im} e_m,\qquad
w_{im}=\frac{\exp(-\|p_i-p_m\|^2/\tau)}{\sum_{m'}\exp(-\|p_i-p_{m'}\|^2/\tau)},
\]
where \(p_i\) are 14-channel coordinates and \(e_m\) are source montage embeddings. This coordinate-based interpolation is directly aligned with montage-agnostic adapter findings and low-coverage transfer results from sparse practical EEG setups  (Jun & Ruotsalo, 2026; Zheng et al., 2025). Parameter-efficient adaptation should then use frozen or partially frozen backbone blocks plus a low-rank or prototype-conditioned adapter, because unconstrained full fine-tuning is reported to cause miscalibration, collapse, and representation drift in label-limited adaptation  (Jin et al., 2026).

If pretraining from scratch on your own data, the best-supported pretext family is **masked modeling plus contrastive structure**, not either one alone  (Fu et al., 2024; Foumani et al., 2024; Sun et al., 2024; Portmann & Morishima, 2026). MAEEG shows masking strategy and masked proportion strongly affect downstream benefit  (Chien et al., 2022). EEG2Rep improves on raw-signal reconstruction by predicting masked latent representations and using semantic subsequence preserving masks, with 50% preserved context working best on average across six tasks  (Foumani et al., 2024). GMAEEG is especially relevant because it embeds masked autoencoding directly in a graph formulation with dynamic adjacency transfer to downstream tasks  (Fu et al., 2024).

For small-data cognitive decoding, a good SSL loss is
\[
\mathcal L_{\text{SSL}}=\lambda_r \|g_\psi(\text{Mask}(X))-T(X)\|_2^2 + \lambda_c \,\mathcal L_{\text{InfoNCE}} + \lambda_s \,\mathcal L_{\text{subject-inv}},
\]
where \(T(X)\) is either the raw target, latent target, or graph target. The contrastive branch should use EEG-specific augmentations and, if possible, treat windows from the same trial as positives while discouraging subject identity leakage  (Li et al., 2024; Eldele et al., 2022; Wei et al., 2026).

## Riemannian and Subject-Invariant Learning

Riemannian methods are unusually well matched to your low-density setting because a 14×14 spatial covariance matrix is stable enough to estimate with shrinkage, small enough for efficient manifold operations, and more robust to amplitude-scale nuisance than raw-signal Euclidean embeddings  (Kim et al., 2023; Shi et al., 2024; Tibermacine et al., 2026). Recent cross-subject systems repeatedly show that preserving SPD geometry helps when subject variability is the main failure mode  (Gao & Deng, 2026; Jin et al., 2026; Tibermacine et al., 2026).

The basic branch is
\[
\Sigma = \frac{1}{T-1}(X-\bar X)(X-\bar X)^\top + \epsilon I \in \mathcal S_{++}^{14}.
\]
Then either map once to tangent space
\[
Z=\log\!\left(\bar\Sigma^{-1/2}\Sigma \bar\Sigma^{-1/2}\right),
\]
or use an end-to-end SPD network with learnable Stiefel projection
\[
\Sigma' = W \Sigma W^\top,\qquad W\in \mathrm{St}(d,14),
\]
followed by tangent-space graph aggregation or log-Euclidean classification  (Tibermacine et al., 2026). This is more novel than a standard tangent-space baseline and still computationally feasible because the cubic cost applies to reduced rank \(d\), not the full channel count  (Tibermacine et al., 2026).

The evidence also supports explicit geometry-aware alignment losses. A useful cross-subject penalty is
\[
\mathcal L_{\text{geo}} = \sum_c d_{\text{LE}}(\mu_c^{(s)},\mu_c^{(t)})^2,
\]
where \(\mu_c\) are classwise Fréchet means and \(d_{\text{LE}}\) is Log-Euclidean distance, echoing recent geodesic adversarial adaptation  (Gao & Deng, 2026). A simpler preprocessing baseline is Euclidean alignment before any network, because it remains an efficient and widely validated transfer-learning primitive across EEG paradigms  (Wu, 2025).

For disentanglement, the cleanest architecture is a shared encoder \(f\) plus two heads:
\[
z_{\text{task}}=g_t(f(X)), \qquad z_{\text{subj}}=g_s(f(X)).
\]
Optimize
\[
\mathcal L = \mathcal L_{\text{CE}}(y,\hat y)
+ \lambda_1 \mathcal L_{\text{adv-subj}}
+ \lambda_2 \mathcal L_{\text{orth}}
+ \lambda_3 \mathcal L_{\text{recon}}
+ \lambda_4 \mathcal L_{\text{center/proto}}.
\]
The adversarial term removes subject information from \(z_{\text{task}}\), the orthogonality term discourages leakage between shared and private spaces, and reconstruction or prototype losses prevent degenerate invariance  (Hu et al., 2023; Zhang et al., 2024; Liu et al., 2025). Recent causal and disentangled formulations further argue that the target is not generic invariance but **spurious-free latent structure** tied to the cognitive variable rather than the acquisition domain  (Zhang et al., 2025; Liu et al., 2023).

## Behavioral Fusion and Three Candidate Pipelines

Behavioral metadata should be treated as a **privileged but risky modality**. Studies on cognitive load and multimodal human-state decoding consistently find gains when neural temporal features are fused with connectivity or contextual signals, but they also warn that simple concatenation underuses cross-modal structure  (Han et al., 2025; Li et al., 2025; Wang et al., 2022). A good fusion pattern is two-stream encoding with cross-attention:
\[
Q = H_{\text{EEG}}W_Q,\quad K = H_{\text{beh}}W_K,\quad V = H_{\text{beh}}W_V,
\]
\[
\text{CrossAttn}(H_{\text{EEG}},H_{\text{beh}})
=\text{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V,
\]
then gated residual fusion
\[
g=\sigma(W_g [h_{\text{EEG}}\|h_{\text{beh}}]),\qquad
h_{\text{fused}}=h_{\text{EEG}} + g \odot h_{\text{ctx}}.
\]
This matches cross-modal transformer practice in multimodal EEG work  (Wang et al., 2022; Abinaya & Dinakaran, 2026).

To prevent behavioral shortcuts, add **modality dropout**, **gating entropy regularization**, and an auxiliary classifier that must still decode class from EEG alone:
\[
\mathcal L_{\text{total}}=\mathcal L_{\text{CE}}^{\text{fused}}+\lambda_a\mathcal L_{\text{CE}}^{\text{EEG-only}}+\lambda_g \|g\|_1.
\]
A stronger version randomly masks RT/difficulty/correctness during training and penalizes excessive prediction drift when metadata are removed. The logic is supported indirectly by work showing that preprocessing and nuisance structure can artificially inflate decoding, so robustness requires constraining non-neural shortcuts  (Kessler et al., 2024; Zhang et al., 2025).

Three candidate architectures fit your project best:

**1. ST-GFormer-RD**  
Input: \((B,14,128)\) raw EEG + bandpass views. Temporal depthwise convs \(\rightarrow\) dynamic residual graph transformer \(\rightarrow\) subject-adversarial latent split \(\rightarrow\) 5-way classifier. Novelty: residual prior-constrained adjacency plus adversarial subject disentanglement. Loss:
\[
\mathcal L=\mathcal L_{\text{CE}}+\lambda_1\mathcal L_{\text{GRL-subj}}+\lambda_2\mathcal L_{\text{orth}}+\lambda_3\mathcal L_{\text{sparse-}A}.
\]
Best target: **IEEE T-NSRE**, **JNE**, **TBME**  (Li et al., 2023; Cheng et al., 2023; Hu et al., 2023; Li et al., 2024).

**2. GeoMAE-AlignNet**  
Input: raw \((B,14,128)\) and covariance \((B,14,14)\). SSL pretraining with graph-masked autoencoding, then Stiefel-SPD branch + log-Euclidean classifier + Euclidean or geodesic alignment. Novelty: joint latent-masked and covariance-geometry transfer for low-density consumer EEG. Loss:
\[
\mathcal L=\mathcal L_{\text{CE}}+\lambda_1\mathcal L_{\text{MAE}}+\lambda_2\mathcal L_{\text{geo-align}}+\lambda_3\mathcal L_{\text{center}}.
\]
Best target: **TBME**, **Neural Networks**, **JNE**  (Fu et al., 2024; Tibermacine et al., 2026; Wu, 2025).

**3. CogFuse-ProAdapter**  
Input: EEG \((B,14,128)\) plus RT, difficulty, correctness. Frozen or partially frozen foundation backbone with coordinate interpolation adapter, prototype-conditioned lightweight adaptation, EEG-behavior cross-attention, and EEG-only auxiliary branch. Novelty: montage-agnostic EFM adaptation plus anti-shortcut gated multimodal fusion. Loss:
\[
\mathcal L=\mathcal L_{\text{CE}}^{\text{fused}}+\lambda_1\mathcal L_{\text{CE}}^{\text{EEG}}+\lambda_2\mathcal L_{\text{proto}}+\lambda_3\mathcal L_{\text{consistency}}.
\]
Best target: **T-NSRE**, **Journal of Neural Engineering**, **NeurIPS/ICML workshop**  (Jun & Ruotsalo, 2026; Jin et al., 2026; Wang et al., 2022; Abinaya & Dinakaran, 2026).

| Pipeline | Core Novelty | Strongest Use Case | Main Risk |
|---|---|---|---|
| ST-GFormer-RD | Dynamic graph + subject disentanglement | End-to-end cognitive state decoding | Overfitting if graph too free |
| GeoMAE-AlignNet | SSL + SPD geometry + alignment | Cross-subject robustness | Heavier training complexity |
| CogFuse-ProAdapter | Foundation adaptation + metadata fusion | Best practical accuracy | Behavioral shortcut leakage |

**Figure 2:** Three candidate pipelines for low-density cognitive EEG

For your 14-channel, small-data, 5-class cognitive EEG problem, the single most important shift in the literature is the move from **Euclidean supervised decoders** to **structure-aware representation learning** that jointly exploits dynamic connectivity, covariance geometry, and self-supervised pretraining. The single biggest open question is not whether these ingredients help, but **how to combine them without losing interpretability or letting subject identity and behavioral shortcuts dominate the latent space**.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
J., & Ruotsalo, T. (2026). Adapting frozen foundation models for montage-agnostic high-resolution EEG event segmentation. *Journal of Neural Engineering, 23*. https://doi.org/10.1088/1741-2552/ae6142
 
J., Wu, F., Xing, Y., Lin, Q., Liu, T., Liu, C., Jia, Z., & Feng, M. (2026). Structured Prototype-Guided Adaptation for EEG Foundation Models. *ArXiv, abs/2602.17251*. https://doi.org/10.48550/arxiv.2602.17251
 
Abinaya, G., & Dinakaran, K. (2026). Adaptive multimodal learning for driver cognitive state monitoring using transformer-based fusion with personalized meta-learning and federated optimization. *Scientific Reports, 16*. https://doi.org/10.1038/s41598-026-51635-3
 
Almohammadi, A., & Wang, Y.-K. (2024). Revealing brain connectivity: graph embeddings for EEG representation learning and comparative analysis of structural and functional connectivity. *Frontiers in Neuroscience, 17*. https://doi.org/10.3389/fnins.2023.1288433
 
Chen, X., Bao, X., Jitian, K., Li, R., Zhu, L., & Kong, W. (2025). Hybrid EEG Feature Learning Method for Cross-Session Human Mental Attention State Classification. *Brain Sciences, 15*. https://doi.org/10.3390/brainsci15080805
 
Cheng, C., Yu, Z., Zhang, Y., & Feng, L. (2023). Hybrid Network Using Dynamic Graph Convolution and Temporal Self-Attention for EEG-Based Emotion Recognition. *IEEE Transactions on Neural Networks and Learning Systems, 35*, 18565-18575. https://doi.org/10.1109/tnnls.2023.3319315
 
Chien, H., Goh, H., Sandino, C. M., & Cheng, J. Y. (2022). MAEEG: Masked Auto-encoder for EEG Representation Learning. *ArXiv, abs/2211.02625*. https://doi.org/10.48550/arxiv.2211.02625
 
Ding, S., Wang, K., Jiang, W., Xu, C., Bo, H., L., & Li, H. (2025). DGAT: a dynamic graph attention neural network framework for EEG emotion recognition. *Frontiers in Psychiatry, 16*. https://doi.org/10.3389/fpsyt.2025.1633860
 
Du, G., Su, J., Zhang, L., Su, K., Wang, X., Teng, S., & Liu, P. (2022). A Multi-Dimensional Graph Convolution Network for EEG Emotion Recognition. *IEEE Transactions on Instrumentation and Measurement, 71*, 1-11. https://doi.org/10.1109/tim.2022.3204314
 
Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C., Li, X., & Guan, C. (2022). Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45*, 15604-15618. https://doi.org/10.1109/tpami.2023.3308189
 
Foumani, N. M., Mackellar, G., Ghane, S., Irtza, S., Nguyen, N., & Salehi, M. (2024). EEG2Rep: Enhancing Self-supervised EEG Representation Through Informative Masked Inputs. *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. https://doi.org/10.1145/3637528.3671600
 
Fu, Z., Zhu, H., Zhao, Y., Huan, R., Zhang, Y., Chen, S., & Pan, Y. (2024). GMAEEG: A Self-Supervised Graph Masked Autoencoder for EEG Representation Learning. *IEEE Journal of Biomedical and Health Informatics, 28*, 6486-6497. https://doi.org/10.1109/jbhi.2024.3443651
 
Gao, Y.-Z., & Deng, H. (2026). Cross-subject EEG emotion recognition using Riemannian Graph Transformers with Geodesic Adversarial Adaptation. *Alexandria Engineering Journal*. https://doi.org/10.1016/j.aej.2026.03.045
 
Han, J., Zhan, G., Wang, L., Liang, D., Zhang, H., Zhang, L., & Kang, X. (2025). Decoding EEG-based cognitive load using fusion of temporal and functional connectivity features.. *Computer methods in biomechanics and biomedical engineering*, 1-16. https://doi.org/10.1080/10255842.2025.2514132
 
Hu, H., Yue, K., Guo, M., Lu, K., & Liu, Y. (2023). Subject Separation Network for Reducing Calibration Time of MI-Based BCI. *Brain Sciences, 13*. https://doi.org/10.3390/brainsci13020221
 
Jiang, C., Dai, Y., Ding, Y., Chen, X., Li, Y., & Tang, Y. (2024). TSANN-TG: Temporal–Spatial Attention Neural Networks with Task-Specific Graph for EEG Emotion Recognition. *Brain Sciences, 14*. https://doi.org/10.3390/brainsci14050516
 
Jin, J., Wang, C., Xu, R., He, X., Wu, X., Li, J., Chen, W., Wang, X., & Cichocki, A. (2026). RUNet: A Zero-Calibration Framework for Cross-Domain EEG Decoding via Riemannian and Unsupervised Representation Learning.. *IEEE transactions on bio-medical engineering, PP*. https://doi.org/10.1109/tbme.2026.3653024
 
Kessler, R., Enge, A., & Skeide, M. (2024). How EEG preprocessing shapes decoding performance. *Communications Biology, 8*. https://doi.org/10.1038/s42003-025-08464-3
 
Keutayeva, A., & Abibullaev, B. (2024). Data Constraints and Performance Optimization for Transformer-Based Models in EEG-Based Brain-Computer Interfaces: A Survey. *IEEE Access, 12*, 62628-62647. https://doi.org/10.1109/access.2024.3394696
 
Kim, B. H., Choi, J. W., Lee, H., & Jo, S.-C. (2023). A discriminative SPD feature learning approach on Riemannian manifolds for EEG classification. *Pattern Recognit., 143*, 109751. https://doi.org/10.1016/j.patcog.2023.109751
 
Kuruppu, G., Wagh, N., Kremen, V., & Varatharajah, Y. (2025). EEG foundation models: a critical review of current progress and future directions. *Journal of Neural Engineering, 23*. https://doi.org/10.1088/1741-2552/ae4455
 
Li, J., Pan, W., Huang, H., Pan, J., & Wang, F. (2023). STGATE: Spatial-temporal graph attention network with a transformer encoder for EEG-based emotion recognition. *Frontiers in Human Neuroscience, 17*. https://doi.org/10.3389/fnhum.2023.1169949
 
Li, W., Li, H., Sun, X., Kang, H., An, S., Wang, G., & Gao, Z. (2024). Self-supervised contrastive learning for EEG-based cross-subject motor imagery recognition. *Journal of Neural Engineering, 21*. https://doi.org/10.1088/1741-2552/ad3986
 
Li, M., Qiu, M., Kong, W., Zhu, L., & Ding, Y. (2023). Fusion Graph Representation of EEG for Emotion Recognition. *Sensors (Basel, Switzerland), 23*. https://doi.org/10.3390/s23031404
 
Li, Y., Zhu, L., Huang, A., Zhang, J., & Yuan, P. (2025). Multimodal MBC-ATT: cross-modality attentional fusion of EEG-fNIRS for cognitive state decoding. *Frontiers in Human Neuroscience, 19*. https://doi.org/10.3389/fnhum.2025.1660532
 
Li, D., Shin, H.-B., Yin, K., & Lee, S.-W. (2024). Domain-Incremental Learning Framework for Continual Motor Imagery EEG Classification Task. *2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)*, 1-5. https://doi.org/10.1109/embc53108.2024.10781886
 
Liu, H., Jin, X., Liu, D., Kong, W., Tang, J., & Peng, Y. (2025). Joint disentangled representation and domain adversarial training for EEG-based cross-session biometric recognition in single-task protocols. *Cognitive Neurodynamics, 19*. https://doi.org/10.1007/s11571-024-10214-w
 
Liu, Z., Chen, G., Li, Z., Qu, S., Knoll, A. C., & Jiang, C. (2023). D2IFLN: Disentangled Domain-Invariant Feature Learning Networks for Domain Generalization. *IEEE Transactions on Cognitive and Developmental Systems, 15*, 2269-2281. https://doi.org/10.1109/tcds.2023.3264615
 
Luo, G., Rao, H., An, P., Li, Y., Hong, R., Chen, W., & Chen, S. (2023). Exploring Adaptive Graph Topologies and Temporal Graph Networks for EEG-Based Depression Detection. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31*, 3947-3957. https://doi.org/10.1109/tnsre.2023.3320693
 
Portmann, H., & Morishima, Y. (2026). Systematic review of self-supervised foundation models for brain network representation using electroencephalography. https://doi.org/10.48550/arxiv.2602.03269
 
Rommel, C., Paillard, J., Moreau, T., & Gramfort, A. (2022). Data augmentation for learning predictive models on EEG: a systematic comparison. *Journal of Neural Engineering, 19*. https://doi.org/10.1088/1741-2552/aca220
 
Shi, Y., Jiang, A., Zhong, J., Li, M., & Zhu, Y. (2024). Multiclass Classification Framework of Motor Imagery EEG by Riemannian Geometry Networks. *IEEE Journal of Biomedical and Health Informatics, 29*, 935-947. https://doi.org/10.1109/jbhi.2024.3496757
 
Sun, L., Lian, Z., Liu, B., & Tao, J. (2024). HiCMAE: Hierarchical Contrastive Masked Autoencoder for Self-Supervised Audio-Visual Emotion Recognition. *ArXiv, abs/2401.05698*. https://doi.org/10.1016/j.inffus.2024.102382
 
Tang, H., Xie, S., Xie, X., Cui, Y., Li, B., Zheng, D., Hao, Y., Wang, X., Jiang, Y., & Tian, Z. (2024). Multi-Domain Based Dynamic Graph Representation Learning for EEG Emotion Recognition. *IEEE Journal of Biomedical and Health Informatics, 28*, 5227-5238. https://doi.org/10.1109/jbhi.2024.3415163
 
Tibermacine, I. E., Russo, S., & Napoli, C. (2026). Stiefel-SPD Manifold Graph Convolution for End-to-End EEG Learning. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 34*, 595-606. https://doi.org/10.1109/tnsre.2026.3652858
 
Wang, B., Miao, M., Zhang, K., Liu, W., Sheng, Z., Xu, B., & Hu, W. (2024). MSMGE-CNN: a multi-scale multi-graph embedding convolutional neural network for motor related EEG decoding. *Machine Learning: Science and Technology, 5*. https://doi.org/10.1088/2632-2153/ad9135
 
Wang, R., Jo, W., Zhao, D., Wang, W., Yang, B., Chen, G., & Min, B.-C. (2022). Husformer: A Multimodal Transformer for Multimodal Human State Recognition. *IEEE Transactions on Cognitive and Developmental Systems, 16*, 1374-1390. https://doi.org/10.1109/tcds.2024.3357618
 
Wei, J., Hong, D., Zhang, Z., Rong, D., He, Q., & Wang, Y. (2026). Self-Supervised Consistency Enhanced Disentangled Learning for Neural Decoding Generalization in Brain-Machine Interface. https://doi.org/10.48550/arxiv.2607.24023
 
Wu, D. (2025). Revisiting Euclidean alignment for transfer learning in EEG-based brain–computer interfaces. *Journal of Neural Engineering, 22*. https://doi.org/10.1088/1741-2552/addd49
 
Xiong, W., Lin, J., Li, J., Li, J., & Jiang, C. (2025). ALFEE: Adaptive Large Foundation Model for EEG Representation. *ArXiv, abs/2505.06291*. https://doi.org/10.48550/arxiv.2505.06291
 
Yan, K., Luo, X., Ye, L., Geng, W., He, J., Mu, J., Hou, X., Zan, X., J., Li, F., Zhang, L., & Chou, X.-J. (2025). Automated seizure detection in epilepsy using a novel dynamic temporal-spatial graph attention network. *Scientific Reports, 15*. https://doi.org/10.1038/s41598-025-01015-0
 
Ye, M., Chen, C. L. P., & Zhang, T. (2022). Hierarchical Dynamic Graph Convolutional Network With Interpretability for EEG-Based Emotion Recognition. *IEEE Transactions on Neural Networks and Learning Systems, 36*, 19489-19500. https://doi.org/10.1109/tnnls.2022.3225855
 
Zhang, X., Zheng, W., Li, Z., Yang, Y., Liu, W., Cai, H., Zhu, J., Liu, J., Hu, B., & Dong, Q. (2025). Constraint-Driven Causal Representation Learning for Vigilance Robust Estimation in Brain–Computer Interface. *IEEE Transactions on Neural Networks and Learning Systems, 36*, 20328-20342. https://doi.org/10.1109/tnnls.2025.3594434
 
Zhang, N., Jian, H., Li, X., Jiang, G., & Tang, X. (2025). LPGGNet: Learning from Local–Partition–Global Graph Representations for Motor Imagery EEG Recognition. *Brain Sciences, 15*. https://doi.org/10.3390/brainsci15121257
 
Zhang, J., Fang, J., Liu, S., Liu, D., Wu, H., & Long, J. (2024). Toward Cross-Brain-Computer Interface: A Prototype-Supervised Adversarial Transfer Learning Approach With Multiple Sources. *IEEE Transactions on Instrumentation and Measurement, 73*, 1-13. https://doi.org/10.1109/tim.2024.3451593
 
Zhao, Y., He, D., Ren, F., Xia, Q., Xu, L., Xie, G., Zhang, X., Yang, R., Zou, S., & Jiang, B. (2026). RMETNet: A cross-subject motor imagery EEG signal classification model based on TSLANet and riemannian geometry features. *PLOS One, 21*. https://doi.org/10.1371/journal.pone.0347671
 
Zheng, L., Jiang, Y., Liu, Z., & Yang, W. (2025). EEG-Based Monitoring of Pilot Training: Transferring Full-Cap Representations to Headphone-Style Electrode Positions. *Proceedings of the ACM on Human-Computer Interaction, 9*, 277 - 291. https://doi.org/10.1145/3773070
 
Zhou, Y., Zhang, X., Chen, G., & Sun, Y. (2024). Hierarchical Dynamic Dependency Graph Convolutional Networks for EEG Emotion Recognition. *2024 8th Asian Conference on Artificial Intelligence Technology (ACAIT)*, 935-942. https://doi.org/10.1109/acait63902.2024.11022291
 
