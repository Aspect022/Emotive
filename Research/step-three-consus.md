# **CogProfile-Net** Integration Roadmap

CogProfile-Net should integrate **unsupervised clustering**, **multi-channel time–frequency stacking**, and a **lightweight adaptive digital twin layer** because these three additions address your baseline’s main limitations: weak latent-state structure, limited temporal–spectral representation, and no personalized uncertainty-aware adaptation.

## System Priorities

The strongest evidence supports three immediate upgrades to the current 73.34% LightGBM baseline: replace purely handcrafted features with learned time–frequency representations, add a clustering layer that discovers sub-states inside each task, and convert the final classifier into a calibrated personalized model rather than a hard-label predictor  (Chakladar et al., 2024; Angkan et al., 2025; Bina et al., 2026; Guo et al., 2017).

Your hardware and montage are suitable for this strategy, but only within consumer-EEG limits. Emotiv EPOC-class systems have been used at 14 channels and 128 Hz for focused attention, working memory, and workload studies, and the exact AF3/F7/F3/FC5/T7/P7/O1/O2/P8/T8/FC6/F4/F8/AF4 layout has already supported cognitive profiling tasks  (Mohamed et al., 2018; Adewale & Panoutsos, 2019). But reliability is still lower than research-grade systems, with smaller effect sizes and lower signal quality for some paradigms, so the model should be optimized for robustness and uncertainty, not treated as clinically precise  (D’Angiulli et al., 2022; Williams et al., 2020).

## Expected Gains

| Evidence Strength | Claim |
|---|---|
| Evidence strength: Strong (8/10) | **Time–frequency deep models** usually outperform handcrafted EEG pipelines for cognitive-state decoding, with examples ranging from 88.33% in multi-task cognitive classification to 88.9% in cross-task workload and 93.1–96.2% in multi-channel scalogram-based affective decoding  (Manikandan et al., 2021; Zhang et al., 2019; Houssein et al., 2024)|
| Evidence strength: Moderate (7/10) | **Clustering or unsupervised representation learning** improves workload-state separation by learning cluster-friendly latent spaces, reaching 98.0% for 0-back vs 2-back with spatio-temporal deep clustering and 93.2% in wearable-EEG unsupervised load estimation  (Chakladar et al., 2024; Hassan et al., 2024)|
| Evidence strength: Moderate (7/10) | **Calibration and uncertainty layers** materially improve probability quality and rejection policies in EEG classifiers, and simple post-hoc temperature scaling remains a strong baseline while more expressive variants can further reduce ECE and improve risk-aware deployment  (Guo et al., 2017; Kull et al., 2019; Nzakuna et al., 2025)|

**Figure 1:** Evidence strength for three CogProfile-Net upgrade directions

The likely gain is not that every advanced method will transfer directly to a 5-class Emotiv task benchmark, but that your current PSD/DE representation is leaving separable structure unused. Studies using fused spatial, spectral, temporal, or connectivity structure repeatedly outperform single-domain baselines  (Pei et al., 2020; Kakkos et al., 2021; Angkan et al., 2025).

## Clustering Layer

For your pipeline, clustering should not replace supervised classification; it should refine it by discovering sub-states within each nominal task and by regularizing subject variability. Deep clustering studies show that low-dimensional latent spaces learned from temporal signals and scalp maps cluster workload states better than raw handcrafted features alone  (Chakladar et al., 2024). ISTDC is especially relevant because it concatenates temporal and spatial latent features before VBGMM clustering, and its multimodal latent space beats unimodal alternatives  (Chakladar et al., 2024).

A second useful line is microstate-style clustering. Task EEG can be segmented into short quasi-stable states using spatial-pattern clustering, Riemannian distance, and autoencoders, enabling unsupervised discretization of cognitive dynamics rather than only trial-level labels  (Pan et al., 2024). GFP-guided and hybrid-metric microstate methods further improve stability and explained variance, suggesting a route for extracting task-transition structure from 1 s sub-windows before final classification  (Wang et al., 2026; Liang et al., 2025).

A practical CogProfile-Net design is:

| Stage | Recommended Method | Rationale |
|---|---|---|
| Window embedding | temporal encoder + topographic encoder | matches spatio-temporal VAE logic  (Chakladar et al., 2024)|
| Unsupervised grouping | VBGMM or deep embedded clustering | cluster-friendly latent geometry  (Chakladar et al., 2024; Wang et al., 2026)|
| Supervised use | cluster IDs or soft memberships appended to classifier | converts hidden sub-states into predictive covariates |
| Personalization | subject-wise prototype updating | handles inter-subject variability  (Gupta et al., 2021; Bina et al., 2026)|

**Figure 2:** Recommended clustering insertion points in CogProfile-Net

More recent end-to-end clustering models strengthen this direction. EEGcMamba uses slice-aware state-space modeling plus instance- and cluster-level contrastive heads, and it outperformed prior unsupervised methods across 11 EEG benchmarks  (Chen et al., 2026). Explainable joint autoencoder-clustering pipelines also avoid the weak two-stage practice of first compressing and then clustering separately  (Ellis et al., 2023).

## Scalogram Stacking

The literature strongly supports moving from PSD/DE vectors to stacked time–frequency images or tensors. CWT-based scalograms repeatedly improve EEG classification because they preserve transient temporal-frequency structure that FFT-only features discard  (Houssein et al., 2024; Fouladi et al., 2022; Türk & Özerdem, 2019). In several domains, CWT or related time–frequency methods outperform alternative transforms or single-domain baselines  (Kolathod et al., 2025; Nouri et al., 2026; Alizadeh et al., 2023).

For CogProfile-Net, the most relevant representation is not a single-channel scalogram but a structured multi-channel stack. Studies have converted per-channel TFRs into fused image grids or three-way tensors so CNNs can learn spatial placement together with frequency evolution  (Guan et al., 2023; Yıldız & Zan, 2026; Wang et al., 2026; Zhang et al., 2019). Yıldız and Zan explicitly built 19-channel fused time–frequency grids and showed that performance depends strongly on the chosen TFR, with subject-wise splitting far more realistic than random splitting  (Yıldız & Zan, 2026).

A direct Emotiv-compatible design is to compute one 2D CWT or CQT image per 1 s or 2 s window for each of the 14 channels, place them on a 2D grid matching the headset topology, and stack either frequency bands or channel images as depth dimensions. This matches methods that build spatially informed time–frequency representations, topographic tensors, or multispectral topography maps before CNN classification  (Kolathod et al., 2025; Angkan et al., 2025; Karmakar et al., 2025).

| Representation | Supporting Evidence | Fit To Your Data |
|---|---|---|
| Channel-wise CWT grid | structured multi-channel fusion works well  (Yıldız & Zan, 2026)| high |
| Time-frequency-channel tensor | cross-task transfer with wavelet tensor features  (Guan et al., 2023)| high |
| Raw + topography dual stream | multi-domain fusion outperforms single domain  (Angkan et al., 2025)| very high |
| Graph-attentive connectivity image | task-independent generalization but lower raw accuracy  (Wei et al., 2025)| moderate |

**Figure 3:** Candidate time-frequency representations for 14-channel consumer EEG

The safest architecture is a dual-stream model: one branch for raw or lightly filtered 1D EEG, one branch for stacked scalograms. That directly follows multidomain representation learning work, where time-domain encoders and frequency/topography encoders are fused with attention and orthogonality constraints to increase inter-class separation and intra-class compactness  (Angkan et al., 2025).

## Digital Twin Layer

For your use case, the digital twin should be lightweight and operational, not a whole-brain simulator. The core twin concept is a personalized computational model continuously updated with incoming neural data to predict latent state, adapt decoding, and support individualized inference  (Bina et al., 2026; Wang et al., 2024). This is already framed in BCI literature as a way to reduce recalibration, absorb session drift, and move from empirical trial-and-error to adaptive model-based loops  (Bina et al., 2026; Zhang et al., 2026).

The Takahashi primate digital twin gives the most concrete operational template. Its variational recurrent architecture used top-down prediction plus bottom-up error-driven latent updates during data assimilation, and it generalized to unseen individuals while reproducing PSD and functional connectivity structure  (Takahashi et al., 2025). For CogProfile-Net, the analog is not ECoG simulation but a subject-specific latent-state updater that ingests recent EEG windows, behavioral metadata, and classifier residuals.

A practical twin for your system can be formalized as a state-space model:
\[
z_t = f_\theta(z_{t-1}, x_t, b_t), \quad \hat{y}_t = g_\phi(z_t), \quad \theta \leftarrow \theta - \eta \nabla \mathcal{L}(y_t,\hat{y}_t)
\]
where \(x_t\) is EEG embedding, \(b_t\) is behavioral metadata, \(z_t\) is personalized latent cognitive state, and \(\hat{y}_t\) is the calibrated task-profile distribution. This matches the latent-state and adaptive-updating framing proposed for personalized digital twins and uncertainty-aware trajectory models  (Soykan et al., 2026; Liu et al., 2025).

- **2017**
  - 1 paper:  (Guo et al., 2017)- **2018**
  - 1 paper:  (Mohamed et al., 2018)- **2019**
  - 3 papers:  (Adewale & Panoutsos, 2019; Zhang et al., 2019; Kull et al., 2019)- **2020**
  - 2 papers:  (Williams et al., 2020; Pei et al., 2020)- **2021**
  - 3 papers:  (Manikandan et al., 2021; Kakkos et al., 2021; Gupta et al., 2021)- **2022**
  - 1 paper:  (D’Angiulli et al., 2022)- **2024**
  - 4 papers:  (Chakladar et al., 2024; Houssein et al., 2024; Hassan et al., 2024; Pan et al., 2024)- **2025**
  - 3 papers:  (Angkan et al., 2025; Nzakuna et al., 2025; Liang et al., 2025)- **2026**
  - 2 papers:  (Bina et al., 2026; Wang et al., 2026)**Figure 4:** Research arc from EEG decoding to adaptive digital twins

EEG cognitive modeling has moved from handcrafted workload indices toward multimodal, adaptive, and personalized systems. The trajectory suggests that the next gain for CogProfile-Net is less likely to come from a single better classifier than from a loop that updates person-specific latent states and uncertainty over time  (Ismail & Karwowski, 2020; Hassan et al., 2024; Soykan et al., 2026).

## Calibration And Fusion

Because your output is a probabilistic cognitive profile, calibration is not optional. Modern neural networks are often overconfident, and temperature scaling is a strong, simple post-hoc baseline for aligning predicted confidence with empirical correctness  (Guo et al., 2017). Dirichlet calibration is more expressive for multiclass settings and improves ECE, log-loss, and Brier score across classifiers  (Kull et al., 2019). Entropy-based adaptive temperature scaling is specifically attractive for small-data settings like yours  (Balanya et al., 2022).

If you want abstention or “profile not confident” outputs, conformal prediction is useful, but plain confidence calibration does not automatically yield smaller efficient conformal sets  (Dabah & Tirer, 2024; Xi et al., 2024). For EEG specifically, ensemble or Bayesian uncertainty methods plus per-model temperature scaling improve ECE and permit risk–coverage operating points where uncertain windows are rejected  (Nzakuna et al., 2025; Hu et al., 2026).

Behavioral metadata should be fused late or in a dedicated side branch rather than concatenated naively with raw EEG. Multimodal studies repeatedly show that middle or late fusion outperforms early fusion when modalities capture partly distinct aspects of cognition  (Vortmann et al., 2022). In cognitive-state settings, multimodal and multi-domain fusion generally beats unimodal pipelines  (Li et al., 2025; Qiu et al., 2022; Rabbani & Islam, 2023).

| Output Layer | Best-Supported Choice | Why |
|---|---|---|
| Probability calibration | temperature scaling, then Dirichlet if needed | simplest strong baseline, multiclass upgrade  (Guo et al., 2017; Kull et al., 2019)|
| Reject option | conformal or entropy threshold | safer low-confidence handling  (Dabah & Tirer, 2024; Nzakuna et al., 2025)|
| Metadata fusion | late fusion MLP or gated branch | avoids destructive early mixing  (Vortmann et al., 2022)|
| Personal adaptation | twin-state update on subject history | reduces recalibration burden  (Bina et al., 2026; Soykan et al., 2026)|

**Figure 5:** Recommended calibrated multimodal output design for cognitive profiling

## Concrete Build Plan

1. **Backbone replacement.** Train a dual-stream network on 1 s windows: raw 1D EEG branch plus channel-topology scalogram branch, with subject-stratified splits and ablations against your PSD/DE LightGBM baseline  (Angkan et al., 2025; Yıldız & Zan, 2026).

2. **Clustering augmentation.** Learn latent embeddings, then add VBGMM or deep embedded clustering soft assignments as auxiliary targets and appended features  (Chakladar et al., 2024; Wang et al., 2026).

3. **Behavior branch.** Add reaction time, difficulty, and score through late fusion, because performance-related variation can materially alter workload labels rather than merely reflect them  (Khanam et al., 2023; Muke & Kozierkiewicz-Hetmanska, 2025).

4. **Calibration stack.** Fit temperature scaling on validation logits, report ECE, Brier score, NLL, and classwise reliability; if overconfidence persists, test Dirichlet or entropy-based scaling  (Guo et al., 2017; Kull et al., 2019; Balanya et al., 2022).

5. **Twin adaptation.** Maintain a subject memory of recent embeddings, confusion patterns, and metadata, and update a small recurrent state model online for personalized profile smoothing and drift correction  (Bina et al., 2026; Takahashi et al., 2025; Liu et al., 2025).

The single most important shift in the literature is from **single-domain supervised EEG classification** to **multi-domain, personalized, uncertainty-aware modeling**  (Angkan et al., 2025; Wang et al., 2024; Soykan et al., 2026). The single biggest open question for CogProfile-Net is whether these gains will persist under **subject-independent, consumer-grade, five-class cognitive profiling with realistic leakage control**, because many strong published accuracies still come from easier binary tasks, richer montages, or less stringent splits  (Yıldız & Zan, 2026; Adewale & Panoutsos, 2019; D’Angiulli et al., 2022).
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Adewale, Q., & Panoutsos, G. (2019). Mental Workload Estimation Using Wireless EEG Signals. *bioRxiv*. https://doi.org/10.1101/755033
 
Alizadeh, N., Afrakhteh, S., & Mosavi, M. (2023). Deep CNN‐based classification of motor imagery tasks from EEG signals using 2D wavelet transformed images of adaptively reconstructed signals from MVMD decomposed modes. *International Journal of Imaging Systems and Technology, 33*, 1988 - 2011. https://doi.org/10.1002/ima.22913
 
Angkan, P., Jalali, A., Hungler, P., & Etemad, A. (2025). Multidomain EEG Representation Learning With Orthogonal Mapping and Attention-Based Fusion for Cognitive Load Classification. *IEEE Transactions on Human-Machine Systems, 56*, 343-355. https://doi.org/10.1109/thms.2026.3651706
 
Balanya, S. A., Maroñas, J., & Ramos, D. (2022). Adaptive temperature scaling for Robust calibration of deep neural networks. *Neural Computing and Applications, 36*, 8073 - 8095. https://doi.org/10.1007/s00521-024-09505-4
 
Bina, M. M. H., Baghernezhad, S., Daliri, M., & Moradi, M. (2026). Neural Digital Twins: Toward Next-Generation Brain-Computer Interfaces. *ArXiv, abs/2601.01539*. https://doi.org/10.48550/arxiv.2601.01539
 
Chakladar, D. D., Roy, P. P., & Chang, V. I. (2024). Integrated Spatio-Temporal Deep Clustering (ISTDC) for cognitive workload assessment. *Biomed. Signal Process. Control., 89*, 105703. https://doi.org/10.1016/j.bspc.2023.105703
 
Chen, J., Pi, D., Jiang, X., Gao, F., Yang, Q., & Chen, Y. (2026). EEGcMamba: EEG Clustering via State Space Model. *IEEE Transactions on Artificial Intelligence, 7*, 2292-2306. https://doi.org/10.1109/tai.2025.3614230
 
D’Angiulli, A., Lockman-Dufour, G., & Buchanan, D. (2022). Promise for Personalized Diagnosis? Assessing the Precision of Wireless Consumer-Grade Electroencephalography across Mental States. *Applied Sciences*. https://doi.org/10.3390/app12136430
 
Dabah, L., & Tirer, T. (2024). On Temperature Scaling and Conformal Prediction of Deep Classifiers. https://doi.org/10.48550/arxiv.2402.05806
 
Ellis, C. A., Miller, R., & Calhoun, V. (2023). A Convolutional Autoencoder-based Explainable Clustering Approach for Resting-State EEG Analysis. *bioRxiv*. https://doi.org/10.1109/embc40787.2023.10340375
 
Fouladi, S., Safaei, A., Mammone, N., Ghaderi, F., & Ebadi, M. J. (2022). Efficient Deep Neural Networks for Classification of Alzheimer’s Disease and Mild Cognitive Impairment from Scalp EEG Recordings. *Cognitive Computation, 14*, 1247 - 1268. https://doi.org/10.1007/s12559-022-10033-3
 
Guan, K., Zhang, Z., Liu, T., & Niu, H.-J. (2023). Cross-Task Mental Workload Recognition Based on EEG Tensor Representation and Transfer Learning. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31*, 2632-2639. https://doi.org/10.1109/tnsre.2023.3277867
 
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ArXiv, abs/1706.04599*. https://doi.org/10.48550/arxiv.1706.04599
 
Gupta, A., Siddhad, G., Pandey, V., Roy, P., & Kim, B.-G. (2021). Subject-Specific Cognitive Workload Classification Using EEG-Based Functional Connectivity and Deep Learning. *Sensors (Basel, Switzerland), 21*. https://doi.org/10.3390/s21206710
 
Hassan, I., Zolezzi, M., Khalil, H., Saady, R. M. A., Pedersen, S., & Chowdhury, M. (2024). Cognitive Load Estimation Using a Hybrid Cluster-Based Unsupervised Machine Learning Technique. *IEEE Access, 12*, 118785-118801. https://doi.org/10.1109/access.2024.3428691
 
Hassan, J., Reza, S., Ahmed, S. U., Anik, N. H., & Khan, M. O. (2024). EEG workload estimation and classification: a systematic review. *Journal of Neural Engineering, 22*. https://doi.org/10.1088/1741-2552/ad705e
 
Houssein, E. H., Hammad, A., Samee, N. A., Alohali, M. A., & Ali, A. (2024). TFCNN-BiGRU with self-attention mechanism for automatic human emotion recognition using multi-channel EEG data. *Cluster Computing, 27*, 14365 - 14385. https://doi.org/10.1007/s10586-024-04590-5
 
Hu, J., Rahman, M. M. U., & Laleg‐Kirati, T. (2026). Calibrated Uncertainty Estimation Based on Bayesian Models for EEG Epileptic Seizure Detection. *IEEE Sensors Journal, 26*, 13533-13545. https://doi.org/10.1109/jsen.2026.3674940
 
Ismail, L., & Karwowski, W. (2020). Applications of EEG indices for the quantification of human cognitive performance: A systematic review and bibliometric analysis. *PLoS ONE, 15*. https://doi.org/10.1371/journal.pone.0242857
 
Kakkos, I., Dimitrakopoulos, G. N., Sun, Y., Yuan, J., Matsopoulos, G., Bezerianos, A., & Sun, Y. (2021). EEG Fingerprints of Task-Independent Mental Workload Discrimination. *IEEE Journal of Biomedical and Health Informatics, 25*, 3824-3833. https://doi.org/10.1109/jbhi.2021.3085131
 
Karmakar, S., Kamilya, S., Koley, C., & Pal, T. (2025). A Deep Learning Technique for Real-Time Detection of Cognitive Load Using Optimal Number of EEG Electrodes. *IEEE Transactions on Instrumentation and Measurement, 74*, 1-11. https://doi.org/10.1109/tim.2024.3509604
 
Khanam, F., Ahmad, M., & Hossain, A. (2023). Investigation of the neural correlation with task performance and its effect on cognitive load level classification. *PLOS ONE, 18*. https://doi.org/10.1371/journal.pone.0291576
 
Kolathod, M. J. T., Sudeep, P., & Sanjay, M. (2025). Motor imagery task classification using spatial–time–frequency features of EEG signals: a deep learning approach for improved performance. *Evolving Systems, 16*. https://doi.org/10.1007/s12530-025-09696-8
 
Kull, M., Perello-Nieto, M., Kängsepp, M., Filho, T. S., Song, H., & Flach, P. A. (2019). Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration. 12295-12305. https://doi.org/10.48550/arxiv.1910.12656
 
Li, Y., Zhu, L., Huang, A., Zhang, J., & Yuan, P. (2025). Multimodal MBC-ATT: cross-modality attentional fusion of EEG-fNIRS for cognitive state decoding. *Frontiers in Human Neuroscience, 19*. https://doi.org/10.3389/fnhum.2025.1660532
 
Liang, J., Yin, X., & Lin, M. (2025). An enhanced microstate clustering algorithm based on canopy, K-means, and genetic simulated annealing. *Biomedical Physics & Engineering Express, 11*. https://doi.org/10.1088/2057-1976/adda50
 
Liu, Y., Yan, K., S., Deng, H., & Kong, J. (2025). AI-Driven Digital Twin Architecture for Multimodal Prediction and Adaptive Intervention in Cognitive Aging. *JMIR AI, 5*. https://doi.org/10.2196/87768
 
Manikandan, S., Madhumitha, R., SornaMeena, M., & Sruthi, R. (2021). Sequential Convolutional Neural Networks for classification of cognitive tasks from EEG signals. *Appl. Soft Comput., 111*, 107664. https://doi.org/10.1016/j.asoc.2021.107664
 
Mohamed, Z., Halaby, M. E., Said, T., Shawky, D., & Badawi, A. H. (2018). Characterizing Focused Attention and Working Memory Using EEG. *Sensors (Basel, Switzerland), 18*. https://doi.org/10.3390/s18113743
 
Muke, P. Z., & Kozierkiewicz-Hetmanska, A. (2025). Machine Learning Techniques to Improve the Cognitive Workload Classification Using Multimodal Sensors’ Data. *IEEE Access, 13*, 173415-173443. https://doi.org/10.1109/access.2025.3616788
 
Nouri, Z., Charmin, A., Kalbkhani, H., & Barghandan, S. (2026). Multivariate synchrosqueezing transform and time-frequency attention for mental workload classification from EEG signals. *Scientific Reports, 16*. https://doi.org/10.1038/s41598-025-34783-w
 
Nzakuna, P. S., Gallo, V., Carratù, M., Paciello, V., Pietrosanto, A., & Lay-Ekuakille, A. (2025). Learned-Weight Ensemble Monte Carlo DropBlock for Uncertainty Estimation and EEG Classification. *IEEE Open Journal of Instrumentation and Measurement, 4*, 1-13. https://doi.org/10.1109/ojim.2025.3638922
 
Pan, S., Shen, T., Lian, Y., & Shi, L. (2024). A Task-Related EEG Microstate Clustering Algorithm Based on Spatial Patterns, Riemannian Distance, and a Deep Autoencoder. *Brain Sciences, 15*. https://doi.org/10.3390/brainsci15010027
 
Pei, Z., Wang, H., Bezerianos, A., & Li, J. (2020). EEG-Based Multiclass Workload Identification Using Feature Fusion and Selection. *IEEE Transactions on Instrumentation and Measurement, 70*, 1-8. https://doi.org/10.1109/tim.2020.3019849
 
Qiu, L., Zhong, Y., He, Z., & Pan, J. (2022). Improved classification performance of EEG-fNIRS multimodal brain-computer interface based on multi-domain features and multi-level progressive learning. *Frontiers in Human Neuroscience, 16*. https://doi.org/10.3389/fnhum.2022.973959
 
Rabbani, M. H. R., & Islam, S. M. R. (2023). Deep learning networks based decision fusion model of EEG and fNIRS for classification of cognitive tasks. *Cognitive Neurodynamics*, 1-18. https://doi.org/10.1007/s11571-023-09986-4
 
Soykan, B., Koksalmis, G. H., Huang, H., & Brattain, L. (2026). Toward Personalized Digital Twins for Cognitive Decline Assessment: A Multimodal, Uncertainty-Aware Framework. *ArXiv, abs/2604.27217*. https://doi.org/10.48550/arxiv.2604.27217
 
Takahashi, Y., Idei, H., Komatsu, M., Tani, J., Tomita, H., & Yamashita, Y. (2025). Digital twin brain simulator for real-time consciousness monitoring and virtual intervention using primate electrocorticogram data. *NPJ Digital Medicine, 8*. https://doi.org/10.1038/s41746-025-01444-1
 
Türk, Ö., & Özerdem, M. S. (2019). Epilepsy Detection by Using Scalogram Based Convolutional Neural Network from EEG Signals. *Brain Sciences, 9*. https://doi.org/10.3390/brainsci9050115
 
Vortmann, L.-M., Ceh, S., & Putze, F. (2022). Multimodal EEG and Eye Tracking Feature Fusion Approaches for Attention Classification in Hybrid BCIs. https://doi.org/10.3389/fcomp.2022.780580
 
Wang, J., Yang, C., Lei, W., & Xiao, Y. (2026). GFP-DWHML: An EEG microstate analysis method using GFP-based dynamic weighting for hybrid metric learning. *2026 29th International Conference on Computer Supported Cooperative Work in Design (CSCWD)*, 1666-1671. https://doi.org/10.1109/cscwd68734.2026.11581750
 
Wang, Z., Huang, G., Chen, Z., Liu, X., Liu, Y., & Hong, K. (2026). Decoupled Bidirectional Spatio-Temporal Fusion Network for Hybrid EEG-fNIRS Cognitive Task Classification. *Brain Sciences, 16*. https://doi.org/10.3390/brainsci16020241
 
Wang, H. E., Triebkorn, P., Breyton, M., Dollomaja, B., Lemaréchal, J.-D., Petkoski, S., Sorrentino, P., Depannemaecker, D., Hashemi, M., & Jirsa, V. (2024). Virtual brain twins: from basic neuroscience to clinical use. *National Science Review, 11*. https://doi.org/10.1093/nsr/nwae079
 
Wei, C., Zhao, X., Song, Y., & Liu, Y. (2025). Task-Independent Cognitive Workload Discrimination Based on EEG with Stacked Graph Attention Convolutional Networks. *Sensors (Basel, Switzerland), 25*. https://doi.org/10.3390/s25082390
 
Williams, N. S., McArthur, G., De Wit, B., Ibrahim, G., & Badcock, N. (2020). A validation of Emotiv EPOC Flex saline for EEG and ERP research. *PeerJ, 8*. https://doi.org/10.7717/peerj.9713
 
Xi, H., Huang, J., Liu, K., Feng, L., & Wei, H. (2024). Does confidence calibration improve conformal prediction?. *Trans. Mach. Learn. Res., 2025*. https://doi.org/10.48550/arxiv.2402.04344
 
Yıldız, A., & Zan, H. (2026). Deep Learning-Based Alzheimer’s Disease Detection from Multi-Channel EEG Using Fused Time–Frequency Image Grids. *Diagnostics, 16*. https://doi.org/10.3390/diagnostics16050746
 
Zhang, P., Wang, X., Zhang, W., & Chen, J. (2019). Learning Spatial–Spectral–Temporal EEG Features With Recurrent 3D Convolutional Neural Networks for Cross-Task Mental Workload Assessment. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27*, 31-42. https://doi.org/10.1109/tnsre.2018.2884641
 
Zhang, T., Zhang, R., Zeng, X., Zeng, M., Xu, Y., Xiong, Y., Zhang, G., Guo, D., & Yao, D. (2026). A new BCI paradigm based on biological brain - digital twin brain dialogue.. *Cognitive neurodynamics, 20 1*, 70. https://doi.org/10.1007/s11571-026-10439-x
 
