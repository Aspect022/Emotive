# Clustering, Scalogram Stacking, and Digital Twin Integration for EEG Cognitive Profiling

## Time–Frequency Representations of EEG Signals

Electroencephalography (EEG) produces non-stationary, multi–frequency brain signals.  Time–frequency transforms such as the Continuous Wavelet Transform (CWT) are widely used to capture both spectral and temporal features.  The CWT of a signal \(x(t)\) with mother wavelet \(\psi(t)\) is defined as  
\[
X_{\mathrm{CWT}}(b,a) \;=\; \frac{1}{\sqrt{a}} \int_{-\infty}^{+\infty} x(t)\,\psi^*\!\Bigl(\frac{t-b}{a}\Bigr)\,dt \,,
\]  
producing a **scalogram** \(X_{\mathrm{CWT}}(b,a)\) (time \(b\) vs scale \(a\)).  In discrete form for sampled signal \(x[n]\), the CWT is 
\[
W_{\mathrm{CWT}}(b,a) \;=\; \frac{1}{\sqrt{a}} \sum_{n=0}^{N-1} x[n]\,\psi^*\!\Bigl(\frac{n-b}{a}\Bigr)\,,
\]  
which yields a 2D matrix of size (time \(\times\) scales).  This scalogram acts like an image representation of the EEG epoch, where vertical axis relates to frequency (via scale) and horizontal axis to time.  

The CWT provides *multi-resolution analysis*: at high frequencies (small scale) it offers fine temporal resolution, and at low frequencies (large scale) fine frequency resolution.  In contrast, a short-time Fourier transform (STFT) uses a fixed window length, trading off time versus frequency resolution uniformly.  Hence CWT is well-suited to EEG, which contains both brief high-frequency bursts (e.g. gamma, 30–100 Hz) and slow waves (e.g. delta, 0.5–4 Hz).  As noted by prior work, “CWT offers a multi-resolution analysis capability that enables simultaneous localization of high-frequency transients of short duration and low-frequency components of long duration”.  Thus scalogram images visualize how the spectral content of EEG varies over time, capturing oscillatory patterns relevant to cognitive tasks.  For example, Shanmugapriya and Mohan (2023) highlight that scalograms reveal subtle speech-related frequency changes; similarly, in EEG, brainwave band-power modulations and transient events are made explicit by a scalogram .

Practically, one selects a suitable mother wavelet (e.g. Morlet) and a range of scales (often covering canonical EEG bands).  The resulting 2D time–frequency map (scalogram) is often converted to a normalized image (e.g. 0–1 grayscale or color).  These scalograms can then be processed by convolutional neural networks (CNNs) or other image-based algorithms.  Formally, given a preprocessed EEG window \(x(t)\), one computes  
\[
W(b,a) = \mathrm{CWT}\{x(t)\}
\]  
and visualizes \(|W(b,a)|^2\) (power) or \(|W(b,a)|\) as an image. 

## Scalogram Stacking and CNN-Based Classification

**Scalogram stacking** refers to combining multiple channel scalograms into a single multi-channel image for deep learning.  In multichannel EEG (e.g. the Emotiv EPOC+ with 14 electrodes), each channel yields its own scalogram.  To exploit spatial patterns and inter-channel dependencies, researchers often “stack” these channel-wise scalograms either as separate image layers or by concatenating them.  For instance, Mohanto *et al.* (2026) segment EEG into epochs, compute CWT scalograms per selected channels, and then stack them to form one composite input image.  This creates a 2D/3D tensor (for example, a 224×224 image with 3 layers for 3 channels, or one grayscale image showing channels side-by-side) that a CNN can ingest.  The pipeline in their study explicitly lists “(v) stacking multiple channel scalograms to form a single image” as a key step.  Stacked scalograms capture both the time–frequency content of each electrode and the spatial layout (depending on how channels are arranged in the stack), enabling CNN filters to learn patterns across time, frequency, and space simultaneously. 

For CogProfile-Net, a plausible approach is to take each 1-second EEG sub-window from all 14 channels, compute its CWT scalogram (e.g. using a Morlet wavelet covering ~1–45 Hz), and then arrange these 14 scalograms into a single image.  One could, for example, create a 14-channel RGB image by assigning 3 channels per color plane (or using 14 gray-scale layers in a 3D tensor).  Alternatively, scalograms could be tiled: e.g. 14 grayscale images concatenated in a grid.  Whatever format, the goal is to feed the resulting image(s) into a CNN.  Deep models like VGG16, ResNet, EfficientNet etc. (pretrained on ImageNet) have been successfully repurposed for EEG scalogram classification via transfer learning. 

Studies report strong performance with scalogram+CNN. For example, Shabber *et al.* (2025) applied scalogram-CNN models (transfer-learned from ImageNet) to dysarthric speech signals and achieved **90–99%** accuracy.  For cognitive tasks, Doğan (2023) transformed EEG to wavelet images for workload classification; EfficientNet-B0 achieved ~67.5% accuracy on Multi-Attribute Task Battery (MATB) difficulty levels.  These results, while for different domains, indicate that deep CNNs on scalograms can extract rich features.  We expect similar efficacy for CogProfile-Net’s five cognitive states.  

Key recommendations for implementation:

- **CWT parameters**: Use a Morlet or complex Gaussian wavelet; choose scales to cover 0.5–45 Hz (matching preprocessed bandpass).  Adjust scale resolution to balance detail vs. redundancy.  
- **Scalogram resolution**: E.g. compute scalogram on 1-second windows at 128 Hz → 128 time points.  A common approach is to interpolate or resize the scalogram to a fixed image size (e.g. 128×128 or 256×256) for CNN input.  
- **Channel stacking**: Experiment with spatial arrangements.  If channels have known montage positions (AF3, F7, ...), one can place their scalograms in a layout reflecting scalp topology.  Otherwise, simply stacking in a fixed order (e.g. frontal vs posterior grouping) also works.  The key is consistency.  
- **CNN choice**: Pretrained 2D CNNs (VGG16, ResNet, EfficientNet) are popular.  Fine-tuning improves results.  Use both pretrained ImageNet weights and domain-specific augmentation (flips, noise).  Early layers capture generic edges in time–frequency, deeper layers capture EEG-specific features.

## Clustering EEG Embeddings for Cognitive Profiling

Beyond supervised classification, **unsupervised clustering** of EEG-derived features can reveal latent cognitive profiles.  After the CNN (or other feature extractor) outputs a learned embedding or probability distribution for each EEG window, clustering can group similar cognitive or neural states across windows/subjects.  In practice, one might extract the penultimate-layer feature vector (or the 5-dimensional softmax probabilities) from CogProfile-Net for each trial, then apply clustering algorithms (k-means, hierarchical, DBSCAN, etc.) to these vectors across the dataset.

Several works have shown clustering utility in EEG.  Lekati *et al.* (2025) clustered cognitive load profiles of learners using K-means on performance metrics and EEG features; they confirmed 3 clusters (“Core Support Needed”, “Developing”, “Advanced”) with Silhouette and Elbow analyses.  In cognitive load studies, frontiers research performed K-means on normalized band-power (Delta–Gamma) features, choosing \(k=3\) via the elbow method.  They used cluster-EEG relations to explore which frequency bands dominated each group.  Formally, given feature vectors \(\mathbf{f}_i\) (e.g. CNN outputs or EEG band powers) for samples \(i\), k-means clustering minimizes 
\[
\sum_{j=1}^k \sum_{i \in C_j} \|\mathbf{f}_i - \boldsymbol{\mu}_j\|^2,
\] 
where \(\boldsymbol{\mu}_j\) is the centroid of cluster \(j\).  The optimal \(k\) can be chosen by the Elbow (inertia vs. \(k\)) or Silhouette method.  Additional clustering methods (e.g. Gaussian Mixture Models, spectral clustering, or topological data analysis such as MapperEEG) could be explored, especially given EEG’s complex structure. 

For CogProfile-Net, clustering can serve two purposes: (1) **Latent profiling** – to discover subtypes of cognitive response.  For example, some subjects might have consistently similar EEG patterns across tasks, forming a cluster of “high cognitive control” vs “low control” profiles.  (2) **Calibration** – to refine the probabilistic cognitive profile.  If a subject’s recent windows consistently fall into a cluster associated with a particular task (or difficulty level), one could adjust the final label probabilities accordingly.  

To integrate clustering: After obtaining feature vectors or CNN outputs for all sub-windows, apply dimensionality reduction (e.g. PCA or UMAP) for visualization, and use K-means (or another) to segment the data.  Evaluate cluster validity via silhouette score \(s = (b(i)-a(i))/\max\{a(i),b(i)\}\) (where \(a(i)\) is intra-cluster distance and \(b(i)\) nearest-cluster distance).  Lekati *et al.* confirmed 3 clusters via Elbow and Silhouette analysis, which is a good model: compare varying \(k\) to see stable clustering.  Clustering metrics (e.g. Davies–Bouldin index) can also guide \(k\).  Once clusters are found, one can label them (manually or via associated task labels) or simply use the cluster ID as a latent trait feature.  

For example, a possible pipeline step: collect all 1-s sub-window embeddings \(\mathbf{f}_{i,t}\) for subject \(i\) at trial \(t\), and cluster across all subjects.  Suppose cluster 1 corresponds mainly to high mental workload patterns, cluster 2 to rest-like EEG, etc.  Then a given trial’s windows might have cluster-membership frequencies \(\{p_j\}\), which can augment the cognitive profile (e.g. “Subject A’s trial 37: 70% windows in cluster 1, 30% in cluster 2”).  These cluster distributions could be fed into a secondary classifier or used as meta-features in mapping to CHC/O*NET. 

## Digital Twin Frameworks for EEG and Cognition

**Digital twins** in neuroscience are computational models that emulate an individual’s brain physiology to predict or interpret neural signals.  In essence, a digital twin creates a personalized “virtual brain” governed by biophysical laws.  Such models can be used to generate **digital biomarkers** that tie observed EEG to underlying neural health or cognitive state.  For instance, in dementia research, Bonmassar *et al.* (2025) used a digital twin of Alzheimer’s disease (the DADD model) to map pathophysiological parameters to EEG/MEG signals.  Their approach “leveraged a digital twin model to derive digital biomarkers linking neurodegeneration mechanisms to alterations in neural activity across multiple modalities”.  These digital biomarkers were then used in transfer learning to classify Mild Cognitive Impairment across different centers, achieving ~83% accuracy on EEG data (vs 58% with standard EEG features).  Crucially, “digital twins, i.e. personalized models of brain activities, have been proposed to […] encapsulate pathophysiological knowledge about disease evolution”, enabling cross-modal generalization.

In cognitive profiling, a digital twin could similarly encode task-related brain dynamics.  One could use platforms like *The Virtual Brain (TVB)*, which model large-scale brain networks with neural mass models.  For example, the DADD twin represents the cortex as 76 interconnected regions (TVB nodes) with Jansen–Rit dynamics; it simulates progression of synaptic and connectivity degeneration via parameters \(lp,\,cp,\,np\).  Specifically, synaptic degeneration is modeled by a parameter \(lp\) that linearly scales the inhibitory/excitatory time constants:  
\[
\tau_i \to \tau_i^{HC} + lp\;(\tau_i^{max}-\tau_i^{HC}), 
\qquad
\tau_e \to \tau_e^{HC} + lp\;(\tau_e^{min}-\tau_e^{HC}),\quad 0<lp<1,
\] 
and connectivity degeneration by parameters \(cp, np\) affecting the structural connectome weights.  Though these equations come from pathology modeling, analogous physiological parameters (e.g. arousal state, network coupling) could be tuned to match EEG patterns under different cognitive tasks.  

One practical way to integrate a digital twin into CogProfile-Net is via **model-based simulation**.  For each subject, one might calibrate a simplified neural mass model on baseline EEG, then simulate its response to cognitive load.  The difference between simulated and actual EEG (or derived features) could refine the cognitive state estimate.  Alternatively, one can use a digital twin to generate synthetic EEG-like data under labeled cognitive states, augmenting the training set for the CNN.  BrainTwin-AI (Saha *et al.*, 2026) implements a multimodal twin that fuses structural MRI (ViT++ tumor analysis) and real-time EEG: it uses a BiLSTM on EEG to classify cognitive conditions (Relaxed/Stress/Fatigue) and fuses this with MRI embeddings for risk scoring.  They achieved an EEG-based macro-F1 of 0.94, demonstrating the feasibility of neural nets on EEG to detect cognitive states.  While BrainTwin targets clinical applications, the concept of a “constantly updating patient-specific model” can inspire how to make CogProfile-Net adaptive: continuously update subject model with incoming EEG to predict future cognitive profile.  

Kong *et al.* (2026) provide a broader AI-driven digital twin architecture for cognitive health.  Their framework ingests multimodal physiological data (EEG, HRV, gait, etc.), uses CNNs/LSTMs to predict cognitive risk trajectories, and implements an adaptive feedback loop.  In CogProfile-Net, the “perception” layer is the EEG acquisition, the “analytics” layer can include our scalogram-CNN plus clustering, and the “decision” layer could use reinforcement learning or rule-based mapping to CHC abilities.  The key insight is viewing the EEG classifier as one component of a larger digital twin that continually updates its internal state of the subject. 

In summary, integrating digital twins means using subject-specific physiologically grounded models (possibly simplified neural mass or graph models) to *interpret* EEG in terms of underlying cognitive state.  This could involve:
- **Forward modeling**: Simulate EEG features for each cognitive class, to improve classifier calibration.  
- **Back-inference**: Fit twin parameters (network coupling, noise) to observed EEG, and interpret parameter changes.  
- **Data fusion**: If other data (MRI, behavioral) exist, combine them with EEG in a unified model (as in BrainTwin).  

## Proposed CogProfile-Net Integration Pipeline

Building on these techniques, we propose the following end-to-end pipeline for CogProfile-Net:

1. **Preprocessing:** Apply existing CAR referencing, 0.5–45 Hz bandpass, and artifact gating to raw EEG (Emotiv EPOC+).  Segment the signal into 4 s trials (already done), then form overlapping 1 s sub-windows (128 samples each).

2. **Scalogram Generation:** For each 1 s sub-window and each of the 14 channels, compute the CWT scalogram (e.g. Morlet wavelet, scales spanning 1–45 Hz).  Normalize the scalogram (e.g. log-power scaling) and resize to a fixed image resolution.  For channel stacking, arrange all 14 scalograms into a composite image; for instance, create a 4×4 grid (16 slots) using 14 scalograms and 2 blank placeholders, or an RGB image with ~5 channels per color.  Consistency in layout is crucial.  (An alternative is to treat each channel’s scalogram as an input “patch” fed through a shared CNN and then merged, but full image stacking is simpler.)

3. **Deep Feature Extraction / Classification:** Feed the stacked scalogram images into a CNN.  Options include fine-tuning a pre-trained architecture (e.g. VGG16, EfficientNet-B0) or training a custom CNN from scratch.  The network outputs a probability vector \(\mathbf{p}=(p_0,\dots,p_4)\) over the five cognitive tasks.  Training is supervised using the labeled tasks; cross-entropy loss and backpropagation adjust the weights.  Transfer learning (freezing early layers, tuning later ones) can accelerate training and leverage large image datasets.  

4. **Feature Embedding Storage:** Extract the CNN’s penultimate-layer embeddings (a fixed-length vector per window) and/or the softmax probabilities for each window.  Collect these feature vectors for all windows across all subjects and tasks.

5. **Unsupervised Clustering:** Apply clustering (e.g. K-means or Gaussian Mixture) to the set of feature vectors.  Normalize features (e.g. z-score) beforehand.  Use Elbow and Silhouette methods to select \(k\).  Prior work suggests \(k=3\) or \(k=4\) often suffices for cognitive states.  Once clusters are found, each sub-window inherits a cluster label.  Compute cluster-level statistics (e.g. the fraction of windows in each cluster for a given trial/subject) and interpret clusters by comparing to known task labels.  

6. **Digital Twin Calibration:** (Optional but recommended for robustness.) Build a simple neural-mass or connectivity-based model of the subject’s EEG.  For example, calibrate a TVB-like network model on the subject’s resting EEG (adjust global coupling \(G\) or local time constants \(\tau_i,\tau_e\) until simulated EEG matches statistics of real EEG).  Then, simulate this model under each of the five task conditions (which may correspond to adjusting input drive or noise levels).  Compare the simulated scalogram features to the real ones.  Use this to refine class probabilities: if a trial’s EEG closely matches the twin’s “task 2” simulation, boost \(p_2\).  More formally, one could compute a **digital biomarker** from the model (e.g. the fitted neurodegeneration parameters lp/cp/np in the DADD model), and input that to a small classifier or ensemble with the CNN output.  This step leverages the twin’s physiologically-grounded prediction to correct the purely data-driven CNN.

7. **Probabilistic Profile Aggregation:** Aggregate window-level predictions into trial-level and subject-level profiles.  For each trial, average the CNN probability vectors (optionally weighted by cluster confidence or twin alignment) to get \(\bar{\mathbf{p}} = (\bar p_0,\dots,\bar p_4)\).  This yields a *soft cognitive label* for that trial.  Across trials and tasks, form the subject’s cognitive profile as the distribution of inferred task probabilities.

8. **CHC/O*NET Mapping:** Finally, map the probabilistic cognitive profile to career aptitudes (per CHC theory and O*NET skills) as planned.  This mapping can itself be a weighted scoring model.  

Throughout, evaluate performance with standard metrics: classification accuracy on held-out windows/trials, macro-averaged F1, and cross-validation.  Given subject variability, use **subject-dependent** evaluation (leave-subject-out cross-validation) to test generalization.  The baseline 73.3% accuracy was subject-dependent; we expect the CNN with scalograms could surpass this, potentially reaching 80–90% as in related work.  For clustering, report silhouette score and qualitative cluster interpretation (e.g. “cluster 1 dominated by high math-skill subjects”).  For digital twin aspects, one could assess how twin-derived features improve classification (e.g. via McNemar’s test on accuracy differences). 

### Key Formulas and Metrics

- **CWT scalogram:** As above, equations (1) and (2) in .  
- **Spectral Power (PSD):** The power of a Fourier transform component \(X(f)\) is \(|X(f)|^2\).  While we focus on wavelets, note that bandpower in a frequency band can be computed by integrating scalogram power over that band.  
- **Clustering:** K-means minimizes intra-cluster distance; silhouette coefficient \(s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}\) measures cluster quality.  
- **Digital Twin (DADD model):** Equations (1) and (2) in  define how degeneration parameters update time constants and connectivity.  These biophysical formulas illustrate how model parameters map to signal features; analogous adjustments could represent cognitive load in a twin.  
- **Classification metrics:** Accuracy \(=\frac{TP+TN}{TP+FP+TN+FN}\), F1 \(=2TP/(2TP+FP+FN)\), and cross-entropy loss \(L=-\sum_{c}y_c\log p_c\) for training CNN.  The BrainTwin study reports macro-F1=0.94 for EEG cognitive-state classification. 

## Implementation Recommendations

- **Libraries:** Use PyWavelets or MNE for CWT computation; PyTorch or TensorFlow for CNN; scikit-learn for clustering; The Virtual Brain (TVB) or NeuroML for building a twin.  Python allows integration via frameworks (e.g. TVB simulator + PyTorch).  
- **Computational cost:** CWT is heavy (computing wavelet at every time/scale).  Consider downsampling or using a coarser scale grid.  The “optCWT” approach suggests adjusting wavelet length and using a hop size to reduce redundancy.  For 1-s windows (128 samples), computation is moderate.  CNN training on ~37k windows is feasible on a GPU.  
- **Data augmentation:** Time-shift or add small noise to EEG before scalogram to augment training.  The SDSF-EGAN method (stacking scalogram features with GAN) could also be used to synthesize underrepresented class examples.  
- **Normalization:** Per-channel scalograms should be normalized (e.g. z-scoring or min-max) to account for amplitude differences between electrodes or subjects.  Also apply batch normalization in the CNN for stable training.  
- **Validation:** Use stratified cross-validation by subject.  In addition to accuracy, track per-class recall/F1, since tasks may not be balanced.  
- **Interpretability:** Grad-CAM or similar can highlight which time-frequency regions in the scalogram the CNN used.  This can validate that the model focuses on plausible EEG rhythms (e.g. parietal alpha for sustained attention).  

## Conclusion

By combining **scalogram stacking**, **deep learning**, **clustering**, and **digital twin modeling**, CogProfile-Net can more richly capture EEG–cognition relationships.  Scalograms ensure time-frequency detail; stacking preserves spatial context.  Clustering on learned embeddings uncovers latent subject profiles.  Digital twins inject physiological realism and cross-domain robustness.  Together, these enhancements promise higher accuracy and more nuanced cognitive profiles.  Implementation should proceed modularly, with careful tuning of CWT parameters, CNN architecture, clustering hyperparameters, and twin calibration.  Evaluation metrics (accuracy, F1, silhouette, etc.) will quantify gains.  This integrated pipeline leverages state-of-the-art research to advance EEG-based cognitive state classification and personalized profiling.

**Sources:** We surveyed recent literature on EEG time–frequency analysis, EEG clustering in cognitive contexts, and digital twin models for brain signals, alongside specific deep-learning studies on EEG classification. These references provided the formulas, methods, and results informing the above design.