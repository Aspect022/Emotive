CogProfile-Net: A Comprehensive Architecture for EEG-Based Cognitive Profiling via Scalogram Stacking, Latent Clustering, and Digital Twin Integration
Introduction to End-to-End Neural Cognitive Profiling
The decoding of human cognitive states from electroencephalographic (EEG) signals constitutes a formidable challenge in computational neuroscience, psychological profiling, and the design of passive brain-computer interfaces (BCIs). The development of the CogProfile-Net architecture represents a fundamental structural evolution from traditional, manual feature-engineering paradigms toward fully integrated, deep representation learning frameworks. Operating on consumer-grade hardware—specifically the 14-channel Emotiv EPOC+ system sampled at 128 Hz—the objective of this system is to accurately classify five distinct cognitive states: Mental Arithmetic (Class 0), Pattern Recognition (Class 1), Working Memory (Class 2), Reading Comprehension (Class 3), and Sustained Attention (Class 4).

The existing baseline architecture for this project utilizes a tuned LightGBM classifier operating on 252 handcrafted features, achieving a subject-dependent test accuracy of 73.34%. This baseline operates across a dataset of approximately 25 subjects, encompassing 2,908 trials (4 seconds each), which are segmented into 37,804 sliding 1-second sub-windows. The handcrafted feature space heavily relies on Power Spectral Density (PSD) extracted via Welch’s periodogram, Differential Entropy (DE), and nonlinear complexity measures such as Sample Entropy (SampEn) and the Higuchi Fractal Dimension (HFD). While LightGBM provides robust decision boundaries for tabular data, it inherently fails to capture the complex, non-stationary spatiotemporal phase-locking dynamics of cortical activity, and its subject-dependent evaluation severely limits real-world generalizability. Furthermore, the reliance on a priori defined frequency bands limits the model's capacity to discover localized, high-resolution temporal biomarkers.   

To transcend these limitations and formulate a probabilistic, computationally validated Digital Twin of the user's cognitive architecture, CogProfile-Net integrates advanced Deep Embedded Clustering (DEC), multi-channel Scalogram Stacking via the Continuous Wavelet Transform (CWT), and robust Domain Adaptation techniques. The output of this neural pipeline is subsequently subjected to temperature scaling to produce a mathematically calibrated probability distribution. This calibrated distribution, combined with behavioral metadata (reaction time, difficulty level, and accuracy), is finally projected onto the Cattell-Horn-Carroll (CHC) psychometric framework and the O*NET occupational database to yield empirical career aptitude mappings.   

Sensor Modality, Signal Fidelity, and Preprocessing Fundamentals
The Emotiv EPOC+ headset acquires neuroelectrical signals via 14 saline-soaked felt pad electrodes positioned according to the international 10-20 system: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, and AF4, utilizing CMS/DRL references situated on the left mastoid process. Given the consumer-grade nature of the device, its 14-bit resolution, and its 128 Hz sampling frequency, the signal-to-noise ratio (SNR) is highly susceptible to artifacts. The temporal precision of the raw signal is frequently compromised by electromyographic (EMG) muscle activity, electrooculographic (EOG) ocular movements (blinks typically presenting as high-amplitude narrow spikes in the 0-15 Hz range), electrocardiographic (ECG) artifacts, and variable skin-electrode impedance driven by the degradation of the saline conductor over extended recording sessions.   

To establish a pristine foundation for deep representation learning, the raw signal undergoes a rigorous, automated preprocessing pipeline prior to scalogram generation.

Contact Quality Gating and Artifact Rejection
The Emotiv EPOC+ proprietary hardware generates a continuous Contact Quality (CQ) metric, which acts as a proxy for impedance, indicating the quality of the electrical signal passing through the sensors and the reference. This metric scales discretely from 0 (poor/black) to 4 (optimal/green). CogProfile-Net implements an automated CQ gating mechanism, discarding any trial where the average CQ across the highly artifact-prone frontal cortices (AF3, AF4, F3, F4) and the vital parietal-occipital cortices (P7, P8, O1, O2) drops below a predefined threshold. This gating acts as an initial physical regularization step, preventing downstream neural networks from fitting to localized sensor noise.   

Following CQ gating, the signals are mathematically re-referenced using a Common Average Reference (CAR) to eliminate global background noise and isolate localized cortical potentials. A finite impulse response (FIR) bandpass filter is then applied between 0.5 Hz and 45 Hz. This passband strategically isolates the canonical neurophysiological frequency bands essential for cognitive state decoding while preventing aliasing effects. High-frequency noise above 45 Hz, largely dominated by muscular artifacts and 50/60 Hz powerline interference, is strictly suppressed.   

Canonical Frequency Band	Range (Hz)	Associated Cognitive and Neurophysiological Functions
Delta (δ)	0.5 – 4 Hz	
Slow-wave activity, deep sleep, unconscious physical restoration.

Theta (θ)	4 – 8 Hz	
Memory consolidation, Working Memory load, relaxation.

Alpha (α)	8 – 13 Hz	
Inhibitory control, visual processing, relaxed wakefulness.

Beta (β)	13 – 30 Hz	
Active cognitive processing, Mental Arithmetic, Sustained Attention.

Gamma (γ)	30 – 45 Hz	
Advanced cognitive processing, multisensory integration, Pattern Recognition.

  
Temporal Segmentation and the Label Noise Paradigm
The dataset consists of approximately 2,908 trials, each lasting 4 seconds. To increase the temporal resolution of the cognitive profile and exponentially expand the training volume for deep learning, CogProfile-Net segments these trials into sliding 1-second sub-windows. Given the 128 Hz sampling rate, each sub-window contains exactly 128 discrete temporal samples across 14 channels. This segmentation yields a high-dimensional dataset of 37,804 sub-windows.

While this expands the sample size, it introduces a critical mathematical challenge: temporal label noise. A subject engaged in a 4-second "Working Memory" or "Sustained Attention" trial may experience transient attentional lapses, micro-distractions, or saccadic eye movements within a specific 1-second sub-window. Assigning a rigid, deterministic "Class 2" (Working Memory) hard label to all constituent sub-windows introduces severely mislabeled instances into the optimization process. Traditional algorithms like LightGBM average out these anomalies across handcrafted temporal features, but highly parameterized deep learning models will memorize this label noise, leading to catastrophic overfitting. This necessitates the integration of unsupervised latent clustering techniques (DEC and GMVAE) to dynamically re-weight or discard mislabeled sub-windows during the training loop.   

Multi-Channel Scalogram Stacking and Time-Frequency Representation
The non-stationary and non-linear characteristics of EEG signals render traditional Fourier-based analyses, such as the Short-Time Fourier Transform (STFT), mathematically suboptimal. The STFT relies on a fixed window size, enforcing a rigid trade-off between time and frequency resolution governed by the Heisenberg-Gabor limit; a narrow window provides excellent temporal resolution but poor frequency resolution, and vice versa. In cognitive profiling, capturing both rapid, high-frequency Gamma bursts (requiring high temporal resolution) and sustained, low-frequency Delta/Theta oscillations (requiring high frequency resolution) is paramount. CogProfile-Net resolves this by mapping the 1D temporal sequences into 2D time-frequency representations using the Continuous Wavelet Transform (CWT).   

Mathematical Formulation of the CWT
The CWT convolves the input EEG signal x(t) with a family of scaled and translated wavelets. The transform is defined as:

CWT 
x
​
 (a,b)= 
∣a∣

​
 
1
​
 ∫ 
−∞
∞
​
 x(t)ψ 
∗
 ( 
a
t−b
​
 )dt
where ψ(t) is the mother wavelet, a is the scale parameter (inversely proportional to frequency), b is the translation parameter (corresponding to time), and ∗ denotes the complex conjugate. The term  
∣a∣

​
 
1
​
  ensures energy normalization across all scales.

CogProfile-Net utilizes the complex Morlet wavelet as the mother wavelet, given its optimal joint time-frequency localization and structural similarity to neurophysiological oscillatory bursts, such as sleep spindles and alpha waves. The Morlet wavelet multiplies a complex sine wave by a Gaussian envelope, ensuring a Gaussian distribution in both the time and frequency domains. It is defined mathematically as:   

ψ(t)=π 
−1/4
 e 
iω 
0
​
 t
 e 
−t 
2
 /2
 
where ω 
0
​
  represents the central frequency of the wavelet. By defining a wavelet kernel of length WL at a given scale, the transformation can be tailored to balance prediction performance and computational cost. Calculating the squared magnitude of the complex wavelet coefficients, ∣CWT 
x
​
 (a,b)∣ 
2
 , yields the wavelet power spectrum, commonly referred to as a 2D scalogram.   

Multi-Dimensional Tensor Construction
For each 1-second sub-window (128 samples), the CWT is applied independently to all 14 electrode channels spanning the 0.5 to 45 Hz range. By discretizing the frequency domain into F logarithmically spaced scales, a single channel yields an F×128 image matrix. Stacking the scalograms from all 14 channels generates a 3D tensor of shape (14,F,128). This multi-channel scalogram acts as a multi-spectral image, allowing advanced computer vision architectures to process spatial (channel-wise anatomical locations), spectral (frequency bands), and temporal (time progression) features simultaneously, capturing inter-channel phase-locking values and long-range synchronization inherently within the convolution or attention operations.   

Deep Representation Learning: Architectural Comparisons
To transcend the 73.34% LightGBM baseline, CogProfile-Net evaluates two state-of-the-art neural architectures capable of parsing the dense 3D scalogram tensors: the highly compact EEGNet and the attention-driven Swin Transformer.

Temporal and Spatial Convolution via EEGNet
EEGNet is a highly compact Convolutional Neural Network (CNN) specifically tailored for BCI applications. Unlike standard deep CNNs (e.g., ResNet or VGG) which employ dense 2D convolutions that rapidly overfit on noisy physiological data, EEGNet leverages depthwise separable convolutions to independently model temporal and spatial relationships, significantly reducing the parameter count.   

The architecture first applies a standard 2D convolutional filter over the temporal dimension of the scalogram to learn frequency-specific bandpass filters. This is followed by a Depthwise Convolution over the spatial dimension (the 14 channels), which acts as a data-driven spatial filter, analogous to the Common Spatial Pattern (CSP) algorithm used extensively in motor imagery tasks. Finally, a Separable Convolution combines a depthwise temporal convolution with a 1×1 pointwise convolution to optimally fuse the extracted feature maps. Empirical evaluations demonstrate that EEGNet combinations with wavelet-based inputs frequently achieve remarkable classification accuracies; for instance, Yan et al. and Liu et al. reported classification accuracies of 94.27% and 90.98%, respectively, when utilizing EEGNet on complex EEG datasets. For the 5-class CogProfile-Net cognitive task, EEGNet serves as a computationally lightweight, real-time capable backbone.   

Hierarchical Self-Attention via the Swin Transformer
As a high-capacity alternative to CNNs, Vision Transformers (ViTs) utilize self-attention mechanisms to model long-range global dependencies across the scalogram. However, standard ViTs exhibit quadratic computational complexity O(N 
2
 ) with respect to the input sequence length (the number of patches), making them computationally prohibitive for dense time-series windows. CogProfile-Net integrates a Swin Transformer (Shifted Window Transformer) backbone to process the scalogram tensors efficiently.   

The Swin Transformer constructs hierarchical feature maps by merging image patches in deeper layers, computing self-attention locally within non-overlapping windows to achieve linear computational complexity O(N). To allow cross-window communication, the window partitioning is shifted by half a window size in successive layers. For an input scalogram patch representation z 
l
​
 , the multi-head self-attention (W-MSA) and shifted window attention (SW-MSA) are computed iteratively as:   

z
^
  
l
 =W-MSA(LN(z 
l−1
 ))+z 
l−1
 
z 
l
 =MLP(LN( 
z
^
  
l
 ))+ 
z
^
  
l
 
z
^
  
l+1
 =SW-MSA(LN(z 
l
 ))+z 
l
 
z 
l+1
 =MLP(LN( 
z
^
  
l+1
 ))+ 
z
^
  
l+1
 
where LN denotes Layer Normalization and MLP is a Multi-Layer Perceptron. By processing the scalograms through the Swin Transformer, the network effectively isolates localized morphological signatures—such as transient frontal theta synchronization during Mental Arithmetic—while preserving global contextual awareness across the entire 1-second sub-window. Studies have shown that neighborhood attention architectures like this improve accuracy over standard ViTs by up to 12.7% on complex physiological datasets like PTB-XL.   

Architecture Model	Primary Inductive Bias	Spatial Modeling Approach	Temporal/Spectral Modeling	Computational Complexity
LightGBM (Baseline)	Tree-based / Gradient Boosting	Handcrafted (PSD/DE features)	Handcrafted (SampEn/HFD)	Low (Inference)
EEGNet	Convolutional	Depthwise Spatial Filters	Temporal Separable Convolutions	Ultra-Low
Swin Transformer	Self-Attention	Patch-based mixing	Shifted Window Attention	High (Training) / Medium (Inference)
Sub-Window Latent Refinement via Unsupervised Clustering
As previously established, the dataset of 37,804 sub-windows suffers from temporal label noise. To cleanse the dataset of mislabeled sub-windows and discover pure, mathematically verifiable cognitive micro-states, CogProfile-Net incorporates Deep Embedded Clustering (DEC) and Gaussian Mixture Variational Autoencoders (GMVAE). These algorithms function on the deep embeddings outputted by the EEGNet or Swin Transformer backbones.   

Deep Embedded Clustering (DEC)
DEC is an unsupervised algorithm that jointly optimizes a deep feature representation and cluster assignments, significantly improving clustering performance on high-dimensional complex data. The network maps the scalogram tensor x 
i
​
  to a lower-dimensional latent representation z 
i
​
 ∈R 
d
 . A Student's t-distribution is used as a kernel to measure the similarity between the latent embedded point z 
i
​
  and a set of learnable cluster centroids μ 
j
​
  (where j∈{0,1,2,3,4} corresponding to the 5 target cognitive classes, plus auxiliary clusters for physiological noise or distraction). The soft assignment q 
ij
​
  of sample i to cluster j is computed as:   

q 
ij
​
 = 
∑ 
j 
′
 
​
 (1+∣∣z 
i
​
 −μ 
j 
′
 
​
 ∣∣ 
2
 /α) 
− 
2
α+1
​
 
 
(1+∣∣z 
i
​
 −μ 
j
​
 ∣∣ 
2
 /α) 
− 
2
α+1
​
 
 
​
 
where α represents the degrees of freedom of the Student's t-distribution. To iteratively refine the clusters and force the network to make high-confidence predictions, DEC defines an auxiliary target distribution p 
ij
​
  that raises the soft assignments to the second power and normalizes by the cluster frequency:   

p 
ij
​
 = 
∑ 
j 
′
 
​
 (q 
ij 
′
 
2
​
 /f 
j 
′
 
​
 )
q 
ij
2
​
 /f 
j
​
 
​
 
where f 
j
​
 =∑ 
i
​
 q 
ij
​
  represents the soft cluster frequencies. The clustering objective minimizes the Kullback-Leibler (KL) divergence between the soft assignments and the target distribution:   

$$ \text{Loss} = \text{KL}(P \vert{}  \vert{} Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}} $$

By minimizing this loss function, DEC pulls representations closer to their respective cluster centers, effectively separating clean cognitive states from noise-contaminated sub-windows based purely on their neuro-electrical morphology, bypassing the flawed behavioral hard labels.   

Gaussian Mixture Variational Autoencoders (GMVAE)
While DEC provides deterministic cluster assignments, GMVAEs model the latent space probabilistically by assuming the underlying data distribution is generated by a Gaussian Mixture Model (GMM). The GMVAE architecture extracts latent factors representing distinct cognitive states, allowing the system to disentangle specific mechanisms—such as working memory load—from general physiological arousal.   

The generative model assumes a categorical variable y∼Cat(π) representing the cluster (the cognitive state), and a continuous latent variable z conditioned on y, such that p(z∣y)=N(μ 
y
​
 ,Σ 
y
​
 ). The Evidence Lower Bound (ELBO) maximized during training incorporates the reconstruction loss and a KL divergence penalty that forces the approximate posterior q(z,y∣x) to match the GMM prior p(z,y):   

$$ \mathcal{L}{ELBO} = \mathbb{E}{q(z,y\vert{}x)}[\log p(x\vert{}z)] - \text{KL}(q(z,y\vert{}x) \vert{}  \vert{} p(z,y)) $$

Integrating GMVAE into CogProfile-Net allows for the identification of unlabelled transitional states. For instance, if a subject's "Mental Arithmetic" sub-window strongly projects into a latent GMVAE cluster typically associated with sustained attention failure or fatigue, the label is dynamically re-weighted during the supervised training phase, drastically reducing the interference of noisy ground truth labels and refining the classifier's accuracy.   

Enhancing Cross-Subject Transferability with Domain Adaptation
The baseline LightGBM model achieves 73.34% accuracy under a subject-dependent evaluation framework, meaning the classifier has access to a specific subject's unique neuro-electrical morphology during training. In practical deployment scenarios, CogProfile-Net must function on zero-shot, unseen subjects. Inter-subject variability—driven by differences in cortical folding, skull conductivity, and baseline resting-state alpha power—creates a severe domain shift between the training data (source domain) and the new user (target domain).   

To bridge this generalization gap and achieve subject-independent classification, CogProfile-Net integrates a Domain Adversarial Neural Network (DANN) framework coupled with Maximum Mean Discrepancy (MMD) constraints.   

Domain Adversarial Neural Networks (DANN)
DANN seeks to learn a feature representation that is simultaneously highly discriminative for the main cognitive task and entirely invariant to the subject identity. The architecture splits into three functional components: a feature extractor G 
f
​
  (e.g., the Swin Transformer), a task classifier G 
y
​
 , and a domain discriminator G 
d
​
 .   

During forward propagation, the feature extractor maps the input scalograms to a deep embedding. The task classifier attempts to predict the cognitive class (0 through 4), while the domain discriminator attempts to predict which subject the data originated from (e.g., Subject A vs. Subject B). During backpropagation, a Gradient Reversal Layer (GRL) is positioned between the feature extractor and the domain discriminator. The GRL mathematically multiplies the gradient by a negative constant −λ during the backward pass.   

The network optimizes a minimax objective function:

E(θ 
f
​
 ,θ 
y
​
 ,θ 
d
​
 )= 
i=1
∑
N 
s
​
 
​
 L 
y
​
 (G 
y
​
 (G 
f
​
 (x 
i
s
​
 ;θ 
f
​
 );θ 
y
​
 ),y 
i
s
​
 )−λ 
i=1
∑
N 
s
​
 +N 
t
​
 
​
 L 
d
​
 (G 
d
​
 (G 
f
​
 (x 
i
​
 ;θ 
f
​
 );θ 
d
​
 ),d 
i
​
 )
where L 
y
​
  is the categorical cross-entropy loss for the cognitive task, L 
d
​
  is the binary cross-entropy loss for domain classification, N 
s
​
  is the number of source samples, and N 
t
​
  is the number of target samples. By maximizing the domain discriminator's loss while minimizing the label classifier's loss, the feature extractor G 
f
​
  is forced to strip away subject-specific idiosyncratic signatures and retain only universally applicable cognitive biomarkers.   

Maximum Mean Discrepancy (MMD)
To explicitly minimize the distance between the source and target distributions in the Reproducing Kernel Hilbert Space (RKHS), CogProfile-Net employs Maximum Mean Discrepancy (MMD) as an auxiliary loss term. Given source features ϕ(X 
s
 ) and target features ϕ(X 
t
 ), the empirical MMD is defined as:   

MMD 
2
 (X 
s
 ,X 
t
 )= 

​
  
N 
s
​
 
1
​
  
i=1
∑
N 
s
​
 
​
 ϕ(x 
i
s
​
 )− 
N 
t
​
 
1
​
  
j=1
∑
N 
t
​
 
​
 ϕ(x 
j
t
​
 ) 

​
  
H
2
​
 
The integration of MMD and DANN ensures that the internal representation space of CogProfile-Net aligns macro-level domain distributions while maintaining tight, class-specific clusters, establishing the robust cross-subject transferability strictly required for a universal Digital Twin.   

Domain Adaptation Technique	Mechanism of Action	Optimization Goal	Impact on Cross-Subject EEG
Deep CORAL	Aligns second-order statistics (covariances) of source and target.	Minimize distance between covariance matrices.	
Reduces basic distribution shifts but may miss non-linear manifolds.

MMD	Aligns distributions in Reproducing Kernel Hilbert Space (RKHS).	Minimize mean embeddings distance in RKHS.	
Highly effective for aligning global feature representations.

DANN (with GRL)	Adversarial training to confuse a domain discriminator.	Minimax optimization (confuse domain, classify label).	
Strips subject-specific anatomical signatures from the latent space.

  
Probabilistic Confidence Calibration for Output Reliability
Deep neural networks, particularly highly parameterized models like Swin Transformers, are notorious for overconfidence; they frequently emit Softmax probabilities approaching 1.0 even when predictions are highly erroneous. Because CogProfile-Net functions as the foundation for a probabilistic Digital Twin mapping directly to career aptitudes, the output of the 5-class classifier must be strictly calibrated. A prediction of 0.8 for "Working Memory" must mathematically equate to exactly an 80% empirical likelihood of the subject genuinely occupying that state.   

Expected Calibration Error (ECE)
Calibration efficacy is quantified using the Expected Calibration Error (ECE), the standard metric for assessing confidence reliability. ECE partitions the probability predictions into M equally spaced bins. For each bin B 
m
​
 , it measures the absolute difference between the average predicted confidence and the empirical accuracy:   

ECE= 
m=1
∑
M
​
  
N
∣B 
m
​
 ∣
​
 ∣acc(B 
m
​
 )−conf(B 
m
​
 )∣
where N is the total number of samples. High ECE values indicate miscalibration (the model is overconfident or underconfident), which would disastrously skew the downstream CHC aptitude mapping algorithms.   

Temperature Scaling and Dirichlet Calibration
To achieve calibration, CogProfile-Net applies Temperature Scaling (TS), a simple yet highly effective post-hoc transformation originally defined by Guo et al. (2017). TS scales the raw output logits z 
i
​
  of the classifier by a single scalar parameter T before the softmax operation:   

p 
i
​
 = 
∑ 
j
​
 exp(z 
j
​
 /T)
exp(z 
i
​
 /T)
​
 
The parameter T is optimized on an independent validation set to minimize the negative log-likelihood (NLL). When T>1, the softmax distribution is softened, reducing overconfidence without altering the maximum logit, thereby preserving the original classification accuracy exactly while dramatically reducing the ECE.   

For multi-class paradigms exhibiting severe class imbalances or distinct inter-class confusion topologies (e.g., distinguishing between highly similar states like Reading Comprehension and Sustained Attention), standard scalar Temperature Scaling may be insufficient. In such cases, CogProfile-Net extends into Dirichlet Calibration, which fits a Dirichlet distribution to the network probabilities, generating a multi-dimensional calibration map that independently scales different cognitive dimensions. This yields highly reliable, mathematically sound probabilistic vectors essential for Digital Twin embedding.   

Digital Twin Integration: Translating Neural States to Career Aptitudes
The ultimate functionality of CogProfile-Net relies on its capacity to convert raw, calibrated EEG probabilities into actionable, real-world human resource data. This is achieved by linking the calibrated cognitive states to the Cattell-Horn-Carroll (CHC) theory of cognitive abilities, and subsequently projecting this psychometric profile onto the O*NET database.   

The Cattell-Horn-Carroll (CHC) Theoretical Framework
The CHC model is the most empirically validated taxonomic framework for human intelligence, organizing cognitive abilities into three hierarchical strata: general intelligence (g) at Stratum III, broad abilities at Stratum II, and narrow specific abilities at Stratum I.   

CogProfile-Net constructs a semantic bridge between the 5 monitored behavioral tasks and specific Stratum II CHC abilities. The model translates sustained classification accuracy and calibrated probabilities into specific psychometric dimensions:

CogProfile-Net Task	Correlated CHC Stratum II Ability	Psychological Definition
Task 0: Mental Arithmetic	Quantitative Knowledge (Gq) & Fluid Reasoning (Gf)	
The depth of acquired mathematical knowledge and the ability to reason using numbers.

Task 1: Pattern Recognition	Visual Processing (Gv) & Fluid Reasoning (Gf)	
The ability to generate, perceive, analyze, and manipulate visual patterns.

Task 2: Working Memory	Short-Term Working Memory (Gwm)	
The ability to apprehend and maintain awareness of elements in immediate awareness.

Task 3: Reading Comprehension	Comprehension-Knowledge (Gc)	
The breadth and depth of acquired knowledge communicated via language.

Task 4: Sustained Attention	Processing Speed (Gs) & Cognitive Efficiency	
The ability to perform automated cognitive tasks fluently while maintaining focus.

  
The calibrated softmax output from the neural classifier over time creates a dynamic, probabilistic vector representing the subject's cognitive load and proficiency in maintaining these specific states under pressure. By integrating behavioral metadata—such as reaction time, difficulty level, and accuracy across the 2,908 trials—CogProfile-Net scales the EEG classification stability into a quantified CHC profile. For instance, a subject who maintains a 95% EEG probability of "Working Memory" during high-difficulty tasks, combined with fast reaction times, will yield a highly elevated Gwm score.   

O*NET Database Structural Mapping
The Occupational Information Network (ONET), developed by the US Department of Labor, contains hundreds of standardized and occupation-specific descriptors. Crucially, ONET defines a specific "Abilities" domain that correlates directly with the CHC taxonomy. For example, O*NET's "Mathematical Reasoning" and "Number Facility" map directly to CHC's Gq; "Deductive/Inductive Reasoning" map to Gf; and "Information Ordering" maps to Gwm.   

The mapping algorithm functions via matrix projection. Let the subject's derived CHC profile be represented by a vector U∈R 
K
 , where K is the number of mapped broad abilities. Let the O*NET occupational database be represented by a matrix O∈R 
J×K
 , where J is the number of distinct careers, and each row contains the importance and level ratings required for that specific career (standardized to a 0-100 scale).   

The Digital Twin algorithm calculates the aptitude alignment using Cosine Similarity and Euclidean Distance metrics:

Similarity(U,O 
j
​
 )= 
∥U∥∥O 
j
​
 ∥
U⋅O 
j
​
 
​
 
Through this integration, a subject exhibiting high EEG classification stability in Working Memory (Task 2) and Pattern Recognition (Task 1), paired with rapid behavioral reaction times, generates a CHC profile dominant in Gwm and Gv. The matrix projection against the O*NET database instantly identifies careers requiring high Gwm and Gv (e.g., Air Traffic Controller, Systems Architect, or Radiologist) providing an automated, biologically grounded career aptitude assessment derived entirely from non-invasive neuroelectrical monitoring.   

Conclusion
The transition from a LightGBM architecture operating on 252 handcrafted features to the fully deep, end-to-end CogProfile-Net paradigm addresses the critical bottlenecks preventing the real-world deployment of EEG-based cognitive profiling. The baseline accuracy of 73.34% represents the upper bound of subject-dependent, traditional machine learning models bounded by rigid 1D temporal structures.

By stacking Continuous Wavelet Transform (CWT) scalograms, the system leverages multi-channel computer vision techniques—specifically EEGNet and Swin Transformers—to model complex frequency-phase relationships and spatial cortical networks. The integration of unsupervised latent space modeling, via Deep Embedded Clustering (DEC) and Gaussian Mixture Variational Autoencoders (GMVAE), purges the inevitable temporal label noise found in sub-windowed physiological data, allowing the network to learn pure cognitive states. Crucially, the implementation of Domain Adversarial Neural Networks (DANN) and Maximum Mean Discrepancy (MMD) ensures the model can generalize across the profound neuroanatomical variance inherent in human subjects, achieving subject-independent functionality. Finally, probabilistic calibration via Temperature Scaling forms the necessary mathematical bridge to map pure neuroelectric phenomenon to the CHC and O*NET structural databases, yielding an objective, neurologically verifiable Digital Twin capable of dynamic and highly accurate career aptitude mapping.


mdpi.com
Automated Multimodal Sleep Staging Using DWT-Based Wavelet Decomposition and Explainable Machine Learning with Signal Sculpting Topographies - MDPI
Opens in a new window

arxiv.org
Classification of Electroencephalograms during Mathematical Calculations Using Deep Learning - arXiv
Opens in a new window

mdpi.com
Systematic Comparison of Electroencephalography Feature Domains for Visual Stimuli Decoding with EEGNet and EEG Conformer - MDPI
Opens in a new window

mdpi.com
Exploring the Relationship between Cognitive Ability Tilt and Job Performance - MDPI
Opens in a new window

oro.open.ac.uk
Testing the relationship between visualisation, written comprehension, and graphic design creativity - Open Research Online
Opens in a new window

psa.drknscience.com
PSA™, Personal SWOT Analysis - Dr. K & Science Innovation
Opens in a new window

pmc.ncbi.nlm.nih.gov
Electroencephalography Signal Processing: A Comprehensive Review and Analysis of Methods and Techniques - PMC
Opens in a new window

pmc.ncbi.nlm.nih.gov
Massage Therapy's Effectiveness on the Decoding EEG Rhythms of Left/Right Motor Imagery and Motion Execution in Patients With Skeletal Muscle Pain - PMC
Opens in a new window

pmc.ncbi.nlm.nih.gov
Convolutional Neural Network for Drowsiness Detection Using EEG Signals - PMC
Opens in a new window

researchgate.net
Empirical Evaluation of the Emotiv EPOC BCI Headset for the Detection of Mental Actions
Opens in a new window

arxiv.org
Recording Brain Activity While Listening to Music Using Wearable EEG Devices Combined with Bidirectional Long Short-Term Memory Networks - arXiv
Opens in a new window

scribd.com
Deep Learning Algorithms | PDF - Scribd
Opens in a new window

air.unimi.it
A neural approach to the Turing Test - AIR Unimi
Opens in a new window

stars.library.ucf.edu
Assessing an EEG and Eye Tracking Interface against Traditional Virtual Reality Input Devices - ucf stars
Opens in a new window

researchgate.net
Discrepancies in Mental Workload Estimation: Self-Reported versus EEG-Based Measures in Data Visualization Evaluation - ResearchGate
Opens in a new window

researchgate.net
Crowdsourced EEG experiments: A proof of concept for remote EEG acquisition using EmotivPRO Builder and EmotivLABS - ResearchGate
Opens in a new window

researchgate.net
(PDF) Advancing Emotional Health Assessments: A Hybrid Deep Learning Approach Using Physiological Signals for Robust Emotion Recognition - ResearchGate
Opens in a new window

mdpi.com
A Parallel Cross Convolutional Recurrent Neural Network for Automatic Imbalanced ECG Arrhythmia Detection with Continuous Wavelet Transform - MDPI
Opens in a new window

pmc.ncbi.nlm.nih.gov
Granular estimation of user cognitive workload using multi-modal physiological sensors
Opens in a new window

pmc.ncbi.nlm.nih.gov
Advanced Bioelectrical Signal Processing Methods: Past, Present and Future Approach—Part II: Brain Signals - PMC
Opens in a new window

pmc.ncbi.nlm.nih.gov
A Task-Related EEG Microstate Clustering Algorithm Based on Spatial Patterns, Riemannian Distance, and a Deep Autoencoder - PMC
Opens in a new window

arxiv.org
Optimal Scalogram for Computational Complexity Reduction in Acoustic Recognition Using Deep Learning - arXiv
Opens in a new window

researchgate.net
The classification algorithm flow chart. | Download Scientific Diagram
Opens in a new window

scribd.com
EEG Signal Classification for Epilepsy | PDF ... - Scribd
Opens in a new window

researchgate.net
Wavelet transforms for feature engineering in EEG data processing: An application on Schizophrenia | Request PDF - ResearchGate
Opens in a new window

mdpi.com
A CNN Deep Local and Global ASD Classification Approach with Continuous Wavelet Transform Using Task-Based FMRI - MDPI
Opens in a new window

frontiersin.org
Sabotage Detection Using DL Models on EEG Data From a Cognitive-Motor Integration Task
Opens in a new window

researchgate.net
Cross-Subject Cognitive Workload Recognition Based on EEG and Deep Domain Adaptation | Request PDF - ResearchGate
Opens in a new window

computer.org
A Novel Vision Transformer Based Multimodal Fusion Approach for Clinical MDD Diagnosis Using EEG and Audio Signals - IEEE Computer Society
Opens in a new window

journals.plos.org
Research on epilepsy detection and recognition based on the combination of time frequency transform and deep learning model | PLOS One
Opens in a new window

sciopen.com
Classification of EEG signals in depression by fusing temporal convolution and feature recalibration - SciOpen
Opens in a new window

frontiersin.org
Spiking neural networks for EEG signal analysis using wavelet transform - Frontiers
Opens in a new window

mdpi.com
Effect of Emotional States on EEG-Based Biometric Identification: A Comparative Study of Classifiers - MDPI
Opens in a new window

ijeeemi.org
MEWT-Enhanced EEGNet for ASD EEG Classification: Performance Evaluation with k-Fold Cross-Validation | Indonesian Journal of Electronics, Electromedical Engineering, and Medical Informatics
Opens in a new window

researchgate.net
(PDF) Unsupervised Time-Series Signal Analysis with Autoencoders and Vision Transformers: A Review of Architectures and Applications - ResearchGate
Opens in a new window

arxiv.org
ECG-NAT: A Self-supervised Neighborhood Attention Transformer for Multi-lead Electrocardiogram Classification - arXiv
Opens in a new window

arxiv.org
Optimal Scalogram for Computational Complexity Reduction in Acoustic Recognition Using Deep Learning - arXiv
Opens in a new window

arxiv.org
Generalized Cauchy-Schwarz Divergence and Its Deep Learning Applications - arXiv
Opens in a new window

escholarship.org
UCLA Electronic Theses and Dissertations - eScholarship
Opens in a new window

researchgate.net
Semi-supervised Deep Embedded Clustering | Request PDF - ResearchGate
Opens in a new window

ruizhang.info
Unsupervised Learning Style Classification for Learning Path Generation in Online Education Platforms - Rui Zhang
Opens in a new window

openaccess.thecvf.com
A Deep Biclustering Framework for Brain Network Analysis - CVF Open Access
Opens in a new window

arxiv.org
Computer Science Sep 2021 - arXiv
Opens in a new window

researchgate.net
Leveraging Variational Autoencoders for Image Clustering | Request PDF - ResearchGate
Opens in a new window

pmc.ncbi.nlm.nih.gov
An Overview of Variational Autoencoders for Source Separation, Finance, and Bio-Signal Applications - PMC
Opens in a new window

mdpi.com
Bi-Hemispheric Adversarial Domain Adaptation Neural Network for EEG-Based Emotion Recognition - MDPI
Opens in a new window

en-journal.org
Domain-generalized Deep Learning for Improved Subject-independent Emotion Recognition Based on Electroencephalography - Experimental Neurobiology
Opens in a new window

pmc.ncbi.nlm.nih.gov
Two-Level Domain Adaptation Neural Network for EEG-Based Emotion Recognition - PMC
Opens in a new window

plos.figshare.com
RMETNet: A cross-subject motor imagery EEG signal classification
Opens in a new window

eureka.patsnap.com
CN122388790A – Cross-subject mi-ee classification method based
Opens in a new window

arxiv.org
Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions
Opens in a new window

arxiv.org
Improving Calibration by Relating Focal Loss, Temperature Scaling, and Properness - arXiv
Opens in a new window

arxiv.org
arXiv:2205.12507v2 [cs.CL] 24 Oct 2022
Opens in a new window

arxiv.org
Full-ECE: A Metric For Token-level Calibration on Large Language Models - arXiv
Opens in a new window

arxiv.org
Scaling of Class-wise Training Losses for Post-hoc Calibration - arXiv
Opens in a new window

arxiv.org
Understanding Model Calibration - A gentle introduction and visual exploration of calibration and the expected calibration error (ECE) - arXiv
Opens in a new window

structural-learning.com
CHC Theory: The Cattell-Horn-Carroll Model Explained - Structural Learning
Opens in a new window

pmc.ncbi.nlm.nih.gov
“Show Me What You Got”: The Nomological Network of the Ability to Pose Facial Emotion Expressions - PMC
Opens in a new window

researchgate.net
Adding a Piece to the Puzzle? The Allocation of Figurative Language Comprehension into the CHC Model of Cognitive Abilities - ResearchGate
Opens in a new window

oecd.org
Assessing artificial intelligence capabilities: AI and the Future of Skills, Volume 1 | OECD
Opens in a new window

researchgate.net
(PDF) The Cattell–Horn–Carroll Model of Cognition for Clinical Assessment - ResearchGate
Opens in a new window

kvmwai.edu.in
Psychological Testing History, Principles, and Applications
Opens in a new window

academic.oup.com
Index | The Oxford Handbook of Personnel Assessment and Selection
