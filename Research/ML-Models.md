# EEG Feature Extraction for Cognitive Tasks 

Common EEG features span time‑domain, frequency‑domain, time‑frequency, and nonlinear measures.  **Time-domain features** include simple statistics (mean, variance) and *Hjorth parameters*.  The Hjorth parameters are defined as: activity = Var(y(t)); mobility = √[Var(dy/dt)/Var(y(t))]; and complexity = mobility(dy/dt)/mobility(y(t)).  **Frequency-domain features** focus on band powers.  Absolute or *relative band power* (e.g. theta, alpha, beta) is usually computed via the power spectral density (PSD).  PSD can be estimated by an FFT on each window (no averaging) or by Welch’s method (which uses overlapping Hanning-windowed segments and averages their periodograms).  A filterbank approach (band-pass filtering each frequency band then computing signal variance or energy) yields equivalent bandpower estimates.  For 4 s windows at 128 Hz (512 samples), typical bands might be δ (1–4 Hz), θ (4–8 Hz), α (8–12 Hz), β (12–30 Hz) etc.  For example, frontal midline θ and α power often **increase** during working memory/arithmetic tasks.  **Bandpower computation:** Welch’s method (e.g. `pwelch` with 50% overlap) gives smoother PSDs but slightly lower frequency resolution.  In practice one may use `scipy.signal.welch` or NumPy’s FFT. Bandpower is then the sum (or integral) of PSD in the target band divided by total power for relative measures.

*Time-frequency features* such as wavelet energies or Hilbert-envelope features are also used (not specified in the question).  **Spectral entropy** measures signal irregularity in frequency: it is the Shannon entropy of the normalized PSD.  If $p_i$ is the PSD fraction at bin $i$, then $H_{\rm spec}=-\sum_i p_i\log p_i$. Higher spectral entropy indicates a flatter (more “noisy”) spectrum. Other entropies include *Rényi entropy* and *Tsallis entropy*, but the classical Shannon spectral entropy is most common.

**Entropy and complexity measures:**  In the *time domain*, popular measures include Approximate Entropy (ApEn), Sample Entropy (SampEn), and Permutation Entropy (PE).  For Sample Entropy (SampEn), given embedding dimension $m$ and tolerance $r$, 
\[
\text{SampEn}(m,r,N)= -\ln\!\Bigl(\frac{A(m+1,r)}{B(m,r)}\Bigr),
\]
where $B(m,r)$ = # of vector pairs of length $m$ within $r$, and $A(m+1,r)$ similarly for length $m+1$.  Permutation Entropy (PE) analyzes the order patterns of the time series: for embedding dimension $D$ and delay $\tau$, it computes the probability $p(\pi)$ of each ordinal permutation $\pi$ of length $D$ and uses $PE=-\sum p(\pi)\ln p(\pi)$.  **Hjorth complexity** (mentioned above) and other nonlinear indices (e.g. **Hurst exponent**, fractal dimensions) quantify long-range correlations and signal irregularity.  The Hurst exponent $H$ (via Detrended Fluctuation Analysis) has been applied in EEG cognitive studies: e.g. Gaurav *et al.* (2021) extracted Hurst and singularity exponents via multifractal DFA for six mental tasks. Higuchi fractal dimension and Lempel-Ziv complexity are also used in some EEG feature sets.  Lyapunov exponents, while theoretically a chaos measure, are rarely used in recent EEG ML due to robustness issues.  

**Frontal Alpha Asymmetry (FAA):** A specialized feature, FAA compares left vs. right frontal alpha power.  Typically, FAA = ln(P<sub>right</sub>) – ln(P<sub>left</sub>), where P is alpha-band power at homologous sites (e.g. F4 vs. F3).  A positive FAA (higher right-alpha) implies relatively greater left-frontal activation.  Variants include subtracting left from right or taking differences/ratios across channels.  FAA is used in affective and cognitive workload studies.

**Inter-channel coherence:** To capture functional connectivity, one may compute the magnitude-squared coherence between electrode pairs.  Coherence $C_{ij}(f)$ is defined as 
\[
C_{ij}(f) = \frac{|S_{ij}(f)|^2}{S_{ii}(f)\,S_{jj}(f)},
\]
where $S_{ij}$ is the cross-spectral density of channels $i,j$.  In practice `scipy.signal.coherence` or MATLAB’s `mscohere` can estimate it (often averaged over a band).  High coherence in, say, theta or alpha between frontal and parietal channels may indicate task-specific coupling.

**Most discriminative features:**  Empirical studies show that **power in theta and alpha bands** often best discriminate cognitive tasks. For example, frontal midline theta tends to rise with memory/arithmetic load.  Parietal alpha power usually decreases (desynchronizes) during visual attention or complex tasks.  Ratios like θ/α or θ/(α+β) are also used as workload indicators.  Nonlinear features (Hurst exponent, Higuchi dimension) have shown utility: e.g. Gaurav *et al.* achieved ~96–97% accuracy on six tasks using Hurst-based features.  However, the “best” features depend on the tasks: attention vs. language tasks may emphasize different channels/bands.  Often, one performs feature-ranking (e.g. via classifier weights or SHAP) to identify which bands/channels most distinguish each task.

# Classical ML Pipeline for Small EEG Datasets

A robust pipeline for small-N EEG feature data (~25 subjects, 2,900 samples, ~250 features) should include careful scaling, class-balance handling, feature selection, and cross-validation.  **Feature scaling:** EEG features can have outliers (e.g. bursts), so using a scaler is important.  *StandardScaler* (z-score) is common, making each feature zero-mean/unit-variance.  If outliers dominate, *RobustScaler* (centering by median and scaling by IQR) is an alternative.  In practice, EEG analyses often use standard normalization unless outliers severely skew a band power or entropy feature.  

**Class imbalance:** If some tasks are underrepresented, apply techniques like SMOTE (Synthetic Minority Oversampling) to generate synthetic examples of minority classes, or use class weights in the classifier.  One can also ensure stratified sampling so each fold in CV has proportional classes.  In EEG literature, class weights or SMOTE are regularly used to equalize classes.  For small datasets, stratified splits + class weighting (e.g. in SVM or logistic loss) may suffice, while SMOTE can augment data artificially.

**Feature selection:** With ~250 features, reducing dimensionality can improve learning.  *Filter* methods (e.g. univariate ANOVA F-values, mutual information, correlation threshold) rank features before any model is trained. *Wrapper* methods (recursive feature elimination, greedy search) use the model’s performance to select subsets. *Embedded* methods leverage regularization (L1 penalty) or tree-based importances built into model fitting.  In EEG ML papers, common practice is to first filter by statistical tests (or mutual info) and then apply a wrapper (like SVM-RFE) within cross-validation.  The key is to perform selection **inside** the training folds to avoid leakage.  E.g., one might use an inner CV loop to pick the top 50 features by F-test, then train models on those features.

**Hyperparameter tuning:** Use grid search or random search over reasonable parameter grids.  Bayesian optimization (e.g. Optuna, Gaussian processes) can more efficiently explore hyperparameters but grid is fine for few models.  Crucially, tune hyperparameters within a *nested* CV framework to avoid overfitting to the test set.  For example, one can do outer LOSO cross-validation and inner CV for tuning: in each outer fold, one subject is held out, and the remaining subjects are split into inner folds for hyperparameter search.  This yields an unbiased estimate of generalization.  In small-data EEG, nested CV ensures the hyperparameter choice is not biased by the test data.  

**Leave-One-Subject-Out (LOSO):** For subject-independent evaluation, use LOSO-CV: for each fold, pick one subject’s entire feature set as test, train on all other subjects.  This reflects real-world use (model never seen the test subject).  Within each training set, one may further do inner folds (e.g. leave-5-out for hyperparameter tuning).  Make sure to recompute any normalization or feature selection using only the training subjects each fold.  Stratification by task label should be maintained within the training data when splitting for inner CV. 

# ML Models and Hyperparameters (40–50 Settings)

A comprehensive EEG baseline should include **linear, kernel, tree, ensemble, boosting,** and **neural** models.  Common choices (with typical hyperparameter ranges) are:

- **Logistic Regression:** L2-regularized with inverse regularization *C* = {0.01, 0.1, 1, 10}. Solvers: ‘liblinear’ (for small data).  
- **Linear SVM:** (C from {0.01, 0.1, 1, 10}) – often a strong baseline for EEG features.  
- **Gaussian Naïve Bayes:** no hyperparameters (but include as a simple baseline).  
- **K-Nearest Neighbors (KNN):** n_neighbors ∈ {3,5,7,9}, distance metric (‘euclidean’/’manhattan’).  
- **LDA/QDA:** Linear Discriminant Analysis (shrinkage can be tuned), and Quadratic DA as a counterpart.  

- **SVM (RBF kernel):** C ∈ {0.1, 1, 10, 100}, γ ∈ {1e-3, 1e-2, 1e-1, 1.0}.  
- **SVM (Polynomial kernel):** C ∈ {0.1, 1, 10}, γ ∈ {0.01, 0.1, 1}, degree ∈ {2,3}.  
- **Decision Tree:** criterion = {“gini”, “entropy”}, max_depth ∈ {5, 10, None}, min_samples_leaf ∈ {1,5}.  
- **Random Forest:** n_estimators ∈ {50, 100, 200}, max_depth ∈ {5, 10, None}, max_features ∈ {“sqrt”, “log2”}.  
- **Extra Trees:** (similar to RF).  
- **AdaBoost:** n_estimators ∈ {50, 100, 200}, learning_rate ∈ {0.01, 0.1, 1.0}, base_estimator = Decision Tree (max_depth=1 or 3).  
- **Gradient Boosting (sklearn’s):** n_estimators ∈ {100,200}, learning_rate ∈ {0.01, 0.1}, max_depth ∈ {3,5}.  
- **XGBoost (or LightGBM):** n_estimators ∈ {100}, learning_rate ∈ {0.01, 0.1}, max_depth ∈ {3,5,7}, (plus subsample/colsample).  
- **VotingClassifier or Bagging:** e.g. Bagging of decision trees or LDA with n_estimators ∈ {10,50}.  (These ensembles add robustness.)  

- **MLP (sklearn Neural Net):** hidden_layer_sizes ∈ {(50,), (100,), (50,50)}, alpha (L2 penalty) ∈ {1e-4, 1e-3, 1e-2}, solver = {“adam”, “lbfgs”}, activation = {“relu”, “tanh”}.  

These give on the order of 40–50 model+hyperparameter combos to try.  In EEG papers (2018–2025), researchers routinely benchmark SVM (linear/RBF), LDA, Random Forest, AdaBoost/GBDT, and MLP as baselines.  Including these ensures publishable rigor. 

# Explainability (XAI) for EEG Features 

For tabular EEG features, standard XAI methods apply.  **SHAP** (Shapley Additive Explanations) is widely used.  For tree models (e.g. Random Forests/GBDT) use TreeSHAP for fast exact Shapley values.  For SVM or other models, use KernelSHAP (approximate, model-agnostic).  **LIME** can provide local linear explanations around a prediction.  Global feature importances via *permutation importance* (randomly shuffle each feature and measure performance drop) are simple and effective.  **Partial Dependence Plots (PDPs)** show how varying one feature (holding others fixed) affects predicted outcome on average.  These illuminate feature-class relationships.  **SHAP interaction values** (computed by SHAP library) can reveal pairwise feature synergies.  

**Interpretation:**  SHAP values quantify the contribution of each feature to the model’s decision for a given example (positive SHAP means pushing toward a class).  For instance, if “frontal theta power” has a high positive SHAP value for a memory-task prediction, it means higher frontal theta significantly drove the classifier to label *memory*.  Neurologically, this aligns with known findings (frontal theta is linked to memory load), giving face validity.  One should explain SHAP results in terms of physiology: e.g. a feature with high average SHAP for the “language” class might indicate that brain activity in that band/location is key for language processing.  

**Visualizations:**  Common plots include bar or beeswarm plots of SHAP values (global importance), force plots or decision plots for single examples, and PDPs.  In EEG-XAI papers, one often sees heatmaps of feature importances across channels/bands, or topographic maps showing which electrodes contribute most (using SHAP/LIME values).  Time-frequency maps or scalp topographies colored by SHAP can also appear.  Permutation importance bar charts and 2D PDP curves are likewise standard.  

# Paper Structure and Statistical Rigor

A well-structured paper should include: (1) **Introduction** (motivation, related work on EEG cognitive tasks and ML), (2) **Methods** (data acquisition, preprocessing, feature extraction formulas, ML pipeline details including scaling and CV), (3) **Models** (list of classifiers and hyperparameter grids), (4) **Evaluation Protocol** (LOSO CV, metrics), (5) **Results** (performance tables, confusion matrices, XAI analyses), (6) **Discussion** (interpretation of results and feature relevancies), and (7) **Conclusion**.  Include figures such as ROC/accuracy bar plots, confusion matrices, feature importance/SHAP visuals, and ablation study charts. Tables should summarize classification accuracies (or F1 scores) per model and per class. Ablation studies may show performance drop when removing a feature set (e.g. no nonlinear features) or the effect of adding feature selection, demonstrating each pipeline component’s value.

**Statistical tests:** To claim a model is significantly better, use non-parametric tests over the cross-validation folds. A common approach is the Friedman test (for comparing multiple classifiers) followed by post-hoc Wilcoxon signed-rank tests between pairs, with Bonferroni (or Holm) correction for multiple comparisons.  For example, many EEG ML papers report Friedman χ² results, then Wilcoxon *p*-values.  Even a pairwise paired *t*-test can be used if normality holds, but non-parametric is safer given small N.  In addition, confidence intervals on accuracy or k-fold results (e.g. via t-distribution or bootstrapping) lend rigor.

**Expected content:**  Reviewers typically expect a table of results (models vs. performance metrics), confusion matrices or class-wise accuracies, and feature importance plots. Ablation experiments – e.g., comparing results with vs. without feature selection, or with different feature groups – strengthen the study. Reporting computation times or model complexity adds completeness.  

**Target venues:** Appropriate journals include IEEE Transactions on Neural Systems and Rehabilitation Engineering (T-NSRE), Journal of Neural Engineering, Frontiers in Neuroscience (Neuroprosthetics/Neural Engineering section), or IEEE EMBC conference proceedings. These venues regularly publish EEG classification baselines and expect the above rigorous evaluation.  

**References:** We have based these guidelines on recent EEG-ML literature and standard ML practices (e.g. nested CV, statistical tests).