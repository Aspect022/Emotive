# A. Executive Conclusion

The combined EEG + Web-test dataset **can** form the basis of a publishable study, but **only if** strict data handling and validation steps are followed. The key challenges are ensuring precise temporal alignment and controlling signal quality. We find that Emotiv consumer headsets can capture valid EEG signatures (especially “late” ERPs), but their lower signal‐to‐noise ratio (SNR) versus medical systems is a limitation. Thus, the raw EEG (primary scientific signal) plus performance data *could* support analyses of, e.g., EEG correlates of task performance or workload, but only with careful preprocessing, subject-level analysis, and an honest assessment of statistical power. If these steps are rigorously followed (see below), a defensible paper is feasible; if not, the data risk being underpowered or misaligned. In summary: **Yes, but only with strict synchronization checks, quality control, and appropriate statistical methods (see final roadmap).** 

# B. Exact Data Architecture

Design a relational schema linking behavioral and EEG data. Key tables are:

- **Participant** (`participant_id`, demographics, group) – one row per subject.  
- **Session** (`session_id`, `participant_id`, `start_time`, `end_time`, `headset_model`, `sampling_rate`, `recording_quality`, etc). Each session corresponds to one EEG recording (Emotiv record) and one web-test session.  
- **Trial** (`trial_id`, `session_id`, `stimulus_onset`, `response_time`, `response`, `correct`, `score`, `condition`, etc). One row per test item/question. Fields from Supabase (e.g. stimulus ID, question, difficulty) appear here.  
- **EEG_Metadata** (`session_id`, `channel`, `timestamp`, `raw_value`, `contact_quality`). Stores the continuous EEG time series in long form (or separate files). Timestamps are relative (ms from session start).  
- **Epoch** (`trial_id`, `channel`, `epoch_start`, `epoch_end`, `artifact_flag`, `epoch_data`). Each row holds one EEG segment (time-locked to a trial event, e.g. stimulus or response).  
- **Features** (`trial_id`, `feature_name`, `feature_value`). Each row is one extracted EEG feature (e.g. band power) for a given trial. Columns include band (theta, alpha, etc.) or other measure. 

The **join keys** are `participant_id` → Session → Trial → EEG. For example, `Session.session_id = Trial.session_id`, and each EEG sample in `EEG_Metadata` is tied to its `session_id`. After synchronization, each `trial_id` will map to an EEG time window (`Epoch`) via `session_id` and timestamps. 

In practice, the master trial-event table would combine behavioral fields (stimulus onset, response) with the aligned EEG data. It might have columns like: `participant_id`, `session_id`, `trial_id`, `stimulus_onset`, `response_time`, `response`, `score`, plus pointers to EEG (or raw data columns like `eeg_sample_index`, `channel`, `value`). The canonical fields from both sources are merged so that each trial links to EEG indices. All non-essential columns (e.g. Emotiv “performance metrics” or raw contact logs) can be stored separately or dropped if not used in analysis. 

<!-- No source citation needed for schema design. -->

# C. Synchronization Methodology

**Strategy:** Align web-event times to EEG sample times via a linear clock mapping. Emotiv’s exported data uses *relative* timestamps (seconds since start). The CSV/EDF header also records an absolute “start timestamp” for the first sample. In contrast, the web app logs events with its own clock. We propose solving: 

\[
\text{EEG_time} = a \times \text{Web_time} + b,
\]

where `EEG_time` and `Web_time` are on a common scale (e.g. seconds since study start). First, identify at least two matching events in both logs (for instance, the first trial onset and a later stimulus or the end of session). Then solve for offset *b* and drift factor *a* by linear regression (minimize squared error). This corrects for any steady clock drift. For example, if `t_web_start` (first trial time in web log) and `t_eeg_start` (first sample time) are known, set `b = t_eeg_start - a * t_web_start`. Then adjust all trial onsets accordingly. Validate by comparing multiple events (e.g. subsequent trial onsets or a synchronization key-press) to ensure residual error is small. 

**Marker-based option:** If the experiment could emit markers (e.g. via the Emotiv Extender or Cortex API) at known events, use those markers. For example, the Emotiv CSV has a `MarkerIndex` and `MarkerType` column. If these were used (e.g. keyboard events injected), align by matching markers to web log events. However, if no markers exist, rely on timestamp fitting. 

**Drift and Jitter:** Emotiv timestamps derive from the headset’s monotonic clock, which can drift relative to the web clock. Model drift explicitly (factor *a*) if the linear fit shows deviation (especially for long sessions). Jitter (random timing noise) may remain (~±1–10 ms depending on sample rate). For Emotiv’s 128 Hz data, timing resolution is ~7.8 ms (1/128 s), so synchronization beyond this is futile. Calculate *sync error* per event as `error = EEG_time - (a*Web_time + b)`. This error (mean, SD) quantifies alignment precision. If error > ~10–20 ms, be cautious in interpreting fast ERPs. 

**Start/end cues:** As a check, use any available “start test” or “end test” events recorded by both systems. For example, a manual keypress at session begin (noted by Emotiv’s Extender as `Marker_HARDWARE`) vs. a logged event in the web app. Aligning on these gives initial *b*. If web and EEG ran on the same PC, absolute system time might suffice, but Emotiv’s 1-second-precision “start time” may be too coarse, so the linear fit method is safer. 

**Validation:** After mapping, overlay the web event timeline onto the EEG and visually inspect. Known patterns (e.g. large event-related deflections) should align to events. If available, compute cross-correlation between the EEG bandpower (or a surrogate signal) and event onset train; the lag should match *b*. Crucially, do not force a fit if markers/events cannot be correlated; report synchronization uncertainty explicitly. 

# D. EEG Preprocessing Methodology

- **Load & Inspect Data:** Read the Emotiv export (CSV/EDF) with a robust library (e.g. pandas, [pyEDFlib](https://github.com/holgern/pyedflib)) or MNE-Python. Verify the *number of channels* (e.g. 14 for EPOC X, 5 for Insight) and *sampling rate* (128 or 256 Hz). Check units (µV) and that all expected streams (EEG channels, accelerometer) appear. Record any missing samples (Emotiv flags interpolated samples) or flat channels. 

- **Resampling:** If the recording is at 256 Hz (Epoc X) but you want standardize to 128 Hz, downsample. Use an anti-alias filter or trust MNE’s built-in filter. For instance, decimate by 2 with low-pass pre-filter. Downsampling makes the data easier to handle (and fulfills Makoto’s advice to “downsample to ~250Hz” if higher). Ensure the new rate is applied to all timing (and re-calculate sample indices). 

- **Filtering:** Apply a band-pass filter to remove slow drifts and high-frequency noise. A common choice is **0.5–1 Hz high-pass** and **40–45 Hz low-pass**. For example, a 1 Hz high-pass suppresses sweating/perspiration drift and biases ICA (if used) towards cognitive bands. A 45 Hz low-pass (with a 10 Hz transition around 50 Hz) avoids aliasing line noise without crushing gamma. Emotiv hardware already has a 50/60 Hz notch filter, but if residual line noise appears, apply a digital notch at 50 or 60 Hz. 

- **Referencing:** Re-reference the EEG to a common average (subtract the mean across channels). This “CAR” reduces global noise (e.g. ambient electrical interference). Emotiv’s native reference is CMS/DRL (at P3/P4), but converting to average reference is customary in EEG analysis. If only 5 channels (Insight), subtracting the mean still helps distribute reference evenly. Record which reference was used. 

- **Bad-Channel Handling:** Use Emotiv’s *Contact Quality* (CQ) and *EEG Quality* (EQ) metrics to flag channels. Each channel’s CQ is 0–4 per sample (and overall 0–100). Exclude or interpolate any channel with consistently poor CQ or many ‘Interpolated’ samples. For instance, if a channel’s median CQ <2 or >20% of its samples were interpolated (Emotiv’s CSV “EEG.Interpolated” flag), mark it bad. With few channels, it may be safer to *reject* epochs containing bad-channel samples rather than try interpolation (since neighbors are sparse). 

- **Artifact Detection:** Identify non-brain artifacts. First, examine the accelerometer data (32 Hz sampling) – large head movements indicate motion artifacts. Mark or drop any epochs with accelerometer magnitude above a threshold. Next, detect eye-blinks: since frontal sensors (e.g. AF3/AF4) pick up blinks, any deflection exceeding ~±75–100 µV could be a blink artifact. With only 5–14 channels, ICA has limited utility, so use simpler methods: e.g. set an amplitude threshold or a machine-learning blink detector on frontal channels, and remove those segments. For muscle artifacts (high-frequency bursts), compute signal variance or spectral entropy per epoch and reject outliers. 

- **Detrending & Baseline Correction:** Remove any linear trend across each epoch (detrend) and subtract the pre-stimulus baseline mean. For example, subtract the average over −200…0 ms from each epoch (if that window is long enough) to correct slow drifts. This is standard ERP practice (Luck 2005) and prevents DC offsets from biasing band powers. 

- **Epoch Rejection:** Define an epoch as “usable” only if all its channels have acceptable quality and no flagged artifact. For instance, reject any epoch where any channel had a large interpolation count or CQ=0 for >10% of samples. Keep track of the fraction of epochs rejected per participant/condition. 

- **(Optional) Component Analysis:** With 14 channels, ICA can in principle separate blink/muscle sources, but results are marginal with such low density. If used, apply ICA only after high-pass filtering at ~1 Hz, then manually remove components correlated with EOG or EMG. However, for 5-channel Insight data, ICA is not advised (insufficient channels). Instead, rely on epoch rejection. 

- **Quality Metrics:** After preprocessing, compute signal-quality measures for each channel (e.g. RMS, flat-line count, remaining interpolation percentage). Record these to quantify usable data per session. 

**Tools:** We recommend using Python with [MNE-Python](https://mne.tools/) (for filtering, referencing, epoching) and `numpy`/`pandas` for data handling. MNE can read EDF/CSV, apply FIR/IIR filters, and epoch the data. Use `scikit-learn` or `statsmodels` for regression. Ensure to log parameters (filter cutoffs, thresholds) and set random seeds. (EEGLAB/MATLAB is an alternative, but Python ensures open reproducibility.) 

**Key References:** Emotiv documentation confirms raw EEG is available in EDF/CSV with µV units. Established EEG practice recommends 0.5–1 Hz high-pass and ~40 Hz low-pass filtering and common-average referencing. Artifacts are often removed by threshold or ICA (but use ICA only if justified by channel count). 

# E. Feature-Engineering Methodology

Focus on scientifically meaningful EEG features (avoid “every possible”):

- **Band Power:** Compute absolute and relative power in canonical bands. Given literature linking them to cognition, include **theta (4–7 Hz)**, **alpha (8–13 Hz)**, **beta (14–30 Hz)** powers. For example, frontal midline theta often increases with working-memory load, and alpha power decreases with attention demands. Use Welch’s PSD or multitaper on each epoch. Also compute *normalized* band power (e.g. band power divided by total 4–30 Hz power) to control for overall amplitude differences. 

- **Band Ratios:** Include theta/alpha and theta/beta ratios if justified (some studies use theta/beta as an “attention” marker). For each trial epoch, compute θ/α and θ/β. Frontal alpha asymmetry (alpha(AF4) – alpha(AF3)) is often studied in emotion, but only if frontal channels exist and well-calibrated. 

- **Spectral Features:** Calculate spectral entropy or slope of the PSD, to capture broadband changes. For example, the 1/f slope or Shannon entropy of the power spectrum can reflect arousal or fatigue. 

- **Event-Related Potentials:** If the task involves discrete stimuli/responses, consider ERP features: peak amplitudes and latencies of N1, P2, etc. (if aligned to stimulus), or CNV ramp before response. With Emotiv’s low SNR, raw ERPs will be noisy, but averaging across trials can give a “trial-average ERP”. Use peak detection or area-under-curve (like mean amplitude in 300–500 ms window after stimulus) as features. Focus on robust components (e.g. P300 if present). 

- **Time-Domain Statistics:** Features like trial-level variance, root-mean-square amplitude, or Hjorth parameters (activity, mobility, complexity). These capture signal variability. 

- **Connectivity/Coherence:** With only 5–14 channels, advanced measures like coherence or Granger causality are not reliable. We skip connectivity unless more channels are available and well-calibrated. 

- **Temporal Dynamics:** For tasks with varying RT, compute response-locked features (e.g. EEG power in the 200 ms before response) to capture decision processes. 

Only include features that have theoretical backing: e.g. theta for memory load, alpha for vigilance. Avoid highly specialized metrics unsupported by literature.  

# F. Statistical Methodology

- **Exploratory Analysis:** Start by visualizing EEG features vs. behavior. Plot e.g. band power vs. RT per trial (for each subject). Check distributions and outliers. Report means±SD of EEG measures by condition or accuracy.

- **Correlation/Regression:** Test trial-level relationships. For continuous outcomes (RT, score), use Pearson or Spearman correlation with each EEG feature. For binary outcomes (correct vs. incorrect), use logistic regression. However, trials are not independent: include **subject** as a random effect in all models. For example, fit a linear mixed-effects model (LMM):  
  `RT ~ theta_power + (1 | subject_id)`.  
  And for accuracy:  
  `logit(P(correct)) ~ alpha_power + (1 | subject_id)`.  

- **Group Comparisons:** If there are conditions (e.g. easy vs. hard trials), perform a repeated-measures ANOVA or mixed ANOVA on features (with subject as blocking factor). Report effect sizes (η² or Cohen’s d) with confidence intervals. 

- **Normalization:** Before modeling, consider within-subject z-scoring of EEG features to remove baseline differences. This can be done by subtracting each subject’s mean and dividing by SD across trials, ensuring features are on comparable scales.

- **Multiple Comparisons:** Adjust for multiple tests (e.g. Bonferroni or FDR) if many EEG features are tested. Report *p*-values *and* effect sizes. 

- **Non-independence:** Explicitly model within-subject covariance. Do NOT treat, say, 1000 trials from 5 participants as 1000 independent observations. Instead, either average features per subject (losing trial-level info) or use mixed models as above. 

- **Software:** Use statistical packages like Python’s `statsmodels` or `pingouin`, or R’s `lme4`. Ensure assumptions (normality, homoscedasticity) are checked. Report 95% confidence intervals for main effects. 

# G. Machine-Learning Methodology

Machine learning is *optional* and only justified if there is sufficient data and a clear prediction goal:

- **Justification:** Given likely modest N, prioritize interpretative statistics over “black-box” methods. If pursued, keep models simple (e.g. logistic regression, Random Forest) to avoid overfitting. Deep learning (e.g. EEGNet) is not recommended unless hundreds of subjects/trials are available. 

- **Prediction Tasks:** Possible targets: classify *correct vs. incorrect* trials, *high vs. low difficulty*, or predict *continuous RT*. Formulate clearly (binary vs. regression).

- **Feature Set:** Use only the hand-crafted features above (band powers, etc.). Avoid raw EEG as input unless using a proven CNN. 

- **Training/Testing:** Split data so that *participants* do not cross between train and test sets (subject-level split). For example, use leave-one-subject-out cross-validation for subject-generalization. Within each train fold, further use nested CV or a validation set for hyperparameter tuning. 

- **Models:** Try logistic regression (with L2 regularization) for classification or linear regression for RT. Also test ensemble methods (Random Forest, XGBoost) for robustness. Compare performance to a simple baseline (e.g. majority class). 

- **Evaluation:** Report balanced accuracy, F1 score, AUC-ROC for classification; RMSE and R² for regression. Use permutation testing to assess significance. Always compute metrics on held-out subjects (never test on data used in training or feature selection). 

- **Avoid Leakage:** Never normalize or select features using information from the test subject. All preprocessing and feature scaling must be fit on the training set only. 

- **Interpretability:** If a model works, examine feature importances or weights to see which EEG bands drive the prediction. 

If ML performance is poor (e.g. near-chance classification), report that result without overclaiming. 

# H. Quality-Control Methodology

Define criteria to label sessions/subjects as **usable** or **unusable**:

- **Sessions/Participants:** A session is usable if EEG recording is continuous and contains few dropouts (e.g. SRQ > 0.95 most of the time), and at least, say, 50% of trials remain after artifact rejection. Mark sessions *questionable* if contact quality was often low or >30% of data was interpolated. Exclude any with massive data loss or aborted runs.

- **Channels:** Flag any channel as bad if >20% of its samples are interpolated or contact quality = 0. If >2 channels per session are bad (for EPOC) or >1 (for Insight), consider the session unusable. 

- **Epochs:** Reject trials if an artifact (blink/movement) was detected or if behavioral data is missing (no response logged). Note the number of usable trials per condition. If a condition has <10 trials after cleaning, it may be too few to analyze. 

- **Behavioral Checks:** Report any anomalous response times (e.g. <100 ms or >5 s, suggesting missed stimulus) and remove them. Check for duplicate or missing trial IDs. 

- **Synchronization Quality:** Compute the alignment error (see Part C). If average synchronization error >20 ms or highly variable, the data may not support precise ERPs. 

Produce an automated report summarizing: number of participants, sessions, total duration, missing files, channels dropped, % interpolated samples, % artifacts, and per-condition trial counts. Use traffic-light thresholds (green/yellow/red) for key metrics to decide `usable` vs. `unusable`. 

# I. Research Questions

Based on the actual tasks and data, candidate questions include:

1. **“EEG Correlates of Task Performance.”** *Hypothesis:* EEG features around stimulus onset (e.g. frontal theta, posterior alpha) predict trial accuracy or RT. *IV:* EEG band-power (theta, alpha, beta), *DV:* Reaction time (linear regression) or Correct/Incorrect (logistic). *Test:* Mixed-effects regression per trial. *Expected:* Higher theta and lower alpha on a trial predict longer RT or errors (reflecting greater cognitive load). *Novelty:* Relates raw EEG to behavioral outcomes in a consumer-EEG study. *Limitations:* Correlation ≠ causation. Requires sufficient error trials.

2. **“Condition/Difficulty Effects on EEG.”** *Hypothesis:* Harder questions elicit different EEG signatures (e.g. sustained theta increase) than easy ones. *IV:* Condition (easy vs. hard), *DV:* trial-averaged band power or ERP amplitudes. *Test:* Repeated-measures ANOVA on EEG features by condition. *Expected:* Eg, slow frontal wave (theta) is larger for hard trials. *Novelty:* Demonstrates cognitive workload effect using Emotiv data. *Limitations:* Need well-balanced conditions; could be confounded by individual ability.

3. **“Predicting Performance with ML.”** *Hypothesis:* A classifier can predict if a response will be correct from EEG before the response. *Input:* EEG features (band powers) in a pre-response window, *Output:* correct vs. incorrect. *Test:* Train Logistic Regression or Random Forest with subject-wise CV. *Expected:* Above-chance accuracy indicates EEG signatures of success. *Novelty:* Objective demonstration of EEG-based prediction in an educational test context. *Limitations:* If performance is near ceiling or data are noisy, accuracy will be low. Must avoid subject-leakage.

4. **“Learning/Fatigue Effects in EEG.”** *Hypothesis:* Over the course of the session, performance and EEG bands change (e.g. alpha power drifts up if subjects tire). *IV:* Trial number or time, *DV:* behavioral score, band powers. *Test:* Correlate trial index with EEG/score; compare first-half vs. second-half with paired tests. *Expected:* Possibly slower RT or increased theta over time (fatigue). *Novelty:* Longitudinal analysis of EEG/task learning within session. *Limitations:* Hard to separate practice (learning) from fatigue without control.

5. **“Individual Differences in EEG.”** *Hypothesis:* Subjects with higher overall scores have different baseline EEG (e.g. higher frontal theta at rest) than low performers. *IV:* Subject group (high vs. low overall score), *DV:* average EEG band power during baseline or low-demand periods. *Test:* Between-subject t-test on band powers. *Expected:* Differences suggest neural basis for ability. *Novelty:* Links Emotiv EEG traits to educational performance. *Limitations:* Very low N (only a few subjects) makes this speculative.

**Ranking (strongest → weakest):** (1) Performance correlation (supported by trials), (2) Condition differences (if tasks have clear difficulty levels), (3) Response prediction (ML; depends on data size), (4) Learning effects (requires temporal analysis), (5) Individual differences (underpowered by few participants). 

# J. Novelty Assessment

A valid contribution would **focus on analyses supported by the data** and avoid overclaims. Novelty can come from: (a) *Methodological integration* of consumer-EEG and web data (this pipeline itself is new), and (b) findings in the specific task domain. For example, demonstrating that Emotiv-measured EEG features (e.g. theta, alpha) can predict test performance under real conditions would be new evidence for using low-cost EEG in educational research. Also, systematically documenting synchronization and preprocessing for such an experiment could be a methodological contribution. 

However, many basic EEG findings (e.g. theta increases with difficulty) are **not novel scientifically** – the value here is showing they hold with Emotiv data. Therefore, claims should be cautious: e.g. “we find that frontal theta power correlates with response time, consistent with prior EEG studies of cognitive load,” rather than “revealing a new neural mechanism.” Any novelty is likely modest and methodological: e.g. *“we demonstrate that wearable EEG can yield analyzable ERPs in an online cognitive task.”* For true novelty, one could incorporate something like “This is among the first field studies linking consumer-EEG measures to real-time quiz performance.” Additional novelty might come from improving synchronization algorithms or quality metrics (a small “methods” innovation). 

If the current dataset is weak (few subjects, low signal quality), admit it: a paper could still focus on “method development and dataset feasibility” rather than bold cognitive conclusions. 

# K. Paper Methodology Blueprint

A structured outline for a manuscript:

- **Title:** e.g. *“Feasibility of Combining Consumer-Grade EEG and Web-Based Cognitive Testing”* or *“EEG Correlates of Online Test Performance Using an EMOTIV Headset.”*

- **Abstract:** *Background:* Importance of low-cost EEG for real-world tasks. *Methods:* Describe participants, EMOTIV headset, web test, synchronization (novel pipeline). *Results:* Summarize key findings (e.g. clear ERP components recorded, correlation with performance). *Conclusion:* State that the integrated data is viable/limits.

- **Introduction:**  
  - *Problem:* EEG is valuable but traditional lab setups are costly and stationary. Consumer devices (like EMOTIV) allow mobile testing, but data quality/synchronization are concerns.  
  - *Motivation:* Bridging EEG and web-based assessments could enable new research in education/neuroscience. However, rigorous methods are needed.  
  - *Existing Work:* Cite EMOTIV validation studies (PeerJ 2013, Duvinage et al. 2013) showing some capabilities/limits. Note common EEG findings (theta/alpha) in cognitive tasks.  
  - *Gap:* No published pipeline for merging EMOTIV EEG with custom online tests and analyzing both together.  
  - *Research Question:* Can we reliably synchronize and analyze EMOTIV EEG with web-test events? Which EEG features relate to performance?  
  - *Contributions:* (1) Detailed data-processing pipeline (synchronization, preprocessing), (2) Demonstration of EEG-behavior coupling in this paradigm, (3) Discussion of dataset limitations and how to improve. 

- **Related Work:**  
  - *Consumer EEG:* Summarize validations of EMOTIV (Badcock et al. 2013, Duvinage 2013, etc) and reviews of consumer EEG reliability.  
  - *EEG + Cognitive Tasks:* Review key findings (e.g. theta/alpha in working memory/attention) but note most are lab-based.  
  - *EEG-Behavior Synchronization:* Mention LSL and hardware-trigger methods as gold-standard, then note our workaround.  
  - *Preprocessing Standards:* Cite pipelines (Makoto’s pipeline, guidelines on filtering/ICA).  
  - *ML with EEG:* Briefly note EEGNet etc., but also caution on small N.

- **Methods:**  
  - *Participants & Procedure:* Describe recruitment, ethics, demographics. Explain the web test (type of tasks, # trials).  
  - *EEG Recording:* Hardware (model, channel layout, sampling, filters). Software (EmotivPro version, what streams were recorded).  
  - *Behavioral Data Collection:* How web app logged events (Supabase details), exported schema.  
  - *Data Synchronization:* Detailed algorithm (as in Part C, with formulas).  
  - *EEG Preprocessing:* Exactly the steps from Part D (filter bands, referencing, artifact handling). Include parameters (e.g. “1–45 Hz Butterworth filter, 12 dB/oct”).  
  - *Feature Extraction:* List bands and how computed.  
  - *Analysis:* Statistical models (mixed-effects), and any ML (train/test split by subject). 

- **Results:**  
  - *Data Quality:* Report actual counts (N subjects, trials, % rejected). Show example raw vs. cleaned EEG trace.  
  - *Synchronization:* Present estimated offset/drift and validation (maybe a figure overlaying event markers on EEG).  
  - *Behavioral:* Summarize accuracy, RT distributions, any condition effects.  
  - *EEG Features:* Show power spectra (PSD) of resting vs. task. Compare mean band power across conditions (with stats).  
  - *EEG-Behavior Coupling:* Report regression/ANOVA results (e.g. theta power significantly predicts RT with p/X, CI; show correlation scatterplot).  
  - *ERP Examples:* If computed, show averaged ERP waveform for a key channel (e.g. frontal sites) for two conditions.  
  - *ML Results:* If done, report classification accuracy/ROC (with CI) in a table. 

- **Discussion:**  
  - Interpret findings: e.g. “We observed that increased frontal theta accompanied slower responses, aligning with workload theory.” Compare with literature.  
  - Assess pipeline: e.g. “Our synchronization method achieved ~X ms accuracy, enabling linking events and EEG.”  
  - Limitations: small sample, consumer EEG SNR (cite Duvinage).  
  - Implications: Suggest that this low-cost setup *can* capture meaningful EEG correlates of cognition, but stress caution (e.g. need >N subjects or >trials).  
  - Future Work: E.g. collecting more subjects, adding hardware triggers for sync, testing different tasks. 

- **Conclusion:**  
  - Summarize key results and feasibility. Emphasize methodological contributions (the pipeline) and any scientific findings (behavior-EEG link).  
  - End with a balanced statement: “This study provides a blueprint for integrating consumer EEG with web-based testing, paving the way for larger-scale neurobehavioral experiments outside the lab.”

# L. Implementation Roadmap

1. **Inspect All Files:** Load every data file from EMOTIV (e.g. `.edf`/`.csv` per session) and the Supabase export (CSV/Excel/JSON). Check filenames for participant/session IDs or timestamps. For EEG: use Python to read headers (e.g. `mne.io.read_raw_edf` or `pandas.read_csv`) and record file size, start/end times (from header keys), sampling rate, channel list. For web data: parse column names (expect fields like participant_id, trial, timestamp). Confirm each participant has matching EEG and web files. *Output:* Summary table of files with metadata (type, start time, rows, channels). Validate no missing files. 

2. **Build Data Dictionary:** From the above, list every field in the behavioral data and each EEG file. For each EEG stream, note channel labels (e.g. "EEG.AF3" etc), units (µV), and special columns (e.g. `Counter`, `Interpolated`). For behavioral data, list columns (Stimulus, RT, etc). Create a canonical schema mapping (as in Part B). *Output:* A documented data dictionary (like a CSV or markdown table) specifying each field, type, source (EEG or web). 

3. **Identify Participant/Session Mapping:** Use participant/session IDs from filenames or file contents to link EEG files to web sessions. If no explicit ID, match by time: e.g., session start time in EEG header vs. first web event. Assign a unique `session_id`. Record in a table: session_id → participant_id, start/end (from EEG), headset model (from header or file metadata). *Output:* Master session table linking data files. 

4. **Recover Timestamps:** In each EEG file, compute absolute times: combine header start timestamp (with microsecond precision) plus each sample’s relative time. In web data, convert all timestamps to a common reference (e.g. seconds since Unix epoch). If needed, apply timezone or clock corrections so that EEG and web times are in the same base unit. 

5. **Synchronize Systems:** For each session: identify at least two event times in both EEG (markers or distinct EEG features) and web logs. If EEG has markers (`MarkerIndex/Type`), match them to logged events. Otherwise, use first trial onset and last trial onset times. Fit `EEG_time = a*Web_time + b` via linear regression. Save offset `b` and scale `a`. Apply transformation to all web-event times to map into EEG timeline. Compute residual error for each aligned event; if large (>±20 ms), review manually. *Output:* Time-transformed event timestamps, alignment error statistics. 

6. **Build Master Trial Table:** Merge the synchronized web events into one table. Each row = one trial, with columns: `trial_id`, `session_id`, stimulus onset (EEG clock), response time, response, correct, score, etc. Also include epoch window boundaries (to use later). Ensure each trial’s EEG time (e.g. stimulus onset) is recorded in `eeg_timestamp`. *Output:* A combined events table ready for epoching. 

7. **Preprocess EEG:** For each session’s continuous EEG: apply the pipeline from Part D (filtering 1–45 Hz, re-reference to common average, etc). Use contact-quality columns to log any channels that drop. Interpolate or drop bad channels. Detect and mark artifacts (e.g. identify blink epochs by amplitude threshold). After cleaning, output: cleaned continuous EEG for each session. *Libraries:* MNE-Python or SciPy. *Checks:* Verify PSD shape; ensure no DC offset remains (mean ~0); check that high amplitude artifacts are removed. 

8. **Quality Control:** Apply the criteria from Part H. Compute per-session metrics: fraction of bad channels, fraction of interpolated samples, percentage epochs rejected. Identify any sessions failing thresholds. Remove them from further analysis (flag as unusable). *Output:* QC report (can be a CSV or printed summary). 

9. **Epoch Extraction:** For each trial in the master table, extract EEG segments. Using the aligned `stimulus_onset` (or `response_time`), define epoch windows (e.g. −0.2…+0.8 s around stimulus). Cut the continuous EEG into `Epoch` records. Also extract baseline period for each epoch. *Validation:* Ensure epoch start/end indices fall within the recording; if not, drop that trial (edge-case). 

10. **Feature Extraction:** Compute features for each epoch: band powers (e.g. via Welch periodogram), band ratios, any ERPs (e.g. mean amplitude in 300–500ms), etc. Assemble into a `features` table (trials × features). *Tools:* NumPy, SciPy for FFT; custom code for band ratios. Save this table for analysis. 

11. **Merge EEG-Behavior:** Join the `features` table with the master trial outcomes. Now each row has behavioral variables (RT, accuracy, condition) and EEG features. *Validation:* Spot-check a few trials to ensure e.g. the alpha power is higher for eyes-closed or baseline periods than during stimulus (sanity check). 

12. **Exploratory Analysis:** Generate summary plots (e.g. histograms of RT, scatter of theta vs. RT). Calculate within-subject correlations. Use `pingouin` or `scipy.stats` to compute ANOVAs/regressions. Examine variance inflation to avoid multicollinearity. *Library:* pandas, seaborn/matplotlib for plots. 

13. **Inferential Statistics:** Formal tests: Fit mixed-effects models (e.g. with `statsmodels` or R) as planned in Part F. Test key hypotheses (e.g. theta predicting RT). *Check:* Residuals of models to ensure no violation of assumptions; otherwise consider robust methods. 

14. **Machine Learning (if justified):** Split data by subject. Use `scikit-learn` pipelines to standardize (fit on train only) and train models. Use `sklearn.model_selection.StratifiedKFold` over subjects. Evaluate metrics and confidence intervals (bootstrap). *Check:* Confusion matrices to inspect errors. 

15. **Validation:** Perform participant-level leave-one-out or shuffle test to confirm findings. E.g., shuffle behavioral labels across trials – performance should drop to chance. Conduct sensitivity analyses: try filtering at 0.5 Hz instead of 1 Hz, see if results hold. 

16. **Generate Plots/Tables:** Create publication-quality figures: ERP waveforms, PSD plots, EEG-behavior scatter, model ROC curves, etc. Tables: summary of participants, results of statistical tests (p-values, effect sizes), and model performance. 

17. **Robustness Checks:** Vary key parameters (filter cutoffs, artifact threshold) to see if conclusions are stable. Document any major changes. 

18. **Manuscript Writing:** Start writing sections above. Include extensive methods (for reproducibility). Cite all used sources. Ensure discussion ties back to hypotheses and acknowledges limitations (low-channel count, small N). 

Throughout, use version control (e.g. git) on code, freeze Python environment (e.g. `pip freeze > requirements.txt`). Validate each stage (for example, after filtering, check that raw vs. filtered plots look plausible). Keep processing logs (files processed, exceptions). 

# M. Code Architecture

Propose modular code organization (Python modules/functions):

- `src/ingestion/`: functions to load raw files. e.g. `read_emotiv_csv(filepath) -> DataFrame`, `read_web_data(filepath)`.  
- `src/synchronization/`: functions like `fit_sync(eeg_times, web_times) -> (a,b)`, `apply_sync(a,b,web_timestamp)`.  
- `src/preprocessing/`: functions for each step: `filter_eeg(raw_eeg)`, `apply_reference(raw_eeg)`, `detect_artifacts(raw_eeg)`.  
- `src/epoching/`: function `make_epochs(raw_eeg, events, tmin, tmax)`.  
- `src/features/`: functions like `compute_band_power(epoch, band)`, `extract_features(epoch_data)`.  
- `src/statistics/`: routines to run LMMs or ANOVAs (possibly wrapping `statsmodels`), e.g. `mixed_model(df, formula)`.  
- `src/modeling/`: ML training/testing, e.g. `train_classifier(X,y, subjects)`, using `sklearn`.  
- `src/plotting/`: functions for key figures (e.g. `plot_psd`, `plot_epochs`).  

Each module should have clear inputs/outputs (DataFrames, NumPy arrays). Use a single configuration file (`configs/`) for parameters (filter bands, thresholds, epoch windows). Write unit tests (in `tests/`) for synchronization (given dummy times, does it recover a&b?) and preprocessing (e.g. filter out DC component). 

Do *not* write large blocks of code until file formats are fully inspected. First write parsers that confirm column names, then build functionality iteratively. 

# N. Required Additional Information

**Before analysis, collect and confirm:**

- **EMOTIV Details:** Exact headset model (e.g. Insight vs. EPOC+ vs. EPOC X) and software version used. This determines channel count and sampling (Insight=5ch@128Hz; EPOC+@128 or 256Hz; EPOC X@128/256Hz). Confirm if dry/saline electrodes. 
- **Raw Data Files:** All exported EEG data (CSV, EDF, or JSON). Include contact quality, marker logs, any edf sidecar. Clarify if any WAV/auxiliary files are present (likely not EEG). 
- **Experiment Logs:** Any log of test start/end times, instructions, or marker usage. Screenshots or docs on whether a synchronization marker (e.g. buzzer, keypress) was used. 
- **Behavioral Data Export:** The Supabase export file(s) with trial data. Verify what columns exist (stimulus ID, timestamps, etc.) and format (CSV, JSON, SQL dump). 
- **Metadata:** Participant IDs, demographics, conditions (if any randomization or grouping). Check if any trial was omitted. 
- **Procedure Notes:** How was the task administered? (e.g. Was EEG recording started then web test launched? Did the experimenter press a button to sync? Was the laptop’s clock synced to an internet time service?). 
- **Consent/Ethics Info:** Ensure we have IRB approval documentation/consent forms to mention. Confirm anonymization strategy (e.g. numeric IDs). 

Having this information is crucial to proceed. Without it, synchronization and interpretation are conjectural.