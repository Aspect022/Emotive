# Calibrated **EEG Aptitude** Profiling

A calibrated EEG-based cognitive profile is technically feasible, but the evidence supports using it as **probabilistic decision support**, not as a stand-alone career screening tool.

Gap 1 is about uncertainty-aware classification, Gap 2 about validated cognitive-to-aptitude theory, Gap 3 about probabilistic career mapping, and Gap 4 about end-to-end validation.

## Gap 1

EDL replaces point probabilities with a Dirichlet distribution over class probabilities, so the network outputs nonnegative evidence for each class and total evidence controls confidence  (Sensoy et al., 2018; Pandey et al., 2025). In this framing, low total evidence means high uncertainty even if one class has the largest mean probability, which is the core advantage over plain softmax confidence  (Sensoy et al., 2018; Hayta et al., 2026).

For standard cross-entropy with logits \(z\), class probabilities are \(p_k=\mathrm{softmax}(z)_k\), and loss is \(L_{CE}=-\sum_{k=1}^{K} y_k \log p_k\). In EDL, the network predicts evidence \(e_k \ge 0\), Dirichlet parameters \(\alpha_k=e_k+1\), total strength \(S=\sum_k \alpha_k\), expected class probability \(\hat p_k=\alpha_k/S\), and uncertainty mass \(u=K/S\)  (Sensoy et al., 2018; Li et al., 2022). The original EDL training objective replaces plain cross-entropy with a Dirichlet-aware risk plus a regularizer that discourages unjustified evidence, commonly written as an expected square or log loss under the Dirichlet plus a KL term toward a flat prior on confusing samples  (Sensoy et al., 2018; Pandey et al., 2025). More recent work shows this loss design is sensitive to activation choice, regularization, and parameterization, which matters in small noisy datasets  (Franzen & Pourkamali-Anaraki, 2026; Pandey et al., 2025).

## Topic Comparison

| Method | Main Strength | Main Weakness | Best Fit For Your 37K / 5-Class EEG Setting |
|---|---|---|---|
| EDL | Single-pass uncertainty, explicit Dirichlet output | Loss design can be fragile | Good research option if carefully benchmarked |
| MC-Dropout | Easy retrofit, uncertainty at test time | Often miscalibrated without post-hoc scaling | Strong baseline |
| Deep Ensembles | Usually strongest epistemic uncertainty | Higher compute and storage | Best-performing practical benchmark |
| Bayesian NNs | Regularization and weight uncertainty | Harder training, mixed OOD gains | Secondary benchmark |
| Temperature Scaling | Cheap, effective post-hoc calibration | Does not create epistemic uncertainty | Essential final calibration layer |

|   (Franzen & Pourkamali-Anaraki, 2026; Milanés Hermosilla et al., 2021; Chetkin et al., 2023; Guo et al., 2017)|
|---|

**Figure 1:** Uncertainty methods for small EEG classification

In EEG specifically, uncertainty quantification is increasingly treated as necessary because EEG is noisy, variable across subjects and sessions, and often data-limited  (Nzakuna et al., 2025; Wang et al., 2022). Deep learning in BCI has not consistently beaten strong classical baselines, especially in small-sample settings, and shrinkage LDA, random forests, and adaptive methods remain competitive  (Lotte et al., 2018). EEGNet remains relevant because it generalizes reasonably well with limited data and compact architectures  (Lawhern et al., 2016).

Direct EEG/BCI evidence favors ensembles and dropout baselines over assuming EDL will win automatically. MC-Dropout has been used to provide uncertainty and reject high-uncertainty motor imagery trials with encouraging results  (Milanés Hermosilla et al., 2021). Bayesian EEGNet and Bayesian Shallow ConvNet improved accuracy in some MI settings and sometimes helped out-of-domain detection, but OOD detection remained participant-dependent  (Chetkin et al., 2023). In cross-subject MI, deep ensembles performed best for both accuracy and uncertainty quantification, while standard softmax CNNs still beat some more advanced uncertainty methods  (Manivannan et al., 2024).

Aleatoric uncertainty in EEG comes from sensor imprecision, artifacts, and intrinsic biological variability, while epistemic uncertainty reflects limited knowledge of model parameters and should fall as training coverage improves  (Nzakuna et al., 2025; Manivannan et al., 2024). A Monte Carlo perturbation study found aleatoric precision errors dominated small systematic calibration shifts and affected confidence differently across subjects  (Nzakuna et al., 2025). For a career profiling product, the output should therefore separate **class probability**, **epistemic confidence**, and **data-quality / aleatoric flags**, rather than collapse them into a single percentage  (Chmiel & Kurpas, 2026).

**Key finding or conclusion:** EDL is a promising single-pass uncertainty method, but for a 37K-sample, 5-class EEG problem the evidence supports benchmarking it against **deep ensembles + temperature scaling** and **MC-Dropout + scaling**, not treating it as the default winner  (Franzen & Pourkamali-Anaraki, 2026; Manivannan et al., 2024; Guo et al., 2017).  
**2–3 directly relevant citations:** *Evidential Deep Learning to Quantify Classification Uncertainty* (Sensoy, Kandemir, Kaplan, 2018, arXiv)  (Sensoy et al., 2018); *Monte Carlo Dropout for Uncertainty Estimation and Motor Imagery Classification* (Hermosilla et al., 2021, Sensors)  (Milanés Hermosilla et al., 2021); *Uncertainty Quantification for cross-subject Motor Imagery classification* (Manivannan et al., 2024, arXiv)  (Manivannan et al., 2024).  
**Practical implication:** Train EEGNet-scale models, compare deterministic + temperature scaling, MC-Dropout, 5-member ensembles, and EDL, then report ECE, Brier score, NLL, entropy-based reject curves, and subject-wise calibration  (Nzakuna et al., 2025; Guo et al., 2017).  
**Open debate:** EDL is efficient, but its uncertainty can couple aleatoric and epistemic effects and can be sensitive to loss design, activation, and OOD assumptions  (Davies et al., 2023; Pandey et al., 2025).

## Gap 2

The strongest validated framework for your five cognitive abilities is CHC theory, not a neuroscience-specific vocational map  (Flanagan & Dixon, 2014; McGrew, 2008). CHC gives a principled taxonomy that maps your tasks fairly well: arithmetic reasoning aligns mainly with **Gq** and partly **Gf**, pattern recognition with **Gf/Gv**, working memory with **Gsm**, reading comprehension with **Grw/Gc**, and sustained attention only partially with CHC broad abilities because attention is less cleanly represented as a classic CHC factor  (Flanagan & Dixon, 2014). CHC broad and narrow abilities are correlated rather than independent, so a probabilistic profile is more defensible than forcing mutually exclusive talent types  (Flanagan & Dixon, 2014).

Evidence linking cognitive abilities to vocational outcomes is strongest at the psychometric level, not the EEG level. Career assessment research argues that human ability structure and occupational aptitude demands are organized in broadly parallel ways, and that both general level and profile matter for person-job match  (Gottfredson, 2003). Meta-analysis shows narrow cognitive abilities can add substantial incremental validity beyond general mental ability for task performance, training performance, and organizational citizenship behavior  (Nye et al., 2022).

## Evidence Coverage Across Aptitude Mapping

| | Psychometrics | EEG Markers | Job Outcomes | Ethics |
|---|---|---|---|---|
| Arithmetic | GAP | GAP | GAP | GAP |
| Pattern | GAP | GAP | GAP | GAP |
| Memory | GAP | GAP | GAP | GAP |
| Reading | GAP | GAP | GAP | GAP |
| Attention | GAP | GAP | GAP | GAP |

The densest evidence is in psychometric structure and general job-validity, especially CHC and cognitive career assessment  (Flanagan & Dixon, 2014; Gottfredson, 2003; Nye et al., 2022). The biggest gap is a direct validated bridge from **EEG-measured** arithmetic/pattern/memory/reading/attention signatures to **career performance**, because most EEG work stays at cognitive state monitoring, workload, or clinical cognition rather than occupational prediction  (Ismail & Karwowski, 2020; Tröger et al., 2025).

There is some encouraging neural-trait evidence. Resting-state and task EEG features can predict working memory capacity, fluid intelligence, or mathematical performance, including cross-dataset generalization in some studies  (Hakim et al., 2021; Van Bueren et al., 2026; Thiele et al., 2022). Consumer-grade EEG has also shown feasibility for real-world cognitive-difference measurement, including mathematical ability associations with EMOTIV hardware  (Van Bueren et al., 2026). But these are still **trait-cognition** links, not validated **neural hiring** links  (Muhl & Andorno, 2023).

Legal and ethical constraints are substantial. Recruitment and selection AI systems are treated as high-risk in EU regulation discussions, with requirements around transparency, validity, and data governance  (Leutner et al., 2023; Bublitz et al., 2024). Workplace neurotechnology raises concerns about privacy, power imbalance, and neurodiscrimination, and current EU legal analysis focuses on GDPR and the AI Act rather than giving employers broad permission to use brain data for screening  (Muhl & Andorno, 2023; Muhl, 2024; Sosa Navarro, 2024).

**Key finding or conclusion:** There is a validated **psychometric** route from your five abilities to vocational aptitude, especially through CHC and cognitive career assessment, but there is not yet a validated **EEG-to-career** framework strong enough for employment screening  (Flanagan & Dixon, 2014; Gottfredson, 2003; Tröger et al., 2025).  
**2–3 directly relevant citations:** *The Cattell-Horn-Carroll Theory of Cognitive Abilities* (Flanagan, Dixon, 2014)  (Flanagan & Dixon, 2014); *The Challenge and Promise of Cognitive Career Assessment* (Gottfredson, 2003, Journal of Career Assessment)  (Gottfredson, 2003); *Cognitive Ability and Job Performance: Meta-analytic Evidence for the Validity of Narrow Cognitive Abilities* (Nye, Ma, Wee, 2022, Journal of Business and Psychology)  (Nye et al., 2022).  
**Practical implication:** Map EEG classes first to CHC-like latent abilities, then map those latents to occupational clusters; do not claim direct neural prediction of job success  (Flanagan & Dixon, 2014).  
**Open debate:** Whether ability profiles beyond general ability add enough stable vocational signal remains debated, and EEG markers of cognition often show heterogeneous and task-dependent validity  (Gottfredson, 2003; Tröger et al., 2025).

## Gap 3

Existing career systems split into three families. Traditional O*NET and Holland systems are rule-based and psychometrically grounded but static  (Roy & Sharma, 2026). Game-based systems such as Pymetrics try to infer behavioral and cognitive-emotional traits with ML, but transparency and psychometric validation remain recurring concerns  (Roy & Sharma, 2026; Leutner et al., 2023). Newer recommenders using O*NET data can build explicit ability-profile datasets across hundreds of careers, which is directly relevant to your mapping layer  (Rahim & Basheer, 2025).

## Mapping Layer Options

| Mapping Layer | Why Use It | Main Risk | Recommendation |
|---|---|---|---|
| Rule-based thresholds | Transparent, easy to justify | Brittle, static | Use only as baseline |
| Fuzzy logic | Handles graded membership well | Rules can grow quickly | Good interpretable middle layer |
| Learned MLP | Captures nonlinear interactions | Needs labeled outcomes | Use only after larger validation dataset |
| Bayesian inference over O*NET | Propagates uncertainty naturally | More modeling effort | Best final target |

|   (Zhang & Gu, 2025; Roy & Sharma, 2026; Alonso et al., 2025)|
|---|

**Figure 2:** Candidate mapping layers for cognitive-to-career inference

For your current stage, Bayesian or fuzzy mapping is better justified than a learned end-to-end MLP. Fuzzy models are explicitly designed to handle uncertainty and graded occupational membership, and they outperformed profile and ML baselines in one career-prediction study  (Zhang & Gu, 2025). O*NET-based multi-tier systems already exist with structured ability profiles for 612 careers, which suggests a practical lookup-plus-ranking layer is feasible before collecting your own longitudinal labels  (Rahim & Basheer, 2025). Bayesian recommenders and probabilistic CF models show how downstream recommendation layers can preserve predictive uncertainty instead of collapsing to a single point estimate  (Wang & Kadioğlu, 2021; Zhou et al., 2020; Abdalla & Forthomme, 2022).

Uncertainty should propagate forward, not be hidden. If the EEG layer outputs a Dirichlet or ensemble predictive distribution, the career layer should ingest samples or moments from that distribution and return **occupation posterior probabilities plus credible intervals**, not only a top-1 job label  (Wang & Kadioğlu, 2021; Abdalla & Forthomme, 2022; Uzun & Lobachev, 2026). This matters because miscalibrated confidence can destabilize downstream decision support  (Uzun & Lobachev, 2026).

**Key finding or conclusion:** The most defensible architecture is a **two-stage probabilistic pipeline**: calibrated EEG-to-ability inference, then Bayesian or fuzzy ability-to-career mapping over O*NET-style occupational profiles  (Rahim & Basheer, 2025; Zhang & Gu, 2025; Wang & Kadioğlu, 2021).  
**2–3 directly relevant citations:** *A Hierarchical and Multi-Tiered Personalized Career Recommender System Tailored to Individual Aptitudes* (Rahim, Basheer, 2025)  (Rahim & Basheer, 2025); *A novel career prediction method based on fuzzy model-Fuzzy clustering approach* (Zhang, Gu, 2025, Acta Psychologica)  (Zhang & Gu, 2025); *Modeling uncertainty to improve personalized recommendations via Bayesian deep learning* (Wang, Kadioğlu, 2021, IJDSA)  (Wang & Kadioğlu, 2021).  
**Practical implication:** Build the first version as a transparent Bayesian/fuzzy ranker over occupational families, with uncertainty intervals and a “needs more data” abstention state.  
**Open debate:** ML recommenders can outperform static rules, but proprietary systems still face transparency, fairness, and psychometric-validity concerns in high-stakes settings  (Roy & Sharma, 2026; Leutner et al., 2023).

## Gap 4

End-to-end validation should not start from career outcomes alone. It should first establish that the EEG profile is reliable, calibrated, and construct-valid against external cognitive measures  (Clayson, 2025; Lopez et al., 2023; Parmigiani et al., 2022). Psychometric work in EEG repeatedly shows that reliability and validity constrain any individual-difference application  (Lopez et al., 2023; Clayson, 2024; Parmigiani et al., 2022).

A practical gold-standard ladder is available from adjacent fields. Computerized cognitive batteries have shown acceptable internal consistency and construct validity across domains  (Gur et al., 2009). Remote brain-health batteries can predict vocational aptitude proxies such as AFQT, with the composite explaining 24.4% of AFQT variance and uniquely accounting for 19.2%  (Attarha et al., 2026). EEG-based cognitive indices have also shown meaningful external validation against MoCA and NIH Toolbox measures  (Anjum et al., 2024).

## Validation Metrics

| Layer | Minimum Metrics |
|---|---|
| EEG classifier | Accuracy, macro-F1, AUROC, confusion matrix |
| Probability quality | ECE, Brier score, NLL, calibration slope/intercept |
| Reliability | Split-half, test-retest ICC, subject-wise stability |
| Construct validity | Convergent, discriminant, invariance |
| Decision support | Risk-coverage, abstention benefit, calibration decision loss |

|   (Chmiel & Kurpas, 2026; Meghdadi et al., 2024; Clayson, 2024; Hu & Wu, 2024)|
|---|

**Figure 3:** Validation metrics for neural aptitude decision support

Published EEG aptitude-adjacent systems exist, but none validate your full claim. EEG has been used for pilots’ at-risk cognitive competency identification with 84.8% accuracy  (Jiang et al., 2023). EEG composite indices have tracked training progression and procedural accuracy in operational tasks  (Ronca et al., 2026). Mobile EEG can assess work-related mental workload, attention, fatigue, and situational awareness in realistic settings  (Wascher et al., 2021). But the systematic reviews still emphasize limited ecological validity, lab-heavy designs, and incomplete real-world validation  (Ismail & Karwowski, 2020; Cheng et al., 2022).

For decision support, calibration is not optional. Modern neural networks are often poorly calibrated  (Guo et al., 2017). Temperature scaling is simple and often effective, but can break when calibration sets are tiny or noisy  (Guo et al., 2017; Mozafari et al., 2018). Evaluation should include not only ECE but also decision-aware metrics, because standard ECE does not fully capture downstream decision loss  (Posocco & Bonnefoy, 2021; Hu & Wu, 2024).

**Key finding or conclusion:** Validation should follow a staged biomarker pathway: **analytic validity** of EEG features, **clinical/psychometric validity** against external cognitive measures, then **context-of-use validity** for career guidance, with abstention and human review built in  (Clayson, 2025; Parmigiani et al., 2022; Chmiel & Kurpas, 2026).  
**2–3 directly relevant citations:** *Translating EEG Biomarkers into Clinical Tools: A Psychometric Blueprint Illustrated with the Error-Related Negativity* (Clayson, 2025, American Psychologist)  (Clayson, 2025); *Construct Validation of a Remote Brain Health Assessment Battery to Evaluate Vocational Aptitude* (Attarha et al., 2026, JMIR Formative Research)  (Attarha et al., 2026); *On Calibration of Modern Neural Networks* (Guo et al., 2017, arXiv)  (Guo et al., 2017).  
**Practical implication:** Validate against standardized cognitive batteries first, then AFQT/GATB-like aptitude proxies, then educational or training outcomes, and only later study occupational outcomes.  
**Open debate:** The field still lacks strong evidence that EEG biomarkers have enough construct specificity, cross-site robustness, and fairness for exclusionary human decisions  (Clayson, 2025; Clayson et al., 2023; Chmiel & Kurpas, 2026).

## Design Recommendation

A practical research architecture is: **EEG encoder → calibrated class posterior / Dirichlet profile → CHC-aligned latent ability layer → Bayesian/fuzzy O*NET occupational family mapper → abstaining recommendation interface**  (Rahim & Basheer, 2025; Zhang & Gu, 2025; Flanagan & Dixon, 2014). Use the interface to report top career clusters, uncertainty bounds, and which latent abilities drove the ranking, while explicitly barring employment-screening use until predictive validity and fairness are demonstrated  (Chmiel & Kurpas, 2026; Muhl, 2024).

The single most important shift in the evidence is moving from **hard EEG class labels** to **calibrated probabilistic profiles with abstention**. The single biggest open question is whether EEG-derived cognitive signatures add enough valid, fair, and stable information beyond conventional psychometrics to justify any role in career guidance at all.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Abdalla, N., & Forthomme, D. (2022). Probabilistic Approach for Recommendation Systems. *2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)*, 629-634. https://doi.org/10.1109/icmla55696.2022.00104
 
Alonso, R., Dessí, D., Meloni, A., & Recupero, D. (2025). A novel approach for job matching and skill recommendation using transformers and the O*NET database. *Big Data Res., 39*, 100509. https://doi.org/10.1016/j.bdr.2025.100509
 
Anjum, M. F., Espinoza, A., Cole, R., Singh, A., May, P., Uc, E., Dasgupta, S., & Narayanan, N. (2024). Resting-state EEG measures cognitive impairment in Parkinson’s disease. *NPJ Parkinson's Disease, 10*. https://doi.org/10.1038/s41531-023-00602-0
 
Attarha, M., Polusny, M. A., Fisher, M., Fitzpatrick, K., Schlinsog, W., Chen, C. S., & Vinogradov, S. (2026). Construct Validation of a Remote Brain Health Assessment Battery to Evaluate Vocational Aptitude and Factors Associated With Cognitive Resilience in the Military: Observational Trial. *JMIR Formative Research, 10*. https://doi.org/10.2196/99490
 
Bublitz, C., Molnár-Gábor, F., & Soekadar, S. (2024). Implications of the novel EU AI Act for neurotechnologies.. *Neuron*. https://doi.org/10.1016/j.neuron.2024.08.011
 
Cheng, B., Fan, C., Fu, H., Huang, J., Chen, H., & Luo, X. (2022). Measuring and Computing Cognitive Statuses of Construction Workers Based on Electroencephalogram: A Critical Review. *IEEE Transactions on Computational Social Systems, 9*, 1644-1659. https://doi.org/10.1109/tcss.2022.3158585
 
Chetkin, E. I., Shishkin, S., & Kozyrskiy, B. (2023). Bayesian Opportunities for Brain-Computer Interfaces: Enhancement of the Existing Classification Algorithms and Out-of-Domain Detection. *Algorithms, 16*, 429. https://doi.org/10.3390/a16090429
 
Chmiel, J., & Kurpas, D. (2026). Mapping Executive Function Performance Based on Resting-State EEG in Healthy Individuals: A Systematic and Mechanistic Review. *Journal of Clinical Medicine, 15*. https://doi.org/10.3390/jcm15031306
 
Clayson, P. (2025). Translating EEG Biomarkers into Clinical Tools: A Psychometric Blueprint Illustrated with the Error-Related Negativity. *The American psychologist, 80*, 1410 - 1424. https://doi.org/10.1037/amp0001620
 
Clayson, P. (2024). The Psychometric Upgrade Psychophysiology Needs. *Psychophysiology, 61*, e14522 - e14522. https://doi.org/10.1111/psyp.14522
 
Clayson, P., Mcdonald, J. B., Park, B., Holbrook, A., Baldwin, S., Riesel, A., & Larson, M. (2023). Registered replication report of the construct validity of the error‐related negativity (ERN): A multi‐site study of task‐specific ERN correlations with internalizing and externalizing symptoms. *Psychophysiology, 62*. https://doi.org/10.1111/psyp.14496
 
Davies, C., Vilamala, M. R., Preece, A., Cerutti, F., Kaplan, L. M., & Chakraborty, S. (2023). Knowledge from Uncertainty in Evidential Deep Learning. *ArXiv, abs/2310.12663*. https://doi.org/10.48550/arxiv.2310.12663
 
Flanagan, D. P., & Dixon, S. G. (2014). The Cattell‐Horn‐Carroll Theory of Cognitive Abilities. 368-382. https://doi.org/10.1002/9781118660584.ese0431
 
Franzen, C., & Pourkamali-Anaraki, F. (2026). Ensemble-Based Dirichlet Modeling for Predictive Uncertainty and Selective Classification. *ArXiv, abs/2604.06032*. https://doi.org/10.48550/arxiv.2604.06032
 
Gottfredson, L. (2003). The Challenge and Promise of Cognitive Career Assessment. *Journal of Career Assessment, 11*, 115 - 135. https://doi.org/10.1177/1069072703011002001
 
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ArXiv, abs/1706.04599*. https://doi.org/10.48550/arxiv.1706.04599
 
Gur, R. C., Richard, J., Hughett, P., Calkins, M., Macy, L., Bilker, W., Brensinger, C., & Gur, R. E. (2009). A cognitive neuroscience based computerized battery for efficient measurement of individual differences: Standardization and initial construct validation. *Journal of neuroscience methods, 187*, 254 - 262. https://doi.org/10.1016/j.jneumeth.2009.11.017
 
Hakim, N., Awh, E., Vogel, E., & Rosenberg, M. (2021). Inter-electrode correlations measured with EEG predict individual differences in cognitive ability. *Current biology : CB, 31*, 4998 - 5008.e6. https://doi.org/10.1016/j.cub.2021.09.036
 
Hayta, B., Laus, H., Mittermaier, S., & Krahmer, F. (2026). Plug-in Losses for Evidential Deep Learning: A Simplified Framework for Uncertainty Estimation that Includes the Softmax Classifier. *ArXiv, abs/2605.22746*. https://doi.org/10.48550/arxiv.2605.22746
 
Hu, L., & Wu, Y. (2024). Calibration Error for Decision Making. https://doi.org/10.48550/arxiv.2404.13503
 
Hu, L., & Wu, Y. (2024). Predict to Minimize Swap Regret for All Payoff-Bounded Tasks. *2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS)*, 244-263. https://doi.org/10.1109/focs61266.2024.00024
 
Ismail, L., & Karwowski, W. (2020). Applications of EEG indices for the quantification of human cognitive performance: A systematic review and bibliometric analysis. *PLoS ONE, 15*. https://doi.org/10.1371/journal.pone.0242857
 
Jiang, S., Chen, W., Ren, Z., & Zhu, H.-X. (2023). EEG-based analysis for pilots’ at-risk cognitive competency identification using RF-CNN algorithm. *Frontiers in Neuroscience, 17*. https://doi.org/10.3389/fnins.2023.1172103
 
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S., Hung, C., & Lance, B. (2016). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering, 15*. https://doi.org/10.1088/1741-2552/aace8c
 
Leutner, F., Codreanu, S.-C., Brink, S., & Bitsakis, T. (2023). Game based assessments of cognitive ability in recruitment: Validity, fairness and test-taking experience. *Frontiers in Psychology, 13*. https://doi.org/10.3389/fpsyg.2022.942662
 
Li, H., Nan, Y., Ser, J., & Yang, G. (2022). Region-based evidential deep learning to quantify uncertainty and improve robustness of brain tumor segmentation. *Neural Computing & Applications, 35*, 22071 - 22085. https://doi.org/10.1007/s00521-022-08016-4
 
Lopez, K., Monachino, A. D., Vincent, K. M., Peck, F. C., & Gabard-Durnam, L. (2023). Stability, change, and reliable individual differences in electroencephalography measures: A lifespan perspective on progress and opportunities. *Neuroimage, 275*. https://doi.org/10.1016/j.neuroimage.2023.120116
 
Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., & Yger, F. (2018). A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update. *Journal of Neural Engineering, 15*. https://doi.org/10.1088/1741-2552/aab2f2
 
Manivannan, P., De Jong, I. P., Valdenegro-Toro, M., & Sburlea, A. (2024). Uncertainty Quantification for cross-subject Motor Imagery classification. *ArXiv, abs/2403.09228*. https://doi.org/10.48550/arxiv.2403.09228
 
McGrew, K. (2008). CHC theory and the human cognitive abilities project: Standing on the shoulders of the giants of psychometric intelligence research. *Intelligence, 37*, 1-10. https://doi.org/10.1016/j.intell.2008.08.004
 
Meghdadi, A., Salat, D., Hamilton, J. M., Hong, Y., Boeve, B. F., St. Louis, E., Verma, A., & Berka, C. (2024). EEG and ERP biosignatures of mild cognitive impairment for longitudinal monitoring of early cognitive decline in Alzheimer’s disease. *PLOS ONE, 19*, e0308137 - e0308137. https://doi.org/10.1371/journal.pone.0308137
 
Milanés Hermosilla, D., Trujillo Codorniú, R., López Baracaldo, R., Sagaró-Zamora, R., Rodríguez, D. D., Villarejo-Mayor, J. J., & Núñez-Álvarez, J. (2021). Monte Carlo Dropout for Uncertainty Estimation and Motor Imagery Classification. *Sensors (Basel, Switzerland), 21*. https://doi.org/10.3390/s21217241
 
Mozafari, A., Gomes, H., Leão, W., Janny, S., & Gagn'e, C. (2018). Attended Temperature Scaling: A Practical Approach for Calibrating Deep Neural Networks. *arXiv: Learning*. https://doi.org/10.48550/arxiv.1810.11586
 
Muhl, E., & Andorno, R. (2023). Neurosurveillance in the workplace: do employers have the right to monitor employees' minds?. *Frontiers in Human Dynamics*. https://doi.org/10.3389/fhumd.2023.1245619
 
Muhl, E. (2024). The challenge of wearable neurodevices for workplace monitoring: an EU legal perspective. *Frontiers in Human Dynamics*. https://doi.org/10.3389/fhumd.2024.1473893
 
Nye, C. D., J., & Wee, S. (2022). Cognitive Ability and Job Performance: Meta-analytic Evidence for the Validity of Narrow Cognitive Abilities. *Journal of Business and Psychology, 37*, 1119 - 1139. https://doi.org/10.1007/s10869-022-09796-1
 
Nzakuna, P. S., Gallo, V., Paciello, V., Lay-Ekuakille, A., & Lusala, A. K. (2025). Monte Carlo-Based Strategy for Assessing the Impact of EEG Data Uncertainty on Confidence in Convolutional Neural Network Classification. *IEEE Access, 13*, 85342-85362. https://doi.org/10.1109/access.2025.3570134
 
Nzakuna, P. S., Gallo, V., Carratù, M., Paciello, V., Pietrosanto, A., & Lay-Ekuakille, A. (2025). Learned-Weight Ensemble Monte Carlo DropBlock for Uncertainty Estimation and EEG Classification. *IEEE Open Journal of Instrumentation and Measurement, 4*, 1-13. https://doi.org/10.1109/ojim.2025.3638922
 
Pandey, D. S., Choi, H., & Yu, Q. (2025). Generalized Regularized Evidential Deep Learning Models: Theory and Comprehensive Evaluation. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 48*, 6865-6879. https://doi.org/10.1109/tpami.2026.3660699
 
Parmigiani, S., Ross, J. M., Cline, C., Minasi, C., Gogulski, J., & Keller, C. (2022). Reliability and validity of TMS-EEG biomarkers. *Biological psychiatry. Cognitive neuroscience and neuroimaging, 8*, 805 - 814. https://doi.org/10.1016/j.bpsc.2022.12.005
 
Posocco, N., & Bonnefoy, A. (2021). Estimating Expected Calibration Errors. 139-150. https://doi.org/10.1007/978-3-030-86380-7_12
 
Rahim, M., & Basheer, K. P. M. (2025). A Hierarchical and Multi-Tiered Personalized Career Recommender System Tailored to Individual Aptitudes. *Indian Journal Of Science And Technology*. https://doi.org/10.17485/ijst/v18i3.3979
 
Ronca, V., Capotorto, R., Buonocore, S., Massa, F., Gironimo, G., Amicis, R., Papa, S., Mattioli, L., Palermo, E., Donato, L. D., Freda, D., Pirozzi, M., Ferraro, A., Flumeri, G. D., Giorgi, A., Borghini, G., & Aricó, P. (2026). A novel neurophysiological approach to evaluate the impact of virtual training on skills acquisition. *Journal of Neural Engineering, 23*. https://doi.org/10.1088/1741-2552/ae5a06
 
Roy, M., & Sharma, K. (2026). AI-Powered Career Guidance: A Scalable Model for Personalized Recommendations. *IEEE Access, 14*, 20927-20941. https://doi.org/10.1109/access.2026.3661573
 
Sensoy, M., Kandemir, M., & Kaplan, L. M. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. *ArXiv, abs/1806.01768*. https://doi.org/10.48550/arxiv.1806.01768
 
Sosa Navarro, M. (2024). The Rise of the Neuroslave:. *European Data Protection Law Review*. https://doi.org/10.21552/edpl/2024/1/6
 
Thiele, J. A., Richter, A., & Hilger, K. (2022). Multimodal Brain Signal Complexity Predicts Human Intelligence. *eNeuro, 10*. https://doi.org/10.1523/eneuro.0345-22.2022
 
Tröger, A., Carmellini, P., Tsapekos, D., Gross, J., Young, A. H., Strawbridge, R., & Ritter, P. (2025). EEG Markers of Cognitive Performance in Bipolar Disorder -A Systematic Review.. *Neuroscience and biobehavioral reviews*, 106157. https://doi.org/10.1016/j.neubiorev.2025.106157
 
Uzun, I., & Lobachev, M. (2026). CALIBRATED RELIABILITY FOR STREAMING DECISION SUPPORT. *European Open Science Space*. https://doi.org/10.70286/eoss-09.03.2026.003.100-102
 
Van Bueren, N. E. R., Van Hoogmoed, A. H., Van Der Ven, S. V. D., & Jonkman, L. (2026). Comparing aperiodic activity in consumer-grade and research-grade EEG: Reliability and association with mathematical ability. *Behavior Research Methods, 58*. https://doi.org/10.3758/s13428-025-02905-x
 
Wang, X., Yang, R., & Huang, M. (2022). An Unsupervised Deep-Transfer-Learning-Based Motor Imagery EEG Classification Scheme for Brain–Computer Interface. *Sensors (Basel, Switzerland), 22*. https://doi.org/10.3390/s22062241
 
Wang, X., & Kadioğlu, S. (2021). Modeling uncertainty to improve personalized recommendations via Bayesian deep learning. *International Journal of Data Science and Analytics, 16*, 191-201. https://doi.org/10.1007/s41060-020-00241-1
 
Wascher, E., Reiser, J., Rinkenauer, G., Larra, M., Dreger, F., Schneider, D., Karthaus, M., Getzmann, S., Gutberlet, M., & Arnau, S. (2021). Neuroergonomics on the Go: An Evaluation of the Potential of Mobile EEG for Workplace Assessment and Design. *Human Factors, 65*, 86 - 106. https://doi.org/10.1177/00187208211007707
 
Zhang, L., & Gu, X. (2025). A novel career prediction method based on fuzzy model-Fuzzy clustering approach.. *Acta psychologica, 259*, 105475. https://doi.org/10.1016/j.actpsy.2025.105475
 
Zhou, F., Mo, Y., Trajcevski, G., Zhang, K., Wu, J., & Zhong, T. (2020). Recommendation via Collaborative Autoregressive Flows. *Neural networks : the official journal of the International Neural Network Society, 126*, 52-64. https://doi.org/10.1016/j.neunet.2020.03.010
 
