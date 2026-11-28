# Predicting ICD-Defined Low Back Pain from Emergency Department Chief Complaints: A Comparative NLP Study

**AI in Healthcare — High-Risk Project | Research plan | 25 November 2026**

## 1. Research objective and question

### Research Motivation

In a previous course assignment, I compared spaCy, scispaCy, and medspaCy to characterize low back pain using ICD-defined cohorts and discharge summaries from MIMIC-IV. I found that different NLP approaches provided complementary information, ranging from general entity recognition to biomedical concept extraction and clinical context detection. However, because the study was exploratory and lacked a common predictive task, I could not formally compare their predictive performance.

This experience led me to consider a more focused question: whether more sophisticated NLP approaches necessarily provide better predictive performance for a specific clinical text classification task. ED chief complaints provide an interesting setting for examining this question because they are typically very short and may contain abbreviations, spelling variations, and limited contextual information. Valmianski et al. (2019) further motivated this question by showing that traditional lexical methods can remain competitive with contextual language models in medical short-text classification.

Building on my previous work with low back pain and clinical NLP, this project establishes a controlled classification task using MIMIC-IV-ED chief complaints and ICD-derived reference labels. Rather than assuming that more complex models are better, I aim to compare their predictive performance and examine how their performance changes with the amount of available labeled training data.

**Research question:** How do different NLP models compare in their ability to predict ICD-defined low back pain from emergency department chief complaints?

This is a comparative NLP study, not a clinical diagnostic validation study. It evaluates predictive performance and training data efficiency through two controlled experiments. Low back pain (LBP) is the application domain; recorded ICD diagnoses provide an operational reference label. The aim is not to develop a deployable diagnostic system or establish which NLP architecture is universally superior.

## 2. Background and literature foundation

ED chief complaints are short, unstructured descriptions that may include abbreviations, spelling variation and multiple symptoms. A chief complaint may not contain all the information available when a diagnosis is recorded. This makes it worthwhile to compare simple lexical methods with contextual representations and generative language models.

The literature supports specific parts of the design rather than determining its scope:

| Reference | Role in this project |
|---|---|
| MIMIC-IV-ED v2.2 documentation; Xie et al. (2022) | Dataset structure, data preparation, benchmark design and fixed test set |
| NCQA / NLM VSAC | Standardized ICD reference label construction |
| Lee et al. (2019) | Chief complaint to ICD-derived category prediction; text preprocessing reference |
| Valmianski et al. (2019) | Medical short-text model comparison; word TF-IDF representation |
| Chang et al. (2020) | ED chief complaint contextual representations and Clinical BERT fine-tuning |
| Miotto et al. (2020) | LBP-specific NLP, keyword baseline, L1 logistic regression, learning curves and validation-based threshold selection |
| Wornow et al. (2023), EHRSHOT | Controlled training-data efficiency evaluation; instructor-recommended methodological inspiration |
| Alsentzer et al. (2019); Qwen and Ollama documentation | Model provenance and technical implementation references, not additional literature-review directions |
| TRIPOD+AI (2024) | Selective reporting checklist; no claim of full compliance |

These studies use different inputs, labels and evaluation settings. Valmianski and Chang address chief complaint extraction or standardization; Lee uses ICD-derived diagnostic categories; Miotto studies longer primary-care clinical notes; EHRSHOT evaluates structured longitudinal EHR representations. Their results motivate our design but are not directly transferable performance benchmarks. We follow relevant methods, adapt them to this task, document deviations, and verify the implementation.

## 3. Data, prediction task and models

### 3.1 Dataset and reference labels

Use MIMIC-IV-ED v2.2 (2011–2019), with `edstays`, `triage` and `diagnosis` as the relevant tables. The study population is all eligible ED encounters, not a preselected LBP cohort. Each encounter is one observation. The only patient-specific model input is the chief complaint. Patient and encounter identifiers are used for linkage and partitioning, never as model features.

The binary outcome is whether an encounter has at least one ICD-9-CM or ICD-10-CM diagnosis matching the fixed NCQA Uncomplicated Low Back Pain value sets. Encounters with valid diagnosis records but no matching code are negative. Encounters without usable diagnosis information are excluded, not treated as negative. No independent clinical exclusions, age restrictions or keyword-based cohort filters are added.

| Reference | OID | Frozen version information | Entries |
|---|---|---|---:|
| ICD-9-CM | 2.16.840.1.113883.3.464.1003.113.11.1002 | Definition 20160331; expansion 2015-04-30; code system 2013 | 24 |
| ICD-10-CM | 2.16.840.1.113883.3.464.1003.113.11.1003 | Definition 20210925; expansion 2025-07-30; code system 2026 | 83 |
| Grouping value set (optional M6 knowledge) | 2.16.840.1.113883.3.464.1003.113.12.1001 | Definition 20170504; expansion 2026-03-27 | 83 ICD-10-CM and 191 SNOMED CT |

Record actual downloaded version metadata and verify counts in P2. These are fixed reference expansions, not year-specific historical clinical definitions. SNOMED CT is not used to construct the outcome. The term ICD-defined LBP is used throughout rather than claiming clinically confirmed nonspecific or uncomplicated LBP.

Earlier feasibility work suggested 425,087 ED encounters, 423,989 with diagnosis records, and 16,331 NCQA-positive encounters (approximately 3.85% among those with diagnoses). These are planning estimates only; the fresh EDA and eligibility checks will establish the final dataset and prevalence.

### 3.2 Model selection

| ID | Model | Purpose |
|---|---|---|
| M0 | Regex keyword baseline | Transparent rule-based reference |
| M1 | Word TF-IDF + logistic regression | Lexical representation |
| M2 | Character TF-IDF + logistic regression | Character-level representation |
| M3 | Frozen Bio_ClinicalBERT + logistic regression | Pretrained contextual representation |
| M4 | Fine-tuned Bio_ClinicalBERT | Task-specific contextual adaptation |
| M5 | Qwen3.5 9B direct zero-shot classification | Prediction without task-specific labeled training |
| M6 | Knowledge-guided Qwen3.5 9B (optional) | Explore additional fixed medical concept knowledge |

M0–M5 are the core methods. M6 is optional and does not affect completion of the core study. Comparisons of word and character representations, frozen and fine-tuned BERT, and supervised versus zero-shot approaches are part of the same model comparison, not separate research questions.

## 4. Experimental protocol

**Experiment 1 — Comparative model performance.** Train M1–M4 on the full training set; evaluate M0–M5 on a common fixed test set using the same reference labels and evaluation protocol.

**Experiment 2 — Training data efficiency.** Train M1–M4 on nested subsets of the same training partition and examine F1 learning curves. Reuse the full-data results from Experiment 1. M0 and M5 provide fixed reference lines; they are not retrained. M6 may be included only if implemented.

### 4.1 Cohort eligibility

Require valid `stay_id`, `subject_id`, usable ICD diagnosis information and nonempty chief complaint. Verify unique encounter-level linkage and remove or resolve invalid records using documented QC rules. Do not restrict by age, admission, vital signs, pain score or LBP keywords. Preserve the natural class distribution. Exclude chief complaints that are missing, empty, whitespace-only or consist entirely of punctuation and symbols.

### 4.2 Text preprocessing

**Literature foundation.** Previous studies use different preprocessing strategies depending on their text sources, prediction tasks and NLP methods. Lee et al. (2019) apply basic text normalization without abbreviation expansion, spelling correction or medical ontology mapping. Valmianski et al. (2019) describe preprocessing specifically for TF-IDF representation, while Chang et al. (2020) provide model-specific input settings for Clinical BERT. Miotto et al. (2020) use more extensive preprocessing for longer clinical notes, which may not be appropriate for short chief complaints. Xie et al. (2022) inform the general data preparation and benchmarking framework.

**Common preprocessing.** Chief complaints are short clinical texts in which capitalization, punctuation, numbers, abbreviations and spelling variations may contain useful information. Removing or standardizing these elements could alter the original meaning or introduce assumptions about clinical language. Apply minimal common preprocessing: exclude missing, empty, whitespace-only and punctuation- or symbol-only complaints; trim and normalize whitespace; preserve valid short texts and identical complaints from different encounters.

**Model-specific preprocessing.** Additional processing, including case handling, tokenization, n-gram construction, padding, truncation and prompt formatting, is handled within the corresponding model pipelines. The study compares complete NLP pipelines using the same source information, eligible encounters, reference labels and experimental partitions, rather than isolating model architecture effects.

**Validation and leakage prevention.** Examine text length, repetition and basic formatting characteristics before preprocessing. Verify record counts, text availability, transformation effects and encounter integrity. Full-dataset descriptive analysis is not used to develop keyword rules or optimize model configurations.

Any learned vocabulary, IDF or other fitted transformation must be fitted on the current training subset only. Task-specific text exploration and supervised model development are restricted to development data after splitting.

This strategy is a project-specific adaptation informed by the literature, not a claim that minimal preprocessing is universally superior.

### 4.3 Patient-level data splitting

Keep the unit of prediction and evaluation at encounter level but separate patients across partitions. Follow Xie's fixed development/test benchmark principle, with our own patient grouping and stratification adaptation.

- Stage 1: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; use the first fold as fixed test, remaining folds as development.
- Stage 2: apply `StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)` to development; use the first fold as validation and the rest as training.
- Group by `subject_id`, stratify by binary label. Target encounter proportions: 70% training, 10% validation, 20% test. Actual proportions and class prevalence are approximate, not guaranteed.
- Save a single encounter-to-partition mapping; verify zero patient and encounter overlap, complete coverage and approximately preserved prevalence. This is a fixed split, not repeated cross-validation.

Use training and validation for all model development, tuning and rule refinement. Freeze all experiment configurations before final test evaluation; do not adapt methods based on test results. M5/M6 prompts are fixed without label-driven refinement to preserve the zero-shot setting.

### 4.4 Class imbalance handling

Use inverse-frequency class weights for supervised models: `class_weight="balanced"` for M1–M3 and weighted cross-entropy for M4. For class c, weight = N/(2 N_c), calculated from the current training subset. No oversampling, undersampling or SMOTE for balancing. Validation and test retain their natural prevalence. M0/M5/M6 do not use class weights. Subsampling in Experiment 2 is an experimental data-budget condition, not a balancing technique.

### 4.5 Training data efficiency sampling

Follow Miotto's random subsampling and learning-curve idea and EHRSHOT's controlled data-efficiency framework; adapt their protocols rather than claim replication. Use encounter-level stratified random subsampling without replacement within the already patient-separated training partition. Preserve approximate natural prevalence, so sample size is the principal experimental variable. Patient-level subsampling is unnecessary for preventing cross-partition leakage here.

| Training fraction | Approximate encounters before final QC | Approximate positives |
|---|---:|---:|
| 0.1% | 297 | 11 |
| 1% | 2,968 | 114 |
| 10% | 29,679 | 1,143 |
| 100% | 296,792 | 11,432 |

For each of three planned sampling seeds (42, 123, 2026; subject to compute feasibility), create nested subsets by class-specific seeded permutations and proportional prefix selection. Verify both classes are present, counts, prevalence and nesting. All supervised models use the same subset IDs for a given fraction and seed. Refit learned transformations and classifiers, recalculate class weights, and train independently from the original checkpoint where applicable. Keep the full validation and test partitions fixed. Report mean and standard deviation across reduced-data runs; the full-data endpoint is reused from Experiment 1 and is a single point unless independently repeated. The small-data conditions measure training data efficiency, not total label-budget efficiency because full labeled validation data and full-training-derived model settings are available. Do not assume that 11 positives guarantee adequate coverage.

### 4.6 Evaluation metrics and thresholds

| Metric | Role | Models |
|---|---|---|
| Positive-class F1 = 2TP/(2TP+FP+FN) | Primary | All |
| Precision = TP/(TP+FP) | Secondary | All |
| Recall = TP/(TP+FN) | Secondary | All |
| AUPRC, implemented as average precision (AP) | Supplementary ranking metric | M1–M4 |
| AUROC | Supplementary ranking metric | M1–M4 |
| Confusion matrix (TP, FP, TN, FN) | Supporting | All |

Use scikit-learn metrics with positive label 1 and a documented zero-division convention (0). Report AP explicitly, distinguishing it from trapezoidal PR-curve area. Accuracy is not a primary metric given the imbalance. Do not invent confidence scores for hard-label models. No computational-cost ranking, calibration experiment or formal significance testing is planned.

For each supervised model/configuration, calculate validation scores once and select the threshold maximizing positive-class F1 over 0.05, 0.10, ..., 0.95 (19 candidates); break ties by choosing the higher threshold. Lock the threshold and apply it to test scores. Repeat independently for each reduced-data model and seed. This adapts Miotto's validation-based threshold selection. M0/M5/M6 use their direct binary outputs. F1 curves reflect classification performance after this common selection procedure, not discrimination alone.

### 4.7 Model training and inference

Use literature-informed configurations and limited validation-based tuning. Use the same full-training-derived hyperparameters across reduced-data conditions, with independent threshold selection as specified above. Do not tune on the test set. Run locally first (CPU/MPS on Apple M3, 16 GB), moving to an authorized GPU environment only if needed. Hardware is not an experimental variable. Pilot runs and sanity checks verify implementation quality, not a new experiment or a required F1 target.

#### 4.7.5 M0 — Regex

Follow Miotto's keyword-search baseline idea. Use case-insensitive Python regex with word boundaries. Initial candidates include `low back pain`, `lower back pain`, `LBP`, `back pain`, `backache`, `lumbar pain`, `lumbago`, and `sciatica`. Refine using reference descriptions and development data only; freeze the rule version. Any matching pattern predicts 1, otherwise 0. No complex clinical rule engine or NegEx implementation. Check unit tests, representative matches and validation performance.

#### 4.7.6 M1–M2 — TF-IDF + logistic regression

Combine Valmianski's word representation approach with Miotto's L1 logistic regression. The character representation is our extension; this is not a replication of either paper.

| Parameter | M1 | M2 |
|---|---|---|
| Analyzer | word | char |
| N-gram range | (1, 2) | (2, 5) |
| Lowercase | True | True |
| Stopwords | None | Not applicable |
| min_df / max_df | 1 / 1.0 | 1 / 1.0 |
| max_features | 50,000 | 50,000 |
| Norm / IDF | L2 / smooth IDF | L2 / smooth IDF |
| Tokenizer | scikit-learn default word tokenizer | Native character analyzer |
| Classifier | L1 LR, liblinear | L1 LR, liblinear |
| C candidates | 0.1, 1, 10 | 0.1, 1, 10 |
| max_iter | 1000 | 1000 |
| Class weights | Balanced | Balanced |

Choose C independently using validation F1 after the common threshold search; freeze the selected C for Experiment 2. Check vocabulary, matrix dimensions, zero-feature rows, convergence and validation behavior. If pilot checks reveal implementation or convergence issues, correct them on development data and record the changes before test evaluation.

#### 4.7.7 M3–M4 — Clinical BERT

Use `emilyalsentzer/Bio_ClinicalBERT` for both, with the exact checkpoint revision recorded. Chang is the main modeling reference; Alsentzer provides model provenance. Use the native tokenizer and maximum sequence length of 64 tokens.

| Parameter | M3 | M4 |
|---|---|---|
| Encoder | Frozen, eval mode | End-to-end fine-tuned |
| Representation | Last-layer CLS, 768 dimensions | BERT classification representation |
| Classifier | L2 logistic regression, balanced | Binary classification head, weighted cross-entropy |
| C candidates | 0.1, 1, 10 | N/A |
| Learning rate | N/A | 2e-5 or 1e-4 |
| Epochs | N/A | 3 |
| Batch size | Pilot-based | 16 or 32, pilot-based |
| Optimizer / weight decay | N/A | AdamW / 0.01 |
| Selection | Validation F1 | Validation F1 |

Cache frozen embeddings locally; no corpus-dependent fitting or label use during extraction. Access test embeddings only for final evaluation. Choose configurations on development data, then freeze. Record truncation rates, embedding dimensions, frozen parameter status, loss, validation behavior, memory and runtime. M3 pooling and L2 regularization and the M4 optimizer settings are adaptations, not claims about Chang's exact configuration. No checkpoint, pooling, LoRA or architecture ablation study.

#### 4.7.8 M5–M6 — LLM classification

Use the same Qwen3.5 9B model (`qwen3.5:9b` in Ollama), model digest and inference configuration. Disable thinking mode, use temperature 0 and a fixed binary output format; no fine-tuning or labeled in-context examples. Record versions and actual options. Temperature 0 does not guarantee determinism.

M5 candidate prompt:

```
Task: Predict whether an emergency department encounter will have an ICD-coded
 diagnosis within the study's low back pain definition, based only on the chief complaint.
Return 1 for ICD-defined low back pain and 0 otherwise.
Return only 0 or 1.
Chief complaint: {chiefcomplaint}
```

Optional M6 adds a fixed block of deduplicated official NCQA ICD and SNOMED CT concept descriptions to the same task prompt. No encounter-specific diagnosis, label or extra patient data is provided. The knowledge is label-definition information, not a clinical gold standard. SNOMED and ICD are not assumed to be equivalent. This is knowledge-guided prompting, not a verified implementation of the instructor's Concept Grounding paper.

Check formatting and runtime using synthetic or unlabeled development examples without optimizing prompt content against outcome labels. Freeze prompts and parsing rules. Record invalid outputs; do not silently exclude failed encounters. Before final evaluation, establish a common deterministic retry/failure policy and verify that all methods can use the same evaluation encounters. If full-test inference is infeasible, use an authorized GPU or predefine a common evaluation subset before viewing test outcomes; document the change and prevalence. No independent RAG, agent, CoT or LLM comparison experiment.

### 4.8 Reproducibility and workflow

Run version-controlled SQL manually in BigQuery; export query results as CSV to local `results/`. Use Jupyter notebooks for analysis and modeling. Record dataset and value set versions, SQL and code revisions, eligibility/QC results, split mapping, subset IDs, seeds (Python, NumPy, PyTorch where applicable), model revisions/digests, prompts, hyperparameters, thresholds, package versions and hardware. Use Git/GitHub and `requirements.txt`; no additional MLOps stack.

Store one structured experiment-results CSV with experiment ID, model ID, fraction, seed, sample counts, configuration, threshold and metrics; runtime can be an execution log only. Keep restricted encounter data, identifiers, text, prediction files, splits, embeddings and trained artifacts in controlled local or authorized environments. Exclude `results/` and `models/` from Git; inspect notebook outputs and repository history before publication. Do not upload restricted data to unapproved cloud services. Cross-hardware exact reproducibility is not guaranteed.

## 5. Research expectations and interpretation criteria

Based on previous exploratory work and Valmianski's findings, I suspect that more sophisticated NLP models may not necessarily perform better on very short clinical texts. I will examine this intuition by comparing the positive-class F1 of the selected models on the same held-out encounters, with precision and recall helping explain differences. AUROC and AP supplement score-based models. Report absolute F1 differences descriptively, without implying statistical significance or formal equivalence.

I will also examine how relative performance changes when labeled training data are reduced to 10%, 1% and 0.1% of the full training set. Plot F1 against training fraction (preferably logarithmic scale), using mean and standard deviation for repeated reduced-data runs and fixed references for M0 and M5. The goal is not to prove a particular model is superior. Results contrary to my intuition are equally informative.

The high-risk component is the integration of pretrained contextual representations, end-to-end fine-tuning, zero-shot LLM evaluation and controlled data-efficiency comparisons. Whether these satisfy the course grading criteria is ultimately determined by the instructor and rubric. No computational-efficiency metric is used for ranking.

## 6. Research limitations

### 6.1 ICD phenotype and reference labels

NCQA value sets were developed for quality measurement, not for independent clinical validation of this task. Recorded ICD labels may be incomplete or inconsistent and may not fully reflect the underlying clinical condition. Fixed modern expansions applied to historical records may not capture every historical coding variation. No clinician adjudication is performed. A negative label means no matching recorded code, not necessarily absence of LBP.

### 6.2 Chief complaint information

Chief complaints may contain limited clinical information, abbreviations and ambiguous expressions. Identical text can correspond to different recorded diagnoses. History, examination and investigations are not available to the models. More sophisticated models may extract additional information from text but cannot reliably recover information that was never recorded. Their relative performance remains an empirical question.

### 6.3 Ground truth and evaluation

No independently validated clinical ground truth is available. Metrics measure agreement with the ICD-derived reference label. A classification error is not necessarily a clinical misdiagnosis, and higher F1 does not establish superior medical reasoning or language understanding independently of the task.

### 6.4 Dataset and generalizability

MIMIC-IV-ED represents historical records from a single medical center. Documentation and coding practices may differ across institutions and time. Class imbalance limits the number of positive cases in small training subsets. Patient-separated internal evaluation does not establish external validity; no external validation is planned.

### 6.5 Model and experimental design

The study compares selected implementations with limited tuning, not all possible NLP models. Bio_ClinicalBERT was pretrained on MIMIC-III; potential overlap with the study data cannot be ruled out. Hardware differs and computational efficiency is not compared. Repeated sampling estimates variability but is not a formal significance test. The comparison evaluates complete NLP pipelines, including model-specific preprocessing and text representations. Differences in performance cannot be attributed solely to model architecture.

### 6.6 Interpretation boundary

Conclusions apply to this specific dataset, reference phenotype, input and selected models. The study cannot establish clinical diagnostic validity, readiness for deployment, universal architectural superiority or the actual clinical disease status of individual patients.

## 7. Fresh implementation roadmap

Start from the original source tables and regenerate all results. Earlier project experience can inform implementation, but old opioid-related results are not part of this study's evidence. Each notebook follows numbered sections, English Markdown rationale, code, execution and result interpretation.

| Phase | Notebook(s) | Main work and deliverables |
|---|---|---|
| P1 Dataset exploration | `01_dataset_exploration.ipynb` | Inventory source tables, identifiers, population, availability, linkage and quality checks; extract required source data |
| P2 Reference labels | `02_reference_label_construction.ipynb` | Document NCQA scope and versions, normalize codes, perform exact matching, aggregate and validate encounter-level labels; `encounter_reference_labels.csv` |
| P3 Dataset preparation and experimental setup | `03_dataset_preparation_and_experimental_setup.ipynb` | Examine chief complaint characteristics, document and apply common preprocessing, integrate and validate eligible encounters, construct patient-grouped splits and nested training subsets, and prepare shared experimental configurations |
| P4 Comparative models | `04a_regex_baseline.ipynb`, `04b_tfidf_models.ipynb`, `04c_clinical_bert_models.ipynb`, `04d_zero_shot_llm.ipynb` | Implement model-specific preprocessing and M0–M5; perform development-only text exploration, pilot, train, validate and freeze model configurations; optional M6 if feasible |
| P5 Training data efficiency | `05_training_data_efficiency.ipynb` | Reuse P4 implementations and fixed nested training subsets; train reduced-data models, freeze all settings and coordinate final test evaluation for both experiments |
| P6 Results and discussion | `06_results_and_discussion.ipynb` | Overall comparison table, learning curves, interpretation, literature comparison, limitations, report and presentation |

P3 produces `modeling_dataset.csv`, `dataset_splits.csv` and `training_subsets.csv`. Common text preprocessing and dataset integration are performed in the same notebook without requiring a separate intermediate cleaned-text file.

Final test evaluation for P4 and P5 occurs only after all core configurations are frozen. Do not use Experiment 1 test results to revise Experiment 2.

Proposed repository structure (name not yet committed):

ed-lbp-nlp-benchmark/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── research_plan.md
│   └── literature_review.md
├── references/
│   └── ncqa/                         # local only
├── sql/
│   └── 01_dataset_exploration/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_reference_label_construction.ipynb
│   ├── 03_dataset_preparation_and_experimental_setup.ipynb
│   ├── 04a_regex_baseline.ipynb
│   ├── 04b_tfidf_models.ipynb
│   ├── 04c_clinical_bert_models.ipynb
│   ├── 04d_zero_shot_llm.ipynb
│   ├── 05_training_data_efficiency.ipynb
│   └── 06_results_and_discussion.ipynb
├── configs/                          # as needed
├── results/                          # local only
└── models/                           # local only

## 8. Scope and open questions

Core scope: one research question, one binary prediction task, six core methods, two experiments. M6 is optional. No opioid prediction, diagnostic granularity study, multimodal model, independent robustness, fairness, XAI, calibration or computational-efficiency experiment.

**Open question (not part of the formal protocol):** Would retrieving relevant NCQA medical concepts from a small vector index be a more effective or efficient way to incorporate knowledge than a fixed concept block? Literature support, retrieval quality, feasibility and cost would need assessment. If pursued later, it would be considered an alternative M6 implementation, not M7. No vector database or RAG implementation is committed.

## References and implementation resources

1. MIMIC-IV-ED v2.2. https://physionet.org/content/mimic-iv-ed/2.2/
2. NLM Value Set Authority Center (NCQA value sets; OIDs in Section 3). https://vsac.nlm.nih.gov/
3. Xie et al. (2022). Benchmarking emergency department prediction models with machine learning and public electronic health records. Scientific Data, 9, 658. https://doi.org/10.1038/s41597-022-01782-9
4. Lee et al. (2019). Chief complaint classification with recurrent neural networks. https://pmc.ncbi.nlm.nih.gov/articles/PMC10563436/
5. Valmianski et al. (2019). Evaluating robustness of language models for chief complaint extraction from patient-generated text. https://arxiv.org/abs/1911.06915
6. Chang et al. (2020). Generating contextual embeddings for emergency department chief complaints. https://pmc.ncbi.nlm.nih.gov/articles/PMC7382638/
7. Miotto et al. (2020). Identifying acute low back pain episodes in primary care practice from clinical notes: observational study. JMIR Medical Informatics, 8(2), e16878. https://doi.org/10.2196/16878
8. Wornow et al. (2023). EHRSHOT: An EHR benchmark for few-shot evaluation of foundation models. https://arxiv.org/abs/2307.02028
9. Alsentzer et al. (2019). Publicly available clinical BERT embeddings. https://aclanthology.org/W19-1909/ ; checkpoint: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
10. Collins et al. (2024). TRIPOD+AI statement. BMJ, 385, e078378. https://doi.org/10.1136/bmj-2023-078378
11. Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B ; Ollama: https://ollama.com/library/qwen3.5:9b
12. scikit-learn StratifiedGroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
