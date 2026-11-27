# Experimental Protocol

**Project:** Predicting ICD-Defined Low Back Pain from Emergency Department Chief Complaints: A Comparative NLP Study

**Milestone:** `hrp-data-ready` | **Scope:** P4–P6

This document records the shared experimental decisions established in P3 and the evaluation controls specified in `research_plan.md`. It is the implementation reference for all subsequent model notebooks. Model-specific configurations and their selected values are documented separately. P1–P3 outputs are reused without regenerating labels, partitions or training subsets.

**Research question:** How do different NLP models compare in their ability to predict ICD-defined low back pain from emergency department chief complaints?


## 1. Dataset and prediction task

| Component | Final decision |
|---|---|
| Modeling population | 422,682 ED encounters from 204,509 unique patients |
| Prediction task | Binary classification of ICD-defined LBP |
| Model input | Common preprocessed chief complaint only |
| Reference label | NCQA ICD-defined binary LBP label |
| Positive prevalence | 3.86% |
| Negative-to-positive ratio | 24.89:1 |
| Unit of prediction and evaluation | Encounter (stay_id) |


### Fixed dataset summary

| Partition | Encounters | Patients | Positive | Negative |
|---|---:|---:|---:|---:|
| Train | 295,877 | 143,119 | 11,429 | 284,448 |
| Validation | 42,268 | 20,483 | 1,632 | 40,636 |
| Test | 84,537 | 40,907 | 3,266 | 81,271 |
| Total | 422,682 | 204,509 | 16,327 | 406,355 |

Dataset: MIMIC-IV-ED v2.2. The model input is `chiefcomplaint_clean`; the target is `lbp_label`. Identifiers are used for linkage and tracking only. The reference label indicates agreement with recorded diagnoses matched to the frozen NCQA ICD value sets, not independently validated clinical disease status.

## 2. Fixed patient-grouped partitions

| Component | Final decision |
|---|---|
| Grouping unit | Patient (subject_id) |
| Stratification target | Encounter-level lbp_label |
| Splitting algorithm | StratifiedGroupKFold |
| Splitting strategy | Two-stage fixed splitting |
| Stage 1 | n_splits=5; first fold as test, remaining folds as development |
| Stage 2 | n_splits=8 within development; first fold as validation, remaining folds as training |
| Target partition proportions | 70% training / 10% validation / 20% test |
| Input ordering | Sort by stay_id before splitting |
| Shuffle | True |
| Splitting random seed | 42 for both stages |
| Cross-validation | Not performed; the algorithm is used to construct fixed partitions |
| Patient overlap across partitions | Strictly zero |
| Encounter overlap across partitions | Strictly zero |
| Encounter coverage | 100% |
| Split proportion tolerance | Absolute deviation ≤ 0.5 percentage points |
| Class prevalence tolerance | Absolute deviation ≤ 0.1 percentage points from overall prevalence |
| Tolerance failure handling | Investigate and document; no arbitrary seed switching or manual encounter reassignment |
| Identical complaint strings across partitions | Allowed when associated with different patients |
| Partition consistency | All models and experiments use the same fixed partitions |

## 3. Training data efficiency

| Component | Final decision |
|---|---|
| Experimental purpose | Training data efficiency |
| Applicable models | M1–M4 supervised models |
| Fixed reference models | M0 and M5; no retraining across fractions |
| Sampling population | Training partition only |
| Sampling unit | Encounter (stay_id) |
| Sampling method | Stratified random sampling without replacement |
| Training fractions | 0.1%, 1%, 10%, 100% |
| Nested sampling | Class-specific seeded permutations followed by proportional prefix selection |
| Nesting requirement | 0.1% ⊆ 1% ⊆ 10% ⊆ 100% |
| Rounding rule | Independent class-wise flooring |
| Positive sample count | max(1, floor(N_pos × fraction)) |
| Negative sample count | floor(N_neg × fraction) |
| Minimum class requirement | Both positive and negative samples must be present |
| Class distribution | Approximately preserve natural training prevalence |
| Sampling seeds | 42, 123, 2026 |
| Class counts across seeds | Identical for the same fraction |
| Subset membership consistency | All applicable models use identical encounter IDs for each fraction and seed |
| Reduced-data repetitions | Three sampling seeds planned, subject to computational feasibility |
| Full-data condition | Reuse the 100% training result from Experiment 1 |
| Validation set | Full, fixed validation partition for all fractions |
| Test set | Full, fixed test partition for all fractions |
| Learning curve reporting | Mean and standard deviation across reduced-data runs; single full-data endpoint |


### Fixed training subset sizes

| Fraction | Encounters | Positive | Negative |
|---|---:|---:|---:|
| 0.1% | 295 | 11 | 284 |
| 1% | 2,958 | 114 | 2,844 |
| 10% | 29,586 | 1,142 | 28,444 |
| 100% | 295,877 | 11,429 | 284,448 |

Counts are identical across sampling seeds. Membership is obtained using the relevant seed column value less than or equal to the requested training fraction.

## 4. Class imbalance and preprocessing

| Component | Final decision |
|---|---|
| Class imbalance handling | Inverse-frequency class weighting |
| M1–M3 class weighting | class_weight="balanced" |
| M4 class weighting | Weighted cross-entropy |
| Class weight calculation | Recalculate using the current training subset |
| Oversampling / undersampling / SMOTE | Not applied for class balancing |
| Common preprocessing | Already completed in Section 3.3 |
| Model-specific preprocessing | Performed within each model pipeline |
| Vocabulary fitting | Current training subset only |
| IDF estimation | Current training subset only |
| Other learned transformations | Current training subset only |
| Model initialization | Independent training from the original initialization or pretrained checkpoint, as applicable |

## 5. Development and evaluation controls

| Component | Final decision |
|---|---|
| Model development | Training and validation partitions only |
| Hyperparameter selection | Development data only |
| Classification threshold selection | Validation partition only, where applicable |
| Zero-shot prompts | Fixed without label-driven refinement |
| Test set access | Final evaluation after all core experimental configurations are frozen |
| Primary evaluation metric | Positive-class F1 |
| Supporting metrics | Precision, recall, confusion matrix, AP and AUROC where applicable |
| Experimental independence | Experiment 1 test results must not influence Experiment 2 development |
| Data efficiency interpretation | Training data efficiency, not strict few-shot or total label-budget efficiency |


### Model comparison and evaluation

| ID | Method | Experiment 1 | Experiment 2 |
|---|---|---|---|
| M0 | Rule-based regex | Fixed classifier | Fixed reference |
| M1 | Word TF-IDF + logistic regression | Full training | Retrain |
| M2 | Character TF-IDF + logistic regression | Full training | Retrain |
| M3 | Frozen Bio_ClinicalBERT + logistic regression | Full training | Retrain classifier |
| M4 | Fine-tuned Bio_ClinicalBERT | Full training | Retrain from original checkpoint |
| M5 | Zero-shot local LLM | Fixed prompt | Fixed reference |

M6 remains optional and is not required for the core study. Experiment 2 reuses full-data results from Experiment 1. Use full-training-derived hyperparameters for reduced-data runs, with independent threshold selection for each run.

Use positive label 1 and `zero_division=0` for classification metrics. Report positive-class F1, precision, recall and confusion matrix for all models; report average precision (AP) and AUROC for M1–M4. AP is not trapezoidal PR-curve area. Do not invent confidence scores for hard-label models.

For supervised models, select the threshold maximizing validation F1 from 0.05, 0.10, ..., 0.95 (19 candidates); resolve ties using the higher threshold. Freeze the selected threshold before test evaluation. M0 and M5 use direct binary predictions. Freeze all core configurations for both experiments before final test evaluation. No independent robustness, fairness, XAI, calibration, temporal drift or computational-efficiency experiment is planned.

For models with multiple regularization candidates, each candidate is evaluated jointly with the predefined classification threshold grid. The combination that achieves the highest positive-class F1 score on the validation set is selected. For tied F1 scores within the same regularization setting, the higher threshold is preferred. If the best F1 scores are tied across different C values, the smaller C is selected. The selected fitted pipeline and classification threshold are retained for subsequent evaluation without refitting.

## 6. Data outputs and reproducibility

| Component | Final decision |
|---|---|
| Modeling dataset | modeling_dataset.csv — already saved and validated |
| Partition membership file | dataset_splits.csv |
| Partition membership schema | stay_id, split |
| Training subset membership file | training_subsets.csv |
| Subset membership format | Wide format, one row per training encounter |
| Subset membership schema | stay_id, seed_42, seed_123, seed_2026 |
| Membership encoding | Smallest training fraction containing each encounter |
| Software reproducibility | Record relevant package versions and random seeds |
| Output validation | Verify saved files against in-memory membership data |
| Data storage | Local results/03_experimental_setup/ |
| GitHub | Code and documentation only; patient-level data excluded |


Store structured experiment results with experiment ID, model ID, training fraction, seed, sample counts, configuration, threshold and metrics. Record relevant software versions, model revisions, prompts and configurations. Keep restricted encounter data, identifiers, clinical text, predictions, splits, embeddings and trained artifacts in controlled local or authorized environments; exclude `results/` and `models/` from Git and inspect notebook outputs before publication. Do not upload restricted data to unapproved services.

## 7. Literature foundation

| Component | Literature reference | Adopted principle |
|---|---|---|
| Fixed experimental partitions | Xie et al. (2022) | Common held-out benchmark |
| Comparative NLP evaluation | Lee et al. (2019); Chang et al. (2020) | Consistent evaluation conditions |
| Data separation | Valmianski et al. (2019) | Control dependencies between training and evaluation data |
| Training data efficiency | Miotto et al. (2020); Wornow et al. (2023), EHRSHOT | Controlled training-data reduction and learning curves |
| Preprocessing and evaluation integrity | Weidman et al. (2026), including implementation review | Critical implementation review informs our decision to avoid evaluation-corpus-dependent transformations |
| Group-aware stratification | Scikit-learn StratifiedGroupKFold | Patient separation with approximate class stratification |
