# Literature Review

**Project:** Predicting ICD-Defined Low Back Pain from Emergency Department Chief Complaints: A Comparative NLP Study

**Research Question:** How do different NLP models compare in their ability to predict ICD-defined low back pain from emergency department chief complaints?

This document summarizes the literature informing my research design. It is a working reference for understanding previous studies, selecting methods, and distinguishing literature-based decisions from project-specific adaptations. It is not intended to be a systematic literature review.

## 1. Literature Inventory

| ID | Reference | Research Focus | Relevance to My Project | Priority |
|---|---|---|---|---|
| R1 | Lee et al. (2019) | Clinical text classification and diagnostic categories | Task formulation | Core |
| R2 | Xie et al. (2022) | MIMIC-IV-ED prediction benchmark | Dataset and evaluation framework | Core |
| R3 | Miotto et al. (2020) | Acute LBP identification from clinical notes | LBP labels, keyword baseline, TF-IDF and training data efficiency | Core |
| R4 | Valmianski et al. (2019) | Traditional NLP versus BERT for patient-generated complaints | Main motivation for comparing model complexity | Core |
| R5 | Alsentzer et al. (2019) | Clinical BERT pretraining | Pretrained checkpoint selection | Supporting |
| R6 | Chang et al. (2020) | Clinical BERT for ED chief complaint classification | BERT implementation and initial configurations | Core |
| R7 | Brown et al. (2020) | Zero-shot and few-shot language modeling | Prompt-based classification | Supporting |
| R8 | Wornow et al. (2023) | EHRSHOT few-shot benchmark | Limited-training-data evaluation | Core |
| R13 | Weidman et al. (2026) | Lexical machine learning using extremely short EMS chief complaints | Predictive value of very short clinical texts and motivation for comparing lexical and contextual NLP methods | Core |

### Dataset and Reference Resources

| ID | Resource | Purpose |
|---|---|---|
| R9 | MIMIC-IV-ED v2.2 | Source dataset |
| R10 | NCQA LBP imaging measure | Context for the original value set purpose |
| R11 | NLM VSAC | ICD reference value sets |
| R12 | Bio_ClinicalBERT model card | Checkpoint and pretraining documentation |

## 2. Comparative Literature Matrix

| Study | Input | Prediction Target | Methods | Main Finding or Contribution |
|---|---|---|---|---|
| Lee [R1] | ED chief complaints and discharge diagnosis text | Diagnostic categories | Traditional and recurrent neural text classifiers | Supports the feasibility of mapping clinical text to diagnostic categories. |
| Xie [R2] | ED encounter data | Hospitalization, critical outcomes and reattendance | ML prediction benchmarks | Provides a reproducible ED benchmarking framework. |
| Miotto [R3] | Primary care clinical notes | Acute LBP episodes | ICD-based identification, keyword rules, TF-IDF + LR, ConvNet | ConvNet trained on manually annotated data outperformed LR approaches; administrative codes and clinical annotations are distinct reference sources. |
| Valmianski [R4] | Patient-generated reasons for visits | 795 complaint categories | TF-IDF and BERT-based approaches | TF-IDF achieved higher test PR-AUC than the strongest evaluated BERT approach. |
| Alsentzer [R5] | Clinical notes | Multiple clinical NLP tasks | Clinical-domain BERT pretraining | Establishes the basis for publicly available clinical BERT representations. |
| Chang [R6] | ED chief complaints | Provider-assigned structured complaint labels | Clinical BERT, LSTM, ELMo | Demonstrates clinical BERT classification and contextual representation extraction. |
| Brown [R7] | Natural-language task instructions | Multiple NLP tasks | Pretrained LLM prompting | Demonstrates zero-shot and few-shot task performance without task-specific gradient updates. |
| Wornow [R8] | Longitudinal structured EHR data | Multiple clinical prediction tasks | Foundation model few-shot benchmark | Provides a framework for evaluating models with limited labeled data. |
| Weidman [R13] | Clinician-recorded EMS chief complaints | Receipt of out-of-hospital lifesaving interventions | Unigram TF-IDF, word count and Random Forest | Demonstrated that chief complaints with a median length of two words can support clinical outcome prediction, with consistent performance across two annual cohorts. |

| **My project** | **ED chief complaint text only** | **Binary ICD-defined LBP** | **Regex, word/character TF-IDF, frozen/fine-tuned Clinical BERT, zero-shot LLM** | **To be evaluated under consistent experimental conditions.** |

The studies use different datasets, inputs, targets, and evaluation metrics. Their reported performance should not be compared directly across studies.

## 3. Literature-to-Methodology Mapping

### 3.1 Dataset and Reference Labels

| Component | Literature Foundation | My Implementation | Relationship |
|---|---|---|---|
| Dataset | Xie [R2]; MIMIC-IV-ED [R9] | MIMIC-IV-ED v2.2 | Adoption of dataset |
| Prediction task | Lee [R1] | Binary LBP classification from chief complaint text alone | Adaptation |
| Reference definition | NCQA / VSAC [R10–R11] | Match recorded ICD-9-CM and ICD-10-CM diagnoses against selected NCQA value sets | Adoption of code lists; adaptation of label construction |
| Label interpretation | Miotto [R3] | Measure agreement with ICD-derived labels, not independently validated clinical diagnoses | Methodological consideration |
| Data preparation | Xie [R2] | Encounter-level integration and quality checks; exclude missing or empty complaints and unavailable valid diagnoses | Adaptation |
| Data splitting | Benchmarking principles [R2] | Patient-grouped train/validation/test partitions | Project-specific |

A positive label requires at least one matching recorded diagnosis. A negative label requires valid diagnosis information but no matching code. The full NCQA imaging measure eligibility criteria are not implemented. Exact OIDs, expansion versions and historical coding compatibility will be documented in P2. No additional clinical inclusion or exclusion rules are introduced.

### 3.2 Model Selection

| Model | Literature Foundation | My Implementation | Relationship |
|---|---|---|---|
| M0 — Regex | Miotto [R3] | Case-insensitive keyword rules with word boundaries, developed on training/validation data | Adaptation; not a reproduction of WordSearch |
| M1 — Word TF-IDF + LR | Valmianski [R4]; Miotto [R3] | Word n-grams and regularized logistic regression | Adaptation |
| M2 — Character TF-IDF + LR | Traditional sparse text representation approach [R3–R4] | Character n-grams and regularized logistic regression | Project-specific representation choice |
| M3 — Frozen Clinical BERT + LR | Alsentzer [R5]; model card [R12] | Frozen final-layer CLS embeddings with logistic regression | Checkpoint adoption; project-specific classifier design |
| M4 — Fine-tuned Clinical BERT | Chang [R6]; Alsentzer [R5] | End-to-end binary classification using the same pretrained checkpoint as M3 | Adaptation |
| M5 — Zero-shot LLM | Brown [R7] | Local Qwen3.5:9b, fixed instruction, binary response, no labeled examples | Adaptation |
| M6 — Concept-informed LLM (optional) | NCQA / VSAC [R10–R11] | Same LLM with fixed reference concept descriptions in the prompt | Proposed project-specific condition; not finalized |

### 3.3 Selected Implementation Details

| Design Decision | Literature Basis | Project Choice |
|---|---|---|
| Word TF-IDF | Valmianski [R4]; Miotto [R3] | Word n-grams (1, 2) |
| Character TF-IDF | Project-specific adaptation | Character n-grams (2, 5) to capture spelling variation and abbreviations |
| Regularized LR | Miotto [R3] | L1 for sparse TF-IDF classifiers; L2 for frozen BERT classifier |
| Clinical BERT checkpoint | Alsentzer [R5, R12] | `emilyalsentzer/Bio_ClinicalBERT` |
| Maximum sequence length | Chang [R6] | 64 tokens |
| Fine-tuning epoch budget | Chang [R6] | 3 epochs |
| Fine-tuning learning rate | Chang [R6] used 1e-4 | Validation candidates: 2e-5 and 1e-4 |
| Zero-shot prompting | Brown [R7] | Fixed English instruction without labeled examples |

These are initial design choices, not a reproduction of any individual paper. Detailed settings and tuning procedures are maintained in `research_plan.md` and the experimental configurations.

### 3.4 Experimental Design and Evaluation

| Component | Literature Foundation | My Implementation | Relationship |
|---|---|---|---|
| Full-data comparison | Valmianski [R4]; Xie [R2] | Compare models using common evaluation partitions | Adaptation |
| Reduced training data | Miotto [R3]; EHRSHOT [R8] | Nested stratified subsets: 0.1%, 1%, 10%, 100% | Adaptation |
| Models in reduced-data experiment | Project-specific | Retrain M1–M4; M0 and M5 remain fixed reference baselines | Project-specific |
| Repeated sampling | Project-specific | Repeat reduced-data conditions subject to computational feasibility | Project-specific |
| Evaluation metrics | Comparative classification framework | Positive-class F1 as primary; precision, recall and confusion matrices; AP and AUROC for score-producing M1–M4 | Project-specific |
| Leakage prevention | Project-specific experimental controls | Patient-grouped splits, train-only feature fitting, validation-only tuning, fixed held-out test set | Project-specific |

The reduced-data experiment evaluates training data efficiency rather than strict few-shot learning because the full validation partition remains available. It does not reproduce EHRSHOT's sampling protocol.

### 3.5 Text Preprocessing

Previous studies adopt different text preprocessing strategies depending on the text source, prediction task and NLP method.

| Literature | Original Preprocessing Approach | Implication for My Study |
|---|---|---|
| Lee [R1] | Special character removal, conversion of digits to written words, and lowercasing; no abbreviation expansion, spelling correction or medical ontology mapping | Chief complaint classification does not necessarily require extensive semantic normalization. |
| Xie [R2] | General data preparation, quality checks and a fixed test set | Maintain consistent input data and experimental conditions across models. |
| Miotto [R3] | Punctuation and number removal, lemmatization, vocabulary filtering and spelling correction; model-specific stopword removal and negation handling | Preprocessing designed for longer clinical notes may not be suitable for short chief complaints. |
| Valmianski [R4] | TF-IDF-specific lowercasing, stopword removal, number replacement, tokenization and n-gram construction | Different text representations can have their own preprocessing pipelines. |
| Chang [R6] | Exclusion of missing chief complaints; Clinical BERT sequence length limited to 64 tokens | Handle model-specific input requirements within the corresponding model pipeline. |

**My approach**

Chief complaints are short clinical texts in which capitalization, punctuation, numbers, abbreviations and spelling variations may carry useful information. For example, removing punctuation may disrupt clinical abbreviations such as `s/p`, removing numbers may discard information about symptom duration or severity, and stopword removal may alter negation. Capitalization may reflect emphasis, documentation conventions or individual preferences.

Since the objective is to compare different NLP methods, extensive common preprocessing could remove information that some models might otherwise capture.

| Design Decision | My Implementation | Relationship |
|---|---|---|
| Common preprocessing | Exclude missing or empty text; normalize whitespace | Project-specific |
| Information preservation | Retain capitalization, punctuation, numbers, abbreviations, stopwords and spelling variations | Literature-informed adaptation |
| Short and repeated complaints | Retain valid short texts and identical complaints from distinct encounters | Project-specific |
| Model-specific preprocessing | Handle case normalization, tokenization, n-grams, padding, truncation and prompt formatting within individual model pipelines | Adaptation |
| Leakage prevention | Fit corpus-dependent transformations using only the current training subset | Project-specific experimental control |

The resulting comparison evaluates complete NLP pipelines under consistent data and experimental conditions, rather than isolating the effects of model architecture alone.

Minimal common preprocessing is a project-specific design choice informed by the literature, not a claim that it is universally superior.


## 4. Key Findings and Research Insights

| Observation | Evidence | Implication for My Study |
|---|---|---|
| More complex models do not necessarily perform better. | TF-IDF outperformed evaluated BERT approaches in Valmianski [R4]. | Retain strong traditional NLP baselines. |
| Model rankings may depend on the task and available text. | Valmianski [R4] and Miotto [R3] report different model comparisons on different tasks. | Compare all models on the same chief complaint classification task; do not attribute differences across papers solely to text length. |
| ICD codes and clinical annotations are not interchangeable. | Miotto [R3]; NCQA measure context [R10]. | Interpret predictions as agreement with the selected ICD reference definition. |
| Clinical BERT can be applied to chief complaint classification. | Chang [R6]; Alsentzer [R5]. | Compare frozen representations with task-specific fine-tuning. |
| Training data availability is an important experimental dimension. | Miotto [R3]; Wornow [R8]. | Examine model performance across multiple training subset sizes. |
| Brief complaints may not contain enough information to determine the eventual diagnosis. | Project motivation informed by the clinical text classification literature [R1, R4, R6]. | Recognize ambiguity and limited context when interpreting errors. |

### Predictive Value of Extremely Short Clinical Text

Weidman et al. (2026) examined whether clinician-recorded chief complaints could predict receipt of lifesaving interventions in out-of-hospital emergency care. The study included approximately 9.2 million EMS encounters from 2023 for model development and 9.8 million encounters from 2024 for temporal validation. Chief complaints were extremely short, with a median word count of two and an interquartile range of one to three words.

The authors applied stopword removal and lemmatization using NLTK, followed by unigram TF-IDF representation. Chief complaint word count was included as an additional feature, and Random Forest models were used for separate binary classification tasks. Six of the seven primary intervention categories achieved AUROCs between 0.821 and 0.929 in the derivation cohort, with generally consistent performance in the temporal validation cohort.

The authors selected traditional machine learning rather than an LLM because of computational considerations and the expectation that predictions from very short complaints would depend primarily on keywords rather than complex word order or context. Their exploratory analysis also found that adding bigrams substantially increased training time without improving predictive performance. However, the study did not directly compare lexical methods with contextual language models.

This study is particularly relevant to my project because the chief complaints in MIMIC-IV-ED have a similarly limited text length, with a median of two words. Together with the findings of Valmianski et al. (2019), it strengthens my motivation to examine whether more sophisticated NLP methods provide additional predictive value when the available clinical text is extremely short.

The preprocessing and evaluation protocols are not directly adopted. In particular, Weidman et al. restricted the vocabulary to words appearing in both annual cohorts, whereas my study fits corpus-dependent transformations using only the current training subset. I also retain minimal common preprocessing to preserve potentially useful linguistic information across different NLP methods.

**Implementation review:** The publicly available 2023 and 2024 notebooks (`eso_2023.ipynb` and `eso_2024.ipynb`) provide additional preprocessing details not explicitly reported in the paper. Both implementations apply lowercasing, punctuation replacement, whitespace normalization, regex-based tokenization, within-complaint token deduplication, NLTK English stopword removal and WordNet lemmatization before unigram TF-IDF vectorization.

Chief complaint length is calculated after basic text normalization. This differs from the raw-text length analysis in my study and should be considered when comparing descriptive statistics.

A methodological difference concerns the fitting of text representations. In the 2023 implementation, TF-IDF is fitted on the full derivation corpus before cross-validation. In the 2024 implementation, TF-IDF is fitted independently on the temporal validation corpus. Both implementations restrict their vocabulary to features shared across the two annual cohorts.

Although the Random Forest classifier is reused without retraining during temporal validation, the complete NLP pipeline is not fully frozen. These design choices introduce evaluation-corpus dependencies without directly using outcome labels.

In my study, I retain minimal common preprocessing and fit all corpus-dependent transformations on the current training subset only. This is particularly important for maintaining a consistent evaluation protocol in the training data efficiency experiment.

### Important Limitations

| Limitation | Implication |
|---|---|
| NCQA value sets originate from a quality measure. | They are not an independent clinical gold standard. |
| Value set expansion years may differ from historical coding years. | Historical code coverage may affect reference labels. |
| Clinical BERT was pretrained on MIMIC-III notes. | Potential source-population overlap with MIMIC-IV-ED should be acknowledged; actual evaluation overlap is not established. |
| Full validation data remain available in reduced-data experiments. | Results measure training-set-size sensitivity, not a strict total labeling budget. |
| Optional concept-informed prompting is not finalized. | It is not a committed experiment or separate research question. |

## 5. Research Gap and Project Positioning

The reviewed studies address related but different tasks, including chief complaint standardization, diagnostic category prediction, acute LBP identification from longer clinical notes, and limited-data evaluation of clinical foundation models.

My project brings these ideas together in a focused binary classification benchmark using ED chief complaint text and ICD-defined LBP labels. Based on my previous exploratory work and the findings of Valmianski et al. (2019), I suspect that more sophisticated NLP models may not necessarily perform better on very short clinical texts. I will examine this intuition by comparing different NLP approaches under consistent experimental conditions and evaluating their performance across different amounts of supervised training data.

The study measures agreement with recorded ICD-derived labels rather than clinical diagnostic accuracy. The research gap is relative to the literature reviewed here; I do not claim that no previous work has addressed this task.

## 6. References

The following working references are retained from the previous literature review. Links, titles, authors and publication metadata should be checked against the original publications before final ACM submission.

### Research Papers

- **[R1]** Lee, S. H., et al. (2019). Chief complaint classification with recurrent neural networks. *Journal of Biomedical Informatics*. https://pubmed.ncbi.nlm.nih.gov/30926471/
- **[R2]** Xie, F., et al. (2022). Benchmarking emergency department prediction models with machine learning and public electronic health records. *Scientific Data*, 9, 658. https://doi.org/10.1038/s41597-022-01782-9
- **[R3]** Miotto, R., et al. (2020). Identifying acute low back pain episodes in primary care practice from clinical notes: Observational study. *JMIR Medical Informatics*, 8(2), e16878. https://doi.org/10.2196/16878
- **[R4]** Valmianski, I., et al. (2019). Evaluating robustness of language models for chief complaint extraction from patient-generated text. *arXiv:1911.06915*. https://arxiv.org/abs/1911.06915
- **[R5]** Alsentzer, E., et al. (2019). Publicly available clinical BERT embeddings. *Proceedings of the 2nd Clinical Natural Language Processing Workshop*. https://aclanthology.org/W19-1909/
- **[R6]** Chang, D., et al. (2020). Generating contextual embeddings for emergency department chief complaints. *JAMIA Open*, 3(2), 160–166. https://doi.org/10.1093/jamiaopen/ooaa022
- **[R7]** Brown, T. B., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33. https://arxiv.org/abs/2005.14165
- **[R8]** Wornow, M., et al. (2023). EHRSHOT: An EHR benchmark for few-shot evaluation of foundation models. *Advances in Neural Information Processing Systems*, 36. https://arxiv.org/abs/2307.02028

### Dataset, Reference Standards, and Model Resources

- **[R9]** MIMIC-IV-ED (version 2.2). PhysioNet. https://doi.org/10.13026/5ntk-km72
- **[R10]** National Committee for Quality Assurance (NCQA). Use of Imaging Studies for Low Back Pain (LBP). https://www.ncqa.org/report-cards/health-plans/state-of-health-care-quality-report/use-of-imaging-studies-for-low-back-pain-lbp/
- **[R11]** National Library of Medicine. Value Set Authority Center (VSAC). https://vsac.nlm.nih.gov/ — selected NCQA Uncomplicated Low Back Pain value sets; exact expansion metadata to be documented in P2.
- **[R12]** Bio_ClinicalBERT model card. https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

