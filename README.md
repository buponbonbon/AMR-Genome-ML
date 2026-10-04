# AMR-Genome-ML

## Comparative Genomic Representations for Meropenem-Resistance Prediction in *Klebsiella pneumoniae*

[![Status](https://img.shields.io/badge/status-manuscript%20prepared-blue)](#manuscript)
[![Target journal](https://img.shields.io/badge/target%20journal-Interdisciplinary%20Sciences%3A%20Computational%20Life%20Sciences-6f42c1)](#manuscript)
[![Organism](https://img.shields.io/badge/organism-Klebsiella%20pneumoniae-2ea44f)](#study-overview)
[![Phenotype](https://img.shields.io/badge/phenotype-meropenem%20resistance-orange)](#study-overview)

This repository contains the analysis code and publication assets for a genomic machine-learning study of **meropenem resistance in *Klebsiella pneumoniae***.

The main question is not simply whether a model can obtain a high AUROC during internal validation. Instead, this study asks:

> **Does strong lineage-aware internal validation actually guarantee that a genomic AMR model will perform well in an independent population?**

The study compares three genomic representations—**known-AMR determinants, pan-genome features, and unitigs**—under random, genomic-cluster-aware, and MLST-aware validation, followed by sequential evaluation on two independent external datasets.

---

## Study overview

The development dataset contains **4,227 *K. pneumoniae* genomes** with meropenem susceptibility labels:

- **1,601 resistant**
- **2,626 susceptible**

Three genomic representations were evaluated:

| Representation | Description | Final feature space |
|---|---|---:|
| **Known-AMR** | Known resistance genes and resistance-associated mutations | 668 features in the original development schema |
| **Pan-genome** | Presence/absence of PLFam protein families | 22,685 features after minor-state filtering |
| **Unitigs** | Reference-free sequence features derived from compacted de Bruijn graphs | 4,718,475 features after minor-state filtering |

For pan-genome and unitig models, supervised feature ranking was performed **inside each outer training fold**, with the top 2,000 features retained for model fitting.

Four machine-learning algorithms were compared:

- L2-regularized logistic regression
- RBF support vector classifier
- ExtraTrees
- Histogram Gradient Boosting

---

## Validation design

A major focus of this project is the difference between **internal validation** and **true external transportability**.

Three 5-fold internal validation schemes were used:

1. **Random validation**  
   Genomes are split approximately at random while preserving phenotype balance.

2. **Genomic-cluster-aware validation**  
   Genomes belonging to the same genomic cluster are kept in the same fold.

3. **MLST-aware validation**  
   Genomes with the same resolved sequence type are kept in the same fold.

Population structure was derived from PLFam profiles and reduced to 50 dimensions before clustering. The final design used **100 genomic clusters**.

MLST typing resolved **4,130/4,227 genomes** into **568 sequence types**; 97 unresolved genomes were retained as independent grouping units.

---

## Main result

The key result is the gap between very strong internal validation and substantially weaker performance on independent data.

| Evaluation | AUROC | Average precision | Balanced accuracy | MCC |
|---|---:|---:|---:|---:|
| Random validation | **0.980** | 0.976 | 0.950 | 0.905 |
| Genomic-cluster-aware | **0.971** | 0.964 | 0.938 | 0.887 |
| MLST-aware | **0.968** | 0.959 | 0.931 | 0.877 |
| External E2 | **0.667** | 0.411 | 0.640 | 0.259 |
| Confirmatory E3 | **0.778** | 0.664 | 0.701 | 0.427 |

The central finding is therefore:

> **Lineage-aware internal validation is necessary, but it is not sufficient to establish transportability to an independent bacterial population.**

The broader pan-genome and unitig representations also did **not consistently outperform** the simpler known-AMR baseline.

---

## External validation

### E2

The first independent external dataset contained:

- **525 genomes**
- **163 resistant**
- **362 susceptible**
- **289 NCBI ERD/SNP grouping units**

The original **668-feature known-AMR L2 logistic model** was fixed before external scoring.

Key E2 results:

- AUROC: **0.667** (95% CI 0.557–0.767)
- Average precision: **0.411**
- Balanced accuracy: **0.640**
- MCC: **0.259**
- Brier score: **0.316**
- Sensitivity: **0.669**
- Specificity: **0.610**

E2 also showed substantial calibration degradation and many high-confidence errors.

### E3

E3 was designed as a second, confirmatory external evaluation and contained:

- **306 genomes**
- **122 resistant**
- **184 susceptible**
- 251 public assemblies
- 55 assemblies generated from public raw SRA reads

Before E3 scoring, the development pipeline was harmonized **using development data only**. Two configurations were then frozen:

- **702-feature primary model**
- **668-feature sensitivity model**

Key E3 results for the primary model:

- AUROC: **0.778** (95% CI 0.722–0.829)
- Average precision: **0.664**
- Balanced accuracy: **0.701**
- MCC: **0.427**
- Brier score: **0.222**

No E3 labels were used for feature selection, recalibration, threshold tuning, or cohort modification.

---

## Biological interpretation

The project also maps high-dimensional pan-genome and unitig signals back to interpretable genomic regions.

Notable findings include:

- **`PLF_570_00004895`** — annotated as a KPC-family class A beta-lactamase
- **`unitig_233683`** — maps directly to KPC-family loci in representative ST11, ST101, and ST1107 genomes
- **`unitig_233720`** — maps near a transposase, approximately 959 bp from the KPC region
- Tn4401-like context was observed in representative carriers, including patterns compatible with Tn4401a-like and Tn4401b-like structures

These analyses are used as **biological context for model signals**, not as claims of a novel resistance mechanism or definitive mobile-element subtype.

---

## Workflow

```text
BV-BRC phenotype-linked genomes
            |
            v
Phenotype reconciliation and source-level QC
            |
            v
Common development cohort (n = 4,227)
            |
            +-------------------+-------------------+
            |                   |                   |
            v                   v                   v
       Known-AMR            Pan-genome            Unitigs
        668 features         22,685 PLFam        4.72M features
            |                   |                   |
            +-------------------+-------------------+
                                |
                                v
                 Model development and comparison
                                |
               +----------------+----------------+
               |                |                |
               v                v                v
             Random       Cluster-aware      MLST-aware
            validation      validation        validation
                                |
                                v
                       Freeze model/config
                                |
                                v
                       External E2 (n=525)
                                |
                                v
                 Post-E2 diagnostic analyses
                                |
                                v
            Development-only harmonization + freeze
                                |
                                v
                   Confirmatory E3 (n=306)
                                |
                                v
                 Final integrity / audit checks
```

---

## Repository organization

The repository is organized around the analysis stages used in the study.

```text
AMR-Genome-ML/
├── README.md
├── .gitignore
├── figure/
│   ├── Fig1.pdf
│   ├── Fig2.pdf
│   ├── Fig3.pdf
│   ├── FigS1.pdf
│   ├── FigS2.pdf
│   └── FigS3.pdf
├── scripts/
│   ├── 07_population_structure_and_validation_design.py
│   ├── 07a_mlst_typing.py
│   ├── 08_model_training_and_feature_selection.py
│   ├── 09_model_evaluation_and_robustness.py
│   ├── 10_explainability_and_biological_validation.py
│   ├── 10i_biological_annotation_prep.py
│   ├── 10j_external_biological_annotation.py
│   ├── file10k_tn4401_sequence_validation.py
│   ├── file11_external_cohort_leakage_firewall.py
│   ├── file11b_complete_snp_cluster_firewall.py
│   ├── file11c_blind_external_validation.py
│   ├── file11e_external_simple_baseline_comparators.py
│   ├── file11f_feature_source_shift.py
│   ├── file13_full_harmonized_amrfinder_development.py
│   ├── file14_build_strict_E3_cohort.py
│   ├── file14_freeze_development_qc_envelope.py
│   ├── file14_independent_E3_blind_validation.py
│   ├── file15_integrity_reporting.py
│   ├── build_unitig_bitpack_resume.cpp
│   └── run_full_unitig.py
└── AMR_Genome_ML_key_metrics_summary.csv
```

Large raw genomic files, intermediate matrices, checkpoints, compiled binaries, logs, and other generated artifacts are intentionally excluded from version control.

---

## Main software and versions

The analysis uses a combination of Python-based machine learning and external genomics tools.

| Tool | Version / setting | Purpose |
|---|---|---|
| AMRFinderPlus | 4.2.7 | Known-AMR annotation |
| AMRFinderPlus DB | 2026-08-07.1 | AMR reference database |
| unitig-caller | 1.3.2 | Unitig generation |
| Bifrost | 1.3.5 | Compacted colored de Bruijn graph |
| `mlst` | 2.35.0 | Sequence typing |
| Shovill | 1.4.2 | Assembly of raw E3 reads |
| SKESA | 2.5.1 | Assembly engine used by Shovill |
| scikit-learn | — | Machine-learning models and evaluation |
| Unitig k-mer size | `k=31` | Sequence representation |

The resumable C++ bit-packing utility in `scripts/build_unitig_bitpack_resume.cpp` was used to make the very large unitig presence/absence representation practical to process while retaining checkpoint and integrity checks.

---

## Data sources

This project uses publicly available genomic and antimicrobial-resistance data from resources including:

- **BV-BRC**
- **NCBI Pathogen Detection**
- **NCBI Sequence Read Archive (SRA)**

Raw genomic datasets are not redistributed in this repository. Large derived matrices are also excluded because of their size.

The repository focuses on the **analysis code, validation logic, integrity checks, manuscript figures, and compact result summaries** needed to understand and reproduce the analytical workflow.

---

## Reproducibility principles

Several safeguards were used throughout the project:

- phenotype-independent source-level genome QC
- explicit duplicate and phenotype reconciliation
- no laboratory metadata or sample identifiers used as predictive features
- feature selection for pan-genome and unitig data performed inside training folds
- group-aware validation by genomic cluster and MLST
- external overlap/leakage checks before E2 scoring
- model configurations frozen before external evaluation
- no E2/E3-driven threshold tuning or recalibration
- development-only harmonization before E3
- clustered bootstrap confidence intervals
- final integrity audit across cohort, feature, prediction, metric, and manifest files

The final E3 audit contained **98 checks: 97 PASS, 1 WARN, and 0 failures**. The warning reflected unresolved grouping information for a large fraction of E3 genomes rather than a model-integrity failure.

---

## Manuscript

**Title**

> *Comparative Genomic Representations for Meropenem-Resistance Prediction in Klebsiella pneumoniae: Lineage-Aware Evaluation, Biological Interpretation, and External Transportability*

**Manuscript type:** Original Article

**Target journal:** *Interdisciplinary Sciences: Computational Life Sciences* (Springer Nature)

**Submission status:** Manuscript prepared for submission.

### Authors

**Tran An Khang**  
FPT University, Vietnam  
Corresponding author: `khangtran.mschcm@gmail.com`

**Nguyen Dang Trinh**  
University of Science, Vietnam National University Ho Chi Minh City, Vietnam

**Equal contribution:** Tran An Khang and Nguyen Dang Trinh contributed equally to this work.

### Author contributions

Tran An Khang and Nguyen Dang Trinh contributed equally to the study conception and design, data curation, computational analysis, interpretation of results, visualization, and manuscript preparation. Both authors reviewed and approved the final manuscript.

---

## Funding and competing interests

**Funding:** The authors received no specific funding for this work.

**Competing interests:** The authors declare no competing interests.

---

## Intended use

This repository supports a research study of genomic AMR prediction and model transportability.

It is **not a clinical diagnostic system** and should not be used to guide antimicrobial treatment decisions without appropriate clinical validation, laboratory confirmation, and regulatory oversight.

---

## Citation

A formal citation will be added after publication.

For now, please cite the manuscript title and this repository when referring to the analysis before publication.

---

## Contact

For questions about the analysis or reproducibility:

**Tran An Khang**  
FPT University, Vietnam  
Email: `khangtran.mschcm@gmail.com`
