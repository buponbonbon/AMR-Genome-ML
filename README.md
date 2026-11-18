# AMR-Genome-ML

## Same predictions, different verdicts: phenotype-reference versioning in genomic resistance prediction

This repository provides the processed data, frozen model parameters, per-genome
predictions, analysis code and figure inputs for the current phenotype-reference
versioning study. It also preserves the earlier genomic model-development
workflow and historical result summaries.

**Start here:** [Current data and reproduction guide](reproducibility/phenotype_reference_2026/README.md).

### Current study

The principal experiment holds a 668-feature model and its predictions fixed
while comparing two dated phenotype references for the same 525 *Klebsiella
pneumoniae* genomes. Substituting 175 source-linked labels changes apparent
AUROC from 0.6673 to 0.9689 and reverses the balanced-accuracy point-estimate
ranking against a fixed broad annotation rule.

Controlled January/August annotation-database replay separates score changes
from the phenotype-reference effect. Additional frozen-model challenges use
354 available CCRE genomes and 170 EuSCAPE genomes, with explicit MIC-boundary
sensitivity and paired source-composition analyses. The wider taxon/drug audit
characterizes extensions of one correlated source event; it is not independent
temporal replication. EuSCAPE categories follow the declared EUCAST14
non-meningitis BMD-derived scenario.

### Reproduce the main results

With Python 3.13, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r reproducibility/phenotype_reference_2026/requirements_review.txt
cd reproducibility/phenotype_reference_2026
python verify_integrity.py
python reproduce_review.py --out /tmp/amr_reference_reproduced
python scripts/analyze_mic_boundary_sensitivity.py --root . --out /tmp/amr_mic_reproduced
python verify_reproduction.py --main-out /tmp/amr_reference_reproduced --mic-out /tmp/amr_mic_reproduced
```

Use a new MIC output directory for each rerun. The quick reproduction verifies
five annotation arms directly from portable frozen coefficients and native
feature matrices. The full MIC rerun independently reconstructs the strata and
10,000 paired country resamples and checks them against the archived results.
No classifier refitting or joblib deserialization is required.

### Data and code map

| Location | Contents |
| --- | --- |
| [`reproducibility/phenotype_reference_2026/`](reproducibility/phenotype_reference_2026/) | Current study data, predictions, coefficients, schema, annotations, cohort/source audits, resampling arrays, scripts and figures |
| [`RAW_METADATA_SOURCES.csv`](reproducibility/phenotype_reference_2026/RAW_METADATA_SOURCES.csv) | Version-specific NCBI metadata URLs, byte counts and SHA-256 hashes; optional checksum-verified fetcher included |
| [`figures/`](reproducibility/phenotype_reference_2026/figures/) within the reproduction directory | Current Fig1–Fig3 and FigS1–FigS4, Figure 2/3 code and backing data |
| [`historical_support/`](reproducibility/phenotype_reference_2026/historical_support/) | Supplied historical E2/E3 per-genome predictions, targeted ANI-neighbour audit, cross-score and AST-provenance records |
| Root `scripts/`, `notebooks/`, `environment/` | Earlier model-development and provenance workflow |
| Root `figure/` and `AMR_Genome_ML_key_metrics_summary.csv` | Historical figure assets and compact result summary; use the reproduction directory for the current manuscript |
| [`docs/README_original_workflow.md`](docs/README_original_workflow.md) | Previous README retained for the original development/E2/legacy-E3 analyses |

The original development matrix contains 4,227 genomes. Its supplied known-AMR
matrix, fold assignments, original 668-feature model artifact and portable
coefficients/schema are included in the current reproduction directory.
The legacy E3 analysis used a separately fitted model and is not another arm
of the current fixed-model comparison.

### Public source inputs and reuse

Raw genomes remain available from their public repositories through the
accession/version and acquisition records. Complete metadata snapshots are
approximately 3 GB and can be recovered from the version-specific URLs with
the supplied checksum-checking fetcher. Included processed inputs support
reproducing the principal fixed-prediction comparisons without those downloads.
Full annotation reacquisition additionally needs the stated AMRFinderPlus and
database releases. Details and source attribution are in the reproduction guide.

The software is released under the [MIT license](LICENSE). Original source data
retain their source terms and attribution. SHA-256 manifests cover the curated
reproduction files. No repository DOI or external archival deposit is claimed.
