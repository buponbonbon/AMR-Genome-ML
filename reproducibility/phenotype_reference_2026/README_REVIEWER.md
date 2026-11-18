# Data and code for editorial and peer review

Manuscript: Same predictions, different verdicts: phenotype-reference versioning
in genomic resistance prediction.

This is the analysis support archive for the current submission. It contains
processed, accession-linked source records, paired dated phenotype comparisons,
cohort/selection records, the supplied 4,227-genome known-AMR development matrix,
fold assignments, the original 668-feature frozen model artifact and portable
CSV coefficients/schema, scored native feature matrices, annotation TSVs,
fixed predictions/rule calls, paired resampling outputs, exact-ranking and
MIC-boundary sensitivity results, and computational scripts.

## Quick reproduction

Use Python with the packages in requirements_review.txt. From this directory:

    python reproduce_review.py --out review_reproduced

This recomputes the main reference-relative confusion matrices, balanced
accuracy, MCC and AUROC; verifies probabilities for all five annotation arms
directly from the 668 coefficients and native feature matrices; and recomputes
the all-case and MIC-boundary-robust model-minus-rule BA contrasts.
It does not refit or tune the classifier and does not deserialize the joblib
model to perform these checks.

To rerun the full MIC-boundary analysis and paired country resampling:

    python scripts/analyze_mic_boundary_sensitivity.py --root . --out mic_reproduced

The output directory must not already exist. Original archived results remain
under results/rsos_readiness_extension_20261113/MIC_boundary_analysis.

Figure 2 and Figure 3 scripts and their backing data are under figures.
They are the current two-panel Fig2 and four-panel Fig3 versions.
Fig1.pdf is the author-supplied workflow diagram, not an algorithmic plot.
The four unchanged supplementary figures are also supplied.

## Large public source data

Raw genome assemblies are not duplicated in this archive; public sample,
assembly and run identifiers, acquisition records and input hashes are retained.
Exact recovery or reacquisition requires the corresponding public repositories,
AMRFinderPlus 4.2.7 and the two declared database releases.

The large NCBI metadata snapshot TSVs are available from version-specific
NCBI URLs in RAW_METADATA_SOURCES.csv, with archived checksums and byte counts.
All 20 URLs responded HTTP 200 during package preparation. The selected-source
paired comparisons and audit outputs are included locally. To restore the
complete public snapshot inputs to their original relative paths:

    python fetch_public_raw_metadata.py --all

This is approximately 3 GB of additional public data. Use --name followed by
a snapshot filename to fetch one file, or run without arguments to list files.
Downloads are checked against the archived SHA-256; a changed download is not
silently accepted as the original snapshot.

The original development workflow remains at the repository root:
https://github.com/buponbonbon/AMR-Genome-ML
The current audit/challenge code and processed inputs are supplied together in
this reproducibility directory. See README.md for the public repository layout
and the integrity/reproduction commands.

## Directory guide

- results/frozen_model_challenge_20261110: supplied development/E2 inputs and
  portable frozen coefficients and specification.
- results/received_bundle_audit_20261024: historical/current source crosswalks,
  focal records and phenotype-provenance checks.
- results/novelty_followup_20261110: source-level multi-taxon/drug comparisons,
  method-reference comparisons and bounded negative controls.
- results/database_reference_factorial_20261111: E2 and CCRE same-DNA
  database/reference comparisons and paired source resampling.
- results/strict_CCRE_followup_20261111: native CCRE354 scoring, technical
  missingness, cluster checks and later retained-metadata audit.
- results/independent_reference_extension_20261112: EuSCAPE170 analysis,
  exact hypothetical reassignment budgets, clinical-method replay and
  additional-taxon reference windows.
- results/rsos_readiness_extension_20261113: MIC stress protocol and results.
- scripts: computational acquisition, reconciliation, scoring, analysis
  and independent-check scripts.

Some historical acquisition records retain the original workspace paths.
The portable reproduction entrypoint uses this archive's relative paths.
Reproducing annotation acquisition requires the external tools and public
genomes; it is separate from reproducing the fixed-prediction comparisons.

## Interpretation

The E2 dated source discrepancy is one correlated event, including its
extensions across taxa and drugs. Neither the source challenges nor additional
isolates are a second temporal event. E2 historical AST was not independently
adjudicated. CCRE uses original PHAS categories; EuSCAPE uses the declared
EUCAST14 non-meningitis BMD-derived scenario. Confusion counts are relative to
these references, not adjudicated clinical error rates. MIC robustness uses a
hypothetical one-twofold-dilution envelope, not measured assay error.

## Integrity and source terms

FILE_MANIFEST.csv and SHA256SUMS.txt cover the included files. Original data
records retain source attribution and are subject to their source terms.
The original project's software license is included as LICENSE.
No new repository DOI or public deposit has been invented for this archive.
