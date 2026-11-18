# Phenotype-reference versioning: data and code

Manuscript: **Same predictions, different verdicts: phenotype-reference versioning in genomic resistance prediction**.

This directory contains the processed data, frozen model specification,
per-genome predictions, analysis code, figure inputs and results for the
current manuscript. The repository root retains the earlier development
workflow; its compact metrics CSV is a historical result summary.

## Reproduce the principal results

Use Python 3.13 and the exact package versions recorded in
`requirements_review.txt` and `REVIEW_ENVIRONMENT.json`. From this directory:

```bash
python -m venv /tmp/amr-reference-venv
source /tmp/amr-reference-venv/bin/activate
python -m pip install -r requirements_review.txt
python verify_integrity.py
python reproduce_review.py --out /tmp/amr_reference_reproduced
python scripts/analyze_mic_boundary_sensitivity.py --root . --out /tmp/amr_mic_reproduced
python verify_reproduction.py --main-out /tmp/amr_reference_reproduced --mic-out /tmp/amr_mic_reproduced
```

Use a fresh output path for the MIC analysis; it refuses to overwrite an
existing directory. The integrity manifest covers archived inputs and outputs,
so keep new analysis output outside this directory. Creating the virtual
environment at the repository root also keeps it outside the archive.

The quick command recomputes 23 reference/database/method comparisons and
checks five sets of frozen probabilities from the original 668 coefficients
and native feature matrices. It also recomputes the CCRE/EuSCAPE all-case and
MIC-boundary-robust balanced-accuracy contrasts. The MIC command independently
reconstructs the per-case strata and 10,000 paired country resamples; the final
command compares the regenerated tables and bootstrap arrays with the archive.
Neither command refits the classifier. Reproduction uses portable CSV
coefficients and does not deserialize the archived joblib model.

To regenerate the full controlled database analysis, including 10,000 paired
source resamples, or the supplied historical E2/E3 sensitivity tables:

```bash
python scripts/analyze_frozen_database_reference_factorial.py --root . --out /tmp/amr_database_reproduced --draws 10000
python reproduce_historical.py --out /tmp/amr_historical_reproduced
```

## Find the data behind each analysis

| Analysis | Included paths under `results/` |
| --- | --- |
| Original 4,227-genome development matrix, fold assignments, original 668-feature model, coefficients and schema | `frozen_model_challenge_20261110/` |
| Historical/current E2 phenotype provenance and source crosswalks | `received_bundle_audit_20261024/` |
| Broader taxon/drug source comparisons and negative controls | `novelty_followup_20261110/`; `independent_reference_extension_20261112/` |
| Fixed-DNA January/August database replay, annotations, native matrices, predictions, paired E2/CCRE case audits and resampling | `database_reference_factorial_20261111/` |
| CCRE primary records, cohort selection and MIC screening | `independent_MIC_screen_20261111/` |
| CCRE354 scoring, technical missingness, cluster and retained-source audits | `strict_CCRE_followup_20261111/` |
| EuSCAPE170 reference construction, predictions, exact reassignment and additional-taxon checks | `independent_reference_extension_20261112/` |
| MIC sensitivity protocol, per-case audit, stratum metrics and country resamples | `rsos_readiness_extension_20261113/` |

`historical_support/project/` additionally preserves the original author-supplied
E2/E3 predictions, targeted ANI-neighbour records, cross-score tables and early
AST-source audits. `historical_support/SOURCE_ARCHIVE.json` identifies the
unchanged source archive. These historical E3 results concern a separately
fitted primary model; the post-hoc 668-feature cross-score is labelled explicitly.

Computational scripts are in `scripts/`. All seven current figure PDFs retain
their original names in `figures/`. Figure 2 and Figure 3 have plotting code,
notebooks and backing CSVs there. Figure 1 is the author-supplied workflow
diagram. `references.bib` provides the accompanying bibliography.

`README_REVIEWER.md` gives the more detailed archive guide.

## Recover large public source inputs

Genome assemblies are accessed through their public accession/version records
and are not duplicated here. Acquisition manifests and hashes are retained.
`RAW_METADATA_SOURCES.csv` records version-specific NCBI metadata URLs, byte
counts and SHA-256 hashes. To list these sources or restore all snapshots:

```bash
python fetch_public_raw_metadata.py
python fetch_public_raw_metadata.py --all
```

The complete metadata snapshots require approximately 3 GB of additional
downloads. The fetcher rejects a checksum mismatch. Repeating genome acquisition
and annotation additionally requires AMRFinderPlus 4.2.7 and the declared
2026-01-21.1 and 2026-08-07.1 database releases; this is separate from reproducing
the comparisons using the included fixed predictions and native matrices.
Historical acquisition receipts retain some absolute workspace paths; the
documented entrypoints use paths relative to this directory.

## Scope and interpretation

The broader taxon/drug differences are extensions of one correlated source
event, not independent temporal replications. Historical E2 AST was not
independently adjudicated. CCRE uses original PHAS categories; EuSCAPE uses the
declared EUCAST14 non-meningitis BMD-derived scenario. The MIC envelope is a
hypothetical one-twofold-dilution stress analysis. The legacy E3 model was fitted
separately and is not part of the frozen 668-feature model comparisons.

## Integrity and reuse

`FILE_MANIFEST.csv` and `SHA256SUMS.txt` cover the archived files.
`verify_integrity.py` checks file sizes and SHA-256 without third-party packages.
The project software license is in `LICENSE`; source data retain their original
attribution and source terms. No DOI or external archival deposit is claimed.
