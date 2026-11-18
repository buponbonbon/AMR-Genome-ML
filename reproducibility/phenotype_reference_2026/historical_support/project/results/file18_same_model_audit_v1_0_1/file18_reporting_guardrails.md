# FILE18 reporting guardrails

1. The exact frozen FILE11C model is used for E2 and E3. FILE13/E3 on-disk order is matched to FILE11C via exact-name, bijective in-memory column permutation if needed; no renaming or imputation. No training or recalibration.
2. **NOT** proof of exactly the same end-to-end preprocessing or feature semantics: original E2 FILE11C annotations and E3 FILE14 harmonized calls may differ even if names match.
3. Do not call FILE18 a prospectively prespecified independent E3 validation of FILE11C.
4. Do not replace E3 primary 702-feature result, and do not compare AUROCs causally across cohorts.
5. No AST phenotype errors are adjudicated; no clinical predictions may be generated for use in care.
6. Interpret with FILE17 near-neighbour selection caveat and FILE13 original-vs-harmonized development bridge audit.
7. The SHA-locked model is loaded with joblib. Run only on your own trusted checkpoint; never deserialize untrusted model files.
