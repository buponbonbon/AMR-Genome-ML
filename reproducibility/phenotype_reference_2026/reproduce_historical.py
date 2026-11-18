"""Recompute supplied historical E2/E3 near-neighbour and cross-score tables."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)


def metrics(frame, score):
    y = frame.y_true.to_numpy(int)
    p = frame[score].to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    return dict(n=len(y), tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp),
                auroc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                mcc=float(matthews_corrcoef(y, p >= .5)), brier=float(brier_score_loss(y, p)))


def check(row, result):
    for key, value in result.items():
        if not np.isclose(float(row[key]), value, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"Historical result differs: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).resolve().parent / "historical_support/project/results"
    near = base / "file17_frozen_neighbour_sensitivity_v1"
    flags = pd.read_csv(near / "file17_per_genome_neighbour_flags.csv")
    expected = pd.read_csv(near / "file17_descriptive_sensitivity_panels.csv")
    outputs = []
    for row in expected.to_dict("records"):
        frame = flags.loc[flags.cohort.eq(row["cohort"])]
        mask = frame.ani_confirmed_near_dev.astype(str).str.lower().isin(["true", "1"])
        if row["panel"] == "TARGETED_ANI_CONFIRMED_NEAR":
            frame = frame.loc[mask]
        elif row["panel"] == "REMAINDER_AFTER_REMOVING_CONFIRMED_NEAR_POSTHOC":
            frame = frame.loc[~mask]
        elif row["panel"] != "FROZEN_FULL_PRIMARY":
            raise AssertionError("Unknown historical panel")
        result = metrics(frame, "prob_resistance")
        check(row, result)
        outputs.append(dict(cohort=row["cohort"], panel=row["panel"], **result))
    pd.DataFrame(outputs).to_csv(args.out / "historical_neighbour_metrics.csv", index=False)
    cross = base / "file18_same_model_audit_v1_0_1"
    expected = pd.read_csv(cross / "file18_same_model_comparison.csv")
    definitions = [
        ("FILE11C_668_ORIGINAL_REPRODUCED", "file18_cross_scored_E2_reproduction.csv",
         "probability_R_reproduced_FILE11C_model"),
        ("FILE11C_668_CROSS_SCORE_POSTHOC", "file18_cross_scored_E3_posthoc.csv",
         "probability_R_FILE11C_cross_score"),
        ("FILE13_702_ORIGINAL_PRIMARY_REFERENCE", "file18_cross_scored_E3_posthoc.csv",
         "original_FILE13_702_primary_probability_R"),
    ]
    outputs = []
    for name, filename, score in definitions:
        frame = pd.read_csv(cross / filename)
        result = metrics(frame, score)
        row = expected.loc[expected.analysis.eq(name)].iloc[0]
        check(row, result)
        outputs.append(dict(cohort=row.cohort, analysis=name, **result))
    pd.DataFrame(outputs).to_csv(args.out / "historical_cross_score_metrics.csv", index=False)
    report = {"status": "PASS", "near_neighbour_rows": 6, "cross_score_rows": 3,
              "legacy_E3_uses_a_separately_fitted_primary_model": True}
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
