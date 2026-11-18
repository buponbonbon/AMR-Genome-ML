#!/usr/bin/env python3
"""Replay fixed-decision method metrics after a complete focal label reversal.

Exact rule point replay needs the preserved BioProject confusion-count table.
Without it, output feasible bounds from archived aggregate totals and certify
score ordering only if the ML point exceeds the whole rule envelope. Never
invent per-isolate predictions or claim recovered rule point estimates.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


FOCAL = "PRJNA1322038"
CM_KEYS = ("tp", "fn", "tn", "fp")
EXPECTED_ML_OLD = {"tp": 109, "fn": 54, "tn": 221, "fp": 141}
EXPECTED_ML_NEW = {"tp": 236, "fn": 30, "tn": 245, "fp": 14}
EXPECTED_RULE_OLD = {"tp": 138, "fn": 25, "tn": 192, "fp": 170}


def metrics(cm):
    tp, fn, tn, fp = (cm[k] for k in CM_KEYS)
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {**cm, "n": tp + fn + tn + fp, "sensitivity": sensitivity,
            "specificity": specificity, "balanced_accuracy": (sensitivity + specificity) / 2,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
            "error_n": fn + fp, "accuracy": (tp + tn) / (tp + fn + tn + fp)}


def flip(cm):
    return {"tp": cm["fp"], "fn": cm["tn"], "tn": cm["fn"], "fp": cm["tp"]}


def add(matrices):
    return {k: sum(c[k] for c in matrices) for k in CM_KEYS}


def matrix(rows, version):
    result = Counter({k: 0 for k in CM_KEYS})
    for row in rows:
        actual = row["historical_label" if version == "historical" else "scenario_label"] == "R"
        predicted = float(row["probability"]) >= 0.5
        result[("tp" if actual else "fp") if predicted else ("fn" if actual else "tn")] += 1
    return dict(result)


def binary_score_metrics(cm):
    p = cm["tp"] + cm["fn"]
    n = cm["tn"] + cm["fp"]
    called_positive = cm["tp"] + cm["fp"]
    recall = cm["tp"] / p
    precision = cm["tp"] / called_positive if called_positive else 0.0
    return {"auroc": (recall + cm["tn"] / n) / 2,
            "average_precision": recall * precision + (1 - recall) * p / (p + n)}


def continuous_score_metrics(rows, version):
    groups = {}
    for row in rows:
        score = float(row["probability"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid frozen probability")
        actual = row["historical_label" if version == "historical" else "scenario_label"] == "R"
        groups.setdefault(score, [0, 0])[int(actual)] += 1
    positives = sum(c[1] for c in groups.values())
    negatives = sum(c[0] for c in groups.values())
    rank_start, rank_sum = 1, 0.0
    for score in sorted(groups):
        negative_n, positive_n = groups[score]
        size = negative_n + positive_n
        rank_sum += positive_n * (rank_start + (size - 1) / 2)
        rank_start += size
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    tp = total = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        negative_n, positive_n = groups[score]
        tp += positive_n
        total += negative_n + positive_n
        ap += positive_n / positives * tp / total
    return {"auroc": auc, "average_precision": ap}


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-crosswalk", type=Path, required=True)
    parser.add_argument("--source-counts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.reference_crosswalk.read_bytes()
    with args.reference_crosswalk.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 525 or len({r["biosample"] for r in rows}) != 525:
        raise ValueError("Unexpected fixed E2 cohort")
    focal = [r for r in rows if r["project"] == FOCAL]
    if len(focal) != 175 or any(r["historical_label"] == r["scenario_label"] for r in focal):
        raise ValueError("Focal labels are not a complete reversal")
    if any({r["historical_label"], r["scenario_label"]} - {"R", "S"} for r in rows):
        raise ValueError("Non-binary reference; cannot use this transformation")
    if any(r["historical_label"] != r["scenario_label"] for r in rows if r["project"] != FOCAL):
        raise ValueError("Changes outside the specified focal source")
    if matrix(rows, "historical") != EXPECTED_ML_OLD or matrix(rows, "october") != EXPECTED_ML_NEW:
        raise ValueError("Frozen model counts do not reproduce archived totals")
    args.out.mkdir(parents=True, exist_ok=False)
    output = [{"method": "frozen_ML", "reference": version, **metrics(matrix(rows, version)),
               **continuous_score_metrics(rows, version)}
              for version in ["historical", "october"]]
    output.append({"method": "frozen_rule", "reference": "historical", **metrics(EXPECTED_RULE_OLD),
                   **binary_score_metrics(EXPECTED_RULE_OLD)})
    report = {"scope": "Fixed-threshold reference-dependent metrics, not clinical safety validation",
              "reference_crosswalk_sha256": hashlib.sha256(raw).hexdigest(),
              "no_training_or_new_rule_selection": True}
    if args.source_counts:
        with args.source_counts.open(encoding="utf-8-sig") as handle:
            sources = list(csv.DictReader(handle))
        if len({r["bioproject_acc"] for r in sources}) != len(sources):
            raise ValueError("Duplicate BioProject count rows")
        ml_old, rule_old, ml_new, rule_new = [], [], [], []
        for row in sources:
            project = row["bioproject_acc"]
            subset = [r for r in rows if r["project"] == project]
            if not subset:
                raise ValueError("Count table includes a source outside fixed E2")
            ml = {k: int(row["ml_" + k]) for k in CM_KEYS}
            rule = {k: int(row["rule_" + k]) for k in CM_KEYS}
            if ml != matrix(subset, "historical") or any(v < 0 for v in rule.values()):
                raise ValueError("Source-specific model counts differ from frozen predictions")
            if sum(rule.values()) != len(subset) or rule["tp"] + rule["fn"] != ml["tp"] + ml["fn"]:
                raise ValueError("Source rule/reference counts mismatch")
            ml_old.append(ml)
            rule_old.append(rule)
            ml_new.append(flip(ml) if project == FOCAL else ml)
            rule_new.append(flip(rule) if project == FOCAL else rule)
        if add(ml_old) != EXPECTED_ML_OLD or add(rule_old) != EXPECTED_RULE_OLD or add(ml_new) != EXPECTED_ML_NEW:
            raise ValueError("Count table does not reproduce the exact published methods")
        new_rule = {**metrics(add(rule_new)), **binary_score_metrics(add(rule_new))}
        output.append({"method": "frozen_rule", "reference": "october", **new_rule})
        report.update(status="EXACT_COUNT_REPLAY_COMPLETE", source_counts_sha256=hashlib.sha256(args.source_counts.read_bytes()).hexdigest())
        contrasts = []
        for version in ["historical", "october"]:
            ml = next(r for r in output if r["method"] == "frozen_ML" and r["reference"] == version)
            rule = next(r for r in output if r["method"] == "frozen_rule" and r["reference"] == version)
            contrasts.append({"reference": version, **{"rule_minus_ML_" + k: rule[k] - ml[k] for k in
                              ["sensitivity", "specificity", "balanced_accuracy", "mcc", "error_n", "auroc", "average_precision"]}})
        write_csv(args.out / "exact_method_contrasts.csv", contrasts)
        report["contrasts"] = contrasts
    else:
        feasible = {}
        focal_r = sum(r["historical_label"] == "R" for r in focal)
        focal_s = len(focal) - focal_r
        for tp_focal in range(focal_r + 1):
            for fp_focal in range(focal_s + 1):
                if not 0 <= EXPECTED_RULE_OLD["tp"] - tp_focal <= 163 - focal_r:
                    continue
                if not 0 <= EXPECTED_RULE_OLD["fp"] - fp_focal <= 362 - focal_s:
                    continue
                tp = EXPECTED_RULE_OLD["tp"] - tp_focal + fp_focal
                cm = {"tp": tp, "fn": 266 - tp, "fp": 308 - tp, "tn": 259 - (308 - tp)}
                if min(cm.values()) < 0:
                    raise ValueError("Invalid feasible transformation")
                feasible[tuple(cm[k] for k in CM_KEYS)] = {**metrics(cm), **binary_score_metrics(cm)}
        write_csv(args.out / "feasible_October_rule_counts_NOT_actual_replay.csv", list(feasible.values()))
        bounds = {k: {"min": min(r[k] for r in feasible.values()), "max": max(r[k] for r in feasible.values())}
                  for k in ["sensitivity", "specificity", "balanced_accuracy", "mcc", "error_n", "auroc", "average_precision"]}
        old_ml, new_ml, old_rule = output[0], output[1], output[2]
        certificate = {
            "scope": "Ordering of continuous frozen ML scores versus the documented fixed binary rule score (0/1), under each common reference",
            "historical_ML_auroc": old_ml["auroc"], "historical_binary_rule_auroc": old_rule["auroc"],
            "october_ML_auroc": new_ml["auroc"], "october_binary_rule_auroc_upper_bound": bounds["auroc"]["max"],
            "historical_ML_AP": old_ml["average_precision"], "historical_binary_rule_AP": old_rule["average_precision"],
            "october_ML_AP": new_ml["average_precision"], "october_binary_rule_AP_upper_bound": bounds["average_precision"]["max"],
            "AUROC_order_reversal_proven": old_ml["auroc"] < old_rule["auroc"] and new_ml["auroc"] > bounds["auroc"]["max"],
            "AP_order_reversal_proven": old_ml["average_precision"] < old_rule["average_precision"] and new_ml["average_precision"] > bounds["average_precision"]["max"],
            "bound_derivation": "Rule has 308 fixed positive calls. October reference has 266 R and 259 S, so TP<=266 and FP>=42. Binary-score AUROC=(TPR+TNR)/2 <=(1+217/259)/2=0.9189189189. The complete feasible count enumeration also bounds binary-score AP at 266/308=0.8636363636.",
            "limits": "New rule point metrics and FN/FP are not recovered. Hard-decision BA/MCC/safety ordering is not certified. AP depends on reference prevalence. This is an established score-format identity applied to the documented comparator, not a new algorithm or clinical superiority claim."}
        (args.out / "AUROC_AP_order_certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
        report.update(status="NEW_RULE_POINTS_MISSING_AUROC_AP_ORDER_CERTIFIED_BY_MARGIN_BOUNDS", feasible_aggregate_matrices=len(feasible),
                      feasible_metric_bounds=bounds,
                      bounds_interpretation="Ranges over unknown rule calls consistent with archived margins only; not confidence intervals, actual replay or a new algorithm.",
                      minimum_missing_input="results/file16i_evidence_lock_v1/file16i_E2_paired_by_bioproject.csv (4030 bytes in preserved inventory)",
                      ordering_certificate=certificate)
    write_csv(args.out / "available_exact_method_metrics.csv", output)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
