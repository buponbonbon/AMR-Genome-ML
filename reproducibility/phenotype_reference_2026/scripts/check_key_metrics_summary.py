#!/usr/bin/env python3
"""
CHECK_KEY_METRICS_SUMMARY.py
============================

Independent verifier for:
    AMR_Genome_ML_key_metrics_summary.csv

What it does
------------
1) Loads the canonical project outputs (FILE09, FILE11C, FILE11D, FILE11E, FILE11F).
2) Recomputes the frozen FILE11C primary metrics directly from
   file11c_blind_predictions.csv.
3) Verifies frozen prediction SHA256 against FILE11C final summary.
4) Reconstructs the expected key-metric rows from the canonical source files.
5) Compares those expected values to the downloaded summary CSV.
6) Writes a row-by-row audit report and exits nonzero if any checked metric differs
   beyond tolerance or is missing.

This script does NOT trust the summary CSV as a source of truth.

Typical run
-----------
python scripts/check_key_metrics_summary.py \
  --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML \
  --summary-csv /mnt/c/Users/ASUS/Downloads/AMR_Genome_ML_key_metrics_summary.csv

If the CSV is copied into the project root:
python scripts/check_key_metrics_summary.py \
  --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML \
  --summary-csv /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML/AMR_Genome_ML_key_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

VERSION = "1.0.0"

KEY_COLS = [
    "section",
    "analysis",
    "scope",
    "cohort_or_model",
    "metric",
]


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def safe_num(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if np.isfinite(v) else None
    s = str(x).strip()
    if s == "":
        return None
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def metric_bundle(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=np.uint8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y)),
        "R": int(y.sum()),
        "S": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def add_expected(
    rows: list[dict[str, Any]],
    section: str,
    analysis: str,
    scope: str,
    cohort_or_model: str,
    metric: str,
    value: Any,
    ci_low: Any = None,
    ci_high: Any = None,
    source: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "analysis": analysis,
            "scope": scope,
            "cohort_or_model": cohort_or_model,
            "metric": metric,
            "expected_value": value,
            "expected_ci95_low": ci_low,
            "expected_ci95_high": ci_high,
            "canonical_source": source,
        }
    )


def build_expected(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # FILE11C primary source
    # -------------------------------------------------------------------------
    p11c_sum = root / "checkpoints/external_validation_file11c/file11c_final_summary.json"
    p11c_pred = root / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_blind_predictions.csv"
    p11c_boot = root / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_cluster_bootstrap_ci.csv"
    p11c_mlst = root / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_mlst_subgroup_metrics.csv"

    for p, label in [
        (p11c_sum, "FILE11C final summary"),
        (p11c_pred, "FILE11C predictions"),
        (p11c_boot, "FILE11C cluster bootstrap"),
        (p11c_mlst, "FILE11C MLST subgroup metrics"),
    ]:
        require(p, label)

    s11c = read_json(p11c_sum)
    pred = pd.read_csv(p11c_pred)
    boot11c = pd.read_csv(p11c_boot)
    mlst11c = pd.read_csv(p11c_mlst)

    observed_pred_sha = sha256_file(p11c_pred)
    expected_pred_sha = s11c["frozen_prediction_sha256"]
    checks["FILE11C_prediction_sha256_match"] = observed_pred_sha == expected_pred_sha

    y = pred["label_binary"].to_numpy(dtype=np.uint8)
    p = pred["probability_R"].to_numpy(dtype=float)
    hard = pred["predicted_R"].to_numpy(dtype=np.uint8)

    checks["FILE11C_threshold_0.5_matches_calls"] = bool(
        np.array_equal((p >= 0.5).astype(np.uint8), hard)
    )

    recomputed = metric_bundle(y, p, hard)
    metric_deltas = {}
    for k, v in recomputed.items():
        if k in s11c["metrics"]:
            metric_deltas[k] = abs(float(v) - float(s11c["metrics"][k]))
    checks["FILE11C_recomputed_metric_max_abs_delta"] = max(metric_deltas.values())
    checks["FILE11C_recomputed_metrics_exact_1e-12"] = bool(
        max(metric_deltas.values()) <= 1e-12
    )

    dev = s11c["development"]
    e2 = s11c["external_meropenem_E2"]

    add_expected(rows, "Cohort", "FILE11C", "primary", "Development", "n", dev["n"],
                 source=str(p11c_sum.relative_to(root)))
    add_expected(rows, "Cohort", "FILE11C", "primary", "Development", "R", dev["R"],
                 source=str(p11c_sum.relative_to(root)))
    add_expected(rows, "Cohort", "FILE11C", "primary", "Development", "S", dev["S"],
                 source=str(p11c_sum.relative_to(root)))

    add_expected(rows, "Cohort", "FILE11C", "primary", "External E2 meropenem", "n", e2["n"],
                 source=str(p11c_sum.relative_to(root)))
    add_expected(rows, "Cohort", "FILE11C", "primary", "External E2 meropenem", "R", e2["R"],
                 source=str(p11c_sum.relative_to(root)))
    add_expected(rows, "Cohort", "FILE11C", "primary", "External E2 meropenem", "S", e2["S"],
                 source=str(p11c_sum.relative_to(root)))
    add_expected(rows, "Cohort", "FILE11C", "primary", "External E2 meropenem",
                 "unique_ERD_SNP_groups", e2["unique_erd_groups"],
                 source=str(p11c_sum.relative_to(root)))

    bootmap = {r.metric: r for r in boot11c.itertuples(index=False)}
    for metric in [
        "roc_auc", "average_precision", "accuracy", "balanced_accuracy",
        "mcc", "sensitivity", "specificity", "f1", "brier", "log_loss",
    ]:
        b = bootmap.get(metric)
        add_expected(
            rows,
            "Blind external validation",
            "FILE11C",
            "primary",
            "Frozen ML — E2",
            metric,
            recomputed[metric],
            getattr(b, "ci95_low", None) if b is not None else None,
            getattr(b, "ci95_high", None) if b is not None else None,
            source=f"{p11c_pred.relative_to(root)} / {p11c_boot.relative_to(root)}",
        )

    for metric in ["tp", "tn", "fp", "fn"]:
        add_expected(
            rows,
            "Blind external validation",
            "FILE11C",
            "primary",
            "Frozen ML — E2",
            metric.upper(),
            recomputed[metric],
            source=str(p11c_pred.relative_to(root)),
        )

    # -------------------------------------------------------------------------
    # FILE09 internal reference
    # -------------------------------------------------------------------------
    p09 = root / "data/evaluation/file09_performance_summary.csv"
    require(p09, "FILE09 performance summary")
    perf09 = pd.read_csv(p09)

    scheme_map = {
        "random_stratified": "Internal random stratified CV",
        "genomic_cluster_aware": "Internal genomic-cluster-aware CV",
        "mlst_aware": "Internal MLST-aware CV",
    }
    metric_cols = [
        ("roc_auc", "roc_auc_mean", "roc_auc_cluster_boot_ci95_low", "roc_auc_cluster_boot_ci95_high"),
        ("average_precision", "average_precision_mean", "average_precision_cluster_boot_ci95_low", "average_precision_cluster_boot_ci95_high"),
        ("balanced_accuracy", "balanced_accuracy_mean", "balanced_accuracy_cluster_boot_ci95_low", "balanced_accuracy_cluster_boot_ci95_high"),
        ("mcc", "mcc_mean", "mcc_cluster_boot_ci95_low", "mcc_cluster_boot_ci95_high"),
        ("brier", "brier_mean", None, None),
        ("log_loss", "log_loss_mean", None, None),
    ]

    sub09 = perf09[
        perf09["representation"].astype(str).eq("known_amr")
        & perf09["model"].astype(str).eq("logreg_l2")
        & perf09["scheme"].isin(scheme_map)
    ]
    if len(sub09) != 3:
        raise RuntimeError(f"Expected 3 FILE09 known_amr/logreg_l2 rows; found {len(sub09)}")

    for _, r in sub09.iterrows():
        for metric, vc, lc, hc in metric_cols:
            add_expected(
                rows,
                "Internal validation",
                "FILE09",
                "internal_reference",
                scheme_map[r["scheme"]],
                metric,
                r[vc],
                r[lc] if lc else None,
                r[hc] if hc else None,
                source=str(p09.relative_to(root)),
            )

    # -------------------------------------------------------------------------
    # FILE11D diagnostics
    # -------------------------------------------------------------------------
    p11d_sum = root / "checkpoints/external_validation_file11d/file11d_final_summary.json"
    p11d_lin = root / "data/external_validation/ncbi_pathogen_detection/file11d/file11d_lineage_cluster_bootstrap.csv"
    p11d_shift = root / "data/external_validation/ncbi_pathogen_detection/file11d/file11d_coefficient_weighted_domain_shift.csv"

    for pth, label in [
        (p11d_sum, "FILE11D final summary"),
        (p11d_lin, "FILE11D lineage bootstrap"),
        (p11d_shift, "FILE11D coefficient-weighted shift"),
    ]:
        require(pth, label)

    s11d = read_json(p11d_sum)
    lin11d = pd.read_csv(p11d_lin)
    shift11d = pd.read_csv(p11d_shift)

    cal = s11d["calibration"]
    for metric in [
        "calibration_slope",
        "calibration_intercept",
        "mean_predicted_probability_R",
        "observed_R_prevalence",
    ]:
        add_expected(
            rows,
            "Calibration",
            "FILE11D",
            "posthoc_diagnostic",
            "Frozen ML — E2",
            metric,
            cal[metric],
            source=str(p11d_sum.relative_to(root)),
        )

    add_expected(
        rows, "Calibration", "FILE11D", "posthoc_diagnostic", "Frozen ML — E2",
        "total_errors", s11d["error_concentration"]["total_errors"],
        source=str(p11d_sum.relative_to(root))
    )
    add_expected(
        rows, "Calibration", "FILE11D", "posthoc_diagnostic", "Frozen ML — E2",
        "high_confidence_errors_ge_0.90", cal["high_confidence_errors_ge_0_90"],
        source=str(p11d_sum.relative_to(root))
    )

    # MLST subgroup point metrics from FILE11C.
    for _, r in mlst11c.iterrows():
        name = str(r["lineage_novelty"])
        for metric in ["n", "R", "S", "roc_auc", "average_precision", "balanced_accuracy", "mcc", "brier"]:
            add_expected(
                rows,
                "Lineage robustness",
                "FILE11C/11D",
                "posthoc_diagnostic",
                name,
                metric,
                r[metric],
                source=str(p11c_mlst.relative_to(root)),
            )

    for _, r in lin11d.iterrows():
        if r["metric"] in {"roc_auc", "balanced_accuracy", "mcc"}:
            add_expected(
                rows,
                "Lineage robustness",
                "FILE11D",
                "posthoc_diagnostic",
                "seen_ST minus unseen_ST",
                f"delta_{r['metric']}",
                r["point_delta"],
                r["ci95_low"],
                r["ci95_high"],
                source=str(p11d_lin.relative_to(root)),
            )

    abs_shift = shift11d["absolute_prevalence_shift"].astype(float)
    for threshold in (0.05, 0.10, 0.15):
        add_expected(
            rows,
            "Domain shift",
            "FILE11D",
            "posthoc_diagnostic",
            "668 frozen features",
            f"features_abs_prevalence_shift_gt_{threshold:.2f}",
            int((abs_shift > threshold).sum()),
            source=str(p11d_shift.relative_to(root)),
        )

    add_expected(
        rows, "Domain shift", "FILE11C/11D", "posthoc_diagnostic", "External E2",
        "out_of_schema_determinants",
        s11c["external_feature_matrix"]["out_of_schema_determinants"],
        source=str(p11c_sum.relative_to(root)),
    )
    add_expected(
        rows, "Domain shift", "FILE11C", "primary", "External E2",
        "zero_feature_genomes",
        s11c["external_feature_matrix"]["zero_feature_genomes"],
        source=str(p11c_sum.relative_to(root)),
    )

    # -------------------------------------------------------------------------
    # FILE11E comparators
    # -------------------------------------------------------------------------
    p11e_sum = root / "checkpoints/external_validation_file11e/file11e_final_summary.json"
    p11e_con = root / "data/external_validation/ncbi_pathogen_detection/file11e/file11e_paired_cluster_bootstrap_contrasts.csv"
    require(p11e_sum, "FILE11E final summary")
    require(p11e_con, "FILE11E paired contrasts")

    s11e = read_json(p11e_sum)
    con11e = pd.read_csv(p11e_con)

    pretty = {
        "frozen_ml": "Frozen ML",
        "development_prevalence_null": "Development-prevalence null",
        "kpc_family_rule": "KPC-family rule",
        "amrfinder_carbapenemase_rule": "AMRFinder broad carbapenemase rule",
    }

    for method, vals in s11e["method_metrics"].items():
        for metric in [
            "roc_auc", "average_precision", "balanced_accuracy", "mcc",
            "sensitivity", "specificity", "f1", "accuracy", "brier", "log_loss",
        ]:
            if vals.get(metric) is None:
                continue
            add_expected(
                rows,
                "Simple comparator",
                "FILE11E",
                "posthoc_comparator",
                pretty[method],
                metric,
                vals[metric],
                source=str(p11e_sum.relative_to(root)),
            )

    q = con11e[
        con11e["comparator"].astype(str).eq("amrfinder_carbapenemase_rule")
        & con11e["metric"].isin(
            ["roc_auc", "average_precision", "balanced_accuracy", "mcc",
             "sensitivity", "specificity", "f1"]
        )
    ]
    for _, r in q.iterrows():
        add_expected(
            rows,
            "Simple comparator",
            "FILE11E",
            "posthoc_comparator",
            "Frozen ML minus AMRFinder broad carbapenemase rule",
            f"delta_{r['metric']}",
            r["bootstrap_delta_mean"],
            r["ci95_low"],
            r["ci95_high"],
            source=str(p11e_con.relative_to(root)),
        )

    # -------------------------------------------------------------------------
    # FILE11F source-shift sensitivity
    # -------------------------------------------------------------------------
    p11f_sum = root / "checkpoints/external_validation_file11f/file11f_final_summary.json"
    p11f_src_cv = root / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_source_classifier_cv.csv"
    p11f_int = root / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_internal_cv_sensitivity.csv"
    p11f_ext = root / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_external_sensitivity_metrics.csv"

    for pth, label in [
        (p11f_sum, "FILE11F final summary"),
        (p11f_src_cv, "FILE11F source classifier CV"),
        (p11f_int, "FILE11F internal CV sensitivity"),
        (p11f_ext, "FILE11F external sensitivity metrics"),
    ]:
        require(pth, label)

    s11f = read_json(p11f_sum)
    src_cv = pd.read_csv(p11f_src_cv)
    int11f = pd.read_csv(p11f_int)
    ext11f = pd.read_csv(p11f_ext)

    src_random = src_cv[src_cv["scheme"].astype(str).eq("stratified_random_5fold")]
    if len(src_random) != 1:
        raise RuntimeError("Cannot identify unique FILE11F random source-CV row.")

    add_expected(
        rows,
        "Feature-generation source shift",
        "FILE11F",
        "posthoc_sensitivity",
        "Source classifier",
        "source_classifier_CV_AUROC",
        float(src_random.iloc[0]["roc_auc"]),
        source=str(p11f_src_cv.relative_to(root)),
    )

    add_expected(
        rows,
        "Feature-generation source shift",
        "FILE11F",
        "posthoc_sensitivity",
        "Source markers",
        "primary_source_markers",
        s11f["source_signature"]["primary_source_marker_count"],
        source=str(p11f_sum.relative_to(root)),
    )
    add_expected(
        rows,
        "Feature-generation source shift",
        "FILE11F",
        "posthoc_sensitivity",
        "Source markers",
        "source_markers_in_top_abs_coef_50",
        s11f["source_signature"]["source_markers_in_top50_abs_resistance_coefficients"],
        source=str(p11f_sum.relative_to(root)),
    )

    name_map_int = {
        "source_debiased_full": "Source-debiased full",
        "local_source_only": "Local-source-only",
        "ncbi_source_only": "NCBI-source-only",
    }
    for _, r in int11f.iterrows():
        if r["model"] not in name_map_int:
            continue
        pretty_name = name_map_int[r["model"]]
        for metric, outmetric in [
            ("roc_auc", "internal_CV_AUROC"),
            ("balanced_accuracy", "internal_CV_balanced_accuracy"),
        ]:
            add_expected(
                rows,
                "Feature-generation source shift",
                "FILE11F",
                "posthoc_sensitivity",
                pretty_name,
                outmetric,
                r[metric],
                source=str(p11f_int.relative_to(root)),
            )

    name_map_ext = {
        "source_debiased_full": "Source-debiased full",
        "local_source_only": "Local-source-only",
        "ncbi_source_only": "NCBI-source-only",
    }
    for _, r in ext11f.iterrows():
        if r["model"] not in name_map_ext:
            continue
        pretty_name = name_map_ext[r["model"]]
        for metric, outmetric in [
            ("roc_auc", "external_E2_AUROC"),
            ("balanced_accuracy", "external_E2_balanced_accuracy"),
            ("mcc", "external_E2_MCC"),
            ("brier", "external_E2_Brier"),
        ]:
            add_expected(
                rows,
                "Feature-generation source shift",
                "FILE11F",
                "posthoc_sensitivity",
                pretty_name,
                outmetric,
                r[metric],
                source=str(p11f_ext.relative_to(root)),
            )

    add_expected(
        rows,
        "Feature-generation source shift",
        "FILE11F",
        "posthoc_sensitivity",
        "Compute triage",
        "decision",
        s11f["compute_triage"]["decision"],
        source=str(p11f_sum.relative_to(root)),
    )

    expected = pd.DataFrame(rows)
    return expected, checks


def compare(summary: pd.DataFrame, expected: pd.DataFrame, atol: float, rtol: float) -> pd.DataFrame:
    # Normalize key columns.
    for c in KEY_COLS:
        summary[c] = summary[c].astype(str).str.strip()
        expected[c] = expected[c].astype(str).str.strip()

    key_dups = summary.duplicated(KEY_COLS, keep=False)
    if key_dups.any():
        d = summary.loc[key_dups, KEY_COLS].sort_values(KEY_COLS)
        raise RuntimeError(
            "Summary CSV contains duplicate metric keys. Examples:\n"
            + d.head(20).to_string(index=False)
        )

    sidx = summary.set_index(KEY_COLS, drop=False)

    audit_rows = []
    for _, er in expected.iterrows():
        key = tuple(er[c] for c in KEY_COLS)

        if key not in sidx.index:
            audit_rows.append(
                {
                    **{c: er[c] for c in KEY_COLS},
                    "status": "MISSING_FROM_SUMMARY",
                    "summary_value": None,
                    "expected_value": er["expected_value"],
                    "abs_diff": None,
                    "summary_ci95_low": None,
                    "expected_ci95_low": er["expected_ci95_low"],
                    "summary_ci95_high": None,
                    "expected_ci95_high": er["expected_ci95_high"],
                    "canonical_source": er["canonical_source"],
                }
            )
            continue

        sr = sidx.loc[key]
        if isinstance(sr, pd.DataFrame):
            raise RuntimeError(f"Duplicate summary key unexpectedly survived: {key}")

        ev = er["expected_value"]
        sv_raw = sr["value"]

        ev_num = safe_num(ev)
        sv_num = safe_num(sv_raw)

        if ev_num is not None and sv_num is not None:
            diff = abs(sv_num - ev_num)
            ok_value = math.isclose(sv_num, ev_num, abs_tol=atol, rel_tol=rtol)
        else:
            diff = None
            ok_value = str(sv_raw).strip() == str(ev).strip()

        # CI comparison only when canonical expected CI exists.
        ci_ok = True
        ci_notes = []
        for field_summary, field_expected in [
            ("ci95_low", "expected_ci95_low"),
            ("ci95_high", "expected_ci95_high"),
        ]:
            expected_ci = er[field_expected]
            expected_num = safe_num(expected_ci)
            if expected_num is None:
                continue
            summary_num = safe_num(sr.get(field_summary, None))
            if summary_num is None:
                ci_ok = False
                ci_notes.append(f"{field_summary}:missing")
            elif not math.isclose(summary_num, expected_num, abs_tol=atol, rel_tol=rtol):
                ci_ok = False
                ci_notes.append(
                    f"{field_summary}:summary={summary_num:.17g},expected={expected_num:.17g}"
                )

        status = "PASS" if ok_value and ci_ok else "FAIL"
        if not ok_value:
            status = "FAIL_VALUE"
        elif not ci_ok:
            status = "FAIL_CI"

        audit_rows.append(
            {
                **{c: er[c] for c in KEY_COLS},
                "status": status,
                "summary_value": sv_raw,
                "expected_value": ev,
                "abs_diff": diff,
                "summary_ci95_low": sr.get("ci95_low", None),
                "expected_ci95_low": er["expected_ci95_low"],
                "summary_ci95_high": sr.get("ci95_high", None),
                "expected_ci95_high": er["expected_ci95_high"],
                "ci_notes": ";".join(ci_notes),
                "canonical_source": er["canonical_source"],
            }
        )

    return pd.DataFrame(audit_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--summary-csv", required=True, type=Path)
    ap.add_argument("--atol", type=float, default=5e-4,
                    help="Absolute tolerance; default 5e-4 to allow 4-decimal rounded entries.")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--strict", action="store_true",
                    help="Use 1e-12 absolute tolerance instead of rounded-value tolerance.")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    root = args.project_root.resolve()
    summary_path = args.summary_csv.resolve()
    require(summary_path, "summary CSV")

    atol = 1e-12 if args.strict else args.atol
    report = (
        args.report.resolve()
        if args.report is not None
        else root / "checkpoints/key_metrics_verification/key_metrics_verification_report.csv"
    )
    report.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("KEY METRICS SUMMARY VERIFIER")
    print("=" * 110)
    print("Version      :", VERSION)
    print("Project root :", root)
    print("Summary CSV  :", summary_path)
    print("Tolerance    :", atol)
    print()

    summary = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
    required_cols = set(KEY_COLS + ["value", "ci95_low", "ci95_high"])
    missing_cols = sorted(required_cols - set(summary.columns))
    if missing_cols:
        raise RuntimeError(f"Summary CSV missing required columns: {missing_cols}")

    expected, checks = build_expected(root)
    audit = compare(summary, expected, atol=atol, rtol=args.rtol)
    audit.to_csv(report, index=False)

    print("Upstream integrity checks")
    print("-------------------------")
    for k, v in checks.items():
        print(f"{k}: {v}")

    status_counts = audit["status"].value_counts().to_dict()
    print()
    print("Metric audit")
    print("------------")
    print("expected metric rows :", len(expected))
    print("summary metric rows  :", len(summary))
    print("status counts        :", status_counts)
    print("report               :", report)

    failed = audit[~audit["status"].eq("PASS")].copy()
    if len(failed):
        print()
        print("FAILED / MISSING METRICS")
        print("------------------------")
        show_cols = KEY_COLS + [
            "status",
            "summary_value",
            "expected_value",
            "abs_diff",
            "summary_ci95_low",
            "expected_ci95_low",
            "summary_ci95_high",
            "expected_ci95_high",
            "canonical_source",
        ]
        with pd.option_context("display.max_rows", 100, "display.max_colwidth", 80):
            print(failed[show_cols].to_string(index=False))

    integrity_ok = all(
        [
            checks["FILE11C_prediction_sha256_match"],
            checks["FILE11C_threshold_0.5_matches_calls"],
            checks["FILE11C_recomputed_metrics_exact_1e-12"],
        ]
    )

    print()
    if len(failed) == 0 and integrity_ok:
        print("FINAL STATUS: PASS_ALL_CHECKED_METRICS")
        return 0

    print("FINAL STATUS: FAIL — inspect the report above")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
