#!/usr/bin/env python3
"""
FILE12 — MANUSCRIPT-READY FIGURES, TABLES, CAPTIONS, AND ASSET MANIFEST
======================================================================

Purpose
-------
Generate the publication/reporting layer for the AMR-Genome-ML project from
already-frozen canonical outputs. FILE12 performs NO model fitting, NO feature
selection, NO threshold tuning, and NO post-hoc optimization.

The script is deliberately reporting-only.

Required upstream state
-----------------------
- FILE09 performance summary available.
- FILE11C PASS_BLIND_EXTERNAL_VALIDATION.
- FILE11D PASS_POSTHOC_DIAGNOSTIC.
- FILE11E PASS_POSTHOC_COMPARATOR.
- FILE11F PASS_POSTHOC_SOURCE_SHIFT_DIAGNOSTIC.
- AMR_Genome_ML_key_metrics_summary_verified.csv exists at project root.
- key_metrics_verification_report.csv contains 142/142 PASS rows.

Primary figures
---------------
Figure 1. Study design and frozen external-validation workflow.
Figure 2. Internal-to-external performance transportability.
Figure 3. Blind external ROC/PR/calibration/probability behavior.
Figure 4. Frozen ML versus simple genotype comparators.
Figure 5. Feature/domain shift and model-weighted genomic changes.
Figure 6. Lineage and source-generation robustness diagnostics.

Supplementary figures
---------------------
Figure S1. Largest absolute coefficients of the frozen external model.
Figure S2. Prediction confidence by TP/TN/FP/FN category.
Figure S3. ERD/SNP-cluster error concentration.
Figure S4. Most prevalent out-of-schema external determinants.
Figure S5. Strongest feature-generation source markers.

Primary tables
--------------
Table 1. Cohort and locked analysis design.
Table 2. Internal validation performance for known-AMR logistic regression.
Table 3. Blind external E2 performance with cluster-bootstrap uncertainty.
Table 4. External comparator performance and paired cluster-bootstrap contrasts.
Table 5. Transportability and robustness diagnostics.

Supplementary tables
--------------------
Copies/curated exports of the verified metrics, model coefficients, feature
shifts, lineage metrics, out-of-schema determinants, error clusters,
source-feature associations, source sensitivities, and analysis guardrails.

Output formats
--------------
Figures: PDF (vector), SVG (vector), PNG (600 dpi).
Tables: CSV (exact numeric values), Markdown, LaTeX.
Backing-data CSV is written for every main figure.

Output root
-----------
results/file12_manuscript_assets/

Typical run
-----------
python scripts/file12_manuscript_assets.py \
  --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML

Scientific guardrails
---------------------
- FILE11C remains the primary external result.
- FILE11D/F/E are explicitly labeled post-hoc where appropriate.
- No metric is selected because it looks favorable.
- Figures do not imply that source shift or lineage novelty causally explains
  external performance loss.
- Comparator figures distinguish threshold-level performance from rank
  discrimination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import textwrap
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)


VERSION = "1.0.2"

# Journal-sized defaults in inches.
DOUBLE_COL_W = 7.15
SINGLE_COL_W = 3.45

# Use Matplotlib defaults for colors; do not hard-code a palette.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 9.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require_file(path, path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt3(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.3f}"


def fmt4(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4f}"


def pct(x: Any, digits: int = 1) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{100*v:.{digits}f}%"



def as_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kwargs in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ]:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, **kwargs)
        paths.append(p)
    plt.close(fig)
    return paths


def latex_escape(s: Any) -> str:
    s = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s


def write_table_bundle(df: pd.DataFrame, stem: Path, caption: str = "") -> list[Path]:
    """
    Write exact CSV plus readable Markdown and LaTeX versions.
    The CSV is the numerical source-of-truth for the table.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    tex_path = stem.with_suffix(".tex")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md = []
    if caption:
        md.append(f"**{caption}**\n")
    md.append(df.to_markdown(index=False))
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Minimal journal-friendly LaTeX tabular.
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    tex_lines = []
    if caption:
        tex_lines.append(f"% {latex_escape(caption)}")
    tex_lines.append(r"\begin{tabular}{" + align + "}")
    tex_lines.append(r"\hline")
    tex_lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    tex_lines.append(r"\hline")
    for row in df.itertuples(index=False):
        tex_lines.append(" & ".join(latex_escape(v) for v in row) + r" \\")
    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    return [csv_path, md_path, tex_path]


def copy_if_exists(src: Path, dst_dir: Path, dst_name: str | None = None) -> Path | None:
    if not src.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (dst_name if dst_name is not None else src.name)
    shutil.copy2(src, dst)
    return dst


def metric_row(master: pd.DataFrame, **filters) -> pd.Series:
    q = master.copy()
    for col, value in filters.items():
        q = q[q[col].astype(str).eq(str(value))]
    if len(q) != 1:
        raise RuntimeError(
            f"Expected exactly one master-metric row for {filters}; found {len(q)}"
        )
    return q.iloc[0]


def verify_master(master: pd.DataFrame, verify_report: pd.DataFrame) -> None:
    if len(master) != 142:
        raise RuntimeError(f"Verified master metrics must contain 142 rows; found {len(master)}")
    if len(verify_report) != 142:
        raise RuntimeError(
            f"Verification report must contain 142 rows; found {len(verify_report)}"
        )
    if not verify_report["status"].astype(str).eq("PASS").all():
        bad = verify_report.loc[
            ~verify_report["status"].astype(str).eq("PASS"),
            ["section", "analysis", "cohort_or_model", "metric", "status"],
        ]
        raise RuntimeError(
            "Key-metrics verification report is not fully PASS:\n"
            + bad.head(30).to_string(index=False)
        )


def ci_string(point: float, low: Any, high: Any, digits: int = 3) -> str:
    if pd.isna(low) or pd.isna(high) or str(low).strip() == "" or str(high).strip() == "":
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} ({float(low):.{digits}f}–{float(high):.{digits}f})"


# =============================================================================
# Input registry
# =============================================================================

def input_paths(root: Path) -> dict[str, Path]:
    return {
        "master": root / "AMR_Genome_ML_key_metrics_summary_verified.csv",
        "verification": root
        / "checkpoints/key_metrics_verification/key_metrics_verification_report.csv",

        "file09": root / "data/evaluation/file09_performance_summary.csv",

        "file11c_summary": root
        / "checkpoints/external_validation_file11c/file11c_final_summary.json",
        "file11c_predictions": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_blind_predictions.csv",
        "file11c_bootstrap": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_cluster_bootstrap_ci.csv",
        "file11c_mlst": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_mlst_subgroup_metrics.csv",
        "file11c_coefficients": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_model_coefficients.csv",
        "file11c_unknown": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_external_out_of_schema_determinants.csv",
        "file11c_design": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_analysis_design.json",

        "file11d_summary": root
        / "checkpoints/external_validation_file11d/file11d_final_summary.json",
        "file11d_calbins": root
        / "data/external_validation/ncbi_pathogen_detection/file11d/file11d_calibration_bins.csv",
        "file11d_lineage_boot": root
        / "data/external_validation/ncbi_pathogen_detection/file11d/file11d_lineage_cluster_bootstrap.csv",
        "file11d_shift": root
        / "data/external_validation/ncbi_pathogen_detection/file11d/file11d_coefficient_weighted_domain_shift.csv",
        "file11d_cluster": root
        / "data/external_validation/ncbi_pathogen_detection/file11d/file11d_erd_cluster_error_profile.csv",
        "file11d_source_errors": root
        / "data/external_validation/ncbi_pathogen_detection/file11d/file11d_known_feature_error_associations.csv",
        "file11d_unknown_errors": root
        / "data/external_validation/ncbi_pathogen_detection/file11d/file11d_out_of_schema_error_associations.csv",

        "file11e_summary": root
        / "checkpoints/external_validation_file11e/file11e_final_summary.json",
        "file11e_metrics": root
        / "data/external_validation/ncbi_pathogen_detection/file11e/file11e_method_metrics.csv",
        "file11e_bootstrap": root
        / "data/external_validation/ncbi_pathogen_detection/file11e/file11e_cluster_bootstrap_ci.csv",
        "file11e_contrasts": root
        / "data/external_validation/ncbi_pathogen_detection/file11e/file11e_paired_cluster_bootstrap_contrasts.csv",
        "file11e_lineage": root
        / "data/external_validation/ncbi_pathogen_detection/file11e/file11e_lineage_method_metrics.csv",
        "file11e_overlap": root
        / "data/external_validation/ncbi_pathogen_detection/file11e/file11e_error_overlap.csv",
        "file11e_discordant": root
        / "data/external_validation/ncbi_pathogen_detection/file11e/file11e_discordant_cases.csv",

        "file11f_summary": root
        / "checkpoints/external_validation_file11f/file11f_final_summary.json",
        "file11f_source_features": root
        / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_source_feature_associations.csv",
        "file11f_source_cv": root
        / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_source_classifier_cv.csv",
        "file11f_internal": root
        / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_internal_cv_sensitivity.csv",
        "file11f_external": root
        / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_external_sensitivity_metrics.csv",
        "file11f_bootstrap": root
        / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_external_paired_cluster_bootstrap.csv",
        "file11f_source_outcome": root
        / "data/external_validation/ncbi_pathogen_detection/file11f/file11f_source_outcome_confounding.csv",
    }


# =============================================================================
# Figure 1 — Study design
# =============================================================================

def figure1_study_design(s11c: dict, s11d: dict, s11e: dict, s11f: dict, out_stem: Path):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 4.4))
    ax.set_axis_off()

    # Coordinates in axes fraction.
    boxes = [
        (0.03, 0.62, 0.23, 0.25, "Development cohort",
         f"n={s11c['development']['n']:,}\n"
         f"R={s11c['development']['R']:,}; S={s11c['development']['S']:,}\n"
         f"Known-AMR: {s11c['model']['feature_count']} features"),
        (0.31, 0.62, 0.23, 0.25, "Internal validation",
         "Random stratified\nGenomic-cluster-aware\nMLST-aware\n5-fold outer validation"),
        (0.59, 0.62, 0.18, 0.25, "Frozen model",
         "Logistic regression\nC=1.0\nThreshold=0.5\nNo E2 tuning"),
        (0.81, 0.62, 0.16, 0.25, "Blind E2",
         f"n={s11c['external_meropenem_E2']['n']}\n"
         f"R={s11c['external_meropenem_E2']['R']}; S={s11c['external_meropenem_E2']['S']}\n"
         f"{s11c['external_meropenem_E2']['unique_erd_groups']} ERD/SNP groups"),
        (0.15, 0.15, 0.22, 0.25, "Post-hoc diagnostics",
         f"Calibration\nLineage robustness\nDomain shift\n{s11c['external_feature_matrix']['out_of_schema_determinants']} out-of-schema determinants"),
        (0.43, 0.15, 0.22, 0.25, "Simple comparators",
         "Development prevalence null\nKPC-family rule\nBroad carbapenemase rule\nPaired cluster bootstrap"),
        (0.71, 0.15, 0.22, 0.25, "Source-shift sensitivity",
         f"Source CV AUROC={s11f['source_signature']['random_5fold_source_classifier_AUROC']:.3f}\n"
         f"{s11f['source_signature']['primary_source_marker_count']} source markers\n"
         f"{s11f['compute_triage']['decision'].replace('_', ' ').title()}"),
    ]

    for x, y, w, h, title, body in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.01",
            linewidth=0.9,
            fill=False,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + 0.012, y + h - 0.04, title, transform=ax.transAxes,
                ha="left", va="top", fontweight="bold", fontsize=8.2)
        ax.text(x + 0.012, y + h - 0.085, body, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.3, linespacing=1.3)

    # Main-row arrows.
    for x1, x2 in [(0.26, 0.31), (0.54, 0.59), (0.77, 0.81)]:
        ax.annotate(
            "",
            xy=(x2, 0.745),
            xytext=(x1, 0.745),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=0.9),
        )

    # Downward arrows from E2.
    for x2 in [0.26, 0.54, 0.82]:
        ax.annotate(
            "",
            xy=(x2, 0.40),
            xytext=(0.89, 0.62),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=0.8),
        )

    ax.text(
        0.5,
        0.02,
        "Primary inference remains the frozen FILE11C blind E2 result; FILE11D–F are explicitly secondary/post-hoc.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    return save_figure(fig, out_stem)


# =============================================================================
# Figure 2 — Internal vs external transportability
# =============================================================================

def figure2_internal_external(master: pd.DataFrame, out_stem: Path, backing: Path):
    schemes = [
        ("Internal random stratified CV", "Random"),
        ("Internal genomic-cluster-aware CV", "Cluster-aware"),
        ("Internal MLST-aware CV", "MLST-aware"),
        ("Frozen ML — E2", "External E2"),
    ]
    metrics = [
        ("roc_auc", "AUROC"),
        ("average_precision", "Average precision"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("mcc", "MCC"),
    ]

    rows = []
    for cohort_key, label in schemes:
        for metric, metric_label in metrics:
            if cohort_key == "Frozen ML — E2":
                r = metric_row(
                    master,
                    section="Blind external validation",
                    analysis="FILE11C",
                    scope="primary",
                    cohort_or_model=cohort_key,
                    metric=metric,
                )
            else:
                r = metric_row(
                    master,
                    section="Internal validation",
                    analysis="FILE09",
                    scope="internal_reference",
                    cohort_or_model=cohort_key,
                    metric=metric,
                )
            rows.append(
                {
                    "validation": label,
                    "metric": metric_label,
                    "value": float(r["value"]),
                    "ci95_low": pd.to_numeric(r["ci95_low"], errors="coerce"),
                    "ci95_high": pd.to_numeric(r["ci95_high"], errors="coerce"),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(backing, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, 5.1))
    for i, ((metric_key, title), ax) in enumerate(zip(metrics, axes.flat)):
        sub = df[df["metric"].eq(title)].copy()
        x = np.arange(len(sub))
        vals = sub["value"].to_numpy(float)

        yerr = None
        if sub["ci95_low"].notna().all() and sub["ci95_high"].notna().all():
            lo = vals - sub["ci95_low"].to_numpy(float)
            hi = sub["ci95_high"].to_numpy(float) - vals
            yerr = np.vstack([lo, hi])

        ax.errorbar(
            x,
            vals,
            yerr=yerr,
            fmt="o",
            capsize=3,
            linewidth=1.0,
        )
        ax.plot(x, vals, linewidth=0.8, alpha=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["validation"], rotation=25, ha="right")
        ax.set_ylim(0, 1.02 if metric_key != "mcc" else 1.02)
        ax.set_ylabel(title)
        ax.axvline(2.5, linestyle="--", linewidth=0.8, alpha=0.6)
        clean_axes(ax)
        panel_label(ax, chr(65 + i))

    fig.subplots_adjust(wspace=0.30, hspace=0.48)
    return save_figure(fig, out_stem)


# =============================================================================
# Figure 3 — External curves/calibration
# =============================================================================

def figure3_external_behavior(
    pred: pd.DataFrame,
    s11c: dict,
    s11d: dict,
    out_stem: Path,
    backing_dir: Path,
):
    y = pred["label_binary"].to_numpy(dtype=int)
    p = pred["probability_R"].to_numpy(dtype=float)

    fpr, tpr, roc_thr = roc_curve(y, p)
    precision, recall, pr_thr = precision_recall_curve(y, p)

    # Equal-frequency calibration bins, 10 bins.
    cal = pred[["label_binary", "probability_R"]].copy()
    cal["bin"] = pd.qcut(cal["probability_R"], q=10, duplicates="drop")
    cal_df = (
        cal.groupby("bin", observed=False)
        .agg(
            n=("label_binary", "size"),
            mean_predicted_probability=("probability_R", "mean"),
            observed_resistant_rate=("label_binary", "mean"),
        )
        .reset_index(drop=True)
    )

    # Save backing data.
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        backing_dir / "Figure3A_ROC_backing.csv", index=False
    )
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(
        backing_dir / "Figure3B_PR_backing.csv", index=False
    )
    cal_df.to_csv(backing_dir / "Figure3C_calibration_backing.csv", index=False)
    pred[["asm_acc", "label_binary", "probability_R", "predicted_R"]].to_csv(
        backing_dir / "Figure3D_probability_backing.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, 5.6))

    # A ROC
    ax = axes[0, 0]
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    b = s11c["metrics"]["roc_auc"]
    ax.text(
        0.97,
        0.05,
        f"AUROC={b:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    clean_axes(ax)
    panel_label(ax, "A")

    # B PR
    ax = axes[0, 1]
    ax.plot(recall, precision)
    prevalence = y.mean()
    ax.axhline(prevalence, linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.text(
        0.97,
        0.05,
        f"AP={s11c['metrics']['average_precision']:.3f}\nPrevalence={prevalence:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    clean_axes(ax)
    panel_label(ax, "B")

    # C Calibration
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    ax.plot(
        cal_df["mean_predicted_probability"],
        cal_df["observed_resistant_rate"],
        marker="o",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability of resistance")
    ax.set_ylabel("Observed resistant fraction")
    ax.text(
        0.03,
        0.97,
        f"Slope={s11d['calibration']['calibration_slope']:.3f}\n"
        f"Intercept={s11d['calibration']['calibration_intercept']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    clean_axes(ax)
    panel_label(ax, "C")

    # D probability distributions
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 21)
    ax.hist(
        pred.loc[pred["label_binary"].eq(0), "probability_R"],
        bins=bins,
        alpha=0.55,
        density=True,
        label="Susceptible",
    )
    ax.hist(
        pred.loc[pred["label_binary"].eq(1), "probability_R"],
        bins=bins,
        alpha=0.55,
        density=True,
        label="Resistant",
    )
    ax.axvline(0.5, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Predicted probability of resistance")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    clean_axes(ax)
    panel_label(ax, "D")

    fig.subplots_adjust(wspace=0.32, hspace=0.34)
    return save_figure(fig, out_stem)


# =============================================================================
# Figure 4 — Comparators
# =============================================================================

def figure4_comparators(
    metrics: pd.DataFrame,
    boot: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_stem: Path,
    backing: Path,
):
    methods = [
        ("frozen_ml", "Frozen ML"),
        ("kpc_family_rule", "KPC rule"),
        ("amrfinder_carbapenemase_rule", "Broad carbapenemase rule"),
    ]

    backing_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 3.3))

    # A: BA / MCC with bootstrap CI.
    ax = axes[0]
    xpos = np.arange(len(methods))
    width = 0.34
    for j, metric in enumerate(["balanced_accuracy", "mcc"]):
        vals = []
        los = []
        his = []
        for method, label in methods:
            point = float(metrics.loc[metrics["method"].eq(method), metric].iloc[0])
            q = boot[
                boot["method"].astype(str).eq(method)
                & boot["metric"].astype(str).eq(metric)
            ]
            if len(q) == 1:
                lo = float(q.iloc[0]["ci95_low"])
                hi = float(q.iloc[0]["ci95_high"])
            else:
                lo = np.nan
                hi = np.nan
            vals.append(point)
            los.append(lo)
            his.append(hi)
            backing_rows.append(
                {
                    "panel": "A",
                    "method": label,
                    "metric": metric,
                    "value": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
        pos = xpos + (j - 0.5) * width
        bars = ax.bar(pos, vals, width=width, alpha=0.82, label=metric.replace("_", " ").title())
        if not np.isnan(los).all():
            loerr = np.array(vals) - np.array(los)
            hierr = np.array(his) - np.array(vals)
            ax.errorbar(pos, vals, yerr=np.vstack([loerr, hierr]), fmt="none", capsize=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels([m[1] for m in methods], rotation=30, ha="right")
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False, fontsize=6.5)
    clean_axes(ax)
    panel_label(ax, "A")

    # B: sensitivity / specificity.
    ax = axes[1]
    for j, metric in enumerate(["sensitivity", "specificity"]):
        vals = [
            float(metrics.loc[metrics["method"].eq(method), metric].iloc[0])
            for method, _ in methods
        ]
        pos = xpos + (j - 0.5) * width
        ax.bar(pos, vals, width=width, alpha=0.82, label=metric.title())
        for (method, label), val in zip(methods, vals):
            backing_rows.append(
                {
                    "panel": "B",
                    "method": label,
                    "metric": metric,
                    "value": val,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                }
            )
    ax.set_xticks(xpos)
    ax.set_xticklabels([m[1] for m in methods], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.legend(frameon=False, fontsize=6.5)
    clean_axes(ax)
    panel_label(ax, "B")

    # C: paired deltas ML minus broad rule.
    ax = axes[2]
    metric_order = [
        ("balanced_accuracy", "BA"),
        ("mcc", "MCC"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1", "F1"),
        ("roc_auc", "AUROC"),
        ("average_precision", "AP"),
    ]
    q = contrasts[
        contrasts["comparator"].astype(str).eq("amrfinder_carbapenemase_rule")
    ].copy()
    yloc = np.arange(len(metric_order))
    points = []
    loerr = []
    hierr = []
    labels = []
    for metric, label in metric_order:
        r = q[q["metric"].astype(str).eq(metric)]
        if len(r) != 1:
            raise RuntimeError(f"Missing FILE11E paired contrast for {metric}")
        point = float(r.iloc[0]["bootstrap_delta_mean"])
        lo = float(r.iloc[0]["ci95_low"])
        hi = float(r.iloc[0]["ci95_high"])
        points.append(point)
        loerr.append(point - lo)
        hierr.append(hi - point)
        labels.append(label)
        backing_rows.append(
            {
                "panel": "C",
                "method": "Frozen ML minus broad carbapenemase rule",
                "metric": metric,
                "value": point,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    ax.errorbar(points, yloc, xerr=np.vstack([loerr, hierr]), fmt="o", capsize=2)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    ax.set_yticks(yloc)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired bootstrap difference\n(ML − broad rule)")
    clean_axes(ax)
    panel_label(ax, "C")

    pd.DataFrame(backing_rows).to_csv(backing, index=False)
    fig.subplots_adjust(wspace=0.48, bottom=0.30)
    return save_figure(fig, out_stem)


# =============================================================================
# Figure 5 — Domain shift
# =============================================================================

def figure5_domain_shift(
    shift: pd.DataFrame,
    unknown: pd.DataFrame,
    out_stem: Path,
    backing_dir: Path,
):
    shift = shift.copy()
    top_weighted = shift.nlargest(15, "coefficient_weighted_abs_shift").copy()
    top_prevalence = shift.nlargest(15, "absolute_prevalence_shift").copy()

    unknown = unknown.copy()
    if "external_genomes_present" in unknown.columns:
        top_unknown = unknown.sort_values("external_genomes_present", ascending=False).head(15)
    else:
        top_unknown = unknown.head(15)

    shift.to_csv(backing_dir / "Figure5A_all_feature_shift_backing.csv", index=False)
    top_weighted.to_csv(backing_dir / "Figure5B_top_weighted_shift_backing.csv", index=False)
    top_unknown.to_csv(backing_dir / "Figure5C_unknown_determinants_backing.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 3.5))

    # A prevalence scatter
    ax = axes[0]
    ax.scatter(
        shift["internal_prevalence"],
        shift["external_prevalence"],
        s=10,
        alpha=0.55,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    annotate = shift.nlargest(8, "coefficient_weighted_abs_shift")
    for _, r in annotate.iterrows():
        ax.annotate(
            str(r["feature"]),
            (r["internal_prevalence"], r["external_prevalence"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.8,
        )
    ax.set_xlim(0, max(0.55, float(shift["internal_prevalence"].max()) + 0.03))
    ax.set_ylim(0, max(0.55, float(shift["external_prevalence"].max()) + 0.03))
    ax.set_xlabel("Development prevalence")
    ax.set_ylabel("External E2 prevalence")
    clean_axes(ax)
    panel_label(ax, "A")

    # B coefficient-weighted shifts
    ax = axes[1]
    tb = top_weighted.sort_values("coefficient_weighted_abs_shift")
    ax.barh(tb["feature"], tb["coefficient_weighted_abs_shift"])
    ax.set_xlabel("|Coefficient| × absolute prevalence shift")
    ax.tick_params(axis="y", labelsize=5.9)
    clean_axes(ax)
    panel_label(ax, "B")

    # C unknown determinants
    ax = axes[2]
    det_col = "determinant_key" if "determinant_key" in top_unknown.columns else top_unknown.columns[0]
    count_col = (
        "external_genomes_present"
        if "external_genomes_present" in top_unknown.columns
        else top_unknown.columns[1]
    )
    tu = top_unknown.sort_values(count_col)
    ax.barh(tu[det_col].astype(str), tu[count_col].astype(float))
    ax.set_xlabel("External genomes carrying determinant")
    ax.tick_params(axis="y", labelsize=5.9)
    clean_axes(ax)
    panel_label(ax, "C")

    fig.subplots_adjust(wspace=0.55, left=0.08, right=0.99)
    return save_figure(fig, out_stem)


# =============================================================================
# Figure 6 — Lineage and source robustness
# =============================================================================

def figure6_robustness(
    mlst: pd.DataFrame,
    lineage_boot: pd.DataFrame,
    source_external: pd.DataFrame,
    source_boot: pd.DataFrame,
    source_cv: pd.DataFrame,
    source_features: pd.DataFrame,
    out_stem: Path,
    backing: Path,
):
    backing_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, 5.2))

    # A: lineage point metrics
    ax = axes[0, 0]
    lineage_order = ["seen_ST", "unseen_ST", "unresolved_ST"]
    metric_order = ["roc_auc", "balanced_accuracy", "mcc"]
    x = np.arange(len(lineage_order))
    width = 0.23
    for j, metric in enumerate(metric_order):
        vals = []
        for lineage in lineage_order:
            r = mlst[mlst["lineage_novelty"].astype(str).eq(lineage)]
            if len(r) != 1:
                vals.append(np.nan)
            else:
                vals.append(float(r.iloc[0][metric]))
                backing_rows.append(
                    {"panel": "A", "group": lineage, "metric": metric, "value": vals[-1]}
                )
        ax.bar(x + (j - 1) * width, vals, width=width, label=metric.upper() if metric == "mcc" else metric.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(["Seen ST", "Unseen ST", "Unresolved ST"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False, fontsize=6.2)
    clean_axes(ax)
    panel_label(ax, "A")

    # B: seen-unseen bootstrap contrast
    ax = axes[0, 1]
    metrics = ["roc_auc", "balanced_accuracy", "mcc"]
    labels = ["AUROC", "Balanced accuracy", "MCC"]
    points, los, his = [], [], []
    for metric in metrics:
        r = lineage_boot[lineage_boot["metric"].astype(str).eq(metric)]
        if len(r) != 1:
            raise RuntimeError(f"Missing lineage contrast for {metric}")
        point = float(r.iloc[0]["bootstrap_delta_mean"])
        lo = float(r.iloc[0]["ci95_low"])
        hi = float(r.iloc[0]["ci95_high"])
        points.append(point)
        los.append(point - lo)
        his.append(hi - point)
        backing_rows.append(
            {
                "panel": "B",
                "group": "seen_ST_minus_unseen_ST",
                "metric": metric,
                "value": point,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    ypos = np.arange(len(metrics))
    ax.errorbar(points, ypos, xerr=np.vstack([los, his]), fmt="o", capsize=2)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Seen-ST − unseen-ST")
    clean_axes(ax)
    panel_label(ax, "B")

    # C: source-sensitivity models external
    ax = axes[1, 0]
    model_order = [
        ("frozen_original", "Frozen"),
        ("source_debiased_full", "Debiased"),
        ("local_source_only", "Local-only"),
        ("ncbi_source_only", "NCBI-only"),
    ]
    x = np.arange(len(model_order))
    for j, metric in enumerate(["roc_auc", "balanced_accuracy", "brier"]):
        vals = []
        for model, label in model_order:
            r = source_external[source_external["model"].astype(str).eq(model)]
            if len(r) != 1:
                vals.append(np.nan)
            else:
                vals.append(float(r.iloc[0][metric]))
                backing_rows.append(
                    {"panel": "C", "group": label, "metric": metric, "value": vals[-1]}
                )
        ax.plot(x, vals, marker="o", label=metric.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in model_order], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False, fontsize=6.2)
    clean_axes(ax)
    panel_label(ax, "C")

    # D: source signature summary
    ax = axes[1, 1]
    random_row = source_cv[source_cv["scheme"].astype(str).eq("stratified_random_5fold")]
    source_auc = float(random_row.iloc[0]["roc_auc"]) if len(random_row) else np.nan
    marker_mask = as_bool_series(source_features["primary_source_marker"])
    marker_n = int(marker_mask.sum())
    top50_overlap = 0
    if "top50_abs_resistance_coefficient" in source_features.columns:
        top50_mask = as_bool_series(source_features["top50_abs_resistance_coefficient"])
        top50_overlap = int((marker_mask & top50_mask).sum())
    else:
        # Recreate from coefficient field if present.
        if "abs_resistance_model_coefficient" in source_features.columns:
            top50 = set(
                source_features.nlargest(
                    50, "abs_resistance_model_coefficient"
                )["feature"]
            )
            top50_overlap = int(
                source_features.loc[marker_mask, "feature"].isin(top50).sum()
            )

    ax.bar(
        ["Source CV\nAUROC", "Source\nmarkers", "Markers in\ntop-|coef| 50"],
        [source_auc, marker_n / 668.0, top50_overlap / 50.0],
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Scaled value")
    ax.text(
        0.5,
        0.97,
        f"Source CV AUROC={source_auc:.3f}\n"
        f"Primary markers={marker_n}/668\n"
        f"Markers in top-|coef|50={top50_overlap}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )
    clean_axes(ax)
    panel_label(ax, "D")

    pd.DataFrame(backing_rows).to_csv(backing, index=False)
    fig.subplots_adjust(wspace=0.42, hspace=0.45, bottom=0.16)
    return save_figure(fig, out_stem)


# =============================================================================
# Supplementary figures
# =============================================================================

def supp_figure_coefficients(coef: pd.DataFrame, out_stem: Path, backing: Path):
    if not {"feature", "coefficient"}.issubset(coef.columns):
        raise RuntimeError("Unexpected coefficient schema.")
    df = coef.copy()
    df["abs_coefficient"] = df["coefficient"].abs()
    top = df.nlargest(25, "abs_coefficient").sort_values("coefficient")
    top.to_csv(backing, index=False)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 4.2))
    ax.barh(top["feature"], top["coefficient"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Frozen logistic-regression coefficient")
    ax.tick_params(axis="y", labelsize=6)
    clean_axes(ax)
    return save_figure(fig, out_stem)


def supp_figure_confidence(pred: pd.DataFrame, out_stem: Path, backing: Path):
    work = pred.copy()
    y = work["label_binary"].to_numpy(int)
    pr = work["predicted_R"].to_numpy(int)
    et = np.full(len(work), "", dtype=object)
    et[(y == 1) & (pr == 1)] = "TP"
    et[(y == 0) & (pr == 0)] = "TN"
    et[(y == 0) & (pr == 1)] = "FP"
    et[(y == 1) & (pr == 0)] = "FN"
    work["error_type"] = et
    work.to_csv(backing, index=False)

    order = ["TP", "TN", "FP", "FN"]
    data = [work.loc[work["error_type"].eq(k), "probability_R"].to_numpy() for k in order]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.3))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.axhline(0.5, linestyle="--", linewidth=0.8)
    ax.set_ylabel("Predicted probability of resistance")
    clean_axes(ax)
    return save_figure(fig, out_stem)


def supp_figure_cluster_errors(cluster: pd.DataFrame, out_stem: Path, backing: Path):
    top = cluster.sort_values(["errors", "n"], ascending=[False, False]).head(25).copy()
    top.to_csv(backing, index=False)
    labels = top["erd_group"].astype(str)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 4.0))
    x = np.arange(len(top))
    ax.bar(x, top["errors"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=5.5)
    ax.set_ylabel("Errors per ERD/SNP cluster")
    ax.set_xlabel("ERD/SNP cluster")
    clean_axes(ax)
    return save_figure(fig, out_stem)


def supp_figure_unknown(unknown: pd.DataFrame, out_stem: Path, backing: Path):
    count_col = "external_genomes_present"
    det_col = "determinant_key"
    if not {count_col, det_col}.issubset(unknown.columns):
        raise RuntimeError("Unexpected out-of-schema determinant schema.")
    top = unknown.sort_values(count_col, ascending=False).head(30).copy()
    top.to_csv(backing, index=False)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 4.4))
    top2 = top.sort_values(count_col)
    ax.barh(top2[det_col], top2[count_col])
    ax.set_xlabel("External genomes carrying determinant")
    ax.tick_params(axis="y", labelsize=5.8)
    clean_axes(ax)
    return save_figure(fig, out_stem)


def supp_figure_source_markers(source_features: pd.DataFrame, out_stem: Path, backing: Path):
    df = source_features.copy()
    if "absolute_prevalence_shift" not in df.columns:
        raise RuntimeError("Unexpected source-feature association schema.")
    top = df.sort_values("absolute_prevalence_shift", ascending=False).head(25).copy()
    top.to_csv(backing, index=False)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 4.4))
    top2 = top.sort_values("local_minus_ncbi_prevalence")
    ax.barh(top2["feature"], top2["local_minus_ncbi_prevalence"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Local-AMRFinder prevalence − NCBI-precomputed prevalence")
    ax.tick_params(axis="y", labelsize=5.8)
    clean_axes(ax)
    return save_figure(fig, out_stem)


# =============================================================================
# Main tables
# =============================================================================

def make_table1(s11c: dict, s11d: dict, s11e: dict, s11f: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Item": "Development cohort",
                "Value": f"{s11c['development']['n']:,} genomes",
                "Details": f"R={s11c['development']['R']:,}; S={s11c['development']['S']:,}",
            },
            {
                "Item": "Primary representation",
                "Value": f"{s11c['model']['feature_count']} known-AMR features",
                "Details": "Frozen schema",
            },
            {
                "Item": "Primary model",
                "Value": "Logistic regression",
                "Details": (
                    f"C={s11c['model']['C']}; solver={s11c['model']['solver']}; "
                    f"threshold={s11c['model']['decision_threshold']}"
                ),
            },
            {
                "Item": "External E2 cohort",
                "Value": f"{s11c['external_meropenem_E2']['n']} genomes",
                "Details": (
                    f"R={s11c['external_meropenem_E2']['R']}; "
                    f"S={s11c['external_meropenem_E2']['S']}; "
                    f"{s11c['external_meropenem_E2']['unique_erd_groups']} ERD/SNP groups"
                ),
            },
            {
                "Item": "External feature matrix",
                "Value": f"{s11c['external_feature_matrix']['shape'][0]} × "
                         f"{s11c['external_feature_matrix']['shape'][1]}",
                "Details": (
                    f"{s11c['external_feature_matrix']['out_of_schema_determinants']} "
                    f"out-of-schema determinants; "
                    f"{s11c['external_feature_matrix']['zero_feature_genomes']} zero-feature genomes"
                ),
            },
            {
                "Item": "Bridge audit",
                "Value": s11c["bridge_audit"]["status"],
                "Details": (
                    f"{s11c['bridge_audit']['exact_row_matches']}/"
                    f"{s11c['bridge_audit']['local_training_genomes_audited']} exact; "
                    f"{s11c['bridge_audit']['mismatched_cells']} mismatched cells"
                ),
            },
            {
                "Item": "E2 used for tuning",
                "Value": "No",
                "Details": "No feature, hyperparameter, or threshold selection on E2",
            },
            {
                "Item": "Post-hoc diagnostics",
                "Value": "FILE11D–F",
                "Details": "Calibration, lineage, domain/source shift, simple comparators",
            },
        ]
    )


def make_table2(file09: pd.DataFrame) -> pd.DataFrame:
    q = file09[
        file09["representation"].astype(str).eq("known_amr")
        & file09["model"].astype(str).eq("logreg_l2")
        & file09["scheme"].isin(
            ["random_stratified", "genomic_cluster_aware", "mlst_aware"]
        )
    ].copy()

    labels = {
        "random_stratified": "Random stratified",
        "genomic_cluster_aware": "Genomic-cluster-aware",
        "mlst_aware": "MLST-aware",
    }
    rows = []
    for _, r in q.iterrows():
        rows.append(
            {
                "Validation scheme": labels[r["scheme"]],
                "AUROC (95% CI)": ci_string(
                    r["roc_auc_mean"],
                    r["roc_auc_cluster_boot_ci95_low"],
                    r["roc_auc_cluster_boot_ci95_high"],
                ),
                "AP (95% CI)": ci_string(
                    r["average_precision_mean"],
                    r["average_precision_cluster_boot_ci95_low"],
                    r["average_precision_cluster_boot_ci95_high"],
                ),
                "Balanced accuracy (95% CI)": ci_string(
                    r["balanced_accuracy_mean"],
                    r["balanced_accuracy_cluster_boot_ci95_low"],
                    r["balanced_accuracy_cluster_boot_ci95_high"],
                ),
                "MCC (95% CI)": ci_string(
                    r["mcc_mean"],
                    r["mcc_cluster_boot_ci95_low"],
                    r["mcc_cluster_boot_ci95_high"],
                ),
                "Brier": fmt3(r["brier_mean"]),
                "Log loss": fmt3(r["log_loss_mean"]),
            }
        )
    return pd.DataFrame(rows)


def make_table3(s11c: dict, boot: pd.DataFrame) -> pd.DataFrame:
    bmap = {r.metric: r for r in boot.itertuples(index=False)}
    metrics = [
        ("roc_auc", "AUROC"),
        ("average_precision", "Average precision"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("mcc", "MCC"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1", "F1"),
        ("brier", "Brier score"),
        ("log_loss", "Log loss"),
    ]
    rows = []
    for key, label in metrics:
        point = float(s11c["metrics"][key])
        b = bmap.get(key)
        rows.append(
            {
                "Metric": label,
                "Point estimate": fmt3(point),
                "95% cluster-bootstrap CI": (
                    f"{float(b.ci95_low):.3f}–{float(b.ci95_high):.3f}"
                    if b is not None
                    else "—"
                ),
            }
        )
    rows.extend(
        [
            {"Metric": "True positives", "Point estimate": str(s11c["metrics"]["tp"]), "95% cluster-bootstrap CI": "—"},
            {"Metric": "False negatives", "Point estimate": str(s11c["metrics"]["fn"]), "95% cluster-bootstrap CI": "—"},
            {"Metric": "True negatives", "Point estimate": str(s11c["metrics"]["tn"]), "95% cluster-bootstrap CI": "—"},
            {"Metric": "False positives", "Point estimate": str(s11c["metrics"]["fp"]), "95% cluster-bootstrap CI": "—"},
        ]
    )
    return pd.DataFrame(rows)


def make_table4(
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Publication-ready comparator table:
    one row per metric, point estimates for all methods, and paired
    ML-minus-broad-rule cluster-bootstrap contrast.
    """
    method_cols = [
        ("frozen_ml", "Frozen ML"),
        ("development_prevalence_null", "Development-prevalence null"),
        ("kpc_family_rule", "KPC-family rule"),
        ("amrfinder_carbapenemase_rule", "Broad carbapenemase rule"),
    ]
    metric_rows = [
        ("roc_auc", "AUROC"),
        ("average_precision", "Average precision"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("mcc", "MCC"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1", "F1"),
        ("accuracy", "Accuracy"),
        ("brier", "Brier score"),
        ("log_loss", "Log loss"),
    ]

    indexed = metrics.set_index("method")
    contrast_q = contrasts[
        contrasts["comparator"].astype(str).eq("amrfinder_carbapenemase_rule")
    ].copy()

    rows = []
    for metric, label in metric_rows:
        row = {"Metric": label}
        for method, colname in method_cols:
            if method not in indexed.index:
                row[colname] = "NA"
                continue
            val = indexed.loc[method, metric]
            row[colname] = "NA" if pd.isna(val) else fmt3(val)

        cq = contrast_q[contrast_q["metric"].astype(str).eq(metric)]
        if len(cq) == 1:
            c = cq.iloc[0]
            row["Paired Δ ML − broad rule (95% CI)"] = (
                f"{float(c['bootstrap_delta_mean']):+.3f} "
                f"({float(c['ci95_low']):+.3f} to {float(c['ci95_high']):+.3f})"
            )
        else:
            row["Paired Δ ML − broad rule (95% CI)"] = "—"
        rows.append(row)

    return pd.DataFrame(rows)


def make_table5(
    s11d: dict,
    mlst: pd.DataFrame,
    s11f: dict,
    source_external: pd.DataFrame,
) -> pd.DataFrame:
    seen = mlst[mlst["lineage_novelty"].eq("seen_ST")].iloc[0]
    unseen = mlst[mlst["lineage_novelty"].eq("unseen_ST")].iloc[0]
    frozen = source_external[source_external["model"].eq("frozen_original")].iloc[0]
    deb = source_external[source_external["model"].eq("source_debiased_full")].iloc[0]
    local = source_external[source_external["model"].eq("local_source_only")].iloc[0]

    return pd.DataFrame(
        [
            {
                "Diagnostic": "Calibration slope",
                "Estimate": fmt3(s11d["calibration"]["calibration_slope"]),
                "Interpretation": "Ideal value = 1; diagnostic only",
            },
            {
                "Diagnostic": "Calibration intercept",
                "Estimate": fmt3(s11d["calibration"]["calibration_intercept"]),
                "Interpretation": "Ideal value = 0; diagnostic only",
            },
            {
                "Diagnostic": "Mean predicted P(R) vs observed R prevalence",
                "Estimate": (
                    f"{fmt3(s11d['calibration']['mean_predicted_probability_R'])} vs "
                    f"{fmt3(s11d['calibration']['observed_R_prevalence'])}"
                ),
                "Interpretation": "External probability overestimation",
            },
            {
                "Diagnostic": "Seen-ST AUROC",
                "Estimate": fmt3(seen["roc_auc"]),
                "Interpretation": f"n={int(seen['n'])}",
            },
            {
                "Diagnostic": "Unseen-ST AUROC",
                "Estimate": fmt3(unseen["roc_auc"]),
                "Interpretation": f"n={int(unseen['n'])}",
            },
            {
                "Diagnostic": "Development source-classifier AUROC",
                "Estimate": fmt3(
                    s11f["source_signature"]["random_5fold_source_classifier_AUROC"]
                ),
                "Interpretation": "Strong source separability; non-causal diagnostic",
            },
            {
                "Diagnostic": "Primary source markers",
                "Estimate": (
                    f"{s11f['source_signature']['primary_source_marker_count']}/668"
                ),
                "Interpretation": (
                    f"{s11f['source_signature']['source_markers_in_top50_abs_resistance_coefficients']} "
                    "in top-|coefficient| 50"
                ),
            },
            {
                "Diagnostic": "Source-debiased external AUROC",
                "Estimate": fmt3(deb["roc_auc"]),
                "Interpretation": f"Frozen AUROC={fmt3(frozen['roc_auc'])}",
            },
            {
                "Diagnostic": "Local-source-only external AUROC",
                "Estimate": fmt3(local["roc_auc"]),
                "Interpretation": (
                    f"BA={fmt3(local['balanced_accuracy'])}; "
                    f"Brier={fmt3(local['brier'])}"
                ),
            },
            {
                "Diagnostic": "Source-shift compute triage",
                "Estimate": s11f["compute_triage"]["decision"],
                "Interpretation": "No clear E2 rescue from pre-specified source sensitivity",
            },
        ]
    )


# =============================================================================
# Captions / README
# =============================================================================

def write_captions(out_root: Path, s11c: dict, s11d: dict, s11f: dict):
    captions = f"""# FILE12 figure captions

## Figure 1. Study design and frozen external-validation workflow
The development cohort comprised {s11c['development']['n']:,} genomes
({s11c['development']['R']:,} resistant and {s11c['development']['S']:,} susceptible)
represented by {s11c['model']['feature_count']} known-AMR features. Internal validation
used random-stratified, genomic-cluster-aware, and MLST-aware partitions. The final
logistic-regression model was frozen before evaluation in the independent E2 meropenem
cohort ({s11c['external_meropenem_E2']['n']} genomes; {s11c['external_meropenem_E2']['unique_erd_groups']}
NCBI ERD/SNP groups). FILE11D–F analyses are explicitly secondary/post-hoc and do not
replace the FILE11C primary external result.

## Figure 2. Internal-to-external performance transportability
Known-AMR logistic-regression performance across the three internal validation schemes
and the frozen blind E2 evaluation. Points show performance estimates; error bars show
available 95% cluster-bootstrap intervals. The vertical dashed separator distinguishes
internal from external evaluation.

## Figure 3. Blind external discrimination and calibration
(A) Receiver-operating-characteristic curve. (B) Precision-recall curve with external
resistance prevalence as the horizontal reference. (C) Equal-frequency calibration
curve. (D) Distribution of predicted resistance probabilities in susceptible and
resistant external isolates. The frozen E2 AUROC was {s11c['metrics']['roc_auc']:.3f};
the post-hoc calibration slope was {s11d['calibration']['calibration_slope']:.3f}.
No recalibration was applied.

## Figure 4. Frozen ML versus simple genotype comparators
(A) Balanced accuracy and MCC. (B) Sensitivity and specificity. (C) Paired
ERD/SNP-cluster-bootstrap differences between the frozen ML model and the broad
AMRFinder carbapenemase rule. Positive differences favor ML for discrimination and
classification metrics. Comparator analyses were secondary/post-hoc and were not used
to alter the primary FILE11C model.

## Figure 5. External genomic domain shift
(A) Development versus external prevalence across the 668 frozen features.
(B) Features with the largest coefficient-weighted absolute prevalence shifts.
(C) Most prevalent AMRFinder determinants observed externally but absent from the
frozen 668-feature schema. These analyses are descriptive and do not establish causal
drivers of external performance loss.

## Figure 6. Lineage and feature-generation robustness
(A) Frozen-model performance by seen, unseen, and unresolved sequence type.
(B) ERD/SNP-cluster-bootstrap contrast between seen-ST and unseen-ST performance.
(C) External performance of pre-specified source-sensitivity models.
(D) Development feature-source separability. Source-generation differences were
detectable (source-classifier AUROC={s11f['source_signature']['random_5fold_source_classifier_AUROC']:.3f}),
but removal of pre-specified source markers did not rescue E2 performance.

# Supplementary figure captions

## Figure S1. Largest coefficients of the frozen external model
Twenty-five features with the largest absolute logistic-regression coefficients in the
frozen known-AMR model. Positive coefficients increase the model log-odds of resistance;
negative coefficients decrease them.

## Figure S2. External prediction confidence by confusion category
Distribution of frozen resistance probabilities among true positives, true negatives,
false positives, and false negatives. The horizontal dashed line marks the fixed 0.5
decision threshold.

## Figure S3. ERD/SNP-cluster error concentration
External classification errors among the 25 ERD/SNP clusters contributing the largest
number of errors. Cluster-level summaries are descriptive and were not used to change
the model.

## Figure S4. Out-of-schema determinants
The 30 most prevalent determinants detected in external AMRFinder output but absent
from the frozen 668-feature schema.

## Figure S5. Feature-generation source markers
Features with the largest prevalence difference between the local-AMRFinder and
NCBI-precomputed development branches. Source assignment was not randomized, so these
differences can represent technical and/or population structure effects.
"""
    (out_root / "file12_figure_captions.md").write_text(
        textwrap.dedent(captions).strip() + "\n", encoding="utf-8"
    )

    table_titles = """# FILE12 table titles

- Table 1. Cohort composition and locked analysis design.
- Table 2. Internal validation performance of the known-AMR logistic-regression model.
- Table 3. Blind external E2 performance with ERD/SNP-cluster-bootstrap uncertainty.
- Table 4. Secondary external comparator performance and paired cluster-bootstrap contrasts.
- Table 5. Post-hoc transportability and robustness diagnostics.

Supplementary tables retain exact machine-readable values from canonical upstream
outputs and are not intended to replace the verified master metrics file.
"""
    (out_root / "file12_table_titles.md").write_text(
        table_titles, encoding="utf-8"
    )


def write_readme(out_root: Path):
    readme = """# FILE12 manuscript assets

This directory was generated from frozen upstream outputs and contains no new model
selection or tuning.

## Recommended main-text assets
- Figures 1–6 in `figures/main/`
- Tables 1–5 in `tables/main/`
- Figure captions in `file12_figure_captions.md`
- Table titles in `file12_table_titles.md`

## Supplementary assets
- Figures S1–S5 in `figures/supplementary/`
- Exact upstream/derived tables in `tables/supplementary/`

## Numerical provenance
- `AMR_Genome_ML_key_metrics_summary_verified.csv` is copied into the supplementary
  tables directory.
- FILE12 requires the strict key-metrics verification report to contain 142 PASS rows.
- Every figure has backing-data CSV under `backing_data/`.

## Figure formats
Each figure is exported as:
- PDF: preferred vector manuscript submission format
- SVG: editable vector source
- PNG: 600-dpi raster fallback

## Interpretation guardrails
The FILE11C blind E2 result remains primary. FILE11D–F are secondary/post-hoc
diagnostics. Source shift and lineage novelty are not interpreted causally.
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate manuscript-ready FILE12 assets.")
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <project-root>/results/file12_manuscript_assets",
    )
    args = ap.parse_args()

    root = args.project_root.resolve()
    out_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / "results/file12_manuscript_assets"
    )

    dirs = {
        "fig_main": mkdir(out_root / "figures/main"),
        "fig_supp": mkdir(out_root / "figures/supplementary"),
        "table_main": mkdir(out_root / "tables/main"),
        "table_supp": mkdir(out_root / "tables/supplementary"),
        "backing": mkdir(out_root / "backing_data"),
        "logs": mkdir(out_root / "metadata"),
    }

    p = input_paths(root)

    log("=" * 112)
    log("FILE12 — MANUSCRIPT-READY FIGURES, TABLES, CAPTIONS, AND ASSET MANIFEST")
    log("=" * 112)
    log(f"Version      : {VERSION}")
    log(f"Project root : {root}")
    log(f"Output root  : {out_root}")
    log("Mode         : REPORTING ONLY — NO MODEL FITTING OR TUNING")

    # Core required inputs.
    for key in [
        "master", "verification", "file09",
        "file11c_summary", "file11c_predictions", "file11c_bootstrap",
        "file11c_mlst", "file11c_coefficients", "file11c_unknown",
        "file11d_summary", "file11d_lineage_boot", "file11d_shift", "file11d_cluster",
        "file11e_summary", "file11e_metrics", "file11e_bootstrap",
        "file11e_contrasts", "file11e_lineage",
        "file11f_summary", "file11f_source_features", "file11f_source_cv",
        "file11f_internal", "file11f_external", "file11f_bootstrap",
    ]:
        require_file(p[key], key)

    # -------------------------------------------------------------------------
    log("Stage 1/6 — verify master metrics and frozen upstream status")
    master = pd.read_csv(p["master"], dtype=str, keep_default_na=False)
    verify_report = pd.read_csv(p["verification"])
    verify_master(master, verify_report)

    s11c = read_json(p["file11c_summary"])
    s11d = read_json(p["file11d_summary"])
    s11e = read_json(p["file11e_summary"])
    s11f = read_json(p["file11f_summary"])

    if s11c.get("status") != "PASS_BLIND_EXTERNAL_VALIDATION":
        raise RuntimeError("FILE11C final status is not PASS.")
    if s11d.get("status") != "PASS_POSTHOC_DIAGNOSTIC":
        raise RuntimeError("FILE11D final status is not PASS.")
    if s11e.get("status") != "PASS_POSTHOC_COMPARATOR":
        raise RuntimeError("FILE11E final status is not PASS.")
    if s11f.get("status") != "PASS_POSTHOC_SOURCE_SHIFT_DIAGNOSTIC":
        raise RuntimeError("FILE11F final status is not PASS.")

    pred = pd.read_csv(p["file11c_predictions"])
    file09 = pd.read_csv(p["file09"])
    boot11c = pd.read_csv(p["file11c_bootstrap"])
    mlst11c = pd.read_csv(p["file11c_mlst"])
    coef11c = pd.read_csv(p["file11c_coefficients"])
    unknown11c = pd.read_csv(p["file11c_unknown"])
    linboot11d = pd.read_csv(p["file11d_lineage_boot"])
    shift11d = pd.read_csv(p["file11d_shift"])
    cluster11d = pd.read_csv(p["file11d_cluster"])
    metrics11e = pd.read_csv(p["file11e_metrics"])
    boot11e = pd.read_csv(p["file11e_bootstrap"])
    contrasts11e = pd.read_csv(p["file11e_contrasts"])
    lineage11e = pd.read_csv(p["file11e_lineage"])
    source_features11f = pd.read_csv(p["file11f_source_features"])
    source_cv11f = pd.read_csv(p["file11f_source_cv"])
    internal11f = pd.read_csv(p["file11f_internal"])
    external11f = pd.read_csv(p["file11f_external"])
    boot11f = pd.read_csv(p["file11f_bootstrap"])

    # Frozen prediction hash recheck.
    pred_hash = sha256_file(p["file11c_predictions"])
    if pred_hash != s11c["frozen_prediction_sha256"]:
        raise RuntimeError("FILE11C prediction hash mismatch at FILE12.")
    log("[PROGRESS] strict metrics verification PASS | 142/142")
    log(f"[PROGRESS] frozen prediction SHA256 PASS | {pred_hash}")

    created: list[Path] = []

    # -------------------------------------------------------------------------
    log("Stage 2/6 — generate main manuscript figures")
    created += figure1_study_design(
        s11c, s11d, s11e, s11f,
        dirs["fig_main"] / "Figure1_study_design_and_validation_workflow",
    )
    log("[PROGRESS] Figure 1 PASS")

    created += figure2_internal_external(
        master,
        dirs["fig_main"] / "Figure2_internal_to_external_transportability",
        dirs["backing"] / "Figure2_backing_data.csv",
    )
    created.append(dirs["backing"] / "Figure2_backing_data.csv")
    log("[PROGRESS] Figure 2 PASS")

    created += figure3_external_behavior(
        pred,
        s11c,
        s11d,
        dirs["fig_main"] / "Figure3_blind_external_discrimination_and_calibration",
        dirs["backing"],
    )
    created += [
        dirs["backing"] / "Figure3A_ROC_backing.csv",
        dirs["backing"] / "Figure3B_PR_backing.csv",
        dirs["backing"] / "Figure3C_calibration_backing.csv",
        dirs["backing"] / "Figure3D_probability_backing.csv",
    ]
    log("[PROGRESS] Figure 3 PASS")

    created += figure4_comparators(
        metrics11e,
        boot11e,
        contrasts11e,
        dirs["fig_main"] / "Figure4_external_simple_comparator_benchmark",
        dirs["backing"] / "Figure4_backing_data.csv",
    )
    created.append(dirs["backing"] / "Figure4_backing_data.csv")
    log("[PROGRESS] Figure 4 PASS")

    created += figure5_domain_shift(
        shift11d,
        unknown11c,
        dirs["fig_main"] / "Figure5_external_genomic_domain_shift",
        dirs["backing"],
    )
    created += [
        dirs["backing"] / "Figure5A_all_feature_shift_backing.csv",
        dirs["backing"] / "Figure5B_top_weighted_shift_backing.csv",
        dirs["backing"] / "Figure5C_unknown_determinants_backing.csv",
    ]
    log("[PROGRESS] Figure 5 PASS")

    created += figure6_robustness(
        mlst11c,
        linboot11d,
        external11f,
        boot11f,
        source_cv11f,
        source_features11f,
        dirs["fig_main"] / "Figure6_lineage_and_source_robustness",
        dirs["backing"] / "Figure6_backing_data.csv",
    )
    created.append(dirs["backing"] / "Figure6_backing_data.csv")
    log("[PROGRESS] Figure 6 PASS")

    # -------------------------------------------------------------------------
    log("Stage 3/6 — generate supplementary figures")
    created += supp_figure_coefficients(
        coef11c,
        dirs["fig_supp"] / "FigureS1_frozen_model_top_coefficients",
        dirs["backing"] / "FigureS1_backing_data.csv",
    )
    created.append(dirs["backing"] / "FigureS1_backing_data.csv")

    created += supp_figure_confidence(
        pred,
        dirs["fig_supp"] / "FigureS2_prediction_confidence_by_error_type",
        dirs["backing"] / "FigureS2_backing_data.csv",
    )
    created.append(dirs["backing"] / "FigureS2_backing_data.csv")

    created += supp_figure_cluster_errors(
        cluster11d,
        dirs["fig_supp"] / "FigureS3_ERD_cluster_error_concentration",
        dirs["backing"] / "FigureS3_backing_data.csv",
    )
    created.append(dirs["backing"] / "FigureS3_backing_data.csv")

    created += supp_figure_unknown(
        unknown11c,
        dirs["fig_supp"] / "FigureS4_out_of_schema_determinants",
        dirs["backing"] / "FigureS4_backing_data.csv",
    )
    created.append(dirs["backing"] / "FigureS4_backing_data.csv")

    created += supp_figure_source_markers(
        source_features11f,
        dirs["fig_supp"] / "FigureS5_feature_generation_source_markers",
        dirs["backing"] / "FigureS5_backing_data.csv",
    )
    created.append(dirs["backing"] / "FigureS5_backing_data.csv")
    log("[PROGRESS] Figures S1–S5 PASS")

    # -------------------------------------------------------------------------
    log("Stage 4/6 — generate main manuscript tables")
    main_tables = [
        (
            make_table1(s11c, s11d, s11e, s11f),
            dirs["table_main"] / "Table1_cohort_and_locked_analysis_design",
            "Table 1. Cohort composition and locked analysis design.",
        ),
        (
            make_table2(file09),
            dirs["table_main"] / "Table2_internal_validation_performance",
            "Table 2. Internal validation performance of the known-AMR logistic-regression model.",
        ),
        (
            make_table3(s11c, boot11c),
            dirs["table_main"] / "Table3_blind_external_E2_performance",
            "Table 3. Blind external E2 performance with ERD/SNP-cluster-bootstrap uncertainty.",
        ),
        (
            make_table4(metrics11e, contrasts11e),
            dirs["table_main"] / "Table4_external_comparator_performance",
            "Table 4. Secondary external comparator performance and paired cluster-bootstrap contrasts.",
        ),
        (
            make_table5(s11d, mlst11c, s11f, external11f),
            dirs["table_main"] / "Table5_transportability_and_robustness",
            "Table 5. Post-hoc transportability and robustness diagnostics.",
        ),
    ]
    for df, stem, caption in main_tables:
        created += write_table_bundle(df, stem, caption)
    log("[PROGRESS] Tables 1–5 PASS")

    # -------------------------------------------------------------------------
    log("Stage 5/6 — export supplementary tables and captions")
    supp_sources = [
        (p["master"], "TableS1_verified_master_metrics.csv"),
        (p["file11c_coefficients"], "TableS2_frozen_model_coefficients.csv"),
        (p["file11d_shift"], "TableS3_feature_prevalence_domain_shift.csv"),
        (p["file11c_unknown"], "TableS4_external_out_of_schema_determinants.csv"),
        (p["file11c_mlst"], "TableS5_external_lineage_metrics.csv"),
        (p["file11e_discordant"], "TableS6_comparator_discordant_cases.csv"),
        (p["file11d_cluster"], "TableS7_ERD_cluster_error_profile.csv"),
        (p["file11f_source_features"], "TableS8_source_feature_associations.csv"),
        (p["file11f_external"], "TableS9_source_sensitivity_external_metrics.csv"),
        (p["file11f_bootstrap"], "TableS10_source_sensitivity_paired_bootstrap.csv"),
    ]
    for src, name in supp_sources:
        copied = copy_if_exists(src, dirs["table_supp"], name)
        if copied is not None:
            created.append(copied)

    # Additional guardrail summary.
    guardrail_rows = []
    for analysis, s in [
        ("FILE11C", s11c),
        ("FILE11D", s11d),
        ("FILE11E", s11e),
        ("FILE11F", s11f),
    ]:
        guards = s.get("scientific_guardrails", s.get("guardrails", {}))
        for key, value in guards.items():
            guardrail_rows.append(
                {"Analysis": analysis, "Guardrail": key, "Value": value}
            )
    guardrail_df = pd.DataFrame(guardrail_rows)
    created += write_table_bundle(
        guardrail_df,
        dirs["table_supp"] / "TableS11_analysis_guardrails",
        "Table S11. Analysis guardrails and frozen-design checks.",
    )

    write_captions(out_root, s11c, s11d, s11f)
    write_readme(out_root)
    created += [
        out_root / "file12_figure_captions.md",
        out_root / "file12_table_titles.md",
        out_root / "README.md",
    ]
    log("[PROGRESS] Supplementary tables + captions PASS")

    # -------------------------------------------------------------------------
    log("Stage 6/6 — verify assets and write manifest")
    # Check every claimed file exists and is non-empty.
    existing_created = []
    for path in created:
        if path.is_file() and path.stat().st_size > 0:
            existing_created.append(path)
        else:
            raise RuntimeError(f"Expected FILE12 asset missing/empty: {path}")

    # Require expected number of main figure formats.
    expected_main_figure_files = 6 * 3
    actual_main_figure_files = sum(
        1 for x in dirs["fig_main"].iterdir()
        if x.suffix.lower() in {".pdf", ".svg", ".png"}
    )
    if actual_main_figure_files != expected_main_figure_files:
        raise RuntimeError(
            f"Expected {expected_main_figure_files} main figure files, "
            f"found {actual_main_figure_files}"
        )

    expected_supp_figure_files = 5 * 3
    actual_supp_figure_files = sum(
        1 for x in dirs["fig_supp"].iterdir()
        if x.suffix.lower() in {".pdf", ".svg", ".png"}
    )
    if actual_supp_figure_files != expected_supp_figure_files:
        raise RuntimeError(
            f"Expected {expected_supp_figure_files} supplementary figure files, "
            f"found {actual_supp_figure_files}"
        )

    manifest_rows = []
    for path in sorted(out_root.rglob("*")):
        if not path.is_file():
            continue
        manifest_rows.append(
            {
                "relative_path": str(path.relative_to(out_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "suffix": path.suffix.lower(),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "file12_asset_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    summary = {
        "script_version": VERSION,
        "status": "PASS_MANUSCRIPT_ASSETS",
        "completed_utc": utc_now(),
        "project_root": str(root),
        "output_root": str(out_root),
        "primary_external_result": {
            "AUROC": s11c["metrics"]["roc_auc"],
            "average_precision": s11c["metrics"]["average_precision"],
            "balanced_accuracy": s11c["metrics"]["balanced_accuracy"],
            "MCC": s11c["metrics"]["mcc"],
            "Brier": s11c["metrics"]["brier"],
        },
        "metrics_verification": {
            "rows": len(verify_report),
            "all_pass": bool(verify_report["status"].astype(str).eq("PASS").all()),
            "master_rows": len(master),
        },
        "asset_counts": {
            "main_figures": 6,
            "supplementary_figures": 5,
            "main_figure_files": actual_main_figure_files,
            "supplementary_figure_files": actual_supp_figure_files,
            "main_tables": 5,
            "manifest_files_total": len(manifest),
        },
        "guardrails": {
            "model_refit": False,
            "feature_selection": False,
            "threshold_tuning": False,
            "E2_recalibration": False,
            "FILE11C_remains_primary": True,
            "FILE11D_F_are_posthoc": True,
        },
        "frozen_prediction_sha256": pred_hash,
    }
    summary_path = out_root / "file12_final_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    log("=" * 112)
    log("FILE12 STATUS : PASS_MANUSCRIPT_ASSETS")
    log("Main figures  : 6 × (PDF, SVG, PNG 600 dpi)")
    log("Supp figures  : 5 × (PDF, SVG, PNG 600 dpi)")
    log("Main tables   : 5 × (CSV, Markdown, LaTeX)")
    log(f"Manifest      : {manifest_path}")
    log(f"Final summary : {summary_path}")
    log(f"Output root   : {out_root}")
    log("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
