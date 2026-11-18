#!/usr/bin/env python3
"""
09_model_evaluation_and_robustness.py
=====================================

CORE PURPOSE
------------
Evaluate File08 out-of-fold (OOF) predictions without retraining, retuning,
or using outer-test labels for any new model/feature-selection decision.

This script is intentionally conservative. Its purpose is not to select a
"winner" post hoc, but to quantify:
    1) predictive discrimination and threshold performance,
    2) robustness to population-structure-aware validation,
    3) incremental value beyond the curated known-AMR baseline,
    4) uncertainty while respecting correlated bacterial lineages,
    5) feature-selection stability across outer folds,
    6) major-lineage subgroup robustness,
    7) integrity / leakage-related invariants of the File08 OOF outputs.

REVIEWER-DEFENSE DESIGN
-----------------------
A. Baselines
   - Null/prevalence-only baseline derived from EACH OUTER TRAINING FOLD only.
   - Curated 668-feature known-AMR baseline evaluated with all four model
     families used for the broader genomic representations.
   - Incremental comparisons use the SAME L2 logistic model so that the
     comparison isolates feature information rather than model-family changes.

B. Leakage protection
   - This file NEVER trains a model and NEVER performs feature selection.
   - It verifies that each Genome ID appears exactly once per
     scheme/representation/model OOF combination.
   - It verifies that the recorded fold for every prediction matches File07.
   - It verifies that prediction labels match the locked File07 phenotype.
   - It verifies the exact expected 45 OOF combinations.
   - It verifies File08's final summary metadata describing train-only
     supervised feature selection and no outer-test use.
   - It audits feature-selection rank/count/uniqueness and task-manifest
     cardinalities against the locked File08 design.

C. Class imbalance
   - Primary reporting includes AUROC, average precision (AP), balanced
     accuracy, sensitivity, specificity, F1 and MCC.
   - Accuracy, PPV and NPV are reported only as secondary metrics.

D. Threshold leakage
   - Threshold metrics use the FIXED predictions already produced in File08.
   - No threshold is optimized on outer-test data.
   - For logistic regression / ExtraTrees / histogram gradient boosting, the
     fixed prediction corresponds to the estimator's default probability
     decision rule.
   - For RBF-SVM, the stored score is a decision-function margin, NOT a
     probability; File09 does not apply a 0.5 threshold to it.

E. Fold aggregation
   - Fold-level metrics are the primary units of model-performance reporting.
   - The script reports mean, SD, min and max across the five locked outer
     folds.
   - Pooled OOF discrimination is retained as a descriptive diagnostic only,
     because score calibration/scale may vary among fold-specific models.
     This is especially important for RBF-SVM decision-function margins.

F. Dependence / uncertainty
   - Naive genome-level bootstrap can underestimate uncertainty when related
     isolates are present.
   - File09 therefore uses CLUSTER bootstrap over population-structure units,
     applying the SAME resampled group multiplicities across all outer folds.
   - Two uncertainty analyses are produced:
       * genomic-cluster bootstrap (primary; File07 label-independent clusters)
       * MLST-group bootstrap (sensitivity; unresolved MLST rows are singletons)
   - Bootstrap confidence intervals are reported for macro-fold AUROC, AP,
     balanced accuracy and MCC.
   - Paired cluster-bootstrap confidence intervals are also used for
     incremental-value and generalization-gap contrasts.
   - No naive DeLong test or genome-level IID p-value is used because those
     assumptions are questionable in a structured bacterial population.

G. Population-structure generalization
   - Same representation/model combinations are compared across:
       random_stratified
       genomic_cluster_aware
       mlst_aware
   - The lineage-aware minus random performance difference is reported with
     paired cluster-bootstrap confidence intervals.

H. Feature-selection stability
   - File08 train-only pangenome and unitig selections are evaluated for
     cross-fold Jaccard overlap and selection-frequency stability.
   - Stability is descriptive; it does not alter the trained models.

I. Subgroup robustness
   - Major exact MLST lineages are evaluated with threshold-based metrics.
   - Score-based subgroup AUROC/AP are intentionally not pooled across
     fold-specific models, avoiding score-scale comparability problems.

PRIMARY OUTPUTS
---------------
data/evaluation/
    file09_fold_metrics.csv
    file09_pooled_oof_diagnostics.csv
    file09_performance_summary.csv
    file09_bootstrap_uncertainty.csv
    file09_null_baseline_fold_metrics.csv
    file09_null_baseline_summary.csv
    file09_incremental_value.csv
    file09_generalization_gap.csv
    file09_feature_selection_stability.csv
    file09_feature_selection_frequency.csv
    file09_lineage_subgroup_performance.csv
    file09_integrity_audit.csv
    file09_evaluation_design.json

checkpoints/model_evaluation/
    bootstrap/*.npz
    file09_final_summary.json
    file09_last_failure.json

SMOKE TEST
----------
    python scripts/09_model_evaluation_and_robustness.py --smoke

FULL
----
    python scripts/09_model_evaluation_and_robustness.py

CPU PARALLELISM
---------------
Default bootstrap execution uses 8 worker PROCESSES.  Each worker is limited
to one inner numerical-library thread to avoid nested oversubscription.
Override only when needed:
    --workers 8

Smoke mode evaluates the six random-stratified logistic-regression OOF
combinations with 50 bootstrap replicates in a separate output/checkpoint
namespace. It does not alter full File08 artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import tempfile
import time
import traceback
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)

SCRIPT_VERSION = "1.1.1"
RANDOM_SEED = 20260921

EXPECTED_COHORT_N = 4227
EXPECTED_PREDICTION_ROWS = 190215
EXPECTED_TASK_ROWS = 225
EXPECTED_FEATURE_SELECTION_ROWS = 60000
EXPECTED_FOLDS = {0, 1, 2, 3, 4}

SCHEME_TO_COLUMN = {
    "random_stratified": "random_fold",
    "genomic_cluster_aware": "genomic_cluster_fold",
    "mlst_aware": "mlst_fold",
}

PRIMARY_REPRESENTATIONS = ("known_amr", "pangenome", "unitig")
INCREMENTAL_REPRESENTATIONS = (
    "known_plus_pangenome",
    "known_plus_unitig",
    "known_plus_pangenome_plus_unitig",
)
PRIMARY_MODELS = ("logreg_l2", "rbf_svm", "extra_trees", "hist_gb")
PROBABILITY_MODELS = {"logreg_l2", "extra_trees", "hist_gb"}

BOOTSTRAP_METRICS = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "mcc",
)

STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_seconds(x: float) -> str:
    if x < 60:
        return f"{x:.1f}s"
    if x < 3600:
        return f"{x/60:.1f}m"
    return f"{x/3600:.2f}h"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp.npz", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    return start


def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"Lock exists: {lock_path}\n"
            "Another File09 process may still be running. If the old process is "
            "definitely dead, inspect the lock before removing it."
        )
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "pid": os.getpid(),
                "started_utc": utc_now(),
                "script_version": SCRIPT_VERSION,
            },
            fh,
            indent=2,
        )


def expected_combo_set(schemes: Iterable[str]) -> set[Tuple[str, str, str]]:
    combos = set()
    for scheme in schemes:
        for rep in PRIMARY_REPRESENTATIONS:
            for model in PRIMARY_MODELS:
                combos.add((scheme, rep, model))
        for rep in INCREMENTAL_REPRESENTATIONS:
            combos.add((scheme, rep, "logreg_l2"))
    return combos


def exact_mlst_group(mlst: str, genome_id: str) -> Tuple[str, bool]:
    s = "" if pd.isna(mlst) else str(mlst).strip()
    if s and s not in {"-", "NA", "NaN", "nan", "None", "none"}:
        return f"ST::{s}", True
    return f"UNRESOLVED::{genome_id}", False


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def confusion_counts(
    y: np.ndarray,
    pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Tuple[float, float, float, float]:
    y = np.asarray(y, dtype=np.uint8)
    pred = np.asarray(pred, dtype=np.uint8)
    if sample_weight is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)

    tp = float(w[(y == 1) & (pred == 1)].sum())
    tn = float(w[(y == 0) & (pred == 0)].sum())
    fp = float(w[(y == 0) & (pred == 1)].sum())
    fn = float(w[(y == 1) & (pred == 0)].sum())
    return tn, fp, fn, tp


def weighted_balanced_accuracy(
    y: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray
) -> float:
    tn, fp, fn, tp = confusion_counts(y, pred, sample_weight)
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    if not np.isfinite(sens) or not np.isfinite(spec):
        return float("nan")
    return float((sens + spec) / 2.0)


def weighted_mcc(
    y: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray
) -> float:
    tn, fp, fn, tp = confusion_counts(y, pred, sample_weight)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return float("nan")
    return float((tp * tn - fp * fn) / den)


def metric_bundle(
    y: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
    is_probability: bool,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.uint8)

    if len(np.unique(y)) != 2:
        raise RuntimeError("Metric computation requires both phenotype classes.")

    tn, fp, fn, tp = confusion_counts(y, pred)

    precision_curve, recall_curve, _ = precision_recall_curve(y, score)
    pr_auc_trap = float(auc(recall_curve, precision_curve))

    out = {
        "n": int(len(y)),
        "n_resistant": int(y.sum()),
        "n_susceptible": int((1 - y).sum()),
        "resistant_fraction": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "pr_auc_trapezoid": pr_auc_trap,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tn": int(round(tn)),
        "fp": int(round(fp)),
        "fn": int(round(fn)),
        "tp": int(round(tp)),
        "brier": float("nan"),
        "log_loss": float("nan"),
    }

    if is_probability:
        eps = 1e-15
        clipped = np.clip(score, eps, 1 - eps)
        out["brier"] = float(brier_score_loss(y, clipped))
        out["log_loss"] = float(log_loss(y, clipped, labels=[0, 1]))

    return out


def score_only_metric(
    metric: str,
    y: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    positive_weight = float(sample_weight[y == 1].sum())
    negative_weight = float(sample_weight[y == 0].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        return float("nan")

    if metric == "roc_auc":
        return float(roc_auc_score(y, score, sample_weight=sample_weight))
    if metric == "average_precision":
        return float(
            average_precision_score(y, score, sample_weight=sample_weight)
        )
    if metric == "balanced_accuracy":
        return weighted_balanced_accuracy(y, pred, sample_weight)
    if metric == "mcc":
        return weighted_mcc(y, pred, sample_weight)
    raise KeyError(metric)


def audit_inputs(
    pred: pd.DataFrame,
    tasks: pd.DataFrame,
    fs: pd.DataFrame,
    folds: pd.DataFrame,
    summary: dict,
    schemes: Tuple[str, ...],
    smoke: bool,
    logger: Logger,
) -> pd.DataFrame:
    audit = []

    def record(check: str, passed: bool, detail: str):
        audit.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )
        if not passed:
            raise RuntimeError(f"Integrity audit failed: {check}: {detail}")

    required_pred = {
        "Genome ID", "y", "scheme", "fold", "representation", "model", "score", "pred"
    }
    record(
        "prediction_columns",
        required_pred.issubset(pred.columns),
        f"columns={list(pred.columns)}",
    )

    required_fold = {
        "Genome ID", "y", "MLST", "Genomic Cluster",
        "random_fold", "genomic_cluster_fold", "mlst_fold",
    }
    record(
        "fold_columns",
        required_fold.issubset(folds.columns),
        f"columns={list(folds.columns)}",
    )

    record(
        "cohort_size",
        len(folds) == EXPECTED_COHORT_N
        and folds["Genome ID"].nunique() == EXPECTED_COHORT_N,
        f"rows={len(folds)}, unique_ids={folds['Genome ID'].nunique()}",
    )

    if not smoke:
        record(
            "prediction_row_count",
            len(pred) == EXPECTED_PREDICTION_ROWS,
            f"rows={len(pred)}, expected={EXPECTED_PREDICTION_ROWS}",
        )
        record(
            "task_row_count",
            len(tasks) == EXPECTED_TASK_ROWS,
            f"rows={len(tasks)}, expected={EXPECTED_TASK_ROWS}",
        )
        record(
            "feature_selection_row_count",
            len(fs) == EXPECTED_FEATURE_SELECTION_ROWS,
            f"rows={len(fs)}, expected={EXPECTED_FEATURE_SELECTION_ROWS}",
        )

    pred["Genome ID"] = pred["Genome ID"].astype(str)
    folds["Genome ID"] = folds["Genome ID"].astype(str)

    record(
        "finite_scores",
        np.isfinite(pd.to_numeric(pred["score"], errors="coerce")).all(),
        "all stored OOF scores finite",
    )
    record(
        "binary_predictions",
        set(pd.to_numeric(pred["pred"], errors="raise").astype(int).unique()).issubset({0, 1}),
        f"values={sorted(pd.to_numeric(pred['pred'], errors='raise').astype(int).unique())}",
    )
    record(
        "binary_labels",
        set(pd.to_numeric(pred["y"], errors="raise").astype(int).unique()).issubset({0, 1}),
        f"values={sorted(pd.to_numeric(pred['y'], errors='raise').astype(int).unique())}",
    )

    observed_combos = set(
        map(tuple, pred[["scheme", "representation", "model"]].drop_duplicates().to_numpy())
    )
    expected = expected_combo_set(schemes)
    if smoke:
        expected = {x for x in expected if x[2] == "logreg_l2"}
    record(
        "expected_oof_grid",
        observed_combos == expected,
        f"observed={len(observed_combos)}, expected={len(expected)}",
    )

    dup = pred.duplicated(
        ["Genome ID", "scheme", "representation", "model"]
    ).sum()
    record(
        "unique_oof_keys",
        int(dup) == 0,
        f"duplicate_keys={int(dup)}",
    )

    # Every active OOF combination must cover the complete locked cohort exactly once.
    coverage = pred.groupby(["scheme", "representation", "model"]).agg(
        rows=("Genome ID", "size"),
        unique_genomes=("Genome ID", "nunique"),
    )
    record(
        "complete_oof_coverage",
        bool(
            (coverage["rows"] == EXPECTED_COHORT_N).all()
            and (coverage["unique_genomes"] == EXPECTED_COHORT_N).all()
        ),
        f"row_range={coverage['rows'].min()}..{coverage['rows'].max()}, "
        f"unique_range={coverage['unique_genomes'].min()}..{coverage['unique_genomes'].max()}",
    )

    fold_idx = folds.set_index("Genome ID", drop=False)
    bad_label = 0
    bad_fold = 0
    for scheme in schemes:
        psub = pred[pred["scheme"] == scheme]
        col = SCHEME_TO_COLUMN[scheme]
        expected_y = psub["Genome ID"].map(fold_idx["y"]).astype(int).to_numpy()
        observed_y = psub["y"].astype(int).to_numpy()
        bad_label += int(np.sum(expected_y != observed_y))

        expected_f = psub["Genome ID"].map(fold_idx[col]).astype(int).to_numpy()
        observed_f = psub["fold"].astype(int).to_numpy()
        bad_fold += int(np.sum(expected_f != observed_f))

    record("phenotype_linkage", bad_label == 0, f"mismatches={bad_label}")
    record("fold_linkage", bad_fold == 0, f"mismatches={bad_fold}")

    probability_rows = pred["model"].isin(PROBABILITY_MODELS)
    p_scores = pred.loc[probability_rows, "score"].astype(float)
    record(
        "probability_score_range",
        bool(((p_scores >= 0) & (p_scores <= 1)).all()),
        f"min={p_scores.min():.6g}, max={p_scores.max():.6g}",
    )

    record(
        "file08_status",
        summary.get("status") == "PASS",
        f"status={summary.get('status')}",
    )
    record(
        "train_only_feature_selection",
        summary.get("supervised_feature_selection_scope") == "outer_training_fold_only",
        f"scope={summary.get('supervised_feature_selection_scope')}",
    )
    record(
        "outer_test_labels_not_used",
        summary.get("outer_test_labels_used_for_selection_or_training") is False,
        f"value={summary.get('outer_test_labels_used_for_selection_or_training')}",
    )
    record(
        "fixed_hyperparameters_recorded",
        summary.get("fixed_model_hyperparameters") is True,
        f"value={summary.get('fixed_model_hyperparameters')}",
    )

    # Feature-selection integrity.
    required_fs = {
        "scheme", "fold", "representation", "rank",
        "feature_index", "chi2_score_train_only",
    }
    record(
        "feature_selection_columns",
        required_fs.issubset(fs.columns),
        f"columns={list(fs.columns)}",
    )

    if len(fs):
        fs_work = fs.copy()
        fs_work["fold"] = pd.to_numeric(fs_work["fold"], errors="raise").astype(int)
        fs_work["rank"] = pd.to_numeric(fs_work["rank"], errors="raise").astype(int)
        fs_work["feature_index"] = pd.to_numeric(
            fs_work["feature_index"], errors="raise"
        ).astype(int)
        fs_work["chi2_score_train_only"] = pd.to_numeric(
            fs_work["chi2_score_train_only"], errors="raise"
        ).astype(float)

        record(
            "feature_selection_scores_finite",
            np.isfinite(fs_work["chi2_score_train_only"]).all(),
            "all train-only chi2 scores finite",
        )
        fs_dup = fs_work.duplicated(
            ["scheme", "fold", "representation", "feature_index"]
        ).sum()
        record(
            "feature_selection_unique_within_fold",
            int(fs_dup) == 0,
            f"duplicates={int(fs_dup)}",
        )

        rank_ok = True
        count_ok = True
        for _, g in fs_work.groupby(["scheme", "fold", "representation"]):
            ranks = sorted(g["rank"].tolist())
            expected_ranks = list(range(1, len(g) + 1))
            if ranks != expected_ranks:
                rank_ok = False
            if not smoke and len(g) != 2000:
                count_ok = False
        record(
            "feature_selection_rank_integrity",
            rank_ok,
            "ranks are contiguous 1..k inside every selection group",
        )
        if not smoke:
            record(
                "feature_selection_2000_each",
                count_ok,
                "every scheme/fold/representation contains exactly 2000 features",
            )

    # Task-manifest integrity in FULL mode.
    if not smoke:
        required_tasks = {
            "scheme", "fold", "representation", "model",
            "test_n", "n_features", "checkpoint",
        }
        record(
            "task_manifest_columns",
            required_tasks.issubset(tasks.columns),
            f"columns={list(tasks.columns)}",
        )
        t = tasks.copy()
        t["fold"] = pd.to_numeric(t["fold"], errors="raise").astype(int)
        t["test_n"] = pd.to_numeric(t["test_n"], errors="raise").astype(int)
        t["n_features"] = pd.to_numeric(t["n_features"], errors="raise").astype(int)
        task_dup = t.duplicated(
            ["scheme", "fold", "representation", "model"]
        ).sum()
        record("task_manifest_unique", task_dup == 0, f"duplicates={task_dup}")

        expected_task_features = {
            "known_amr": 668,
            "pangenome": 2000,
            "unitig": 2000,
            "known_plus_pangenome": 2668,
            "known_plus_unitig": 2668,
            "known_plus_pangenome_plus_unitig": 4668,
        }
        nf_ok = all(
            int(row.n_features) == expected_task_features[row.representation]
            for row in t.itertuples(index=False)
        )
        record(
            "task_feature_counts",
            nf_ok,
            "n_features match locked File08 representation design",
        )

        fold_sizes = {
            scheme: folds[SCHEME_TO_COLUMN[scheme]].value_counts().to_dict()
            for scheme in SCHEME_TO_COLUMN
        }
        tn_ok = all(
            int(row.test_n) == int(fold_sizes[row.scheme][int(row.fold)])
            for row in t.itertuples(index=False)
        )
        record(
            "task_test_sizes_match_file07",
            tn_ok,
            "every task test_n matches its locked File07 outer fold size",
        )

    logger(f"Integrity/leakage audit PASS: {len(audit)} checks.")
    return pd.DataFrame(audit)


def compute_fold_metrics(pred: pd.DataFrame, logger: Logger) -> pd.DataFrame:
    rows = []
    group_cols = ["scheme", "fold", "representation", "model"]
    groups = pred.groupby(group_cols, sort=True)
    for key, g in groups:
        scheme, fold, rep, model = key
        y = g["y"].to_numpy(dtype=np.uint8)
        score = g["score"].to_numpy(dtype=np.float64)
        pr = g["pred"].to_numpy(dtype=np.uint8)
        m = metric_bundle(y, score, pr, model in PROBABILITY_MODELS)
        m.update(
            {
                "scheme": scheme,
                "fold": int(fold),
                "representation": rep,
                "model": model,
                "score_type": (
                    "probability" if model in PROBABILITY_MODELS
                    else "decision_margin"
                ),
                "threshold_source": "fixed_estimator_default_from_File08",
            }
        )
        rows.append(m)

    out = pd.DataFrame(rows)
    logger(f"Fold metrics computed: {len(out):,} rows.")
    return out


def compute_pooled_diagnostics(pred: pd.DataFrame, logger: Logger) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(
        ["scheme", "representation", "model"], sort=True
    ):
        scheme, rep, model = key
        y = g["y"].to_numpy(dtype=np.uint8)
        score = g["score"].to_numpy(dtype=np.float64)
        pr = g["pred"].to_numpy(dtype=np.uint8)
        m = metric_bundle(y, score, pr, model in PROBABILITY_MODELS)
        m.update(
            {
                "scheme": scheme,
                "representation": rep,
                "model": model,
                "score_type": (
                    "probability" if model in PROBABILITY_MODELS
                    else "decision_margin"
                ),
                "interpretation": (
                    "DESCRIPTIVE_POOLED_OOF_ONLY; fold-macro metrics are primary"
                ),
            }
        )
        rows.append(m)
    out = pd.DataFrame(rows)
    logger(f"Pooled OOF diagnostics computed: {len(out):,} rows.")
    return out


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc", "average_precision", "pr_auc_trapezoid",
        "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "ppv", "npv", "f1", "mcc", "brier", "log_loss",
    ]
    rows = []
    for key, g in fold_metrics.groupby(
        ["scheme", "representation", "model"], sort=True
    ):
        scheme, rep, model = key
        row = {
            "scheme": scheme,
            "representation": rep,
            "model": model,
            "n_outer_folds": len(g),
        }
        for metric in metric_cols:
            vals = g[metric].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite):
                row[f"{metric}_mean"] = float(np.mean(finite))
                row[f"{metric}_sd"] = (
                    float(np.std(finite, ddof=1))
                    if len(finite) > 1 else float("nan")
                )
                row[f"{metric}_min"] = float(np.min(finite))
                row[f"{metric}_max"] = float(np.max(finite))
            else:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_sd"] = float("nan")
                row[f"{metric}_min"] = float("nan")
                row[f"{metric}_max"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_null_baseline(
    folds: pd.DataFrame,
    schemes: Tuple[str, ...],
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    y_all = folds["y"].to_numpy(dtype=np.uint8)

    for scheme in schemes:
        col = SCHEME_TO_COLUMN[scheme]
        fvals = folds[col].to_numpy(dtype=int)
        for fold in sorted(EXPECTED_FOLDS):
            test = fvals == fold
            train = ~test
            y_train = y_all[train]
            y_test = y_all[test]
            p_train = float(y_train.mean())
            score = np.full(int(test.sum()), p_train, dtype=np.float64)
            pred = np.full(
                int(test.sum()),
                1 if p_train >= 0.5 else 0,
                dtype=np.uint8,
            )
            m = metric_bundle(y_test, score, pred, is_probability=True)
            m.update(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "representation": "null_training_prevalence",
                    "model": "constant_training_prevalence",
                    "train_resistant_fraction": p_train,
                    "threshold_source": "training_prevalence_only",
                }
            )
            rows.append(m)

    fold_df = pd.DataFrame(rows)
    summary = summarize_fold_metrics(fold_df)
    logger(
        f"Null prevalence-only baseline computed for {len(fold_df):,} outer folds."
    )
    return fold_df, summary


def build_bootstrap_units(folds: pd.DataFrame) -> Dict[str, np.ndarray]:
    genomic = folds["Genomic Cluster"].astype(str).to_numpy()
    mlst_groups = []
    for gid, st in zip(folds["Genome ID"].astype(str), folds["MLST"]):
        group, _ = exact_mlst_group(st, gid)
        mlst_groups.append(group)
    return {
        "genomic_cluster": genomic,
        "mlst_group": np.asarray(mlst_groups, dtype=object),
    }


def bootstrap_multiplicity_matrix(
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, group_index = np.unique(groups.astype(str), return_inverse=True)
    g = len(unique)
    rng = np.random.default_rng(seed)
    probs = np.full(g, 1.0 / g, dtype=np.float64)
    mult = rng.multinomial(g, probs, size=n_boot).astype(np.int16)
    return unique, group_index.astype(np.int32), mult


def bootstrap_checkpoint_path(
    base: Path, unit: str, scheme: str, rep: str, model: str
) -> Path:
    return base / f"{unit}__{scheme}__{rep}__{model}.npz"


def bootstrap_one_combo(
    g: pd.DataFrame,
    folds: pd.DataFrame,
    scheme: str,
    rep: str,
    model: str,
    unit_name: str,
    group_index_all: np.ndarray,
    multiplicities: np.ndarray,
    checkpoint: Path,
    logger: Logger,
) -> Dict[str, np.ndarray]:
    n_boot = multiplicities.shape[0]

    if checkpoint.exists():
        try:
            z = np.load(checkpoint, allow_pickle=False)
            if int(z["n_boot"][0]) == n_boot:
                out = {m: z[m].astype(np.float64) for m in BOOTSTRAP_METRICS}
                if all(len(v) == n_boot for v in out.values()):
                    if logger is not None:
                        logger(
                            f"  SKIP bootstrap checkpoint {unit_name} "
                            f"{scheme}/{rep}/{model}"
                        )
                    return out
        except Exception:
            pass

    pos = {gid: i for i, gid in enumerate(folds["Genome ID"].astype(str))}
    row_pos = g["Genome ID"].map(pos)
    if row_pos.isna().any():
        raise RuntimeError("Bootstrap combo contains unknown Genome ID.")
    g = g.assign(_cohort_pos=row_pos.astype(int)).sort_values("_cohort_pos")

    if len(g) != EXPECTED_COHORT_N:
        raise RuntimeError(
            f"Bootstrap requires full OOF cohort; got {len(g)} for "
            f"{scheme}/{rep}/{model}"
        )
    if not np.array_equal(
        g["_cohort_pos"].to_numpy(dtype=int),
        np.arange(EXPECTED_COHORT_N, dtype=int),
    ):
        raise RuntimeError("OOF combo does not map one-to-one to File07 cohort order.")

    y = g["y"].to_numpy(dtype=np.uint8)
    score = g["score"].to_numpy(dtype=np.float64)
    predv = g["pred"].to_numpy(dtype=np.uint8)
    foldv = g["fold"].to_numpy(dtype=int)

    out = {
        m: np.full(n_boot, np.nan, dtype=np.float64)
        for m in BOOTSTRAP_METRICS
    }

    fold_masks = {
        fold: (foldv == fold) for fold in sorted(EXPECTED_FOLDS)
    }
    t0 = time.time()

    for b in range(n_boot):
        if STOP_REQUESTED:
            raise KeyboardInterrupt

        sample_weight = multiplicities[b, group_index_all].astype(np.float64)

        macro = {m: [] for m in BOOTSTRAP_METRICS}
        valid = True
        for _, mask in fold_masks.items():
            w = sample_weight[mask]
            yy = y[mask]
            ss = score[mask]
            pp = predv[mask]

            for metric in BOOTSTRAP_METRICS:
                val = score_only_metric(metric, yy, ss, pp, w)
                if not np.isfinite(val):
                    valid = False
                    break
                macro[metric].append(val)
            if not valid:
                break

        if valid:
            for metric in BOOTSTRAP_METRICS:
                out[metric][b] = float(np.mean(macro[metric]))

    payload = {
        "n_boot": np.array([n_boot], dtype=np.int32),
        "unit_name": np.array([unit_name]),
    }
    payload.update({m: v.astype(np.float32) for m, v in out.items()})
    atomic_save_npz(checkpoint, **payload)

    valid_counts = {m: int(np.isfinite(v).sum()) for m, v in out.items()}
    if logger is not None:
        logger(
            f"  bootstrap {unit_name} {scheme}/{rep}/{model}: "
            f"{human_seconds(time.time()-t0)}; valid={valid_counts}"
        )
    return out



def _bootstrap_parallel_job(
    g: pd.DataFrame,
    folds: pd.DataFrame,
    scheme: str,
    rep: str,
    model: str,
    unit_name: str,
    group_index: np.ndarray,
    multiplicities: np.ndarray,
    checkpoint: Path,
):
    """
    One independent population-structure bootstrap job.

    This function is top-level so joblib/loky can safely execute it in a
    separate process.  Each job writes to a unique checkpoint path.
    """
    t0 = time.time()
    arrs = bootstrap_one_combo(
        g=g,
        folds=folds,
        scheme=scheme,
        rep=rep,
        model=model,
        unit_name=unit_name,
        group_index_all=group_index,
        multiplicities=multiplicities,
        checkpoint=checkpoint,
        logger=None,
    )
    return {
        "unit_name": unit_name,
        "scheme": scheme,
        "representation": rep,
        "model": model,
        "arrays": arrs,
        "elapsed_seconds": time.time() - t0,
    }


def compute_bootstrap_uncertainty(
    pred: pd.DataFrame,
    folds: pd.DataFrame,
    units: Dict[str, np.ndarray],
    n_boot: int,
    checkpoint_dir: Path,
    logger: Logger,
    workers: int,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str, str, str], np.ndarray]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache: Dict[Tuple[str, str, str, str, str], np.ndarray] = {}
    rows = []

    unit_boot = {}
    for i, (unit_name, labels) in enumerate(units.items()):
        unique, group_index, mult = bootstrap_multiplicity_matrix(
            labels, n_boot, RANDOM_SEED + 1000 * (i + 1)
        )
        unit_boot[unit_name] = (unique, group_index, mult)
        logger(
            f"Bootstrap unit {unit_name}: {len(unique):,} groups; "
            f"{n_boot:,} replicates."
        )

    combo_groups = list(
        pred.groupby(["scheme", "representation", "model"], sort=True)
    )

    jobs = []
    for key, g in combo_groups:
        scheme, rep, model = key
        # Copy only once per combo; joblib/loky may memmap large NumPy payloads.
        g_job = g.copy()
        for unit_name, (_, group_index, mult) in unit_boot.items():
            ckpt = bootstrap_checkpoint_path(
                checkpoint_dir, unit_name, scheme, rep, model
            )
            jobs.append(
                (
                    g_job,
                    folds,
                    scheme,
                    rep,
                    model,
                    unit_name,
                    group_index,
                    mult,
                    ckpt,
                )
            )

    logger(
        f"START cluster bootstrap: {len(combo_groups)} OOF combinations x "
        f"{len(units)} population-structure units = {len(jobs)} independent jobs."
    )
    logger(
        f"Parallel bootstrap workers: {workers} processes; "
        "inner numerical-library threads capped at 1/worker."
    )

    t0 = time.time()

    # CPU-bound work: processes avoid the Python GIL.  Capping inner threads
    # prevents 8 workers from each spawning their own BLAS/OpenMP thread pool.
    with parallel_config(
        backend="loky",
        n_jobs=workers,
        inner_max_num_threads=1,
    ):
        results = Parallel(
            n_jobs=workers,
            backend="loky",
            batch_size=1,
            pre_dispatch=workers,
            verbose=0,
        )(
            delayed(_bootstrap_parallel_job)(*job)
            for job in jobs
        )

    for i, result in enumerate(results, start=1):
        unit_name = result["unit_name"]
        scheme = result["scheme"]
        rep = result["representation"]
        model = result["model"]
        arrs = result["arrays"]

        logger(
            f"  bootstrap job {i}/{len(results)} PASS: "
            f"{unit_name} {scheme}/{rep}/{model}; "
            f"{human_seconds(float(result['elapsed_seconds']))}"
        )

        for metric, vals in arrs.items():
            vals = np.asarray(vals, dtype=np.float64)
            cache[(unit_name, scheme, rep, model, metric)] = vals
            finite = vals[np.isfinite(vals)]
            # Some lineage-aware cluster-bootstrap draws can make one outer
            # test fold contain effectively only one phenotype class after
            # cluster resampling. AUROC/AP are undefined for those draws.
            # They are therefore excluded rather than assigned an artificial
            # score. This is an estimability issue caused by strongly
            # structured lineages, not a model failure.
            #
            # Require >=80% valid draws (and >=500 valid draws in a standard
            # 1000-replicate full run). This leaves ample Monte-Carlo support
            # for percentile CIs while explicitly recording the valid fraction.
            min_valid = (
                max(25, int(0.80 * n_boot))
                if n_boot < 100
                else max(500, int(0.80 * n_boot))
            )
            if len(finite) < min_valid:
                raise RuntimeError(
                    f"Too few estimable cluster-bootstrap replicates for "
                    f"{unit_name}/{scheme}/{rep}/{model}/{metric}: "
                    f"{len(finite)}/{n_boot} valid "
                    f"({len(finite)/n_boot:.1%}); required >= {min_valid}. "
                    "Invalid draws occur when cluster resampling leaves an "
                    "outer test fold with only one phenotype class."
                )
            rows.append(
                {
                    "bootstrap_unit": unit_name,
                    "scheme": scheme,
                    "representation": rep,
                    "model": model,
                    "metric": metric,
                    "n_boot_requested": n_boot,
                    "n_boot_valid": len(finite),
                    "valid_fraction": float(len(finite) / n_boot),
                    "invalid_draws_reason": (
                        "A resampled outer test fold contained effectively "
                        "only one phenotype class; metric undefined."
                    ),
                    "bootstrap_mean": float(np.mean(finite)),
                    "ci95_low": float(np.quantile(finite, 0.025)),
                    "ci95_high": float(np.quantile(finite, 0.975)),
                }
            )

    logger(
        f"DONE parallel cluster-bootstrap uncertainty in "
        f"{human_seconds(time.time()-t0)}."
    )
    return pd.DataFrame(rows), cache


def attach_primary_ci(
    summary: pd.DataFrame,
    boot_unc: pd.DataFrame,
) -> pd.DataFrame:
    out = summary.copy()
    primary = boot_unc[boot_unc["bootstrap_unit"] == "genomic_cluster"].copy()

    for metric in BOOTSTRAP_METRICS:
        s = primary[primary["metric"] == metric][
            ["scheme", "representation", "model", "ci95_low", "ci95_high"]
        ].rename(
            columns={
                "ci95_low": f"{metric}_cluster_boot_ci95_low",
                "ci95_high": f"{metric}_cluster_boot_ci95_high",
            }
        )
        out = out.merge(
            s,
            on=["scheme", "representation", "model"],
            how="left",
            validate="one_to_one",
        )
    return out


def incremental_value_table(
    fold_metrics: pd.DataFrame,
    boot_cache: Dict[Tuple[str, str, str, str, str], np.ndarray],
    units: Iterable[str],
    schemes: Iterable[str],
) -> pd.DataFrame:
    baseline_rep = "known_amr"
    candidates = list(INCREMENTAL_REPRESENTATIONS)
    metrics = list(BOOTSTRAP_METRICS)
    rows = []

    for scheme in schemes:
        base_fold = fold_metrics[
            (fold_metrics["scheme"] == scheme)
            & (fold_metrics["representation"] == baseline_rep)
            & (fold_metrics["model"] == "logreg_l2")
        ].sort_values("fold")

        for candidate in candidates:
            cand_fold = fold_metrics[
                (fold_metrics["scheme"] == scheme)
                & (fold_metrics["representation"] == candidate)
                & (fold_metrics["model"] == "logreg_l2")
            ].sort_values("fold")

            if list(base_fold["fold"]) != list(cand_fold["fold"]):
                raise RuntimeError("Incremental comparison fold alignment failed.")

            for metric in metrics:
                delta_fold = (
                    cand_fold[metric].to_numpy(float)
                    - base_fold[metric].to_numpy(float)
                )

                for unit in units:
                    cand_boot = boot_cache[
                        (unit, scheme, candidate, "logreg_l2", metric)
                    ]
                    base_boot = boot_cache[
                        (unit, scheme, baseline_rep, "logreg_l2", metric)
                    ]
                    delta_boot = cand_boot - base_boot
                    finite = delta_boot[np.isfinite(delta_boot)]

                    rows.append(
                        {
                            "bootstrap_unit": unit,
                            "scheme": scheme,
                            "baseline_representation": baseline_rep,
                            "candidate_representation": candidate,
                            "model": "logreg_l2",
                            "metric": metric,
                            "mean_fold_delta_candidate_minus_baseline": float(
                                np.mean(delta_fold)
                            ),
                            "sd_fold_delta": (
                                float(np.std(delta_fold, ddof=1))
                                if len(delta_fold) > 1 else float("nan")
                            ),
                            "min_fold_delta": float(np.min(delta_fold)),
                            "max_fold_delta": float(np.max(delta_fold)),
                            "folds_candidate_higher": int(np.sum(delta_fold > 0)),
                            "folds_equal": int(np.sum(delta_fold == 0)),
                            "folds_candidate_lower": int(np.sum(delta_fold < 0)),
                            "bootstrap_delta_ci95_low": float(
                                np.quantile(finite, 0.025)
                            ),
                            "bootstrap_delta_ci95_high": float(
                                np.quantile(finite, 0.975)
                            ),
                            "n_boot_valid": len(finite),
                            "interpretation": (
                                "Positive delta means candidate representation "
                                "outperformed known-AMR under the same fixed "
                                "logistic-regression model."
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def generalization_gap_table(
    summary: pd.DataFrame,
    boot_cache: Dict[Tuple[str, str, str, str, str], np.ndarray],
    units: Iterable[str],
) -> pd.DataFrame:
    rows = []
    base_scheme = "random_stratified"
    target_schemes = ("genomic_cluster_aware", "mlst_aware")

    combos = (
        summary[["representation", "model"]]
        .drop_duplicates()
        .sort_values(["representation", "model"])
    )

    for rep, model in combos.itertuples(index=False, name=None):
        random_row = summary[
            (summary["scheme"] == base_scheme)
            & (summary["representation"] == rep)
            & (summary["model"] == model)
        ]
        if len(random_row) != 1:
            continue

        for target_scheme in target_schemes:
            target_row = summary[
                (summary["scheme"] == target_scheme)
                & (summary["representation"] == rep)
                & (summary["model"] == model)
            ]
            if len(target_row) != 1:
                continue

            for metric in BOOTSTRAP_METRICS:
                point = float(
                    target_row.iloc[0][f"{metric}_mean"]
                    - random_row.iloc[0][f"{metric}_mean"]
                )
                for unit in units:
                    target_boot = boot_cache[
                        (unit, target_scheme, rep, model, metric)
                    ]
                    random_boot = boot_cache[
                        (unit, base_scheme, rep, model, metric)
                    ]
                    delta = target_boot - random_boot
                    finite = delta[np.isfinite(delta)]
                    rows.append(
                        {
                            "bootstrap_unit": unit,
                            "representation": rep,
                            "model": model,
                            "metric": metric,
                            "reference_scheme": base_scheme,
                            "lineage_aware_scheme": target_scheme,
                            "macro_fold_delta_lineage_minus_random": point,
                            "bootstrap_delta_ci95_low": float(
                                np.quantile(finite, 0.025)
                            ),
                            "bootstrap_delta_ci95_high": float(
                                np.quantile(finite, 0.975)
                            ),
                            "n_boot_valid": len(finite),
                            "interpretation": (
                                "Negative delta indicates lower performance under "
                                "the lineage-aware validation design."
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def feature_selection_stability(
    fs: pd.DataFrame,
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stab_rows = []
    freq_rows = []

    for (scheme, rep), g in fs.groupby(
        ["scheme", "representation"], sort=True
    ):
        fold_sets = {}
        for fold, gf in g.groupby("fold"):
            fold_sets[int(fold)] = set(gf["feature_index"].astype(int))

        pairwise = []
        for a, b in combinations(sorted(fold_sets), 2):
            A, B = fold_sets[a], fold_sets[b]
            union = len(A | B)
            j = len(A & B) / union if union else float("nan")
            pairwise.append(j)

        all_features = g.groupby("feature_index").agg(
            selection_count=("fold", "nunique"),
            mean_train_only_chi2=("chi2_score_train_only", "mean"),
            max_train_only_chi2=("chi2_score_train_only", "max"),
        ).reset_index()

        counts = (
            all_features["selection_count"].value_counts().sort_index().to_dict()
        )
        stable5 = int((all_features["selection_count"] == 5).sum())
        stable4 = int((all_features["selection_count"] >= 4).sum())

        stab_rows.append(
            {
                "scheme": scheme,
                "representation": rep,
                "n_folds": len(fold_sets),
                "mean_pairwise_jaccard": float(np.mean(pairwise)),
                "sd_pairwise_jaccard": float(np.std(pairwise, ddof=1)),
                "min_pairwise_jaccard": float(np.min(pairwise)),
                "max_pairwise_jaccard": float(np.max(pairwise)),
                "unique_features_selected_across_folds": len(all_features),
                "selected_in_all_5_folds": stable5,
                "selected_in_at_least_4_folds": stable4,
                "selected_once": int(counts.get(1, 0)),
                "selected_twice": int(counts.get(2, 0)),
                "selected_three_times": int(counts.get(3, 0)),
                "selected_four_times": int(counts.get(4, 0)),
                "selected_five_times": int(counts.get(5, 0)),
            }
        )

        for row in all_features.itertuples(index=False):
            freq_rows.append(
                {
                    "scheme": scheme,
                    "representation": rep,
                    "feature_index": int(row.feature_index),
                    "selection_count": int(row.selection_count),
                    "mean_train_only_chi2": float(row.mean_train_only_chi2),
                    "max_train_only_chi2": float(row.max_train_only_chi2),
                }
            )

    logger(
        f"Feature-selection stability computed: "
        f"{len(stab_rows)} summaries, {len(freq_rows):,} unique feature rows."
    )
    return pd.DataFrame(stab_rows), pd.DataFrame(freq_rows)


def lineage_subgroup_performance(
    pred: pd.DataFrame,
    folds: pd.DataFrame,
    min_lineage_n: int,
    logger: Logger,
) -> pd.DataFrame:
    meta = folds[["Genome ID", "MLST", "y"]].copy()
    meta["Genome ID"] = meta["Genome ID"].astype(str)

    groups = []
    is_exact = []
    for gid, st in zip(meta["Genome ID"], meta["MLST"]):
        group, exact = exact_mlst_group(st, gid)
        groups.append(group if exact else None)
        is_exact.append(exact)
    meta["exact_mlst"] = groups

    exact = meta[meta["exact_mlst"].notna()].copy()
    lineage_stats = exact.groupby("exact_mlst").agg(
        lineage_n=("Genome ID", "size"),
        resistant_n=("y", "sum"),
    )
    lineage_stats["susceptible_n"] = (
        lineage_stats["lineage_n"] - lineage_stats["resistant_n"]
    )

    keep = lineage_stats[
        (lineage_stats["lineage_n"] >= min_lineage_n)
        & (lineage_stats["resistant_n"] > 0)
        & (lineage_stats["susceptible_n"] > 0)
    ].index

    gid_to_lineage = meta.set_index("Genome ID")["exact_mlst"]
    p = pred.copy()
    p["exact_mlst"] = p["Genome ID"].map(gid_to_lineage)
    p = p[p["exact_mlst"].isin(keep)].copy()

    rows = []
    for key, g in p.groupby(
        ["scheme", "representation", "model", "exact_mlst"], sort=True
    ):
        scheme, rep, model, lineage = key
        y = g["y"].to_numpy(dtype=np.uint8)
        pr = g["pred"].to_numpy(dtype=np.uint8)
        tn, fp, fn, tp = confusion_counts(y, pr)
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        ba = (
            float((sens + spec) / 2)
            if np.isfinite(sens) and np.isfinite(spec)
            else float("nan")
        )
        mcc = (
            float(matthews_corrcoef(y, pr))
            if len(np.unique(y)) == 2 else float("nan")
        )
        rows.append(
            {
                "scheme": scheme,
                "representation": rep,
                "model": model,
                "MLST_group": lineage,
                "n": len(g),
                "resistant_n": int(y.sum()),
                "susceptible_n": int((1 - y).sum()),
                "accuracy": float(accuracy_score(y, pr)),
                "balanced_accuracy": ba,
                "sensitivity": sens,
                "specificity": spec,
                "f1": float(f1_score(y, pr, zero_division=0)),
                "mcc": mcc,
                "errors": int(np.sum(y != pr)),
                "note": (
                    "Threshold-based subgroup metrics only; score-based AUROC/AP "
                    "are intentionally omitted to avoid cross-fold score-scale pooling."
                ),
            }
        )

    logger(
        f"Major-lineage subgroup analysis: {len(keep)} eligible exact MLST groups; "
        f"{len(rows):,} output rows."
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Parallel bootstrap worker processes (default 8). "
            "Each worker is capped to one inner BLAS/OpenMP thread."
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Cluster-bootstrap replicates for FULL mode (default 1000).",
    )
    parser.add_argument(
        "--min-lineage-n",
        type=int,
        default=30,
        help="Minimum exact-MLST sample size for subgroup reporting.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Evaluate only random_stratified logistic-regression OOF combinations "
            "with 50 bootstrap replicates and separate outputs/checkpoints."
        ),
    )
    args = parser.parse_args()

    if not (1 <= args.workers <= 12):
        raise SystemExit("--workers must be between 1 and 12.")
    if args.bootstrap_replicates < 100 and not args.smoke:
        raise SystemExit("Use at least 100 bootstrap replicates in FULL mode.")
    if args.min_lineage_n < 10:
        raise SystemExit("--min-lineage-n must be >=10.")

    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else find_project_root(Path.cwd())
    )

    if args.smoke:
        schemes = ("random_stratified",)
        n_boot = 50
        namespace = "model_evaluation_smoke"
        out_dir = root / "data/evaluation/smoke"
    else:
        schemes = tuple(SCHEME_TO_COLUMN)
        n_boot = args.bootstrap_replicates
        namespace = "model_evaluation"
        out_dir = root / "data/evaluation"

    paths = {
        "pred": root / "data/modeling/file08_oof_predictions.csv",
        "tasks": root / "data/modeling/file08_task_manifest.csv",
        "fs": root / "data/modeling/file08_feature_selection_manifest.csv",
        "folds": root / "data/splits/file07_outer_folds.csv",
        "file08_summary": root / "checkpoints/model_training/file08_final_summary.json",
    }

    ckpt_dir = root / "checkpoints" / namespace
    boot_dir = ckpt_dir / "bootstrap"
    log_path = root / "logs" / (
        "09_model_evaluation_smoke.log" if args.smoke
        else "09_model_evaluation.log"
    )
    failure_path = ckpt_dir / "09_last_failure.json"
    final_summary_path = ckpt_dir / "file09_final_summary.json"
    lock_path = ckpt_dir / "09_model_evaluation.lock"

    outputs = {
        "fold_metrics": out_dir / "file09_fold_metrics.csv",
        "pooled": out_dir / "file09_pooled_oof_diagnostics.csv",
        "performance": out_dir / "file09_performance_summary.csv",
        "bootstrap": out_dir / "file09_bootstrap_uncertainty.csv",
        "null_fold": out_dir / "file09_null_baseline_fold_metrics.csv",
        "null_summary": out_dir / "file09_null_baseline_summary.csv",
        "incremental": out_dir / "file09_incremental_value.csv",
        "gap": out_dir / "file09_generalization_gap.csv",
        "stability": out_dir / "file09_feature_selection_stability.csv",
        "frequency": out_dir / "file09_feature_selection_frequency.csv",
        "lineage": out_dir / "file09_lineage_subgroup_performance.csv",
        "audit": out_dir / "file09_integrity_audit.csv",
        "design": out_dir / "file09_evaluation_design.json",
    }

    logger = Logger(log_path)
    stage = "startup"
    started = time.time()

    acquire_lock(lock_path)

    try:
        logger(f"File09 script version: {SCRIPT_VERSION}")
        logger(f"Mode: {'SMOKE' if args.smoke else 'FULL'}")
        logger(f"Project root: {root}")
        logger(f"Bootstrap replicates: {n_boot:,}")
        logger(f"Bootstrap workers: {args.workers}")
        logger(f"Major-lineage minimum n: {args.min_lineage_n}")

        for name, path in paths.items():
            if not path.is_file():
                raise RuntimeError(f"Required input missing ({name}): {path}")
            logger(f"Input OK: {path.relative_to(root)}")

        stage = "load_inputs"
        pred = pd.read_csv(paths["pred"], dtype={"Genome ID": str})
        tasks = pd.read_csv(paths["tasks"])
        fs = pd.read_csv(paths["fs"])
        folds = pd.read_csv(paths["folds"], dtype={"Genome ID": str})
        with paths["file08_summary"].open("r", encoding="utf-8") as fh:
            file08_summary = json.load(fh)

        folds["y"] = pd.to_numeric(folds["y"], errors="raise").astype(int)
        for col in SCHEME_TO_COLUMN.values():
            folds[col] = pd.to_numeric(folds[col], errors="raise").astype(int)

        pred["y"] = pd.to_numeric(pred["y"], errors="raise").astype(int)
        pred["fold"] = pd.to_numeric(pred["fold"], errors="raise").astype(int)
        pred["pred"] = pd.to_numeric(pred["pred"], errors="raise").astype(int)
        pred["score"] = pd.to_numeric(pred["score"], errors="raise").astype(float)

        if args.smoke:
            pred = pred[
                (pred["scheme"] == "random_stratified")
                & (pred["model"] == "logreg_l2")
            ].copy()
            fs = fs[fs["scheme"] == "random_stratified"].copy()

        stage = "integrity_audit"
        audit_df = audit_inputs(
            pred, tasks, fs, folds, file08_summary, schemes, args.smoke, logger
        )
        atomic_write_csv(outputs["audit"], audit_df)

        design = {
            "script_version": SCRIPT_VERSION,
            "created_utc": utc_now(),
            "mode": "SMOKE" if args.smoke else "FULL",
            "primary_evaluation_unit": "outer_fold",
            "outer_folds_per_scheme": 5,
            "score_based_primary_metrics": ["roc_auc", "average_precision"],
            "threshold_based_primary_metrics": [
                "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc"
            ],
            "secondary_metrics": ["accuracy", "ppv", "npv"],
            "probability_only_metrics": ["brier", "log_loss"],
            "threshold_optimization_on_outer_test": False,
            "model_selection_on_outer_test": False,
            "pooled_oof_discrimination_role": "descriptive_only",
            "uncertainty_method": (
                "global population-structure cluster bootstrap with macro-fold estimand"
            ),
            "bootstrap_units": ["genomic_cluster", "mlst_group"],
            "bootstrap_replicates": n_boot,
            "bootstrap_workers": args.workers,
            "inner_threads_per_worker": 1,
            "primary_bootstrap_unit": "genomic_cluster",
            "paired_incremental_baseline_model": "logreg_l2",
            "incremental_baseline_representation": "known_amr",
            "generalization_reference": "random_stratified",
            "lineage_subgroup_min_n": args.min_lineage_n,
            "lineage_subgroup_score_metrics_omitted": True,
            "reason_no_naive_iid_pvalues": (
                "Related bacterial isolates violate a simple IID-genome assumption; "
                "cluster-bootstrap confidence intervals are used instead."
            ),
            "reason_no_delong": (
                "Outer-fold models and structured samples complicate the assumptions "
                "of naive paired AUROC tests; paired cluster bootstrap is used."
            ),
            "bootstrap_undefined_draw_policy": (
                "Cluster-bootstrap draws are excluded only when an outer test fold "
                "contains effectively one phenotype class, making AUROC/AP undefined; "
                "valid counts and fractions are reported, with >=80% validity required."
            ),
        }
        atomic_write_text(outputs["design"], json.dumps(design, indent=2) + "\n")

        stage = "metrics"
        fold_metrics = compute_fold_metrics(pred, logger)
        pooled = compute_pooled_diagnostics(pred, logger)
        perf_summary = summarize_fold_metrics(fold_metrics)
        null_fold, null_summary = compute_null_baseline(folds, schemes, logger)

        atomic_write_csv(outputs["fold_metrics"], fold_metrics)
        atomic_write_csv(outputs["pooled"], pooled)
        atomic_write_csv(outputs["null_fold"], null_fold)
        atomic_write_csv(outputs["null_summary"], null_summary)

        stage = "bootstrap"
        units = build_bootstrap_units(folds)
        boot_unc, boot_cache = compute_bootstrap_uncertainty(
            pred,
            folds,
            units,
            n_boot,
            boot_dir,
            logger,
            args.workers,
        )
        atomic_write_csv(outputs["bootstrap"], boot_unc)

        perf_summary = attach_primary_ci(perf_summary, boot_unc)
        atomic_write_csv(outputs["performance"], perf_summary)

        stage = "contrasts"
        incremental = incremental_value_table(
            fold_metrics,
            boot_cache,
            units.keys(),
            schemes,
        )
        generalization = generalization_gap_table(
            perf_summary,
            boot_cache,
            units.keys(),
        )
        atomic_write_csv(outputs["incremental"], incremental)
        atomic_write_csv(outputs["gap"], generalization)

        stage = "feature_stability"
        stability, frequency = feature_selection_stability(fs, logger)
        atomic_write_csv(outputs["stability"], stability)
        atomic_write_csv(outputs["frequency"], frequency)

        stage = "lineage_subgroups"
        lineage = lineage_subgroup_performance(
            pred, folds, args.min_lineage_n, logger
        )
        atomic_write_csv(outputs["lineage"], lineage)

        stage = "final_audit"
        if not args.smoke:
            if len(fold_metrics) != 225:
                raise RuntimeError(
                    f"Expected 225 fold-metric rows, found {len(fold_metrics)}."
                )
            if len(perf_summary) != 45:
                raise RuntimeError(
                    f"Expected 45 performance-summary rows, found {len(perf_summary)}."
                )
            if len(pooled) != 45:
                raise RuntimeError(
                    f"Expected 45 pooled-diagnostic rows, found {len(pooled)}."
                )
            if len(null_fold) != 15:
                raise RuntimeError(
                    f"Expected 15 null-baseline fold rows, found {len(null_fold)}."
                )

        summary = {
            "script_version": SCRIPT_VERSION,
            "status": "PASS",
            "mode": "SMOKE" if args.smoke else "FULL",
            "completed_utc": utc_now(),
            "cohort_n": EXPECTED_COHORT_N,
            "validation_schemes": list(schemes),
            "oof_combinations_evaluated": int(
                pred[["scheme", "representation", "model"]]
                .drop_duplicates()
                .shape[0]
            ),
            "fold_metric_rows": len(fold_metrics),
            "performance_summary_rows": len(perf_summary),
            "bootstrap_uncertainty_rows": len(boot_unc),
            "null_baseline_fold_rows": len(null_fold),
            "incremental_contrast_rows": len(incremental),
            "generalization_gap_rows": len(generalization),
            "feature_stability_rows": len(stability),
            "lineage_subgroup_rows": len(lineage),
            "bootstrap_replicates": n_boot,
            "bootstrap_workers": args.workers,
            "inner_threads_per_worker": 1,
            "bootstrap_units": list(units),
            "threshold_tuning_on_outer_test": False,
            "posthoc_model_retraining": False,
            "elapsed_seconds": time.time() - started,
            "outputs": {k: str(v.relative_to(root)) for k, v in outputs.items()},
        }
        atomic_write_text(
            final_summary_path, json.dumps(summary, indent=2) + "\n"
        )

        if failure_path.exists():
            failure_path.unlink()

        logger("=" * 78)
        logger("FILE09 — MODEL EVALUATION AND ROBUSTNESS")
        logger("=" * 78)
        logger(f"Mode                           : {'SMOKE' if args.smoke else 'FULL'}")
        logger(f"Cohort                         : {EXPECTED_COHORT_N:,}")
        logger(f"OOF combinations evaluated     : {summary['oof_combinations_evaluated']}")
        logger(f"Fold metric rows               : {len(fold_metrics):,}")
        logger(f"Performance summary rows       : {len(perf_summary):,}")
        logger(f"Bootstrap units                : {', '.join(units)}")
        logger(f"Bootstrap replicates/unit      : {n_boot:,}")
        logger(f"Parallel bootstrap workers     : {args.workers}")
        logger("Inner threads / worker         : 1")
        logger("Outer-test threshold tuning    : NO")
        logger("Post-hoc model retraining      : NO")
        logger("IID genome-level p-values      : NO")
        logger(f"Elapsed                        : {human_seconds(time.time()-started)}")
        logger(f"Final summary                  : {final_summary_path}")
        logger("FILE09 STATUS                  : PASS")
        logger("=" * 78)
        return 0

    except KeyboardInterrupt:
        failure = {
            "script_version": SCRIPT_VERSION,
            "status": "INTERRUPTED",
            "stage": stage,
            "time_utc": utc_now(),
            "message": (
                "Interrupted. Completed bootstrap-combination checkpoints are "
                "preserved. Re-run the same command to resume."
            ),
        }
        atomic_write_text(failure_path, json.dumps(failure, indent=2) + "\n")
        logger(
            "FILE09 INTERRUPTED. Completed checkpoints are preserved; "
            "re-run the same command to resume."
        )
        return 130

    except Exception as exc:
        failure = {
            "script_version": SCRIPT_VERSION,
            "status": "FAILED",
            "stage": stage,
            "time_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_text(failure_path, json.dumps(failure, indent=2) + "\n")
        logger("")
        logger("FILE09 FAILED")
        logger(f"Stage : {stage}")
        logger(f"Error : {exc}")
        logger(f"Crash record: {failure_path}")
        logger(
            "Fix the cause and re-run the same command; validated bootstrap "
            "checkpoints will be reused."
        )
        return 1

    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    raise SystemExit(main())
