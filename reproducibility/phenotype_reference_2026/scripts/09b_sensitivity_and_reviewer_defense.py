#!/usr/bin/env python3
"""
09b_sensitivity_and_reviewer_defense.py
=======================================

PURPOSE
-------
Reviewer-defense sensitivity analysis layered on top of locked File08/File09.

This script DOES NOT replace the primary File08/File09 analysis.  It addresses
the most plausible methodological objections to the conclusion that broader
genomic representations do not add robust predictive value beyond the curated
known-AMR baseline.

SCIENTIFIC QUESTIONS ADDRESSED
------------------------------
1) Was the incremental-result conclusion an artifact of C=1?
   -> Sweep L2 logistic C = {0.01, 0.1, 1, 10}.

2) Was it an artifact of selecting exactly 2,000 genomic features?
   -> Sweep train-only top-k = {500, 1,000, 2,000} using the LOCKED File08
      outer-training-fold rankings. No new test-label feature selection occurs.

3) Was it an artifact of not weighting the modestly imbalanced phenotype?
   -> Compare class_weight=None and class_weight="balanced".

4) Is the broader-vs-known-AMR conclusion model-family dependent?
   -> Re-analyze the locked File09 fold results across all four model families
      using matched model-family comparisons.

5) Did unresolved MLST rows drive the conclusions?
   -> Re-evaluate locked File08 OOF predictions on exact-MLST rows only.

6) Are aggregate results dominated by one or two very large sequence types?
   -> Cumulative major-lineage exclusion sensitivity for the top 1, 2, and 5
      exact MLST groups, reporting how macro-fold metrics change.

ANTI-LEAKAGE / ANTI-CHERRY-PICKING RULES
----------------------------------------
- Outer folds are read from locked File07 and never changed.
- The File08 train-only feature ranking is reused exactly.
- Top-500 and top-1000 are prefixes of the train-only top-2000 ranking.
- No feature ranking is recomputed using outer-test labels.
- ALL C x class-weight x k settings are reported.
- No "best" hyperparameter configuration is selected from outer-test results.
- Matched incremental comparisons always use the SAME C and class weighting
  for candidate and known-AMR baseline.
- This is a sensitivity analysis, not a second model-selection stage.

PRIMARY GRID
------------
C:
    0.01, 0.1, 1, 10

class_weight:
    none, balanced

k for broader genomic representations:
    500, 1000, 2000

representations:
    known_amr
    pangenome
    unitig
    known_plus_pangenome
    known_plus_unitig
    known_plus_pangenome_plus_unitig

Full configuration count per validation scheme:
    known-AMR: 4 C x 2 weights = 8
    broader : 5 reps x 3 k x 4 C x 2 weights = 120
    total   : 128 OOF configurations / scheme

Three validation schemes:
    384 complete OOF sensitivity configurations total.

OPERATIONAL DESIGN
------------------
- 8 process workers by default.
- Each worker handles one complete outer fold, materializes the maximum top-k
  matrices once, then reuses prefixes for smaller k values.
- One checkpoint per outer fold.
- Crash-safe / resumable.
- BLAS/OpenMP threads capped to one per worker to avoid oversubscription.

SMOKE
-----
    python scripts/09b_sensitivity_and_reviewer_defense.py --smoke

FULL
----
    python scripts/09b_sensitivity_and_reviewer_defense.py --workers 8

OUTPUTS
-------
data/evaluation/reviewer_defense/
    file09b_sensitivity_oof_predictions.csv.gz
    file09b_sensitivity_fold_metrics.csv
    file09b_sensitivity_performance_summary.csv
    file09b_matched_incremental_deltas.csv
    file09b_incremental_robustness_summary.csv
    file09b_model_family_consistency.csv
    file09b_exact_mlst_subset_summary.csv
    file09b_major_lineage_exclusion_sensitivity.csv
    file09b_integrity_audit.csv
    file09b_analysis_design.json

checkpoints/model_evaluation_reviewer_defense/
    folds/*.csv.gz
    file09b_final_summary.json
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
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

SCRIPT_VERSION = "1.0.0"
RANDOM_SEED = 20260921

EXPECTED_COHORT_N = 4227
EXPECTED_KNOWN_FEATURES = 668
EXPECTED_PANGENOME_FEATURES = 22685
EXPECTED_UNITIG_FEATURES = 4718475
UNITIG_BYTES_PER_FEATURE = 529
EXPECTED_UNITIG_BYTES = 2496073275

SCHEME_TO_COLUMN = {
    "random_stratified": "random_fold",
    "genomic_cluster_aware": "genomic_cluster_fold",
    "mlst_aware": "mlst_fold",
}

BROADER_REPS = (
    "pangenome",
    "unitig",
    "known_plus_pangenome",
    "known_plus_unitig",
    "known_plus_pangenome_plus_unitig",
)

PRIMARY_GRID_K = (500, 1000, 2000)
PRIMARY_GRID_C = (0.01, 0.1, 1.0, 10.0)
PRIMARY_CLASS_WEIGHTS = ("none", "balanced")

PRIMARY_METRICS = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "mcc",
    "sensitivity",
    "specificity",
    "f1",
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


def signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    return start


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


def atomic_write_csv_gz(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp.csv.gz", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_csv(tmp_path, index=False, compression="gzip")
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


def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"Lock exists: {lock_path}\n"
            "Another File09b process may still be running. Inspect the old "
            "process before removing the lock."
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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def metric_bundle(
    y: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.uint8)

    if len(np.unique(y)) < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "mcc": float("nan"),
            "n": int(len(y)),
            "n_resistant": int(y.sum()),
            "n_susceptible": int((1 - y).sum()),
        }

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "n": int(len(y)),
        "n_resistant": int(y.sum()),
        "n_susceptible": int((1 - y).sum()),
    }


def normalize_c_label(c: float) -> str:
    return f"{c:g}"


def exact_mlst(st) -> bool:
    if pd.isna(st):
        return False
    s = str(st).strip()
    return bool(s) and s not in {"-", "NA", "NaN", "nan", "None", "none"}


def load_folds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Genome ID": str})
    required = {
        "Genome ID", "y", "MLST", "Genomic Cluster",
        "random_fold", "genomic_cluster_fold", "mlst_fold",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"File07 fold file missing columns: {sorted(missing)}")
    if len(df) != EXPECTED_COHORT_N or df["Genome ID"].nunique() != EXPECTED_COHORT_N:
        raise RuntimeError("Unexpected File07 cohort size or duplicate Genome IDs.")
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)
    for col in SCHEME_TO_COLUMN.values():
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
        if set(df[col].unique()) != {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{col} does not contain exactly folds 0..4.")
    return df


def load_known(path: Path, cohort_ids: List[str]) -> np.ndarray:
    df = pd.read_csv(path, dtype={"Genome ID": str})
    if "Genome ID" not in df.columns:
        raise RuntimeError("Known-AMR matrix lacks Genome ID.")
    features = [c for c in df.columns if c != "Genome ID"]
    if len(features) != EXPECTED_KNOWN_FEATURES:
        raise RuntimeError(
            f"Known-AMR feature count mismatch: {len(features)} != {EXPECTED_KNOWN_FEATURES}"
        )
    if "emrD" in features:
        raise RuntimeError("emrD unexpectedly present in final known-AMR matrix.")
    indexed = df.set_index("Genome ID")
    missing = [x for x in cohort_ids if x not in indexed.index]
    if missing:
        raise RuntimeError(f"Known-AMR missing cohort IDs: {missing[:10]}")
    X = indexed.loc[cohort_ids, features].to_numpy(dtype=np.uint8, copy=True)
    if not np.all(np.isin(np.unique(X), [0, 1])):
        raise RuntimeError("Known-AMR matrix is not binary.")
    return X


def load_pangenome(
    npz_path: Path,
    row_index_path: Path,
    cohort_ids: List[str],
) -> sparse.csr_matrix:
    X = load_npz(npz_path).tocsr()
    rows = pd.read_csv(row_index_path, dtype=str)
    if X.shape[1] != EXPECTED_PANGENOME_FEATURES:
        raise RuntimeError(
            f"Pangenome feature count mismatch: {X.shape[1]} != "
            f"{EXPECTED_PANGENOME_FEATURES}"
        )
    ids = rows["Genome ID"].astype(str).tolist()
    if len(ids) != X.shape[0] or len(set(ids)) != len(ids):
        raise RuntimeError("Pangenome row index invalid.")
    if ids == cohort_ids:
        return X
    pos = {gid: i for i, gid in enumerate(ids)}
    missing = [gid for gid in cohort_ids if gid not in pos]
    if missing:
        raise RuntimeError(f"Pangenome missing cohort IDs: {missing[:10]}")
    return X[[pos[gid] for gid in cohort_ids], :].tocsr()


def decode_unitigs(
    bin_path: Path,
    feature_indices: np.ndarray,
) -> np.ndarray:
    if bin_path.stat().st_size != EXPECTED_UNITIG_BYTES:
        raise RuntimeError("Unitig bit-packed file size mismatch.")
    idx = np.asarray(feature_indices, dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= EXPECTED_UNITIG_FEATURES):
        raise RuntimeError("Unitig feature index out of range.")
    mm = np.memmap(bin_path, mode="r", dtype=np.uint8)
    rows = np.asarray(
        mm.reshape(EXPECTED_UNITIG_FEATURES, UNITIG_BYTES_PER_FEATURE)[idx]
    )
    bits = np.unpackbits(
        rows, axis=1, bitorder="little"
    )[:, :EXPECTED_COHORT_N]
    return np.ascontiguousarray(bits.T, dtype=np.uint8)


def selection_indices(
    fs: pd.DataFrame,
    scheme: str,
    fold: int,
    representation: str,
) -> np.ndarray:
    g = fs[
        (fs["scheme"] == scheme)
        & (pd.to_numeric(fs["fold"], errors="raise").astype(int) == fold)
        & (fs["representation"] == representation)
    ].copy()
    if len(g) != 2000:
        raise RuntimeError(
            f"Expected 2000 {representation} selections for {scheme} fold {fold}; "
            f"found {len(g)}."
        )
    g["rank"] = pd.to_numeric(g["rank"], errors="raise").astype(int)
    g["feature_index"] = pd.to_numeric(
        g["feature_index"], errors="raise"
    ).astype(int)
    g = g.sort_values("rank")
    if g["rank"].tolist() != list(range(1, 2001)):
        raise RuntimeError(
            f"Selection ranks are not exactly 1..2000 for "
            f"{scheme} fold {fold} {representation}."
        )
    if g["feature_index"].duplicated().any():
        raise RuntimeError(
            f"Duplicate selected feature index for {scheme} fold {fold} "
            f"{representation}."
        )
    return g["feature_index"].to_numpy(dtype=np.int64)


def make_logreg(c: float, class_weight_label: str) -> LogisticRegression:
    cw = None if class_weight_label == "none" else "balanced"
    return LogisticRegression(
        C=float(c),
        solver="liblinear",
        class_weight=cw,
        max_iter=3000,
        random_state=RANDOM_SEED,
    )


def expected_config_count(
    k_values: Tuple[int, ...],
    c_values: Tuple[float, ...],
    class_weights: Tuple[str, ...],
) -> int:
    return len(c_values) * len(class_weights) * (
        1 + len(BROADER_REPS) * len(k_values)
    )


def fold_checkpoint_valid(
    path: Path,
    expected_test_n: int,
    expected_configs: int,
) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, compression="gzip", dtype={"Genome ID": str})
        if len(df) != expected_test_n * expected_configs:
            return False
        if df["config_id"].nunique() != expected_configs:
            return False
        counts = df.groupby("config_id").size()
        if not bool((counts == expected_test_n).all()):
            return False
        if not np.isfinite(pd.to_numeric(df["score"], errors="coerce")).all():
            return False
        return True
    except Exception:
        return False


def run_outer_fold_job(
    root_str: str,
    namespace: str,
    scheme: str,
    fold: int,
    k_values: Tuple[int, ...],
    c_values: Tuple[float, ...],
    class_weights: Tuple[str, ...],
) -> Dict[str, object]:
    """
    Independent fold worker. Loads data locally to avoid sending large matrices
    through inter-process serialization.
    """
    root = Path(root_str)
    t0 = time.time()

    folds = load_folds(root / "data/splits/file07_outer_folds.csv")
    cohort_ids = folds["Genome ID"].astype(str).tolist()
    y = folds["y"].to_numpy(dtype=np.uint8)
    fold_col = SCHEME_TO_COLUMN[scheme]
    fvals = folds[fold_col].to_numpy(dtype=int)
    train_idx = np.flatnonzero(fvals != fold)
    test_idx = np.flatnonzero(fvals == fold)

    expected_configs = expected_config_count(
        k_values, c_values, class_weights
    )
    ckpt = (
        root / "checkpoints" / namespace / "folds"
        / f"{scheme}__fold{fold}.csv.gz"
    )
    if fold_checkpoint_valid(ckpt, len(test_idx), expected_configs):
        return {
            "scheme": scheme,
            "fold": fold,
            "status": "SKIP",
            "rows": len(test_idx) * expected_configs,
            "configs": expected_configs,
            "elapsed_seconds": 0.0,
            "checkpoint": str(ckpt),
        }

    if STOP_REQUESTED:
        raise KeyboardInterrupt

    known = load_known(
        root / "data/features/known_amr/X_known_amr_final.csv",
        cohort_ids,
    )
    pang = load_pangenome(
        root / "data/features/pangenome/X_pangenome_plfam_binary.npz",
        root / "data/features/pangenome/X_pangenome_row_index.csv",
        cohort_ids,
    )
    fs = pd.read_csv(
        root / "data/modeling/file08_feature_selection_manifest.csv"
    )

    pidx = selection_indices(fs, scheme, fold, "pangenome")
    uidx = selection_indices(fs, scheme, fold, "unitig")

    Xp_max = np.ascontiguousarray(
        pang[:, pidx].toarray(), dtype=np.uint8
    )
    Xu_max = decode_unitigs(
        root / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin",
        uidx,
    )

    if Xp_max.shape != (EXPECTED_COHORT_N, 2000):
        raise RuntimeError("Pangenome selected matrix shape mismatch.")
    if Xu_max.shape != (EXPECTED_COHORT_N, 2000):
        raise RuntimeError("Unitig selected matrix shape mismatch.")

    out_frames = []

    def fit_one(
        representation: str,
        k_value: int | None,
        c_value: float,
        cw_label: str,
        X: np.ndarray,
    ) -> None:
        model = make_logreg(c_value, cw_label)
        Xtr = np.asarray(X[train_idx], dtype=np.float32, order="C")
        Xte = np.asarray(X[test_idx], dtype=np.float32, order="C")
        model.fit(Xtr, y[train_idx])
        score = model.predict_proba(Xte)[:, 1].astype(np.float64)
        predv = model.predict(Xte).astype(np.uint8)

        k_label = "NA" if k_value is None else str(int(k_value))
        config_id = (
            f"{representation}__k{k_label}__C{normalize_c_label(c_value)}"
            f"__cw{cw_label}"
        )
        out_frames.append(
            pd.DataFrame(
                {
                    "Genome ID": np.asarray(cohort_ids, dtype=object)[test_idx],
                    "y": y[test_idx].astype(int),
                    "scheme": scheme,
                    "fold": fold,
                    "representation": representation,
                    "k": -1 if k_value is None else int(k_value),
                    "C": float(c_value),
                    "class_weight": cw_label,
                    "config_id": config_id,
                    "score": score,
                    "pred": predv.astype(int),
                }
            )
        )

    # Matched known-AMR baseline for every C x class-weight setting.
    for c in c_values:
        for cw in class_weights:
            fit_one("known_amr", None, c, cw, known)

    # Broader genomic representations.
    for k in k_values:
        Xp = Xp_max[:, :k]
        Xu = Xu_max[:, :k]
        X_kp = np.concatenate([known, Xp], axis=1)
        X_ku = np.concatenate([known, Xu], axis=1)
        X_kpu = np.concatenate([known, Xp, Xu], axis=1)

        rep_mats = {
            "pangenome": Xp,
            "unitig": Xu,
            "known_plus_pangenome": X_kp,
            "known_plus_unitig": X_ku,
            "known_plus_pangenome_plus_unitig": X_kpu,
        }
        for rep, Xrep in rep_mats.items():
            for c in c_values:
                for cw in class_weights:
                    fit_one(rep, k, c, cw, Xrep)

    out = pd.concat(out_frames, ignore_index=True)
    if out["config_id"].nunique() != expected_configs:
        raise RuntimeError(
            f"Fold config count mismatch: {out['config_id'].nunique()} "
            f"!= {expected_configs}"
        )
    if len(out) != len(test_idx) * expected_configs:
        raise RuntimeError("Fold prediction row-count mismatch.")
    if out.duplicated(["Genome ID", "config_id"]).any():
        raise RuntimeError("Duplicate Genome ID/config within fold sensitivity output.")
    if not np.isfinite(out["score"]).all():
        raise RuntimeError("Non-finite sensitivity score produced.")

    atomic_write_csv_gz(ckpt, out)
    return {
        "scheme": scheme,
        "fold": fold,
        "status": "PASS",
        "rows": len(out),
        "configs": expected_configs,
        "elapsed_seconds": time.time() - t0,
        "checkpoint": str(ckpt),
    }


def compute_sensitivity_fold_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "scheme", "fold", "representation", "k",
        "C", "class_weight", "config_id",
    ]
    for key, g in pred.groupby(group_cols, sort=True):
        (
            scheme, fold, rep, k, c, cw, config_id
        ) = key
        m = metric_bundle(
            g["y"].to_numpy(dtype=np.uint8),
            g["score"].to_numpy(dtype=float),
            g["pred"].to_numpy(dtype=np.uint8),
        )
        m.update(
            {
                "scheme": scheme,
                "fold": int(fold),
                "representation": rep,
                "k": int(k),
                "C": float(c),
                "class_weight": cw,
                "config_id": config_id,
            }
        )
        rows.append(m)
    return pd.DataFrame(rows)


def summarize_sensitivity_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc", "average_precision", "accuracy",
        "balanced_accuracy", "mcc", "sensitivity",
        "specificity", "f1",
    ]
    rows = []
    group_cols = [
        "scheme", "representation", "k", "C", "class_weight", "config_id"
    ]
    for key, g in fold_metrics.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row["n_outer_folds"] = len(g)
        for metric in metric_cols:
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = (
                float(np.mean(vals)) if len(vals) else float("nan")
            )
            row[f"{metric}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            )
            row[f"{metric}_min"] = (
                float(np.min(vals)) if len(vals) else float("nan")
            )
            row[f"{metric}_max"] = (
                float(np.max(vals)) if len(vals) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def matched_incremental_deltas(
    fold_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = fold_metrics[
        fold_metrics["representation"] == "known_amr"
    ].copy()

    baseline_cols = [
        "scheme", "fold", "C", "class_weight", *PRIMARY_METRICS
    ]
    baseline = baseline[baseline_cols].rename(
        columns={m: f"baseline_{m}" for m in PRIMARY_METRICS}
    )

    cand = fold_metrics[
        fold_metrics["representation"] != "known_amr"
    ].copy()

    merged = cand.merge(
        baseline,
        on=["scheme", "fold", "C", "class_weight"],
        how="left",
        validate="many_to_one",
    )
    if merged[[f"baseline_{m}" for m in PRIMARY_METRICS]].isna().all(axis=1).any():
        raise RuntimeError("Matched known-AMR baseline missing for sensitivity row.")

    long_rows = []
    for row in merged.itertuples(index=False):
        for metric in PRIMARY_METRICS:
            cand_val = float(getattr(row, metric))
            base_val = float(getattr(row, f"baseline_{metric}"))
            long_rows.append(
                {
                    "scheme": row.scheme,
                    "fold": int(row.fold),
                    "candidate_representation": row.representation,
                    "k": int(row.k),
                    "C": float(row.C),
                    "class_weight": row.class_weight,
                    "metric": metric,
                    "candidate_value": cand_val,
                    "matched_known_amr_value": base_val,
                    "delta_candidate_minus_known_amr": cand_val - base_val,
                }
            )
    delta = pd.DataFrame(long_rows)

    summary_rows = []

    def add_summary(g, grouping: dict):
        vals = g["delta_candidate_minus_known_amr"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        summary_rows.append(
            {
                **grouping,
                "n_matched_fold_hyperparameter_comparisons": len(vals),
                "mean_delta": float(np.mean(vals)),
                "median_delta": float(np.median(vals)),
                "min_delta": float(np.min(vals)),
                "max_delta": float(np.max(vals)),
                "fraction_candidate_higher": float(np.mean(vals > 0)),
                "fraction_equal": float(np.mean(vals == 0)),
                "fraction_candidate_lower": float(np.mean(vals < 0)),
                "all_comparisons_candidate_leq_baseline": bool(np.all(vals <= 0)),
            }
        )

    # k-specific robustness.
    for key, g in delta.groupby(
        ["scheme", "candidate_representation", "k", "metric"],
        sort=True,
    ):
        scheme, rep, k, metric = key
        add_summary(
            g,
            {
                "scheme": scheme,
                "candidate_representation": rep,
                "k_scope": str(int(k)),
                "metric": metric,
            },
        )

    # Across all tested k values.
    for key, g in delta.groupby(
        ["scheme", "candidate_representation", "metric"],
        sort=True,
    ):
        scheme, rep, metric = key
        add_summary(
            g,
            {
                "scheme": scheme,
                "candidate_representation": rep,
                "k_scope": "ALL_500_1000_2000",
                "metric": metric,
            },
        )

    return delta, pd.DataFrame(summary_rows)


def model_family_consistency(
    file09_fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    candidates = (
        "pangenome", "unitig",
        "known_plus_pangenome", "known_plus_unitig",
        "known_plus_pangenome_plus_unitig",
    )
    rows = []

    for scheme in SCHEME_TO_COLUMN:
        for candidate in candidates:
            cmodels = sorted(
                file09_fold_metrics[
                    (file09_fold_metrics["scheme"] == scheme)
                    & (file09_fold_metrics["representation"] == candidate)
                ]["model"].unique()
            )
            for model in cmodels:
                cand = file09_fold_metrics[
                    (file09_fold_metrics["scheme"] == scheme)
                    & (file09_fold_metrics["representation"] == candidate)
                    & (file09_fold_metrics["model"] == model)
                ].sort_values("fold")
                base = file09_fold_metrics[
                    (file09_fold_metrics["scheme"] == scheme)
                    & (file09_fold_metrics["representation"] == "known_amr")
                    & (file09_fold_metrics["model"] == model)
                ].sort_values("fold")
                if len(cand) != 5 or len(base) != 5:
                    continue
                if not np.array_equal(
                    cand["fold"].to_numpy(), base["fold"].to_numpy()
                ):
                    raise RuntimeError("File09 model-family fold alignment failed.")

                for metric in PRIMARY_METRICS:
                    d = (
                        cand[metric].to_numpy(dtype=float)
                        - base[metric].to_numpy(dtype=float)
                    )
                    rows.append(
                        {
                            "scheme": scheme,
                            "candidate_representation": candidate,
                            "matched_model": model,
                            "metric": metric,
                            "mean_fold_delta_candidate_minus_known_amr": float(np.mean(d)),
                            "median_fold_delta": float(np.median(d)),
                            "min_fold_delta": float(np.min(d)),
                            "max_fold_delta": float(np.max(d)),
                            "folds_candidate_higher": int(np.sum(d > 0)),
                            "folds_candidate_equal": int(np.sum(d == 0)),
                            "folds_candidate_lower": int(np.sum(d < 0)),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_locked_predictions_on_subset(
    pred: pd.DataFrame,
    keep_ids: set[str],
    subset_name: str,
) -> pd.DataFrame:
    p = pred[pred["Genome ID"].astype(str).isin(keep_ids)].copy()
    fold_rows = []
    for key, g in p.groupby(
        ["scheme", "fold", "representation", "model"], sort=True
    ):
        scheme, fold, rep, model = key
        m = metric_bundle(
            g["y"].to_numpy(dtype=np.uint8),
            g["score"].to_numpy(dtype=float),
            g["pred"].to_numpy(dtype=np.uint8),
        )
        m.update(
            {
                "subset": subset_name,
                "scheme": scheme,
                "fold": int(fold),
                "representation": rep,
                "model": model,
            }
        )
        fold_rows.append(m)

    fdf = pd.DataFrame(fold_rows)
    summary_rows = []
    for key, g in fdf.groupby(
        ["subset", "scheme", "representation", "model"], sort=True
    ):
        subset, scheme, rep, model = key
        row = {
            "subset": subset,
            "scheme": scheme,
            "representation": rep,
            "model": model,
            "n_valid_folds_roc_auc": int(np.isfinite(g["roc_auc"]).sum()),
            "n_valid_folds_average_precision": int(
                np.isfinite(g["average_precision"]).sum()
            ),
            "subset_n_unique_genomes": int(
                p[
                    (p["scheme"] == scheme)
                    & (p["representation"] == rep)
                    & (p["model"] == model)
                ]["Genome ID"].nunique()
            ),
        }
        for metric in PRIMARY_METRICS:
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = (
                float(np.mean(vals)) if len(vals) else float("nan")
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def exact_mlst_subset_analysis(
    pred: pd.DataFrame,
    folds: pd.DataFrame,
    file09_perf: pd.DataFrame,
) -> pd.DataFrame:
    exact_ids = set(
        folds.loc[
            folds["MLST"].apply(exact_mlst), "Genome ID"
        ].astype(str)
    )
    out = summarize_locked_predictions_on_subset(
        pred, exact_ids, "exact_MLST_only"
    )

    base_cols = ["scheme", "representation", "model"] + [
        f"{m}_mean" for m in PRIMARY_METRICS
    ]
    base = file09_perf[base_cols].copy().rename(
        columns={f"{m}_mean": f"all_cohort_{m}_mean" for m in PRIMARY_METRICS}
    )
    out = out.merge(
        base,
        on=["scheme", "representation", "model"],
        how="left",
        validate="one_to_one",
    )
    for metric in PRIMARY_METRICS:
        out[f"delta_exact_mlst_minus_all_{metric}"] = (
            out[f"{metric}_mean"] - out[f"all_cohort_{metric}_mean"]
        )
    return out


def major_lineage_exclusion_analysis(
    pred: pd.DataFrame,
    folds: pd.DataFrame,
    file09_perf: pd.DataFrame,
) -> pd.DataFrame:
    exact = folds[folds["MLST"].apply(exact_mlst)].copy()
    counts = exact["MLST"].astype(str).value_counts()
    top = counts.index.tolist()

    scenarios = [
        ("exclude_top1_ST", top[:1]),
        ("exclude_top2_STs", top[:2]),
        ("exclude_top5_STs", top[:5]),
    ]

    base_cols = ["scheme", "representation", "model"] + [
        f"{m}_mean" for m in PRIMARY_METRICS
    ]
    base = file09_perf[base_cols].copy().rename(
        columns={f"{m}_mean": f"all_cohort_{m}_mean" for m in PRIMARY_METRICS}
    )

    gid_to_st = folds.set_index("Genome ID")["MLST"].astype(str)
    rows = []
    all_ids = set(folds["Genome ID"].astype(str))

    for scenario, excluded_sts in scenarios:
        excluded_ids = set(
            folds.loc[
                folds["MLST"].astype(str).isin(excluded_sts), "Genome ID"
            ].astype(str)
        )
        keep_ids = all_ids - excluded_ids
        s = summarize_locked_predictions_on_subset(
            pred, keep_ids, scenario
        )
        s["excluded_STs"] = ";".join(excluded_sts)
        s["excluded_genomes_n"] = len(excluded_ids)
        s["remaining_genomes_n"] = len(keep_ids)
        rows.append(s)

    out = pd.concat(rows, ignore_index=True)
    out = out.merge(
        base,
        on=["scheme", "representation", "model"],
        how="left",
        validate="many_to_one",
    )
    for metric in PRIMARY_METRICS:
        out[f"delta_exclusion_minus_all_{metric}"] = (
            out[f"{metric}_mean"] - out[f"all_cohort_{metric}_mean"]
        )
    return out


def integrity_audit(
    root: Path,
    folds: pd.DataFrame,
    fs: pd.DataFrame,
    file08_pred: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, detail: str):
        rows.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )
        if not passed:
            raise RuntimeError(f"Integrity audit failed: {name}: {detail}")

    add(
        "cohort_size",
        len(folds) == EXPECTED_COHORT_N
        and folds["Genome ID"].nunique() == EXPECTED_COHORT_N,
        f"rows={len(folds)}, unique_ids={folds['Genome ID'].nunique()}",
    )
    add(
        "file08_oof_rows",
        len(file08_pred) == 190215,
        f"rows={len(file08_pred)}",
    )
    add(
        "file08_oof_unique_keys",
        not file08_pred.duplicated(
            ["Genome ID", "scheme", "representation", "model"]
        ).any(),
        "Genome ID/scheme/representation/model keys unique",
    )
    add(
        "feature_selection_rows",
        len(fs) == 60000,
        f"rows={len(fs)}",
    )

    # Verify all 30 scheme/fold/representation selection groups.
    ok_groups = 0
    for scheme in SCHEME_TO_COLUMN:
        for fold in range(5):
            for rep in ("pangenome", "unitig"):
                g = fs[
                    (fs["scheme"] == scheme)
                    & (pd.to_numeric(fs["fold"], errors="raise").astype(int) == fold)
                    & (fs["representation"] == rep)
                ].copy()
                g["rank"] = pd.to_numeric(g["rank"], errors="raise").astype(int)
                if (
                    len(g) == 2000
                    and sorted(g["rank"].tolist()) == list(range(1, 2001))
                    and not g["feature_index"].duplicated().any()
                ):
                    ok_groups += 1
    add(
        "train_only_selection_groups",
        ok_groups == 30,
        f"validated_groups={ok_groups}/30",
    )

    unitig = (
        root
        / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin"
    )
    add(
        "unitig_binary_size",
        unitig.stat().st_size == EXPECTED_UNITIG_BYTES,
        f"bytes={unitig.stat().st_size}",
    )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not (1 <= args.workers <= 12):
        raise SystemExit("--workers must be between 1 and 12.")

    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else find_project_root(Path.cwd())
    )

    if args.smoke:
        schemes = ("random_stratified",)
        active_folds = (0,)
        k_values = (500, 2000)
        c_values = (0.1, 1.0)
        class_weights = ("none",)
        namespace = "model_evaluation_reviewer_defense_smoke"
        out_dir = root / "data/evaluation/reviewer_defense/smoke"
    else:
        schemes = tuple(SCHEME_TO_COLUMN)
        active_folds = (0, 1, 2, 3, 4)
        k_values = PRIMARY_GRID_K
        c_values = PRIMARY_GRID_C
        class_weights = PRIMARY_CLASS_WEIGHTS
        namespace = "model_evaluation_reviewer_defense"
        out_dir = root / "data/evaluation/reviewer_defense"

    ckpt_dir = root / "checkpoints" / namespace
    fold_ckpt_dir = ckpt_dir / "folds"
    lock_path = ckpt_dir / "09b.lock"
    failure_path = ckpt_dir / "09b_last_failure.json"
    final_summary_path = ckpt_dir / "file09b_final_summary.json"
    log_path = root / "logs" / (
        "09b_reviewer_defense_smoke.log"
        if args.smoke
        else "09b_reviewer_defense.log"
    )

    outputs = {
        "pred": out_dir / "file09b_sensitivity_oof_predictions.csv.gz",
        "fold_metrics": out_dir / "file09b_sensitivity_fold_metrics.csv",
        "perf": out_dir / "file09b_sensitivity_performance_summary.csv",
        "deltas": out_dir / "file09b_matched_incremental_deltas.csv",
        "robustness": out_dir / "file09b_incremental_robustness_summary.csv",
        "family": out_dir / "file09b_model_family_consistency.csv",
        "exact_mlst": out_dir / "file09b_exact_mlst_subset_summary.csv",
        "lineage_exclusion": out_dir / "file09b_major_lineage_exclusion_sensitivity.csv",
        "audit": out_dir / "file09b_integrity_audit.csv",
        "design": out_dir / "file09b_analysis_design.json",
    }

    logger = Logger(log_path)
    started = time.time()
    stage = "startup"

    acquire_lock(lock_path)

    try:
        logger(f"File09b script version: {SCRIPT_VERSION}")
        logger(f"Mode: {'SMOKE' if args.smoke else 'FULL'}")
        logger(f"Project root: {root}")
        logger(f"Workers: {args.workers}")
        logger(f"k grid: {k_values}")
        logger(f"C grid: {c_values}")
        logger(f"class_weight grid: {class_weights}")

        required = {
            "folds": root / "data/splits/file07_outer_folds.csv",
            "known": root / "data/features/known_amr/X_known_amr_final.csv",
            "pang": root / "data/features/pangenome/X_pangenome_plfam_binary.npz",
            "pang_rows": root / "data/features/pangenome/X_pangenome_row_index.csv",
            "unitig": root / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin",
            "fs": root / "data/modeling/file08_feature_selection_manifest.csv",
            "file08_pred": root / "data/modeling/file08_oof_predictions.csv",
            "file09_fold": root / "data/evaluation/file09_fold_metrics.csv",
            "file09_perf": root / "data/evaluation/file09_performance_summary.csv",
        }
        for name, path in required.items():
            if not path.is_file():
                raise RuntimeError(f"Missing required input ({name}): {path}")
            logger(f"Input OK: {path.relative_to(root)}")

        stage = "input_audit"
        folds = load_folds(required["folds"])
        fs = pd.read_csv(required["fs"])
        file08_pred = pd.read_csv(
            required["file08_pred"], dtype={"Genome ID": str}
        )
        file09_fold = pd.read_csv(required["file09_fold"])
        file09_perf = pd.read_csv(required["file09_perf"])

        audit = integrity_audit(root, folds, fs, file08_pred)
        atomic_write_csv(outputs["audit"], audit)
        logger(f"Integrity audit PASS: {len(audit)} checks.")

        design = {
            "script_version": SCRIPT_VERSION,
            "created_utc": utc_now(),
            "mode": "SMOKE" if args.smoke else "FULL",
            "analysis_role": "post-primary sensitivity / reviewer defense",
            "outer_folds_changed": False,
            "outer_test_labels_used_for_feature_selection": False,
            "feature_ranking_source": (
                "Locked File08 outer-training-fold chi-square ranking"
            ),
            "hyperparameter_selection_from_outer_test": False,
            "all_grid_settings_reported": True,
            "k_grid": list(k_values),
            "C_grid": list(c_values),
            "class_weight_grid": list(class_weights),
            "matched_incremental_baseline": (
                "known_amr with identical C and class_weight"
            ),
            "representations": ["known_amr", *BROADER_REPS],
            "additional_robustness_checks": [
                "matched model-family consistency",
                "exact-MLST-only evaluation",
                "cumulative top-1/top-2/top-5 MLST exclusion",
            ],
            "workers": args.workers,
            "inner_threads_per_worker": 1,
            "interpretation_rule": (
                "This file does not select a best configuration. Robustness is "
                "judged from consistency of conclusions across the full grid."
            ),
        }
        atomic_write_text(
            outputs["design"], json.dumps(design, indent=2) + "\n"
        )

        stage = "sensitivity_training"
        jobs = [
            (scheme, fold)
            for scheme in schemes
            for fold in active_folds
        ]
        logger(
            f"START sensitivity training: {len(jobs)} outer-fold jobs, "
            f"{expected_config_count(k_values, c_values, class_weights)} "
            "configurations/job."
        )

        with parallel_config(
            backend="loky",
            n_jobs=args.workers,
            inner_max_num_threads=1,
        ):
            results = Parallel(
                n_jobs=args.workers,
                backend="loky",
                batch_size=1,
                pre_dispatch=args.workers,
            )(
                delayed(run_outer_fold_job)(
                    str(root),
                    namespace,
                    scheme,
                    fold,
                    k_values,
                    c_values,
                    class_weights,
                )
                for scheme, fold in jobs
            )

        for i, r in enumerate(results, start=1):
            logger(
                f"  fold job {i}/{len(results)} {r['status']}: "
                f"{r['scheme']} fold {r['fold']}; "
                f"configs={r['configs']}; rows={r['rows']:,}; "
                f"{human_seconds(float(r['elapsed_seconds']))}"
            )

        stage = "aggregate_sensitivity"
        fold_frames = []
        for scheme, fold in jobs:
            p = fold_ckpt_dir / f"{scheme}__fold{fold}.csv.gz"
            if not p.exists():
                raise RuntimeError(f"Missing fold checkpoint: {p}")
            fold_frames.append(
                pd.read_csv(
                    p, compression="gzip", dtype={"Genome ID": str}
                )
            )
        sens_pred = pd.concat(fold_frames, ignore_index=True)

        # OOF uniqueness relative to active folds.
        if sens_pred.duplicated(
            ["Genome ID", "scheme", "config_id"]
        ).any():
            raise RuntimeError("Duplicate sensitivity OOF keys.")
        if not np.isfinite(sens_pred["score"]).all():
            raise RuntimeError("Non-finite sensitivity OOF scores.")

        atomic_write_csv_gz(outputs["pred"], sens_pred)

        fold_metrics = compute_sensitivity_fold_metrics(sens_pred)
        perf = summarize_sensitivity_metrics(fold_metrics)
        deltas, robustness = matched_incremental_deltas(fold_metrics)

        atomic_write_csv(outputs["fold_metrics"], fold_metrics)
        atomic_write_csv(outputs["perf"], perf)
        atomic_write_csv(outputs["deltas"], deltas)
        atomic_write_csv(outputs["robustness"], robustness)

        stage = "locked_result_robustness"
        family = model_family_consistency(file09_fold)
        exact_subset = exact_mlst_subset_analysis(
            file08_pred, folds, file09_perf
        )
        lineage_excl = major_lineage_exclusion_analysis(
            file08_pred, folds, file09_perf
        )

        atomic_write_csv(outputs["family"], family)
        atomic_write_csv(outputs["exact_mlst"], exact_subset)
        atomic_write_csv(outputs["lineage_exclusion"], lineage_excl)

        stage = "final_audit"
        if not args.smoke:
            expected_configs_scheme = expected_config_count(
                k_values, c_values, class_weights
            )
            expected_total_configs = (
                expected_configs_scheme * len(schemes)
            )
            observed_total_configs = (
                sens_pred[
                    ["scheme", "config_id"]
                ].drop_duplicates().shape[0]
            )
            if observed_total_configs != expected_total_configs:
                raise RuntimeError(
                    f"Expected {expected_total_configs} scheme/config pairs, "
                    f"found {observed_total_configs}."
                )
            if len(fold_metrics) != expected_total_configs * 5:
                raise RuntimeError(
                    "Sensitivity fold-metric row count mismatch."
                )
            if len(perf) != expected_total_configs:
                raise RuntimeError(
                    "Sensitivity performance-summary row count mismatch."
                )

        summary = {
            "script_version": SCRIPT_VERSION,
            "status": "PASS",
            "mode": "SMOKE" if args.smoke else "FULL",
            "completed_utc": utc_now(),
            "workers": args.workers,
            "schemes": list(schemes),
            "active_folds": list(active_folds),
            "k_grid": list(k_values),
            "C_grid": list(c_values),
            "class_weight_grid": list(class_weights),
            "sensitivity_prediction_rows": len(sens_pred),
            "sensitivity_fold_metric_rows": len(fold_metrics),
            "sensitivity_configuration_rows": len(perf),
            "matched_delta_rows": len(deltas),
            "robustness_summary_rows": len(robustness),
            "model_family_consistency_rows": len(family),
            "exact_mlst_subset_rows": len(exact_subset),
            "major_lineage_exclusion_rows": len(lineage_excl),
            "best_configuration_selected_from_outer_test": False,
            "outer_test_feature_selection": False,
            "elapsed_seconds": time.time() - started,
            "outputs": {
                k: str(v.relative_to(root)) for k, v in outputs.items()
            },
        }
        atomic_write_text(
            final_summary_path, json.dumps(summary, indent=2) + "\n"
        )

        if failure_path.exists():
            failure_path.unlink()

        logger("=" * 80)
        logger("FILE09b — SENSITIVITY AND REVIEWER DEFENSE")
        logger("=" * 80)
        logger(f"Mode                              : {'SMOKE' if args.smoke else 'FULL'}")
        logger(f"Workers                           : {args.workers}")
        logger(f"Sensitivity prediction rows       : {len(sens_pred):,}")
        logger(f"Fold metric rows                  : {len(fold_metrics):,}")
        logger(f"Complete OOF configuration rows   : {len(perf):,}")
        logger(f"Matched delta rows                : {len(deltas):,}")
        logger(f"Model-family consistency rows     : {len(family):,}")
        logger(f"Exact-MLST subset rows            : {len(exact_subset):,}")
        logger(f"Lineage-exclusion rows            : {len(lineage_excl):,}")
        logger("Outer-test feature selection      : NO")
        logger("Best config selected on test      : NO")
        logger(f"Elapsed                           : {human_seconds(time.time()-started)}")
        logger(f"Final summary                     : {final_summary_path}")
        logger("FILE09b STATUS                    : PASS")
        logger("=" * 80)
        return 0

    except KeyboardInterrupt:
        failure = {
            "script_version": SCRIPT_VERSION,
            "status": "INTERRUPTED",
            "stage": stage,
            "time_utc": utc_now(),
            "message": (
                "Interrupted. Completed outer-fold checkpoints are preserved. "
                "Re-run the same command to resume."
            ),
        }
        atomic_write_text(
            failure_path, json.dumps(failure, indent=2) + "\n"
        )
        logger(
            "FILE09b INTERRUPTED. Completed fold checkpoints are preserved."
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
        atomic_write_text(
            failure_path, json.dumps(failure, indent=2) + "\n"
        )
        logger("")
        logger("FILE09b FAILED")
        logger(f"Stage : {stage}")
        logger(f"Error : {exc}")
        logger(f"Crash record: {failure_path}")
        logger(
            "Fix the cause and re-run the same command; completed fold "
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
