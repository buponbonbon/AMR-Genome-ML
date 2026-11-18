#!/usr/bin/env python3
"""
09c_nonlinear_incremental_reviewer_defense.py
=============================================

PURPOSE
-------
Final nonlinear reviewer-defense sensitivity analysis.

File09 established the primary performance/generalization results.
File09b established that the incremental conclusions were robust to:
    - L2-logistic regularization strength,
    - feature-count sensitivity,
    - class weighting,
    - exact-MLST-only evaluation,
    - dominant-lineage exclusion,
    - matched model-family comparisons for representations evaluated in File08.

The remaining plausible objection is:
    "Could broader genomic features provide useful NONLINEAR interactions with
     known-AMR determinants that a linear additive model cannot capture?"

File09c directly addresses that question with a targeted 12-setting nonlinear sensitivity grid.

NONLINEAR MODEL FAMILIES
------------------------
1. ExtraTrees
2. Histogram Gradient Boosting
3. RBF-SVM

INCREMENTAL REPRESENTATIONS
---------------------------
Baseline:
    known_amr

Candidates:
    known_plus_pangenome
    known_plus_unitig
    known_plus_pangenome_plus_unitig

BROADER-FEATURE COUNTS
----------------------
    k = 1000, 2000

The selected pangenome/unitig features are prefixes of the LOCKED File08
outer-training-fold rankings.  No feature is re-ranked with outer-test labels.

NONLINEAR SENSITIVITY GRID
--------------------------
ExtraTrees:
    max_features   = {"sqrt", 0.3}
    min_samples_leaf = {1, 5}
    class_weight   = {None, "balanced"}
    n_estimators   = 300

Histogram Gradient Boosting:
    max_leaf_nodes = {15, 31}
    l2_regularization = {1, 10}
    class_weight   = {None, "balanced"}
    learning_rate  = 0.05
    max_iter       = 200

RBF-SVM:
    C              = {0.1, 1, 10}
    gamma          = "scale"
    class_weight   = {None, "balanced"}

This version uses 12 targeted nonlinear settings (4 per model family).

For each outer fold:
    22 matched known-AMR baselines
    + 3 incremental representations x 2 k values x 22 settings
    = 57 configurations / fold

Across 15 locked outer folds:
    855 fold/config fits.

ANTI-LEAKAGE / ANTI-CHERRY-PICKING RULES
----------------------------------------
- Outer folds are the locked File07 folds.
- File08 train-only rankings are reused exactly.
- No outer-test label is used for feature selection or hyperparameter choice.
- ALL nonlinear settings are reported.
- No "best" nonlinear configuration is selected from outer-test performance.
- Every candidate is compared against known-AMR with the SAME:
      validation fold,
      model family,
      model hyperparameters,
      class weighting.
- This is a sensitivity analysis, not a second model-selection stage.

IMPLEMENTATION DEFENSE
----------------------
The default setting for each nonlinear family reproduces the corresponding
locked File08 known-AMR model:
    ExtraTrees: sqrt / min_leaf=1 / no class weighting
    HistGB    : max_leaf_nodes=31 / l2=1 / no class weighting
    RBF-SVM   : C=1 / gamma=scale / no class weighting

File09c explicitly compares those OOF scores/predictions against File08 and
fails if the reproduction check exceeds strict numerical tolerances.

SMOKE
-----
    python scripts/09c_nonlinear_incremental_reviewer_defense.py --smoke --workers 8

FULL
----
    python scripts/09c_nonlinear_incremental_reviewer_defense.py --workers 8

OUTPUTS
-------
data/evaluation/reviewer_defense_nonlinear/
    file09c_oof_predictions.csv.gz
    file09c_fold_metrics.csv
    file09c_performance_summary.csv
    file09c_matched_incremental_deltas.csv
    file09c_nonlinear_incremental_robustness.csv
    file09c_default_baseline_reproduction_audit.csv
    file09c_integrity_audit.csv
    file09c_analysis_design.json

checkpoints/model_evaluation_reviewer_defense_nonlinear/
    folds/*.csv.gz
    file09c_final_summary.json
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.svm import SVC

SCRIPT_VERSION = "1.0.3"
COMPATIBLE_CHECKPOINT_VERSIONS = {"1.0.2", "1.0.3"}
DESIGN_ID = "09c_12setting_config_checkpoint_v1"
RANDOM_SEED = 20260920

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

INCREMENTAL_REPS = (
    "known_plus_pangenome",
    "known_plus_unitig",
    "known_plus_pangenome_plus_unitig",
)

PRIMARY_K = (1000, 2000)

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
    if lock_path.exists():
        try:
            old = json.loads(lock_path.read_text(encoding="utf-8"))
            old_pid = int(old.get("pid", -1))
        except Exception:
            old_pid = -1
        if old_pid > 0 and Path(f"/proc/{old_pid}").exists():
            raise RuntimeError(
                f"Active File09c lock exists: {lock_path} (PID {old_pid})."
            )
        lock_path.unlink(missing_ok=True)

    atomic_write_text(
        lock_path,
        json.dumps(
            {
                "pid": os.getpid(),
                "started_utc": utc_now(),
                "script_version": SCRIPT_VERSION,
                "design_id": DESIGN_ID,
            },
            indent=2,
        ) + "\n",
    )


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def metric_bundle(
    y: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.uint8)

    if len(np.unique(y)) != 2:
        raise RuntimeError("Each outer test fold must contain both classes.")

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    return {
        "n": int(len(y)),
        "n_resistant": int(y.sum()),
        "n_susceptible": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def load_folds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Genome ID": str})
    required = {
        "Genome ID", "y", "random_fold", "genomic_cluster_fold", "mlst_fold"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"File07 fold file missing: {sorted(missing)}")
    if len(df) != EXPECTED_COHORT_N or df["Genome ID"].nunique() != EXPECTED_COHORT_N:
        raise RuntimeError("Unexpected File07 cohort size / duplicate Genome IDs.")
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)
    for col in SCHEME_TO_COLUMN.values():
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
        if set(df[col].unique()) != {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{col} must contain exactly folds 0..4.")
    return df


def load_known(path: Path, cohort_ids: List[str]) -> np.ndarray:
    df = pd.read_csv(path, dtype={"Genome ID": str})
    if "Genome ID" not in df.columns:
        raise RuntimeError("Known-AMR matrix lacks Genome ID.")
    features = [c for c in df.columns if c != "Genome ID"]
    if len(features) != EXPECTED_KNOWN_FEATURES:
        raise RuntimeError(
            f"Known-AMR feature mismatch: {len(features)} != {EXPECTED_KNOWN_FEATURES}"
        )
    indexed = df.set_index("Genome ID")
    missing = [gid for gid in cohort_ids if gid not in indexed.index]
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
        raise RuntimeError("Unexpected pangenome feature count.")
    ids = rows["Genome ID"].astype(str).tolist()
    if len(ids) != X.shape[0] or len(set(ids)) != len(ids):
        raise RuntimeError("Invalid pangenome row index.")
    if ids == cohort_ids:
        return X
    pos = {gid: i for i, gid in enumerate(ids)}
    missing = [gid for gid in cohort_ids if gid not in pos]
    if missing:
        raise RuntimeError(f"Pangenome missing IDs: {missing[:10]}")
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
            f"Expected 2000 selections for {scheme}/fold{fold}/{representation}; "
            f"found {len(g)}."
        )
    g["rank"] = pd.to_numeric(g["rank"], errors="raise").astype(int)
    g["feature_index"] = pd.to_numeric(
        g["feature_index"], errors="raise"
    ).astype(int)
    g = g.sort_values("rank")
    if g["rank"].tolist() != list(range(1, 2001)):
        raise RuntimeError(
            f"Feature-selection ranks not exactly 1..2000 for "
            f"{scheme}/fold{fold}/{representation}."
        )
    if g["feature_index"].duplicated().any():
        raise RuntimeError("Duplicate selected feature index.")
    return g["feature_index"].to_numpy(dtype=np.int64)


def full_model_settings() -> List[dict]:
    """Targeted 12-setting nonlinear sensitivity grid."""
    settings = [
        # ExtraTrees: default + one-factor sensitivity.
        {
            "family": "extra_trees",
            "setting_id": "extra_trees__mfsqrt__leaf1__cwnone",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "none",
        },
        {
            "family": "extra_trees",
            "setting_id": "extra_trees__mfsqrt__leaf5__cwnone",
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "class_weight": "none",
        },
        {
            "family": "extra_trees",
            "setting_id": "extra_trees__mf0.3__leaf1__cwnone",
            "max_features": 0.3,
            "min_samples_leaf": 1,
            "class_weight": "none",
        },
        {
            "family": "extra_trees",
            "setting_id": "extra_trees__mfsqrt__leaf1__cwbalanced",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "balanced",
        },

        # HistGradientBoosting: default + one-factor sensitivity.
        {
            "family": "hist_gb",
            "setting_id": "hist_gb__leaf31__l21__cwnone",
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "class_weight": "none",
        },
        {
            "family": "hist_gb",
            "setting_id": "hist_gb__leaf15__l21__cwnone",
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "class_weight": "none",
        },
        {
            "family": "hist_gb",
            "setting_id": "hist_gb__leaf31__l210__cwnone",
            "max_leaf_nodes": 31,
            "l2_regularization": 10.0,
            "class_weight": "none",
        },
        {
            "family": "hist_gb",
            "setting_id": "hist_gb__leaf31__l21__cwbalanced",
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
        },

        # RBF-SVM: C sensitivity + class weighting.
        {
            "family": "rbf_svm",
            "setting_id": "rbf_svm__C0.1__cwnone",
            "C": 0.1,
            "class_weight": "none",
        },
        {
            "family": "rbf_svm",
            "setting_id": "rbf_svm__C1__cwnone",
            "C": 1.0,
            "class_weight": "none",
        },
        {
            "family": "rbf_svm",
            "setting_id": "rbf_svm__C10__cwnone",
            "C": 10.0,
            "class_weight": "none",
        },
        {
            "family": "rbf_svm",
            "setting_id": "rbf_svm__C1__cwbalanced",
            "C": 1.0,
            "class_weight": "balanced",
        },
    ]
    if len(settings) != 12:
        raise RuntimeError(f"Internal nonlinear setting count != 12: {len(settings)}")
    return settings


def smoke_model_settings() -> List[dict]:
    return [
        {
            "family": "extra_trees",
            "setting_id": "extra_trees__mfsqrt__leaf1__cwnone",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "none",
        },
        {
            "family": "hist_gb",
            "setting_id": "hist_gb__leaf31__l21__cwnone",
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "class_weight": "none",
        },
        {
            "family": "rbf_svm",
            "setting_id": "rbf_svm__C1__cwnone",
            "C": 1.0,
            "class_weight": "none",
        },
    ]


def make_model(setting: dict):
    family = setting["family"]
    cw = None if setting["class_weight"] == "none" else "balanced"

    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_features=setting["max_features"],
            min_samples_leaf=int(setting["min_samples_leaf"]),
            class_weight=cw,
            n_jobs=1,
            random_state=RANDOM_SEED,
        )

    if family == "hist_gb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=int(setting["max_leaf_nodes"]),
            l2_regularization=float(setting["l2_regularization"]),
            class_weight=cw,
            random_state=RANDOM_SEED,
        )

    if family == "rbf_svm":
        return SVC(
            C=float(setting["C"]),
            kernel="rbf",
            gamma="scale",
            class_weight=cw,
            cache_size=2048,
        )

    raise KeyError(family)


def prediction_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    raise RuntimeError(f"No score method for {type(model).__name__}")


def config_id(
    representation: str,
    k: int,
    setting_id: str,
) -> str:
    klabel = "NA" if k < 0 else str(int(k))
    return f"{representation}__k{klabel}__{setting_id}"


def settings_for_k(settings: List[dict], k: int) -> List[dict]:
    # k=2000 gets all 12 settings. k=1000 gets only the three File08 defaults.
    if k == 2000:
        return list(settings)
    if k == 1000:
        defaults = {
            "extra_trees__mfsqrt__leaf1__cwnone",
            "hist_gb__leaf31__l21__cwnone",
            "rbf_svm__C1__cwnone",
        }
        return [s for s in settings if s["setting_id"] in defaults]
    return list(settings)


def config_plan(settings: List[dict], k_values: Tuple[int, ...]) -> List[dict]:
    plan = []
    for setting in settings:
        plan.append(
            {"representation": "known_amr", "k": -1, "setting": setting}
        )
    for k in k_values:
        for rep in INCREMENTAL_REPS:
            for setting in settings_for_k(settings, int(k)):
                plan.append(
                    {"representation": rep, "k": int(k), "setting": setting}
                )
    return plan


def expected_configs_per_fold(
    settings: List[dict],
    k_values: Tuple[int, ...],
) -> int:
    return len(config_plan(settings, k_values))


def config_checkpoint_path(
    root: Path,
    namespace: str,
    scheme: str,
    fold: int,
    cid: str,
) -> Path:
    return (
        root / "checkpoints" / namespace / "configs"
        / f"{scheme}__fold{fold}__{cid}.csv.gz"
    )


def config_checkpoint_valid(
    path: Path,
    expected_test_n: int,
    expected_scheme: str,
    expected_fold: int,
    expected_cid: str,
) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, compression="gzip", dtype={"Genome ID": str})
        required = {
            "Genome ID", "y", "scheme", "fold", "representation", "k",
            "model_family", "setting_id", "class_weight", "config_id",
            "score", "pred", "fit_seconds", "script_version", "design_id",
        }
        if not required.issubset(df.columns):
            return False
        if len(df) != expected_test_n:
            return False
        if df["Genome ID"].nunique() != expected_test_n:
            return False
        if df["script_version"].nunique() != 1 or df["script_version"].iloc[0] not in COMPATIBLE_CHECKPOINT_VERSIONS:
            return False
        if df["design_id"].nunique() != 1 or df["design_id"].iloc[0] != DESIGN_ID:
            return False
        if df["scheme"].nunique() != 1 or df["scheme"].iloc[0] != expected_scheme:
            return False
        if int(df["fold"].iloc[0]) != int(expected_fold):
            return False
        if df["config_id"].nunique() != 1 or df["config_id"].iloc[0] != expected_cid:
            return False
        if not np.isfinite(pd.to_numeric(df["score"], errors="coerce")).all():
            return False
        return True
    except Exception:
        return False


def count_config_checkpoints(root: Path, namespace: str) -> int:
    d = root / "checkpoints" / namespace / "configs"
    if not d.exists():
        return 0
    return sum(1 for _ in d.glob("*.csv.gz"))


def run_fold_job(
    root_str: str,
    namespace: str,
    scheme: str,
    fold: int,
    settings: List[dict],
    k_values: Tuple[int, ...],
    expected_global_configs: int,
) -> dict:
    root = Path(root_str)
    started = time.time()

    folds = load_folds(root / "data/splits/file07_outer_folds.csv")
    cohort_ids = folds["Genome ID"].astype(str).tolist()
    y = folds["y"].to_numpy(dtype=np.uint8)

    col = SCHEME_TO_COLUMN[scheme]
    fvals = folds[col].to_numpy(dtype=int)
    train_idx = np.flatnonzero(fvals != fold)
    test_idx = np.flatnonzero(fvals == fold)

    plan = config_plan(settings, k_values)

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

    Xp_max = np.ascontiguousarray(pang[:, pidx].toarray(), dtype=np.uint8)
    Xu_max = decode_unitigs(
        root / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin",
        uidx,
    )

    matrix_cache = {("known_amr", -1): known}

    def get_matrix(rep: str, k: int) -> np.ndarray:
        key = (rep, k)
        if key in matrix_cache:
            return matrix_cache[key]
        Xp = Xp_max[:, :k]
        Xu = Xu_max[:, :k]
        if rep == "known_plus_pangenome":
            X = np.concatenate([known, Xp], axis=1)
        elif rep == "known_plus_unitig":
            X = np.concatenate([known, Xu], axis=1)
        elif rep == "known_plus_pangenome_plus_unitig":
            X = np.concatenate([known, Xp, Xu], axis=1)
        else:
            raise KeyError(rep)
        matrix_cache[key] = np.ascontiguousarray(X, dtype=np.uint8)
        return matrix_cache[key]

    completed = 0
    skipped = 0

    for local_i, spec in enumerate(plan, start=1):
        rep = spec["representation"]
        k = int(spec["k"])
        setting = spec["setting"]
        cid = config_id(rep, k, setting["setting_id"])
        ckpt = config_checkpoint_path(root, namespace, scheme, fold, cid)

        if config_checkpoint_valid(
            ckpt, len(test_idx), scheme, fold, cid
        ):
            completed += 1
            skipped += 1
            continue

        X = get_matrix(rep, k)
        Xtr = np.asarray(X[train_idx], dtype=np.float32, order="C")
        Xte = np.asarray(X[test_idx], dtype=np.float32, order="C")

        model = make_model(setting)
        fit_t0 = time.time()
        model.fit(Xtr, y[train_idx])
        fit_seconds = time.time() - fit_t0

        score = prediction_score(model, Xte)
        predv = np.asarray(model.predict(Xte), dtype=np.uint8)

        if not np.isfinite(score).all():
            raise RuntimeError(
                f"Non-finite score: {scheme}/fold{fold}/{cid}"
            )

        out = pd.DataFrame(
            {
                "Genome ID": np.asarray(cohort_ids, dtype=object)[test_idx],
                "y": y[test_idx].astype(int),
                "scheme": scheme,
                "fold": fold,
                "representation": rep,
                "k": k,
                "model_family": setting["family"],
                "setting_id": setting["setting_id"],
                "class_weight": setting["class_weight"],
                "config_id": cid,
                "score": score,
                "pred": predv.astype(int),
                "fit_seconds": fit_seconds,
                "script_version": SCRIPT_VERSION,
                "design_id": DESIGN_ID,
            }
        )

        atomic_write_csv_gz(ckpt, out)
        completed += 1

        global_done = count_config_checkpoints(root, namespace)
        pct = 100.0 * global_done / expected_global_configs
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"[PROGRESS] global {global_done}/{expected_global_configs} "
            f"({pct:5.1f}%) | {scheme} fold {fold} "
            f"{completed}/{len(plan)} | {cid} | "
            f"fit={human_seconds(fit_seconds)}",
            flush=True,
        )

    return {
        "scheme": scheme,
        "fold": fold,
        "status": "PASS" if skipped < len(plan) else "SKIP_COMPLETE",
        "configs": len(plan),
        "skipped_configs": skipped,
        "elapsed_seconds": time.time() - started,
    }


def compute_fold_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "scheme", "fold", "representation", "k",
        "model_family", "setting_id", "class_weight", "config_id",
    ]
    for key, g in pred.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row.update(
            metric_bundle(
                g["y"].to_numpy(dtype=np.uint8),
                g["score"].to_numpy(dtype=float),
                g["pred"].to_numpy(dtype=np.uint8),
            )
        )
        row["fit_seconds_mean_recorded"] = float(g["fit_seconds"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc", "average_precision", "accuracy",
        "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc",
    ]
    rows = []
    group_cols = [
        "scheme", "representation", "k", "model_family",
        "setting_id", "class_weight", "config_id",
    ]
    for key, g in fold_metrics.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row["n_outer_folds"] = len(g)
        row["mean_fit_seconds"] = float(np.mean(g["fit_seconds_mean_recorded"]))
        for metric in metric_cols:
            vals = g[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            )
            row[f"{metric}_min"] = float(np.min(vals))
            row[f"{metric}_max"] = float(np.max(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def matched_incremental_analysis(
    fold_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "roc_auc", "average_precision", "balanced_accuracy",
        "mcc", "sensitivity", "specificity", "f1",
    )

    base = fold_metrics[
        fold_metrics["representation"] == "known_amr"
    ][
        ["scheme", "fold", "model_family", "setting_id", "class_weight", *metrics]
    ].copy()

    base = base.rename(columns={m: f"baseline_{m}" for m in metrics})

    cand = fold_metrics[
        fold_metrics["representation"] != "known_amr"
    ].copy()

    merged = cand.merge(
        base,
        on=["scheme", "fold", "model_family", "setting_id", "class_weight"],
        how="left",
        validate="many_to_one",
    )

    if merged[[f"baseline_{m}" for m in metrics]].isna().any().any():
        raise RuntimeError("Missing matched nonlinear known-AMR baseline.")

    long_rows = []
    for row in merged.itertuples(index=False):
        for metric in metrics:
            cv = float(getattr(row, metric))
            bv = float(getattr(row, f"baseline_{metric}"))
            long_rows.append(
                {
                    "scheme": row.scheme,
                    "fold": int(row.fold),
                    "candidate_representation": row.representation,
                    "k": int(row.k),
                    "model_family": row.model_family,
                    "setting_id": row.setting_id,
                    "class_weight": row.class_weight,
                    "metric": metric,
                    "candidate_value": cv,
                    "matched_known_amr_value": bv,
                    "delta_candidate_minus_known_amr": cv - bv,
                }
            )
    delta = pd.DataFrame(long_rows)

    summary_rows = []

    # Setting-level mean across five folds.
    setting_mean = (
        delta.groupby(
            [
                "scheme", "candidate_representation", "k",
                "model_family", "setting_id", "class_weight", "metric",
            ],
            sort=True,
        )["delta_candidate_minus_known_amr"]
        .mean()
        .reset_index(name="mean_fold_delta")
    )

    # Robustness by family/k/metric across every tested nonlinear setting.
    for key, g in setting_mean.groupby(
        [
            "scheme", "candidate_representation", "k",
            "model_family", "metric",
        ],
        sort=True,
    ):
        scheme, rep, k, family, metric = key
        vals = g["mean_fold_delta"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "scheme": scheme,
                "candidate_representation": rep,
                "k_scope": str(int(k)),
                "model_family_scope": family,
                "metric": metric,
                "n_settings": len(vals),
                "mean_setting_delta": float(np.mean(vals)),
                "median_setting_delta": float(np.median(vals)),
                "min_setting_delta": float(np.min(vals)),
                "max_setting_delta": float(np.max(vals)),
                "fraction_settings_candidate_higher": float(np.mean(vals > 0)),
                "fraction_settings_candidate_lower": float(np.mean(vals < 0)),
                "all_settings_candidate_leq_baseline": bool(np.all(vals <= 0)),
            }
        )

    # Across both k values within each model family.
    for key, g in setting_mean.groupby(
        [
            "scheme", "candidate_representation", "model_family", "metric"
        ],
        sort=True,
    ):
        scheme, rep, family, metric = key
        vals = g["mean_fold_delta"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "scheme": scheme,
                "candidate_representation": rep,
                "k_scope": "ALL_1000_2000",
                "model_family_scope": family,
                "metric": metric,
                "n_settings": len(vals),
                "mean_setting_delta": float(np.mean(vals)),
                "median_setting_delta": float(np.median(vals)),
                "min_setting_delta": float(np.min(vals)),
                "max_setting_delta": float(np.max(vals)),
                "fraction_settings_candidate_higher": float(np.mean(vals > 0)),
                "fraction_settings_candidate_lower": float(np.mean(vals < 0)),
                "all_settings_candidate_leq_baseline": bool(np.all(vals <= 0)),
            }
        )

    # Across all nonlinear families and both k values.
    for key, g in setting_mean.groupby(
        ["scheme", "candidate_representation", "metric"],
        sort=True,
    ):
        scheme, rep, metric = key
        vals = g["mean_fold_delta"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "scheme": scheme,
                "candidate_representation": rep,
                "k_scope": "ALL_1000_2000",
                "model_family_scope": "ALL_NONLINEAR_FAMILIES",
                "metric": metric,
                "n_settings": len(vals),
                "mean_setting_delta": float(np.mean(vals)),
                "median_setting_delta": float(np.median(vals)),
                "min_setting_delta": float(np.min(vals)),
                "max_setting_delta": float(np.max(vals)),
                "fraction_settings_candidate_higher": float(np.mean(vals > 0)),
                "fraction_settings_candidate_lower": float(np.mean(vals < 0)),
                "all_settings_candidate_leq_baseline": bool(np.all(vals <= 0)),
            }
        )

    return delta, pd.DataFrame(summary_rows)


def default_setting_ids() -> Dict[str, str]:
    return {
        "extra_trees": "extra_trees__mfsqrt__leaf1__cwnone",
        "hist_gb": "hist_gb__leaf31__l21__cwnone",
        "rbf_svm": "rbf_svm__C1__cwnone",
    }


def baseline_reproduction_audit(
    pred09c: pd.DataFrame,
    pred08: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for family, setting_id in default_setting_ids().items():
        model08 = family
        for scheme in SCHEME_TO_COLUMN:
            a = pred09c[
                (pred09c["scheme"] == scheme)
                & (pred09c["representation"] == "known_amr")
                & (pred09c["setting_id"] == setting_id)
            ][["Genome ID", "fold", "score", "pred"]].copy()

            b = pred08[
                (pred08["scheme"] == scheme)
                & (pred08["representation"] == "known_amr")
                & (pred08["model"] == model08)
            ][["Genome ID", "fold", "score", "pred"]].copy()

            a = a.rename(
                columns={
                    "score": "score_09c",
                    "pred": "pred_09c",
                    "fold": "fold_09c",
                }
            )
            b = b.rename(
                columns={
                    "score": "score_08",
                    "pred": "pred_08",
                    "fold": "fold_08",
                }
            )

            m = a.merge(
                b, on="Genome ID", how="inner", validate="one_to_one"
            )
            if len(m) != EXPECTED_COHORT_N:
                raise RuntimeError(
                    f"Baseline reproduction merge incomplete for {scheme}/{family}."
                )

            fold_match = bool((m["fold_09c"] == m["fold_08"]).all())
            pred_match = bool((m["pred_09c"] == m["pred_08"]).all())
            max_abs_score_diff = float(
                np.max(np.abs(m["score_09c"] - m["score_08"]))
            )

            # Tree ensemble and SVM should normally be exact; allow only tiny
            # numerical noise across process/thread layouts.
            score_tol = 1e-12 if family != "hist_gb" else 1e-10
            passed = fold_match and pred_match and max_abs_score_diff <= score_tol

            rows.append(
                {
                    "scheme": scheme,
                    "model_family": family,
                    "setting_id": setting_id,
                    "n_genomes": len(m),
                    "fold_match": fold_match,
                    "prediction_match": pred_match,
                    "max_abs_score_difference": max_abs_score_diff,
                    "score_tolerance": score_tol,
                    "status": "PASS" if passed else "FAIL",
                }
            )

            if not passed:
                raise RuntimeError(
                    f"File09c default known-AMR baseline does not reproduce "
                    f"File08 for {scheme}/{family}: "
                    f"fold_match={fold_match}, pred_match={pred_match}, "
                    f"max_score_diff={max_abs_score_diff:.3e}, tol={score_tol:.3e}"
                )

    return pd.DataFrame(rows)


def integrity_audit(
    root: Path,
    folds: pd.DataFrame,
    fs: pd.DataFrame,
    pred08: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, detail: str):
        rows.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )
        if not passed:
            raise RuntimeError(f"Integrity audit failed: {check}: {detail}")

    add(
        "cohort_size",
        len(folds) == EXPECTED_COHORT_N
        and folds["Genome ID"].nunique() == EXPECTED_COHORT_N,
        f"rows={len(folds)}, unique_ids={folds['Genome ID'].nunique()}",
    )
    add(
        "file08_prediction_rows",
        len(pred08) == 190215,
        f"rows={len(pred08)}",
    )
    add(
        "file08_unique_oof_keys",
        not pred08.duplicated(
            ["Genome ID", "scheme", "representation", "model"]
        ).any(),
        "OOF keys unique",
    )
    add(
        "feature_selection_rows",
        len(fs) == 60000,
        f"rows={len(fs)}",
    )

    valid_groups = 0
    for scheme in SCHEME_TO_COLUMN:
        for fold in range(5):
            for rep in ("pangenome", "unitig"):
                try:
                    selection_indices(fs, scheme, fold, rep)
                    valid_groups += 1
                except Exception:
                    pass
    add(
        "locked_train_only_feature_selection_groups",
        valid_groups == 30,
        f"valid_groups={valid_groups}/30",
    )

    unitig_path = (
        root
        / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin"
    )
    add(
        "unitig_binary_size",
        unitig_path.stat().st_size == EXPECTED_UNITIG_BYTES,
        f"bytes={unitig_path.stat().st_size}",
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
        k_values = (2000,)
        settings = smoke_model_settings()
        namespace = "model_evaluation_reviewer_defense_nonlinear_v102_smoke"
        out_dir = root / "data/evaluation/reviewer_defense_nonlinear_v102/smoke"
    else:
        schemes = tuple(SCHEME_TO_COLUMN)
        active_folds = (0, 1, 2, 3, 4)
        k_values = PRIMARY_K
        settings = full_model_settings()
        namespace = "model_evaluation_reviewer_defense_nonlinear_v102"
        out_dir = root / "data/evaluation/reviewer_defense_nonlinear_v102"

    ckpt_dir = root / "checkpoints" / namespace
    fold_ckpt_dir = ckpt_dir / "folds"
    lock_path = ckpt_dir / "09c.lock"
    failure_path = ckpt_dir / "09c_last_failure.json"
    summary_path = ckpt_dir / "file09c_final_summary.json"
    log_path = root / "logs" / (
        "09c_nonlinear_reviewer_defense_v102_smoke.log"
        if args.smoke
        else "09c_nonlinear_reviewer_defense_v102.log"
    )

    outputs = {
        "pred": out_dir / "file09c_oof_predictions.csv.gz",
        "fold_metrics": out_dir / "file09c_fold_metrics.csv",
        "perf": out_dir / "file09c_performance_summary.csv",
        "deltas": out_dir / "file09c_matched_incremental_deltas.csv",
        "robustness": out_dir / "file09c_nonlinear_incremental_robustness.csv",
        "reproduction": out_dir / "file09c_default_baseline_reproduction_audit.csv",
        "audit": out_dir / "file09c_integrity_audit.csv",
        "design": out_dir / "file09c_analysis_design.json",
    }

    logger = Logger(log_path)
    stage = "startup"
    started = time.time()

    acquire_lock(lock_path)

    try:
        logger(f"File09c script version: {SCRIPT_VERSION}")
        logger(f"Design ID: {DESIGN_ID}")
        logger(f"Mode: {'SMOKE' if args.smoke else 'FULL'}")
        logger(f"Project root: {root}")
        logger(f"Workers: {args.workers}")
        logger(f"k values: {k_values}")
        logger(f"Nonlinear settings: {len(settings)}")
        logger("Checkpoint granularity: ONE CONFIGURATION")
        logger("Live progress: ENABLED")
        logger(
            f"Expected configurations/fold: "
            f"{expected_configs_per_fold(settings, k_values)}"
        )

        required = {
            "folds": root / "data/splits/file07_outer_folds.csv",
            "known": root / "data/features/known_amr/X_known_amr_final.csv",
            "pang": root / "data/features/pangenome/X_pangenome_plfam_binary.npz",
            "pang_rows": root / "data/features/pangenome/X_pangenome_row_index.csv",
            "unitig": root / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin",
            "fs": root / "data/modeling/file08_feature_selection_manifest.csv",
            "pred08": root / "data/modeling/file08_oof_predictions.csv",
            "summary08": root / "checkpoints/model_training/file08_final_summary.json",
        }
        for name, path in required.items():
            if not path.is_file():
                raise RuntimeError(f"Missing required input ({name}): {path}")
            logger(f"Input OK: {path.relative_to(root)}")

        stage = "integrity_audit"
        folds = load_folds(required["folds"])
        fs = pd.read_csv(required["fs"])
        pred08 = pd.read_csv(required["pred08"], dtype={"Genome ID": str})
        with required["summary08"].open("r", encoding="utf-8") as fh:
            summary08 = json.load(fh)

        if summary08.get("supervised_feature_selection_scope") != "outer_training_fold_only":
            raise RuntimeError(
                "File08 summary does not confirm outer-training-fold-only selection."
            )
        if summary08.get("outer_test_labels_used_for_selection_or_training") is not False:
            raise RuntimeError("File08 summary indicates unexpected outer-test use.")

        audit = integrity_audit(root, folds, fs, pred08)
        atomic_write_csv(outputs["audit"], audit)
        logger(f"Integrity audit PASS: {len(audit)} checks.")

        design = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "created_utc": utc_now(),
            "mode": "SMOKE" if args.smoke else "FULL",
            "analysis_role": "final nonlinear incremental reviewer-defense sensitivity",
            "outer_folds_changed": False,
            "outer_test_labels_used_for_feature_selection": False,
            "outer_test_hyperparameter_selection": False,
            "all_tested_settings_reported": True,
            "feature_ranking_source": "locked File08 outer-training-fold ranking",
            "k_values": list(k_values),
            "nonlinear_settings": settings,
            "representations": ["known_amr", *INCREMENTAL_REPS],
            "matched_baseline_rule": (
                "same scheme, fold, nonlinear family, nonlinear hyperparameters, "
                "and class weighting"
            ),
            "default_baseline_reproduction_required": not args.smoke,
            "workers": args.workers,
            "inner_threads_per_worker": 1,
            "checkpoint_granularity": "one configuration",
            "live_progress": True,
        }
        atomic_write_text(
            outputs["design"], json.dumps(design, indent=2) + "\n"
        )

        stage = "nonlinear_training"
        jobs = [
            (scheme, fold)
            for scheme in schemes
            for fold in active_folds
        ]
        expected_global_configs = len(jobs) * expected_configs_per_fold(settings, k_values)
        logger(
            f"START nonlinear sensitivity: {len(jobs)} outer-fold jobs x "
            f"{expected_configs_per_fold(settings, k_values)} configs/fold "
            f"= {expected_global_configs} resumable configs."
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
                delayed(run_fold_job)(
                    str(root),
                    namespace,
                    scheme,
                    fold,
                    settings,
                    k_values,
                    expected_global_configs,
                )
                for scheme, fold in jobs
            )

        for i, r in enumerate(results, start=1):
            logger(
                f"  fold job {i}/{len(results)} {r['status']}: "
                f"{r['scheme']} fold {r['fold']}; "
                f"configs={r['configs']}; skipped={r['skipped_configs']}; "
                f"{human_seconds(float(r['elapsed_seconds']))}"
            )

        stage = "aggregate"
        config_frames = []
        plan = config_plan(settings, k_values)

        for scheme, fold in jobs:
            test_n = int((folds[SCHEME_TO_COLUMN[scheme]] == fold).sum())
            for spec in plan:
                cid = config_id(
                    spec["representation"],
                    int(spec["k"]),
                    spec["setting"]["setting_id"],
                )
                p = config_checkpoint_path(root, namespace, scheme, fold, cid)
                if not config_checkpoint_valid(p, test_n, scheme, fold, cid):
                    raise RuntimeError(
                        f"Missing/invalid config checkpoint: "
                        f"{scheme}/fold{fold}/{cid}"
                    )
                config_frames.append(
                    pd.read_csv(p, compression="gzip", dtype={"Genome ID": str})
                )

        pred = pd.concat(config_frames, ignore_index=True)

        if pred.duplicated(["Genome ID", "scheme", "config_id"]).any():
            raise RuntimeError("Duplicate File09c OOF keys.")
        if not np.isfinite(pred["score"]).all():
            raise RuntimeError("Non-finite File09c scores.")

        atomic_write_csv_gz(outputs["pred"], pred)

        fold_metrics = compute_fold_metrics(pred)
        perf = summarize_fold_metrics(fold_metrics)
        deltas, robustness = matched_incremental_analysis(fold_metrics)

        atomic_write_csv(outputs["fold_metrics"], fold_metrics)
        atomic_write_csv(outputs["perf"], perf)
        atomic_write_csv(outputs["deltas"], deltas)
        atomic_write_csv(outputs["robustness"], robustness)

        stage = "baseline_reproduction"
        if args.smoke:
            reproduction = pd.DataFrame(
                [{
                    "status": "NOT_RUN_IN_SMOKE",
                    "detail": "Full 4,227-genome OOF coverage required for reproduction audit."
                }]
            )
        else:
            reproduction = baseline_reproduction_audit(pred, pred08)
        atomic_write_csv(outputs["reproduction"], reproduction)

        stage = "final_audit"
        if not args.smoke:
            expected_per_scheme = expected_configs_per_fold(settings, k_values)
            expected_config_rows = expected_per_scheme * len(schemes)

            observed_config_rows = (
                pred[["scheme", "config_id"]].drop_duplicates().shape[0]
            )
            if observed_config_rows != expected_config_rows:
                raise RuntimeError(
                    f"Scheme/config count mismatch: {observed_config_rows} != "
                    f"{expected_config_rows}"
                )
            if len(fold_metrics) != expected_config_rows * 5:
                raise RuntimeError(
                    f"Fold-metric count mismatch: {len(fold_metrics)} != "
                    f"{expected_config_rows * 5}"
                )
            if len(perf) != expected_config_rows:
                raise RuntimeError(
                    f"Performance-summary count mismatch: {len(perf)} != "
                    f"{expected_config_rows}"
                )
            if not bool((reproduction["status"] == "PASS").all()):
                raise RuntimeError("Default baseline reproduction audit did not fully PASS.")

        summary = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "PASS",
            "mode": "SMOKE" if args.smoke else "FULL",
            "completed_utc": utc_now(),
            "workers": args.workers,
            "schemes": list(schemes),
            "active_folds": list(active_folds),
            "k_values": list(k_values),
            "nonlinear_setting_count": len(settings),
            "configurations_per_fold": expected_configs_per_fold(settings, k_values),
            "prediction_rows": len(pred),
            "fold_metric_rows": len(fold_metrics),
            "performance_summary_rows": len(perf),
            "matched_delta_rows": len(deltas),
            "robustness_rows": len(robustness),
            "baseline_reproduction_rows": len(reproduction),
            "outer_test_feature_selection": False,
            "best_configuration_selected_from_outer_test": False,
            "checkpoint_granularity": "one configuration",
            "elapsed_seconds": time.time() - started,
            "outputs": {
                k: str(v.relative_to(root)) for k, v in outputs.items()
            },
        }
        atomic_write_text(
            summary_path, json.dumps(summary, indent=2) + "\n"
        )

        if failure_path.exists():
            failure_path.unlink()

        logger("=" * 82)
        logger("FILE09c — NONLINEAR INCREMENTAL REVIEWER DEFENSE")
        logger("=" * 82)
        logger(f"Mode                               : {'SMOKE' if args.smoke else 'FULL'}")
        logger(f"Workers                            : {args.workers}")
        logger(f"Nonlinear settings                 : {len(settings)}")
        logger(f"k values                           : {k_values}")
        logger(f"Prediction rows                    : {len(pred):,}")
        logger(f"Fold metric rows                   : {len(fold_metrics):,}")
        logger(f"Complete scheme/config rows        : {len(perf):,}")
        logger(f"Matched delta rows                 : {len(deltas):,}")
        logger(f"Robustness rows                    : {len(robustness):,}")
        if not args.smoke:
            logger(
                f"Default File08 reproduction        : "
                f"{int((reproduction['status']=='PASS').sum())}/"
                f"{len(reproduction)} PASS"
            )
        logger("Checkpoint granularity             : ONE CONFIGURATION")
        logger("Outer-test feature selection       : NO")
        logger("Best config selected on test       : NO")
        logger(f"Elapsed                            : {human_seconds(time.time()-started)}")
        logger(f"Final summary                      : {summary_path}")
        logger("FILE09c STATUS                     : PASS")
        logger("=" * 82)
        return 0

    except KeyboardInterrupt:
        failure = {
            "script_version": SCRIPT_VERSION,
            "status": "INTERRUPTED",
            "stage": stage,
            "time_utc": utc_now(),
            "saved_config_checkpoints": count_config_checkpoints(root, namespace),
            "message": (
                "Interrupted. Every completed configuration checkpoint is preserved. "
                "Re-run the same command to resume."
            ),
        }
        atomic_write_text(
            failure_path, json.dumps(failure, indent=2) + "\n"
        )
        logger(
            "FILE09c INTERRUPTED. Completed configuration checkpoints are preserved."
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
            "saved_config_checkpoints": count_config_checkpoints(root, namespace),
        }
        atomic_write_text(
            failure_path, json.dumps(failure, indent=2) + "\n"
        )
        logger("")
        logger("FILE09c FAILED")
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
    raise SystemExit(main())
