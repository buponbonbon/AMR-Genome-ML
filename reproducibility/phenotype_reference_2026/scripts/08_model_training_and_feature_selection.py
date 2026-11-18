#!/usr/bin/env python3
"""
08_model_training_and_feature_selection.py
==========================================

CORE PURPOSE
------------
Train leakage-aware genomic prediction models for meropenem resistance
using the File07 outer folds.

Scientific rules enforced by this script
----------------------------------------
1. The fixed analysis cohort is the 4,227 genomes in
   data/splits/file07_outer_folds.csv.
2. Genome ID is used only for linkage and never as a predictor.
3. MLST and genomic-cluster labels are used only to define validation folds.
4. Known-AMR features are used as a curated baseline without supervised
   feature selection.
5. Pangenome and unitig supervised feature ranking is performed separately
   inside each OUTER TRAINING FOLD using training labels only.
6. No outer-test labels are used for feature selection or model fitting.
7. Fixed model hyperparameters are used to avoid hidden inner-CV leakage.
8. Test-fold predictions are written for evaluation in File09; File08 does
   not select a "best" model from outer-test performance.

Representations
---------------
Primary representations, evaluated with four model families:
    - known_amr
    - pangenome
    - unitig

Incremental representations, evaluated with the same fixed L2 logistic
regression to isolate added information beyond the known-AMR baseline:
    - known_plus_pangenome
    - known_plus_unitig
    - known_plus_pangenome_plus_unitig

Model families
--------------
    - logreg_l2       : L2 logistic regression
    - rbf_svm         : RBF-kernel support-vector classifier
    - extra_trees     : ExtraTrees tree ensemble
    - hist_gb         : histogram gradient boosting

Default supervised feature counts
---------------------------------
    - pangenome: top 2,000 chi-square-ranked features per outer training fold
    - unitig   : top 2,000 chi-square-ranked features per outer training fold

The feature count is fixed a priori. File09 can perform sensitivity analyses
without changing the primary outer-test predictions.

Operational safety
------------------
- strict preflight and cohort/order checks
- label-leakage checks
- unitig bit-order validation against the original pyseer file
- resumable unitig feature-selection scan
- per-task atomic checkpoints for model predictions
- process lock to prevent duplicate runs
- crash/interruption record
- final atomic outputs
- smoke mode for a cheap end-to-end validation before the full run

Typical use
-----------
Smoke test first:
    python scripts/08_model_training_and_feature_selection.py --smoke

Full run:
    python scripts/08_model_training_and_feature_selection.py

Outputs
-------
data/modeling/file08_oof_predictions.csv
data/modeling/file08_task_manifest.csv
data/modeling/file08_feature_selection_manifest.csv
checkpoints/model_training/file08_final_summary.json

Selected-feature files:
checkpoints/model_training/selected_features/<scheme>/fold_<k>/...

Per-task prediction checkpoints:
checkpoints/model_training/tasks/*.npz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

SCRIPT_VERSION = "1.0.2"
RANDOM_SEED = 20260920

EXPECTED_COHORT_N = 4227
EXPECTED_KNOWN_FEATURES = 668
EXPECTED_PANGENOME_FEATURES = 22685
EXPECTED_UNITIG_FEATURES = 4718475
UNITIG_BYTES_PER_FEATURE = 529
EXPECTED_UNITIG_BYTES = 2496073275

PRIMARY_REPRESENTATIONS = ("known_amr", "pangenome", "unitig")
INCREMENTAL_REPRESENTATIONS = (
    "known_plus_pangenome",
    "known_plus_unitig",
    "known_plus_pangenome_plus_unitig",
)
MODEL_NAMES = ("logreg_l2", "rbf_svm", "extra_trees", "hist_gb")
SCHEME_TO_COLUMN = {
    "random_stratified": "random_fold",
    "genomic_cluster_aware": "genomic_cluster_fold",
    "mlst_aware": "mlst_fold",
}

STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_seconds(x: float) -> str:
    if x < 60:
        return f"{x:.1f}s"
    if x < 3600:
        return f"{x/60:.1f}m"
    return f"{x/3600:.2f}h"


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
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
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_csv(tmp_path, index=False)
        with tmp_path.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.savez_compressed(tmp_path, **arrays)
        with tmp_path.open("rb") as fh:
            os.fsync(fh.fileno())
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
            "Another File08 process may still be running. If the old process "
            "is definitely dead, inspect the lock before removing it."
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


@dataclass(frozen=True)
class FoldSpec:
    scheme: str
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def normalize_unitig_sample_name(x: str) -> str:
    x = x.strip()
    if x.endswith(".fna"):
        x = x[:-4]
    return x


def validate_binary_dense(a: np.ndarray, name: str) -> None:
    vals = np.unique(a)
    if not set(vals.tolist()).issubset({0, 1}):
        raise RuntimeError(f"{name} is not binary 0/1. Values include {vals[:20]!r}")


def load_folds(path: Path, logger: Logger) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {
        "Genome ID", "Phenotype", "y", "MLST", "Genomic Cluster",
        "random_fold", "genomic_cluster_fold", "mlst_fold",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Fold file missing columns: {sorted(missing)}")

    if len(df) != EXPECTED_COHORT_N:
        raise RuntimeError(
            f"Expected {EXPECTED_COHORT_N:,} fold rows, found {len(df):,}."
        )
    if df["Genome ID"].nunique() != EXPECTED_COHORT_N:
        raise RuntimeError("Genome ID values are not unique in fold file.")

    y = pd.to_numeric(df["y"], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise RuntimeError("y must contain only 0/1.")
    df["y"] = y

    for scheme, col in SCHEME_TO_COLUMN.items():
        f = pd.to_numeric(df[col], errors="raise").astype(int)
        if set(f.unique()) != {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{col} must contain exactly folds 0..4.")
        df[col] = f

    logger(
        f"Fold cohort PASS: n={len(df):,}; "
        f"R={int(df['y'].sum()):,}; S={int((1-df['y']).sum()):,}."
    )
    return df


def make_fold_specs(folds: pd.DataFrame, schemes: Iterable[str]) -> List[FoldSpec]:
    specs = []
    for scheme in schemes:
        col = SCHEME_TO_COLUMN[scheme]
        fvals = folds[col].to_numpy(dtype=int)
        for fold in range(5):
            test_idx = np.flatnonzero(fvals == fold)
            train_idx = np.flatnonzero(fvals != fold)
            if len(test_idx) == 0 or len(train_idx) == 0:
                raise RuntimeError(f"Empty train/test split for {scheme} fold {fold}.")
            specs.append(FoldSpec(scheme, fold, train_idx, test_idx))
    return specs


def load_known_amr(
    path: Path, cohort_ids: List[str], logger: Logger
) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, dtype={"Genome ID": str})
    if "Genome ID" not in df.columns:
        raise RuntimeError("Known-AMR matrix lacks Genome ID.")
    if df["Genome ID"].nunique() != len(df):
        raise RuntimeError("Known-AMR Genome ID values are not unique.")

    feature_cols = [c for c in df.columns if c != "Genome ID"]
    if len(feature_cols) != EXPECTED_KNOWN_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_KNOWN_FEATURES} known-AMR features, "
            f"found {len(feature_cols)}."
        )
    if "emrD" in feature_cols:
        raise RuntimeError("emrD unexpectedly present in final known-AMR matrix.")

    indexed = df.set_index("Genome ID", drop=True)
    missing = [gid for gid in cohort_ids if gid not in indexed.index]
    if missing:
        raise RuntimeError(
            f"Known-AMR matrix missing {len(missing)} cohort genomes; first={missing[:10]}"
        )
    aligned = indexed.loc[cohort_ids, feature_cols]
    X = aligned.to_numpy(dtype=np.uint8, copy=True)
    validate_binary_dense(X, "Known-AMR matrix")
    logger(f"Known-AMR aligned: {X.shape[0]:,} x {X.shape[1]:,}.")
    return X, feature_cols


def load_pangenome(
    npz_path: Path, row_index_path: Path, cohort_ids: List[str], logger: Logger
) -> sparse.csr_matrix:
    X = load_npz(npz_path).tocsr()
    rows = pd.read_csv(row_index_path, dtype=str)
    if "Genome ID" not in rows.columns:
        raise RuntimeError("Pangenome row index lacks Genome ID.")
    if len(rows) != X.shape[0]:
        raise RuntimeError("Pangenome row-index length does not match matrix rows.")
    if X.shape[1] != EXPECTED_PANGENOME_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_PANGENOME_FEATURES:,} pangenome features, "
            f"found {X.shape[1]:,}."
        )

    row_ids = rows["Genome ID"].astype(str).tolist()
    if len(set(row_ids)) != len(row_ids):
        raise RuntimeError("Pangenome row index contains duplicate Genome IDs.")

    if row_ids == cohort_ids:
        aligned = X
    else:
        pos = {gid: i for i, gid in enumerate(row_ids)}
        missing = [gid for gid in cohort_ids if gid not in pos]
        if missing:
            raise RuntimeError(
                f"Pangenome matrix missing {len(missing)} cohort genomes; "
                f"first={missing[:10]}"
            )
        aligned = X[[pos[gid] for gid in cohort_ids], :].tocsr()

    if aligned.shape != (EXPECTED_COHORT_N, EXPECTED_PANGENOME_FEATURES):
        raise RuntimeError(f"Unexpected aligned pangenome shape: {aligned.shape}")
    if aligned.nnz and (
        aligned.data.min() < 0
        or aligned.data.max() > 1
        or not np.all(np.isin(aligned.data, [0, 1]))
    ):
        raise RuntimeError("Pangenome matrix is not binary 0/1.")

    logger(
        f"Pangenome aligned: {aligned.shape[0]:,} x {aligned.shape[1]:,}; "
        f"nnz={aligned.nnz:,}."
    )
    return aligned


def validate_unitig_layout(
    bin_path: Path,
    index_path: Path,
    sample_order_path: Path,
    cohort_ids: List[str],
    logger: Logger,
) -> None:
    if bin_path.stat().st_size != EXPECTED_UNITIG_BYTES:
        raise RuntimeError(
            f"Unitig binary size mismatch: expected {EXPECTED_UNITIG_BYTES:,}, "
            f"found {bin_path.stat().st_size:,}."
        )
    if EXPECTED_UNITIG_FEATURES * UNITIG_BYTES_PER_FEATURE != EXPECTED_UNITIG_BYTES:
        raise RuntimeError("Internal expected unitig size constants are inconsistent.")

    order = [
        normalize_unitig_sample_name(x)
        for x in sample_order_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(order) != EXPECTED_COHORT_N:
        raise RuntimeError(
            f"Unitig sample-order count {len(order)} != {EXPECTED_COHORT_N}."
        )
    if order != cohort_ids:
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(order, cohort_ids)) if a != b),
            None,
        )
        raise RuntimeError(
            f"Unitig sample order does not exactly match File07 cohort order. "
            f"First mismatch index={mismatch}."
        )

    idx_head = pd.read_csv(index_path, sep="\t", nrows=5, dtype=str)
    required = {
        "Feature Index", "Raw Record", "Sequence",
        "Prevalence", "Minor State Count", "Byte Offset",
    }
    if required - set(idx_head.columns):
        raise RuntimeError(
            f"Unitig feature index schema mismatch: {list(idx_head.columns)}"
        )
    offsets = pd.to_numeric(idx_head["Byte Offset"], errors="raise").astype(int).tolist()
    expected_offsets = [i * UNITIG_BYTES_PER_FEATURE for i in range(len(offsets))]
    if offsets != expected_offsets:
        raise RuntimeError(
            f"Unitig byte offsets do not begin as expected: {offsets}"
        )
    logger(
        "Unitig layout PASS: "
        f"{EXPECTED_UNITIG_FEATURES:,} features x "
        f"{UNITIG_BYTES_PER_FEATURE} bytes/feature."
    )


def validate_unitig_bitorder_against_pyseer(
    bin_path: Path,
    pyseer_path: Path,
    cohort_ids: List[str],
    logger: Logger,
    n_features: int = 8,
) -> None:
    """
    Validate sample-bit orientation by comparing the first retained bit-packed
    features against the original pyseer membership lists. This is a semantic
    orientation check, not merely a popcount check.
    """
    mm = np.memmap(bin_path, mode="r", dtype=np.uint8)
    packed = np.asarray(
        mm[: n_features * UNITIG_BYTES_PER_FEATURE]
    ).reshape(n_features, UNITIG_BYTES_PER_FEATURE)
    decoded = np.unpackbits(packed, axis=1, bitorder="little")[:, :EXPECTED_COHORT_N]

    id_to_pos = {f"{gid}.fna": i for i, gid in enumerate(cohort_ids)}
    matched = []
    with gzip.open(pyseer_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(matched) >= n_features:
                break
            line = line.rstrip("\n")
            if "|" not in line:
                continue
            seq, members = line.split("|", 1)
            member_tokens_raw = [m.strip() for m in members.split() if m.strip()]
            if not member_tokens_raw:
                continue

            # unitig-caller/pyseer membership tokens are emitted as
            # "<sample_name>:1" (e.g. "1284787.3.fna:1").  The bit-packed
            # builder stores biological presence only, so strip the explicit
            # ":1" presence suffix before matching against the fixed sample order.
            member_tokens = []
            bad_suffix = []
            for token in member_tokens_raw:
                base, sep, value = token.rpartition(":")
                if sep:
                    if value == "1" and base:
                        token = base
                    else:
                        bad_suffix.append(token)
                member_tokens.append(token)

            if bad_suffix:
                raise RuntimeError(
                    "Unexpected pyseer membership suffix(es) in orientation audit: "
                    f"{bad_suffix[:5]}"
                )
            if len(set(member_tokens)) != len(member_tokens):
                raise RuntimeError(
                    "Duplicate sample membership detected within a pyseer unitig record."
                )

            prevalence = len(member_tokens)
            minor = min(prevalence, EXPECTED_COHORT_N - prevalence)
            if minor < 5:
                continue
            row = np.zeros(EXPECTED_COHORT_N, dtype=np.uint8)
            bad = []
            for token in member_tokens:
                if token not in id_to_pos:
                    bad.append(token)
                else:
                    row[id_to_pos[token]] = 1
            if bad:
                raise RuntimeError(
                    f"Unexpected sample token(s) in pyseer orientation audit: {bad[:5]}"
                )
            matched.append(row)

    if len(matched) != n_features:
        raise RuntimeError(
            f"Could only recover {len(matched)}/{n_features} retained pyseer features "
            "for unitig orientation audit."
        )
    expected = np.stack(matched, axis=0)
    if not np.array_equal(decoded, expected):
        # Try big-endian only to provide a useful diagnostic.
        decoded_big = np.unpackbits(
            packed, axis=1, bitorder="big"
        )[:, :EXPECTED_COHORT_N]
        if np.array_equal(decoded_big, expected):
            raise RuntimeError(
                "Unitig bit-order is BIG, but File08 is configured for LITTLE. "
                "Do not continue until decoder is updated."
            )
        raise RuntimeError(
            "Unitig semantic orientation audit failed for both little and big bit order."
        )
    logger(
        f"Unitig semantic orientation PASS: first {n_features} retained features "
        "match original pyseer membership exactly (little-endian bit order)."
    )


def chi2_binary_counts(
    present_pos: np.ndarray,
    present_neg: np.ndarray,
    n_pos: int,
    n_neg: int,
) -> np.ndarray:
    a = present_pos.astype(np.float64)
    b = present_neg.astype(np.float64)
    c = n_pos - a
    d = n_neg - b
    n = float(n_pos + n_neg)

    # Pearson chi-square for a 2x2 table.
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    numer = n * (a * d - b * c) ** 2
    out = np.zeros_like(numer, dtype=np.float64)
    good = denom > 0
    out[good] = numer[good] / denom[good]
    return out


def merge_topk(
    current_idx: np.ndarray,
    current_score: np.ndarray,
    block_idx: np.ndarray,
    block_score: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(block_score)
    if not np.any(finite):
        return current_idx, current_score

    b_idx = block_idx[finite]
    b_score = block_score[finite]

    if len(b_score) > k:
        take = np.argpartition(b_score, -k)[-k:]
        b_idx = b_idx[take]
        b_score = b_score[take]

    if len(current_idx):
        idx = np.concatenate([current_idx, b_idx])
        score = np.concatenate([current_score, b_score])
    else:
        idx = b_idx
        score = b_score

    if len(score) > k:
        take = np.argpartition(score, -k)[-k:]
        idx = idx[take]
        score = score[take]

    # Deterministic ordering: descending score, then ascending feature index.
    order = np.lexsort((idx, -score))
    return idx[order], score[order]


def pangenome_selected_path(base: Path, spec: FoldSpec) -> Path:
    return base / spec.scheme / f"fold_{spec.fold}" / "pangenome_selected.npz"


def unitig_selected_path(base: Path, spec: FoldSpec) -> Path:
    return base / spec.scheme / f"fold_{spec.fold}" / "unitig_selected.npz"


def ensure_pangenome_feature_selection(
    X: sparse.csr_matrix,
    y: np.ndarray,
    specs: List[FoldSpec],
    k: int,
    selected_base: Path,
    logger: Logger,
) -> None:
    logger("START pangenome train-only feature selection")
    for i, spec in enumerate(specs, start=1):
        out = pangenome_selected_path(selected_base, spec)
        if out.exists():
            try:
                z = np.load(out)
                if (
                    len(z["feature_index"]) == min(k, X.shape[1])
                    and np.all(np.isfinite(z["score"]))
                ):
                    logger(
                        f"  SKIP {spec.scheme} fold {spec.fold}: validated selection exists."
                    )
                    continue
            except Exception:
                pass

        if STOP_REQUESTED:
            raise KeyboardInterrupt

        X_train = X[spec.train_idx]
        y_train = y[spec.train_idx]
        scores, _ = chi2(X_train, y_train)
        scores = np.asarray(scores, dtype=np.float64)
        scores[~np.isfinite(scores)] = -np.inf
        kk = min(k, X.shape[1])
        take = np.argpartition(scores, -kk)[-kk:]
        take = take[np.lexsort((take, -scores[take]))]

        atomic_save_npz(
            out,
            feature_index=take.astype(np.int32),
            score=scores[take].astype(np.float32),
            train_n=np.array([len(spec.train_idx)], dtype=np.int32),
            test_n=np.array([len(spec.test_idx)], dtype=np.int32),
        )
        logger(
            f"  {i}/{len(specs)} {spec.scheme} fold {spec.fold}: "
            f"selected {len(take):,}/{X.shape[1]:,}."
        )
    logger("DONE pangenome feature selection")


def unitig_scan_state_path(ckpt_dir: Path, smoke: bool) -> Path:
    return ckpt_dir / ("unitig_selection_scan_smoke.npz" if smoke else "unitig_selection_scan.npz")


def ensure_unitig_feature_selection(
    bin_path: Path,
    y: np.ndarray,
    specs: List[FoldSpec],
    k: int,
    selected_base: Path,
    ckpt_dir: Path,
    logger: Logger,
    block_features: int,
    max_features: int | None,
    smoke: bool,
) -> None:
    """
    Scan the bit-packed matrix once and update top-k train-only chi-square
    features for every requested scheme/fold in parallel.
    """
    total_features = EXPECTED_UNITIG_FEATURES if max_features is None else min(
        EXPECTED_UNITIG_FEATURES, max_features
    )

    # If every final selection exists and validates, no scan is needed.
    all_ok = True
    for spec in specs:
        p = unitig_selected_path(selected_base, spec)
        if not p.exists():
            all_ok = False
            break
        try:
            z = np.load(p)
            if len(z["feature_index"]) != min(k, total_features):
                all_ok = False
                break
        except Exception:
            all_ok = False
            break
    if all_ok:
        logger("SKIP unitig feature-selection scan: all validated selections exist.")
        return

    state_path = unitig_scan_state_path(ckpt_dir, smoke)
    start_feature = 0
    top_idx: Dict[str, np.ndarray] = {}
    top_score: Dict[str, np.ndarray] = {}

    keys = [f"{s.scheme}__{s.fold}" for s in specs]
    if state_path.exists():
        z = np.load(state_path, allow_pickle=False)
        start_feature = int(z["next_feature"][0])
        saved_total = int(z["total_features"][0])
        saved_k = int(z["k"][0])
        if saved_total != total_features or saved_k != k:
            raise RuntimeError(
                "Existing unitig selection checkpoint is incompatible with current "
                "feature limit or k. Use a separate smoke/full run or remove only the "
                "incompatible checkpoint after inspection."
            )
        for key in keys:
            top_idx[key] = z[f"{key}__idx"].astype(np.int64)
            top_score[key] = z[f"{key}__score"].astype(np.float64)
        logger(
            f"RESUME unitig selection scan at feature {start_feature:,}/{total_features:,}."
        )
    else:
        for key in keys:
            top_idx[key] = np.empty(0, dtype=np.int64)
            top_score[key] = np.empty(0, dtype=np.float64)
        logger(
            f"START unitig train-only feature-selection scan: "
            f"{total_features:,} features, block={block_features:,}, "
            f"{len(specs)} outer folds evaluated per decoded block."
        )

    mm = np.memmap(bin_path, mode="r", dtype=np.uint8)
    t0 = time.time()
    last_save = time.time()

    for start in range(start_feature, total_features, block_features):
        if STOP_REQUESTED:
            raise KeyboardInterrupt

        end = min(total_features, start + block_features)
        n = end - start
        packed = np.asarray(
            mm[
                start * UNITIG_BYTES_PER_FEATURE:
                end * UNITIG_BYTES_PER_FEATURE
            ]
        ).reshape(n, UNITIG_BYTES_PER_FEATURE)

        bits = np.unpackbits(
            packed, axis=1, bitorder="little"
        )[:, :EXPECTED_COHORT_N]

        block_global_idx = np.arange(start, end, dtype=np.int64)

        for spec in specs:
            key = f"{spec.scheme}__{spec.fold}"
            train_idx = spec.train_idx
            y_train = y[train_idx]
            pos_idx = train_idx[y_train == 1]
            neg_idx = train_idx[y_train == 0]
            n_pos = len(pos_idx)
            n_neg = len(neg_idx)

            present_pos = bits[:, pos_idx].sum(axis=1, dtype=np.int32)
            present_neg = bits[:, neg_idx].sum(axis=1, dtype=np.int32)
            scores = chi2_binary_counts(present_pos, present_neg, n_pos, n_neg)

            top_idx[key], top_score[key] = merge_topk(
                top_idx[key],
                top_score[key],
                block_global_idx,
                scores,
                k,
            )

        done = end
        elapsed = time.time() - t0
        rate = max(done - start_feature, 1) / max(elapsed, 1e-9)
        remain = total_features - done
        eta = remain / max(rate, 1e-9)
        logger(
            f"  unitig scan {done:,}/{total_features:,} "
            f"({done/total_features:.1%}); "
            f"rate={rate:,.0f} features/s; ETA={human_seconds(eta)}"
        )

        # Crash-safe checkpoint approximately every 60 seconds or at end.
        if (time.time() - last_save >= 60) or done == total_features:
            payload = {
                "next_feature": np.array([done], dtype=np.int64),
                "total_features": np.array([total_features], dtype=np.int64),
                "k": np.array([k], dtype=np.int64),
            }
            for key in keys:
                payload[f"{key}__idx"] = top_idx[key].astype(np.int64)
                payload[f"{key}__score"] = top_score[key].astype(np.float32)
            atomic_save_npz(state_path, **payload)
            last_save = time.time()

        del bits, packed

    # Promote fold-specific selections.
    for spec in specs:
        key = f"{spec.scheme}__{spec.fold}"
        out = unitig_selected_path(selected_base, spec)
        atomic_save_npz(
            out,
            feature_index=top_idx[key].astype(np.int32),
            score=top_score[key].astype(np.float32),
            train_n=np.array([len(spec.train_idx)], dtype=np.int32),
            test_n=np.array([len(spec.test_idx)], dtype=np.int32),
            feature_universe_n=np.array([total_features], dtype=np.int64),
        )
    logger(
        f"DONE unitig feature selection in {human_seconds(time.time()-t0)}."
    )


def decode_selected_unitigs(
    bin_path: Path, feature_indices: np.ndarray
) -> np.ndarray:
    """
    Return sample x selected-feature uint8 matrix.
    """
    idx = np.asarray(feature_indices, dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= EXPECTED_UNITIG_FEATURES):
        raise RuntimeError("Selected unitig index out of range.")
    mm = np.memmap(bin_path, mode="r", dtype=np.uint8)
    byte_rows = np.asarray(
        mm.reshape(EXPECTED_UNITIG_FEATURES, UNITIG_BYTES_PER_FEATURE)[idx]
    )
    bits = np.unpackbits(
        byte_rows, axis=1, bitorder="little"
    )[:, :EXPECTED_COHORT_N]
    return np.ascontiguousarray(bits.T, dtype=np.uint8)


def make_model(name: str, model_threads: int):
    if name == "logreg_l2":
        return LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_SEED,
        )
    if name == "rbf_svm":
        return SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=False,
            cache_size=2048,
            random_state=RANDOM_SEED,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=model_threads,
            random_state=RANDOM_SEED,
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
    raise KeyError(name)


def prediction_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return np.asarray(p[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    raise RuntimeError(f"Model {type(model).__name__} exposes no score method.")


def task_key(scheme: str, fold: int, representation: str, model: str) -> str:
    return f"{scheme}__fold{fold}__{representation}__{model}"


def validate_task_checkpoint(
    path: Path,
    expected_test_idx: np.ndarray,
    expected_n_features: int,
) -> bool:
    if not path.exists():
        return False
    try:
        z = np.load(path, allow_pickle=False)
        if not np.array_equal(z["test_idx"].astype(int), expected_test_idx.astype(int)):
            return False
        if int(z["n_features"][0]) != expected_n_features:
            return False
        n = len(expected_test_idx)
        if len(z["score"]) != n or len(z["pred"]) != n:
            return False
        if not np.all(np.isfinite(z["score"])):
            return False
        if not set(np.unique(z["pred"]).tolist()).issubset({0, 1}):
            return False
        return True
    except Exception:
        return False


def fit_and_checkpoint_task(
    task_path: Path,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    spec: FoldSpec,
    representation: str,
    model_threads: int,
    logger: Logger,
) -> None:
    expected_features = X.shape[1]
    if validate_task_checkpoint(task_path, spec.test_idx, expected_features):
        logger(
            f"    SKIP {representation}/{model_name}: validated task checkpoint."
        )
        return

    if STOP_REQUESTED:
        raise KeyboardInterrupt

    X_train = np.asarray(X[spec.train_idx], dtype=np.float32, order="C")
    X_test = np.asarray(X[spec.test_idx], dtype=np.float32, order="C")
    y_train = y[spec.train_idx]

    if len(np.unique(y_train)) != 2:
        raise RuntimeError(
            f"Training fold has only one class: {spec.scheme} fold {spec.fold}"
        )

    model = make_model(model_name, model_threads)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    score = prediction_score(model, X_test)
    pred = np.asarray(model.predict(X_test), dtype=np.uint8)

    atomic_save_npz(
        task_path,
        test_idx=spec.test_idx.astype(np.int32),
        score=score.astype(np.float64),
        pred=pred.astype(np.uint8),
        n_features=np.array([expected_features], dtype=np.int32),
        fit_seconds=np.array([fit_seconds], dtype=np.float64),
    )
    logger(
        f"    PASS {representation}/{model_name}: "
        f"features={expected_features:,}; fit={human_seconds(fit_seconds)}."
    )


def assemble_representation(
    representation: str,
    known: np.ndarray,
    pangenome_selected: np.ndarray | None,
    unitig_selected: np.ndarray | None,
) -> np.ndarray:
    if representation == "known_amr":
        return known
    if representation == "pangenome":
        assert pangenome_selected is not None
        return pangenome_selected
    if representation == "unitig":
        assert unitig_selected is not None
        return unitig_selected
    if representation == "known_plus_pangenome":
        assert pangenome_selected is not None
        return np.concatenate([known, pangenome_selected], axis=1)
    if representation == "known_plus_unitig":
        assert unitig_selected is not None
        return np.concatenate([known, unitig_selected], axis=1)
    if representation == "known_plus_pangenome_plus_unitig":
        assert pangenome_selected is not None and unitig_selected is not None
        return np.concatenate([known, pangenome_selected, unitig_selected], axis=1)
    raise KeyError(representation)


def aggregate_outputs(
    root: Path,
    folds: pd.DataFrame,
    specs: List[FoldSpec],
    task_dir: Path,
    selected_base: Path,
    primary_models: Tuple[str, ...],
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    task_rows = []
    fs_rows = []

    cohort_ids = folds["Genome ID"].astype(str).to_numpy()
    y = folds["y"].to_numpy(dtype=int)

    for spec in specs:
        psel = np.load(pangenome_selected_path(selected_base, spec))
        usel = np.load(unitig_selected_path(selected_base, spec))

        for source, z in [("pangenome", psel), ("unitig", usel)]:
            idx = z["feature_index"].astype(int)
            score = z["score"].astype(float)
            for rank, (fi, sc) in enumerate(zip(idx, score), start=1):
                fs_rows.append(
                    {
                        "scheme": spec.scheme,
                        "fold": spec.fold,
                        "representation": source,
                        "rank": rank,
                        "feature_index": fi,
                        "chi2_score_train_only": sc,
                    }
                )

        task_plan = []
        for rep in PRIMARY_REPRESENTATIONS:
            for model in primary_models:
                task_plan.append((rep, model))
        for rep in INCREMENTAL_REPRESENTATIONS:
            task_plan.append((rep, "logreg_l2"))

        for rep, model in task_plan:
            p = task_dir / f"{task_key(spec.scheme, spec.fold, rep, model)}.npz"
            if not p.exists():
                raise RuntimeError(f"Missing task checkpoint during aggregation: {p}")
            z = np.load(p)
            test_idx = z["test_idx"].astype(int)
            score = z["score"].astype(float)
            pred = z["pred"].astype(int)
            n_features = int(z["n_features"][0])
            fit_seconds = float(z["fit_seconds"][0])

            for j, idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "Genome ID": cohort_ids[idx],
                        "y": int(y[idx]),
                        "scheme": spec.scheme,
                        "fold": spec.fold,
                        "representation": rep,
                        "model": model,
                        "score": float(score[j]),
                        "pred": int(pred[j]),
                    }
                )

            task_rows.append(
                {
                    "scheme": spec.scheme,
                    "fold": spec.fold,
                    "representation": rep,
                    "model": model,
                    "test_n": len(test_idx),
                    "n_features": n_features,
                    "fit_seconds": fit_seconds,
                    "checkpoint": str(p.relative_to(root)),
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    task_df = pd.DataFrame(task_rows)
    fs_df = pd.DataFrame(fs_rows)

    # Strong OOF completeness audit relative to the ACTIVE fold specs.
    # In a full run, each validation scheme has all five outer folds and the
    # expected OOF coverage is therefore the complete 4,227-sample cohort.
    # In smoke mode, only one fold is intentionally active, so the correct
    # expected coverage is that fold's test set rather than all 4,227 samples.
    expected_idx_by_scheme = {}
    for scheme in {s.scheme for s in specs}:
        scheme_specs = [s for s in specs if s.scheme == scheme]
        expected_idx = np.concatenate([s.test_idx for s in scheme_specs]).astype(int)
        if len(np.unique(expected_idx)) != len(expected_idx):
            raise RuntimeError(
                f"Active outer test folds overlap for scheme {scheme}."
            )
        expected_idx_by_scheme[scheme] = expected_idx

    for (scheme, rep, model), g in pred_df.groupby(
        ["scheme", "representation", "model"], sort=False
    ):
        expected_idx = expected_idx_by_scheme[scheme]
        expected_ids = set(cohort_ids[expected_idx].tolist())

        if len(g) != len(expected_idx):
            raise RuntimeError(
                f"OOF prediction count mismatch for {scheme}/{rep}/{model}: "
                f"{len(g)} != {len(expected_idx)} active-test samples"
            )
        if g["Genome ID"].nunique() != len(expected_idx):
            raise RuntimeError(
                f"Duplicate OOF Genome IDs for {scheme}/{rep}/{model}."
            )
        observed_ids = set(g["Genome ID"].astype(str).tolist())
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)[:10]
            extra = sorted(observed_ids - expected_ids)[:10]
            raise RuntimeError(
                f"OOF Genome-ID coverage mismatch for {scheme}/{rep}/{model}; "
                f"missing={missing}, extra={extra}."
            )

    logger(
        f"Aggregation PASS: predictions={len(pred_df):,}; "
        f"tasks={len(task_df):,}; selected-feature rows={len(fs_df):,}."
    )
    return pred_df, task_df, fs_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--pangenome-k", type=int, default=2000)
    parser.add_argument("--unitig-k", type=int, default=2000)
    parser.add_argument("--unitig-block-features", type=int, default=4096)
    parser.add_argument("--model-threads", type=int, default=6)
    parser.add_argument(
        "--schemes",
        default="random_stratified,genomic_cluster_aware,mlst_aware",
        help="Comma-separated validation schemes."
    )
    parser.add_argument(
        "--models",
        default="logreg_l2,rbf_svm,extra_trees,hist_gb",
        help="Comma-separated primary-representation model families."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Cheap end-to-end smoke test: random_stratified fold 0 only, "
            "first 100,000 unitigs, k<=200, and logreg_l2 only. "
            "Uses separate smoke checkpoints and does not contaminate full outputs."
        ),
    )
    args = parser.parse_args()

    if args.pangenome_k < 1 or args.unitig_k < 1:
        raise SystemExit("Feature k values must be >=1.")
    if not (256 <= args.unitig_block_features <= 32768):
        raise SystemExit("--unitig-block-features must be between 256 and 32768.")
    if not (1 <= args.model_threads <= 12):
        raise SystemExit("--model-threads must be between 1 and 12.")

    schemes = tuple(x.strip() for x in args.schemes.split(",") if x.strip())
    bad_schemes = set(schemes) - set(SCHEME_TO_COLUMN)
    if bad_schemes:
        raise SystemExit(f"Unknown schemes: {sorted(bad_schemes)}")

    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    bad_models = set(models) - set(MODEL_NAMES)
    if bad_models:
        raise SystemExit(f"Unknown models: {sorted(bad_models)}")

    if args.smoke:
        schemes = ("random_stratified",)
        models = ("logreg_l2",)
        pangenome_k = min(args.pangenome_k, 200)
        unitig_k = min(args.unitig_k, 200)
        unitig_max_features = 100000
        namespace = "model_training_smoke"
    else:
        pangenome_k = args.pangenome_k
        unitig_k = args.unitig_k
        unitig_max_features = None
        namespace = "model_training"

    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else find_project_root(Path.cwd())
    )

    paths = {
        "folds": root / "data/splits/file07_outer_folds.csv",
        "known": root / "data/features/known_amr/X_known_amr_final.csv",
        "pangenome": root / "data/features/pangenome/X_pangenome_plfam_binary.npz",
        "pangenome_rows": root / "data/features/pangenome/X_pangenome_row_index.csv",
        "unitig_bin": root / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin",
        "unitig_idx": root / "data/features/sequence_variation/X_sequence_variation_unitig_feature_index.tsv",
        "unitig_pyseer": root / "data/features/sequence_variation/unitig_full_k31.pyseer.gz",
        "unitig_order": root / "checkpoints/sequence_variation/unitig_full_k31_sample_order.txt",
    }

    ckpt_dir = root / "checkpoints" / namespace
    selected_base = ckpt_dir / "selected_features"
    task_dir = ckpt_dir / "tasks"
    log_path = root / "logs" / ("08_model_training_smoke.log" if args.smoke else "08_model_training.log")
    failure_path = ckpt_dir / "08_last_failure.json"
    lock_path = ckpt_dir / "08_model_training.lock"
    final_summary = ckpt_dir / "file08_final_summary.json"

    if args.smoke:
        out_dir = root / "data/modeling/smoke"
    else:
        out_dir = root / "data/modeling"

    pred_out = out_dir / "file08_oof_predictions.csv"
    task_out = out_dir / "file08_task_manifest.csv"
    fs_out = out_dir / "file08_feature_selection_manifest.csv"

    logger = Logger(log_path)
    stage = "startup"
    started = time.time()

    acquire_lock(lock_path)

    try:
        logger(f"File08 script version: {SCRIPT_VERSION}")
        logger(f"Mode: {'SMOKE' if args.smoke else 'FULL'}")
        logger(f"Project root: {root}")
        logger(f"Python: {sys.executable}")
        logger(
            f"Parameters: pangenome_k={pangenome_k:,}; unitig_k={unitig_k:,}; "
            f"unitig_block={args.unitig_block_features:,}; model_threads={args.model_threads}"
        )
        logger(f"Schemes: {', '.join(schemes)}")
        logger(f"Primary models: {', '.join(models)}")

        for name, path in paths.items():
            if not path.is_file():
                raise RuntimeError(f"Required input missing ({name}): {path}")
            logger(f"Input OK: {path.relative_to(root)}")

        stage = "preflight"
        folds = load_folds(paths["folds"], logger)
        cohort_ids = folds["Genome ID"].astype(str).tolist()
        y = folds["y"].to_numpy(dtype=np.uint8)

        known, known_features = load_known_amr(paths["known"], cohort_ids, logger)
        pangenome = load_pangenome(
            paths["pangenome"], paths["pangenome_rows"], cohort_ids, logger
        )
        validate_unitig_layout(
            paths["unitig_bin"],
            paths["unitig_idx"],
            paths["unitig_order"],
            cohort_ids,
            logger,
        )
        validate_unitig_bitorder_against_pyseer(
            paths["unitig_bin"],
            paths["unitig_pyseer"],
            cohort_ids,
            logger,
            n_features=8,
        )

        specs = make_fold_specs(folds, schemes)
        if args.smoke:
            specs = [s for s in specs if s.scheme == "random_stratified" and s.fold == 0]
        logger(f"Outer fold specifications active: {len(specs)}")

        # Audit that group-defining metadata are not part of predictive inputs.
        forbidden = {"Genome ID", "MLST", "Genomic Cluster", "Phenotype", "y"}
        if forbidden & set(known_features):
            raise RuntimeError(
                f"Leakage-risk columns found in known-AMR predictors: "
                f"{sorted(forbidden & set(known_features))}"
            )
        logger("Predictor leakage audit PASS: identifiers/MLST/clusters/phenotype excluded.")

        stage = "pangenome_feature_selection"
        ensure_pangenome_feature_selection(
            pangenome, y, specs, pangenome_k, selected_base, logger
        )

        stage = "unitig_feature_selection"
        ensure_unitig_feature_selection(
            paths["unitig_bin"],
            y,
            specs,
            unitig_k,
            selected_base,
            ckpt_dir,
            logger,
            args.unitig_block_features,
            unitig_max_features,
            args.smoke,
        )

        stage = "model_training"
        task_dir.mkdir(parents=True, exist_ok=True)

        for spec_i, spec in enumerate(specs, start=1):
            logger(
                f"START outer task group {spec_i}/{len(specs)}: "
                f"{spec.scheme} fold {spec.fold}"
            )

            psel = np.load(pangenome_selected_path(selected_base, spec))
            pidx = psel["feature_index"].astype(int)
            Xp = np.ascontiguousarray(
                pangenome[:, pidx].toarray(), dtype=np.uint8
            )

            usel = np.load(unitig_selected_path(selected_base, spec))
            uidx = usel["feature_index"].astype(int)
            Xu = decode_selected_unitigs(paths["unitig_bin"], uidx)

            if Xp.shape != (EXPECTED_COHORT_N, len(pidx)):
                raise RuntimeError("Materialized pangenome selected matrix shape mismatch.")
            if Xu.shape != (EXPECTED_COHORT_N, len(uidx)):
                raise RuntimeError("Materialized unitig selected matrix shape mismatch.")

            # Primary representations: all requested model families.
            for rep in PRIMARY_REPRESENTATIONS:
                Xrep = assemble_representation(rep, known, Xp, Xu)
                for model_name in models:
                    p = task_dir / f"{task_key(spec.scheme, spec.fold, rep, model_name)}.npz"
                    fit_and_checkpoint_task(
                        p,
                        model_name,
                        Xrep,
                        y,
                        spec,
                        rep,
                        args.model_threads,
                        logger,
                    )

            # Incremental feature combinations: fixed logistic regression only.
            for rep in INCREMENTAL_REPRESENTATIONS:
                Xrep = assemble_representation(rep, known, Xp, Xu)
                p = task_dir / f"{task_key(spec.scheme, spec.fold, rep, 'logreg_l2')}.npz"
                fit_and_checkpoint_task(
                    p,
                    "logreg_l2",
                    Xrep,
                    y,
                    spec,
                    rep,
                    args.model_threads,
                    logger,
                )

            logger(
                f"DONE outer task group: {spec.scheme} fold {spec.fold}"
            )

        stage = "aggregation"
        pred_df, task_df, fs_df = aggregate_outputs(
            root,
            folds,
            specs,
            task_dir,
            selected_base,
            models,
            logger,
        )

        atomic_write_csv(pred_out, pred_df)
        atomic_write_csv(task_out, task_df)
        atomic_write_csv(fs_out, fs_df)

        summary = {
            "script_version": SCRIPT_VERSION,
            "status": "PASS",
            "mode": "SMOKE" if args.smoke else "FULL",
            "completed_utc": utc_now(),
            "cohort_n": EXPECTED_COHORT_N,
            "phenotype_resistant_n": int(y.sum()),
            "phenotype_susceptible_n": int((1 - y).sum()),
            "known_amr_features": EXPECTED_KNOWN_FEATURES,
            "pangenome_features_total": EXPECTED_PANGENOME_FEATURES,
            "pangenome_features_selected_per_outer_fold": pangenome_k,
            "unitig_features_total": (
                unitig_max_features if unitig_max_features is not None
                else EXPECTED_UNITIG_FEATURES
            ),
            "unitig_features_selected_per_outer_fold": unitig_k,
            "validation_schemes": list(schemes),
            "primary_models": list(models),
            "primary_representations": list(PRIMARY_REPRESENTATIONS),
            "incremental_representations": list(INCREMENTAL_REPRESENTATIONS),
            "incremental_model": "logreg_l2",
            "supervised_feature_selection_scope": "outer_training_fold_only",
            "outer_test_labels_used_for_selection_or_training": False,
            "fixed_model_hyperparameters": True,
            "prediction_rows": len(pred_df),
            "task_rows": len(task_df),
            "feature_selection_rows": len(fs_df),
            "outputs": {
                "oof_predictions": str(pred_out.relative_to(root)),
                "task_manifest": str(task_out.relative_to(root)),
                "feature_selection_manifest": str(fs_out.relative_to(root)),
            },
            "elapsed_seconds": time.time() - started,
        }
        atomic_write_text(final_summary, json.dumps(summary, indent=2) + "\n")

        if failure_path.exists():
            failure_path.unlink()

        logger("=" * 76)
        logger("FILE08 — MODEL TRAINING AND FEATURE SELECTION")
        logger("=" * 76)
        logger(f"Mode                          : {'SMOKE' if args.smoke else 'FULL'}")
        logger(f"Cohort                        : {EXPECTED_COHORT_N:,}")
        logger(f"Known-AMR features            : {EXPECTED_KNOWN_FEATURES:,}")
        logger(f"Pangenome selected/fold       : {pangenome_k:,}")
        logger(f"Unitig selected/fold          : {unitig_k:,}")
        logger(f"Validation schemes            : {', '.join(schemes)}")
        logger(f"Primary models                : {', '.join(models)}")
        logger("Supervised selection scope    : OUTER TRAINING FOLD ONLY")
        logger("Outer-test labels used        : NO")
        logger(f"Prediction rows               : {len(pred_df):,}")
        logger(f"Task checkpoints              : {len(task_df):,}")
        logger(f"Elapsed                       : {human_seconds(time.time()-started)}")
        logger(f"Final summary                 : {final_summary}")
        logger("FILE08 STATUS                 : PASS")
        logger("=" * 76)
        return 0

    except KeyboardInterrupt:
        failure = {
            "script_version": SCRIPT_VERSION,
            "status": "INTERRUPTED",
            "stage": stage,
            "time_utc": utc_now(),
            "message": (
                "User/system interruption. Valid feature-selection and task checkpoints "
                "are preserved. Re-run the same command to resume."
            ),
        }
        atomic_write_text(failure_path, json.dumps(failure, indent=2) + "\n")
        logger(
            "FILE08 INTERRUPTED. Valid checkpoints are preserved; "
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
        logger("FILE08 FAILED")
        logger(f"Stage : {stage}")
        logger(f"Error : {exc}")
        logger(f"Crash record: {failure_path}")
        logger(
            "Fix the cause and re-run the same command; validated checkpoints will be skipped."
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
