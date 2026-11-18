#!/usr/bin/env python3
"""
File10 — Explainability and Biological Validation
Version 1.0.1

Bug fix vs 1.0.0
----------------
- File07 MLST labels are stored as ST<number>; 10D now recognizes that exact format
  and audits the number of parsed exact MLST calls before within-lineage analysis.

Scientific role
---------------
File10 does NOT search for a "better" predictive model. Predictive performance
and reviewer-defense model robustness were locked in File08/File09/File09b/File09c.

File10 addresses interpretation/biology questions:
  10A. Locked-input, cohort, row-order and representation integrity.
  10B. Train-only feature-selection stability across folds/schemes.
  10C. Cross-fitted held-out effect estimates for stable broader features.
  10D. Population-structure-aware association (genomic cluster + MLST),
       within-lineage direction consistency, dominant-lineage sensitivity.
  10E. Redundancy/correlation structure and linkage to known-AMR features.
  10F. Cross-fitted SHAP attribution using ExtraTrees on lineage-aware splits.
  10G. Integrated biological-candidate tables (association, stability,
       known-AMR linkage, redundancy, SHAP).
  10H. Reviewer-defense matrix and manuscript-ready audit summary.

Interpretation guardrails
-------------------------
- Associations and SHAP values are NOT treated as causal evidence.
- Full-cohort population-adjusted tests are post-selection/exploratory.
- Cross-fitted held-out effects only use each fold's test labels after that
  feature was selected using that fold's training data.
- No predictive hyperparameter/model selection is performed here.
- No metadata, Genome ID, MLST or cluster labels are used as predictive X.
- Broader-feature biological mapping is limited by available local annotations;
  final literature/external sequence annotation should be performed only after
  candidate prioritization.

Operational design
------------------
- Atomic outputs.
- Block-level checkpoints.
- Heavy sub-tasks (cross-fit and SHAP) checkpoint individually.
- Live progress, elapsed time and ETA.
- Resume without recomputing completed tasks.
- Smoke mode uses a separate namespace and exercises all blocks on a smaller
  candidate set and one SHAP fold.

Usage
-----
SMOKE FIRST:
  python scripts/10_explainability_and_biological_validation.py --smoke --workers 8

FULL:
  python scripts/10_explainability_and_biological_validation.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from scipy.stats import chi2, norm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

SCRIPT_VERSION = "1.0.1"
DESIGN_ID = "file10_explainability_biology_v1"
RANDOM_SEED = 20260920

EXPECTED_N = 4227
EXPECTED_KNOWN_FEATURES = 668
EXPECTED_PANGENOME_FEATURES = 22685
EXPECTED_UNITIG_FEATURES = 4718475

SCHEME_TO_COL = {
    "random_stratified": "random_fold",
    "genomic_cluster_aware": "genomic_cluster_fold",
    "mlst_aware": "mlst_fold",
}

LINEAGE_AWARE_SCHEMES = ("genomic_cluster_aware", "mlst_aware")
BROAD_REPS = ("pangenome", "unitig")


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_seconds(sec: float) -> str:
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


def human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024
    return str(n)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    return start


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
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
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_csv_gz(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.csv.gz", dir=path.parent)
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


def atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.npz", dir=path.parent)
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


class Progress:
    def __init__(self, total: int, label: str):
        self.total = max(int(total), 1)
        self.label = label
        self.started = time.time()
        self.done = 0

    def update(self, detail: str = ""):
        self.done += 1
        elapsed = time.time() - self.started
        rate = self.done / elapsed if elapsed > 0 else 0
        remain = self.total - self.done
        eta = remain / rate if rate > 0 else float("nan")
        pct = 100 * self.done / self.total
        eta_text = human_seconds(eta) if np.isfinite(eta) else "?"
        log(
            f"[PROGRESS] {self.label}: {self.done}/{self.total} "
            f"({pct:5.1f}%) | elapsed={human_seconds(elapsed)} "
            f"| ETA={eta_text}"
            + (f" | {detail}" if detail else "")
        )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def block_marker_path(checkpoint_root: Path, block: str) -> Path:
    return checkpoint_root / "blocks" / f"{block}.json"


def block_done(checkpoint_root: Path, block: str, outputs: Iterable[Path]) -> bool:
    marker = block_marker_path(checkpoint_root, block)
    if not marker.is_file():
        return False
    try:
        d = read_json(marker)
    except Exception:
        return False
    if d.get("script_version") != SCRIPT_VERSION or d.get("design_id") != DESIGN_ID:
        return False
    if d.get("status") != "PASS":
        return False
    return all(Path(p).is_file() for p in outputs)


def mark_block(
    checkpoint_root: Path,
    block: str,
    outputs: Iterable[Path],
    elapsed: float,
    detail: dict | None = None,
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "design_id": DESIGN_ID,
        "block": block,
        "status": "PASS",
        "completed_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "outputs": [str(p) for p in outputs],
    }
    if detail:
        payload.update(detail)
    atomic_write_text(
        block_marker_path(checkpoint_root, block),
        json.dumps(payload, indent=2) + "\n",
    )


def task_marker_valid(path: Path, task_id: str, outputs: Iterable[Path]) -> bool:
    if not path.is_file():
        return False
    try:
        d = read_json(path)
    except Exception:
        return False
    return (
        d.get("script_version") == SCRIPT_VERSION
        and d.get("design_id") == DESIGN_ID
        and d.get("task_id") == task_id
        and d.get("status") == "PASS"
        and all(Path(p).is_file() for p in outputs)
    )


def mark_task(path: Path, task_id: str, outputs: Iterable[Path], elapsed: float, detail=None):
    d = {
        "script_version": SCRIPT_VERSION,
        "design_id": DESIGN_ID,
        "task_id": task_id,
        "status": "PASS",
        "completed_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "outputs": [str(p) for p in outputs],
    }
    if detail:
        d.update(detail)
    atomic_write_text(path, json.dumps(d, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Input loading / row alignment
# ---------------------------------------------------------------------------

def load_folds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Genome ID": str, "MLST": str, "Genomic Cluster": str})
    required = {
        "Genome ID", "y", "Phenotype", "MLST", "Genomic Cluster",
        "random_fold", "genomic_cluster_fold", "mlst_fold",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"File07 folds missing columns: {sorted(missing)}")
    if len(df) != EXPECTED_N or df["Genome ID"].nunique() != EXPECTED_N:
        raise RuntimeError(
            f"Unexpected common cohort: rows={len(df)}, unique={df['Genome ID'].nunique()}"
        )
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(np.uint8)
    for c in SCHEME_TO_COL.values():
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
        if set(df[c].unique()) != {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{c} is not exactly folds 0..4")
    return df


def load_known(path: Path, cohort_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, dtype={"Genome ID": str})
    features = [c for c in df.columns if c != "Genome ID"]
    if len(features) != EXPECTED_KNOWN_FEATURES:
        raise RuntimeError(
            f"Known-AMR feature count {len(features)} != {EXPECTED_KNOWN_FEATURES}"
        )
    if "emrD" in features:
        raise RuntimeError("Artifact emrD unexpectedly present.")
    if df["Genome ID"].duplicated().any():
        raise RuntimeError("Known-AMR matrix has duplicate Genome IDs.")
    idx = df.set_index("Genome ID")
    missing = [g for g in cohort_ids if g not in idx.index]
    if missing:
        raise RuntimeError(f"Known-AMR missing cohort IDs, first={missing[:5]}")
    X = idx.loc[cohort_ids, features].to_numpy(dtype=np.uint8)
    if not np.isin(X, [0, 1]).all():
        raise RuntimeError("Known-AMR matrix not binary.")
    return np.ascontiguousarray(X), features


def load_pangenome(
    matrix_path: Path,
    row_index_path: Path,
    cohort_ids: List[str],
) -> sparse.csr_matrix:
    X = load_npz(matrix_path).tocsr()
    rows = pd.read_csv(row_index_path, dtype={"Genome ID": str})
    if "Genome ID" not in rows.columns:
        raise RuntimeError("Pangenome row index missing Genome ID.")
    ids = rows["Genome ID"].astype(str).tolist()
    if len(ids) != X.shape[0] or len(set(ids)) != len(ids):
        raise RuntimeError("Invalid pangenome row index.")
    if X.shape[1] != EXPECTED_PANGENOME_FEATURES:
        raise RuntimeError(
            f"Pangenome feature count {X.shape[1]} != {EXPECTED_PANGENOME_FEATURES}"
        )
    if ids == cohort_ids:
        return X
    pos = {g: i for i, g in enumerate(ids)}
    missing = [g for g in cohort_ids if g not in pos]
    if missing:
        raise RuntimeError(f"Pangenome missing cohort IDs, first={missing[:5]}")
    return X[[pos[g] for g in cohort_ids], :].tocsr()


def normalize_sample_id(line: str) -> str:
    name = Path(line.strip()).name
    for suffix in (".fna.gz", ".fasta.gz", ".fa.gz", ".fna", ".fasta", ".fa"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_unitig_sample_order(path: Path) -> List[str]:
    ids = [normalize_sample_id(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(ids) != EXPECTED_N or len(set(ids)) != EXPECTED_N:
        raise RuntimeError(
            f"Unexpected unitig sample order: rows={len(ids)}, unique={len(set(ids))}"
        )
    return ids


def decode_unitigs(
    bin_path: Path,
    feature_indices: np.ndarray,
    unitig_sample_ids: List[str],
    cohort_ids: List[str],
) -> np.ndarray:
    idx = np.asarray(feature_indices, dtype=np.int64)
    idx = np.unique(idx)
    if len(idx) == 0:
        return np.zeros((len(cohort_ids), 0), dtype=np.uint8)

    bytes_per_feature = (len(unitig_sample_ids) + 7) // 8
    file_size = bin_path.stat().st_size
    if file_size % bytes_per_feature != 0:
        raise RuntimeError(
            f"Unitig bitpack size {file_size} not divisible by {bytes_per_feature}"
        )
    n_features = file_size // bytes_per_feature
    if n_features != EXPECTED_UNITIG_FEATURES:
        raise RuntimeError(
            f"Unitig feature count from bitpack {n_features} != {EXPECTED_UNITIG_FEATURES}"
        )
    if idx.min() < 0 or idx.max() >= n_features:
        raise RuntimeError("Unitig feature index out of range.")

    mm = np.memmap(bin_path, mode="r", dtype=np.uint8)
    byte_rows = np.asarray(mm.reshape(n_features, bytes_per_feature)[idx])
    bits = np.unpackbits(byte_rows, axis=1, bitorder="little")[:, :len(unitig_sample_ids)].T

    unitig_pos = {g: i for i, g in enumerate(unitig_sample_ids)}
    missing = [g for g in cohort_ids if g not in unitig_pos]
    if missing:
        raise RuntimeError(f"Unitig sample order missing cohort IDs, first={missing[:5]}")
    reorder = np.array([unitig_pos[g] for g in cohort_ids], dtype=np.int64)
    return np.ascontiguousarray(bits[reorder, :], dtype=np.uint8)


def read_unitig_index_selected(path: Path, wanted: Iterable[int]) -> pd.DataFrame:
    wanted = set(int(x) for x in wanted)
    if not wanted:
        return pd.DataFrame(
            columns=["Feature Index", "Raw Record", "Sequence", "Prevalence", "Minor State Count", "Byte Offset"]
        )
    max_idx = max(wanted)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            i = int(row["Feature Index"])
            if i in wanted:
                rows.append(row)
                if len(rows) == len(wanted):
                    break
            if i > max_idx and len(rows) == len(wanted):
                break
    df = pd.DataFrame(rows)
    if len(df) != len(wanted):
        got = set(pd.to_numeric(df["Feature Index"], errors="coerce").dropna().astype(int)) if len(df) else set()
        miss = sorted(wanted - got)
        raise RuntimeError(f"Unitig index missing selected features, first={miss[:10]}")
    for c in ["Feature Index", "Raw Record", "Prevalence", "Minor State Count", "Byte Offset"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df.sort_values("Feature Index").reset_index(drop=True)


def load_feature_selection(path: Path) -> pd.DataFrame:
    fs = pd.read_csv(path)
    required = {"scheme", "fold", "representation", "rank", "feature_index", "chi2_score_train_only"}
    missing = required - set(fs.columns)
    if missing:
        raise RuntimeError(f"Feature-selection manifest missing: {sorted(missing)}")
    fs = fs[fs["representation"].isin(BROAD_REPS)].copy()
    fs["fold"] = pd.to_numeric(fs["fold"], errors="raise").astype(int)
    fs["rank"] = pd.to_numeric(fs["rank"], errors="raise").astype(int)
    fs["feature_index"] = pd.to_numeric(fs["feature_index"], errors="raise").astype(int)
    fs["chi2_score_train_only"] = pd.to_numeric(fs["chi2_score_train_only"], errors="coerce")
    if len(fs) != 60000:
        raise RuntimeError(f"Expected 60,000 File08 selected rows; found {len(fs)}")
    return fs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def binary_table(x: np.ndarray, y: np.ndarray) -> Tuple[int, int, int, int]:
    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    a = int(np.sum((x == 1) & (y == 1)))  # R, present
    b = int(np.sum((x == 1) & (y == 0)))  # S, present
    c = int(np.sum((x == 0) & (y == 1)))  # R, absent
    d = int(np.sum((x == 0) & (y == 0)))  # S, absent
    return a, b, c, d


def log_or_se(a: int, b: int, c: int, d: int) -> Tuple[float, float, bool]:
    corrected = min(a, b, c, d) == 0
    aa, bb, cc, dd = map(float, (a, b, c, d))
    if corrected:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5
    lor = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return lor, se, corrected


def meta_analyze(log_ors: np.ndarray, ses: np.ndarray) -> dict:
    log_ors = np.asarray(log_ors, dtype=float)
    ses = np.asarray(ses, dtype=float)
    ok = np.isfinite(log_ors) & np.isfinite(ses) & (ses > 0)
    log_ors, ses = log_ors[ok], ses[ok]
    k = len(log_ors)
    if k == 0:
        return {
            "n_folds": 0, "fixed_log_or": np.nan, "fixed_or": np.nan,
            "fixed_ci_low": np.nan, "fixed_ci_high": np.nan, "fixed_p": np.nan,
            "random_log_or": np.nan, "random_or": np.nan,
            "random_ci_low": np.nan, "random_ci_high": np.nan, "random_p": np.nan,
            "Q": np.nan, "I2": np.nan, "tau2": np.nan,
            "direction_concordance": np.nan,
        }

    w = 1 / (ses ** 2)
    mu = float(np.sum(w * log_ors) / np.sum(w))
    se_mu = float(math.sqrt(1 / np.sum(w)))
    z = mu / se_mu
    p = 2 * norm.sf(abs(z))
    ci = (mu - 1.96 * se_mu, mu + 1.96 * se_mu)

    Q = float(np.sum(w * (log_ors - mu) ** 2))
    df = max(k - 1, 0)
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w)) if k > 1 else np.nan
    tau2 = max(0.0, (Q - df) / C) if k > 1 and C > 0 else 0.0
    wr = 1 / (ses ** 2 + tau2)
    mur = float(np.sum(wr * log_ors) / np.sum(wr))
    se_r = float(math.sqrt(1 / np.sum(wr)))
    pr = 2 * norm.sf(abs(mur / se_r))
    cir = (mur - 1.96 * se_r, mur + 1.96 * se_r)

    I2 = max(0.0, (Q - df) / Q) if k > 1 and Q > 0 else 0.0
    direction = np.sign(mur) if mur != 0 else 0
    concord = float(np.mean(np.sign(log_ors) == direction)) if direction != 0 else np.nan

    return {
        "n_folds": k,
        "fixed_log_or": mu,
        "fixed_or": math.exp(mu),
        "fixed_ci_low": math.exp(ci[0]),
        "fixed_ci_high": math.exp(ci[1]),
        "fixed_p": p,
        "random_log_or": mur,
        "random_or": math.exp(mur),
        "random_ci_low": math.exp(cir[0]),
        "random_ci_high": math.exp(cir[1]),
        "random_p": pr,
        "Q": Q,
        "I2": I2,
        "tau2": tau2,
        "direction_concordance": concord,
    }


def bh_fdr(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], alpha=0.05, method="fdr_bh")[1]
    return out


def cmh_binary(x: np.ndarray, y: np.ndarray, strata: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    strata = np.asarray(strata, dtype=object)

    num = 0.0
    den = 0.0
    score = 0.0
    varsum = 0.0
    informative = 0
    n_informative = 0
    stratum_logors = []
    stratum_ses = []

    # factorize preserves missing strings as values if converted first.
    codes, uniques = pd.factorize(pd.Series(strata).astype(str), sort=False)
    for code in range(len(uniques)):
        m = codes == code
        if m.sum() < 2:
            continue
        xx, yy = x[m], y[m]
        a, b, c, d = binary_table(xx, yy)
        n = a + b + c + d
        if n <= 1:
            continue

        num += a * d / n
        den += b * c / n

        r1 = a + b
        r0 = c + d
        c1 = a + c
        c0 = b + d
        if r1 > 0 and r0 > 0 and c1 > 0 and c0 > 0:
            expa = r1 * c1 / n
            var = (r1 * r0 * c1 * c0) / (n * n * (n - 1))
            if var > 0:
                score += a - expa
                varsum += var
                informative += 1
                n_informative += n
                lor, se, _ = log_or_se(a, b, c, d)
                stratum_logors.append(lor)
                stratum_ses.append(se)

    mh_or = num / den if den > 0 else np.nan
    stat = (score ** 2) / varsum if varsum > 0 else np.nan
    p = chi2.sf(stat, 1) if np.isfinite(stat) else np.nan

    if stratum_logors:
        dirs = np.sign(np.asarray(stratum_logors))
        target = np.sign(math.log(mh_or)) if np.isfinite(mh_or) and mh_or > 0 and mh_or != 1 else 0
        concord = float(np.mean(dirs == target)) if target != 0 else np.nan
    else:
        concord = np.nan

    return {
        "mh_or": mh_or,
        "cmh_chi2": stat,
        "cmh_p": p,
        "informative_strata": informative,
        "n_informative_samples": n_informative,
        "within_stratum_direction_concordance": concord,
    }


def phi_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson/phi correlation between binary columns of A and B."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    num = Ac.T @ Bc
    da = np.sqrt(np.sum(Ac * Ac, axis=0))
    db = np.sqrt(np.sum(Bc * Bc, axis=0))
    den = da[:, None] * db[None, :]
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out


def center_within_groups(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    out = X.copy()
    codes, uniques = pd.factorize(pd.Series(groups).astype(str), sort=False)
    for code in range(len(uniques)):
        idx = np.flatnonzero(codes == code)
        if len(idx):
            out[idx] -= out[idx].mean(axis=0, keepdims=True)
    return out


def correlation_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    num = A.T @ B
    da = np.sqrt(np.sum(A * A, axis=0))
    db = np.sqrt(np.sum(B * B, axis=0))
    den = da[:, None] * db[None, :]
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]


# ---------------------------------------------------------------------------
# Block 10A
# ---------------------------------------------------------------------------

def run_block_10A(root: Path, out_dir: Path, checkpoint_root: Path) -> None:
    block = "10A_integrity"
    out_audit = out_dir / "file10_input_integrity_audit.csv"
    out_summary = out_dir / "file10_cohort_summary.json"
    outputs = [out_audit, out_summary]

    if block_done(checkpoint_root, block, outputs):
        log("10A SKIP_COMPLETE — integrity outputs already checkpointed.")
        return

    t0 = time.time()
    log("10A START — locked-input / row-order / cohort integrity")

    paths = {
        "folds": root / "data/splits/file07_outer_folds.csv",
        "known": root / "data/features/known_amr/X_known_amr_final.csv",
        "known_meta": root / "data/features/known_amr/known_amr_feature_metadata_final.csv",
        "pang": root / "data/features/pangenome/X_pangenome_plfam_binary.npz",
        "pang_rows": root / "data/features/pangenome/X_pangenome_row_index.csv",
        "pang_index": root / "data/features/pangenome/X_pangenome_feature_index.csv",
        "unitig_bin": root / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin",
        "unitig_index": root / "data/features/sequence_variation/X_sequence_variation_unitig_feature_index.tsv",
        "unitig_order": root / "checkpoints/sequence_variation/unitig_full_k31_sample_order.txt",
        "fs": root / "data/modeling/file08_feature_selection_manifest.csv",
        "oof": root / "data/modeling/file08_oof_predictions.csv",
        "file08_summary": root / "checkpoints/model_training/file08_final_summary.json",
        "file09_audit": root / "data/evaluation/file09_integrity_audit.csv",
        "file09b_audit": root / "data/evaluation/reviewer_defense/file09b_integrity_audit.csv",
        "file09c_audit": root / "data/evaluation/reviewer_defense_nonlinear_v102/file09c_integrity_audit.csv",
        "file09c_repro": root / "data/evaluation/reviewer_defense_nonlinear_v102/file09c_default_baseline_reproduction_audit.csv",
    }

    audit = []

    def add(check, passed, detail):
        audit.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": str(detail)})
        if not passed:
            raise RuntimeError(f"10A FAIL {check}: {detail}")

    for name, p in paths.items():
        add(f"path_{name}", p.is_file(), p)

    folds = load_folds(paths["folds"])
    cohort_ids = folds["Genome ID"].astype(str).tolist()
    y = folds["y"].to_numpy(dtype=np.uint8)

    Xk, known_names = load_known(paths["known"], cohort_ids)
    Xp = load_pangenome(paths["pang"], paths["pang_rows"], cohort_ids)
    unitig_ids = load_unitig_sample_order(paths["unitig_order"])
    fs = load_feature_selection(paths["fs"])

    add("cohort_n", len(folds) == EXPECTED_N, len(folds))
    add("phenotype_binary", set(np.unique(y)) == {0, 1}, np.unique(y).tolist())
    add("known_shape", Xk.shape == (EXPECTED_N, EXPECTED_KNOWN_FEATURES), Xk.shape)
    add("pangenome_shape", Xp.shape == (EXPECTED_N, EXPECTED_PANGENOME_FEATURES), Xp.shape)
    add("unitig_sample_membership", set(unitig_ids) == set(cohort_ids), f"{len(unitig_ids)} IDs")
    add("feature_selection_rows", len(fs) == 60000, len(fs))

    # Unitig orientation / prevalence audit on three fixed selected features.
    test_idx = fs[fs["representation"] == "unitig"]["feature_index"].drop_duplicates().head(3).to_numpy()
    Xu = decode_unitigs(paths["unitig_bin"], test_idx, unitig_ids, cohort_ids)
    meta = read_unitig_index_selected(paths["unitig_index"], test_idx)
    prev_map = dict(zip(meta["Feature Index"].astype(int), meta["Prevalence"].astype(int)))
    decoded_prev = {int(fi): int(Xu[:, j].sum()) for j, fi in enumerate(sorted(np.unique(test_idx)))}
    add("unitig_orientation_prevalence", decoded_prev == {int(k): int(v) for k, v in prev_map.items()},
        f"decoded={decoded_prev}; index={prev_map}")

    summary08 = read_json(paths["file08_summary"])
    add(
        "file08_train_only_selection",
        summary08.get("supervised_feature_selection_scope") == "outer_training_fold_only",
        summary08.get("supervised_feature_selection_scope"),
    )
    add(
        "file08_outer_test_not_used",
        summary08.get("outer_test_labels_used_for_selection_or_training") is False,
        summary08.get("outer_test_labels_used_for_selection_or_training"),
    )

    if paths["file09c_repro"].is_file():
        repro = pd.read_csv(paths["file09c_repro"])
        add(
            "file09c_default_reproduction",
            len(repro) == 9 and (repro["status"] == "PASS").all(),
            f"{int((repro['status']=='PASS').sum())}/{len(repro)} PASS",
        )

    # Ensure prohibited metadata fields are not in predictive matrices.
    prohibited = {"Genome ID", "Phenotype", "y", "MLST", "Genomic Cluster"}
    add(
        "known_predictor_names_no_prohibited",
        not bool(prohibited.intersection(known_names)),
        sorted(prohibited.intersection(known_names)),
    )

    atomic_write_csv(out_audit, pd.DataFrame(audit))

    summary = {
        "script_version": SCRIPT_VERSION,
        "design_id": DESIGN_ID,
        "n": len(folds),
        "n_resistant": int(y.sum()),
        "n_susceptible": int((1-y).sum()),
        "known_features": len(known_names),
        "pangenome_features": int(Xp.shape[1]),
        "unitig_features": EXPECTED_UNITIG_FEATURES,
        "fold_sizes": {
            scheme: folds.groupby(col).size().astype(int).to_dict()
            for scheme, col in SCHEME_TO_COL.items()
        },
        "input_fingerprints": {k: file_fingerprint(v) for k, v in paths.items() if v.is_file()},
        "causal_claim_guardrail": "File10 reports associations/attributions, not causality.",
    }
    atomic_write_text(out_summary, json.dumps(summary, indent=2) + "\n")

    mark_block(checkpoint_root, block, outputs, time.time()-t0, {"n_checks": len(audit)})
    log(f"10A PASS — {len(audit)} integrity checks | {human_seconds(time.time()-t0)}")


# ---------------------------------------------------------------------------
# Block 10B — feature-selection stability + candidate/cached matrices
# ---------------------------------------------------------------------------

def run_block_10B(
    root: Path,
    out_dir: Path,
    checkpoint_root: Path,
    top_n: int,
    shap_top: int,
    shap_tasks: List[Tuple[str, int]],
) -> None:
    block = "10B_selection_stability"
    out_stability = out_dir / "file10_feature_selection_stability.csv"
    out_jaccard = out_dir / "file10_feature_selection_jaccard.csv"
    out_candidates = out_dir / "file10_candidate_features.csv"
    out_pang_cache = checkpoint_root / "cache/file10_pangenome_selected_cache.npz"
    out_unitig_cache = checkpoint_root / "cache/file10_unitig_selected_cache.npz"
    out_unitig_meta = checkpoint_root / "cache/file10_unitig_selected_metadata.csv"
    outputs = [
        out_stability, out_jaccard, out_candidates,
        out_pang_cache, out_unitig_cache, out_unitig_meta,
    ]

    if block_done(checkpoint_root, block, outputs):
        log("10B SKIP_COMPLETE — selection stability/caches already checkpointed.")
        return

    t0 = time.time()
    log("10B START — train-only selection stability + selected-feature caches")

    folds = load_folds(root / "data/splits/file07_outer_folds.csv")
    cohort_ids = folds["Genome ID"].astype(str).tolist()
    fs = load_feature_selection(root / "data/modeling/file08_feature_selection_manifest.csv")

    rows = []
    for rep, g in fs.groupby("representation"):
        for fi, h in g.groupby("feature_index"):
            rows.append({
                "representation": rep,
                "feature_index": int(fi),
                "selected_tasks": int(len(h)),
                "selected_schemes": int(h["scheme"].nunique()),
                "random_stratified_folds": int((h["scheme"] == "random_stratified").sum()),
                "genomic_cluster_aware_folds": int((h["scheme"] == "genomic_cluster_aware").sum()),
                "mlst_aware_folds": int((h["scheme"] == "mlst_aware").sum()),
                "median_rank": float(h["rank"].median()),
                "mean_rank": float(h["rank"].mean()),
                "best_rank": int(h["rank"].min()),
                "mean_train_chi2": float(h["chi2_score_train_only"].mean()),
            })
    stab = pd.DataFrame(rows)
    stab = stab.sort_values(
        ["representation", "selected_tasks", "median_rank", "best_rank"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    stab["stability_rank_within_representation"] = (
        stab.groupby("representation").cumcount() + 1
    )
    atomic_write_csv(out_stability, stab)

    # Pairwise overlap of File08 top-2000 training-only selections.
    jac_rows = []
    tasks = (
        fs[["scheme", "fold", "representation"]]
        .drop_duplicates()
        .sort_values(["representation", "scheme", "fold"])
    )
    for rep in BROAD_REPS:
        tr = tasks[tasks["representation"] == rep]
        sets = {}
        for r in tr.itertuples(index=False):
            key = (r.scheme, int(r.fold))
            sets[key] = set(
                fs[
                    (fs["representation"] == rep)
                    & (fs["scheme"] == r.scheme)
                    & (fs["fold"] == int(r.fold))
                ]["feature_index"].astype(int)
            )
        keys = list(sets)
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a, b = keys[i], keys[j]
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                jac_rows.append({
                    "representation": rep,
                    "scheme_a": a[0], "fold_a": a[1],
                    "scheme_b": b[0], "fold_b": b[1],
                    "intersection": inter,
                    "union": union,
                    "jaccard": inter / union if union else np.nan,
                })
    atomic_write_csv(out_jaccard, pd.DataFrame(jac_rows))

    # Top stable candidates per representation.
    cand = (
        stab.sort_values(
            ["representation", "selected_tasks", "median_rank", "best_rank"],
            ascending=[True, False, True, True],
        )
        .groupby("representation", as_index=False, group_keys=False)
        .head(top_n)
        .copy()
    )

    # Add pangenome IDs.
    pang_idx = pd.read_csv(root / "data/features/pangenome/X_pangenome_feature_index.csv")
    pang_idx["Feature Index"] = pd.to_numeric(pang_idx["Feature Index"], errors="raise").astype(int)
    pang_map = pang_idx.set_index("Feature Index")["PLFam ID"].astype(str).to_dict()
    cand["feature_id"] = cand.apply(
        lambda r: pang_map.get(int(r["feature_index"]), "")
        if r["representation"] == "pangenome"
        else f"unitig_{int(r['feature_index'])}",
        axis=1,
    )

    # Build union required by candidates + SHAP fold-specific train-only top features.
    needed = {"pangenome": set(), "unitig": set()}
    for rep in BROAD_REPS:
        needed[rep].update(
            cand.loc[cand["representation"] == rep, "feature_index"].astype(int).tolist()
        )
    for scheme, fold in shap_tasks:
        for rep in BROAD_REPS:
            h = fs[
                (fs["scheme"] == scheme)
                & (fs["fold"] == fold)
                & (fs["representation"] == rep)
                & (fs["rank"] <= shap_top)
            ]
            needed[rep].update(h["feature_index"].astype(int).tolist())

    # Pangenome cache.
    Xp = load_pangenome(
        root / "data/features/pangenome/X_pangenome_plfam_binary.npz",
        root / "data/features/pangenome/X_pangenome_row_index.csv",
        cohort_ids,
    )
    pidx = np.array(sorted(needed["pangenome"]), dtype=np.int64)
    Xpc = np.ascontiguousarray(Xp[:, pidx].toarray(), dtype=np.uint8)
    atomic_write_npz(out_pang_cache, feature_index=pidx, X=Xpc)

    # Unitig cache + selected metadata.
    unitig_order = load_unitig_sample_order(
        root / "checkpoints/sequence_variation/unitig_full_k31_sample_order.txt"
    )
    uidx = np.array(sorted(needed["unitig"]), dtype=np.int64)
    Xuc = decode_unitigs(
        root / "data/features/sequence_variation/X_sequence_variation_unitig_bitpacked.bin",
        uidx, unitig_order, cohort_ids,
    )
    atomic_write_npz(out_unitig_cache, feature_index=uidx, X=Xuc)

    log(
        f"10B unitig metadata scan — extracting {len(uidx)} selected rows "
        "from 4.7M-feature index; this is sequential and checkpointed."
    )
    umeta = read_unitig_index_selected(
        root / "data/features/sequence_variation/X_sequence_variation_unitig_feature_index.tsv",
        uidx,
    )
    atomic_write_csv(out_unitig_meta, umeta)

    seq_map = umeta.set_index("Feature Index")["Sequence"].astype(str).to_dict()
    cand["sequence"] = cand.apply(
        lambda r: seq_map.get(int(r["feature_index"]), "")
        if r["representation"] == "unitig" else "",
        axis=1,
    )
    atomic_write_csv(out_candidates, cand)

    mark_block(
        checkpoint_root, block, outputs, time.time()-t0,
        {
            "top_n_per_representation": top_n,
            "pangenome_cached_features": len(pidx),
            "unitig_cached_features": len(uidx),
        },
    )
    log(
        f"10B PASS — candidates={len(cand)}, cache pang={len(pidx)}, "
        f"unitig={len(uidx)} | {human_seconds(time.time()-t0)}"
    )


def load_selected_cache(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return z["feature_index"].astype(int), z["X"].astype(np.uint8)


def subset_cache(indices: np.ndarray, X: np.ndarray, wanted: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    pos = {int(fi): j for j, fi in enumerate(indices)}
    wanted = [int(x) for x in wanted]
    missing = [x for x in wanted if x not in pos]
    if missing:
        raise RuntimeError(f"Selected cache missing features, first={missing[:10]}")
    cols = np.array([pos[x] for x in wanted], dtype=int)
    return np.array(wanted, dtype=int), np.ascontiguousarray(X[:, cols], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Block 10C — cross-fitted held-out effects
# ---------------------------------------------------------------------------

def run_block_10C(root: Path, out_dir: Path, checkpoint_root: Path, smoke: bool) -> None:
    block = "10C_crossfitted_effects"
    out_fold = out_dir / "file10_crossfitted_fold_effects.csv"
    out_meta = out_dir / "file10_crossfitted_meta_effects.csv"
    outputs = [out_fold, out_meta]

    if block_done(checkpoint_root, block, outputs):
        log("10C SKIP_COMPLETE — cross-fitted held-out effects already checkpointed.")
        return

    t0 = time.time()
    log("10C START — held-out effect estimates after train-only selection")

    folds = load_folds(root / "data/splits/file07_outer_folds.csv")
    y = folds["y"].to_numpy(dtype=np.uint8)
    fs = load_feature_selection(root / "data/modeling/file08_feature_selection_manifest.csv")
    cand = pd.read_csv(out_dir / "file10_candidate_features.csv")
    pidx, Xp = load_selected_cache(checkpoint_root / "cache/file10_pangenome_selected_cache.npz")
    uidx, Xu = load_selected_cache(checkpoint_root / "cache/file10_unitig_selected_cache.npz")

    tasks = [
        (scheme, fold, rep)
        for scheme in SCHEME_TO_COL
        for fold in range(5)
        for rep in BROAD_REPS
    ]
    if smoke:
        tasks = [(s, f, r) for s, f, r in tasks if s == "mlst_aware"]

    task_dir = checkpoint_root / "tasks/10C"
    task_dir.mkdir(parents=True, exist_ok=True)
    prog = Progress(len(tasks), "10C cross-fit tasks")
    task_outputs = []

    for scheme, fold, rep in tasks:
        task_id = f"{scheme}__fold{fold}__{rep}"
        out_task = task_dir / f"{task_id}.csv"
        marker = task_dir / f"{task_id}.json"
        task_outputs.append(out_task)

        if task_marker_valid(marker, task_id, [out_task]):
            prog.update(f"{task_id} SKIP")
            continue

        tt = time.time()
        test = folds[SCHEME_TO_COL[scheme]].to_numpy(dtype=int) == fold
        selected = set(
            fs[
                (fs["scheme"] == scheme)
                & (fs["fold"] == fold)
                & (fs["representation"] == rep)
            ]["feature_index"].astype(int)
        )
        wanted = (
            cand[
                (cand["representation"] == rep)
                & (cand["feature_index"].astype(int).isin(selected))
            ]["feature_index"].astype(int).tolist()
        )

        if rep == "pangenome":
            fidx, X = subset_cache(pidx, Xp, wanted)
        else:
            fidx, X = subset_cache(uidx, Xu, wanted)

        rows = []
        for j, fi in enumerate(fidx):
            a, b, c, d = binary_table(X[test, j], y[test])
            lor, se, corrected = log_or_se(a, b, c, d)
            rows.append({
                "scheme": scheme,
                "fold": fold,
                "representation": rep,
                "feature_index": int(fi),
                "a_R_present": a, "b_S_present": b,
                "c_R_absent": c, "d_S_absent": d,
                "log_or": lor, "se_log_or": se,
                "or": math.exp(lor),
                "zero_cell_correction": corrected,
                "n_test": int(test.sum()),
            })

        df = pd.DataFrame(rows)
        atomic_write_csv(out_task, df)
        mark_task(marker, task_id, [out_task], time.time()-tt, {"n_features": len(df)})
        prog.update(f"{task_id} | features={len(df)}")

    fold_effects = pd.concat(
        [pd.read_csv(p) for p in task_outputs if p.is_file()],
        ignore_index=True,
    )
    atomic_write_csv(out_fold, fold_effects)

    meta_rows = []
    for key, g in fold_effects.groupby(["scheme", "representation", "feature_index"], sort=True):
        scheme, rep, fi = key
        m = meta_analyze(g["log_or"].to_numpy(), g["se_log_or"].to_numpy())
        meta_rows.append({
            "scheme": scheme,
            "representation": rep,
            "feature_index": int(fi),
            **m,
        })
    meta = pd.DataFrame(meta_rows)
    if len(meta):
        meta["random_p_fdr"] = (
            meta.groupby(["scheme", "representation"], group_keys=False)["random_p"]
            .transform(lambda s: bh_fdr(s))
        )
    atomic_write_csv(out_meta, meta)

    mark_block(
        checkpoint_root, block, outputs, time.time()-t0,
        {"task_count": len(tasks), "fold_effect_rows": len(fold_effects), "meta_rows": len(meta)},
    )
    log(f"10C PASS — fold effects={len(fold_effects):,}, meta rows={len(meta):,} | {human_seconds(time.time()-t0)}")


# ---------------------------------------------------------------------------
# Block 10D — population structure, within-lineage, dominant ST sensitivity
# ---------------------------------------------------------------------------

def run_block_10D(root: Path, out_dir: Path, checkpoint_root: Path, top_lineage_features: int) -> None:
    block = "10D_population_structure"
    out_cmh = out_dir / "file10_population_adjusted_association.csv"
    out_within = out_dir / "file10_within_mlst_effects.csv"
    out_excl = out_dir / "file10_dominant_lineage_exclusion.csv"
    outputs = [out_cmh, out_within, out_excl]

    if block_done(checkpoint_root, block, outputs):
        log("10D SKIP_COMPLETE — population-structure analyses already checkpointed.")
        return

    t0 = time.time()
    log("10D START — CMH population adjustment + within-lineage/exclusion sensitivity")

    folds = load_folds(root / "data/splits/file07_outer_folds.csv")
    y = folds["y"].to_numpy(dtype=np.uint8)
    cand = pd.read_csv(out_dir / "file10_candidate_features.csv")
    pidx, Xp = load_selected_cache(checkpoint_root / "cache/file10_pangenome_selected_cache.npz")
    uidx, Xu = load_selected_cache(checkpoint_root / "cache/file10_unitig_selected_cache.npz")

    cmh_rows = []
    tasks = [(rep, strat) for rep in BROAD_REPS for strat in ("genomic_cluster", "mlst")]
    prog = Progress(len(tasks), "10D CMH tasks")

    for rep, strat in tasks:
        wanted = cand[cand["representation"] == rep]["feature_index"].astype(int).tolist()
        fidx, X = subset_cache(pidx, Xp, wanted) if rep == "pangenome" else subset_cache(uidx, Xu, wanted)
        groups = (
            folds["Genomic Cluster"].astype(str).to_numpy()
            if strat == "genomic_cluster"
            else folds["MLST"].astype(str).to_numpy()
        )
        for j, fi in enumerate(fidx):
            a, b, c, d = binary_table(X[:, j], y)
            lor, se, corrected = log_or_se(a, b, c, d)
            cmh = cmh_binary(X[:, j], y, groups)
            cmh_rows.append({
                "representation": rep,
                "feature_index": int(fi),
                "stratifier": strat,
                "unadjusted_or": math.exp(lor),
                "unadjusted_log_or": lor,
                "unadjusted_se": se,
                "unadjusted_zero_cell_correction": corrected,
                **cmh,
            })
        prog.update(f"{rep}/{strat} | features={len(fidx)}")

    cmh_df = pd.DataFrame(cmh_rows)
    cmh_df["cmh_p_fdr"] = (
        cmh_df.groupby(["representation", "stratifier"], group_keys=False)["cmh_p"]
        .transform(lambda s: bh_fdr(s))
    )
    atomic_write_csv(out_cmh, cmh_df)

    # Exact numeric MLST groups for within-lineage effects.
    mlst = folds["MLST"].astype(str)
    exact = mlst.str.fullmatch(r"ST\d+", case=False)
    exact_counts = mlst[exact].value_counts()
    if int(exact.sum()) < 4000:
        raise RuntimeError(
            f"Unexpected exact-MLST parsing: recognized {int(exact.sum())} / {len(mlst)} "
            "rows as ST<number>. Expected approximately 4,130 exact calls."
        )
    major_sts = exact_counts.head(5).index.tolist()

    top_cand = (
        cand.sort_values(["representation", "stability_rank_within_representation"])
        .groupby("representation", as_index=False, group_keys=False)
        .head(top_lineage_features)
    )

    within_rows = []
    exclusion_rows = []

    for rep in BROAD_REPS:
        wanted = top_cand[top_cand["representation"] == rep]["feature_index"].astype(int).tolist()
        fidx, X = subset_cache(pidx, Xp, wanted) if rep == "pangenome" else subset_cache(uidx, Xu, wanted)

        for j, fi in enumerate(fidx):
            # within exact STs n>=20; keep informative 2x2 only.
            for st, nst in exact_counts.items():
                if nst < 20:
                    continue
                m = (mlst == st).to_numpy()
                xx, yy = X[m, j], y[m]
                a, b, c, d = binary_table(xx, yy)
                if len(np.unique(xx)) < 2 or len(np.unique(yy)) < 2:
                    continue
                lor, se, corrected = log_or_se(a, b, c, d)
                within_rows.append({
                    "representation": rep,
                    "feature_index": int(fi),
                    "MLST": st,
                    "n": int(m.sum()),
                    "a_R_present": a, "b_S_present": b,
                    "c_R_absent": c, "d_S_absent": d,
                    "log_or": lor, "or": math.exp(lor),
                    "se_log_or": se,
                    "zero_cell_correction": corrected,
                })

            scenarios = [("none", [])]
            if major_sts:
                scenarios.append((f"exclude_{major_sts[0]}", major_sts[:1]))
            if len(major_sts) >= 2:
                scenarios.append((f"exclude_top2_{'_'.join(major_sts[:2])}", major_sts[:2]))
            if len(major_sts) >= 5:
                scenarios.append((f"exclude_top5_{'_'.join(major_sts[:5])}", major_sts[:5]))

            for label, excluded in scenarios:
                m = ~mlst.isin(excluded).to_numpy()
                a, b, c, d = binary_table(X[m, j], y[m])
                lor, se, corrected = log_or_se(a, b, c, d)
                exclusion_rows.append({
                    "representation": rep,
                    "feature_index": int(fi),
                    "scenario": label,
                    "excluded_STs": ";".join(excluded),
                    "n": int(m.sum()),
                    "or": math.exp(lor),
                    "log_or": lor,
                    "se_log_or": se,
                    "zero_cell_correction": corrected,
                })

    atomic_write_csv(out_within, pd.DataFrame(within_rows))
    atomic_write_csv(out_excl, pd.DataFrame(exclusion_rows))

    mark_block(
        checkpoint_root, block, outputs, time.time()-t0,
        {"major_STs": major_sts, "top_lineage_features_per_rep": top_lineage_features},
    )
    log(f"10D PASS — CMH rows={len(cmh_df)}, within-ST rows={len(within_rows)} | {human_seconds(time.time()-t0)}")


# ---------------------------------------------------------------------------
# Block 10E — redundancy + known-AMR linkage
# ---------------------------------------------------------------------------

def run_block_10E(root: Path, out_dir: Path, checkpoint_root: Path, redundancy_threshold: float = 0.95) -> None:
    block = "10E_redundancy_known_amr_linkage"
    out_red = out_dir / "file10_broader_feature_redundancy.csv"
    out_link = out_dir / "file10_known_amr_linkage.csv"
    out_seq = out_dir / "file10_unitig_sequence_containment.csv"
    outputs = [out_red, out_link, out_seq]

    if block_done(checkpoint_root, block, outputs):
        log("10E SKIP_COMPLETE — redundancy/linkage outputs already checkpointed.")
        return

    t0 = time.time()
    log("10E START — redundancy + known-AMR co-occurrence/linkage")

    folds = load_folds(root / "data/splits/file07_outer_folds.csv")
    cohort_ids = folds["Genome ID"].astype(str).tolist()
    Xk, known_names = load_known(root / "data/features/known_amr/X_known_amr_final.csv", cohort_ids)
    cand = pd.read_csv(out_dir / "file10_candidate_features.csv")
    pidx, Xp = load_selected_cache(checkpoint_root / "cache/file10_pangenome_selected_cache.npz")
    uidx, Xu = load_selected_cache(checkpoint_root / "cache/file10_unitig_selected_cache.npz")

    # Combined candidate matrix.
    matrices = []
    labels = []
    for rep in BROAD_REPS:
        wanted = cand[cand["representation"] == rep]["feature_index"].astype(int).tolist()
        fidx, X = subset_cache(pidx, Xp, wanted) if rep == "pangenome" else subset_cache(uidx, Xu, wanted)
        matrices.append(X)
        labels.extend([(rep, int(fi)) for fi in fidx])
    Xb = np.concatenate(matrices, axis=1) if matrices else np.zeros((EXPECTED_N, 0), dtype=np.uint8)

    # Redundancy clusters based on absolute phi >= threshold.
    corr = phi_matrix(Xb, Xb)
    uf = UnionFind(len(labels))
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if abs(float(corr[i, j])) >= redundancy_threshold:
                uf.union(i, j)

    cluster_members = defaultdict(list)
    for i in range(len(labels)):
        cluster_members[uf.find(i)].append(i)
    cluster_id_map = {}
    for cid, members in enumerate(sorted(cluster_members.values(), key=lambda z: (-len(z), z[0])), 1):
        for i in members:
            cluster_id_map[i] = cid

    red_rows = []
    for i, (rep, fi) in enumerate(labels):
        cid = cluster_id_map[i]
        members = cluster_members[uf.find(i)]
        max_other = 0.0
        if len(members) > 1:
            max_other = max(abs(float(corr[i, j])) for j in members if j != i)
        red_rows.append({
            "representation": rep,
            "feature_index": fi,
            "redundancy_cluster": cid,
            "cluster_size": len(members),
            "max_abs_phi_within_cluster": max_other,
            "threshold": redundancy_threshold,
        })
    red_df = pd.DataFrame(red_rows)
    atomic_write_csv(out_red, red_df)

    # Known-AMR linkage: raw phi and within-population centered correlations.
    raw = phi_matrix(Xb, Xk)
    Xk_cluster = center_within_groups(Xk, folds["Genomic Cluster"].astype(str).to_numpy())
    Xb_cluster = center_within_groups(Xb, folds["Genomic Cluster"].astype(str).to_numpy())
    adj_cluster = correlation_matrix(Xb_cluster, Xk_cluster)

    Xk_mlst = center_within_groups(Xk, folds["MLST"].astype(str).to_numpy())
    Xb_mlst = center_within_groups(Xb, folds["MLST"].astype(str).to_numpy())
    adj_mlst = correlation_matrix(Xb_mlst, Xk_mlst)

    link_rows = []
    for i, (rep, fi) in enumerate(labels):
        order = np.argsort(-np.abs(raw[i]))[:3]
        best_raw = int(order[0])
        best_gc = int(np.argmax(np.abs(adj_cluster[i])))
        best_mlst = int(np.argmax(np.abs(adj_mlst[i])))

        row = {
            "representation": rep,
            "feature_index": fi,
            "max_raw_abs_phi": float(abs(raw[i, best_raw])),
            "max_raw_phi": float(raw[i, best_raw]),
            "max_raw_known_amr_feature": known_names[best_raw],
            "max_genomic_cluster_centered_abs_r": float(abs(adj_cluster[i, best_gc])),
            "max_genomic_cluster_centered_r": float(adj_cluster[i, best_gc]),
            "max_genomic_cluster_centered_known_amr_feature": known_names[best_gc],
            "max_mlst_centered_abs_r": float(abs(adj_mlst[i, best_mlst])),
            "max_mlst_centered_r": float(adj_mlst[i, best_mlst]),
            "max_mlst_centered_known_amr_feature": known_names[best_mlst],
        }
        for rank, j in enumerate(order, 1):
            row[f"raw_top{rank}_known_amr"] = known_names[int(j)]
            row[f"raw_top{rank}_phi"] = float(raw[i, int(j)])
        link_rows.append(row)
    atomic_write_csv(out_link, pd.DataFrame(link_rows))

    # Unitig sequence containment / reverse-complement containment.
    umeta = pd.read_csv(checkpoint_root / "cache/file10_unitig_selected_metadata.csv")
    unitig_candidates = cand[cand["representation"] == "unitig"][["feature_index", "sequence"]].copy()
    seqs = [(int(r.feature_index), str(r.sequence).upper()) for r in unitig_candidates.itertuples(index=False)]
    seq_rows = []
    for i in range(len(seqs)):
        fi, si = seqs[i]
        rci = reverse_complement(si)
        for j in range(i+1, len(seqs)):
            fj, sj = seqs[j]
            rcj = reverse_complement(sj)
            relationship = None
            if si == sj:
                relationship = "identical"
            elif si == rcj:
                relationship = "reverse_complement_identical"
            elif si in sj or sj in si:
                relationship = "direct_containment"
            elif rci in sj or rcj in si:
                relationship = "reverse_complement_containment"
            if relationship:
                seq_rows.append({
                    "feature_index_a": fi,
                    "feature_index_b": fj,
                    "relationship": relationship,
                    "len_a": len(si),
                    "len_b": len(sj),
                })
    atomic_write_csv(out_seq, pd.DataFrame(seq_rows))

    mark_block(
        checkpoint_root, block, outputs, time.time()-t0,
        {"candidate_features": len(labels), "redundancy_threshold": redundancy_threshold},
    )
    log(f"10E PASS — broader candidates={len(labels)}, linkage to {len(known_names)} known-AMR features | {human_seconds(time.time()-t0)}")


# ---------------------------------------------------------------------------
# Block 10F — cross-fitted SHAP on lineage-aware splits
# ---------------------------------------------------------------------------

def stratified_subsample(indices: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= max_n:
        return indices
    rng = np.random.default_rng(seed)
    yy = y[indices]
    out = []
    for cls in (0, 1):
        cls_idx = indices[yy == cls]
        n_take = max(1, int(round(max_n * len(cls_idx) / len(indices))))
        n_take = min(n_take, len(cls_idx))
        out.extend(rng.choice(cls_idx, size=n_take, replace=False).tolist())
    out = np.array(sorted(set(out)), dtype=int)
    if len(out) > max_n:
        out = np.sort(rng.choice(out, size=max_n, replace=False))
    elif len(out) < max_n:
        remaining = np.setdiff1d(indices, out, assume_unique=False)
        add_n = min(max_n - len(out), len(remaining))
        if add_n:
            out = np.sort(np.concatenate([out, rng.choice(remaining, size=add_n, replace=False)]))
    return out


def normalize_shap_array(values, n_samples: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        arr = np.asarray(values[1] if len(values) > 1 else values[0])
    else:
        arr = np.asarray(values)

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        # Current SHAP sklearn-tree convention is often (samples, features, outputs).
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            arr = arr[:, :, 1] if arr.shape[2] >= 2 else arr[:, :, 0]
        elif arr.shape[1] == n_samples and arr.shape[2] == n_features:
            arr = arr[1] if arr.shape[0] >= 2 else arr[0]
        else:
            raise RuntimeError(f"Unrecognized SHAP shape {arr.shape}")
    else:
        raise RuntimeError(f"Unrecognized SHAP ndim/shape: {arr.ndim}/{arr.shape}")

    if arr.shape != (n_samples, n_features):
        raise RuntimeError(
            f"Normalized SHAP shape {arr.shape} != {(n_samples, n_features)}"
        )
    return np.asarray(arr, dtype=np.float64)


def run_block_10F(
    root: Path,
    out_dir: Path,
    checkpoint_root: Path,
    shap_tasks: List[Tuple[str, int]],
    shap_top: int,
    shap_max_samples: int,
    workers: int,
) -> None:
    block = "10F_crossfitted_shap"
    out_importance = out_dir / "file10_shap_feature_importance.csv"
    out_group = out_dir / "file10_shap_group_summary.csv"
    out_task_summary = out_dir / "file10_shap_task_summary.csv"
    outputs = [out_importance, out_group, out_task_summary]

    if block_done(checkpoint_root, block, outputs):
        log("10F SKIP_COMPLETE — SHAP outputs already checkpointed.")
        return

    t0 = time.time()
    log("10F START — cross-fitted SHAP on lineage-aware held-out folds")
    log(
        f"10F design — tasks={len(shap_tasks)}, top broader features/fold/rep={shap_top}, "
        f"max explained samples/fold={shap_max_samples}"
    )

    try:
        import shap
    except Exception as exc:
        raise RuntimeError(
            "SHAP import failed. Verify the amr-genome-ml environment: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    folds = load_folds(root / "data/splits/file07_outer_folds.csv")
    cohort_ids = folds["Genome ID"].astype(str).tolist()
    y = folds["y"].to_numpy(dtype=np.uint8)
    Xk, known_names = load_known(root / "data/features/known_amr/X_known_amr_final.csv", cohort_ids)
    fs = load_feature_selection(root / "data/modeling/file08_feature_selection_manifest.csv")

    pidx, Xp = load_selected_cache(checkpoint_root / "cache/file10_pangenome_selected_cache.npz")
    uidx, Xu = load_selected_cache(checkpoint_root / "cache/file10_unitig_selected_cache.npz")

    pang_idx = pd.read_csv(root / "data/features/pangenome/X_pangenome_feature_index.csv")
    pang_idx["Feature Index"] = pd.to_numeric(pang_idx["Feature Index"], errors="raise").astype(int)
    pang_id = pang_idx.set_index("Feature Index")["PLFam ID"].astype(str).to_dict()

    task_dir = checkpoint_root / "tasks/10F"
    task_dir.mkdir(parents=True, exist_ok=True)
    prog = Progress(len(shap_tasks), "10F SHAP tasks")
    imp_files = []
    summary_rows = []

    for task_num, (scheme, fold) in enumerate(shap_tasks, 1):
        task_id = f"{scheme}__fold{fold}"
        out_imp = task_dir / f"{task_id}__importance.csv"
        out_sum = task_dir / f"{task_id}__summary.json"
        marker = task_dir / f"{task_id}.json"
        imp_files.append(out_imp)

        if task_marker_valid(marker, task_id, [out_imp, out_sum]):
            summary_rows.append(read_json(out_sum))
            prog.update(f"{task_id} SKIP")
            continue

        tt = time.time()
        fcol = SCHEME_TO_COL[scheme]
        test_idx = np.flatnonzero(folds[fcol].to_numpy(dtype=int) == fold)
        train_idx = np.flatnonzero(folds[fcol].to_numpy(dtype=int) != fold)

        selected = {}
        X_parts = [Xk]
        names = [f"known::{x}" for x in known_names]
        types = ["known_amr"] * len(known_names)
        feature_indices = list(range(len(known_names)))

        for rep, cache_idx, cache_X in [
            ("pangenome", pidx, Xp),
            ("unitig", uidx, Xu),
        ]:
            h = fs[
                (fs["scheme"] == scheme)
                & (fs["fold"] == fold)
                & (fs["representation"] == rep)
                & (fs["rank"] <= shap_top)
            ].sort_values("rank")
            wanted = h["feature_index"].astype(int).tolist()
            fidx, Xsub = subset_cache(cache_idx, cache_X, wanted)
            X_parts.append(Xsub)
            if rep == "pangenome":
                names.extend([f"pangenome::{pang_id.get(int(fi), fi)}" for fi in fidx])
            else:
                names.extend([f"unitig::{int(fi)}" for fi in fidx])
            types.extend([rep] * len(fidx))
            feature_indices.extend([int(fi) for fi in fidx])
            selected[rep] = len(fidx)

        X = np.concatenate(X_parts, axis=1).astype(np.float32, copy=False)

        model = ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=workers,
            random_state=RANDOM_SEED,
        )
        fit_start = time.time()
        model.fit(X[train_idx], y[train_idx])
        fit_sec = time.time() - fit_start

        score = model.predict_proba(X[test_idx])[:, 1]
        auc = float(roc_auc_score(y[test_idx], score))
        ap = float(average_precision_score(y[test_idx], score))

        explain_idx = stratified_subsample(
            test_idx, y, shap_max_samples,
            RANDOM_SEED + fold + 100 * list(SCHEME_TO_COL).index(scheme),
        )
        exp_start = time.time()
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        try:
            sv_raw = explainer.shap_values(
                X[explain_idx], check_additivity=False, approximate=True
            )
        except TypeError:
            sv_raw = explainer.shap_values(
                X[explain_idx], check_additivity=False
            )
        sv = normalize_shap_array(sv_raw, len(explain_idx), X.shape[1])
        explain_sec = time.time() - exp_start

        mean_abs = np.mean(np.abs(sv), axis=0)
        mean_signed = np.mean(sv, axis=0)
        order = np.argsort(-mean_abs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order)+1)

        rows = []
        for j in range(X.shape[1]):
            rows.append({
                "scheme": scheme,
                "fold": fold,
                "feature_type": types[j],
                "feature_name": names[j],
                "feature_index": feature_indices[j],
                "mean_abs_shap": float(mean_abs[j]),
                "mean_signed_shap": float(mean_signed[j]),
                "shap_rank": int(ranks[j]),
                "n_explained_samples": len(explain_idx),
            })
        imp = pd.DataFrame(rows)
        atomic_write_csv(out_imp, imp)

        total_abs = float(mean_abs.sum())
        group_share = {}
        for typ in ("known_amr", "pangenome", "unitig"):
            m = np.array(types) == typ
            group_share[f"{typ}_total_abs_shap"] = float(mean_abs[m].sum())
            group_share[f"{typ}_share_abs_shap"] = float(mean_abs[m].sum() / total_abs) if total_abs else np.nan
            group_share[f"{typ}_mean_abs_shap_per_feature"] = float(mean_abs[m].mean()) if m.any() else np.nan

        summary = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "task_id": task_id,
            "scheme": scheme,
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_explained": len(explain_idx),
            "features_total": X.shape[1],
            "known_features": len(known_names),
            "pangenome_features": selected["pangenome"],
            "unitig_features": selected["unitig"],
            "heldout_auroc_sanity": auc,
            "heldout_average_precision_sanity": ap,
            "fit_seconds": fit_sec,
            "explain_seconds": explain_sec,
            **group_share,
        }
        atomic_write_text(out_sum, json.dumps(summary, indent=2) + "\n")
        mark_task(marker, task_id, [out_imp, out_sum], time.time()-tt)
        summary_rows.append(summary)
        prog.update(
            f"{task_id} | fit={human_seconds(fit_sec)} "
            f"| SHAP={human_seconds(explain_sec)} | AUROC={auc:.4f}"
        )

    all_imp = pd.concat([pd.read_csv(p) for p in imp_files if p.is_file()], ignore_index=True)
    atomic_write_csv(out_importance, all_imp)

    task_df = pd.DataFrame(summary_rows)
    atomic_write_csv(out_task_summary, task_df)

    group_cols = [
        "scheme", "fold",
        "known_amr_total_abs_shap", "known_amr_share_abs_shap", "known_amr_mean_abs_shap_per_feature",
        "pangenome_total_abs_shap", "pangenome_share_abs_shap", "pangenome_mean_abs_shap_per_feature",
        "unitig_total_abs_shap", "unitig_share_abs_shap", "unitig_mean_abs_shap_per_feature",
        "heldout_auroc_sanity", "heldout_average_precision_sanity",
    ]
    atomic_write_csv(out_group, task_df[group_cols].copy())

    mark_block(
        checkpoint_root, block, outputs, time.time()-t0,
        {"shap_tasks": len(shap_tasks), "shap_top_per_rep": shap_top, "max_samples": shap_max_samples},
    )
    log(f"10F PASS — SHAP rows={len(all_imp):,}, tasks={len(shap_tasks)} | {human_seconds(time.time()-t0)}")


# ---------------------------------------------------------------------------
# Block 10G — integrated biological candidates
# ---------------------------------------------------------------------------

def run_block_10G(root: Path, out_dir: Path, checkpoint_root: Path) -> None:
    block = "10G_integrated_candidates"
    out_broad = out_dir / "file10_biological_candidate_summary.csv"
    out_known = out_dir / "file10_known_amr_shap_summary.csv"
    outputs = [out_broad, out_known]

    if block_done(checkpoint_root, block, outputs):
        log("10G SKIP_COMPLETE — integrated candidate tables already checkpointed.")
        return

    t0 = time.time()
    log("10G START — integrate stability, association, lineage, redundancy, linkage, SHAP")

    cand = pd.read_csv(out_dir / "file10_candidate_features.csv")
    meta = pd.read_csv(out_dir / "file10_crossfitted_meta_effects.csv")
    cmh = pd.read_csv(out_dir / "file10_population_adjusted_association.csv")
    red = pd.read_csv(out_dir / "file10_broader_feature_redundancy.csv")
    link = pd.read_csv(out_dir / "file10_known_amr_linkage.csv")
    shap_imp = pd.read_csv(out_dir / "file10_shap_feature_importance.csv")

    broad = cand.merge(red, on=["representation", "feature_index"], how="left", validate="one_to_one")
    broad = broad.merge(link, on=["representation", "feature_index"], how="left", validate="one_to_one")

    # Cross-fitted meta effects: wide by scheme.
    meta_keep = meta[
        ["scheme", "representation", "feature_index", "n_folds", "random_or",
         "random_ci_low", "random_ci_high", "random_p", "random_p_fdr",
         "I2", "direction_concordance"]
    ].copy()
    for scheme in meta_keep["scheme"].drop_duplicates():
        h = meta_keep[meta_keep["scheme"] == scheme].drop(columns=["scheme"])
        rename = {
            c: f"{scheme}__{c}"
            for c in h.columns
            if c not in {"representation", "feature_index"}
        }
        broad = broad.merge(
            h.rename(columns=rename),
            on=["representation", "feature_index"],
            how="left",
            validate="one_to_one",
        )

    # CMH wide by stratifier.
    cmh_keep = cmh[
        ["representation", "feature_index", "stratifier", "mh_or", "cmh_p",
         "cmh_p_fdr", "informative_strata", "within_stratum_direction_concordance"]
    ].copy()
    for strat in cmh_keep["stratifier"].drop_duplicates():
        h = cmh_keep[cmh_keep["stratifier"] == strat].drop(columns=["stratifier"])
        rename = {
            c: f"{strat}__{c}"
            for c in h.columns
            if c not in {"representation", "feature_index"}
        }
        broad = broad.merge(
            h.rename(columns=rename),
            on=["representation", "feature_index"],
            how="left",
            validate="one_to_one",
        )

    # SHAP aggregation for broader features.
    s = shap_imp[shap_imp["feature_type"].isin(BROAD_REPS)].copy()
    s["feature_index"] = pd.to_numeric(s["feature_index"], errors="raise").astype(int)
    shap_agg = (
        s.groupby(["feature_type", "feature_index"])
        .agg(
            shap_task_appearances=("mean_abs_shap", "size"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            median_abs_shap=("mean_abs_shap", "median"),
            median_shap_rank=("shap_rank", "median"),
            best_shap_rank=("shap_rank", "min"),
        )
        .reset_index()
        .rename(columns={"feature_type": "representation"})
    )
    broad = broad.merge(
        shap_agg, on=["representation", "feature_index"], how="left", validate="one_to_one"
    )

    # Interpretation tags: transparent descriptive flags, not causal labels.
    broad["high_raw_known_amr_correlation"] = broad["max_raw_abs_phi"] >= 0.80
    broad["retains_population_centered_known_amr_correlation"] = (
        (broad["max_genomic_cluster_centered_abs_r"] >= 0.50)
        | (broad["max_mlst_centered_abs_r"] >= 0.50)
    )
    broad["high_redundancy_cluster"] = broad["cluster_size"].fillna(1) > 1

    atomic_write_csv(out_broad, broad)

    # Known-AMR SHAP summary.
    known_s = shap_imp[shap_imp["feature_type"] == "known_amr"].copy()
    known_s["Feature"] = known_s["feature_name"].str.replace(r"^known::", "", regex=True)
    known_agg = (
        known_s.groupby("Feature")
        .agg(
            shap_task_appearances=("mean_abs_shap", "size"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            median_abs_shap=("mean_abs_shap", "median"),
            median_shap_rank=("shap_rank", "median"),
            best_shap_rank=("shap_rank", "min"),
        )
        .reset_index()
    )
    known_meta = pd.read_csv(root / "data/features/known_amr/known_amr_feature_metadata_final.csv")
    known_out = known_agg.merge(known_meta, on="Feature", how="left", validate="one_to_one")
    known_out = known_out.sort_values(["mean_abs_shap", "median_shap_rank"], ascending=[False, True])
    atomic_write_csv(out_known, known_out)

    mark_block(
        checkpoint_root, block, outputs, time.time()-t0,
        {"broader_candidates": len(broad), "known_amr_features_with_shap": len(known_out)},
    )
    log(f"10G PASS — broader candidates={len(broad)}, known-AMR SHAP={len(known_out)} | {human_seconds(time.time()-t0)}")


# ---------------------------------------------------------------------------
# Block 10H — reviewer-defense matrix / final summary
# ---------------------------------------------------------------------------

def run_block_10H(
    root: Path,
    out_dir: Path,
    checkpoint_root: Path,
    smoke: bool,
    config: dict,
) -> None:
    block = "10H_reviewer_summary"
    out_review = out_dir / "file10_reviewer_defense_matrix.csv"
    out_summary = checkpoint_root / "file10_final_summary.json"
    outputs = [out_review, out_summary]

    if block_done(checkpoint_root, block, outputs):
        log("10H SKIP_COMPLETE — final reviewer summary already checkpointed.")
        return

    t0 = time.time()
    log("10H START — reviewer-defense matrix + final audit")

    audit = pd.read_csv(out_dir / "file10_input_integrity_audit.csv")
    stab = pd.read_csv(out_dir / "file10_feature_selection_stability.csv")
    jac = pd.read_csv(out_dir / "file10_feature_selection_jaccard.csv")
    meta = pd.read_csv(out_dir / "file10_crossfitted_meta_effects.csv")
    cmh = pd.read_csv(out_dir / "file10_population_adjusted_association.csv")
    red = pd.read_csv(out_dir / "file10_broader_feature_redundancy.csv")
    link = pd.read_csv(out_dir / "file10_known_amr_linkage.csv")
    shap_group = pd.read_csv(out_dir / "file10_shap_group_summary.csv")
    candidates = pd.read_csv(out_dir / "file10_biological_candidate_summary.csv")

    rows = [
        {
            "reviewer_question": "Were cohort linkage, row order, and feature representations verified before interpretation?",
            "analysis": "10A locked-input/cohort/orientation integrity audit",
            "status": "ASSESSED",
            "evidence": f"{int((audit['status']=='PASS').sum())}/{len(audit)} integrity checks PASS",
            "interpretation_guardrail": "No predictive metadata added.",
        },
        {
            "reviewer_question": "Are broader-feature findings stable across outer folds and validation schemes?",
            "analysis": "10B train-only selection frequency/rank + pairwise Jaccard",
            "status": "ASSESSED",
            "evidence": f"{len(stab):,} unique selected broader features; {len(jac):,} pairwise overlap comparisons",
            "interpretation_guardrail": "Stability is quantified rather than assumed.",
        },
        {
            "reviewer_question": "Are broader-feature associations evaluated out of sample after train-only selection?",
            "analysis": "10C cross-fitted held-out 2x2 effects + fixed/random-effects meta-analysis",
            "status": "ASSESSED",
            "evidence": f"{len(meta):,} feature/scheme meta-effect rows",
            "interpretation_guardrail": "Held-out effects are association estimates, not causal effects.",
        },
        {
            "reviewer_question": "Could top broader features be population-structure / lineage markers?",
            "analysis": "10D CMH adjustment by genomic cluster and MLST",
            "status": "ASSESSED",
            "evidence": f"{len(cmh):,} population-adjusted candidate tests with BH-FDR",
            "interpretation_guardrail": "Full-cohort adjusted tests are post-selection/exploratory.",
        },
        {
            "reviewer_question": "Could a few dominant lineages drive biological associations?",
            "analysis": "10D within-MLST effects + major-ST exclusion sensitivity",
            "status": "ASSESSED",
            "evidence": "Within-lineage effect table and top-lineage exclusion table generated.",
            "interpretation_guardrail": "Sparse/non-informative lineages are not forced into effect estimates.",
        },
        {
            "reviewer_question": "Are top broader features redundant / correlated proxies?",
            "analysis": "10E absolute-phi redundancy clusters + unitig sequence containment",
            "status": "ASSESSED",
            "evidence": f"{int((red['cluster_size']>1).sum())}/{len(red)} candidates belong to >=2-feature high-correlation clusters",
            "interpretation_guardrail": "Correlation/redundancy is not treated as physical linkage without locus mapping.",
        },
        {
            "reviewer_question": "Do broader features largely tag known-AMR determinants?",
            "analysis": "10E raw and within-lineage-centered correlation to all 668 known-AMR features",
            "status": "ASSESSED",
            "evidence": f"{int((link['max_raw_abs_phi']>=0.8).sum())}/{len(link)} candidates have |raw phi|>=0.8 to a known-AMR feature",
            "interpretation_guardrail": "Co-occurrence/correlation does not establish same locus or mechanism.",
        },
        {
            "reviewer_question": "Are model attributions stable under lineage-aware held-out evaluation?",
            "analysis": "10F cross-fitted ExtraTrees SHAP on genomic-cluster and MLST-aware folds",
            "status": "ASSESSED",
            "evidence": f"{len(shap_group)} lineage-aware SHAP fold summaries",
            "interpretation_guardrail": "SHAP explains model behavior; it is not causal biology.",
        },
        {
            "reviewer_question": "Was the negative incremental result specific to linear models or one hyperparameter choice?",
            "analysis": "File09b + File09c locked reviewer-defense analyses",
            "status": "ALREADY_ADDRESSED",
            "evidence": "Sensitivity grid plus 12 nonlinear settings across ExtraTrees, HistGB, and RBF-SVM.",
            "interpretation_guardrail": "File10 does not re-tune predictive models.",
        },
        {
            "reviewer_question": "Are PLFam/unitig candidates fully assigned to genes/loci and independently supported in literature?",
            "analysis": "10G local candidate prioritization; external sequence/literature annotation follows candidate extraction",
            "status": "REQUIRES_EXTERNAL_ANNOTATION",
            "evidence": f"{len(candidates)} prioritized broader candidates with local stability/association/linkage/SHAP evidence",
            "interpretation_guardrail": "Do not call unannotated PLFam/unitig signals novel mechanisms.",
        },
        {
            "reviewer_question": "Does the conclusion generalize to an independent external cohort?",
            "analysis": "File11",
            "status": "PENDING_FILE11",
            "evidence": "Not claimed by File10.",
            "interpretation_guardrail": "Internal lineage-aware validation is not a substitute for external validation.",
        },
    ]
    review = pd.DataFrame(rows)
    atomic_write_csv(out_review, review)

    summary = {
        "script_version": SCRIPT_VERSION,
        "design_id": DESIGN_ID,
        "status": "PASS",
        "mode": "SMOKE" if smoke else "FULL",
        "completed_utc": utc_now(),
        "config": config,
        "integrity_checks_pass": int((audit["status"] == "PASS").sum()),
        "integrity_checks_total": len(audit),
        "selection_stability_rows": len(stab),
        "jaccard_rows": len(jac),
        "crossfitted_meta_rows": len(meta),
        "population_adjusted_rows": len(cmh),
        "redundancy_rows": len(red),
        "known_amr_linkage_rows": len(link),
        "shap_fold_summaries": len(shap_group),
        "candidate_rows": len(candidates),
        "reviewer_defense_rows": len(review),
        "causal_claims": False,
        "predictive_retuning": False,
        "outer_test_feature_selection_for_prediction": False,
        "external_annotation_pending": True,
        "external_validation_pending_file11": True,
        "outputs": [str(p) for p in sorted(out_dir.glob("file10_*"))],
    }
    atomic_write_text(out_summary, json.dumps(summary, indent=2) + "\n")

    mark_block(checkpoint_root, block, outputs, time.time()-t0)
    log(f"10H PASS — reviewer-defense rows={len(review)} | {human_seconds(time.time()-t0)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--shap-top-per-rep", type=int, default=50)
    ap.add_argument("--shap-max-samples", type=int, default=300)
    ap.add_argument("--top-lineage-features", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not (1 <= args.workers <= 12):
        raise SystemExit("--workers must be 1..12")

    root = Path(args.project_root).resolve() if args.project_root else find_project_root(Path.cwd())

    if args.smoke:
        namespace = "explainability_biology_smoke"
        out_dir = root / "data/explainability_biology/smoke"
        top_n = min(args.top_n, 10)
        shap_top = min(args.shap_top_per_rep, 10)
        shap_max_samples = min(args.shap_max_samples, 50)
        top_lineage_features = min(args.top_lineage_features, 10)
        shap_tasks = [("mlst_aware", 0)]
    else:
        namespace = "explainability_biology"
        out_dir = root / "data/explainability_biology"
        top_n = args.top_n
        shap_top = args.shap_top_per_rep
        shap_max_samples = args.shap_max_samples
        top_lineage_features = args.top_lineage_features
        shap_tasks = [
            (scheme, fold)
            for scheme in LINEAGE_AWARE_SCHEMES
            for fold in range(5)
        ]

    checkpoint_root = root / "checkpoints" / namespace
    failure_path = checkpoint_root / "file10_last_failure.json"
    lock_path = checkpoint_root / "file10.lock"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Single-instance lock with stale-PID recovery.
    if lock_path.exists():
        try:
            old = read_json(lock_path)
            pid = int(old.get("pid", -1))
        except Exception:
            pid = -1
        if pid > 0 and Path(f"/proc/{pid}").exists():
            raise SystemExit(f"Active File10 lock exists (PID {pid}): {lock_path}")
        lock_path.unlink(missing_ok=True)

    atomic_write_text(
        lock_path,
        json.dumps(
            {
                "pid": os.getpid(),
                "script_version": SCRIPT_VERSION,
                "design_id": DESIGN_ID,
                "mode": "SMOKE" if args.smoke else "FULL",
                "started_utc": utc_now(),
            },
            indent=2,
        ) + "\n",
    )

    config = {
        "workers": args.workers,
        "top_n_per_broader_representation": top_n,
        "shap_top_per_representation_per_fold": shap_top,
        "shap_max_explained_samples_per_fold": shap_max_samples,
        "top_lineage_features_per_representation": top_lineage_features,
        "shap_tasks": shap_tasks,
        "mode": "SMOKE" if args.smoke else "FULL",
    }

    start = time.time()
    stage = "startup"
    try:
        log("=" * 92)
        log("FILE10 — EXPLAINABILITY + BIOLOGICAL VALIDATION")
        log("=" * 92)
        log(f"Script version : {SCRIPT_VERSION}")
        log(f"Design ID      : {DESIGN_ID}")
        log(f"Mode           : {'SMOKE' if args.smoke else 'FULL'}")
        log(f"Project root   : {root}")
        log(f"Workers        : {args.workers}")
        log(f"Top candidates : {top_n} / broader representation")
        log(f"SHAP design    : {len(shap_tasks)} tasks, top {shap_top}/rep/fold, max {shap_max_samples} samples")
        log("Checkpointing  : BLOCK + HEAVY SUB-TASK")
        log("Live progress  : ENABLED")

        stage = "10A"
        run_block_10A(root, out_dir, checkpoint_root)

        stage = "10B"
        run_block_10B(root, out_dir, checkpoint_root, top_n, shap_top, shap_tasks)

        stage = "10C"
        run_block_10C(root, out_dir, checkpoint_root, args.smoke)

        stage = "10D"
        run_block_10D(root, out_dir, checkpoint_root, top_lineage_features)

        stage = "10E"
        run_block_10E(root, out_dir, checkpoint_root)

        stage = "10F"
        run_block_10F(
            root, out_dir, checkpoint_root, shap_tasks,
            shap_top, shap_max_samples, args.workers,
        )

        stage = "10G"
        run_block_10G(root, out_dir, checkpoint_root)

        stage = "10H"
        run_block_10H(root, out_dir, checkpoint_root, args.smoke, config)

        failure_path.unlink(missing_ok=True)

        log("=" * 92)
        log("FILE10 STATUS : PASS")
        log(f"Mode          : {'SMOKE' if args.smoke else 'FULL'}")
        log(f"Elapsed       : {human_seconds(time.time()-start)}")
        log(f"Outputs       : {out_dir}")
        log(f"Final summary : {checkpoint_root / 'file10_final_summary.json'}")
        log("NOTE          : local computational biology is complete; external sequence/literature")
        log("                annotation of prioritized PLFam/unitig candidates remains a separate")
        log("                candidate-annotation step before File10 is scientifically LOCKED.")
        log("=" * 92)
        return 0

    except KeyboardInterrupt:
        failure = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "INTERRUPTED",
            "stage": stage,
            "time_utc": utc_now(),
            "elapsed_seconds": time.time()-start,
            "message": "Completed block/task checkpoints are preserved; re-run the same command to resume.",
        }
        atomic_write_text(failure_path, json.dumps(failure, indent=2) + "\n")
        log(f"FILE10 INTERRUPTED at {stage}. Checkpoints preserved.")
        return 130

    except Exception as exc:
        failure = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "FAILED",
            "stage": stage,
            "time_utc": utc_now(),
            "elapsed_seconds": time.time()-start,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "message": "Fix cause and re-run same command; compatible completed checkpoints are preserved.",
        }
        atomic_write_text(failure_path, json.dumps(failure, indent=2) + "\n")
        log(f"FILE10 FAILED at {stage}: {type(exc).__name__}: {exc}")
        log(f"Crash record: {failure_path}")
        return 1

    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
