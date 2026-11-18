#!/usr/bin/env python3
"""
File07 — Population Structure and Validation Design
===================================================

Purpose
-------
Build a reproducible, crash-resistant validation design for the common
Klebsiella pneumoniae cohort used by the genomic feature pipelines.

This script intentionally DOES NOT use MLST, QC statistics, Genome ID, or
population-structure variables as predictive model features. They are used
only for cohort linkage, population-structure auditing, and validation splits.

Main outputs
------------
1) Common cohort table with phenotype and lineage metadata.
2) Label-independent pangenome SVD embedding.
3) Label-independent pangenome-derived genomic clusters.
4) 5-fold random stratified outer CV assignments.
5) 5-fold genomic-cluster-aware outer CV assignments.
6) 5-fold MLST-aware outer CV assignments when MLST coverage is sufficient.
7) Split/population-structure audit tables and publication-ready figures.
8) Stage checkpoints, crash/interruption records, and final run summary.

Crash safety
------------
- Single-process lock prevents accidental concurrent runs.
- Stage outputs are written atomically via temporary files + os.replace().
- Every completed stage writes a JSON checkpoint.
- On restart, validated completed stages are skipped automatically.
- SIGINT/SIGTERM and uncaught exceptions write an interruption/crash record.
- --force recomputes all stages from scratch without overwriting inputs.

Scientific safeguards
---------------------
- Pangenome dimensionality reduction and genomic clustering are completely
  label-independent.
- Phenotype is used only for stratifying/diagnosing validation folds.
- Supervised feature selection is NOT performed here; it belongs inside
  training folds in File08.
- Genomic clusters from this script are a pangenome gene-content structure
  proxy, not a phylogeny. They should be described that way in the manuscript.

Expected project defaults
-------------------------
data/processed/unitig_full_k31_input_manifest.csv
  Master manifest containing Genome ID and meropenem phenotype.

data/features/pangenome/X_pangenome_plfam_binary.npz
  4,227 x 22,685 label-independent PLFam presence/absence matrix.

data/features/pangenome/X_pangenome_row_index.csv
  Row Index, Genome ID mapping for the pangenome matrix.

Run
---
From the repository root inside the amr-genome-ml WSL environment:

    python scripts/07_population_structure_and_validation_design.py

Useful options:

    python scripts/07_population_structure_and_validation_design.py --help
    python scripts/07_population_structure_and_validation_design.py --force
    python scripts/07_population_structure_and_validation_design.py --no-figures

Authoring note
--------------
Designed for the AMR-Genome-ML project after File06 was locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# -----------------------------------------------------------------------------
# Dependency preflight with a useful failure message.
# -----------------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.sparse import load_npz
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.preprocessing import normalize
except Exception as exc:  # pragma: no cover - preflight path
    print("\nFATAL: File07 Python dependencies are incomplete.\n", file=sys.stderr)
    print(f"Original import error: {exc}\n", file=sys.stderr)
    print(
        "Activate the project environment first:\n"
        "  micromamba activate amr-genome-ml\n\n"
        "Then ensure these packages exist:\n"
        "  numpy pandas scipy scikit-learn matplotlib\n",
        file=sys.stderr,
    )
    raise


SCRIPT_VERSION = "1.0.1"
DEFAULT_SEED = 20260918
DEFAULT_EXPECTED_N = 4227
DEFAULT_EXPECTED_R = 1601
DEFAULT_EXPECTED_S = 2626
DEFAULT_OUTER_FOLDS = 5
DEFAULT_SVD_COMPONENTS = 50
DEFAULT_MLST_MIN_COVERAGE = 0.80
DEFAULT_CLUSTER_KS = (20, 30, 40, 50, 60, 80, 100)

CURRENT_STAGE = "startup"
STOP_REQUESTED = False


# =============================================================================
# Utility helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.2f}h"


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.3f} {unit}"
        value /= 1024
    return f"{n} B"


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def normalize_genome_id(value: Any) -> str:
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def normalize_phenotype(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip().upper()
    mapping = {
        "R": "R",
        "RESISTANT": "R",
        "RESISTANCE": "R",
        "RES": "R",
        "S": "S",
        "SUSCEPTIBLE": "S",
        "SENSITIVE": "S",
        "SUSCEPTIBILITY": "S",
        "SUS": "S",
    }
    return mapping.get(s)


def normalize_mlst(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    upper = s.upper()
    if upper in {"NA", "N/A", "NONE", "NAN", "UNKNOWN", "UNDEFINED", "NOT AVAILABLE", "-"}:
        return None

    # Normalize common representations while preserving non-numeric schemes.
    m = re.fullmatch(r"(?:ST\s*)?(\d+)(?:\.0)?", upper)
    if m:
        return f"ST{int(m.group(1))}"

    return s


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fsync_file_handle(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def atomic_write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        fsync_file_handle(f)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_dataframe(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    ensure_parent(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=index)
    # pandas closes file internally; reopen for fsync.
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    ensure_parent(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def sha256_small_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_signature(path: Path, *, hash_if_under_mb: int = 64) -> dict[str, Any]:
    stat = path.stat()
    out: dict[str, Any] = {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stat.st_size <= hash_if_under_mb * 1024 * 1024:
        out["sha256"] = sha256_small_file(path)
    return out


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


@contextmanager
def stage(name: str):
    global CURRENT_STAGE
    CURRENT_STAGE = name
    t0 = time.time()
    log(f"START {name}")
    try:
        yield
    except Exception:
        log(f"FAILED {name} after {human_seconds(time.time() - t0)}")
        raise
    else:
        log(f"DONE  {name} in {human_seconds(time.time() - t0)}")


# =============================================================================
# Run locking and crash/interruption handling
# =============================================================================

def write_crash_record(checkpoint_dir: Path, *, kind: str, detail: str) -> None:
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": utc_now(),
            "kind": kind,
            "stage": CURRENT_STAGE,
            "pid": os.getpid(),
            "detail": detail,
        }
        atomic_write_json(checkpoint_dir / "07_last_failure.json", payload)
    except Exception:
        # Never mask the original failure.
        pass


def signal_handler(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    raise KeyboardInterrupt(f"Received signal {signum}")


@contextmanager
def exclusive_lock(lock_path: Path):
    """Linux/WSL advisory lock. Prevents two File07 runs at once."""
    import fcntl

    ensure_parent(lock_path)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another File07 process appears to be running. Lock: {lock_path}"
        ) from exc

    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\nstarted_utc={utc_now()}\n")
        fsync_file_handle(handle)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class Config:
    project_root: Path
    manifest: Path
    pangenome_matrix: Path
    pangenome_row_index: Path
    mlst_metadata: Optional[Path]

    processed_dir: Path
    split_dir: Path
    checkpoint_dir: Path
    figure_dir: Path

    expected_n: int
    expected_r: int
    expected_s: int
    strict_counts: bool

    seed: int
    outer_folds: int
    svd_components: int
    cluster_ks: tuple[int, ...]
    silhouette_sample_size: int
    mlst_min_coverage: float

    force: bool
    make_figures: bool

    @property
    def cohort_csv(self) -> Path:
        return self.processed_dir / "file07_common_cohort.csv"

    @property
    def svd_npz(self) -> Path:
        return self.checkpoint_dir / "file07_pangenome_svd.npz"

    @property
    def svd_csv(self) -> Path:
        return self.processed_dir / "file07_pangenome_svd_coordinates.csv"

    @property
    def cluster_tuning_csv(self) -> Path:
        return self.processed_dir / "file07_genomic_cluster_tuning.csv"

    @property
    def population_csv(self) -> Path:
        return self.processed_dir / "file07_population_structure.csv"

    @property
    def folds_csv(self) -> Path:
        return self.split_dir / "file07_outer_folds.csv"

    @property
    def split_audit_csv(self) -> Path:
        return self.split_dir / "file07_split_audit.csv"

    @property
    def cluster_audit_csv(self) -> Path:
        return self.processed_dir / "file07_cluster_audit.csv"

    @property
    def summary_json(self) -> Path:
        return self.checkpoint_dir / "file07_final_summary.json"

    @property
    def run_log(self) -> Path:
        return self.checkpoint_dir / "file07_run.log"


# =============================================================================
# Checkpoint helpers
# =============================================================================

def checkpoint_path(cfg: Config, stage_number: int, stage_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_name)
    return cfg.checkpoint_dir / f"07_stage_{stage_number:02d}_{safe}.json"


def write_stage_checkpoint(
    cfg: Config,
    stage_number: int,
    stage_name: str,
    outputs: Iterable[Path],
    details: dict[str, Any],
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": utc_now(),
        "stage_number": stage_number,
        "stage_name": stage_name,
        "status": "COMPLETE",
        "outputs": [file_signature(p) for p in outputs if p.exists()],
        "details": details,
    }
    atomic_write_json(checkpoint_path(cfg, stage_number, stage_name), payload)


def stage_checkpoint_exists(cfg: Config, stage_number: int, stage_name: str) -> bool:
    return checkpoint_path(cfg, stage_number, stage_name).exists()


# =============================================================================
# Metadata discovery
# =============================================================================
GENOME_ID_CANDIDATES = {
    "genomeid",
    "genome",
    "genomeidentifier",
    "patricgenomeid",
    "bvbrcgenomeid",
}

PHENOTYPE_CANDIDATES = {
    "phenotype",
    "meropenem",
    "meropenemphenotype",
    "meropenemsusceptibility",
    "meropenemresistance",
    "resistancephenotype",
    "resistantphenotype",
    "amrphenotype",
}

MLST_CANDIDATES = {
    "mlst",
    "mlstst",
    "sequencetype",
    "sequence type",
    "st",
    "mlstsequencetype",
}
MLST_CANDIDATES_NORMALIZED = {normalize_colname(x) for x in MLST_CANDIDATES}


def detect_genome_id_column(df: pd.DataFrame) -> str:
    normalized = {normalize_colname(c): c for c in df.columns}
    for key in GENOME_ID_CANDIDATES:
        if key in normalized:
            return normalized[key]
    raise ValueError(
        "Could not identify Genome ID column. Columns: " + ", ".join(map(str, df.columns))
    )


def score_phenotype_column(series: pd.Series) -> tuple[int, float]:
    vals = series.map(normalize_phenotype)
    valid = vals.notna()
    return int(valid.sum()), float(valid.mean()) if len(vals) else 0.0


def detect_phenotype_column(df: pd.DataFrame) -> str:
    candidates: list[tuple[int, float, str]] = []
    for col in df.columns:
        norm = normalize_colname(col)
        if norm in PHENOTYPE_CANDIDATES or "meropenem" in norm or norm == "phenotype":
            n_valid, frac = score_phenotype_column(df[col])
            candidates.append((n_valid, frac, str(col)))

    if not candidates:
        # Fallback: score every low-cardinality string-like column.
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 20:
                n_valid, frac = score_phenotype_column(df[col])
                if n_valid > 0:
                    candidates.append((n_valid, frac, str(col)))

    if not candidates:
        raise ValueError("Could not identify a meropenem R/S phenotype column.")

    candidates.sort(reverse=True)
    best = candidates[0]
    if best[0] == 0:
        raise ValueError("No R/S phenotype values detected.")
    return best[2]


def detect_mlst_column(df: pd.DataFrame) -> Optional[str]:
    scored: list[tuple[int, str]] = []
    for col in df.columns:
        if normalize_colname(col) in MLST_CANDIDATES_NORMALIZED:
            normalized = df[col].map(normalize_mlst)
            scored.append((int(normalized.notna().sum()), str(col)))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def load_table_with_genome_id(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    gid_col = detect_genome_id_column(df)
    df[gid_col] = df[gid_col].map(normalize_genome_id)
    return df, gid_col


def discover_best_mlst_source(
    cfg: Config,
    cohort_ids: set[str],
    manifest_df: pd.DataFrame,
    manifest_gid_col: str,
) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str], float]:
    """Return dataframe, gid column, MLST column, coverage fraction."""

    candidates: list[tuple[float, int, str, pd.DataFrame, str, str]] = []

    def consider(path_label: str, df: pd.DataFrame, gid_col: str) -> None:
        mlst_col = detect_mlst_column(df)
        if not mlst_col:
            return
        sub = df[[gid_col, mlst_col]].copy()
        sub[gid_col] = sub[gid_col].map(normalize_genome_id)
        sub[mlst_col] = sub[mlst_col].map(normalize_mlst)
        sub = sub[sub[gid_col].isin(cohort_ids)]
        sub = sub.drop_duplicates(subset=[gid_col], keep="first")
        coverage = float(sub[mlst_col].notna().sum() / len(cohort_ids))
        n_unique = int(sub[mlst_col].nunique(dropna=True))
        candidates.append((coverage, n_unique, path_label, sub, gid_col, mlst_col))

    consider(str(cfg.manifest), manifest_df, manifest_gid_col)

    if cfg.mlst_metadata is not None:
        df, gid_col = load_table_with_genome_id(cfg.mlst_metadata)
        consider(str(cfg.mlst_metadata), df, gid_col)
    else:
        # Conservative discovery: only scan processed CSVs, never huge feature files.
        for path in sorted(cfg.processed_dir.glob("*.csv")):
            if path.resolve() == cfg.manifest.resolve():
                continue
            try:
                # Avoid accidentally reading a giant table in this discovery pass.
                if path.stat().st_size > 512 * 1024 * 1024:
                    continue
                df, gid_col = load_table_with_genome_id(path)
                consider(str(path), df, gid_col)
            except Exception:
                continue

    if not candidates:
        return None, None, None, 0.0

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    coverage, _n_unique, source, df, gid_col, mlst_col = candidates[0]
    return df, gid_col, mlst_col, coverage


# =============================================================================
# Stage 1 — common cohort and metadata linkage
# =============================================================================

def build_common_cohort(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_index = pd.read_csv(cfg.pangenome_row_index, dtype=str)

    normalized_cols = {normalize_colname(c): c for c in row_index.columns}
    if "rowindex" not in normalized_cols or "genomeid" not in normalized_cols:
        raise ValueError(
            "Pangenome row index must contain 'Row Index' and 'Genome ID'."
        )

    row_col = normalized_cols["rowindex"]
    gid_col = normalized_cols["genomeid"]

    row_index[row_col] = pd.to_numeric(row_index[row_col], errors="raise").astype(int)
    row_index[gid_col] = row_index[gid_col].map(normalize_genome_id)
    row_index = row_index.sort_values(row_col).reset_index(drop=True)

    if row_index[gid_col].duplicated().any():
        dup = row_index.loc[row_index[gid_col].duplicated(), gid_col].iloc[0]
        raise ValueError(f"Duplicate Genome ID in pangenome row index: {dup}")

    expected_rows = np.arange(len(row_index), dtype=int)
    if not np.array_equal(row_index[row_col].to_numpy(), expected_rows):
        raise ValueError("Pangenome Row Index is not contiguous 0..N-1.")

    manifest_df, manifest_gid_col = load_table_with_genome_id(cfg.manifest)
    phenotype_col = detect_phenotype_column(manifest_df)

    manifest_df["__phenotype__"] = manifest_df[phenotype_col].map(normalize_phenotype)
    manifest_df = manifest_df.drop_duplicates(subset=[manifest_gid_col], keep="first")

    phenotype_map = manifest_df.set_index(manifest_gid_col)["__phenotype__"]

    cohort = pd.DataFrame(
        {
            "Row Index": row_index[row_col].astype(int),
            "Genome ID": row_index[gid_col].astype(str),
        }
    )
    cohort["Phenotype"] = cohort["Genome ID"].map(phenotype_map)

    if cohort["Phenotype"].isna().any():
        missing = cohort.loc[cohort["Phenotype"].isna(), "Genome ID"].head(10).tolist()
        raise ValueError(
            f"Missing meropenem phenotype for {cohort['Phenotype'].isna().sum()} cohort samples. "
            f"Examples: {missing}"
        )

    cohort["y"] = (cohort["Phenotype"] == "R").astype(np.uint8)

    cohort_ids = set(cohort["Genome ID"])
    mlst_df, mlst_gid_col, mlst_col, mlst_coverage = discover_best_mlst_source(
        cfg, cohort_ids, manifest_df, manifest_gid_col
    )

    mlst_source = None
    if mlst_df is not None and mlst_gid_col and mlst_col:
        mlst_source = "autodetected"
        mlst_map = (
            mlst_df.drop_duplicates(subset=[mlst_gid_col], keep="first")
            .set_index(mlst_gid_col)[mlst_col]
            .map(normalize_mlst)
        )
        cohort["MLST"] = cohort["Genome ID"].map(mlst_map)
    else:
        cohort["MLST"] = pd.Series([None] * len(cohort), dtype="object")

    n = len(cohort)
    n_r = int((cohort["Phenotype"] == "R").sum())
    n_s = int((cohort["Phenotype"] == "S").sum())

    if n != cfg.expected_n:
        raise ValueError(f"Expected {cfg.expected_n:,} samples, found {n:,}.")

    if cfg.strict_counts and (n_r != cfg.expected_r or n_s != cfg.expected_s):
        raise ValueError(
            "Phenotype counts differ from locked File05 common cohort: "
            f"observed R={n_r}, S={n_s}; expected R={cfg.expected_r}, S={cfg.expected_s}."
        )

    details = {
        "n_samples": n,
        "n_R": n_r,
        "n_S": n_s,
        "phenotype_column": phenotype_col,
        "manifest_genome_id_column": manifest_gid_col,
        "mlst_column": mlst_col,
        "mlst_coverage": float(cohort["MLST"].notna().mean()),
        "mlst_unique_nonmissing": int(cohort["MLST"].nunique(dropna=True)),
        "mlst_source_detected": mlst_source,
    }

    return cohort, details


def validate_cohort_file(cfg: Config) -> bool:
    try:
        df = pd.read_csv(cfg.cohort_csv, dtype={"Genome ID": str})
        if len(df) != cfg.expected_n:
            return False
        if df["Genome ID"].duplicated().any():
            return False
        if not set(df["Phenotype"]).issubset({"R", "S"}):
            return False
        if cfg.strict_counts:
            counts = df["Phenotype"].value_counts().to_dict()
            if int(counts.get("R", 0)) != cfg.expected_r:
                return False
            if int(counts.get("S", 0)) != cfg.expected_s:
                return False
        return True
    except Exception:
        return False


# =============================================================================
# Stage 2 — label-independent pangenome SVD
# =============================================================================

def compute_svd(cfg: Config, cohort: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = load_npz(cfg.pangenome_matrix)
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()

    if X.shape[0] != len(cohort):
        raise ValueError(
            f"Pangenome matrix rows {X.shape[0]} != cohort rows {len(cohort)}."
        )

    expected_rows = cohort["Row Index"].to_numpy(dtype=int)
    if not np.array_equal(expected_rows, np.arange(len(cohort))):
        raise ValueError("Cohort row order no longer matches pangenome matrix row order.")

    n_components = min(cfg.svd_components, X.shape[0] - 1, X.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Not enough dimensions for SVD.")

    log(
        f"Pangenome matrix: {X.shape[0]:,} x {X.shape[1]:,}; "
        f"nnz={X.nnz:,}; normalizing rows (label-independent)."
    )

    # Cosine-style gene-content geometry; preserves sparsity.
    X_norm = normalize(X, norm="l2", axis=1, copy=True)

    log(
        f"Running randomized TruncatedSVD with {n_components} components. "
        "This is the heaviest File07 stage and may take several minutes."
    )
    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        n_iter=7,
        random_state=cfg.seed,
    )
    Z = svd.fit_transform(X_norm).astype(np.float32, copy=False)
    explained = svd.explained_variance_ratio_.astype(np.float64, copy=False)

    atomic_save_npz(
        cfg.svd_npz,
        embedding=Z,
        explained_variance_ratio=explained,
        genome_ids=cohort["Genome ID"].astype(str).to_numpy(dtype="U"),
        seed=np.array([cfg.seed], dtype=np.int64),
    )

    coord_cols = {"Genome ID": cohort["Genome ID"].astype(str)}
    for i in range(min(10, Z.shape[1])):
        coord_cols[f"SVD{i + 1}"] = Z[:, i]
    atomic_write_dataframe(pd.DataFrame(coord_cols), cfg.svd_csv, index=False)

    return Z, explained


def load_valid_svd(cfg: Config, cohort: pd.DataFrame) -> Optional[tuple[np.ndarray, np.ndarray]]:
    try:
        data = np.load(cfg.svd_npz, allow_pickle=False)
        Z = data["embedding"]
        explained = data["explained_variance_ratio"]
        ids = data["genome_ids"].astype(str)
        if Z.shape[0] != len(cohort):
            return None
        if not np.array_equal(ids, cohort["Genome ID"].astype(str).to_numpy()):
            return None
        if Z.ndim != 2 or Z.shape[1] < 2:
            return None
        return Z, explained
    except Exception:
        return None


# =============================================================================
# Stage 3 — genomic-cluster tuning (label-independent)
# =============================================================================

def tune_genomic_clusters(cfg: Config, Z: np.ndarray) -> pd.DataFrame:
    # Normalize embedding before Euclidean KMeans to reduce scale dominance.
    Zc = normalize(Z, norm="l2", axis=1, copy=True)

    records: list[dict[str, Any]] = []
    n = Zc.shape[0]
    sample_size = min(cfg.silhouette_sample_size, n)

    for i, k in enumerate(cfg.cluster_ks, start=1):
        if k >= n:
            continue
        log(f"Cluster tuning {i}/{len(cfg.cluster_ks)}: K={k}")
        t0 = time.time()
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=500,
            random_state=cfg.seed + k,
        )
        labels = model.fit_predict(Zc)
        sizes = np.bincount(labels, minlength=k)

        sil = silhouette_score(
            Zc,
            labels,
            metric="euclidean",
            sample_size=sample_size if sample_size < n else None,
            random_state=cfg.seed,
        )

        frac_in_clusters_ge5 = float(sizes[sizes >= 5].sum() / n)
        max_share = float(sizes.max() / n)
        record = {
            "k": int(k),
            "silhouette": float(sil),
            "min_cluster_size": int(sizes.min()),
            "median_cluster_size": float(np.median(sizes)),
            "max_cluster_size": int(sizes.max()),
            "max_cluster_share": max_share,
            "fraction_samples_in_clusters_ge5": frac_in_clusters_ge5,
            "elapsed_seconds": float(time.time() - t0),
        }
        records.append(record)
        log(
            f"K={k}: silhouette={sil:.4f}, min={sizes.min()}, "
            f"median={np.median(sizes):.1f}, max={sizes.max()} "
            f"({max_share:.1%}); {human_seconds(record['elapsed_seconds'])}"
        )

    tuning = pd.DataFrame(records)
    if tuning.empty:
        raise ValueError("No valid genomic-cluster K candidates.")

    # Pre-specified quality constraints are label-independent.
    eligible = tuning[
        (tuning["fraction_samples_in_clusters_ge5"] >= 0.95)
        & (tuning["max_cluster_share"] <= 0.25)
    ]
    if eligible.empty:
        log(
            "WARNING: no K candidate met cluster-size constraints; "
            "selecting highest silhouette across all candidates."
        )
        chosen_idx = tuning["silhouette"].idxmax()
        selection_rule = "max_silhouette_no_candidate_met_size_constraints"
    else:
        chosen_idx = eligible["silhouette"].idxmax()
        selection_rule = "max_silhouette_among_size_eligible_candidates"

    tuning["selected"] = False
    tuning.loc[chosen_idx, "selected"] = True
    tuning["selection_rule"] = selection_rule

    atomic_write_dataframe(tuning, cfg.cluster_tuning_csv, index=False)
    return tuning


def fit_selected_clusters(cfg: Config, Z: np.ndarray, tuning: pd.DataFrame) -> np.ndarray:
    selected = tuning.loc[tuning["selected"] == True, "k"]  # noqa: E712
    if len(selected) != 1:
        raise ValueError("Expected exactly one selected genomic-cluster K.")
    k = int(selected.iloc[0])
    log(f"Fitting final label-independent genomic clusters with K={k}.")

    Zc = normalize(Z, norm="l2", axis=1, copy=True)
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=50,
        max_iter=1000,
        random_state=cfg.seed + k,
    )
    labels = model.fit_predict(Zc)
    return labels.astype(np.int32)


# =============================================================================
# Stage 4 — validation folds
# =============================================================================

def assign_random_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    folds = np.full(len(y), -1, dtype=np.int16)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((len(y), 1), dtype=np.uint8)
    for fold, (_, test_idx) in enumerate(cv.split(dummy, y)):
        folds[test_idx] = fold
    if np.any(folds < 0):
        raise RuntimeError("Random fold assignment incomplete.")
    return folds


def assign_group_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique groups; found {len(np.unique(groups))}."
        )
    folds = np.full(len(y), -1, dtype=np.int16)
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros((len(y), 1), dtype=np.uint8)
    for fold, (_, test_idx) in enumerate(cv.split(dummy, y, groups=groups)):
        folds[test_idx] = fold
    if np.any(folds < 0):
        raise RuntimeError("Group fold assignment incomplete.")
    return folds


def audit_fold_scheme(
    df: pd.DataFrame,
    scheme: str,
    fold_col: str,
    group_col: Optional[str],
    n_splits: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    assigned = df[fold_col].dropna().astype(int)
    if len(assigned) != len(df):
        raise ValueError(f"{scheme}: not every sample has a fold assignment.")

    if set(assigned.unique()) != set(range(n_splits)):
        raise ValueError(f"{scheme}: expected folds 0..{n_splits - 1}.")

    if group_col is not None:
        group_span = df.groupby(group_col, dropna=False)[fold_col].nunique(dropna=False)
        if int(group_span.max()) != 1:
            offenders = group_span[group_span > 1].head(10).index.tolist()
            raise ValueError(f"{scheme}: groups cross folds: {offenders}")

    overall_r = float((df["Phenotype"] == "R").mean())

    for fold in range(n_splits):
        sub = df[df[fold_col].astype(int) == fold]
        n = len(sub)
        n_r = int((sub["Phenotype"] == "R").sum())
        n_s = int((sub["Phenotype"] == "S").sum())
        if n_r == 0 or n_s == 0:
            raise ValueError(f"{scheme}: fold {fold} lacks one phenotype class.")

        row: dict[str, Any] = {
            "scheme": scheme,
            "fold": fold,
            "n": n,
            "R": n_r,
            "S": n_s,
            "R_fraction": n_r / n,
            "absolute_R_fraction_shift": abs((n_r / n) - overall_r),
        }
        if group_col is not None:
            row["n_groups"] = int(sub[group_col].nunique(dropna=False))
        else:
            row["n_groups"] = np.nan
        rows.append(row)

    return rows


# =============================================================================
# Stage 5 — population/split audit tables
# =============================================================================

def build_cluster_audit(population: pd.DataFrame) -> pd.DataFrame:
    grouped = population.groupby("Genomic Cluster", sort=True, dropna=False)
    rows = []
    for cluster, sub in grouped:
        n = len(sub)
        n_r = int((sub["Phenotype"] == "R").sum())
        rows.append(
            {
                "Genomic Cluster": int(cluster),
                "n": n,
                "R": n_r,
                "S": n - n_r,
                "R_fraction": n_r / n,
                "unique_MLST_nonmissing": int(sub["MLST"].nunique(dropna=True)),
                "MLST_coverage": float(sub["MLST"].notna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "Genomic Cluster"], ascending=[False, True])


# =============================================================================
# Stage 6 — figures
# =============================================================================

def make_figures(cfg: Config, population: pd.DataFrame, split_audit: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Figure A: first two SVD dimensions by phenotype. Phenotype is used only
    # for visualization after label-independent embedding construction.
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for phenotype in ["S", "R"]:
        sub = population[population["Phenotype"] == phenotype]
        ax.scatter(
            sub["SVD1"],
            sub["SVD2"],
            s=10,
            alpha=0.55,
            label=phenotype,
            linewidths=0,
        )
    ax.set_xlabel("Pangenome SVD1")
    ax.set_ylabel("Pangenome SVD2")
    ax.set_title("Pangenome-derived population structure")
    ax.legend(title="Meropenem phenotype", frameon=False)
    fig.tight_layout()

    png = cfg.figure_dir / "file07_population_structure_svd.png"
    pdf = cfg.figure_dir / "file07_population_structure_svd.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    outputs.extend([png, pdf])

    # Figure B: phenotype fraction by fold and split scheme.
    schemes = list(split_audit["scheme"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    width = 0.8 / max(1, len(schemes))
    x = np.arange(cfg.outer_folds)
    for i, scheme in enumerate(schemes):
        sub = split_audit[split_audit["scheme"] == scheme].sort_values("fold")
        positions = x - 0.4 + width / 2 + i * width
        ax.bar(positions, sub["R_fraction"].to_numpy(), width=width, label=scheme)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(cfg.outer_folds)])
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Resistant fraction")
    ax.set_title("Phenotype balance across validation schemes")
    ax.legend(frameon=False)
    fig.tight_layout()

    png = cfg.figure_dir / "file07_validation_fold_balance.png"
    pdf = cfg.figure_dir / "file07_validation_fold_balance.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    outputs.extend([png, pdf])

    return outputs


# =============================================================================
# Main pipeline
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="File07 population structure and validation design",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/unitig_full_k31_input_manifest.csv"),
    )
    parser.add_argument(
        "--pangenome-matrix",
        type=Path,
        default=Path("data/features/pangenome/X_pangenome_plfam_binary.npz"),
    )
    parser.add_argument(
        "--pangenome-row-index",
        type=Path,
        default=Path("data/features/pangenome/X_pangenome_row_index.csv"),
    )
    parser.add_argument(
        "--mlst-metadata",
        type=Path,
        default=None,
        help="Optional metadata CSV containing Genome ID and MLST/ST.",
    )
    parser.add_argument("--expected-n", type=int, default=DEFAULT_EXPECTED_N)
    parser.add_argument("--expected-r", type=int, default=DEFAULT_EXPECTED_R)
    parser.add_argument("--expected-s", type=int, default=DEFAULT_EXPECTED_S)
    parser.add_argument(
        "--no-strict-counts",
        action="store_true",
        help="Do not enforce the locked R/S counts from File05.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    parser.add_argument("--svd-components", type=int, default=DEFAULT_SVD_COMPONENTS)
    parser.add_argument(
        "--cluster-ks",
        type=str,
        default=",".join(map(str, DEFAULT_CLUSTER_KS)),
        help="Comma-separated candidate K values for label-independent genomic clustering.",
    )
    parser.add_argument("--silhouette-sample-size", type=int, default=2000)
    parser.add_argument("--mlst-min-coverage", type=float, default=DEFAULT_MLST_MIN_COVERAGE)
    parser.add_argument("--force", action="store_true", help="Recompute completed stages.")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> Config:
    root = args.project_root.resolve()

    def rp(path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        return path if path.is_absolute() else (root / path)

    cluster_ks = tuple(sorted({int(x.strip()) for x in args.cluster_ks.split(",") if x.strip()}))
    if not cluster_ks:
        raise ValueError("--cluster-ks produced an empty K list.")

    cfg = Config(
        project_root=root,
        manifest=rp(args.manifest),
        pangenome_matrix=rp(args.pangenome_matrix),
        pangenome_row_index=rp(args.pangenome_row_index),
        mlst_metadata=rp(args.mlst_metadata),
        processed_dir=root / "data/processed",
        split_dir=root / "data/splits",
        checkpoint_dir=root / "checkpoints/population_structure",
        figure_dir=root / "manuscript/figures",
        expected_n=args.expected_n,
        expected_r=args.expected_r,
        expected_s=args.expected_s,
        strict_counts=not args.no_strict_counts,
        seed=args.seed,
        outer_folds=args.outer_folds,
        svd_components=args.svd_components,
        cluster_ks=cluster_ks,
        silhouette_sample_size=args.silhouette_sample_size,
        mlst_min_coverage=args.mlst_min_coverage,
        force=args.force,
        make_figures=not args.no_figures,
    )
    return cfg


def preflight(cfg: Config) -> None:
    log(f"File07 script version: {SCRIPT_VERSION}")
    log(f"Project root: {cfg.project_root}")
    log(f"Python: {sys.executable}")
    log(f"Python version: {sys.version.split()[0]}")

    required = [cfg.manifest, cfg.pangenome_matrix, cfg.pangenome_row_index]
    if cfg.mlst_metadata is not None:
        required.append(cfg.mlst_metadata)

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
        log(f"Input OK: {path.relative_to(cfg.project_root) if path.is_relative_to(cfg.project_root) else path} ({human_bytes(path.stat().st_size)})")

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.split_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if cfg.make_figures:
        cfg.figure_dir.mkdir(parents=True, exist_ok=True)

    # Refuse obviously wrong environment when run accidentally with Windows Python.
    if os.name != "posix":
        raise RuntimeError(
            "File07 is intended to run inside the WSL amr-genome-ml environment. "
            f"Detected os.name={os.name!r}."
        )

    # Record reproducibility metadata.
    environment = {
        "timestamp_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": __import__("sklearn").__version__,
        "seed": cfg.seed,
        "outer_folds": cfg.outer_folds,
        "svd_components": cfg.svd_components,
        "cluster_ks": list(cfg.cluster_ks),
        "inputs": [file_signature(p) for p in required],
    }
    atomic_write_json(cfg.checkpoint_dir / "07_environment.json", environment)


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with exclusive_lock(cfg.checkpoint_dir / "07_run.lock"):
        preflight(cfg)

        # ------------------------------------------------------------------
        # Stage 1: common cohort linkage
        # ------------------------------------------------------------------
        stage_name = "common_cohort"
        with stage(stage_name):
            if (
                not cfg.force
                and stage_checkpoint_exists(cfg, 1, stage_name)
                and validate_cohort_file(cfg)
            ):
                cohort = pd.read_csv(cfg.cohort_csv, dtype={"Genome ID": str, "MLST": str})
                cohort["MLST"] = cohort["MLST"].replace({"nan": np.nan})
                log("SKIP stage 1: validated common-cohort checkpoint found.")
                stage1_details = {
                    "n_samples": len(cohort),
                    "n_R": int((cohort["Phenotype"] == "R").sum()),
                    "n_S": int((cohort["Phenotype"] == "S").sum()),
                    "mlst_coverage": float(cohort["MLST"].notna().mean()),
                    "mlst_unique_nonmissing": int(cohort["MLST"].nunique(dropna=True)),
                    "resumed_from_checkpoint": True,
                }
            else:
                cohort, stage1_details = build_common_cohort(cfg)
                atomic_write_dataframe(cohort, cfg.cohort_csv, index=False)
                write_stage_checkpoint(cfg, 1, stage_name, [cfg.cohort_csv], stage1_details)

            log(
                f"Common cohort: n={len(cohort):,}; "
                f"R={(cohort['Phenotype'] == 'R').sum():,}; "
                f"S={(cohort['Phenotype'] == 'S').sum():,}; "
                f"MLST coverage={cohort['MLST'].notna().mean():.1%}."
            )

        # ------------------------------------------------------------------
        # Stage 2: pangenome SVD
        # ------------------------------------------------------------------
        stage_name = "pangenome_svd"
        with stage(stage_name):
            cached = None if cfg.force else load_valid_svd(cfg, cohort)
            if cached is not None and stage_checkpoint_exists(cfg, 2, stage_name):
                Z, explained = cached
                log("SKIP stage 2: validated SVD checkpoint found.")
            else:
                Z, explained = compute_svd(cfg, cohort)
                write_stage_checkpoint(
                    cfg,
                    2,
                    stage_name,
                    [cfg.svd_npz, cfg.svd_csv],
                    {
                        "shape": list(Z.shape),
                        "explained_variance_ratio_sum": float(explained.sum()),
                        "label_used": False,
                    },
                )
            log(
                f"SVD shape={Z.shape}; cumulative explained variance="
                f"{explained.sum():.4f}."
            )

        # ------------------------------------------------------------------
        # Stage 3: genomic cluster tuning
        # ------------------------------------------------------------------
        stage_name = "genomic_cluster_tuning"
        with stage(stage_name):
            if (
                not cfg.force
                and stage_checkpoint_exists(cfg, 3, stage_name)
                and cfg.cluster_tuning_csv.exists()
            ):
                tuning = pd.read_csv(cfg.cluster_tuning_csv)
                if tuning["selected"].astype(str).str.lower().eq("true").sum() != 1:
                    raise ValueError("Cached cluster tuning has invalid selected-row count.")
                # Restore boolean reliably across pandas versions.
                tuning["selected"] = tuning["selected"].astype(str).str.lower().eq("true")
                log("SKIP stage 3: validated cluster-tuning checkpoint found.")
            else:
                tuning = tune_genomic_clusters(cfg, Z)
                chosen_k = int(tuning.loc[tuning["selected"], "k"].iloc[0])
                write_stage_checkpoint(
                    cfg,
                    3,
                    stage_name,
                    [cfg.cluster_tuning_csv],
                    {
                        "candidate_ks": list(cfg.cluster_ks),
                        "selected_k": chosen_k,
                        "selection_rule": tuning.loc[tuning["selected"], "selection_rule"].iloc[0],
                        "label_used": False,
                    },
                )

            chosen_k = int(tuning.loc[tuning["selected"], "k"].iloc[0])
            log(f"Selected genomic-cluster K={chosen_k} (label-independent).")

        # ------------------------------------------------------------------
        # Stage 4: population table + validation folds
        # ------------------------------------------------------------------
        stage_name = "population_and_outer_folds"
        with stage(stage_name):
            outputs_exist = cfg.population_csv.exists() and cfg.folds_csv.exists() and cfg.split_audit_csv.exists()
            if not cfg.force and stage_checkpoint_exists(cfg, 4, stage_name) and outputs_exist:
                population = pd.read_csv(cfg.population_csv, dtype={"Genome ID": str, "MLST": str})
                population["MLST"] = population["MLST"].replace({"nan": np.nan})
                folds = pd.read_csv(cfg.folds_csv, dtype={"Genome ID": str, "MLST": str})
                split_audit = pd.read_csv(cfg.split_audit_csv)
                if len(folds) != cfg.expected_n or len(population) != cfg.expected_n:
                    raise ValueError("Cached File07 split outputs have wrong row count.")
                log("SKIP stage 4: validated population/fold checkpoint found.")
            else:
                genomic_cluster = fit_selected_clusters(cfg, Z, tuning)

                population = cohort.copy()
                population["Genomic Cluster"] = genomic_cluster
                for i in range(min(10, Z.shape[1])):
                    population[f"SVD{i + 1}"] = Z[:, i]
                atomic_write_dataframe(population, cfg.population_csv, index=False)

                y = cohort["y"].to_numpy(dtype=np.uint8)
                random_fold = assign_random_folds(y, cfg.outer_folds, cfg.seed)
                cluster_fold = assign_group_folds(
                    y,
                    genomic_cluster.astype(str),
                    cfg.outer_folds,
                    cfg.seed + 1000,
                )

                mlst_coverage = float(cohort["MLST"].notna().mean())
                mlst_unique = int(cohort["MLST"].nunique(dropna=True))
                mlst_eligible = (
                    mlst_coverage >= cfg.mlst_min_coverage
                    and mlst_unique >= cfg.outer_folds
                )

                if mlst_eligible:
                    # Missing MLST samples get unique singleton groups rather than one
                    # shared "missing" group. This avoids forcing unrelated unknown
                    # lineages into the same fold. Coverage is reported transparently.
                    mlst_groups = np.array(
                        [
                            mlst if pd.notna(mlst) else f"MLST_MISSING__{gid}"
                            for gid, mlst in zip(cohort["Genome ID"], cohort["MLST"])
                        ],
                        dtype=object,
                    )
                    mlst_fold = assign_group_folds(
                        y,
                        mlst_groups,
                        cfg.outer_folds,
                        cfg.seed + 2000,
                    )
                else:
                    mlst_groups = np.array(
                        [
                            mlst if pd.notna(mlst) else f"MLST_MISSING__{gid}"
                            for gid, mlst in zip(cohort["Genome ID"], cohort["MLST"])
                        ],
                        dtype=object,
                    )
                    mlst_fold = np.full(len(cohort), -1, dtype=np.int16)
                    log(
                        "MLST-aware folds NOT activated: "
                        f"coverage={mlst_coverage:.1%}, unique non-missing STs={mlst_unique}. "
                        f"Required coverage >= {cfg.mlst_min_coverage:.0%}."
                    )

                folds = pd.DataFrame(
                    {
                        "Genome ID": cohort["Genome ID"].astype(str),
                        "Phenotype": cohort["Phenotype"].astype(str),
                        "y": y,
                        "MLST": cohort["MLST"],
                        "Genomic Cluster": genomic_cluster,
                        "random_fold": random_fold,
                        "genomic_cluster_fold": cluster_fold,
                        "mlst_fold": pd.Series(mlst_fold).replace(-1, pd.NA),
                    }
                )

                audit_rows: list[dict[str, Any]] = []
                audit_rows.extend(
                    audit_fold_scheme(
                        folds,
                        "random_stratified",
                        "random_fold",
                        None,
                        cfg.outer_folds,
                    )
                )
                audit_rows.extend(
                    audit_fold_scheme(
                        folds,
                        "genomic_cluster_aware",
                        "genomic_cluster_fold",
                        "Genomic Cluster",
                        cfg.outer_folds,
                    )
                )
                if mlst_eligible:
                    folds["MLST Group"] = mlst_groups
                    audit_rows.extend(
                        audit_fold_scheme(
                            folds,
                            "mlst_aware",
                            "mlst_fold",
                            "MLST Group",
                            cfg.outer_folds,
                        )
                    )
                    folds = folds.drop(columns=["MLST Group"])

                split_audit = pd.DataFrame(audit_rows)

                atomic_write_dataframe(folds, cfg.folds_csv, index=False)
                atomic_write_dataframe(split_audit, cfg.split_audit_csv, index=False)

                cluster_audit = build_cluster_audit(population)
                atomic_write_dataframe(cluster_audit, cfg.cluster_audit_csv, index=False)

                write_stage_checkpoint(
                    cfg,
                    4,
                    stage_name,
                    [cfg.population_csv, cfg.folds_csv, cfg.split_audit_csv, cfg.cluster_audit_csv],
                    {
                        "n_samples": len(folds),
                        "n_genomic_clusters": int(population["Genomic Cluster"].nunique()),
                        "mlst_coverage": mlst_coverage,
                        "mlst_unique_nonmissing": mlst_unique,
                        "mlst_aware_split_enabled": mlst_eligible,
                        "outer_folds": cfg.outer_folds,
                        "phenotype_used_for_clustering": False,
                        "phenotype_used_for_split_stratification": True,
                    },
                )

            log("Validation fold audit:")
            for scheme, sub in split_audit.groupby("scheme", sort=False):
                max_shift = float(sub["absolute_R_fraction_shift"].max())
                sizes = sub["n"].astype(int).tolist()
                log(f"  {scheme}: fold sizes={sizes}; max |R-fraction shift|={max_shift:.4f}")

        # ------------------------------------------------------------------
        # Stage 5: figures
        # ------------------------------------------------------------------
        figure_outputs: list[Path] = []
        if cfg.make_figures:
            stage_name = "figures"
            with stage(stage_name):
                expected_figures = [
                    cfg.figure_dir / "file07_population_structure_svd.png",
                    cfg.figure_dir / "file07_population_structure_svd.pdf",
                    cfg.figure_dir / "file07_validation_fold_balance.png",
                    cfg.figure_dir / "file07_validation_fold_balance.pdf",
                ]
                if (
                    not cfg.force
                    and stage_checkpoint_exists(cfg, 5, stage_name)
                    and all(p.exists() and p.stat().st_size > 0 for p in expected_figures)
                ):
                    figure_outputs = expected_figures
                    log("SKIP stage 5: figure checkpoint found.")
                else:
                    figure_outputs = make_figures(cfg, population, split_audit)
                    write_stage_checkpoint(
                        cfg,
                        5,
                        stage_name,
                        figure_outputs,
                        {"png_dpi": 300, "n_figures": len(figure_outputs)},
                    )
        else:
            log("Figures disabled by --no-figures.")

        # ------------------------------------------------------------------
        # Final summary / lock
        # ------------------------------------------------------------------
        stage_name = "final_summary"
        with stage(stage_name):
            selected_k = int(tuning.loc[tuning["selected"], "k"].iloc[0])
            mlst_coverage = float(cohort["MLST"].notna().mean())
            mlst_eligible = "mlst_aware" in set(split_audit["scheme"])

            summary = {
                "timestamp_utc": utc_now(),
                "script_version": SCRIPT_VERSION,
                "status": "PASS",
                "scientific_guardrails": {
                    "population_structure_features_used_as_predictive_X": False,
                    "phenotype_used_for_pangenome_svd": False,
                    "phenotype_used_for_genomic_clustering": False,
                    "phenotype_used_for_outer_fold_stratification": True,
                    "supervised_feature_selection_performed": False,
                },
                "cohort": {
                    "n": int(len(cohort)),
                    "R": int((cohort["Phenotype"] == "R").sum()),
                    "S": int((cohort["Phenotype"] == "S").sum()),
                },
                "population_structure": {
                    "representation": "BV-BRC PLFam presence/absence",
                    "svd_components": int(Z.shape[1]),
                    "svd_explained_variance_ratio_sum": float(explained.sum()),
                    "genomic_cluster_method": "KMeans on L2-normalized pangenome SVD embedding",
                    "selected_k": selected_k,
                    "cluster_selection": tuning.loc[tuning["selected"], "selection_rule"].iloc[0],
                    "interpretation": "pangenome gene-content structure proxy; not a phylogeny",
                },
                "mlst": {
                    "coverage": mlst_coverage,
                    "unique_nonmissing": int(cohort["MLST"].nunique(dropna=True)),
                    "mlst_aware_outer_cv_enabled": bool(mlst_eligible),
                    "minimum_coverage_required": cfg.mlst_min_coverage,
                },
                "validation_schemes": sorted(split_audit["scheme"].unique().tolist()),
                "outer_folds": cfg.outer_folds,
                "seed": cfg.seed,
                "outputs": {
                    "common_cohort": str(cfg.cohort_csv),
                    "svd_coordinates": str(cfg.svd_csv),
                    "cluster_tuning": str(cfg.cluster_tuning_csv),
                    "population_structure": str(cfg.population_csv),
                    "outer_folds": str(cfg.folds_csv),
                    "split_audit": str(cfg.split_audit_csv),
                    "cluster_audit": str(cfg.cluster_audit_csv),
                    "figures": [str(p) for p in figure_outputs],
                },
                "output_signatures": [
                    file_signature(p)
                    for p in [
                        cfg.cohort_csv,
                        cfg.svd_csv,
                        cfg.cluster_tuning_csv,
                        cfg.population_csv,
                        cfg.folds_csv,
                        cfg.split_audit_csv,
                        cfg.cluster_audit_csv,
                    ]
                    if p.exists()
                ],
            }
            atomic_write_json(cfg.summary_json, summary)
            write_stage_checkpoint(
                cfg,
                6,
                stage_name,
                [cfg.summary_json],
                {"status": "PASS"},
            )

        # Clear old failure marker only after successful finalization.
        failure_path = cfg.checkpoint_dir / "07_last_failure.json"
        if failure_path.exists():
            failure_path.unlink()

        print("\n" + "=" * 72)
        print("FILE07 — POPULATION STRUCTURE AND VALIDATION DESIGN")
        print("=" * 72)
        print(f"Cohort                         : {len(cohort):,}")
        print(f"Phenotype                      : R={(cohort['Phenotype'] == 'R').sum():,} / S={(cohort['Phenotype'] == 'S').sum():,}")
        print(f"Pangenome SVD dimensions       : {Z.shape[1]}")
        print(f"Selected genomic clusters K    : {int(tuning.loc[tuning['selected'], 'k'].iloc[0])}")
        print(f"MLST coverage                  : {cohort['MLST'].notna().mean():.2%}")
        print(f"MLST-aware CV enabled          : {'YES' if 'mlst_aware' in set(split_audit['scheme']) else 'NO'}")
        print(f"Outer CV folds                 : {cfg.outer_folds}")
        print(f"Validation schemes             : {', '.join(sorted(split_audit['scheme'].unique()))}")
        print("Phenotype used for clustering  : NO")
        print("Supervised feature selection   : NO")
        print("Final summary                  :", cfg.summary_json)
        print("FILE07 STATUS                  : PASS")
        print("=" * 72)

        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt as exc:
        # Resolve checkpoint dir even if argument parsing/config failed partially.
        try:
            root = Path.cwd()
            cp = root / "checkpoints/population_structure"
            write_crash_record(cp, kind="INTERRUPTED", detail=str(exc))
        finally:
            print("\nINTERRUPTED: File07 stopped safely. Completed stage checkpoints remain reusable.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        try:
            root = Path.cwd()
            cp = root / "checkpoints/population_structure"
            write_crash_record(cp, kind="CRASH", detail=traceback.format_exc())
        finally:
            print("\nFILE07 FAILED", file=sys.stderr)
            print(f"Stage : {CURRENT_STAGE}", file=sys.stderr)
            print(f"Error : {exc}", file=sys.stderr)
            print(
                "A crash record was written to checkpoints/population_structure/07_last_failure.json.\n"
                "Fix the cause and rerun the same command; completed validated stages will be skipped.",
                file=sys.stderr,
            )
        sys.exit(1)
