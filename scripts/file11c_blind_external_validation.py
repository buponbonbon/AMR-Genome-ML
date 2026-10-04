#!/usr/bin/env python3
"""
FILE11C — BLIND EXTERNAL VALIDATION (MEROPENEM)
==============================================

Purpose
-------
Freeze and externally validate the pre-specified meropenem resistance model
without using external labels for feature selection, hyperparameter tuning, or
threshold tuning.

Primary model (locked from File08 design)
-----------------------------------------
Representation : 668 known-AMR binary features
Estimator      : LogisticRegression(C=1.0, solver="liblinear")
Class weight   : None
Threshold      : estimator default / probability >= 0.5
Training set   : all 4,227 internal File07 genomes
Endpoint       : meropenem only

Known-AMR feature-generation bridge
-----------------------------------
The external feature caller reproduces the local branch of File04:
  AMRFinderPlus 4.2.7
  database 2026-08-07.1
  nucleotide mode
  organism Klebsiella_pneumoniae
  retain Type == AMR
  retain Subtype in {AMR, POINT, POINT_DISRUPT}
  feature name = Element symbol
  case-insensitive determinant key
  remove emrD
  align exactly to the frozen 668-column schema

Before external prediction, the script requires an exact bridge audit against
the original File04 local-AMRFinder training artifacts when they are present.
This demonstrates that the parser used for external data reproduces the locked
training feature matrix on the 382 locally screened development genomes.

Reviewer-facing safeguards
--------------------------
- verifies frozen E2 SHA256
- evaluates ONLY meropenem E2 records
- requires 525 unique meropenem assemblies and R=163/S=362 by default
- verifies zero exact-ID and SNP-cluster firewall reason fields
- freezes model specification before external labels are read for scoring
- never tunes threshold on E2
- no feature selection on E2
- no hyperparameter selection on E2
- exact AMRFinder software/database preflight
- exact 668-feature schema/order lock
- exact local-branch reconstruction audit
- final full-development model serialized before E2 scoring
- cluster bootstrap over external NCBI ERD/SNP groups
- optional MLST typing for seen-ST / unseen-ST descriptive robustness
- atomic outputs, resumable per-assembly AMRFinder checkpoints, process lock

Typical run
-----------
python scripts/file11c_blind_external_validation.py \
  --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML \
  --workers 6

Requirements
------------
Python: numpy, pandas, scikit-learn, joblib
Executables:
  datasets   (NCBI Datasets CLI; only needed if external FASTAs are absent)
  amrfinder  (must be 4.2.7 with DB 2026-08-07.1)
  mlst       (optional but recommended; expected 2.35.0)

Outputs
-------
data/external_validation/ncbi_pathogen_detection/file11c/
  file11c_external_meropenem_manifest.csv
  file11c_external_known_amr_668.csv
  file11c_blind_predictions.csv
  file11c_external_metrics.csv
  file11c_cluster_bootstrap_ci.csv
  file11c_mlst_subgroup_metrics.csv
  file11c_feature_prevalence_shift.csv
  file11c_bridge_audit.csv
  file11c_model_coefficients.csv
  file11c_analysis_design.json

checkpoints/external_validation_file11c/
  file11c_final_model.joblib
  file11c_model_lock.json
  file11c_final_summary.json
  amrfinder/*.tsv
  mlst/*.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

VERSION = "1.0.1"
RANDOM_SEED = 20260922

EXPECTED_E2_SHA256 = "181ad34e9087234b28ee06e72ed59861a1b199b699be45c5549206b4f0555c19"
EXPECTED_INTERNAL_N = 4227
EXPECTED_INTERNAL_R = 1601
EXPECTED_INTERNAL_S = 2626
EXPECTED_FEATURES = 668
EXPECTED_E2_MERO_N = 525
EXPECTED_E2_MERO_R = 163
EXPECTED_E2_MERO_S = 362

AMRFINDER_VERSION = "4.2.7"
AMRFINDER_DB_VERSION = "2026-08-07.1"
AMRFINDER_ORGANISM = "Klebsiella_pneumoniae"
ALLOWED_SUBTYPES = {"AMR", "POINT", "POINT_DISRUPT"}

STOP = False


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
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


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.csv", dir=path.parent)
    os.close(fd)
    p = Path(tmp)
    try:
        df.to_csv(p, index=False)
        with p.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(p, path)
    except Exception:
        p.unlink(missing_ok=True)
        raise


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            pid = int(old.get("pid", -1))
        except Exception:
            pid = -1
        if pid > 0 and Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"Active FILE11C lock exists: {path} (PID {pid})")
        path.unlink(missing_ok=True)
    atomic_json(path, {"pid": os.getpid(), "started_utc": utc_now(), "version": VERSION})


def signal_handler(_signum, _frame):
    global STOP
    STOP = True


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def find_exe(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return str(p.resolve())
    q = shutil.which(name_or_path)
    if not q:
        raise FileNotFoundError(f"Executable not found: {name_or_path}")
    return q


def norm_gid(x: Any) -> str:
    return str(x).strip()


def norm_feature_key(x: Any) -> str:
    return str(x).strip().lower()


# =============================================================================
# Input loading and locks
# =============================================================================

def load_e2(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_hash = sha256_file(path)
    if observed_hash != EXPECTED_E2_SHA256:
        raise RuntimeError(
            "Frozen E2 SHA256 mismatch.\n"
            f"Expected: {EXPECTED_E2_SHA256}\n"
            f"Observed: {observed_hash}"
        )

    df = pd.read_csv(path, dtype=str)
    required = {
        "target_acc", "antibiotic", "phenotype", "asm_acc", "erd_group",
        "label_binary", "exact_overlap_reason", "cluster_firewall_reason",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"E2 schema missing columns: {sorted(missing)}")

    mero = df[df["antibiotic"].astype(str).str.lower().eq("meropenem")].copy()
    mero["label_binary"] = pd.to_numeric(mero["label_binary"], errors="raise").astype(int)

    if len(mero) != EXPECTED_E2_MERO_N:
        raise RuntimeError(f"Expected {EXPECTED_E2_MERO_N} meropenem rows, found {len(mero)}")
    if mero["asm_acc"].nunique() != EXPECTED_E2_MERO_N:
        raise RuntimeError("Meropenem E2 must contain one unique assembly per row.")
    if int(mero["label_binary"].sum()) != EXPECTED_E2_MERO_R:
        raise RuntimeError("Unexpected meropenem resistant count.")
    if int((1 - mero["label_binary"]).sum()) != EXPECTED_E2_MERO_S:
        raise RuntimeError("Unexpected meropenem susceptible count.")
    if mero["asm_acc"].isna().any() or mero["asm_acc"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Meropenem E2 contains missing assembly accessions.")
    if mero["erd_group"].isna().any() or mero["erd_group"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Strict E2 contains missing erd_group values.")

    # Firewall fields must be empty on rows that survived.
    for col in ("exact_overlap_reason", "cluster_firewall_reason"):
        bad = mero[col].fillna("").astype(str).str.strip().ne("")
        if bad.any():
            raise RuntimeError(f"Strict E2 survivor unexpectedly has {col}: {int(bad.sum())} rows")

    return df, mero


def load_internal(
    folds_path: Path,
    known_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    folds = pd.read_csv(folds_path, dtype={"Genome ID": str})
    known = pd.read_csv(known_path, dtype={"Genome ID": str})

    if len(folds) != EXPECTED_INTERNAL_N or folds["Genome ID"].nunique() != EXPECTED_INTERNAL_N:
        raise RuntimeError("Internal File07 cohort size/uniqueness mismatch.")
    folds["y"] = pd.to_numeric(folds["y"], errors="raise").astype(int)
    if int(folds["y"].sum()) != EXPECTED_INTERNAL_R:
        raise RuntimeError("Internal resistant count mismatch.")
    if int((1 - folds["y"]).sum()) != EXPECTED_INTERNAL_S:
        raise RuntimeError("Internal susceptible count mismatch.")

    features = [c for c in known.columns if c != "Genome ID"]
    if len(features) != EXPECTED_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_FEATURES} known-AMR features, found {len(features)}")
    if any(norm_feature_key(c) == "emrd" for c in features):
        raise RuntimeError("emrD unexpectedly present in final known-AMR schema.")
    if len({norm_feature_key(c) for c in features}) != len(features):
        raise RuntimeError("Case-insensitive duplicate known-AMR feature names detected.")

    known["Genome ID"] = known["Genome ID"].map(norm_gid)
    folds["Genome ID"] = folds["Genome ID"].map(norm_gid)
    idx = known.set_index("Genome ID")
    missing = [g for g in folds["Genome ID"] if g not in idx.index]
    if missing:
        raise RuntimeError(f"Known-AMR matrix missing internal cohort genomes: {missing[:10]}")

    X = idx.loc[folds["Genome ID"], features].to_numpy(dtype=np.uint8, copy=True)
    if not np.all(np.isin(np.unique(X), [0, 1])):
        raise RuntimeError("Internal known-AMR matrix is not binary.")

    y = folds["y"].to_numpy(dtype=np.uint8)
    return folds, known, X, y, features


# =============================================================================
# AMRFinder preflight and parsing
# =============================================================================

def amrfinder_env(exe: str) -> dict[str, str]:
    env = os.environ.copy()
    bindir = str(Path(exe).resolve().parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env.setdefault("CONDA_PREFIX", str(Path(bindir).parent))
    return env


def verify_amrfinder(exe: str, database: Path) -> dict[str, str]:
    env = amrfinder_env(exe)
    if not database.is_dir():
        raise FileNotFoundError(f"AMRFinder database directory not found: {database}")

    p = subprocess.run(
        [exe, "--version"], capture_output=True, text=True, env=env, check=True
    )
    text = (p.stdout + "\n" + p.stderr).strip()
    m = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
    if not m or m.group(1) != AMRFINDER_VERSION:
        raise RuntimeError(
            f"AMRFinderPlus version mismatch; expected {AMRFINDER_VERSION}, output={text!r}"
        )

    p = subprocess.run(
        [exe, "--database", str(database), "--database_version"],
        capture_output=True, text=True, env=env, check=True
    )
    text_db = p.stdout + "\n" + p.stderr
    m = re.search(r"Database version:\s*([^\s]+)", text_db)
    if not m or m.group(1) != AMRFINDER_DB_VERSION:
        raise RuntimeError(
            f"AMRFinderPlus DB mismatch; expected {AMRFINDER_DB_VERSION}, output={text_db!r}"
        )

    return {
        "amrfinder_executable": exe,
        "amrfinder_version": AMRFINDER_VERSION,
        "database_version": AMRFINDER_DB_VERSION,
        "organism": AMRFINDER_ORGANISM,
        "mode": "nucleotide",
        "database_path": str(database),
    }


def parse_amrfinder_tsv(path: Path) -> set[str]:
    df = pd.read_csv(path, sep="\t")
    required = {"Element symbol", "Type", "Subtype"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Unexpected AMRFinder schema in {path}; columns={list(df.columns)}")

    keep = (
        df["Type"].astype(str).str.upper().eq("AMR")
        & df["Subtype"].astype(str).str.upper().isin(ALLOWED_SUBTYPES)
    )
    out: set[str] = set()
    for x in df.loc[keep, "Element symbol"]:
        if pd.isna(x):
            continue
        s = str(x).strip()
        if s:
            out.add(norm_feature_key(s))
    return out


def feature_vector_from_keys(keys: set[str], features: list[str]) -> np.ndarray:
    schema_keys = [norm_feature_key(x) for x in features]
    return np.asarray([1 if k in keys else 0 for k in schema_keys], dtype=np.uint8)


# =============================================================================
# File04 bridge audit
# =============================================================================

def run_bridge_audit(
    root: Path,
    known: pd.DataFrame,
    features: list[str],
    output_path: Path,
) -> dict[str, Any]:
    manifest = root / "data/features/known_amr/amrfinder_local_completion_manifest.csv"
    raw_dir = root / "data/features/known_amr/raw_amrfinder"

    require_file(manifest, "File04 local AMRFinder completion manifest")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing File04 raw AMRFinder directory: {raw_dir}")

    m = pd.read_csv(manifest, dtype={"Genome ID": str})
    if "Genome ID" not in m.columns:
        raise RuntimeError("Local completion manifest lacks Genome ID.")
    m["Genome ID"] = m["Genome ID"].map(norm_gid)
    if len(m) != 382 or m["Genome ID"].nunique() != 382:
        raise RuntimeError(
            f"Expected exactly 382 File04 local-screened genomes, found {len(m)} rows / "
            f"{m['Genome ID'].nunique()} unique IDs."
        )

    known_idx = known.set_index("Genome ID")
    rows = []
    exact_rows = 0
    total_mismatch_cells = 0

    for i, gid in enumerate(m["Genome ID"], start=1):
        tsv = raw_dir / f"{gid}.amrfinder.tsv"
        require_file(tsv, f"raw AMRFinder output for {gid}")
        keys = parse_amrfinder_tsv(tsv)
        keys.discard("emrd")
        reconstructed = feature_vector_from_keys(keys, features)
        expected = known_idx.loc[gid, features].to_numpy(dtype=np.uint8)
        mismatches = int(np.sum(reconstructed != expected))
        if mismatches == 0:
            exact_rows += 1
        total_mismatch_cells += mismatches
        rows.append(
            {
                "Genome ID": gid,
                "expected_present": int(expected.sum()),
                "reconstructed_present": int(reconstructed.sum()),
                "mismatched_cells": mismatches,
                "exact_row_match": mismatches == 0,
            }
        )
        if i == 1 or i % 25 == 0 or i == len(m):
            log(
                f"[PROGRESS] bridge audit {i}/{len(m)} | "
                f"exact_rows={exact_rows} | mismatched_cells={total_mismatch_cells}"
            )

    audit = pd.DataFrame(rows)
    atomic_csv(output_path, audit)

    if exact_rows != 382 or total_mismatch_cells != 0:
        raise RuntimeError(
            "FILE11C bridge audit FAILED: external AMRFinder parser does not exactly "
            "reproduce the locked File04 known-AMR matrix for all 382 locally screened "
            f"development genomes. exact_rows={exact_rows}/382, mismatched_cells={total_mismatch_cells}"
        )

    return {
        "local_training_genomes_audited": 382,
        "exact_row_matches": exact_rows,
        "mismatched_cells": total_mismatch_cells,
        "status": "PASS_EXACT",
    }


# =============================================================================
# Final model lock
# =============================================================================

def train_and_lock_model(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    model_path: Path,
    lock_path: Path,
    coef_path: Path,
    source_paths: dict[str, Path],
) -> tuple[LogisticRegression, dict[str, Any]]:
    spec = {
        "representation": "known_amr",
        "feature_count": EXPECTED_FEATURES,
        "estimator": "sklearn.linear_model.LogisticRegression",
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": None,
        "max_iter": 3000,
        "random_state": 20260920,
        "decision_threshold": 0.5,
        "threshold_source": "fixed estimator default; never tuned on E2",
        "selection_source": "pre-specified File08 primary known-AMR baseline",
        "external_endpoint": "meropenem",
        "external_labels_used_for_model_or_feature_selection": False,
    }

    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight=None,
        max_iter=3000,
        random_state=20260920,
    )
    model.fit(np.asarray(X, dtype=np.float32), y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp.joblib")
    joblib.dump(
        {
            "model": model,
            "features": features,
            "specification": spec,
        },
        tmp,
    )
    os.replace(tmp, model_path)

    coefs = pd.DataFrame(
        {
            "feature": features,
            "coefficient": model.coef_[0].astype(float),
            "abs_coefficient": np.abs(model.coef_[0].astype(float)),
        }
    ).sort_values("abs_coefficient", ascending=False)
    atomic_csv(coef_path, coefs)

    feature_schema_text = "\n".join(features) + "\n"
    lock = {
        "locked_utc": utc_now(),
        "script_version": VERSION,
        "specification": spec,
        "feature_schema_sha256": sha256_text(feature_schema_text),
        "model_artifact_sha256": sha256_file(model_path),
        "training_X_shape": list(X.shape),
        "training_y_R": int(y.sum()),
        "training_y_S": int((1 - y).sum()),
        "source_hashes": {
            k: sha256_file(v) for k, v in source_paths.items()
        },
    }
    atomic_json(lock_path, lock)
    return model, lock


# =============================================================================
# External assembly acquisition
# =============================================================================

def discover_external_fastas(fasta_root: Path, accessions: list[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not fasta_root.exists():
        return found

    # Prefer NCBI datasets directory names, but allow a flat user-provided folder.
    all_fastas = list(fasta_root.rglob("*.fna")) + list(fasta_root.rglob("*.fa")) + list(fasta_root.rglob("*.fasta"))
    for acc in accessions:
        candidates = [
            p for p in all_fastas
            if acc in str(p) or acc.split(".")[0] in str(p)
        ]
        if candidates:
            candidates.sort(key=lambda p: (0 if "genomic" in p.name else 1, len(str(p))))
            found[acc] = candidates[0]
    return found


def download_external_fastas(
    datasets_exe: str,
    fasta_root: Path,
    accessions: list[str],
    checkpoint_dir: Path,
) -> dict[str, Path]:
    found = discover_external_fastas(fasta_root, accessions)
    missing = [a for a in accessions if a not in found]
    if not missing:
        return found

    log(f"External FASTAs missing: {len(missing)}; invoking NCBI Datasets CLI.")
    fasta_root.mkdir(parents=True, exist_ok=True)
    acc_file = checkpoint_dir / "file11c_external_assembly_accessions.txt"
    atomic_text(acc_file, "\n".join(missing) + "\n")
    zip_path = checkpoint_dir / "file11c_ncbi_datasets.zip"

    cmd = [
        datasets_exe,
        "download", "genome", "accession",
        "--inputfile", str(acc_file),
        "--include", "genome",
        "--filename", str(zip_path),
    ]
    log("Datasets command: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    extract_dir = fasta_root / "ncbi_datasets_download"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    found = discover_external_fastas(fasta_root, accessions)
    missing = [a for a in accessions if a not in found]
    if missing:
        raise RuntimeError(f"Could not locate downloaded FASTAs for {len(missing)} accessions; first={missing[:10]}")
    return found


# =============================================================================
# External AMRFinder screening
# =============================================================================

def run_one_amrfinder(
    exe: str,
    database: Path,
    fasta: Path,
    output: Path,
) -> tuple[bool, str]:
    if output.is_file() and output.stat().st_size > 0:
        try:
            parse_amrfinder_tsv(output)
            return True, "SKIP"
        except Exception:
            output.unlink(missing_ok=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp.tsv")
    env = amrfinder_env(exe)
    cmd = [
        exe,
        "-n", str(fasta),
        "-O", AMRFINDER_ORGANISM,
        "--database", str(database),
        "--threads", "1",
        "-o", str(tmp),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False, (p.stdout + "\n" + p.stderr)[-4000:]
    os.replace(tmp, output)
    parse_amrfinder_tsv(output)
    return True, "PASS"


def screen_external(
    exe: str,
    database: Path,
    manifest: pd.DataFrame,
    output_dir: Path,
    workers: int,
) -> None:
    jobs = []
    for row in manifest.itertuples(index=False):
        acc = row.asm_acc
        fasta = Path(row.fasta_path)
        out = output_dir / f"{acc}.amrfinder.tsv"
        jobs.append((acc, fasta, out))

    done = 0
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(run_one_amrfinder, exe, database, fasta, out): (acc, out)
            for acc, fasta, out in jobs
        }
        for fut in as_completed(futs):
            if STOP:
                raise KeyboardInterrupt
            acc, out = futs[fut]
            try:
                ok, detail = fut.result()
            except Exception as exc:
                ok, detail = False, repr(exc)
            done += 1
            if not ok:
                failures.append((acc, detail))
            if done == 1 or done % 10 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9)
                log(
                    f"[PROGRESS] external AMRFinder {done}/{len(jobs)} "
                    f"({done/len(jobs):.1%}) | failures={len(failures)} | ETA={human_seconds(eta)}"
                )

    if failures:
        detail = "\n".join(f"{a}: {d}" for a, d in failures[:10])
        raise RuntimeError(f"External AMRFinder failed for {len(failures)} assemblies.\n{detail}")


def build_external_matrix(
    manifest: pd.DataFrame,
    amr_dir: Path,
    features: list[str],
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = {norm_feature_key(f): f for f in features}
    rows = []
    unknown_counts: dict[str, int] = {}

    for row in manifest.itertuples(index=False):
        acc = row.asm_acc
        keys = parse_amrfinder_tsv(amr_dir / f"{acc}.amrfinder.tsv")
        keys.discard("emrd")

        for k in keys:
            if k not in schema:
                unknown_counts[k] = unknown_counts.get(k, 0) + 1

        vec = feature_vector_from_keys(keys, features)
        record = {"asm_acc": acc}
        record.update({f: int(v) for f, v in zip(features, vec)})
        rows.append(record)

    X = pd.DataFrame(rows)
    if X.shape != (len(manifest), EXPECTED_FEATURES + 1):
        raise RuntimeError(f"External feature matrix shape mismatch: {X.shape}")
    if X["asm_acc"].nunique() != len(manifest):
        raise RuntimeError("Duplicate assembly accessions in external feature matrix.")
    if X[features].isna().any().any():
        raise RuntimeError("Missing external feature values.")
    if not np.all(np.isin(np.unique(X[features].to_numpy()), [0, 1])):
        raise RuntimeError("External feature matrix is not binary.")

    atomic_csv(output_path, X)

    unknown = pd.DataFrame(
        [
            {"determinant_key": k, "external_genomes_present": n}
            for k, n in sorted(unknown_counts.items(), key=lambda x: (-x[1], x[0]))
        ]
    )
    return X, unknown


# =============================================================================
# Optional external MLST
# =============================================================================

def verify_mlst(exe: str) -> dict[str, str]:
    p = subprocess.run([exe, "--version"], capture_output=True, text=True)
    txt = p.stdout + "\n" + p.stderr
    if p.returncode != 0:
        raise RuntimeError(f"mlst --version failed: {txt}")
    return {"mlst_executable": exe, "mlst_version_output": txt.strip()}


def parse_mlst_line(line: str) -> str | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    scheme = parts[1].strip()
    st = parts[2].strip()
    if scheme != "klebsiella":
        return None
    if re.fullmatch(r"\d+", st):
        return f"ST{int(st)}"
    return None


def type_external_mlst(
    exe: str,
    manifest: pd.DataFrame,
    ckpt_dir: Path,
    workers: int,
) -> pd.DataFrame:
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def one(acc: str, fasta: Path) -> tuple[str, str | None, str]:
        out = ckpt_dir / f"{acc}.tsv"
        if out.exists() and out.stat().st_size > 0:
            text = out.read_text(encoding="utf-8", errors="replace")
            first = next((x for x in text.splitlines() if x.strip()), "")
            return acc, parse_mlst_line(first), "SKIP"
        p = subprocess.run(
            [exe, "--scheme", "klebsiella", str(fasta)],
            capture_output=True, text=True, timeout=600,
        )
        if p.returncode != 0:
            return acc, None, "FAIL"
        atomic_text(out, p.stdout)
        first = next((x for x in p.stdout.splitlines() if x.strip()), "")
        return acc, parse_mlst_line(first), "PASS"

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(one, r.asm_acc, Path(r.fasta_path)): r.asm_acc
            for r in manifest.itertuples(index=False)
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            acc, st, status = fut.result()
            rows.append({"asm_acc": acc, "external_MLST": st, "mlst_status": status})
            if done == 1 or done % 25 == 0 or done == len(futs):
                log(f"[PROGRESS] external MLST {done}/{len(futs)}")
    return pd.DataFrame(rows)


# =============================================================================
# Metrics and cluster bootstrap
# =============================================================================

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def metric_bundle(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=np.uint8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return {
        "n": int(len(y)),
        "R": int(y.sum()),
        "S": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


BOOT_METRICS = (
    "roc_auc", "average_precision", "balanced_accuracy",
    "sensitivity", "specificity", "mcc", "brier",
)


def cluster_bootstrap(
    pred_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    groups = pred_df["erd_group"].astype(str).to_numpy()
    unique = np.unique(groups)
    by_group = {
        g: np.flatnonzero(groups == g)
        for g in unique
    }
    rng = np.random.default_rng(seed)
    values = {m: [] for m in BOOT_METRICS}
    valid = 0

    for b in range(n_boot):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled_groups])
        y = pred_df["label_binary"].to_numpy(dtype=np.uint8)[idx]
        if len(np.unique(y)) < 2:
            continue
        p = pred_df["probability_R"].to_numpy(dtype=float)[idx]
        pr = pred_df["predicted_R"].to_numpy(dtype=np.uint8)[idx]
        m = metric_bundle(y, p, pr)
        for metric in BOOT_METRICS:
            values[metric].append(m[metric])
        valid += 1

    if valid < max(500, int(0.8 * n_boot)):
        raise RuntimeError(
            f"Too few valid ERD-group bootstrap draws: {valid}/{n_boot}"
        )

    rows = []
    for metric, vals in values.items():
        a = np.asarray(vals, dtype=float)
        rows.append(
            {
                "bootstrap_unit": "NCBI_ERD_SNP_cluster",
                "metric": metric,
                "n_boot_requested": n_boot,
                "n_boot_valid": len(a),
                "bootstrap_mean": float(np.mean(a)),
                "ci95_low": float(np.quantile(a, 0.025)),
                "ci95_high": float(np.quantile(a, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, g in pred.groupby(group_col, dropna=False):
        if len(g) < 10 or g["label_binary"].nunique() < 2:
            continue
        m = metric_bundle(
            g["label_binary"].to_numpy(dtype=np.uint8),
            g["probability_R"].to_numpy(dtype=float),
            g["predicted_R"].to_numpy(dtype=np.uint8),
        )
        m[group_col] = group
        rows.append(m)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="FILE11C blind external validation for meropenem resistance."
    )
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--amrfinder", default="amrfinder")
    ap.add_argument(
        "--amrfinder-database",
        type=Path,
        default=None,
        help=(
            "Path to the exact AMRFinderPlus database directory. Required for "
            "reproducible external validation; must report database version "
            f"{AMRFINDER_DB_VERSION}."
        ),
    )
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--mlst", default="mlst")
    ap.add_argument("--skip-mlst", action="store_true")
    ap.add_argument(
        "--external-fasta-dir",
        type=Path,
        default=None,
        help="Existing external assembly directory. If omitted, FILE11C uses its managed download directory.",
    )
    args = ap.parse_args()

    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be 1..16")
    if args.bootstrap_replicates < 1000:
        raise SystemExit("--bootstrap-replicates must be >=1000")

    root = args.project_root.resolve()
    out_dir = root / "data/external_validation/ncbi_pathogen_detection/file11c"
    ckpt = root / "checkpoints/external_validation_file11c"
    amr_ckpt = ckpt / "amrfinder"
    mlst_ckpt = ckpt / "mlst"
    lock_file = ckpt / "file11c.lock"
    failure = ckpt / "file11c_last_failure.json"

    paths = {
        "e2": root / "data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv",
        "folds": root / "data/splits/file07_outer_folds.csv",
        "known": root / "data/features/known_amr/X_known_amr_final.csv",
        "known_meta": root / "data/features/known_amr/known_amr_feature_metadata_final.csv",
        "harmonization": root / "data/features/known_amr/known_amr_final_harmonization_decision.csv",
        "integrity": root / "data/features/known_amr/known_amr_final_integrity_audit.csv",
        "file08_summary": root / "checkpoints/model_training/file08_final_summary.json",
        "file09_summary": root / "checkpoints/model_evaluation/file09_final_summary.json",
        "file09b_summary": root / "checkpoints/model_evaluation_reviewer_defense/file09b_final_summary.json",
        "file09c_summary": root / "checkpoints/model_evaluation_reviewer_defense_nonlinear_v102/file09c_final_summary.json",
    }

    outputs = {
        "manifest": out_dir / "file11c_external_meropenem_manifest.csv",
        "X_external": out_dir / "file11c_external_known_amr_668.csv",
        "predictions": out_dir / "file11c_blind_predictions.csv",
        "metrics": out_dir / "file11c_external_metrics.csv",
        "bootstrap": out_dir / "file11c_cluster_bootstrap_ci.csv",
        "mlst_subgroups": out_dir / "file11c_mlst_subgroup_metrics.csv",
        "shift": out_dir / "file11c_feature_prevalence_shift.csv",
        "bridge": out_dir / "file11c_bridge_audit.csv",
        "coefficients": out_dir / "file11c_model_coefficients.csv",
        "unknown_features": out_dir / "file11c_external_out_of_schema_determinants.csv",
        "design": out_dir / "file11c_analysis_design.json",
    }

    model_path = ckpt / "file11c_final_model.joblib"
    model_lock_path = ckpt / "file11c_model_lock.json"
    final_summary = ckpt / "file11c_final_summary.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    acquire_lock(lock_file)

    stage = "startup"
    started = time.time()

    try:
        log("=" * 104)
        log("FILE11C — BLIND EXTERNAL VALIDATION")
        log("=" * 104)
        log(f"Version       : {VERSION}")
        log(f"Project root  : {root}")
        log(f"Workers       : {args.workers}")
        log(f"Bootstrap     : {args.bootstrap_replicates}")

        for label, path in paths.items():
            require_file(path, label)
            log(f"Input OK      : {path.relative_to(root)}")

        # ---------------------------------------------------------------------
        stage = "stage1_frozen_input_audit"
        log("Stage 1/10 — audit frozen E2 and locked internal artifacts")
        e2_all, e2 = load_e2(paths["e2"])
        folds, known, X_internal, y_internal, features = load_internal(
            paths["folds"], paths["known"]
        )

        meta = pd.read_csv(paths["known_meta"])
        if "Feature" not in meta.columns or set(meta["Feature"].astype(str)) != set(features):
            raise RuntimeError("known_amr_feature_metadata_final.csv does not match the 668-feature schema.")

        for sname in ("file08_summary", "file09_summary", "file09b_summary", "file09c_summary"):
            with paths[sname].open("r", encoding="utf-8") as fh:
                s = json.load(fh)
            if s.get("status") != "PASS" or s.get("mode") != "FULL":
                raise RuntimeError(f"{sname} is not PASS/FULL.")

        log(
            f"[PROGRESS] frozen E2 PASS | total={len(e2_all)} | "
            f"meropenem={len(e2)} R={int(e2['label_binary'].sum())} "
            f"S={int((1-e2['label_binary']).sum())}"
        )

        design = {
            "script_version": VERSION,
            "created_utc": utc_now(),
            "primary_endpoint": "meropenem",
            "development_cohort_n": EXPECTED_INTERNAL_N,
            "external_strict_E2_n": EXPECTED_E2_MERO_N,
            "external_firewall": [
                "exact identifier overlap removed",
                "same NCBI ERD/SNP cluster as internal removed",
                "external rows lacking ERD/SNP cluster excluded",
            ],
            "model": {
                "representation": "known_amr_668",
                "estimator": "LogisticRegression",
                "C": 1.0,
                "solver": "liblinear",
                "class_weight": None,
                "threshold": 0.5,
            },
            "external_model_selection": False,
            "external_feature_selection": False,
            "external_threshold_tuning": False,
            "amrfinder": {
                "version": AMRFINDER_VERSION,
                "database": AMRFINDER_DB_VERSION,
                "organism": AMRFINDER_ORGANISM,
                "mode": "nucleotide",
                "allowed_subtypes": sorted(ALLOWED_SUBTYPES),
                "excluded_feature": "emrD",
            },
            "primary_uncertainty": "NCBI ERD/SNP-cluster bootstrap",
            "bootstrap_replicates": args.bootstrap_replicates,
        }
        atomic_json(outputs["design"], design)

        # ---------------------------------------------------------------------
        stage = "stage2_amrfinder_preflight"
        log("Stage 2/10 — exact AMRFinderPlus runtime preflight")
        amrfinder = find_exe(args.amrfinder)
        if args.amrfinder_database is None:
            raise RuntimeError(
                "--amrfinder-database is required in FILE11C v1.0.1 so the exact "
                f"database {AMRFINDER_DB_VERSION} is explicit rather than inferred from a latest symlink."
            )
        amrfinder_database = args.amrfinder_database.expanduser().resolve()
        runtime = verify_amrfinder(amrfinder, amrfinder_database)
        log(
            f"[PROGRESS] AMRFinder PASS | version={runtime['amrfinder_version']} | "
            f"DB={runtime['database_version']}"
        )

        # ---------------------------------------------------------------------
        stage = "stage3_bridge_audit"
        log("Stage 3/10 — exact File04 local-branch reconstruction audit")
        bridge = run_bridge_audit(root, known, features, outputs["bridge"])
        log(
            f"[PROGRESS] bridge PASS | exact rows={bridge['exact_row_matches']}/382 | "
            f"mismatched cells={bridge['mismatched_cells']}"
        )

        # ---------------------------------------------------------------------
        stage = "stage4_final_model_lock"
        log("Stage 4/10 — train and freeze final 4,227-genome internal model")
        model, model_lock = train_and_lock_model(
            X_internal,
            y_internal,
            features,
            model_path,
            model_lock_path,
            outputs["coefficients"],
            {
                "folds": paths["folds"],
                "known": paths["known"],
                "e2": paths["e2"],
                "file08_summary": paths["file08_summary"],
                "file09_summary": paths["file09_summary"],
                "file09b_summary": paths["file09b_summary"],
                "file09c_summary": paths["file09c_summary"],
            },
        )
        log(
            f"[PROGRESS] model frozen | model_sha256={model_lock['model_artifact_sha256']}"
        )

        # ---------------------------------------------------------------------
        stage = "stage5_external_assembly_acquisition"
        log("Stage 5/10 — acquire/verify 525 external meropenem assemblies")
        fasta_root = (
            args.external_fasta_dir.resolve()
            if args.external_fasta_dir
            else root / "data/external_validation/ncbi_pathogen_detection/file11c/assemblies"
        )
        accessions = e2["asm_acc"].astype(str).tolist()
        found = discover_external_fastas(fasta_root, accessions)
        if len(found) != len(accessions):
            datasets = find_exe(args.datasets)
            found = download_external_fastas(datasets, fasta_root, accessions, ckpt)

        manifest = e2[
            [
                "target_acc", "asm_acc", "biosample_acc", "erd_group",
                "collection_date", "geo_loc_name", "host", "isolation_source",
                "phenotype", "label_binary",
            ]
        ].copy()
        manifest["fasta_path"] = manifest["asm_acc"].map(lambda a: str(found[str(a)]))
        manifest["fasta_sha256"] = manifest["fasta_path"].map(lambda p: sha256_file(Path(p)))
        atomic_csv(outputs["manifest"], manifest)
        log(f"[PROGRESS] external assemblies PASS | {len(manifest)}/{len(manifest)}")

        # ---------------------------------------------------------------------
        stage = "stage6_external_amrfinder"
        log("Stage 6/10 — external AMRFinderPlus screening")
        screen_external(amrfinder, amrfinder_database, manifest, amr_ckpt, args.workers)
        log("[PROGRESS] external AMRFinder PASS")

        # ---------------------------------------------------------------------
        stage = "stage7_external_feature_matrix"
        log("Stage 7/10 — construct exact 668-column external feature matrix")
        X_ext_df, unknown = build_external_matrix(
            manifest, amr_ckpt, features, outputs["X_external"]
        )
        atomic_csv(outputs["unknown_features"], unknown)
        X_ext = X_ext_df[features].to_numpy(dtype=np.uint8)
        log(
            f"[PROGRESS] external X PASS | shape={X_ext.shape} | "
            f"out-of-schema determinants={len(unknown)}"
        )

        # Feature prevalence shift is descriptive only.
        internal_prev = X_internal.mean(axis=0)
        external_prev = X_ext.mean(axis=0)
        shift = pd.DataFrame(
            {
                "feature": features,
                "internal_prevalence": internal_prev,
                "external_prevalence": external_prev,
                "absolute_prevalence_shift": np.abs(external_prev - internal_prev),
            }
        ).sort_values("absolute_prevalence_shift", ascending=False)
        atomic_csv(outputs["shift"], shift)

        # ---------------------------------------------------------------------
        stage = "stage8_optional_mlst"
        log("Stage 8/10 — external MLST seen/unseen lineage audit")
        if args.skip_mlst:
            mlst_df = pd.DataFrame(
                {"asm_acc": manifest["asm_acc"], "external_MLST": pd.NA, "mlst_status": "SKIPPED"}
            )
            mlst_runtime = {"status": "SKIPPED"}
        else:
            mlst_exe = find_exe(args.mlst)
            mlst_runtime = verify_mlst(mlst_exe)
            mlst_df = type_external_mlst(mlst_exe, manifest, mlst_ckpt, args.workers)

        training_st = set(
            folds["MLST"].dropna().astype(str).str.strip()
        )
        # File07 may store ST as ST11 or raw numeric depending source; normalize both.
        training_st_norm = set()
        for st in training_st:
            m = re.fullmatch(r"(?:ST)?(\d+)(?:\.0)?", st, flags=re.I)
            if m:
                training_st_norm.add(f"ST{int(m.group(1))}")
        mlst_df["lineage_novelty"] = mlst_df["external_MLST"].map(
            lambda st: (
                "unseen_ST" if pd.notna(st) and st not in training_st_norm
                else ("seen_ST" if pd.notna(st) else "unresolved_ST")
            )
        )

        # ---------------------------------------------------------------------
        stage = "stage9_blind_prediction"
        log("Stage 9/10 — BLIND E2 prediction (no tuning)")
        # Important: model prediction happens before metrics are computed.
        p = model.predict_proba(np.asarray(X_ext, dtype=np.float32))[:, 1]
        predv = (p >= 0.5).astype(np.uint8)

        predictions = manifest.copy()
        predictions["probability_R"] = p
        predictions["predicted_R"] = predv
        predictions = predictions.merge(
            mlst_df[["asm_acc", "external_MLST", "lineage_novelty"]],
            on="asm_acc", how="left", validate="one_to_one"
        )
        atomic_csv(outputs["predictions"], predictions)
        prediction_hash = sha256_file(outputs["predictions"])
        log(f"[PROGRESS] blind predictions frozen | sha256={prediction_hash}")

        # ---------------------------------------------------------------------
        stage = "stage10_metrics"
        log("Stage 10/10 — external metrics, ERD-cluster bootstrap CI, lineage robustness")
        y_ext = predictions["label_binary"].to_numpy(dtype=np.uint8)
        primary = metric_bundle(y_ext, p, predv)
        primary_df = pd.DataFrame([{"endpoint": "meropenem", **primary}])
        atomic_csv(outputs["metrics"], primary_df)

        boot = cluster_bootstrap(predictions, args.bootstrap_replicates, RANDOM_SEED)
        atomic_csv(outputs["bootstrap"], boot)

        lineage = subgroup_metrics(predictions, "lineage_novelty")
        atomic_csv(outputs["mlst_subgroups"], lineage)

        summary = {
            "script_version": VERSION,
            "status": "PASS_BLIND_EXTERNAL_VALIDATION",
            "completed_utc": utc_now(),
            "elapsed_seconds": time.time() - started,
            "frozen_e2_sha256": sha256_file(paths["e2"]),
            "frozen_model_sha256": sha256_file(model_path),
            "frozen_prediction_sha256": prediction_hash,
            "development": {
                "n": EXPECTED_INTERNAL_N,
                "R": EXPECTED_INTERNAL_R,
                "S": EXPECTED_INTERNAL_S,
            },
            "external_meropenem_E2": {
                "n": int(primary["n"]),
                "R": int(primary["R"]),
                "S": int(primary["S"]),
                "unique_erd_groups": int(predictions["erd_group"].nunique()),
            },
            "model": model_lock["specification"],
            "feature_schema_sha256": model_lock["feature_schema_sha256"],
            "bridge_audit": bridge,
            "amrfinder_runtime": runtime,
            "mlst_runtime": mlst_runtime,
            "metrics": primary,
            "external_feature_matrix": {
                "shape": list(X_ext.shape),
                "out_of_schema_determinants": int(len(unknown)),
                "zero_feature_genomes": int((X_ext.sum(axis=1) == 0).sum()),
            },
            "scientific_guardrails": {
                "E2_used_for_feature_selection": False,
                "E2_used_for_hyperparameter_selection": False,
                "E2_used_for_threshold_tuning": False,
                "threshold": 0.5,
                "same_known_amr_schema_as_development": True,
                "exact_file04_local_branch_reproduction": True,
                "same_amrfinder_software_version": True,
                "same_amrfinder_database_version": True,
                "same_amrfinder_organism": True,
                "same_amrfinder_mode": True,
                "same_emrD_exclusion": True,
                "clonal_overlap_firewall": True,
            },
            "outputs": {k: str(v.relative_to(root)) for k, v in outputs.items()},
        }
        atomic_json(final_summary, summary)

        if failure.exists():
            failure.unlink()

        log("=" * 104)
        log("FILE11C STATUS : PASS_BLIND_EXTERNAL_VALIDATION")
        log(f"External E2    : n={primary['n']} | R={primary['R']} | S={primary['S']}")
        log(f"AUROC          : {primary['roc_auc']:.4f}")
        log(f"AveragePrec.   : {primary['average_precision']:.4f}")
        log(f"Bal. accuracy  : {primary['balanced_accuracy']:.4f}")
        log(f"MCC            : {primary['mcc']:.4f}")
        log(f"Sensitivity    : {primary['sensitivity']:.4f}")
        log(f"Specificity    : {primary['specificity']:.4f}")
        log(f"Brier          : {primary['brier']:.4f}")
        log(f"Final summary  : {final_summary}")
        log("=" * 104)
        return 0

    except KeyboardInterrupt:
        atomic_json(
            failure,
            {
                "script_version": VERSION,
                "status": "INTERRUPTED",
                "stage": stage,
                "utc": utc_now(),
                "message": "Completed AMRFinder/MLST checkpoints are preserved; rerun to resume.",
            },
        )
        log("FILE11C INTERRUPTED. Completed checkpoints are preserved.")
        return 130

    except Exception as exc:
        atomic_json(
            failure,
            {
                "script_version": VERSION,
                "status": "FAILED",
                "stage": stage,
                "utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        log("")
        log("FILE11C FAILED")
        log(f"Stage : {stage}")
        log(f"Error : {exc}")
        log(f"Crash record: {failure}")
        return 1

    finally:
        lock_file.unlink(missing_ok=True)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    raise SystemExit(main())
