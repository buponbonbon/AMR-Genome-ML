#!/usr/bin/env python3
"""
FILE14 — independent E3 construction, firewall, harmonized AMRFinder annotation,
and blind scoring of the frozen FILE13 model.

Scientific intent
-----------------
FILE14 is confirmatory external validation. It MUST NOT tune, reselect, recalibrate,
or refit the FILE13 primary model using E3. The E3 cohort is locked before any
model scoring. FILE13 model/schema/protocol hashes are verified before E3 work.

Default stages
--------------
  1/10 Verify FILE13 freeze, hashes, and E3 timing precondition.
  2/10 Load + freeze E3 manifest (no silent row selection).
  3/10 Independence firewall vs development and frozen E2.
  4/10 Acquire/validate E3 FASTAs with per-genome checkpointing.
  5/10 Harmonized AMRFinder annotation with per-genome checkpoint/resume.
  6/10 Parse calls and project onto frozen FILE13 schemas.
  7/10 Blind score frozen primary (+ frozen 668 sensitivity if available).
  8/10 Compute confirmatory metrics + cluster bootstrap + calibration.
  9/10 Descriptive transportability diagnostics (no model selection).
 10/10 Integrity manifest, final lock, and summary.

Expected E3 manifest columns (aliases accepted)
-----------------------------------------------
Required in strict mode:
  Genome ID
  Phenotype                  Resistant/Susceptible or 1/0
  Assembly Accession         e.g. GCA_... / GCF_...
  BioSample                  e.g. SAMN...
  NCBI SNP/ERD Group         an NCBI SNP/ERD cluster/group identifier
Optional:
  MLST
  FASTA Path                 local/Drive FASTA; otherwise Assembly Accession can
                             be downloaded with NCBI datasets CLI.

Example
-------
python scripts/file14_independent_E3_blind_validation.py \
  --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML \
  --e3-manifest data/external_validation/e3/e3_candidate_manifest.csv \
  --workers 10 \
  --bootstrap-replicates 2000

If FASTAs are not local, add --download-missing and ensure `datasets` is installed.
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
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
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

VERSION = "2.0.5"
SEED = 20260923
DEFAULT_ROOT = Path("/mnt/c/Users/ASUS/Desktop/AMR-Genome-ML")
SUBTYPES = {"AMR", "POINT", "POINT_DISRUPT"}
BOOT_METRICS = ["roc_auc", "average_precision", "balanced_accuracy", "mcc", "brier"]
PRIMARY_NAME = "harmonized_rebuilt_primary"
SENS_NAME = "harmonized_original668_projection"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


LOG_FILE: Path | None = None

def log(s: str) -> None:
    line = f"[{ts()}] {s}"
    print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def shafile(p: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def shatext(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_text(p: Path, s: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f".tmp.{os.getpid()}")
    q.write_text(s, encoding="utf-8")
    os.replace(q, p)


def atomic_json(p: Path, obj: Any) -> None:
    atomic_text(p, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_csv(p: Path, df: pd.DataFrame, compression: str | None = None) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f".tmp.{os.getpid()}")
    df.to_csv(q, index=False, encoding="utf-8-sig", compression=compression)
    os.replace(q, p)


def reqfile(p: Path, label: str) -> None:
    if not Path(p).is_file():
        raise FileNotFoundError(f"Missing {label}: {p}")


def reqdir(p: Path, label: str) -> None:
    if not Path(p).is_dir():
        raise FileNotFoundError(f"Missing {label}: {p}")


def normid(x: Any) -> str:
    return "" if pd.isna(x) else str(x).strip()


def safe(g: str) -> str:
    a = re.sub(r"[^A-Za-z0-9._-]+", "_", g).strip("._") or "genome"
    return a + "__" + hashlib.sha1(g.encode()).hexdigest()[:10]


def normalize_accession(x: Any) -> str:
    s = normid(x).upper()
    if not s:
        return ""
    # Keep version because versioned assemblies can differ; also store base separately where needed.
    return s


def accession_base(x: Any) -> str:
    s = normalize_accession(x)
    return s.split(".")[0] if s else ""


def normalize_biosample(x: Any) -> str:
    return normid(x).upper()


def normalize_cluster(x: Any) -> str:
    return normid(x).upper()


def file_mtime_iso(p: Path) -> str:
    return datetime.fromtimestamp(Path(p).stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


# -----------------------------------------------------------------------------
# Column aliases + phenotype parsing
# -----------------------------------------------------------------------------

ALIASES = {
    "genome_id": ["Genome ID", "genome_id", "GenomeID", "genome", "ID"],
    "phenotype": ["Phenotype", "phenotype", "Resistant Phenotype", "Resistance Phenotype", "label", "y"],
    "assembly": ["Assembly Accession", "Assembly", "assembly_accession", "Assembly accession", "AssemblyAccession"],
    "biosample": ["BioSample", "Biosample", "biosample", "BioSample Accession", "BioSample accession"],
    "cluster": [
        "NCBI SNP/ERD Group", "SNP/ERD Group", "SNP ERD Group", "SNP Cluster", "SNP cluster",
        "ERD Cluster", "ERD cluster", "SNP/ERD cluster", "NCBI Cluster"
    ],
    "mlst": ["MLST", "ST", "Sequence Type", "sequence_type"],
    "fasta": ["FASTA Path", "Fasta Path", "fasta_path", "FASTA", "fasta", "Genome FASTA"],
}


def find_col(cols: Iterable[str], key: str) -> str | None:
    cols = list(cols)
    lower = {str(c).strip().casefold(): c for c in cols}
    for a in ALIASES[key]:
        if a.casefold() in lower:
            return lower[a.casefold()]
    return None


def phenotype_to_y(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        if int(x) in (0, 1):
            return int(x)
    if isinstance(x, (float, np.floating)) and not pd.isna(x):
        if float(x) in (0.0, 1.0):
            return int(x)
    s = normid(x).casefold()
    resistant = {"r", "resistant", "resistance", "1", "true", "yes"}
    susceptible = {"s", "susceptible", "sensitive", "0", "false", "no"}
    if s in resistant:
        return 1
    if s in susceptible:
        return 0
    raise ValueError(f"Unrecognized binary phenotype: {x!r}")


def canonical_e3_manifest(path: Path, allow_limited: bool) -> pd.DataFrame:
    reqfile(path, "E3 manifest")
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    gc = find_col(d.columns, "genome_id")
    pc = find_col(d.columns, "phenotype")
    ac = find_col(d.columns, "assembly")
    bc = find_col(d.columns, "biosample")
    cc = find_col(d.columns, "cluster")
    mc = find_col(d.columns, "mlst")
    fc = find_col(d.columns, "fasta")
    if gc is None or pc is None:
        raise RuntimeError(f"E3 manifest must contain Genome ID and Phenotype. Columns={list(d.columns)}")
    if not allow_limited:
        miss = []
        if ac is None: miss.append("Assembly Accession")
        if bc is None: miss.append("BioSample")
        if cc is None: miss.append("NCBI SNP/ERD Group")
        if miss:
            raise RuntimeError(
                "Strict E3 firewall requires identity fields: " + ", ".join(miss) +
                ". Add them to the manifest; do not use --allow-limited-firewall for the confirmatory E3 without documenting why."
            )
    out = pd.DataFrame()
    out["Genome ID"] = d[gc].map(normid)
    out["y"] = [phenotype_to_y(x) for x in d[pc]]
    out["Phenotype"] = np.where(out["y"].eq(1), "Resistant", "Susceptible")
    out["Assembly Accession"] = d[ac].map(normalize_accession) if ac else ""
    out["BioSample"] = d[bc].map(normalize_biosample) if bc else ""
    out["NCBI SNP/ERD Group"] = d[cc].map(normalize_cluster) if cc else ""
    out["MLST"] = d[mc].map(normid) if mc else ""
    out["FASTA Path"] = d[fc].map(normid) if fc else ""

    if out["Genome ID"].eq("").any():
        raise RuntimeError("Blank Genome ID in E3 manifest")
    if out["Genome ID"].duplicated().any():
        ex = out.loc[out["Genome ID"].duplicated(False), "Genome ID"].head(20).tolist()
        raise RuntimeError(f"Duplicate E3 Genome IDs: {ex}")
    if (out["Assembly Accession"] != "").any():
        dup = out.loc[out["Assembly Accession"].ne("") & out["Assembly Accession"].duplicated(False), "Assembly Accession"]
        if len(dup):
            raise RuntimeError(f"Duplicate E3 Assembly Accessions: {dup.head(20).tolist()}")
    if (out["BioSample"] != "").any():
        dup = out.loc[out["BioSample"].ne("") & out["BioSample"].duplicated(False), "BioSample"]
        if len(dup):
            raise RuntimeError(f"Duplicate E3 BioSamples: {dup.head(20).tolist()}")
    return out


def cohort_hash(df: pd.DataFrame) -> str:
    cols = ["Genome ID", "y", "Phenotype", "Assembly Accession", "BioSample", "NCBI SNP/ERD Group", "MLST", "FASTA Path"]
    z = df[cols].copy().sort_values("Genome ID").reset_index(drop=True)
    return shatext(z.to_csv(index=False, lineterminator="\n"))


# -----------------------------------------------------------------------------
# FILE13 freeze verification
# -----------------------------------------------------------------------------

def resolve_artifact(root: Path, value: str | None, fallback: Path) -> Path:
    if value:
        p = Path(value)
        if p.is_file():
            return p.resolve()
        # allow moved project: use basename under fallback parent
        q = fallback.parent / p.name
        if q.is_file():
            return q.resolve()
    return fallback.resolve()


def load_file13_freeze(root: Path) -> dict[str, Any]:
    cp = root / "checkpoints/file13_harmonized_amrfinder"
    md = root / "models/file13"
    flagp = cp / "FREEZE_COMPLETE.flag"
    manp = md / "file13_freeze_manifest.json"
    lockp = cp / "file13_protocol_lock.json"
    reqfile(flagp, "FILE13 FREEZE_COMPLETE.flag")
    reqfile(manp, "FILE13 freeze manifest")
    reqfile(lockp, "FILE13 protocol lock")

    flag = json.loads(flagp.read_text())
    man = json.loads(manp.read_text())
    lock = json.loads(lockp.read_text())
    if flag.get("status") != "PASS_HARMONIZED_DEVELOPMENT_FREEZE":
        raise RuntimeError(f"FILE13 freeze flag status is not PASS: {flag.get('status')}")
    if man.get("status") != "PASS_HARMONIZED_DEVELOPMENT_FREEZE":
        raise RuntimeError(f"FILE13 manifest status is not PASS: {man.get('status')}")

    protocol_sha = man.get("protocol_sha256")
    if not protocol_sha or protocol_sha != flag.get("protocol_sha256") or protocol_sha != lock.get("protocol_sha256"):
        raise RuntimeError("FILE13 protocol hash disagreement among manifest/flag/protocol lock")

    primary_model = resolve_artifact(
        root,
        man.get("primary_model", {}).get("model_path"),
        md / "file13_harmonized_primary_model.joblib",
    )
    primary_schema = resolve_artifact(
        root,
        man.get("primary_feature_schema", {}).get("path"),
        md / "file13_harmonized_primary_feature_schema.csv",
    )
    reqfile(primary_model, "FILE13 primary model")
    reqfile(primary_schema, "FILE13 primary schema")

    expected_model_sha = man["primary_model"]["model_sha256"]
    expected_schema_sha = man["primary_feature_schema"]["sha256"]
    if shafile(primary_model) != expected_model_sha:
        raise RuntimeError("FILE13 primary model SHA256 mismatch")
    if shafile(primary_schema) != expected_schema_sha:
        raise RuntimeError("FILE13 primary schema SHA256 mismatch")
    if flag.get("primary_model_sha256") != expected_model_sha or flag.get("primary_schema_sha256") != expected_schema_sha:
        raise RuntimeError("FILE13 freeze flag model/schema hash disagreement")

    sens_model = resolve_artifact(
        root,
        man.get("sensitivity_model", {}).get("model_path"),
        md / "file13_harmonized_original668_sensitivity_model.joblib",
    )
    sens_schema = resolve_artifact(
        root,
        man.get("sensitivity_feature_schema", {}).get("path"),
        md / "file13_harmonized_original668_sensitivity_feature_schema.csv",
    )
    if not sens_model.is_file(): sens_model = None
    if not sens_schema.is_file(): sens_schema = None

    return {
        "checkpoint_dir": cp,
        "flag_path": flagp,
        "manifest_path": manp,
        "protocol_lock_path": lockp,
        "flag": flag,
        "manifest": man,
        "protocol_lock": lock,
        "protocol_sha256": protocol_sha,
        "primary_model_path": primary_model,
        "primary_model_sha256": expected_model_sha,
        "primary_schema_path": primary_schema,
        "primary_schema_sha256": expected_schema_sha,
        "sensitivity_model_path": sens_model,
        "sensitivity_schema_path": sens_schema,
        "freeze_flag_mtime": flagp.stat().st_mtime,
        "freeze_manifest_sha256": shafile(manp),
    }


def load_schema(path: Path) -> list[str]:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "feature" not in d.columns:
        raise RuntimeError(f"Schema missing feature column: {path}")
    feats = d["feature"].map(str).str.strip().tolist()
    if not feats or any(not x for x in feats):
        raise RuntimeError(f"Invalid/blank features in schema: {path}")
    if len({x.casefold() for x in feats}) != len(feats):
        raise RuntimeError(f"Case-insensitive duplicate features in schema: {path}")
    return feats


# -----------------------------------------------------------------------------
# Firewall reference harvesting
# -----------------------------------------------------------------------------

def read_table(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, usecols=usecols)


def header_cols(path: Path) -> list[str]:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        return list(pd.read_csv(path, sep=sep, nrows=0).columns)
    except Exception:
        return []


def candidate_firewall_tables(root: Path, explicit: list[Path]) -> list[Path]:
    out = []
    for p in explicit:
        if p and Path(p).is_file(): out.append(Path(p).resolve())
    known = [
        root / "data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv",
        root / "data/processed/file07_common_cohort.csv",
    ]
    for p in known:
        if p.is_file(): out.append(p.resolve())
    data = root / "data"
    if data.is_dir():
        for ext in ("*.csv", "*.tsv"):
            for p in data.rglob(ext):
                s = str(p).lower()
                if "file14" in s or "known_amr_harmonized" in s or "evaluation" in s:
                    continue
                try:
                    if p.stat().st_size > 150_000_000:
                        continue
                except OSError:
                    continue
                name = p.name.lower()
                if any(k in name for k in ["manifest", "metadata", "cohort", "snp", "erd", "isolate", "genome"]):
                    out.append(p.resolve())
    return sorted(set(out), key=str)


def harvest_identity(root: Path, target_ids: set[str], explicit: list[Path], label: str) -> dict[str, Any]:
    vals = {"genome_id": set(target_ids), "assembly": set(), "assembly_base": set(), "biosample": set(), "cluster": set()}
    sources = []
    for p in candidate_firewall_tables(root, explicit):
        cols = header_cols(p)
        if not cols:
            continue
        gc = find_col(cols, "genome_id")
        if gc is None:
            continue
        ac = find_col(cols, "assembly")
        bc = find_col(cols, "biosample")
        cc = find_col(cols, "cluster")
        if not any([ac, bc, cc]):
            continue
        use = [gc] + [x for x in [ac, bc, cc] if x is not None]
        try:
            d = read_table(p, usecols=list(dict.fromkeys(use)))
        except Exception:
            continue
        d[gc] = d[gc].map(normid)
        q = d[d[gc].isin(target_ids)].copy()
        if q.empty:
            continue
        before = {k: len(v) for k, v in vals.items()}
        vals["genome_id"].update({normid(x) for x in q[gc] if normid(x)})
        if ac:
            aa = {normalize_accession(x) for x in q[ac] if normalize_accession(x)}
            vals["assembly"].update(aa)
            vals["assembly_base"].update(accession_base(x) for x in aa)
        if bc:
            vals["biosample"].update(normalize_biosample(x) for x in q[bc] if normalize_biosample(x))
        if cc:
            vals["cluster"].update(normalize_cluster(x) for x in q[cc] if normalize_cluster(x))
        after = {k: len(v) for k, v in vals.items()}
        if after != before:
            sources.append({"path": str(p), "matched_rows": int(len(q)), "columns": use})
    return {"label": label, "values": vals, "sources": sources}


def firewall_e3(e3: pd.DataFrame, devref: dict, e2ref: dict, allow_limited: bool) -> tuple[pd.DataFrame, dict]:
    e3_ids = set(e3["Genome ID"])
    e3_asm = {normalize_accession(x) for x in e3["Assembly Accession"] if normalize_accession(x)}
    e3_ab = {accession_base(x) for x in e3_asm}
    e3_bio = {normalize_biosample(x) for x in e3["BioSample"] if normalize_biosample(x)}
    e3_cl = {normalize_cluster(x) for x in e3["NCBI SNP/ERD Group"] if normalize_cluster(x)}

    def compare(ref: dict, name: str) -> dict:
        v = ref["values"]
        return {
            "reference": name,
            "reference_sources": ref["sources"],
            "reference_counts": {k: len(x) for k, x in v.items()},
            "overlap_genome_id": sorted(e3_ids & v["genome_id"]),
            "overlap_assembly": sorted(e3_asm & v["assembly"]),
            "overlap_assembly_base": sorted(e3_ab & v["assembly_base"]),
            "overlap_biosample": sorted(e3_bio & v["biosample"]),
            "overlap_ncbi_cluster": sorted(e3_cl & v["cluster"]),
        }

    dev = compare(devref, "development")
    e2 = compare(e2ref, "E2")
    checks = {"development": dev, "E2": e2}

    overlap_any = False
    rows = []
    for name, c in checks.items():
        for field in ["genome_id", "assembly", "assembly_base", "biosample", "ncbi_cluster"]:
            key = "overlap_" + field
            n = len(c[key])
            rows.append({"reference": name, "identity_type": field, "overlap_n": n, "examples": ";".join(c[key][:20])})
            overlap_any |= n > 0
    limitations = []
    # For a reviewer-defensible E3, dev cluster firewall is critical because FILE11 used NCBI SNP/ERD firewall.
    if len(devref["values"]["cluster"]) == 0:
        limitations.append("No development NCBI SNP/ERD cluster references were found")
    if len(e2ref["values"]["genome_id"]) == 0:
        limitations.append("No E2 reference identities were found")
    if len(e3_cl) == 0:
        limitations.append("E3 manifest contains no NCBI SNP/ERD cluster values")
    if overlap_any:
        status = "FAIL_OVERLAP"
    elif limitations and not allow_limited:
        status = "FAIL_INCOMPLETE_FIREWALL"
    else:
        status = "PASS_STRICT_INDEPENDENCE_FIREWALL" if not limitations else "PASS_WITH_LIMITED_FIREWALL"
    return pd.DataFrame(rows), {"status": status, "limitations": limitations, "development": dev, "E2": e2}


# -----------------------------------------------------------------------------
# FASTA acquisition/checkpoint
# -----------------------------------------------------------------------------

def is_fasta(p: Path) -> bool:
    return any(p.name.lower().endswith(e) for e in [".fa", ".fna", ".fasta", ".fas", ".ffn", ".fa.gz", ".fna.gz", ".fasta.gz"])


def fasta_content_sha(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def find_manifest_fasta(raw: str, manifest_dir: Path, root: Path) -> Path | None:
    if not raw:
        return None
    p = Path(raw).expanduser()
    candidates = [p] if p.is_absolute() else [manifest_dir / p, root / p]
    for q in candidates:
        if q.is_file() and is_fasta(q):
            return q.resolve()
    return None


def datasets_version(exe: str) -> str:
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
        return (r.stdout + "\n" + r.stderr).strip()
    except Exception:
        return ""


def download_one(gid: str, acc: str, outdir: Path, done_dir: Path, datasets_exe: str, force: bool) -> dict:
    s = safe(gid)
    fasta = outdir / f"{s}.fna"
    done = done_dir / f"{s}.download.done.json"
    if not force and fasta.is_file() and done.is_file():
        try:
            d = json.loads(done.read_text())
            if d.get("status") == "PASS" and d.get("assembly_accession") == acc and d.get("fasta_content_sha256") == fasta_content_sha(fasta):
                return {"Genome ID": gid, "status": "SKIP_CHECKPOINT", "fasta_path": str(fasta), "message": ""}
        except Exception:
            pass
    if not acc:
        return {"Genome ID": gid, "status": "FAIL", "fasta_path": "", "message": "No FASTA path and no Assembly Accession"}
    outdir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    work = outdir / (s + ".download_tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    z = work / "genome.zip"
    cmd = [datasets_exe, "download", "genome", "accession", acc, "--include", "genome", "--filename", str(z)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"datasets returncode={r.returncode}; stderr={(r.stderr or '')[-3000:]}")
        if not z.is_file():
            raise RuntimeError("datasets finished but ZIP was not created")
        with zipfile.ZipFile(z) as zh:
            names = [n for n in zh.namelist() if n.lower().endswith((".fna", ".fa", ".fasta")) and "/genomic.fna" in n.lower() or n.lower().endswith("_genomic.fna")]
            if not names:
                names = [n for n in zh.namelist() if n.lower().endswith((".fna", ".fa", ".fasta"))]
            if len(names) != 1:
                raise RuntimeError(f"Expected one genomic FASTA in datasets ZIP, found {len(names)}: {names[:10]}")
            src = zh.open(names[0])
            tmp = fasta.with_name(fasta.name + f".tmp.{os.getpid()}")
            with src, tmp.open("wb") as oh:
                shutil.copyfileobj(src, oh, 1 << 20)
            os.replace(tmp, fasta)
        sha = fasta_content_sha(fasta)
        atomic_json(done, {
            "Genome ID": gid, "status": "PASS", "assembly_accession": acc,
            "completed_utc": utc(), "command": cmd, "fasta_path": str(fasta),
            "fasta_content_sha256": sha, "elapsed_seconds": time.time() - t0,
        })
        return {"Genome ID": gid, "status": "PASS", "fasta_path": str(fasta), "message": ""}
    except Exception as e:
        return {"Genome ID": gid, "status": "FAIL", "fasta_path": "", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def acquire_fastas(e3: pd.DataFrame, root: Path, manifest_path: Path, fd: Path, cp: Path,
                   workers: int, datasets_exe: str, download_missing: bool, force: bool) -> pd.DataFrame:
    dl = fd / "downloaded_fastas"
    dd = cp / "per_genome_download_done"
    rows = []
    pending = []
    for _, r in e3.iterrows():
        gid = r["Genome ID"]
        p = find_manifest_fasta(r["FASTA Path"], manifest_path.parent, root)
        if p:
            rows.append({"Genome ID": gid, "status": "LOCAL", "fasta_path": str(p), "message": ""})
        else:
            pending.append((gid, r["Assembly Accession"]))
    if pending and not download_missing:
        raise RuntimeError(
            f"{len(pending)} E3 genomes lack a valid FASTA Path. Re-run with --download-missing after verifying Assembly Accession values, or provide local FASTAs."
        )
    if pending:
        ver = datasets_version(datasets_exe)
        if not ver:
            raise RuntimeError(f"NCBI datasets CLI not available: {datasets_exe}")
        log(f"[PROGRESS] downloading/checking {len(pending)} missing FASTAs with datasets | version={ver}")
        with ThreadPoolExecutor(max_workers=min(workers, 6)) as ex:
            fut = {ex.submit(download_one, gid, acc, dl, dd, datasets_exe, force): gid for gid, acc in pending}
            done_n = 0
            for f in as_completed(fut):
                rr = f.result(); rows.append(rr); done_n += 1
                if rr["status"] == "FAIL": log(f"[FAIL] FASTA {rr['Genome ID']} | {rr['message']}")
                if done_n == 1 or done_n % 20 == 0 or done_n == len(pending):
                    c = Counter(x["status"] for x in rows)
                    log(f"[PROGRESS] FASTA acquisition {done_n}/{len(pending)} pending | status={dict(c)}")
    z = pd.DataFrame(rows)
    bad = z[z.status.eq("FAIL")]
    if len(bad):
        atomic_csv(cp / "file14_fasta_acquisition_failures.csv", bad)
        raise RuntimeError(f"{len(bad)} E3 FASTAs failed acquisition")
    z = e3[["Genome ID"]].merge(z, on="Genome ID", validate="one_to_one")
    z["fasta_path"] = z["fasta_path"].map(str)
    z["fasta_content_sha256"] = [fasta_content_sha(Path(p)) for p in z["fasta_path"]]
    z["bytes"] = [Path(p).stat().st_size for p in z["fasta_path"]]
    z["mtime_ns"] = [Path(p).stat().st_mtime_ns for p in z["fasta_path"]]
    return z


# -----------------------------------------------------------------------------
# AMRFinder annotation
# -----------------------------------------------------------------------------

def valid_tsv(p: Path) -> tuple[bool, str]:
    p = Path(p)
    if not p.is_file() or p.stat().st_size == 0:
        return False, "missing/empty"
    try:
        head = p.open(encoding="utf-8", errors="replace").readline().rstrip("\r\n").split("\t")
    except Exception as e:
        return False, str(e)
    need = {"Type", "Subtype", "Element symbol"}
    return (need <= set(head), "ok" if need <= set(head) else f"missing {sorted(need-set(head))}")


def prep_fasta(fa: Path, tmp: Path) -> tuple[Path, Path | None]:
    if not fa.name.lower().endswith(".gz"):
        return fa, None
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(fa, "rb") as a, tmp.open("wb") as b:
        shutil.copyfileobj(a, b, 1 << 20)
    return tmp, tmp


def amrfinder_version(exe: Path) -> str:
    for arg in ["--version", "-V"]:
        try:
            r = subprocess.run([str(exe), arg], capture_output=True, text=True, timeout=30)
            t = (r.stdout + "\n" + r.stderr).strip()
            if r.returncode == 0 and t:
                return t
        except Exception:
            pass
    return ""


def run_amr_one(gid: str, fa: Path, fasta_sha: str, raw: Path, done: Path, amr: Path, db: Path,
                threads: int, file13_protocol_sha: str, annotation_protocol_sha: str,
                tmpdir: Path, timeout: int, retries: int, force: bool) -> dict:
    if not force and raw.is_file() and done.is_file():
        try:
            d = json.loads(done.read_text()); ok, _ = valid_tsv(raw)
            if (
                ok and d.get("status") == "PASS" and
                d.get("file13_protocol_sha256") == file13_protocol_sha and
                d.get("file14_annotation_protocol_sha256") == annotation_protocol_sha and
                d.get("fasta_content_sha256") == fasta_sha and
                d.get("output_sha256") == shafile(raw)
            ):
                return {"Genome ID": gid, "status": "SKIP_CHECKPOINT", "elapsed_seconds": 0.0, "raw_path": str(raw), "message": ""}
        except Exception:
            pass
    t0 = time.time(); err = ""
    for attempt in range(1, retries + 2):
        work = tmpdir / safe(gid); work.mkdir(parents=True, exist_ok=True)
        tmpout = raw.with_name(raw.name + f".tmp.{os.getpid()}.{attempt}")
        cleanup = None
        try:
            nuc, cleanup = prep_fasta(fa, work / (safe(gid) + ".fna"))
            cmd = [str(amr), "-n", str(nuc), "-O", "Klebsiella_pneumoniae", "--database", str(db), "--threads", str(threads)]
            st = time.time()
            with tmpout.open("w", encoding="utf-8") as oh:
                r = subprocess.run(cmd, stdout=oh, stderr=subprocess.PIPE, text=True, timeout=timeout)
            if r.returncode != 0:
                raise RuntimeError(f"returncode={r.returncode}; stderr={(r.stderr or '')[-4000:]}")
            ok, why = valid_tsv(tmpout)
            if not ok:
                raise RuntimeError(f"invalid TSV: {why}")
            raw.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmpout, raw)
            atomic_json(done, {
                "Genome ID": gid, "status": "PASS", "completed_utc": utc(),
                "file13_protocol_sha256": file13_protocol_sha,
                "file14_annotation_protocol_sha256": annotation_protocol_sha,
                "fasta_path": str(fa), "fasta_content_sha256": fasta_sha,
                "raw_path": str(raw), "output_sha256": shafile(raw),
                "attempt": attempt, "elapsed_seconds": time.time() - st,
                "command": cmd, "stderr_tail": (r.stderr or "")[-4000:],
            })
            return {"Genome ID": gid, "status": "PASS", "elapsed_seconds": time.time() - t0, "raw_path": str(raw), "message": ""}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            try: tmpout.unlink(missing_ok=True)
            except Exception: pass
            if attempt <= retries: time.sleep(min(10, 2 * attempt))
        finally:
            if cleanup:
                try: cleanup.unlink(missing_ok=True)
                except Exception: pass
    return {"Genome ID": gid, "status": "FAIL", "elapsed_seconds": time.time() - t0, "raw_path": str(raw), "message": err}


def annotate_e3(e3: pd.DataFrame, fmanifest: pd.DataFrame, rawdir: Path, donedir: Path, tmpdir: Path, cp: Path,
                amr: Path, db: Path, workers: int, threads: int, file13_protocol_sha: str,
                annotation_protocol_sha: str, timeout: int, retries: int, force: bool) -> pd.DataFrame:
    fm = fmanifest.set_index("Genome ID")
    rows = []; n = len(e3); start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {}
        for gid in e3["Genome ID"]:
            fa = Path(fm.loc[gid, "fasta_path"])
            fsha = str(fm.loc[gid, "fasta_content_sha256"])
            fut[ex.submit(
                run_amr_one, gid, fa, fsha, rawdir / (safe(gid) + ".tsv"), donedir / (safe(gid) + ".done.json"),
                amr, db, threads, file13_protocol_sha, annotation_protocol_sha, tmpdir, timeout, retries, force
            )] = gid
        for f in as_completed(fut):
            rr = f.result(); rows.append(rr)
            k = len(rows)
            if rr["status"] == "FAIL": log(f"[FAIL] AMRFinder {rr['Genome ID']} | {rr['message']}")
            if k == 1 or k % 20 == 0 or k == n:
                c = Counter(x["status"] for x in rows)
                elapsed = time.time() - start
                done_rate = k / max(elapsed, 1e-9)
                eta = (n - k) / done_rate if done_rate else math.nan
                log(f"[PROGRESS] E3 AMRFinder {k}/{n} ({100*k/n:5.1f}%) | PASS={c['PASS']} SKIP={c['SKIP_CHECKPOINT']} FAIL={c['FAIL']} | elapsed={elapsed/3600:.2f}h ETA={(eta/3600 if math.isfinite(eta) else float('nan')):.2f}h")
                atomic_csv(cp / "file14_annotation_status.csv", pd.DataFrame(rows))
    d = pd.DataFrame(rows)
    bad = d[d.status.eq("FAIL")]
    if len(bad):
        atomic_csv(cp / "file14_annotation_failures.csv", bad)
        raise RuntimeError(f"{len(bad)} E3 AMRFinder annotations failed; fix and rerun same command")
    return d


def parse_tsv(p: Path) -> list[str]:
    d = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    need = {"Type", "Subtype", "Element symbol"}
    if not need <= set(d.columns):
        raise RuntimeError(f"{p} missing columns {sorted(need-set(d.columns))}")
    q = d[d.Type.str.strip().eq("AMR") & d.Subtype.str.strip().isin(SUBTYPES)]
    seen = set(); out = []
    for x in q["Element symbol"]:
        s = str(x).strip(); k = s.casefold()
        if s and k not in seen:
            seen.add(k); out.append(s)
    return out


# -----------------------------------------------------------------------------
# Matrix projection + scoring
# -----------------------------------------------------------------------------

def project_calls(ids: list[str], rawdir: Path, primary_feats: list[str], sens_feats: list[str] | None):
    pmap = {f.casefold(): j for j, f in enumerate(primary_feats)}
    smap = {f.casefold(): j for j, f in enumerate(sens_feats or [])}
    xp = np.zeros((len(ids), len(primary_feats)), dtype=np.uint8)
    xs = np.zeros((len(ids), len(sens_feats or [])), dtype=np.uint8) if sens_feats else None
    out_schema = Counter(); call_counts = []
    for i, gid in enumerate(ids):
        calls = parse_tsv(rawdir / (safe(gid) + ".tsv"))
        keys = {c.casefold(): c for c in calls}
        call_counts.append(len(keys))
        for k, spelling in keys.items():
            if k in pmap:
                xp[i, pmap[k]] = 1
            else:
                out_schema[spelling] += 1
            if xs is not None and k in smap:
                xs[i, smap[k]] = 1
    return xp, xs, out_schema, call_counts


def score_bundle(model_path: Path, schema_feats: list[str], X: np.ndarray) -> tuple[np.ndarray, float, dict]:
    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    threshold = float(bundle.get("threshold", 0.5)) if isinstance(bundle, dict) else 0.5
    bfeats = bundle.get("feature_names") if isinstance(bundle, dict) else None
    if bfeats is not None and [str(x) for x in bfeats] != [str(x) for x in schema_feats]:
        raise RuntimeError(f"Model bundle feature_names do not exactly match frozen schema: {model_path}")
    p = model.predict_proba(X.astype(float))[:, 1]
    meta = {
        "model_path": str(model_path), "model_sha256": shafile(model_path),
        "threshold": threshold, "n_features": X.shape[1],
    }
    return p, threshold, meta


# -----------------------------------------------------------------------------
# Metrics + bootstrap + calibration
# -----------------------------------------------------------------------------

def metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "R": int(y.sum()), "S": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    # Newton-Raphson logistic calibration: logit(P(Y=1)) = a + b*logit(p)
    eps = 1e-8
    pp = np.clip(np.asarray(p, float), eps, 1 - eps)
    x = np.log(pp / (1 - pp))
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(100):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-np.clip(eta, -40, 40)))
        w = np.maximum(mu * (1 - mu), 1e-9)
        grad = X.T @ (y - mu)
        h = X.T @ (w[:, None] * X)
        try:
            step = np.linalg.solve(h, grad)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta2 = beta + step
        if np.max(np.abs(beta2 - beta)) < 1e-10:
            beta = beta2; break
        beta = beta2
    return float(beta[0]), float(beta[1])


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, groups: pd.Series, threshold: float,
                      nrep: int, seed: int, label: str) -> pd.DataFrame:
    g = groups.fillna("UNRESOLVED").astype(str).to_numpy()
    ug = np.array(pd.unique(g), dtype=object)
    mem = {u: np.flatnonzero(g == u) for u in ug}
    rng = np.random.default_rng(seed); rows = []
    for r in range(1, nrep + 1):
        sm = rng.choice(ug, size=len(ug), replace=True)
        ix = np.concatenate([mem[u] for u in sm])
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            continue
        mm = metrics(yy, p[ix], threshold)
        rows.append({"replicate": r, **{k: mm[k] for k in BOOT_METRICS}})
        if r == 1 or r % 250 == 0 or r == nrep:
            log(f"[PROGRESS] bootstrap {label} {r}/{nrep} | valid={len(rows)}")
    return pd.DataFrame(rows)


def load_or_run_bootstrap(path: Path, y: np.ndarray, p: np.ndarray, groups: pd.Series, threshold: float,
                          nrep: int, seed: int, label: str) -> pd.DataFrame:
    if path.is_file():
        try:
            b = pd.read_csv(path, compression="gzip")
            need = {"replicate", *BOOT_METRICS}
            if need <= set(b.columns) and len(b) == nrep and int(b["replicate"].min()) == 1 and int(b["replicate"].max()) == nrep:
                log(f"[PASS] bootstrap checkpoint reused {label} | {len(b)}/{nrep}")
                return b
        except Exception:
            pass
    b = cluster_bootstrap(y, p, groups, threshold, nrep, seed, label)
    atomic_csv(path, b, compression="gzip")
    return b


def bootstrap_summary(point: dict, b: pd.DataFrame, representation: str) -> pd.DataFrame:
    rows = []
    for k in BOOT_METRICS:
        rows.append({
            "representation": representation, "metric": k, "point_estimate": point[k],
            "bootstrap_mean": float(b[k].mean()), "ci95_low": float(b[k].quantile(0.025)),
            "ci95_high": float(b[k].quantile(0.975)), "valid_replicates": int(len(b)),
        })
    return pd.DataFrame(rows)


def subgroup_metric_rows(e3: pd.DataFrame, p: np.ndarray, threshold: float, dev_mlst: set[str]) -> pd.DataFrame:
    mlst = e3["MLST"].map(normid)
    group = np.where(mlst.eq(""), "unresolved_ST", np.where(mlst.isin(dev_mlst), "seen_ST", "unseen_ST"))
    rows = []
    for g in ["seen_ST", "unseen_ST", "unresolved_ST"]:
        ix = np.flatnonzero(group == g)
        if len(ix) == 0:
            continue
        yy = e3["y"].to_numpy(int)[ix]; pp = p[ix]
        row = {"subgroup": g, "n": int(len(ix)), "R": int(yy.sum()), "S": int((yy == 0).sum())}
        if len(np.unique(yy)) == 2:
            row.update(metrics(yy, pp, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Self-test / template
# -----------------------------------------------------------------------------

def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.tsv"
        p.write_text("Type\tSubtype\tElement symbol\nAMR\tAMR\tblaKPC-2\nAMR\tPOINT\tgyrA_S83I\nAMR\tSTRESS\temrD\n", encoding="utf-8")
        assert valid_tsv(p)[0]
        assert parse_tsv(p) == ["blaKPC-2", "gyrA_S83I"]
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.6, 0.9])
    assert abs(metrics(y, p)["roc_auc"] - 1.0) < 1e-12
    a, b = calibration_intercept_slope(y, p)
    assert math.isfinite(a) and math.isfinite(b)
    print("FILE14 self-test: PASS")
    return 0


def write_template(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=[
        "Genome ID", "Phenotype", "Assembly Accession", "BioSample",
        "NCBI SNP/ERD Group", "MLST", "FASTA Path"
    ]).to_csv(path, index=False)
    print(path)
    return 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# =============================================================================
# FILE14 v2 consolidated production helpers
# =============================================================================

EXPECTED_HISTORICAL_ORIGINAL = 4270
EXPECTED_HISTORICAL_GOOD = 4233
EXPECTED_COMMON_REFERENCE = 4227
EXPECTED_QC_ONLY = 6
EXPECTED_QC_ONLY_IDS = {
    "573.29887", "573.34250", "573.34254", "573.34255", "573.46106", "72407.549"
}


def tool_prefix(tool: str, env_name: str | None) -> list[str]:
    """Resolve a command either on PATH or through `micromamba run -n ENV`."""
    p = shutil.which(tool)
    if p:
        return [p]
    mm = shutil.which("micromamba")
    if env_name and mm:
        probe = subprocess.run(
            [mm, "run", "-n", env_name, "bash", "-lc", f"command -v {tool}"],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return [mm, "run", "-n", env_name, tool]
    raise RuntimeError(
        f"Required tool not found: {tool}. Install it in PATH or in --assembly-env {env_name!r}."
    )


def command_version(prefix: list[str], args: list[str] | None = None) -> str:
    args = args or ["--version"]
    try:
        r = subprocess.run(prefix + args, capture_output=True, text=True, timeout=30)
        return (r.stdout + "\n" + r.stderr).strip()[:4000]
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def run_checked(cmd: list[str], *, cwd: Path | None = None, timeout: int | None = None,
                stdout_path: Path | None = None) -> subprocess.CompletedProcess:
    if stdout_path is None:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    else:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as oh:
            r = subprocess.run(cmd, cwd=cwd, stdout=oh, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed ({r.returncode}): {' '.join(map(str, cmd))}\n"
            f"stderr tail: {(r.stderr or '')[-5000:]}"
        )
    return r


def write_status_summary(path: Path, *, stage: str, status: str, next_step: str,
                         extra: dict[str, Any] | None = None) -> None:
    extra = extra or {}
    lines = [
        "FILE14 FINAL STATUS SUMMARY",
        "=" * 112,
        f"Updated UTC                    : {utc()}",
        f"Script version                 : {VERSION}",
        f"Status                         : {status}",
        f"Current/last completed stage   : {stage}",
        "",
        "LOCKED SCIENTIFIC CONTEXT",
        f"- Historical source-level QC   : {EXPECTED_HISTORICAL_ORIGINAL} -> {EXPECTED_HISTORICAL_GOOD} BV-BRC Good",
        f"- Common development cohort    : {EXPECTED_COMMON_REFERENCE}",
        "- Historical QC must NOT be rewritten as an N50/QUAST/contig rule.",
        "- E3 selection by model performance is prohibited.",
        "- No E3 tuning, feature reselection, recalibration, or threshold change.",
        "- Strict pre-assembly E3 source is the output of file14_build_strict_E3_cohort.py.",
        "",
        "RUN DETAILS",
    ]
    for k, v in extra.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "NEXT", next_step, "=" * 112, ""]
    atomic_text(path, "\n".join(lines))


def id_col(df: pd.DataFrame) -> str:
    for c in ["Genome ID", "Genome ID String", "genome_id", "GenomeID"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot locate genome ID column; columns={list(df.columns)}")


def find_development_fasta(root: Path, gid: str) -> Path | None:
    candidates = [
        root / "data/genomes" / f"{gid}.fna.gz",
        root / "data/genomes" / f"{gid}.fna",
        root / "data/genomes" / f"{gid}.fa.gz",
        root / "data/genomes" / f"{gid}.fa",
        root / "data/genomes" / f"{gid}.fasta.gz",
        root / "data/genomes" / f"{gid}.fasta",
        root / "data/genomes/fasta_gz" / f"{gid}.fna.gz",
        root / "data/genomes/fasta" / f"{gid}.fna",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def sequence_metrics(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    lens: list[int] = []
    total = amb = gc = 0
    cur = camb = cgc = 0

    def flush() -> None:
        nonlocal cur, camb, cgc, total, amb, gc
        if cur:
            lens.append(cur)
            total += cur
            amb += camb
            gc += cgc
        cur = camb = cgc = 0

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                flush(); continue
            s = "".join(line.split()).upper()
            if not s:
                continue
            cur += len(s)
            cgc += s.count("G") + s.count("C")
            camb += sum(ch not in {"A", "C", "G", "T"} for ch in s)
        flush()
    if not lens or total <= 0:
        raise RuntimeError(f"Invalid/empty FASTA: {path}")
    sl = sorted(lens, reverse=True)
    half = total / 2.0
    csum = 0
    n50 = l50 = 0
    for i, L in enumerate(sl, start=1):
        csum += L
        if csum >= half:
            n50, l50 = L, i
            break
    return {
        "assembly_length_bp": int(total),
        "contig_count": int(len(sl)),
        "n50_bp": int(n50),
        "l50": int(l50),
        "largest_contig_bp": int(max(sl)),
        "ambiguous_bases": int(amb),
        "ambiguous_fraction": float(amb / total),
        "gc_fraction": float(gc / total),
        "contigs_ge_500": int(sum(x >= 500 for x in sl)),
        "contigs_ge_1000": int(sum(x >= 1000 for x in sl)),
    }


def outer_fence(series: pd.Series) -> dict[str, float]:
    q1 = float(series.quantile(0.25, interpolation="linear"))
    med = float(series.quantile(0.50, interpolation="linear"))
    q3 = float(series.quantile(0.75, interpolation="linear"))
    iqr = q3 - q1
    return {
        "q1": q1, "median": med, "q3": q3, "iqr": iqr,
        "outer_lower": q1 - 3 * iqr, "outer_upper": q3 + 3 * iqr,
        "observed_min": float(series.min()), "observed_max": float(series.max()),
    }


def ensure_reference_envelope(root: Path, cp: Path, workers: int) -> tuple[Path, dict[str, Any]]:
    """Freeze/reuse the 4,227 common-development sequence compatibility reference."""
    qdir = root / "data/external_validation/e3/qc_reference"
    qdir.mkdir(parents=True, exist_ok=True)
    envp = qdir / "file14_E3_sequence_qc_envelope.json"
    flagp = qdir / "FILE14_QC_ENVELOPE_FROZEN.flag"
    metricsp = qdir / "file14_development_common_4227_assembly_metrics.csv"

    if envp.is_file() and flagp.is_file():
        flag = json.loads(flagp.read_text())
        env = json.loads(envp.read_text())
        if (
            flag.get("status") in {"PASS_FILE14_QC_REFERENCE_FROZEN", "FILE14_QC_ENVELOPE_FROZEN"}
            and int(env.get("common_development_sequence_reference", {}).get("n", 0)) == EXPECTED_COMMON_REFERENCE
            and int(env.get("reference_failures_under_hard_rule", 0)) == 0
        ):
            expected_sha = flag.get("envelope_sha256")
            if expected_sha and expected_sha != shafile(envp):
                raise RuntimeError("Existing QC envelope freeze flag SHA does not match envelope JSON")
            log(f"[PASS] Reusing frozen E3 sequence reference | n={EXPECTED_COMMON_REFERENCE} | sha={shafile(envp)}")
            return envp, env

    goodp = root / "data/features/known_amr/X_known_amr_final.csv"
    commonp = root / "data/processed/file07_common_cohort.csv"
    reqfile(goodp, "4,233 historical-Good membership")
    reqfile(commonp, "4,227 common development cohort")
    gd = pd.read_csv(goodp, dtype=str, keep_default_na=False)
    cd = pd.read_csv(commonp, dtype=str, keep_default_na=False)
    gc, cc = id_col(gd), id_col(cd)
    good = [normid(x) for x in gd[gc] if normid(x)]
    common = [normid(x) for x in cd[cc] if normid(x)]
    if len(good) != EXPECTED_HISTORICAL_GOOD or len(set(good)) != EXPECTED_HISTORICAL_GOOD:
        raise RuntimeError(f"Historical Good membership invariant failed: {len(good)}")
    if len(common) != EXPECTED_COMMON_REFERENCE or len(set(common)) != EXPECTED_COMMON_REFERENCE:
        raise RuntimeError(f"Common cohort invariant failed: {len(common)}")
    qconly = set(good) - set(common)
    if qconly != EXPECTED_QC_ONLY_IDS:
        raise RuntimeError(f"Unexpected 4233-4227 identity difference: {sorted(qconly)}")
    if set(common) - set(good):
        raise RuntimeError("Common cohort contains IDs outside historical Good membership")

    fmap: dict[str, Path] = {}
    missing = []
    for gid in common:
        p = find_development_fasta(root, gid)
        if p is None: missing.append(gid)
        else: fmap[gid] = p
    if missing:
        raise RuntimeError(f"Missing {len(missing)} common-development FASTAs: {missing[:30]}")
    log(f"[PASS] Reference pre-audit: {len(fmap)}/{EXPECTED_COMMON_REFERENCE} common-development FASTAs resolved")

    refcp = cp / "reference_metrics_per_genome"
    refcp.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def one(gid: str) -> dict[str, Any]:
        fa = fmap[gid]
        jp = refcp / f"{safe(gid)}.json"
        st = fa.stat()
        if jp.is_file():
            try:
                z = json.loads(jp.read_text())
                if z.get("fasta_path") == str(fa) and int(z.get("bytes", -1)) == st.st_size and int(z.get("mtime_ns", -1)) == st.st_mtime_ns:
                    return z
            except Exception:
                pass
        m = sequence_metrics(fa)
        z = {"Genome ID": gid, "fasta_path": str(fa), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, **m}
        atomic_json(jp, z)
        return z

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as ex:
        fut = {ex.submit(one, gid): gid for gid in common}
        for f in as_completed(fut):
            rows.append(f.result())
            k = len(rows)
            if k == 1 or k % 250 == 0 or k == len(common):
                elapsed = time.time() - start
                rate = k / max(elapsed, 1e-9)
                eta = (len(common) - k) / rate if rate else math.nan
                log(f"[PROGRESS] reference metrics {k}/{len(common)} ({100*k/len(common):5.1f}%) | elapsed={elapsed/60:.1f}m ETA={(eta/60 if math.isfinite(eta) else float('nan')):.1f}m")

    order = {g: i for i, g in enumerate(common)}
    rows.sort(key=lambda z: order[z["Genome ID"]])
    d = pd.DataFrame(rows)
    atomic_csv(metricsp, d)
    hard = {
        "assembly_length_bp": {"min": int(d.assembly_length_bp.min()), "max": int(d.assembly_length_bp.max())},
        "contig_count": {"max": int(d.contig_count.max())},
        "n50_bp": {"min": int(d.n50_bp.min())},
        "largest_contig_bp": {"min": int(d.largest_contig_bp.min())},
        "ambiguous_fraction": {"max": float(d.ambiguous_fraction.max())},
    }
    refpass = (
        d.assembly_length_bp.between(hard["assembly_length_bp"]["min"], hard["assembly_length_bp"]["max"])
        & d.contig_count.le(hard["contig_count"]["max"])
        & d.n50_bp.ge(hard["n50_bp"]["min"])
        & d.largest_contig_bp.ge(hard["largest_contig_bp"]["min"])
        & d.ambiguous_fraction.le(hard["ambiguous_fraction"]["max"])
    )
    if int((~refpass).sum()) != 0:
        raise RuntimeError("Internal error: empirical reference envelope excludes a reference genome")
    warns = {m: outer_fence(d[m]) for m in ["assembly_length_bp", "contig_count", "n50_bp", "largest_contig_bp", "ambiguous_fraction"]}
    env = {
        "version": VERSION,
        "created_utc": utc(),
        "purpose": "E3 sequence-compatibility reference; not a rewrite of historical BV-BRC Good/Poor QC",
        "historical_development_qc": {
            "original_n": EXPECTED_HISTORICAL_ORIGINAL,
            "retained_good_n": EXPECTED_HISTORICAL_GOOD,
            "rule": "BV-BRC genome_quality == 'Good' -> Keep; otherwise Exclude",
            "phenotype_used_for_qc": False,
        },
        "common_development_sequence_reference": {
            "n": EXPECTED_COMMON_REFERENCE,
            "source": str(commonp), "source_sha256": shafile(commonp),
            "historical_good_minus_common_ids": sorted(qconly),
        },
        "development_metrics_csv": str(metricsp),
        "development_metrics_csv_sha256": shafile(metricsp),
        "hard_pass_fail_envelope": hard,
        "reference_failures_under_hard_rule": 0,
        "robust_warning_outer_fences": warns,
        "phenotype_used": False, "model_predictions_used": False, "e3_outcomes_used": False,
    }
    atomic_json(envp, env)
    atomic_json(flagp, {
        "status": "PASS_FILE14_QC_REFERENCE_FROZEN", "version": VERSION, "created_utc": utc(),
        "common_reference_n": EXPECTED_COMMON_REFERENCE, "envelope_sha256": shafile(envp),
        "metrics_sha256": shafile(metricsp), "historical_good_minus_common_ids": sorted(qconly),
    })
    log(f"[PASS] E3 sequence reference frozen | n={EXPECTED_COMMON_REFERENCE} | envelope_sha={shafile(envp)}")
    return envp, env


def load_strict_preassembly(root: Path) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    p = root / "data/external_validation/e3/strict_build/file14_strict_E3_preassembly_pool.csv"
    auditp = root / "data/external_validation/e3/strict_build/file14_strict_E3_build_audit.json"
    reqfile(p, "strict E3 pre-assembly pool")
    reqfile(auditp, "strict E3 build audit")
    audit = json.loads(auditp.read_text())
    exp = audit.get("output_sha256", {}).get("preassembly_pool")
    if exp and shafile(p) != exp:
        raise RuntimeError("Strict pre-assembly pool SHA does not match cohort-builder audit")
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    need = ["Genome ID", "Phenotype", "Assembly Accession", "BioSample", "NCBI SNP/ERD Group", "SRA Run", "Source"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise RuntimeError(f"Strict pre-assembly pool missing columns: {miss}")
    if d["Genome ID"].duplicated().any() or d["BioSample"].replace("", np.nan).dropna().duplicated().any():
        raise RuntimeError("Strict pre-assembly pool contains duplicate Genome ID/BioSample")
    y = d["Phenotype"].map(phenotype_to_y).astype(int)
    n, nr, ns = len(d), int(y.sum()), int((1-y).sum())
    if not (n >= 300 and nr >= 100 and ns >= 100):
        raise RuntimeError(f"Strict pre-assembly pool no longer meets locked gate: n={n} R={nr} S={ns}")
    return d, p, audit


def download_public_assembly_one(row: dict[str, str], outdir: Path, done_dir: Path,
                                 datasets_cmd: list[str], timeout: int, force: bool) -> dict[str, Any]:
    gid = row["Genome ID"]; acc = row["Assembly Accession"]
    s = safe(gid)
    fa = outdir / f"{s}.fna"
    dp = done_dir / f"{s}.json"
    if not force and fa.is_file() and dp.is_file():
        try:
            z = json.loads(dp.read_text())
            if z.get("status") == "PASS" and z.get("assembly_accession") == acc and z.get("fasta_content_sha256") == fasta_content_sha(fa):
                return {"Genome ID": gid, "status": "SKIP_CHECKPOINT", "fasta_path": str(fa), "fasta_content_sha256": z["fasta_content_sha256"], "message": ""}
        except Exception:
            pass
    if not acc:
        return {"Genome ID": gid, "status": "FAIL", "fasta_path": "", "message": "missing Assembly Accession"}
    work = outdir / f".{s}.tmp"
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True, exist_ok=True)
    zipp = work / "genome.zip"
    cmd = datasets_cmd + ["download", "genome", "accession", acc, "--include", "genome", "--filename", str(zipp)]
    t0 = time.time()
    try:
        run_checked(cmd, timeout=timeout)
        with zipfile.ZipFile(zipp) as zh:
            names = [n for n in zh.namelist() if n.lower().endswith((".fna", ".fa", ".fasta"))]
            preferred = [n for n in names if n.lower().endswith("_genomic.fna") or n.lower().endswith("/genomic.fna")]
            use = preferred if len(preferred) == 1 else names
            if len(use) != 1:
                raise RuntimeError(f"Expected one genomic FASTA in datasets ZIP; found {len(use)} candidates")
            tmp = fa.with_name(fa.name + f".tmp.{os.getpid()}")
            with zh.open(use[0]) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(tmp, fa)
        sha = fasta_content_sha(fa)
        atomic_json(dp, {"status": "PASS", "Genome ID": gid, "assembly_accession": acc, "fasta_path": str(fa),
                         "fasta_content_sha256": sha, "completed_utc": utc(), "elapsed_seconds": time.time()-t0,
                         "command": cmd})
        return {"Genome ID": gid, "status": "PASS", "fasta_path": str(fa), "fasta_content_sha256": sha, "message": ""}
    except Exception as e:
        return {"Genome ID": gid, "status": "FAIL", "fasta_path": "", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def sra_assembly_one(row: dict[str, str], *, workroot: Path, outdir: Path, done_dir: Path,
                     prefetch_cmd: list[str], fasterq_cmd: list[str], fastp_cmd: list[str], shovill_cmd: list[str],
                     cpus: int, ram_gb: int, assembler: str, timeout: int, keep_fastq: bool, force: bool) -> dict[str, Any]:
    """
    Checkpointed SRA -> clean reads -> Shovill assembly.

    v2.0.1 resource/resume fixes:
      - Shovill requires --ram >= 8 GB; validated before invocation.
      - If fastp-clean reads already exist, prefetch/fasterq are NOT repeated.
      - After fastp succeeds, raw FASTQ + prefetch payload are deleted when
        --keep-fastq is not requested. Clean compressed reads are sufficient to
        retry Shovill.
      - fasterq temporary space is always removed, including on failure.
      - After final assembly checkpoint is written, the per-SRA work directory
        is removed when --keep-fastq is not requested.
    """
    gid = row["Genome ID"]; sra = normid(row.get("SRA Run", ""))
    s = safe(gid)
    finalfa = outdir / f"{s}.fna"
    finaldone = done_dir / f"{s}.assembly.done.json"

    if not force and finalfa.is_file() and finaldone.is_file():
        try:
            z = json.loads(finaldone.read_text())
            if (
                z.get("status") == "PASS"
                and z.get("sra_run") == sra
                and z.get("assembler") == assembler
                and z.get("fasta_content_sha256") == fasta_content_sha(finalfa)
            ):
                return {
                    "Genome ID": gid,
                    "status": "SKIP_CHECKPOINT",
                    "fasta_path": str(finalfa),
                    "fasta_content_sha256": z["fasta_content_sha256"],
                    "message": "",
                }
        except Exception:
            pass

    if not sra:
        return {"Genome ID": gid, "status": "FAIL", "fasta_path": "", "message": "missing SRA Run"}
    if ram_gb < 8:
        return {
            "Genome ID": gid,
            "status": "FAIL",
            "fasta_path": "",
            "message": f"assembly_ram_gb={ram_gb}; Shovill 1.4.2 requires --ram >= 8",
        }

    w = workroot / s
    w.mkdir(parents=True, exist_ok=True)
    prefdir = w / "prefetch"
    rawdir = w / "raw"
    cleandir = w / "clean"
    shodir = w / "shovill"
    for x in [prefdir, rawdir, cleandir]:
        x.mkdir(parents=True, exist_ok=True)

    pj = w / "01_prefetch.done.json"
    fj = w / "02_fasterq.done.json"
    fpj = w / "03_fastp.done.json"
    sj = w / "04_shovill.done.json"

    sra_file = prefdir / sra / f"{sra}.sra"
    r1 = rawdir / f"{sra}_1.fastq"
    r2 = rawdir / f"{sra}_2.fastq"
    c1 = cleandir / f"{sra}_R1.clean.fastq.gz"
    c2 = cleandir / f"{sra}_R2.clean.fastq.gz"
    fpjson = cleandir / f"{sra}.fastp.json"
    fphtml = cleandir / f"{sra}.fastp.html"
    contigs = shodir / "contigs.fa"

    t0 = time.time()

    def clean_reads_ready() -> bool:
        return (
            fpj.is_file()
            and c1.is_file() and c1.stat().st_size > 0
            and c2.is_file() and c2.stat().st_size > 0
        )

    try:
        # If clean reads survived the previous failed run, they are the correct
        # restart point. Do not waste disk by reconstructing raw FASTQ.
        if not force and clean_reads_ready():
            log(f"[SRA-STEP] {gid} | reuse clean FASTQ | assembler={assembler}")
            if not keep_fastq:
                shutil.rmtree(rawdir, ignore_errors=True)
                shutil.rmtree(prefdir, ignore_errors=True)
                shutil.rmtree(w / "fasterq_tmp", ignore_errors=True)
        else:
            # 1) prefetch
            if force or not (pj.is_file() and sra_file.is_file() and sra_file.stat().st_size > 0):
                log(f"[SRA-STEP] {gid} | prefetch START | {sra}")
                run_checked(
                    prefetch_cmd + [sra, "--output-directory", str(prefdir), "--max-size", "100G"],
                    timeout=timeout,
                )
                if not sra_file.is_file():
                    hits = list(prefdir.rglob(f"{sra}.sra"))
                    if len(hits) != 1:
                        raise RuntimeError(f"prefetch did not yield one {sra}.sra")
                    sra_file = hits[0]
                atomic_json(
                    pj,
                    {
                        "status": "PASS",
                        "sra_run": sra,
                        "path": str(sra_file),
                        "bytes": sra_file.stat().st_size,
                        "completed_utc": utc(),
                    },
                )
                log(f"[SRA-STEP] {gid} | prefetch PASS | {sra}")

            # 2) fasterq paired split
            if force or not (
                fj.is_file()
                and r1.is_file() and r1.stat().st_size > 0
                and r2.is_file() and r2.stat().st_size > 0
            ):
                for q in rawdir.glob(f"{sra}*.fastq"):
                    q.unlink(missing_ok=True)
                tmpq = w / "fasterq_tmp"
                shutil.rmtree(tmpq, ignore_errors=True)
                tmpq.mkdir(parents=True, exist_ok=True)
                log(f"[SRA-STEP] {gid} | fasterq START | {sra}")
                try:
                    run_checked(
                        fasterq_cmd
                        + [
                            str(sra_file),
                            "--split-files",
                            "-e", str(cpus),
                            "-O", str(rawdir),
                            "-t", str(tmpq),
                        ],
                        timeout=timeout,
                    )
                finally:
                    # Partial fasterq temp can be enormous. Never retain it.
                    shutil.rmtree(tmpq, ignore_errors=True)

                if not (
                    r1.is_file() and r1.stat().st_size > 0
                    and r2.is_file() and r2.stat().st_size > 0
                ):
                    produced = [p.name for p in rawdir.glob("*.fastq")]
                    raise RuntimeError(f"paired FASTQ not produced for {sra}; files={produced}")

                atomic_json(
                    fj,
                    {
                        "status": "PASS",
                        "sra_run": sra,
                        "r1": str(r1),
                        "r2": str(r2),
                        "completed_utc": utc(),
                    },
                )
                log(f"[SRA-STEP] {gid} | fasterq PASS | {sra}")

            # 3) fastp defaults; phenotype-independent
            if force or not clean_reads_ready():
                log(f"[SRA-STEP] {gid} | fastp START | {sra}")
                run_checked(
                    fastp_cmd
                    + [
                        "-i", str(r1),
                        "-I", str(r2),
                        "-o", str(c1),
                        "-O", str(c2),
                        "--thread", str(cpus),
                        "--json", str(fpjson),
                        "--html", str(fphtml),
                    ],
                    timeout=timeout,
                )
                atomic_json(
                    fpj,
                    {
                        "status": "PASS",
                        "sra_run": sra,
                        "r1": str(c1),
                        "r2": str(c2),
                        "completed_utc": utc(),
                    },
                )
                log(f"[SRA-STEP] {gid} | fastp PASS | {sra}")

            # Once clean compressed reads exist, raw/prefetch are unnecessary
            # for the assembly retry path and consume substantial disk.
            if not keep_fastq and clean_reads_ready():
                shutil.rmtree(rawdir, ignore_errors=True)
                shutil.rmtree(prefdir, ignore_errors=True)
                shutil.rmtree(w / "fasterq_tmp", ignore_errors=True)

        # 4) Shovill assembly (protocol-selected engine)
        shovill_ckpt_ok = False
        if not force and sj.is_file() and contigs.is_file() and contigs.stat().st_size > 0:
            try:
                zsj = json.loads(sj.read_text())
                shovill_ckpt_ok = (
                    zsj.get("status") == "PASS"
                    and zsj.get("sra_run") == sra
                    and zsj.get("assembler") == assembler
                )
            except Exception:
                shovill_ckpt_ok = False

        if not shovill_ckpt_ok:
            shutil.rmtree(shodir, ignore_errors=True)
            log(f"[SRA-STEP] {gid} | shovill/{assembler} START | cpus={cpus} ram={ram_gb}GB")
            cmd = shovill_cmd + [
                "--R1", str(c1),
                "--R2", str(c2),
                "--outdir", str(shodir),
                "--cpus", str(cpus),
                "--ram", str(ram_gb),
                "--assembler", assembler,
            ]
            run_checked(cmd, timeout=timeout)
            if not contigs.is_file() or contigs.stat().st_size == 0:
                raise RuntimeError("Shovill completed without non-empty contigs.fa")
            atomic_json(
                sj,
                {
                    "status": "PASS",
                    "sra_run": sra,
                    "assembler": assembler,
                    "contigs": str(contigs),
                    "command": cmd,
                    "completed_utc": utc(),
                },
            )
            log(f"[SRA-STEP] {gid} | shovill/{assembler} PASS")

        outdir.mkdir(parents=True, exist_ok=True)
        tmp = finalfa.with_name(finalfa.name + f".tmp.{os.getpid()}")
        shutil.copy2(contigs, tmp)
        os.replace(tmp, finalfa)
        sha = fasta_content_sha(finalfa)
        atomic_json(
            finaldone,
            {
                "status": "PASS",
                "Genome ID": gid,
                "sra_run": sra,
                "assembler": assembler,
                "fasta_path": str(finalfa),
                "fasta_content_sha256": sha,
                "completed_utc": utc(),
                "elapsed_seconds": time.time() - t0,
            },
        )

        if not keep_fastq:
            # Final FASTA + finaldone are outside w, so the entire scratch tree
            # can be removed safely after the durable checkpoint is written.
            shutil.rmtree(w, ignore_errors=True)

        return {
            "Genome ID": gid,
            "status": "PASS",
            "fasta_path": str(finalfa),
            "fasta_content_sha256": sha,
            "message": "",
        }

    except Exception as e:
        # Keep clean reads on Shovill failure so the retry does not redownload or
        # reconstruct raw FASTQ. Remove only disposable temp.
        shutil.rmtree(w / "fasterq_tmp", ignore_errors=True)
        return {
            "Genome ID": gid,
            "status": "FAIL",
            "fasta_path": "",
            "message": f"{type(e).__name__}: {e}",
        }


def acquire_or_assemble_all(pre: pd.DataFrame, *, root: Path, cp: Path, fd: Path,
                            assembly_env: str, download_workers: int, assembly_workers: int,
                            assembly_cpus: int, assembly_ram_gb: int, sra_assembler: str, timeout: int,
                            keep_fastq: bool, force: bool) -> pd.DataFrame:
    pub = pre[pre["Assembly Accession"].map(normid).ne("")].copy()
    sra = pre[pre["Assembly Accession"].map(normid).eq("")].copy()
    if sra["SRA Run"].map(normid).eq("").any():
        bad = sra.loc[sra["SRA Run"].map(normid).eq(""), "Genome ID"].tolist()
        raise RuntimeError(f"No assembly and no SRA for strict E3 candidates: {bad}")

    datasets_cmd = tool_prefix("datasets", assembly_env)
    prefetch_cmd = tool_prefix("prefetch", assembly_env)
    fasterq_cmd = tool_prefix("fasterq-dump", assembly_env)
    fastp_cmd = tool_prefix("fastp", assembly_env)
    shovill_cmd = tool_prefix("shovill", assembly_env)
    log(f"[TOOLS] datasets={command_version(datasets_cmd)}")
    log(f"[TOOLS] prefetch={command_version(prefetch_cmd)}")
    log(f"[TOOLS] fasterq={command_version(fasterq_cmd)}")
    log(f"[TOOLS] fastp={command_version(fastp_cmd)}")
    log(f"[TOOLS] shovill={command_version(shovill_cmd)}")

    pubout = fd / "public_assembly_fastas"; pubdone = cp / "public_assembly_done"
    asmout = fd / "sra_assembled_fastas"; asmdone = cp / "sra_assembly_done"; workroot = cp / "sra_work"
    for x in [pubout, pubdone, asmout, asmdone, workroot]: x.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    # Public assemblies
    pstart = time.time()
    pubrecords = pub.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, min(download_workers, 6))) as ex:
        fut = {ex.submit(download_public_assembly_one, r, pubout, pubdone, datasets_cmd, timeout, force): r["Genome ID"] for r in pubrecords}
        for f in as_completed(fut):
            rr = f.result(); rows.append(rr); k = sum(1 for x in rows if x["Genome ID"] in set(pub["Genome ID"]))
            if rr["status"] == "FAIL": log(f"[FAIL] public FASTA {rr['Genome ID']} | {rr['message']}")
            if k == 1 or k % 25 == 0 or k == len(pubrecords):
                elapsed = time.time()-pstart; rate=k/max(elapsed,1e-9); eta=(len(pubrecords)-k)/rate if rate else math.nan
                log(f"[PROGRESS] public assemblies {k}/{len(pubrecords)} | elapsed={elapsed/3600:.2f}h ETA={(eta/3600 if math.isfinite(eta) else float('nan')):.2f}h")

    badpub = [x for x in rows if x["status"] == "FAIL"]
    if badpub:
        atomic_csv(cp / "file14_public_assembly_failures.csv", pd.DataFrame(badpub))
        raise RuntimeError(f"{len(badpub)} public assembly downloads failed; checkpoints preserved")

    # SRA assemblies
    # Resume optimization: process candidates with already-clean FASTQs first.
    # This changes execution order only; it does NOT change cohort membership,
    # labels, QC, assembler, or any downstream scientific rule.
    astart = time.time(); srarows: list[dict[str, Any]] = []
    srecords = sra.to_dict("records")

    def _clean_resume_ready(r: dict[str, str]) -> bool:
        gid = r["Genome ID"]; sr = normid(r.get("SRA Run", ""))
        w = workroot / safe(gid)
        c = w / "clean"
        return (
            (w / "03_fastp.done.json").is_file()
            and (c / f"{sr}_R1.clean.fastq.gz").is_file()
            and (c / f"{sr}_R1.clean.fastq.gz").stat().st_size > 0
            and (c / f"{sr}_R2.clean.fastq.gz").is_file()
            and (c / f"{sr}_R2.clean.fastq.gz").stat().st_size > 0
        )

    ready = [r for r in srecords if _clean_resume_ready(r)]
    cold = [r for r in srecords if not _clean_resume_ready(r)]
    srecords = ready + cold
    log(
        f"[RESUME] SRA scheduling | clean-ready={len(ready)} | "
        f"needs earlier step={len(cold)} | clean-ready processed first"
    )

    with ThreadPoolExecutor(max_workers=max(1, assembly_workers)) as ex:
        fut = {ex.submit(sra_assembly_one, r, workroot=workroot, outdir=asmout, done_dir=asmdone,
                         prefetch_cmd=prefetch_cmd, fasterq_cmd=fasterq_cmd, fastp_cmd=fastp_cmd, shovill_cmd=shovill_cmd,
                         cpus=assembly_cpus, ram_gb=assembly_ram_gb, assembler=sra_assembler,
                         timeout=timeout, keep_fastq=keep_fastq, force=force): r["Genome ID"] for r in srecords}
        for f in as_completed(fut):
            rr = f.result(); srarows.append(rr); rows.append(rr); k=len(srarows)
            if rr["status"] == "FAIL": log(f"[FAIL] SRA assembly {rr['Genome ID']} | {rr['message']}")
            if k == 1 or k % 5 == 0 or k == len(srecords):
                elapsed=time.time()-astart; rate=k/max(elapsed,1e-9); eta=(len(srecords)-k)/rate if rate else math.nan
                c=Counter(x["status"] for x in srarows)
                log(f"[PROGRESS] SRA assembly {k}/{len(srecords)} | PASS={c['PASS']} SKIP={c['SKIP_CHECKPOINT']} FAIL={c['FAIL']} | elapsed={elapsed/3600:.2f}h ETA={(eta/3600 if math.isfinite(eta) else float('nan')):.2f}h")
                atomic_csv(cp / "file14_sra_assembly_status.csv", pd.DataFrame(srarows))

    bad = [x for x in srarows if x["status"] == "FAIL"]
    if bad:
        atomic_csv(cp / "file14_sra_assembly_failures.csv", pd.DataFrame(bad))
        raise RuntimeError(f"{len(bad)} SRA assemblies failed; fix cause and rerun identical command")

    z = pd.DataFrame(rows)
    if len(z) != len(pre) or z["Genome ID"].nunique() != len(pre):
        raise RuntimeError(f"FASTA acquisition/assembly coverage mismatch: rows={len(z)} unique={z['Genome ID'].nunique()} expected={len(pre)}")
    z = pre[["Genome ID"]].merge(z, on="Genome ID", validate="one_to_one")
    z["bytes"] = [Path(p).stat().st_size for p in z.fasta_path]
    z["mtime_ns"] = [Path(p).stat().st_mtime_ns for p in z.fasta_path]
    return z


def apply_frozen_e3_qc(pre: pd.DataFrame, fm: pd.DataFrame, envelope_path: Path, *, cp: Path, fd: Path,
                        min_total: int, min_r: int, min_s: int, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    env = json.loads(envelope_path.read_text())
    hard = env["hard_pass_fail_envelope"]
    warn = env.get("robust_warning_outer_fences", {})
    mcp = cp / "e3_sequence_metrics_per_genome"; mcp.mkdir(parents=True, exist_ok=True)
    fidx = fm.set_index("Genome ID")

    def one(gid: str) -> dict[str, Any]:
        fa = Path(fidx.loc[gid, "fasta_path"]); fsha = str(fidx.loc[gid, "fasta_content_sha256"])
        jp = mcp / f"{safe(gid)}.json"
        if jp.is_file():
            try:
                z = json.loads(jp.read_text())
                if z.get("fasta_content_sha256") == fsha and z.get("envelope_sha256") == shafile(envelope_path):
                    return z
            except Exception: pass
        m = sequence_metrics(fa)
        reasons=[]
        if not (hard["assembly_length_bp"]["min"] <= m["assembly_length_bp"] <= hard["assembly_length_bp"]["max"]): reasons.append("assembly_length_outside_reference_range")
        if m["contig_count"] > hard["contig_count"]["max"]: reasons.append("contig_count_above_reference_max")
        if m["n50_bp"] < hard["n50_bp"]["min"]: reasons.append("n50_below_reference_min")
        if m["largest_contig_bp"] < hard["largest_contig_bp"]["min"]: reasons.append("largest_contig_below_reference_min")
        if m["ambiguous_fraction"] > hard["ambiguous_fraction"]["max"]: reasons.append("ambiguous_fraction_above_reference_max")
        warnings=[]
        for key in ["assembly_length_bp","contig_count","n50_bp","largest_contig_bp","ambiguous_fraction"]:
            if key in warn:
                lo=warn[key].get("outer_lower", -float("inf")); hi=warn[key].get("outer_upper", float("inf")); val=m[key]
                if val < lo or val > hi: warnings.append(key)
        z={"Genome ID":gid,"fasta_path":str(fa),"fasta_content_sha256":fsha,"envelope_sha256":shafile(envelope_path),
           **m,"technical_qc_pass":len(reasons)==0,"technical_qc_reasons":";".join(reasons),"diagnostic_outer_fence_warnings":";".join(warnings)}
        atomic_json(jp,z); return z

    rows=[]; start=time.time(); gids=pre["Genome ID"].tolist()
    with ThreadPoolExecutor(max_workers=max(1,min(workers,8))) as ex:
        fut={ex.submit(one,g):g for g in gids}
        for f in as_completed(fut):
            rows.append(f.result()); k=len(rows)
            if k==1 or k%25==0 or k==len(gids):
                elapsed=time.time()-start; rate=k/max(elapsed,1e-9); eta=(len(gids)-k)/rate if rate else math.nan
                log(f"[PROGRESS] E3 technical QC {k}/{len(gids)} | elapsed={elapsed/60:.1f}m ETA={(eta/60 if math.isfinite(eta) else float('nan')):.1f}m")
    md=pd.DataFrame(rows)
    audit=pre.merge(fm[["Genome ID","fasta_path","fasta_content_sha256"]],on="Genome ID",validate="one_to_one").merge(md.drop(columns=["fasta_path","fasta_content_sha256"]),on="Genome ID",validate="one_to_one")
    auditp=fd/"file14_E3_technical_qc_audit.csv"; atomic_csv(auditp,audit)
    final=audit[audit.technical_qc_pass.astype(bool)].copy().reset_index(drop=True)
    final["FASTA Path"]=final["fasta_path"]
    y=final.Phenotype.map(phenotype_to_y).astype(int); n=len(final); nr=int(y.sum()); ns=int((1-y).sum())
    log(f"[QC] final technical-pass candidates: n={n} R={nr} S={ns} | excluded={len(audit)-n}")
    if n < min_total or nr < min_r or ns < min_s:
        raise RuntimeError(f"FINAL E3 gate failed after technical QC: n={n} R={nr} S={ns}; required n>={min_total},R>={min_r},S>={min_s}. QC must not be relaxed.")

    outcols=["Genome ID","Phenotype","Assembly Accession","BioSample","NCBI SNP/ERD Group","SRA Run","Source","FASTA Path","fasta_content_sha256",
             "assembly_length_bp","contig_count","n50_bp","l50","largest_contig_bp","ambiguous_bases","ambiguous_fraction","gc_fraction","contigs_ge_500","contigs_ge_1000","diagnostic_outer_fence_warnings"]
    finalmanifest=fd/"file14_E3_FINAL_FROZEN.csv"
    final[outcols].to_csv(finalmanifest,index=False)
    canon=canonical_e3_manifest(finalmanifest, True)
    ch=cohort_hash(canon)
    lockp=cp/"file14_final_e3_cohort_lock.json"
    lock={"status":"FINAL_E3_FROZEN_BEFORE_SCORING","created_utc":utc(),"source_preassembly_n":len(pre),"final_n":n,"R":nr,"S":ns,
          "technical_qc_audit_sha256":shafile(auditp),"envelope_sha256":shafile(envelope_path),"final_manifest":str(finalmanifest),"final_manifest_sha256":shafile(finalmanifest),"cohort_sha256":ch}
    if lockp.is_file():
        old=json.loads(lockp.read_text())
        for key in ["final_manifest_sha256","cohort_sha256","envelope_sha256"]:
            if old.get(key)!=lock.get(key): raise RuntimeError(f"Final E3 lock mismatch on {key}; use a new checkpoint dir for an intentionally different cohort")
        log(f"[PASS] Reused final E3 cohort lock | n={n} R={nr} S={ns} | sha={ch}")
    else:
        atomic_json(lockp,lock); log(f"[PASS] Final E3 frozen BEFORE scoring | n={n} R={nr} S={ns} | sha={ch}")
    return final, audit, ch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--checkpoint-dir", type=Path)
    ap.add_argument("--assembly-env", default="e3-assembly")
    ap.add_argument("--reference-workers", type=int, default=4)
    ap.add_argument("--download-workers", type=int, default=4)
    ap.add_argument("--assembly-workers", type=int, default=1)
    ap.add_argument("--assembly-cpus", type=int, default=4)
    ap.add_argument("--assembly-ram-gb", type=int, default=8)
    ap.add_argument("--sra-assembler", choices=["skesa", "spades", "megahit", "velvet"], default="skesa")
    ap.add_argument("--amrfinder-workers", type=int, default=8)
    ap.add_argument("--amrfinder-threads", type=int, default=1)
    ap.add_argument("--timeout-seconds", type=int, default=21600)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--min-total", type=int, default=300)
    ap.add_argument("--min-resistant", type=int, default=100)
    ap.add_argument("--min-susceptible", type=int, default=100)
    ap.add_argument("--keep-fastq", action="store_true")
    ap.add_argument("--force-acquisition", action="store_true")
    ap.add_argument("--force-reannotation", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a=ap.parse_args()
    if a.self_test: return selftest()

    root=a.project_root.resolve(); reqdir(root,"project root")
    cp=a.checkpoint_dir.resolve() if a.checkpoint_dir else root/"checkpoints/file14_final_e3_validation"
    fd=root/"data/external_validation/e3/file14_final"; ed=root/"data/evaluation/file14"
    raw=fd/"raw_amrfinder"; done=cp/"per_genome_amrfinder_done"; tmp=cp/"tmp"
    for d in [cp,fd,ed,raw,done,tmp]: d.mkdir(parents=True,exist_ok=True)
    failjson=cp/"file14_failure.json"; summaryp=root/"results/FILE14_FINAL_SUMMARY.log"
    global LOG_FILE; LOG_FILE=root/"results/FILE14_FINAL_RUN.log"
    LOG_FILE.parent.mkdir(parents=True,exist_ok=True)

    current_stage="startup"
    try:
        log("="*112); log("FILE14 — CONSOLIDATED E3 ASSEMBLY/QC/FREEZE/BLIND VALIDATION"); log("="*112)
        log(f"Version       : {VERSION}"); log(f"Project root  : {root}"); log(f"Checkpoint dir: {cp}")
        log("Resume        : ENABLED at reference-genome, public-assembly, SRA-step, E3-QC, AMRFinder, projection, bootstrap stages")
        log("E3 performance-based selection/tuning/recalibration: PROHIBITED")
        write_status_summary(summaryp,stage=current_stage,status="RUNNING",next_step="Preflight and reference/QC freeze.")

        current_stage="Stage 0/10 — preflight"
        log(current_stage)

        # Resource guardrails for Shovill 1.4.2.
        if a.assembly_ram_gb < 8:
            raise RuntimeError(
                f"--assembly-ram-gb={a.assembly_ram_gb} is invalid: "
                "Shovill 1.4.2 requires at least 8 GB."
            )
        try:
            mem_kb = 0
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        break
            mem_gb = mem_kb / 1024 / 1024 if mem_kb else 0.0
        except Exception:
            mem_gb = 0.0
        if mem_gb and a.assembly_workers * a.assembly_ram_gb > mem_gb * 0.90:
            raise RuntimeError(
                f"Assembly concurrency requests about "
                f"{a.assembly_workers * a.assembly_ram_gb} GB "
                f"({a.assembly_workers} workers x {a.assembly_ram_gb} GB), "
                f"but WSL reports only {mem_gb:.2f} GB RAM. "
                "Use one assembly worker on this machine."
            )

        pre, prep, buildaudit=load_strict_preassembly(root)
        f13=load_file13_freeze(root)
        primary_feats=load_schema(f13["primary_schema_path"])
        sens_feats=load_schema(f13["sensitivity_schema_path"]) if f13["sensitivity_schema_path"] else None
        # Fail early on external tools before overnight work.
        tool_versions={}
        for tool in ["datasets","prefetch","fasterq-dump","fastp","shovill"]:
            pfx=tool_prefix(tool,a.assembly_env); tool_versions[tool]=command_version(pfx)
            if tool_versions[tool].startswith("ERROR"):
                raise RuntimeError(f"Tool preflight failed: {tool}: {tool_versions[tool]}")
        disk = shutil.disk_usage(cp)
        log(
            f"[PASS] preflight | strict preassembly n={len(pre)} | "
            f"FILE13 model/schema hashes verified | external tools resolved | "
            f"assembly={a.assembly_workers}x{a.assembly_cpus}CPU/{a.assembly_ram_gb}GB/{a.sra_assembler} | "
            f"checkpoint_free_disk={disk.free/(1024**3):.1f}GiB"
        )
        write_status_summary(summaryp,stage=current_stage,status="RUNNING",next_step="Freeze/reuse 4,227-development sequence compatibility reference.",extra={"preassembly_n":len(pre),"tool_versions":json.dumps(tool_versions)})
        if a.preflight_only:
            log("STATUS: PASS_PREFLIGHT_ONLY"); return 0

        current_stage="Stage 1/10 — freeze/reuse 4,227 development sequence reference"
        log(current_stage)
        envp,env=ensure_reference_envelope(root,cp,a.reference_workers)
        write_status_summary(summaryp,stage=current_stage,status="RUNNING",next_step="Acquire 251 public assemblies and assemble 55 SRA candidates with resume using the predeclared SRA assembler.",extra={"envelope_sha256":shafile(envp)})

        current_stage="Stage 2/10 — acquire public FASTAs + assemble SRA queue"
        log(current_stage)
        fm=acquire_or_assemble_all(pre,root=root,cp=cp,fd=fd,assembly_env=a.assembly_env,download_workers=a.download_workers,
                                   assembly_workers=a.assembly_workers,assembly_cpus=a.assembly_cpus,assembly_ram_gb=a.assembly_ram_gb,
                                   sra_assembler=a.sra_assembler,
                                   timeout=a.timeout_seconds,keep_fastq=a.keep_fastq,force=a.force_acquisition)
        fmp=fd/"file14_E3_all_preQC_fasta_manifest.csv"; atomic_csv(fmp,fm); fmsha=shafile(fmp)
        log(f"[PASS] all strict preassembly FASTAs available | n={len(fm)} | sha={fmsha}")
        write_status_summary(summaryp,stage=current_stage,status="RUNNING",next_step="Apply frozen technical sequence compatibility envelope to all E3 FASTAs.",extra={"fasta_manifest_sha256":fmsha})

        current_stage="Stage 3/10 — apply frozen technical QC and freeze FINAL E3"
        log(current_stage)
        finalfull,qcaudit,ch=apply_frozen_e3_qc(pre,fm,envp,cp=cp,fd=fd,min_total=a.min_total,min_r=a.min_resistant,min_s=a.min_susceptible,workers=a.reference_workers)
        finalmanifest=fd/"file14_E3_FINAL_FROZEN.csv"
        e3=canonical_e3_manifest(finalmanifest,True)
        n=len(e3); nr=int(e3.y.sum()); ns=int((e3.y==0).sum())
        write_status_summary(summaryp,stage=current_stage,status="RUNNING",next_step="Harmonized AMRFinder annotation using frozen FILE13 protocol.",extra={"final_E3_n":n,"R":nr,"S":ns,"cohort_sha256":ch})

        current_stage="Stage 4/10 — verify FILE13 freeze again + annotation protocol lock"
        log(current_stage)
        f13=load_file13_freeze(root)
        if len(primary_feats)!=int(f13["manifest"]["primary_feature_schema"]["feature_count"]): raise RuntimeError("FILE13 primary feature count mismatch")
        p13=f13["protocol_lock"]; amr=Path(p13["amrfinder"]["executable"]).expanduser().resolve(); db=Path(p13["amrfinder"]["database"]).expanduser().resolve()
        reqfile(amr,"AMRFinder executable from FILE13"); reqdir(db,"AMRFinder DB from FILE13")
        ver=amrfinder_version(amr); expected_ver=str(p13["amrfinder"].get("version_text",""))
        if expected_ver and expected_ver.split()[0] not in ver and "4.2.7" not in ver: raise RuntimeError(f"AMRFinder version differs from FILE13: {expected_ver!r} vs {ver!r}")
        finalfm=finalfull[["Genome ID"]].merge(fm,on="Genome ID",validate="one_to_one")
        finalfmp=fd/"file14_E3_FINAL_fasta_manifest.csv"
        atomic_csv(finalfmp,finalfm)
        finalfm_file_sha=shafile(finalfmp)

        # Scientific FASTA identity must be invariant to resume/runtime metadata
        # such as status, message, bytes timestamps, or checkpoint state.
        stable_fasta_rows=(
            finalfm[["Genome ID","fasta_content_sha256"]]
            .astype(str)
            .sort_values("Genome ID",kind="stable")
            .to_dict("records")
        )
        finalfm_content_sha=shatext(
            json.dumps(stable_fasta_rows,sort_keys=True,separators=(",",":"))
        )

        ann_obj={"file":"FILE14","file13_protocol_sha256":f13["protocol_sha256"],"amrfinder_executable":str(amr),"amrfinder_version_text":ver,
                 "database":str(db),"organism":"Klebsiella_pneumoniae","mode":"nucleotide","threads_per_genome":a.amrfinder_threads,
                 "parser":{"Type":"AMR","Subtype_in":sorted(SUBTYPES),"symbol":"Element symbol"},"e3_cohort_sha256":ch,
                 "fasta_content_manifest_sha256":finalfm_content_sha}
        ann_sha=shatext(json.dumps(ann_obj,sort_keys=True,separators=(",",":"))); ann_lock=cp/"file14_annotation_protocol_lock.json"
        if ann_lock.is_file():
            old=json.loads(ann_lock.read_text());
            if old.get("protocol_sha256")!=ann_sha: raise RuntimeError("FILE14 annotation protocol changed after checkpoint")
        else:
            atomic_json(ann_lock,{
                **ann_obj,
                "fasta_manifest_file_sha256_audit":finalfm_file_sha,
                "protocol_sha256":ann_sha,
                "locked_utc":utc()
            })
        log(f"[PASS] FILE13 + annotation protocol verified | model={f13['primary_model_sha256']} | schema={f13['primary_schema_sha256']}")

        current_stage="Stage 5/10 — harmonized AMRFinder annotation (per-genome resume)"
        log(current_stage)
        annotate_e3(e3,finalfm,raw,done,tmp,cp,amr,db,a.amrfinder_workers,a.amrfinder_threads,f13["protocol_sha256"],ann_sha,a.timeout_seconds,2,a.force_reannotation)
        log("[PASS] E3 AMRFinder complete/checkpointed")
        write_status_summary(summaryp,stage=current_stage,status="RUNNING",next_step="Project calls onto frozen 702/668 schemas.",extra={"annotation_protocol_sha256":ann_sha})

        current_stage="Stage 6/10 — project E3 calls onto frozen FILE13 schemas"
        log(current_stage)
        prim_mat_path=fd/"file14_e3_primary_702_matrix.csv.gz"; sens_mat_path=fd/"file14_e3_sensitivity_668_matrix.csv.gz"; parse_summary_path=fd/"file14_e3_projection_summary.json"; out_schema_path=fd/"file14_e3_out_of_schema_determinants.csv"
        if prim_mat_path.is_file() and parse_summary_path.is_file():
            ps=json.loads(parse_summary_path.read_text())
            if ps.get("e3_cohort_sha256")==ch and ps.get("primary_schema_sha256")==f13["primary_schema_sha256"]:
                pm=pd.read_csv(prim_mat_path,compression="gzip",dtype={"Genome ID":str});
                if len(pm)!=n or list(pm.columns[1:])!=primary_feats: raise RuntimeError("Cached primary projection failed validation")
                Xp=pm[primary_feats].to_numpy(np.uint8); Xs=None
                if sens_feats and sens_mat_path.is_file():
                    sm=pd.read_csv(sens_mat_path,compression="gzip",dtype={"Genome ID":str});
                    if len(sm)==n and list(sm.columns[1:])==sens_feats: Xs=sm[sens_feats].to_numpy(np.uint8)
                log(f"[PASS] reused cached projection | primary={Xp.shape}")
            else: raise RuntimeError("Cached projection does not match final E3/schema")
        else:
            Xp,Xs,oos,call_counts=project_calls(e3["Genome ID"].tolist(),raw,primary_feats,sens_feats)
            pm=pd.DataFrame(Xp,columns=primary_feats); pm.insert(0,"Genome ID",e3["Genome ID"].tolist()); atomic_csv(prim_mat_path,pm,compression="gzip")
            if Xs is not None and sens_feats:
                sm=pd.DataFrame(Xs,columns=sens_feats); sm.insert(0,"Genome ID",e3["Genome ID"].tolist()); atomic_csv(sens_mat_path,sm,compression="gzip")
            ood=pd.DataFrame([{"determinant":k,"genome_count":v,"prevalence":v/n} for k,v in oos.items()]).sort_values(["genome_count","determinant"],ascending=[False,True]) if oos else pd.DataFrame(columns=["determinant","genome_count","prevalence"]); atomic_csv(out_schema_path,ood)
            ps={"e3_cohort_sha256":ch,"primary_schema_sha256":f13["primary_schema_sha256"],"primary_feature_count":len(primary_feats),"sensitivity_feature_count":len(sens_feats or []),"zero_primary_feature_genomes":int((Xp.sum(1)==0).sum()),"out_of_schema_unique_determinants":len(oos),"median_total_calls":float(np.median(call_counts))}; atomic_json(parse_summary_path,ps)
            log(f"[PASS] projected final E3 | primary={Xp.shape} | zero-feature={ps['zero_primary_feature_genomes']}")

        current_stage="Stage 7/10 — BLIND score frozen FILE13 model(s), then lock scores"
        log(current_stage)
        score_lock_path=cp/"file14_blind_score_lock.json"; blind_primary=ed/"file14_E3_blind_primary_scores.csv"
        p_primary,thr_primary,_=score_bundle(f13["primary_model_path"],primary_feats,Xp)
        blind=pd.DataFrame({"Genome ID":e3["Genome ID"],"probability_R":p_primary,"predicted_R":(p_primary>=thr_primary).astype(int)}); atomic_csv(blind_primary,blind)
        score_obj={"status":"BLIND_SCORES_FROZEN_BEFORE_METRICS","created_utc":utc(),"e3_cohort_sha256":ch,"file13_protocol_sha256":f13["protocol_sha256"],"primary_model_sha256":f13["primary_model_sha256"],"primary_schema_sha256":f13["primary_schema_sha256"],"threshold":thr_primary,"blind_primary_scores":str(blind_primary),"blind_primary_scores_sha256":shafile(blind_primary),"no_tuning":True,"no_recalibration":True,"no_model_reselection":True}
        p_sens=None; thr_sens=None
        if f13["sensitivity_model_path"] and sens_feats and Xs is not None:
            p_sens,thr_sens,_=score_bundle(f13["sensitivity_model_path"],sens_feats,Xs); blind_sens=ed/"file14_E3_blind_sensitivity_scores.csv"; atomic_csv(blind_sens,pd.DataFrame({"Genome ID":e3["Genome ID"],"probability_R":p_sens,"predicted_R":(p_sens>=thr_sens).astype(int)})); score_obj.update({"sensitivity_model_sha256":shafile(f13["sensitivity_model_path"]),"blind_sensitivity_scores":str(blind_sens),"blind_sensitivity_scores_sha256":shafile(blind_sens)})
        if score_lock_path.is_file():
            old=json.loads(score_lock_path.read_text())

            # Scientific identity must remain exact for cohort/model/schema/threshold.
            for k in ["e3_cohort_sha256","primary_model_sha256","primary_schema_sha256","threshold"]:
                if old.get(k)!=score_obj.get(k):
                    raise RuntimeError(f"Blind score lock mismatch on {k}")

            if old.get("sensitivity_model_sha256") is not None and score_obj.get("sensitivity_model_sha256") is not None:
                if old.get("sensitivity_model_sha256") != score_obj.get("sensitivity_model_sha256"):
                    raise RuntimeError("Blind score lock mismatch on sensitivity_model_sha256")

            if old.get("blind_primary_scores_sha256") == score_obj.get("blind_primary_scores_sha256"):
                log("[PASS] existing blind score lock reproduced exactly")
            else:
                # Raw CSV SHA can differ from IEEE-754 machine-precision jitter.
                # Verify against predictions from the previous successful locked run.
                prevp=ed/"file14_E3_primary_predictions_labeled.csv"
                if not prevp.is_file():
                    raise RuntimeError(
                        "Blind score raw SHA changed and previous labeled predictions "
                        "are unavailable for numerical-equivalence verification"
                    )

                prev=pd.read_csv(
                    prevp,
                    dtype={"Genome ID":str},
                    keep_default_na=False,
                    usecols=["Genome ID","probability_R","predicted_R"],
                )
                cur=blind[["Genome ID","probability_R","predicted_R"]].copy()
                cur["Genome ID"]=cur["Genome ID"].astype(str)

                if prev["Genome ID"].tolist() != cur["Genome ID"].tolist():
                    raise RuntimeError("Blind score reproduction changed Genome ID/order")

                if not np.array_equal(
                    prev["predicted_R"].to_numpy(int),
                    cur["predicted_R"].to_numpy(int),
                ):
                    raise RuntimeError("Blind score reproduction changed predicted_R")

                diff=np.abs(
                    prev["probability_R"].to_numpy(float)
                    - cur["probability_R"].to_numpy(float)
                )
                maxdiff=float(diff.max()) if len(diff) else 0.0

                if (not np.isfinite(maxdiff)) or maxdiff > 1e-12:
                    raise RuntimeError(
                        f"Blind score reproduction differs beyond numerical tolerance: "
                        f"max_abs_diff={maxdiff:.17g}"
                    )

                # Preserve the original raw-byte lock before migrating its audit SHA.
                prep=score_lock_path.with_name(
                    score_lock_path.stem + ".PRE_NUMERIC_TOLERANCE.json"
                )
                if not prep.exists():
                    prep.write_bytes(score_lock_path.read_bytes())

                score_obj["previous_blind_primary_scores_sha256_audit"] = old.get(
                    "blind_primary_scores_sha256"
                )
                score_obj["numerical_reproduction_tolerance"] = 1e-12
                score_obj["numerical_reproduction_max_abs_diff"] = maxdiff
                score_obj["score_lock_note"] = (
                    "Raw CSV SHA changed only by machine-precision floating-point "
                    "reproduction; IDs/order and thresholded predictions identical."
                )

                atomic_json(score_lock_path,score_obj)
                log(
                    f"[PASS] blind scores numerically reproduced | "
                    f"max_abs_diff={maxdiff:.3e} <= 1e-12 | "
                    f"predicted_R identical | lock SHA audit migrated"
                )
        else:
            atomic_json(score_lock_path,score_obj)
            log(f"[PASS] blind scores frozen BEFORE metrics | sha={score_obj['blind_primary_scores_sha256']}")

        current_stage="Stage 8/10 — confirmatory metrics + cluster bootstrap + calibration"
        log(current_stage)
        y=e3.y.to_numpy(int)
        cluster=e3["NCBI SNP/ERD Group"].astype("string").fillna("").str.strip()
        unresolved=cluster.eq("") | cluster.str.upper().isin(["UNRESOLVED","NA","N/A","NAN","NONE"])
        cluster=cluster.astype(object)
        cluster.loc[unresolved]="UNRESOLVED::"+e3.loc[unresolved,"Genome ID"].astype(str)
        log(f"[BOOTSTRAP] clusters={cluster.nunique()} | unresolved_as_singletons={int(unresolved.sum())} | max_cluster_size={int(cluster.value_counts().max())}")
        pnt_primary=metrics(y,p_primary,thr_primary); ci,cs=calibration_intercept_slope(y,p_primary); pnt_primary.update({"calibration_intercept":ci,"calibration_slope":cs,"mean_predicted_R":float(np.mean(p_primary)),"observed_prevalence_R":float(np.mean(y))})
        bb=load_or_run_bootstrap(ed/"file14_E3_primary_cluster_bootstrap.csv.gz",y,p_primary,cluster,thr_primary,a.bootstrap_replicates,SEED+101,PRIMARY_NAME); bs=bootstrap_summary(pnt_primary,bb,PRIMARY_NAME)
        point_rows=[{"representation":PRIMARY_NAME,**pnt_primary}]; bsum_rows=[bs]
        if p_sens is not None:
            p2=metrics(y,p_sens,thr_sens); ci2,cs2=calibration_intercept_slope(y,p_sens); p2.update({"calibration_intercept":ci2,"calibration_slope":cs2,"mean_predicted_R":float(np.mean(p_sens)),"observed_prevalence_R":float(np.mean(y))}); b2=load_or_run_bootstrap(ed/"file14_E3_sensitivity_cluster_bootstrap.csv.gz",y,p_sens,cluster,thr_sens,a.bootstrap_replicates,SEED+202,SENS_NAME); point_rows.append({"representation":SENS_NAME,**p2}); bsum_rows.append(bootstrap_summary(p2,b2,SENS_NAME))
        atomic_csv(ed/"file14_E3_point_metrics.csv",pd.DataFrame(point_rows)); atomic_csv(ed/"file14_E3_cluster_bootstrap_summary.csv",pd.concat(bsum_rows,ignore_index=True)); labeled=e3[["Genome ID","Phenotype","y","Assembly Accession","BioSample","NCBI SNP/ERD Group","MLST"]].merge(blind,on="Genome ID",validate="one_to_one"); atomic_csv(ed/"file14_E3_primary_predictions_labeled.csv",labeled)
        log(f"[RESULT] PRIMARY E3 | AUROC={pnt_primary['roc_auc']:.4f} AP={pnt_primary['average_precision']:.4f} BA={pnt_primary['balanced_accuracy']:.4f} MCC={pnt_primary['mcc']:.4f} Brier={pnt_primary['brier']:.4f}")

        current_stage="Stage 9/10 — descriptive transportability diagnostics"
        log(current_stage)
        dev=pd.read_csv(root/"data/processed/file07_common_cohort.csv",dtype={"Genome ID":str,"MLST":str},keep_default_na=False); dev_mlst={normid(x) for x in dev.get("MLST",pd.Series(dtype=str)) if normid(x)}; sub=subgroup_metric_rows(e3,p_primary,thr_primary,dev_mlst); atomic_csv(ed/"file14_E3_mlst_subgroup_metrics.csv",sub)
        pred=(p_primary>=thr_primary).astype(int); err=e3[["Genome ID","Phenotype","y","MLST","Assembly Accession","BioSample","NCBI SNP/ERD Group"]].copy(); err["probability_R"]=p_primary; err["predicted_R"]=pred; err["error"]=err.predicted_R!=err.y; err["high_confidence_wrong"]=((err.y.eq(0)&err.probability_R.ge(.90))|(err.y.eq(1)&err.probability_R.le(.10))); atomic_csv(ed/"file14_E3_error_diagnostics.csv",err)
        devmatp=root/"data/features/known_amr_harmonized/file13/X_known_amr_harmonized_rebuilt.csv"
        if devmatp.is_file():
            dv=pd.read_csv(devmatp,usecols=primary_feats); dp=dv.mean(0).to_numpy(float); ep=Xp.mean(0); sh=pd.DataFrame({"feature":primary_feats,"development_prevalence":dp,"E3_prevalence":ep,"delta_E3_minus_development":ep-dp}); sh["abs_delta"]=sh.delta_E3_minus_development.abs(); sh=sh.sort_values(["abs_delta","feature"],ascending=[False,True]); atomic_csv(ed/"file14_E3_feature_prevalence_shift.csv",sh)
        diagnostics={"errors":int(err.error.sum()),"high_confidence_wrong":int(err.high_confidence_wrong.sum()),"note":"Descriptive only; not used to modify/reselect/recalibrate the frozen FILE13 model."}; atomic_json(ed/"file14_E3_diagnostics_summary.json",diagnostics)

        current_stage="Stage 10/10 — final integrity + summary/flag"
        log(current_stage)
        if shafile(f13["primary_model_path"])!=f13["primary_model_sha256"]: raise RuntimeError("FILE13 primary model changed during FILE14")
        if shafile(f13["primary_schema_path"])!=f13["primary_schema_sha256"]: raise RuntimeError("FILE13 primary schema changed during FILE14")
        if shafile(blind_primary)!=score_obj["blind_primary_scores_sha256"]: raise RuntimeError("Blind score file changed after score lock")
        final_status="PASS_E3_BLIND_EXTERNAL_VALIDATION"
        final={"file":"FILE14","script_version":VERSION,"status":final_status,"completed_utc":utc(),"E3":{"n":n,"R":nr,"S":ns,"cohort_sha256":ch,"final_manifest":str(finalmanifest),"final_manifest_sha256":shafile(finalmanifest)},"preassembly":{"manifest":str(prep),"manifest_sha256":shafile(prep)},"technical_qc":{"envelope":str(envp),"envelope_sha256":shafile(envp),"audit_sha256":shafile(fd/"file14_E3_technical_qc_audit.csv")},"FILE13":{"protocol_sha256":f13["protocol_sha256"],"primary_model_sha256":f13["primary_model_sha256"],"primary_schema_sha256":f13["primary_schema_sha256"],"freeze_manifest_sha256":f13["freeze_manifest_sha256"]},"fasta_manifest_sha256":finalfm_file_sha,"fasta_content_manifest_sha256":finalfm_content_sha,"annotation_protocol_sha256":ann_sha,"blind_score_lock":score_obj,"primary_metrics":pnt_primary,"primary_bootstrap_summary":bs.to_dict("records"),"diagnostics":diagnostics,"guardrails":{"historical_development_qc_rewritten":False,"E3_used_for_tuning":False,"E3_used_for_reselection":False,"E3_used_for_recalibration":False,"decision_threshold_changed":False,"blind_scores_frozen_before_metrics":True},"next_step":"FILE15 integrity/reporting only."}
        finalp=cp/"file14_final_summary.json"; atomic_json(finalp,final); flag=cp/"E3_VALIDATION_COMPLETE.flag"; atomic_json(flag,{"status":final_status,"created_utc":utc(),"file14_final_summary":str(finalp),"file14_final_summary_sha256":shafile(finalp),"e3_cohort_sha256":ch,"file13_primary_model_sha256":f13["primary_model_sha256"],"blind_primary_scores_sha256":shafile(blind_primary)})
        failjson.unlink(missing_ok=True)
        write_status_summary(summaryp,stage=current_stage,status=final_status,next_step="FILE14 complete. Proceed to FILE15 integrity/reporting only.",extra={"final_E3_n":n,"R":nr,"S":ns,"cohort_sha256":ch,"AUROC":pnt_primary["roc_auc"],"AP":pnt_primary["average_precision"],"BA":pnt_primary["balanced_accuracy"],"MCC":pnt_primary["mcc"],"Brier":pnt_primary["brier"],"blind_score_sha256":shafile(blind_primary),"final_summary":str(finalp)})
        log("="*112); log(f"FILE14 STATUS : {final_status}"); log(f"E3             : n={n} R={nr} S={ns}"); log(f"E3 AUROC       : {pnt_primary['roc_auc']:.6f}"); log(f"E3 AP          : {pnt_primary['average_precision']:.6f}"); log(f"E3 BA          : {pnt_primary['balanced_accuracy']:.6f}"); log(f"E3 MCC         : {pnt_primary['mcc']:.6f}"); log(f"E3 Brier       : {pnt_primary['brier']:.6f}"); log(f"Final summary  : {finalp}"); log(f"Completion flag: {flag}"); log("="*112)
        return 0
    except KeyboardInterrupt:
        atomic_json(failjson,{"file":"FILE14","version":VERSION,"status":"INTERRUPTED","stage":current_stage,"time_utc":utc(),"traceback":traceback.format_exc()}); write_status_summary(summaryp,stage=current_stage,status="INTERRUPTED",next_step="Rerun the identical command. Per-genome/per-step checkpoints are preserved."); log("[INTERRUPTED] checkpoints preserved; rerun identical command"); return 130
    except Exception as e:
        atomic_json(failjson,{"file":"FILE14","version":VERSION,"status":"FAILED","stage":current_stage,"time_utc":utc(),"error_type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}); write_status_summary(summaryp,stage=current_stage,status="FAILED",next_step="Fix only the reported cause, then rerun the identical command. Do not change cohort/QC thresholds after performance inspection.",extra={"error":f"{type(e).__name__}: {e}","failure_json":str(failjson)}); log("="*112); log("FILE14 STATUS : FAILED"); log(f"Stage          : {current_stage}"); log(f"Error          : {type(e).__name__}: {e}"); log(f"Failure JSON   : {failjson}"); log("All completed checkpoints are preserved; rerun identical command after fixing the cause."); log("="*112); return 1


if __name__ == "__main__":
    raise SystemExit(main())
