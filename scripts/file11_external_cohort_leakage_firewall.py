#!/usr/bin/env python3
"""
FILE11 — INDEPENDENT EXTERNAL COHORT BUILDER + LEAKAGE FIREWALL
Version: 1.0.0

Goal
----
Build an external phenotype cohort from NCBI Pathogen Detection and keep it
separate from the development/training cohort.

This version DOES NOT retrain, tune, re-select features, or choose thresholds.
It freezes the current project state first, then builds:

  E1 = external-clean:
       phenotype-eligible NCBI isolates after exact identifier overlap removal

  E2 = external-strict:
       E1 after excluding NCBI Pathogen Detection SNP-cluster overlap with the
       internal cohort, when cluster mapping is available

Checkpointing
-------------
- Stage-level checkpoint files
- Per-internal-genome BV-BRC crosswalk cache
- NCBI raw query cache
- NCBI internal-cluster mapping cache
- Safe resume on rerun
- Refuses to silently reuse checkpoints if the run configuration changes

Live progress
-------------
Every potentially long stage prints [PROGRESS] lines with counts / percentages.

Default external scope
----------------------
Organism    : Klebsiella pneumoniae
Antibiotics : ertapenem, imipenem, meropenem

Phenotypes are NEVER pooled across antibiotics. Each antibiotic remains a
separate endpoint.

NCBI source
-----------
Preferred: NCBI Pathogen Detection BigQuery public tables:
  ncbi-pathogen-detect.pdbrowser.ast
  ncbi-pathogen-detect.pdbrowser.isolates

The script uses the `bq` command-line client when available/authenticated.
If BigQuery is unavailable, it writes the exact SQL query to disk and exits
cleanly with NEEDS_SOURCE_DATA. Export that query as CSV and rerun with:
  --source-csv /path/to/export.csv

Internal identifiers
--------------------
Pass --internal-manifest if possible. The manifest may contain any of:
  Genome ID / genome_id              (BV-BRC/PATRIC genome IDs)
  assembly accession / asm_acc
  BioSample / biosample_acc
  target_acc
  SRA / run accession
  erd_group                          (NCBI Pathogen Detection SNP cluster)

If only BV-BRC Genome IDs are present, FILE11 queries the BV-BRC genome API
and checkpoints each crosswalk to NCBI assembly/BioSample/SRA identifiers.

Scientific guardrail
--------------------
The external cohort is not used for model tuning. FILE11 v1.0.0 only freezes,
builds, audits, and exports the independent cohort. Blind prediction should be
performed only after the cohort and the original model/feature schema are frozen.
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
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests


VERSION = "1.0.0"
DESIGN_ID = "file11_external_cohort_leakage_firewall_v1"
NCBI_AST_TABLE = "ncbi-pathogen-detect.pdbrowser.ast"
NCBI_ISOLATES_TABLE = "ncbi-pathogen-detect.pdbrowser.isolates"
BVBRC_API = "https://www.bv-brc.org/api"


# --------------------------------------------------------------------------------------
# Logging / checkpoints
# --------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(label: str, i: int, n: int, extra: str = "", t0: Optional[float] = None) -> None:
    pct = 100.0 * i / max(1, n)
    eta = ""
    if t0 is not None and i > 0:
        elapsed = time.time() - t0
        remaining = max(0.0, elapsed / i * (n - i))
        eta = f" | elapsed={elapsed:.1f}s | ETA={remaining:.1f}s"
    suffix = f" | {extra}" if extra else ""
    log(f"[PROGRESS] {label}: {i}/{n} ({pct:5.1f}%){eta}{suffix}")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def stage_done_path(ckpt_dir: Path, stage: int) -> Path:
    return ckpt_dir / f"stage_{stage:02d}.done.json"


def mark_stage_done(ckpt_dir: Path, stage: int, payload: dict) -> None:
    atomic_json(stage_done_path(ckpt_dir, stage), payload)


def read_stage_done(ckpt_dir: Path, stage: int) -> Optional[dict]:
    p = stage_done_path(ckpt_dir, stage)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------

def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_upper(x) -> str:
    return norm_str(x).upper()


def first_existing_col(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lowered:
            return lowered[a.lower()]
    return None


ALIASES = {
    "genome_id": [
        "Genome ID", "genome_id", "genome id", "patric_genome_id", "bvbrc_genome_id"
    ],
    "target_acc": [
        "target_acc", "target accession", "isolate", "pathogen_detection_accession", "pdt"
    ],
    "biosample_acc": [
        "biosample_acc", "biosample", "BioSample", "biosample accession", "biosample_accession"
    ],
    "asm_acc": [
        "asm_acc", "assembly", "Assembly", "assembly accession", "assembly_accession",
        "genbank_assembly_accession"
    ],
    "sra_acc": [
        "sra_acc", "sra accession", "sra_accession", "run", "run_acc",
        "run accession", "sra_run", "sra_run_acc"
    ],
    "erd_group": [
        "erd_group", "snp_cluster", "snp cluster", "cluster", "ncbi_snp_cluster"
    ],
}


def canonical_identifier_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for key, aliases in ALIASES.items():
        c = first_existing_col(df, aliases)
        out[key] = df[c].map(norm_str) if c is not None else ""
    for c in out.columns:
        out[c] = out[c].replace({"nan": "", "None": "", "<NA>": ""})
    return out


def assembly_base(acc: str) -> str:
    acc = norm_upper(acc)
    return acc.split(".")[0] if acc else ""


def sra_tokens(v: str) -> List[str]:
    if not v:
        return []
    toks = re.split(r"[;,|\s]+", v.strip())
    return [x.upper() for x in toks if x]


def quote_sql_string(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def config_signature(args) -> str:
    payload = {
        "version": VERSION,
        "organism": args.organism,
        "antibiotics": sorted([x.strip().lower() for x in args.antibiotics.split(",") if x.strip()]),
        "include_intermediate_as_resistant": bool(args.include_intermediate_as_resistant),
        "min_per_class": int(args.min_per_class),
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


# --------------------------------------------------------------------------------------
# Stage 1 — freeze current project/model state
# --------------------------------------------------------------------------------------

MODEL_SUFFIXES = {
    ".joblib", ".pkl", ".pickle", ".onnx", ".pt", ".pth", ".keras", ".h5",
}
FREEZE_NAME_HINTS = (
    "model", "feature", "selected", "task", "split", "scaler", "encoder",
    "threshold", "hyperparam", "training", "train_", "cv_", "fold"
)


def discover_freeze_files(root: Path, ckpt_dir: Path, max_files: int = 2000) -> List[Path]:
    candidates: List[Path] = []
    search_roots = [root / "models", root / "checkpoints", root / "data"]
    for base in search_roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if len(candidates) >= max_files:
                break
            if not p.is_file():
                continue
            try:
                p.relative_to(ckpt_dir)
                continue
            except Exception:
                pass
            name = p.name.lower()
            # Freeze true model binaries and small configuration/schema artifacts.
            if p.suffix.lower() in MODEL_SUFFIXES:
                candidates.append(p)
            elif p.suffix.lower() in {".json", ".yaml", ".yml", ".txt", ".csv", ".tsv"}:
                if any(h in name for h in FREEZE_NAME_HINTS):
                    try:
                        if p.stat().st_size <= 25 * 1024 * 1024:
                            candidates.append(p)
                    except OSError:
                        pass
    return sorted(set(candidates))


def freeze_project_state(root: Path, ckpt_dir: Path) -> dict:
    manifest_path = ckpt_dir / "file11_pre_external_freeze_manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        log(f"Pre-external freeze manifest reused: {manifest_path}")
        return data

    files = discover_freeze_files(root, ckpt_dir)
    t0 = time.time()
    rows = []
    for i, p in enumerate(files, start=1):
        try:
            stat = p.stat()
            digest = sha256_file(p)
            rows.append({
                "path": str(p),
                "relative_path": str(p.relative_to(root)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            })
        except Exception as e:
            rows.append({"path": str(p), "error": str(e)})
        if i == 1 or i == len(files) or i % 25 == 0:
            progress("11 freeze artifacts", i, len(files), p.name, t0)

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "purpose": "Freeze pre-external-validation model/feature state before cohort inspection",
        "file_count": len(rows),
        "files": rows,
    }
    atomic_json(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------------------
# Stage 2 — NCBI source acquisition
# --------------------------------------------------------------------------------------

def build_ncbi_sql(organism: str, antibiotics: List[str]) -> str:
    drugs = ", ".join(quote_sql_string(x.lower()) for x in antibiotics)
    org = organism.lower().replace("%", r"\%")
    return f"""
SELECT
  ast.target_acc,
  ast.taxgroup_name,
  ast.antibiotic,
  ast.phenotype,
  iso.biosample_acc,
  iso.asm_acc,
  iso.erd_group
FROM `{NCBI_AST_TABLE}` AS ast
JOIN `{NCBI_ISOLATES_TABLE}` AS iso
  ON ast.target_acc = iso.target_acc
WHERE LOWER(ast.taxgroup_name) LIKE {quote_sql_string('%' + org + '%')}
  AND LOWER(ast.antibiotic) IN ({drugs})
  AND LOWER(ast.phenotype) IN (
    'resistant', 'susceptible', 'intermediate', 'nonsusceptible',
    'non-susceptible', 'r', 's', 'i'
  )
  AND iso.asm_acc IS NOT NULL
ORDER BY ast.antibiotic, ast.target_acc
""".strip()


def run_bq_csv(sql: str, out_csv: Path, bq_project: Optional[str], max_rows: int = 500000) -> None:
    bq = shutil.which("bq")
    if not bq:
        raise RuntimeError("bq command not found")

    cmd = [
        bq, "query",
        "--quiet",
        "--use_legacy_sql=false",
        "--format=csv",
        f"--max_rows={max_rows}",
    ]
    if bq_project:
        cmd.append(f"--project_id={bq_project}")
    cmd.append(sql)

    log("Running NCBI Pathogen Detection BigQuery query...")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "BigQuery query failed.\n"
            f"STDERR:\n{proc.stderr[-4000:]}"
        )
    if not proc.stdout.strip():
        raise RuntimeError("BigQuery returned no CSV output")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(proc.stdout)


def acquire_ncbi_source(args, raw_dir: Path, ckpt_dir: Path) -> Tuple[Optional[Path], str]:
    raw_csv = raw_dir / "file11_ncbi_ast_isolates_joined_raw.csv"
    sql_path = raw_dir / "file11_ncbi_ast_isolates_query.sql"
    antibiotics = [x.strip().lower() for x in args.antibiotics.split(",") if x.strip()]
    sql = build_ncbi_sql(args.organism, antibiotics)
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    sql_path.write_text(sql + "\n")

    if args.source_csv:
        src = Path(args.source_csv)
        if not src.exists():
            raise FileNotFoundError(f"--source-csv not found: {src}")
        if not raw_csv.exists() or args.refresh_source:
            shutil.copy2(src, raw_csv)
        return raw_csv, "LOCAL_EXPORTED_CSV"

    if raw_csv.exists() and not args.refresh_source:
        return raw_csv, "CACHE"

    if shutil.which("bq"):
        try:
            run_bq_csv(sql, raw_csv, args.bq_project)
            return raw_csv, "BIGQUERY"
        except Exception as e:
            log(f"BigQuery automatic acquisition failed: {e}")

    needs = {
        "status": "NEEDS_SOURCE_DATA",
        "sql_query": str(sql_path),
        "expected_csv": str(raw_csv),
        "instruction": (
            "Run file11_ncbi_ast_isolates_query.sql in Google BigQuery and export "
            "the result as CSV, then rerun with --source-csv <export.csv>."
        ),
    }
    atomic_json(ckpt_dir / "file11_needs_source_data.json", needs)
    return None, "NEEDS_SOURCE_DATA"


# --------------------------------------------------------------------------------------
# Stage 3 — phenotype harmonization
# --------------------------------------------------------------------------------------

def harmonize_ncbi(raw_csv: Path, out_dir: Path, include_i_as_r: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)

    required = ["target_acc", "antibiotic", "phenotype", "biosample_acc", "asm_acc", "erd_group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"NCBI source CSV is missing columns: {missing}. "
            "Use the SQL generated by FILE11."
        )

    for c in required + (["taxgroup_name"] if "taxgroup_name" in df.columns else []):
        df[c] = df[c].map(norm_str)

    df["antibiotic"] = df["antibiotic"].str.lower()
    ph = df["phenotype"].str.strip().str.lower()

    label = pd.Series([None] * len(df), index=df.index, dtype=object)
    label[ph.isin(["resistant", "r"])] = 1
    label[ph.isin(["susceptible", "s"])] = 0
    if include_i_as_r:
        label[ph.isin(["intermediate", "i", "nonsusceptible", "non-susceptible"])] = 1

    df["label_binary"] = label
    df["asm_acc_base"] = df["asm_acc"].map(assembly_base)

    excluded_nonbinary = df[df["label_binary"].isna()].copy()
    x = df[df["label_binary"].notna()].copy()
    x["label_binary"] = x["label_binary"].astype(int)

    # Collapse exact duplicate AST rows first.
    x = x.drop_duplicates(
        subset=["target_acc", "antibiotic", "label_binary", "biosample_acc", "asm_acc", "erd_group"]
    )

    # Exclude any isolate/drug pair with conflicting binary labels.
    counts = (
        x.groupby(["target_acc", "antibiotic"])["label_binary"]
        .nunique()
        .reset_index(name="n_labels")
    )
    conflicts = counts[counts["n_labels"] > 1][["target_acc", "antibiotic"]]
    if not conflicts.empty:
        conflict_keys = set(map(tuple, conflicts.to_numpy()))
        is_conf = [
            (r.target_acc, r.antibiotic) in conflict_keys
            for r in x[["target_acc", "antibiotic"]].itertuples(index=False)
        ]
        conflict_rows = x[pd.Series(is_conf, index=x.index)].copy()
        x = x[~pd.Series(is_conf, index=x.index)].copy()
    else:
        conflict_rows = x.iloc[0:0].copy()

    # Prefer highest assembly version if duplicate BioSample/drug entries exist.
    def asm_version(acc: str) -> int:
        m = re.search(r"\.(\d+)$", acc or "")
        return int(m.group(1)) if m else 0

    x["_asm_version"] = x["asm_acc"].map(asm_version)
    x = x.sort_values(
        ["antibiotic", "biosample_acc", "_asm_version", "asm_acc"],
        ascending=[True, True, False, True],
    )

    # One assembly per antibiotic; then one BioSample per antibiotic.
    x = x.drop_duplicates(["antibiotic", "asm_acc_base"], keep="first")
    has_bs = x["biosample_acc"] != ""
    xb = x[has_bs].drop_duplicates(["antibiotic", "biosample_acc"], keep="first")
    xn = x[~has_bs]
    x = pd.concat([xb, xn], ignore_index=True)
    x = x.drop(columns=["_asm_version"])

    exclusions = pd.concat(
        [
            excluded_nonbinary.assign(exclusion_reason="NON_BINARY_PHENOTYPE"),
            conflict_rows.assign(exclusion_reason="CONFLICTING_BINARY_PHENOTYPE"),
        ],
        ignore_index=True,
        sort=False,
    )

    harm_path = out_dir / "file11_external_harmonized.csv"
    excl_path = out_dir / "file11_external_harmonization_exclusions.csv"
    x.to_csv(harm_path, index=False)
    exclusions.to_csv(excl_path, index=False)

    for drug, sub in x.groupby("antibiotic"):
        vc = sub["label_binary"].value_counts().to_dict()
        log(
            f"[PROGRESS] 11 harmonized {drug}: n={len(sub)} | "
            f"R={vc.get(1,0)} S={vc.get(0,0)}"
        )
    return x, exclusions


# --------------------------------------------------------------------------------------
# Internal manifest discovery + BV-BRC crosswalk
# --------------------------------------------------------------------------------------

def inspect_candidate_file(path: Path) -> Tuple[int, List[str]]:
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path, nrows=5)
        elif suffix in {".tsv", ".txt"}:
            df = pd.read_csv(path, sep="\t", nrows=5)
        elif suffix == ".parquet":
            df = pd.read_parquet(path).head(5)
        else:
            return 0, []
        hits = []
        for key, aliases in ALIASES.items():
            if first_existing_col(df, aliases) is not None:
                hits.append(key)
        return len(hits), hits
    except Exception:
        return 0, []


def discover_internal_manifest(root: Path, out_csv: Path) -> Optional[Path]:
    name_hints = ("metadata", "manifest", "cohort", "sample", "phenotype", "genome")
    rows = []
    for base in [root / "data", root / "checkpoints"]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".csv", ".tsv", ".txt", ".parquet"}:
                continue
            name = p.name.lower()
            if not any(h in name for h in name_hints):
                continue
            try:
                if p.stat().st_size > 500 * 1024 * 1024:
                    continue
            except OSError:
                continue
            score, hits = inspect_candidate_file(p)
            if score:
                rows.append({
                    "path": str(p),
                    "score": score,
                    "identifier_columns_detected": ";".join(hits),
                    "size_bytes": p.stat().st_size,
                })

    cand = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out_csv, index=False)
    if cand.empty:
        return None

    cand = cand.sort_values(["score", "size_bytes"], ascending=[False, True])
    top_score = int(cand.iloc[0]["score"])
    top = cand[cand["score"] == top_score]
    # Only auto-select when unambiguous and at least two identifier types exist.
    if top_score >= 2 and len(top) == 1:
        return Path(top.iloc[0]["path"])
    return None


def load_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).astype(str)
    raise ValueError(f"Unsupported internal manifest format: {path}")


def bvbrc_fetch_genome_crosswalk(genome_id: str, timeout: int = 30) -> dict:
    url = f"{BVBRC_API}/genome/"
    params = {
        "eq(genome_id," + genome_id + ")": "",
    }
    # requests cannot naturally express RQL key-only query; build URL directly.
    qurl = f"{BVBRC_API}/genome/?eq(genome_id,{genome_id})"
    r = requests.get(
        qurl,
        timeout=timeout,
        headers={"Accept": "application/json", "User-Agent": f"FILE11/{VERSION}"},
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        g = data[0]
    elif isinstance(data, dict) and data.get("genome_id"):
        g = data
    else:
        return {
            "genome_id": genome_id,
            "assembly_accession": "",
            "biosample_accession": "",
            "sra_accession": "",
            "status": "NOT_FOUND",
        }
    return {
        "genome_id": genome_id,
        "assembly_accession": norm_str(g.get("assembly_accession")),
        "biosample_accession": norm_str(g.get("biosample_accession")),
        "sra_accession": norm_str(g.get("sra_accession")),
        "status": "OK",
    }


def bvbrc_crosswalk(
    ids: Sequence[str],
    ckpt_dir: Path,
    workers: int = 6,
) -> pd.DataFrame:
    cache_dir = ckpt_dir / "bvbrc_genome_crosswalk"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted(set(x for x in ids if x))
    rows: Dict[str, dict] = {}
    missing = []

    for gid in ids:
        p = cache_dir / f"{gid.replace('/','_')}.json"
        if p.exists():
            try:
                rows[gid] = json.loads(p.read_text())
                continue
            except Exception:
                pass
        missing.append(gid)

    t0 = time.time()
    completed = len(rows)
    if completed:
        progress("11 BV-BRC crosswalk", completed, len(ids), "cached", t0)

    def task(gid: str) -> Tuple[str, dict]:
        last = None
        for attempt in range(1, 4):
            try:
                return gid, bvbrc_fetch_genome_crosswalk(gid)
            except Exception as e:
                last = e
                time.sleep(min(2 ** attempt, 8))
        return gid, {
            "genome_id": gid,
            "assembly_accession": "",
            "biosample_accession": "",
            "sra_accession": "",
            "status": f"ERROR: {last}",
        }

    if missing:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(task, gid): gid for gid in missing}
            for fut in as_completed(futs):
                gid, rec = fut.result()
                rows[gid] = rec
                atomic_json(cache_dir / f"{gid.replace('/','_')}.json", rec)
                completed += 1
                if completed == 1 or completed == len(ids) or completed % 10 == 0:
                    progress(
                        "11 BV-BRC crosswalk", completed, len(ids),
                        f"{gid} | {rec.get('status','')}", t0
                    )

    return pd.DataFrame([rows[x] for x in ids])


def enrich_internal_identifiers(
    internal_raw: pd.DataFrame,
    ckpt_dir: Path,
    workers: int,
) -> pd.DataFrame:
    ids = canonical_identifier_frame(internal_raw)

    need_bv = (
        (ids["genome_id"] != "")
        & (ids["asm_acc"] == "")
        & (ids["biosample_acc"] == "")
    )
    gids = ids.loc[need_bv, "genome_id"].tolist()

    if gids:
        log(f"BV-BRC identifier crosswalk required for {len(set(gids))} genome IDs")
        cross = bvbrc_crosswalk(gids, ckpt_dir, workers=workers)
        cross = cross.rename(columns={
            "assembly_accession": "bv_asm_acc",
            "biosample_accession": "bv_biosample_acc",
            "sra_accession": "bv_sra_acc",
        })
        ids = ids.merge(cross, on="genome_id", how="left")
        for base, bv in [
            ("asm_acc", "bv_asm_acc"),
            ("biosample_acc", "bv_biosample_acc"),
            ("sra_acc", "bv_sra_acc"),
        ]:
            if bv in ids.columns:
                ids[base] = ids[base].where(ids[base] != "", ids[bv].fillna(""))
        ids = ids.drop(
            columns=[c for c in ["bv_asm_acc", "bv_biosample_acc", "bv_sra_acc", "status"]
                     if c in ids.columns]
        )

    ids["asm_acc_base"] = ids["asm_acc"].map(assembly_base)
    return ids.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Stage 4 — exact leakage firewall
# --------------------------------------------------------------------------------------

def build_internal_sets(ids: pd.DataFrame) -> Dict[str, set]:
    out = {
        "target_acc": set(x.upper() for x in ids["target_acc"] if x),
        "biosample_acc": set(x.upper() for x in ids["biosample_acc"] if x),
        "asm_acc": set(x.upper() for x in ids["asm_acc"] if x),
        "asm_acc_base": set(x.upper() for x in ids["asm_acc_base"] if x),
        "sra_acc": set(),
        "erd_group": set(x for x in ids["erd_group"] if x),
    }
    for v in ids["sra_acc"]:
        out["sra_acc"].update(sra_tokens(v))
    return out


def exact_firewall(external: pd.DataFrame, internal_ids: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sets = build_internal_sets(internal_ids)
    audit = []

    t0 = time.time()
    n = len(external)
    for i, row in enumerate(external.itertuples(index=False), start=1):
        reasons = []
        target = norm_upper(getattr(row, "target_acc"))
        bs = norm_upper(getattr(row, "biosample_acc"))
        asm = norm_upper(getattr(row, "asm_acc"))
        asm_base = assembly_base(asm)

        if target and target in sets["target_acc"]:
            reasons.append("TARGET_ACC_OVERLAP")
        if bs and bs in sets["biosample_acc"]:
            reasons.append("BIOSAMPLE_OVERLAP")
        if asm and asm in sets["asm_acc"]:
            reasons.append("ASSEMBLY_EXACT_OVERLAP")
        if asm_base and asm_base in sets["asm_acc_base"]:
            reasons.append("ASSEMBLY_BASE_OVERLAP")

        audit.append(";".join(sorted(set(reasons))))
        if i == 1 or i == n or i % max(1, n // 20) == 0:
            progress("11 exact leakage firewall", i, n, target or asm, t0)

    x = external.copy()
    x["exact_overlap_reason"] = audit
    excluded = x[x["exact_overlap_reason"] != ""].copy()
    clean = x[x["exact_overlap_reason"] == ""].copy()
    return clean, excluded


# --------------------------------------------------------------------------------------
# Stage 5 — SNP-cluster firewall
# --------------------------------------------------------------------------------------

def bq_map_internal_clusters(
    internal_ids: pd.DataFrame,
    cache_csv: Path,
    bq_project: Optional[str],
    ckpt_dir: Path,
    chunk_size: int = 200,
) -> Tuple[pd.DataFrame, str]:
    if cache_csv.exists():
        return pd.read_csv(cache_csv, dtype=str, keep_default_na=False), "CACHE"

    if not shutil.which("bq"):
        return pd.DataFrame(), "BQ_UNAVAILABLE"

    asm = sorted(set(x for x in internal_ids["asm_acc"] if x))
    bs = sorted(set(x for x in internal_ids["biosample_acc"] if x))
    target = sorted(set(x for x in internal_ids["target_acc"] if x))

    tokens = [("asm_acc", x) for x in asm] + [("biosample_acc", x) for x in bs] + [
        ("target_acc", x) for x in target
    ]
    if not tokens:
        return pd.DataFrame(), "NO_QUERYABLE_IDS"

    rows = []
    t0 = time.time()
    n_chunks = math.ceil(len(tokens) / chunk_size)

    for ci in range(n_chunks):
        chunk = tokens[ci * chunk_size:(ci + 1) * chunk_size]
        by_field: Dict[str, List[str]] = {}
        for field, value in chunk:
            by_field.setdefault(field, []).append(value)

        clauses = []
        for field, vals in by_field.items():
            val_sql = ", ".join(quote_sql_string(v) for v in vals)
            clauses.append(f"{field} IN ({val_sql})")

        sql = f"""
SELECT target_acc, biosample_acc, asm_acc, erd_group
FROM `{NCBI_ISOLATES_TABLE}`
WHERE {" OR ".join(clauses)}
""".strip()

        chunk_csv = ckpt_dir / "ncbi_internal_cluster_chunks" / f"chunk_{ci+1:05d}.csv"
        chunk_csv.parent.mkdir(parents=True, exist_ok=True)
        if not chunk_csv.exists():
            try:
                run_bq_csv(sql, chunk_csv, bq_project, max_rows=100000)
            except Exception as e:
                log(f"Internal SNP-cluster mapping chunk failed: {e}")
                continue

        try:
            d = pd.read_csv(chunk_csv, dtype=str, keep_default_na=False)
            rows.append(d)
        except Exception:
            pass

        progress(
            "11 internal SNP-cluster mapping", ci + 1, n_chunks,
            f"chunk rows={0 if not rows else len(rows[-1])}", t0
        )

    if not rows:
        return pd.DataFrame(), "NO_MAPPING_ROWS"

    out = pd.concat(rows, ignore_index=True).drop_duplicates()
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_csv, index=False)
    return out, "BIGQUERY"


def strict_cluster_firewall(
    clean: pd.DataFrame,
    internal_ids: pd.DataFrame,
    internal_cluster_map: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups = set(x for x in internal_ids.get("erd_group", pd.Series(dtype=str)).map(norm_str) if x)

    if not internal_cluster_map.empty and "erd_group" in internal_cluster_map.columns:
        groups.update(x for x in internal_cluster_map["erd_group"].map(norm_str) if x)

    x = clean.copy()
    x["erd_group"] = x["erd_group"].map(norm_str)
    x["cluster_overlap_internal"] = x["erd_group"].isin(groups) & (x["erd_group"] != "")
    x["strict_cluster_verifiable"] = x["erd_group"] != ""

    excluded_overlap = x[x["cluster_overlap_internal"]].copy()

    # Strict E2 requires a known external NCBI SNP cluster and no internal cluster overlap.
    strict = x[
        (~x["cluster_overlap_internal"])
        & x["strict_cluster_verifiable"]
    ].copy()

    stats = {
        "internal_cluster_count": len(groups),
        "external_clean_with_cluster": int((x["erd_group"] != "").sum()),
        "external_clean_missing_cluster": int((x["erd_group"] == "").sum()),
        "excluded_same_snp_cluster": int(x["cluster_overlap_internal"].sum()),
    }
    return strict, excluded_overlap, stats


# --------------------------------------------------------------------------------------
# Stage 6/7 — freeze cohort and report readiness
# --------------------------------------------------------------------------------------

def class_count_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["antibiotic", "label_binary", "n"])
    return (
        df.groupby(["antibiotic", "label_binary"])
        .size()
        .reset_index(name="n")
        .sort_values(["antibiotic", "label_binary"])
    )


def readiness_by_drug(df: pd.DataFrame, min_per_class: int) -> pd.DataFrame:
    drugs = sorted(df["antibiotic"].unique()) if not df.empty else []
    rows = []
    for drug in drugs:
        sub = df[df["antibiotic"] == drug]
        r = int((sub["label_binary"] == 1).sum())
        s = int((sub["label_binary"] == 0).sum())
        rows.append({
            "antibiotic": drug,
            "n_total": len(sub),
            "n_resistant": r,
            "n_susceptible": s,
            "min_per_class_required": min_per_class,
            "blind_validation_ready": r >= min_per_class and s >= min_per_class,
        })
    return pd.DataFrame(rows)


def write_accession_lists(df: pd.DataFrame, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for drug, sub in df.groupby("antibiotic"):
        acc = sorted(set(x for x in sub["asm_acc"] if x))
        p = final_dir / f"file11_{drug}_external_strict_assembly_accessions.txt"
        p.write_text("\n".join(acc) + ("\n" if acc else ""))


def parse_args():
    p = argparse.ArgumentParser(
        description="FILE11 — independent external cohort + leakage firewall"
    )
    p.add_argument(
        "--project-root",
        default="/mnt/c/Users/ASUS/Desktop/AMR-Genome-ML",
    )
    p.add_argument(
        "--organism",
        default="Klebsiella pneumoniae",
    )
    p.add_argument(
        "--antibiotics",
        default="ertapenem,imipenem,meropenem",
        help="Comma-separated; endpoints remain separate",
    )
    p.add_argument(
        "--internal-manifest",
        default=None,
        help="CSV/TSV/Parquet containing original internal cohort identifiers",
    )
    p.add_argument(
        "--source-csv",
        default=None,
        help="CSV exported from the FILE11 BigQuery SQL if automatic bq access is unavailable",
    )
    p.add_argument(
        "--bq-project",
        default=None,
        help="Optional Google Cloud billing project for bq query",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="BV-BRC crosswalk workers",
    )
    p.add_argument(
        "--min-per-class",
        type=int,
        default=30,
        help="Minimum resistant AND susceptible isolates per drug for readiness",
    )
    p.add_argument(
        "--include-intermediate-as-resistant",
        action="store_true",
        help="Off by default; do not use unless pre-specified by the original endpoint",
    )
    p.add_argument(
        "--refresh-source",
        action="store_true",
        help="Re-query/copy NCBI source instead of reusing frozen raw cache",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Reset FILE11 stage checkpoints; raw NCBI cache is preserved unless --refresh-source",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root)

    data_root = root / "data/external_validation/ncbi_pathogen_detection"
    raw_dir = data_root / "raw"
    inter_dir = data_root / "intermediate"
    final_dir = data_root / "final"
    ckpt_dir = root / "checkpoints/external_validation_ncbi_pathogen_detection_file11"

    for d in [raw_dir, inter_dir, final_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sig = config_signature(args)
    config_path = ckpt_dir / "file11_run_config.json"

    if args.reset:
        for p in ckpt_dir.glob("stage_*.done.json"):
            p.unlink(missing_ok=True)
        (ckpt_dir / "file11_final_summary.json").unlink(missing_ok=True)

    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("config_signature") != sig:
            raise RuntimeError(
                "FILE11 configuration changed since checkpoints were created. "
                "Rerun with --reset to start a new cohort definition."
            )
    else:
        atomic_json(config_path, {
            "script_version": VERSION,
            "design_id": DESIGN_ID,
            "config_signature": sig,
            "organism": args.organism,
            "antibiotics": [
                x.strip().lower() for x in args.antibiotics.split(",") if x.strip()
            ],
            "include_intermediate_as_resistant": args.include_intermediate_as_resistant,
            "min_per_class": args.min_per_class,
            "created_utc": pd.Timestamp.utcnow().isoformat(),
        })

    log("=" * 104)
    log("FILE11 — INDEPENDENT EXTERNAL COHORT + LEAKAGE FIREWALL")
    log("=" * 104)
    log(f"Version       : {VERSION}")
    log(f"Project root  : {root}")
    log(f"Organism      : {args.organism}")
    log(f"Antibiotics   : {args.antibiotics}")
    log("Endpoints     : KEPT SEPARATE (no phenotype pooling)")
    log("Checkpointing : PER STAGE + PER BV-BRC GENOME + PER NCBI QUERY CHUNK")
    log("Live progress : ENABLED")
    log("Model policy  : FROZEN; NO TUNING ON EXTERNAL COHORT")

    # ----------------------------------------------------------------------------------
    # Stage 1/7
    # ----------------------------------------------------------------------------------
    log("Stage 1/7 — freeze pre-external model/feature state")
    freeze = freeze_project_state(root, ckpt_dir)
    mark_stage_done(ckpt_dir, 1, {
        "status": "PASS",
        "frozen_files": freeze.get("file_count", 0),
        "manifest": str(ckpt_dir / "file11_pre_external_freeze_manifest.json"),
    })
    log(f"[PROGRESS] 11 freeze: {freeze.get('file_count',0)} artifacts hashed/frozen")

    # ----------------------------------------------------------------------------------
    # Stage 2/7
    # ----------------------------------------------------------------------------------
    log("Stage 2/7 — acquire NCBI Pathogen Detection AST + isolate metadata")
    raw_csv, source_mode = acquire_ncbi_source(args, raw_dir, ckpt_dir)
    if raw_csv is None:
        log("=" * 104)
        log("FILE11 STATUS : NEEDS_SOURCE_DATA")
        log(f"SQL query saved: {raw_dir / 'file11_ncbi_ast_isolates_query.sql'}")
        log("Export the query result as CSV, then rerun with --source-csv <export.csv>.")
        log("All completed checkpoints have been preserved.")
        log("=" * 104)
        return 20

    raw_hash = sha256_file(raw_csv)
    raw_n = max(0, sum(1 for _ in raw_csv.open("r", errors="ignore")) - 1)
    mark_stage_done(ckpt_dir, 2, {
        "status": "PASS",
        "source_mode": source_mode,
        "raw_csv": str(raw_csv),
        "raw_sha256": raw_hash,
        "raw_rows_approx": raw_n,
    })
    log(f"[PROGRESS] 11 source: rows≈{raw_n} | mode={source_mode} | cache=frozen")

    # ----------------------------------------------------------------------------------
    # Stage 3/7
    # ----------------------------------------------------------------------------------
    log("Stage 3/7 — harmonize phenotype labels and remove conflicts")
    harm_path = inter_dir / "file11_external_harmonized.csv"
    excl_harm_path = inter_dir / "file11_external_harmonization_exclusions.csv"

    st3 = read_stage_done(ckpt_dir, 3)
    if st3 and harm_path.exists() and st3.get("raw_sha256") == raw_hash:
        harm = pd.read_csv(harm_path, dtype={"label_binary": int}, keep_default_na=False)
        harm_excl = (
            pd.read_csv(excl_harm_path, keep_default_na=False)
            if excl_harm_path.exists() else pd.DataFrame()
        )
        log(f"[PROGRESS] 11 harmonization checkpoint reused: n={len(harm)}")
    else:
        harm, harm_excl = harmonize_ncbi(
            raw_csv, inter_dir, args.include_intermediate_as_resistant
        )
        mark_stage_done(ckpt_dir, 3, {
            "status": "PASS",
            "raw_sha256": raw_hash,
            "harmonized_rows": len(harm),
            "excluded_rows": len(harm_excl),
            "output": str(harm_path),
        })

    # ----------------------------------------------------------------------------------
    # Stage 4/7
    # ----------------------------------------------------------------------------------
    log("Stage 4/7 — exact identifier leakage firewall against internal cohort")

    manifest_path: Optional[Path] = None
    if args.internal_manifest:
        manifest_path = Path(args.internal_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"--internal-manifest not found: {manifest_path}")
    else:
        cand_path = ckpt_dir / "file11_internal_manifest_candidates.csv"
        manifest_path = discover_internal_manifest(root, cand_path)
        if manifest_path:
            log(f"Auto-selected internal manifest: {manifest_path}")
        else:
            needs = {
                "status": "NEEDS_INTERNAL_MANIFEST",
                "candidate_list": str(cand_path),
                "instruction": (
                    "Rerun with --internal-manifest <file>. The file should contain all "
                    "original study isolates and preferably Genome ID, assembly accession, "
                    "BioSample, SRA, or NCBI target accession."
                ),
            }
            atomic_json(ckpt_dir / "file11_needs_internal_manifest.json", needs)
            log("=" * 104)
            log("FILE11 STATUS : NEEDS_INTERNAL_MANIFEST")
            log(f"External phenotype cohort checkpointed: {harm_path}")
            log(f"Candidate internal files listed at: {cand_path}")
            log("Rerun with: --internal-manifest <original-study-manifest.csv>")
            log("No external model evaluation has been performed.")
            log("=" * 104)
            return 21

    internal_raw = load_manifest(manifest_path)
    internal_ids_path = inter_dir / "file11_internal_identifier_crosswalk.csv"
    manifest_hash = sha256_file(manifest_path)

    st4 = read_stage_done(ckpt_dir, 4)
    clean_path = inter_dir / "file11_external_clean_E1.csv"
    exact_excl_path = inter_dir / "file11_exact_overlap_exclusions.csv"

    if (
        st4
        and st4.get("internal_manifest_sha256") == manifest_hash
        and clean_path.exists()
        and internal_ids_path.exists()
    ):
        internal_ids = pd.read_csv(internal_ids_path, dtype=str, keep_default_na=False)
        clean = pd.read_csv(clean_path, dtype={"label_binary": int}, keep_default_na=False)
        exact_excl = (
            pd.read_csv(exact_excl_path, keep_default_na=False)
            if exact_excl_path.exists() else pd.DataFrame()
        )
        log(f"[PROGRESS] 11 exact firewall checkpoint reused: E1 n={len(clean)}")
    else:
        internal_ids = enrich_internal_identifiers(
            internal_raw, ckpt_dir, workers=args.workers
        )
        internal_ids.to_csv(internal_ids_path, index=False)

        usable = (
            (internal_ids["asm_acc"] != "")
            | (internal_ids["biosample_acc"] != "")
            | (internal_ids["target_acc"] != "")
            | (internal_ids["sra_acc"] != "")
        )
        usable_n = int(usable.sum())
        if usable_n == 0:
            raise RuntimeError(
                "Internal manifest could not be cross-walked to any NCBI identifiers. "
                "Do not claim an independent cohort yet."
            )

        clean, exact_excl = exact_firewall(harm, internal_ids)
        clean.to_csv(clean_path, index=False)
        exact_excl.to_csv(exact_excl_path, index=False)

        mark_stage_done(ckpt_dir, 4, {
            "status": "PASS",
            "internal_manifest": str(manifest_path),
            "internal_manifest_sha256": manifest_hash,
            "internal_rows": len(internal_ids),
            "internal_rows_with_ncbi_identifier": usable_n,
            "external_before_firewall": len(harm),
            "exact_overlap_excluded": len(exact_excl),
            "external_clean_E1": len(clean),
        })

    log(
        f"[PROGRESS] 11 exact firewall: internal={len(internal_ids)} | "
        f"excluded={len(exact_excl)} | E1={len(clean)}"
    )

    # ----------------------------------------------------------------------------------
    # Stage 5/7
    # ----------------------------------------------------------------------------------
    log("Stage 5/7 — NCBI SNP-cluster clonal-overlap firewall")

    internal_cluster_cache = inter_dir / "file11_internal_ncbi_snp_clusters.csv"
    cluster_map, cluster_mode = bq_map_internal_clusters(
        internal_ids,
        internal_cluster_cache,
        args.bq_project,
        ckpt_dir,
    )

    strict, cluster_excl, cluster_stats = strict_cluster_firewall(
        clean, internal_ids, cluster_map
    )

    strict_path = final_dir / "file11_external_strict_E2.csv"
    cluster_excl_path = inter_dir / "file11_snp_cluster_overlap_exclusions.csv"
    strict.to_csv(strict_path, index=False)
    cluster_excl.to_csv(cluster_excl_path, index=False)

    mark_stage_done(ckpt_dir, 5, {
        "status": "PASS" if cluster_stats["internal_cluster_count"] > 0 else "PARTIAL",
        "cluster_mapping_mode": cluster_mode,
        **cluster_stats,
        "external_strict_E2": len(strict),
        "output": str(strict_path),
    })

    log(
        f"[PROGRESS] 11 clonal firewall: mode={cluster_mode} | "
        f"internal_clusters={cluster_stats['internal_cluster_count']} | "
        f"same-cluster excluded={cluster_stats['excluded_same_snp_cluster']} | "
        f"E2={len(strict)}"
    )

    # ----------------------------------------------------------------------------------
    # Stage 6/7
    # ----------------------------------------------------------------------------------
    log("Stage 6/7 — freeze final external cohort and assembly accession lists")

    freeze_csv = final_dir / "file11_external_strict_E2_FROZEN.csv"
    strict.to_csv(freeze_csv, index=False)
    cohort_hash = sha256_file(freeze_csv)
    write_accession_lists(strict, final_dir)

    class_counts = class_count_table(strict)
    class_counts_path = final_dir / "file11_external_strict_class_counts.csv"
    class_counts.to_csv(class_counts_path, index=False)

    readiness = readiness_by_drug(strict, args.min_per_class)
    readiness_path = final_dir / "file11_external_validation_readiness.csv"
    readiness.to_csv(readiness_path, index=False)

    for _, r in readiness.iterrows():
        log(
            f"[PROGRESS] 11 readiness {r['antibiotic']}: "
            f"n={r['n_total']} | R={r['n_resistant']} | S={r['n_susceptible']} | "
            f"ready={bool(r['blind_validation_ready'])}"
        )

    mark_stage_done(ckpt_dir, 6, {
        "status": "PASS",
        "frozen_cohort": str(freeze_csv),
        "frozen_cohort_sha256": cohort_hash,
        "rows": len(strict),
        "class_counts": str(class_counts_path),
        "readiness": str(readiness_path),
    })

    # ----------------------------------------------------------------------------------
    # Stage 7/7
    # ----------------------------------------------------------------------------------
    log("Stage 7/7 — external-validation readiness summary")

    exact_firewall_verified = len(internal_ids) > 0
    cluster_firewall_verified = cluster_stats["internal_cluster_count"] > 0
    ready_any = (
        bool(readiness["blind_validation_ready"].any())
        if not readiness.empty else False
    )
    ready_all = (
        bool(readiness["blind_validation_ready"].all())
        if not readiness.empty else False
    )

    if exact_firewall_verified and cluster_firewall_verified and ready_any:
        status = "PASS_COHORT_FROZEN"
    elif exact_firewall_verified and ready_any:
        status = "PASS_EXACT_FIREWALL_ONLY"
    else:
        status = "PARTIAL_REVIEW_REQUIRED"

    final_summary = {
        "script_version": VERSION,
        "design_id": DESIGN_ID,
        "status": status,
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
        "organism": args.organism,
        "antibiotics": [
            x.strip().lower() for x in args.antibiotics.split(",") if x.strip()
        ],
        "external_source": "NCBI Pathogen Detection AST + Isolates Browser BigQuery",
        "source_mode": source_mode,
        "model_policy": (
            "Pre-external model/feature artifacts were frozen. External cohort was "
            "not used for feature selection, hyperparameter tuning, or threshold selection."
        ),
        "phenotype_policy": (
            "Drug endpoints kept separate. Resistant=1, Susceptible=0. "
            + (
                "Intermediate/nonsusceptible mapped to resistant by explicit CLI request."
                if args.include_intermediate_as_resistant
                else "Intermediate/nonsusceptible excluded from binary validation."
            )
        ),
        "internal_manifest": str(manifest_path),
        "internal_manifest_sha256": manifest_hash,
        "pre_external_freeze_manifest": str(
            ckpt_dir / "file11_pre_external_freeze_manifest.json"
        ),
        "ncbi_raw_sha256": raw_hash,
        "frozen_external_cohort_sha256": cohort_hash,
        "counts": {
            "raw_ncbi_rows_approx": raw_n,
            "harmonized_external": len(harm),
            "harmonization_exclusions": len(harm_excl),
            "exact_overlap_excluded": len(exact_excl),
            "external_clean_E1": len(clean),
            "same_snp_cluster_excluded": len(cluster_excl),
            "external_strict_E2": len(strict),
        },
        "cluster_firewall": {
            "mode": cluster_mode,
            **cluster_stats,
            "verified": cluster_firewall_verified,
        },
        "blind_validation_readiness": {
            "min_per_class": args.min_per_class,
            "any_endpoint_ready": ready_any,
            "all_endpoints_ready": ready_all,
            "table": str(readiness_path),
        },
        "outputs": {
            "external_harmonized": str(harm_path),
            "internal_identifier_crosswalk": str(internal_ids_path),
            "external_clean_E1": str(clean_path),
            "external_strict_E2": str(strict_path),
            "external_strict_E2_frozen": str(freeze_csv),
            "class_counts": str(class_counts_path),
            "readiness": str(readiness_path),
            "exact_overlap_exclusions": str(exact_excl_path),
            "snp_cluster_overlap_exclusions": str(cluster_excl_path),
        },
        "next_stage": (
            "Blindly reconstruct the original frozen feature representation on E2 and "
            "evaluate the original frozen model without retraining/tuning."
        ),
    }

    final_summary_path = ckpt_dir / "file11_final_summary.json"
    atomic_json(final_summary_path, final_summary)
    mark_stage_done(ckpt_dir, 7, {"status": status, "summary": str(final_summary_path)})

    log("=" * 104)
    log(f"FILE11 STATUS : {status}")
    log(f"NCBI harmonized            : {len(harm)}")
    log(f"Exact-overlap excluded     : {len(exact_excl)}")
    log(f"External-clean E1          : {len(clean)}")
    log(f"Same SNP-cluster excluded  : {len(cluster_excl)}")
    log(f"External-strict E2         : {len(strict)}")
    log(f"Frozen cohort SHA256       : {cohort_hash}")
    log(f"Frozen cohort              : {freeze_csv}")
    log(f"Final summary              : {final_summary_path}")
    log("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
