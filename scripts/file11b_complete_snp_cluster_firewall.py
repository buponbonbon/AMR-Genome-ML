#!/usr/bin/env python3
"""
FILE11B — COMPLETE NCBI SNP-CLUSTER FIREWALL
Version: 1.0.0

Purpose
-------
Complete FILE11 Stage 5 after FILE11 v1.0.0 reached:
    PASS_EXACT_FIREWALL_ONLY

This helper NEVER reruns Stage 4 / BV-BRC crosswalk.

Run 1:
    python scripts/file11b_complete_snp_cluster_firewall.py \
      --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML

It generates:
    data/external_validation/ncbi_pathogen_detection/raw/
      file11_internal_snp_cluster_query.sql

Run that SQL in Google BigQuery and export the result to CSV.

Run 2:
    python scripts/file11b_complete_snp_cluster_firewall.py \
      --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML \
      --cluster-csv /path/to/bigquery-export.csv

Then it:
  - derives NCBI SNP clusters represented in the internal cohort
  - removes E1 external isolates sharing those clusters
  - freezes a true E2 strict cohort
  - writes class counts/readiness
  - updates FILE11 stage 5/6/7 checkpoints and final summary

Checkpointing + live progress are enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd


VERSION = "1.0.0"
DESIGN_ID = "file11b_complete_ncbi_snp_cluster_firewall_v1"
NCBI_ISOLATES_TABLE = "ncbi-pathogen-detect.pdbrowser.isolates"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(label: str, i: int, n: int, extra: str = "", t0=None) -> None:
    pct = 100.0 * i / max(1, n)
    eta = ""
    if t0 is not None and i > 0:
        elapsed = time.time() - t0
        remain = max(0.0, elapsed / i * (n - i))
        eta = f" | elapsed={elapsed:.1f}s | ETA={remain:.1f}s"
    suffix = f" | {extra}" if extra else ""
    log(f"[PROGRESS] {label}: {i}/{n} ({pct:5.1f}%){eta}{suffix}")


def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def assembly_base(acc: str) -> str:
    acc = norm(acc).upper()
    return acc.split(".")[0] if acc else ""


def sql_quote(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sql_array(values):
    values = sorted(set(norm(x) for x in values if norm(x)))
    if not values:
        return "[]"
    return "[\n    " + ",\n    ".join(sql_quote(x) for x in values) + "\n  ]"


def build_query(ids: pd.DataFrame) -> str:
    asm = sorted(set(norm(x) for x in ids.get("asm_acc", pd.Series(dtype=str)) if norm(x)))
    asm_base = sorted(set(
        assembly_base(x) for x in ids.get("asm_acc", pd.Series(dtype=str)) if assembly_base(x)
    ))
    biosample = sorted(set(
        norm(x) for x in ids.get("biosample_acc", pd.Series(dtype=str)) if norm(x)
    ))
    target = sorted(set(
        norm(x) for x in ids.get("target_acc", pd.Series(dtype=str)) if norm(x)
    ))

    return f"""-- FILE11 internal-cohort NCBI SNP-cluster mapping
-- Generated from the already checkpointed FILE11 internal identifier crosswalk.
-- Run in Google BigQuery, then export Results as CSV.

WITH internal_ids AS (
  SELECT
    {sql_array(asm)} AS asm_exact,
    {sql_array(asm_base)} AS asm_base,
    {sql_array(biosample)} AS biosamples,
    {sql_array(target)} AS target_accessions
)

SELECT DISTINCT
  iso.target_acc,
  iso.biosample_acc,
  iso.asm_acc,
  iso.erd_group
FROM `{NCBI_ISOLATES_TABLE}` AS iso
CROSS JOIN internal_ids AS ids
WHERE
      iso.asm_acc IN UNNEST(ids.asm_exact)
   OR REGEXP_REPLACE(UPPER(iso.asm_acc), r'\\.\\d+$', '') IN UNNEST(ids.asm_base)
   OR iso.biosample_acc IN UNNEST(ids.biosamples)
   OR iso.target_acc IN UNNEST(ids.target_accessions)
ORDER BY iso.erd_group, iso.asm_acc;
"""


def readiness(df: pd.DataFrame, min_per_class: int) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(columns=[
            "antibiotic", "n_total", "n_resistant", "n_susceptible",
            "min_per_class_required", "blind_validation_ready"
        ])
    for drug, sub in df.groupby("antibiotic"):
        r = int((pd.to_numeric(sub["label_binary"]) == 1).sum())
        s = int((pd.to_numeric(sub["label_binary"]) == 0).sum())
        rows.append({
            "antibiotic": drug,
            "n_total": len(sub),
            "n_resistant": r,
            "n_susceptible": s,
            "min_per_class_required": min_per_class,
            "blind_validation_ready": r >= min_per_class and s >= min_per_class,
        })
    return pd.DataFrame(rows).sort_values("antibiotic")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        default="/mnt/c/Users/ASUS/Desktop/AMR-Genome-ML",
    )
    ap.add_argument(
        "--cluster-csv",
        default=None,
        help="BigQuery export from file11_internal_snp_cluster_query.sql",
    )
    ap.add_argument("--min-per-class", type=int, default=30)
    args = ap.parse_args()

    root = Path(args.project_root)
    data_root = root / "data/external_validation/ncbi_pathogen_detection"
    raw_dir = data_root / "raw"
    inter_dir = data_root / "intermediate"
    final_dir = data_root / "final"
    ckpt_dir = root / "checkpoints/external_validation_ncbi_pathogen_detection_file11"

    ids_path = inter_dir / "file11_internal_identifier_crosswalk.csv"
    e1_path = inter_dir / "file11_external_clean_E1.csv"

    if not ids_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-4 identifier crosswalk: {ids_path}\n"
            "Run FILE11 through Stage 4 first."
        )
    if not e1_path.exists():
        raise FileNotFoundError(
            f"Missing E1 cohort: {e1_path}\n"
            "Run FILE11 through Stage 4 first."
        )

    log("=" * 104)
    log("FILE11B — COMPLETE NCBI SNP-CLUSTER FIREWALL")
    log("=" * 104)
    log(f"Version       : {VERSION}")
    log(f"Project root  : {root}")
    log("Stage 4       : REUSED — no BV-BRC crosswalk rerun")
    log("Checkpointing : ENABLED")
    log("Live progress : ENABLED")

    ids = pd.read_csv(ids_path, dtype=str, keep_default_na=False)
    e1 = pd.read_csv(e1_path, dtype=str, keep_default_na=False)

    log(f"Internal IDs  : {len(ids)} rows")
    log(f"External E1   : {len(e1)} rows")

    # ------------------------------------------------------------------
    # First run: make SQL and stop.
    # ------------------------------------------------------------------
    query_path = raw_dir / "file11_internal_snp_cluster_query.sql"
    query_path.parent.mkdir(parents=True, exist_ok=True)
    sql = build_query(ids)
    query_path.write_text(sql)

    if not args.cluster_csv:
        state = {
            "script_version": VERSION,
            "design_id": DESIGN_ID,
            "status": "NEEDS_INTERNAL_CLUSTER_CSV",
            "internal_identifier_crosswalk": str(ids_path),
            "internal_identifier_crosswalk_sha256": sha256_file(ids_path),
            "external_clean_E1": str(e1_path),
            "external_clean_E1_sha256": sha256_file(e1_path),
            "sql_query": str(query_path),
            "instruction": (
                "Run the SQL in Google BigQuery, export Results as CSV, then rerun "
                "FILE11B with --cluster-csv <export.csv>."
            ),
        }
        atomic_json(ckpt_dir / "file11_stage5_needs_cluster_csv.json", state)

        log("[PROGRESS] 11B SQL generation: 1/1 (100.0%)")
        log("=" * 104)
        log("FILE11B STATUS : NEEDS_INTERNAL_CLUSTER_CSV")
        log(f"SQL query      : {query_path}")
        log("Run this SQL in the SAME BigQuery web UI used for the external cohort.")
        log("Export Results as CSV, then rerun with --cluster-csv <CSV>.")
        log("=" * 104)
        return 20

    cluster_csv = Path(args.cluster_csv)
    if not cluster_csv.exists():
        raise FileNotFoundError(f"--cluster-csv not found: {cluster_csv}")

    # ------------------------------------------------------------------
    # Stage 5 — load cluster mapping
    # ------------------------------------------------------------------
    log("Stage 5/7 — complete NCBI SNP-cluster clonal-overlap firewall")
    cmap = pd.read_csv(cluster_csv, dtype=str, keep_default_na=False)

    required = {"target_acc", "biosample_acc", "asm_acc", "erd_group"}
    missing = sorted(required - set(cmap.columns))
    if missing:
        raise RuntimeError(
            f"Cluster CSV missing columns {missing}. "
            "Use FILE11B-generated SQL without editing its SELECT columns."
        )

    for c in required:
        cmap[c] = cmap[c].map(norm)

    cmap = cmap.drop_duplicates()
    cmap.to_csv(inter_dir / "file11_internal_ncbi_snp_clusters.csv", index=False)

    internal_clusters = sorted(set(
        x for x in cmap["erd_group"] if x
    ) | set(
        norm(x) for x in ids.get("erd_group", pd.Series(dtype=str)) if norm(x)
    ))

    log(
        f"[PROGRESS] 11B internal cluster map: rows={len(cmap)} | "
        f"unique_internal_clusters={len(internal_clusters)}"
    )

    if len(internal_clusters) == 0:
        raise RuntimeError(
            "BigQuery export contains no non-empty erd_group values. "
            "Cannot claim SNP-cluster filtering."
        )

    if "erd_group" not in e1.columns:
        raise RuntimeError(
            "E1 cohort has no erd_group column. Rebuild the external NCBI export "
            "with erd_group included."
        )

    e1["erd_group"] = e1["erd_group"].map(norm)
    cluster_set = set(internal_clusters)

    reasons = []
    t0 = time.time()
    n = len(e1)
    for i, g in enumerate(e1["erd_group"], start=1):
        if not g:
            reasons.append("MISSING_EXTERNAL_SNP_CLUSTER")
        elif g in cluster_set:
            reasons.append("SAME_NCBI_SNP_CLUSTER_AS_INTERNAL")
        else:
            reasons.append("")
        if i == 1 or i == n or i % max(1, n // 20) == 0:
            progress("11B clonal firewall", i, n, g or "no_cluster", t0)

    x = e1.copy()
    x["cluster_firewall_reason"] = reasons

    same_cluster = x[x["cluster_firewall_reason"] == "SAME_NCBI_SNP_CLUSTER_AS_INTERNAL"].copy()
    missing_cluster = x[x["cluster_firewall_reason"] == "MISSING_EXTERNAL_SNP_CLUSTER"].copy()
    e2 = x[x["cluster_firewall_reason"] == ""].copy()

    same_cluster_path = inter_dir / "file11_snp_cluster_overlap_exclusions.csv"
    missing_cluster_path = inter_dir / "file11_missing_snp_cluster_exclusions.csv"
    e2_path = final_dir / "file11_external_strict_E2.csv"
    frozen_path = final_dir / "file11_external_strict_E2_FROZEN.csv"

    same_cluster.to_csv(same_cluster_path, index=False)
    missing_cluster.to_csv(missing_cluster_path, index=False)
    e2.to_csv(e2_path, index=False)
    e2.to_csv(frozen_path, index=False)

    stage5 = {
        "status": "PASS",
        "cluster_mapping_mode": "BIGQUERY_WEB_EXPORT",
        "internal_cluster_mapping_csv": str(cluster_csv),
        "internal_cluster_mapping_sha256": sha256_file(cluster_csv),
        "internal_cluster_count": len(internal_clusters),
        "external_clean_E1": len(e1),
        "external_clean_with_cluster": int((e1["erd_group"] != "").sum()),
        "external_clean_missing_cluster": len(missing_cluster),
        "excluded_same_snp_cluster": len(same_cluster),
        "external_strict_E2": len(e2),
        "output": str(e2_path),
    }
    atomic_json(ckpt_dir / "stage_05.done.json", stage5)

    log(
        f"[PROGRESS] 11B Stage 5 PASS | internal_clusters={len(internal_clusters)} | "
        f"same-cluster excluded={len(same_cluster)} | "
        f"missing-cluster excluded={len(missing_cluster)} | E2={len(e2)}"
    )

    # ------------------------------------------------------------------
    # Stage 6 — freeze E2
    # ------------------------------------------------------------------
    log("Stage 6/7 — freeze TRUE external-strict E2 cohort")
    cohort_hash = sha256_file(frozen_path)

    counts = (
        e2.groupby(["antibiotic", "label_binary"])
        .size()
        .reset_index(name="n")
        .sort_values(["antibiotic", "label_binary"])
    )
    counts_path = final_dir / "file11_external_strict_class_counts.csv"
    counts.to_csv(counts_path, index=False)

    rd = readiness(e2, args.min_per_class)
    readiness_path = final_dir / "file11_external_validation_readiness.csv"
    rd.to_csv(readiness_path, index=False)

    for _, r in rd.iterrows():
        log(
            f"[PROGRESS] 11B readiness {r['antibiotic']}: "
            f"n={int(r['n_total'])} | R={int(r['n_resistant'])} | "
            f"S={int(r['n_susceptible'])} | ready={bool(r['blind_validation_ready'])}"
        )

    for drug, sub in e2.groupby("antibiotic"):
        acc = sorted(set(norm(x) for x in sub["asm_acc"] if norm(x)))
        (final_dir / f"file11_{drug}_external_strict_assembly_accessions.txt").write_text(
            "\n".join(acc) + ("\n" if acc else "")
        )

    atomic_json(ckpt_dir / "stage_06.done.json", {
        "status": "PASS",
        "frozen_cohort": str(frozen_path),
        "frozen_cohort_sha256": cohort_hash,
        "rows": len(e2),
        "class_counts": str(counts_path),
        "readiness": str(readiness_path),
    })

    # ------------------------------------------------------------------
    # Stage 7 — patch final summary
    # ------------------------------------------------------------------
    log("Stage 7/7 — finalize external-validation readiness summary")
    summary_path = ckpt_dir / "file11_final_summary.json"
    old = {}
    if summary_path.exists():
        try:
            old = json.loads(summary_path.read_text())
        except Exception:
            old = {}

    ready_any = bool(rd["blind_validation_ready"].any()) if not rd.empty else False
    ready_all = bool(rd["blind_validation_ready"].all()) if not rd.empty else False

    summary = dict(old)
    summary.update({
        "script_version_cluster_completion": VERSION,
        "cluster_completion_design_id": DESIGN_ID,
        "status": "PASS_COHORT_FROZEN" if ready_any else "PARTIAL_REVIEW_REQUIRED",
        "completed_cluster_firewall_utc": pd.Timestamp.now("UTC").isoformat(),
        "cluster_firewall": {
            "mode": "BIGQUERY_WEB_EXPORT",
            "verified": True,
            "internal_cluster_count": len(internal_clusters),
            "external_clean_with_cluster": int((e1["erd_group"] != "").sum()),
            "external_clean_missing_cluster": len(missing_cluster),
            "excluded_same_snp_cluster": len(same_cluster),
        },
        "counts": {
            **old.get("counts", {}),
            "external_clean_E1": len(e1),
            "same_snp_cluster_excluded": len(same_cluster),
            "missing_snp_cluster_excluded_from_strict_E2": len(missing_cluster),
            "external_strict_E2": len(e2),
        },
        "frozen_external_cohort_sha256": cohort_hash,
        "blind_validation_readiness": {
            "min_per_class": args.min_per_class,
            "any_endpoint_ready": ready_any,
            "all_endpoints_ready": ready_all,
            "table": str(readiness_path),
        },
        "outputs": {
            **old.get("outputs", {}),
            "external_strict_E2": str(e2_path),
            "external_strict_E2_frozen": str(frozen_path),
            "class_counts": str(counts_path),
            "readiness": str(readiness_path),
            "snp_cluster_overlap_exclusions": str(same_cluster_path),
            "missing_snp_cluster_exclusions": str(missing_cluster_path),
            "internal_ncbi_snp_clusters": str(
                inter_dir / "file11_internal_ncbi_snp_clusters.csv"
            ),
        },
        "next_stage": (
            "Blindly reconstruct the ORIGINAL frozen feature representation on the "
            "TRUE E2 cohort and evaluate the ORIGINAL frozen model without retraining, "
            "feature reselection, hyperparameter tuning, or threshold tuning."
        ),
    })
    atomic_json(summary_path, summary)

    atomic_json(ckpt_dir / "stage_07.done.json", {
        "status": summary["status"],
        "summary": str(summary_path),
    })

    log("=" * 104)
    log(f"FILE11B STATUS : {summary['status']}")
    log(f"External-clean E1                  : {len(e1)}")
    log(f"Internal NCBI SNP clusters         : {len(internal_clusters)}")
    log(f"Same-cluster external excluded     : {len(same_cluster)}")
    log(f"Missing-cluster external excluded  : {len(missing_cluster)}")
    log(f"TRUE external-strict E2            : {len(e2)}")
    log(f"Frozen cohort SHA256               : {cohort_hash}")
    log(f"Frozen cohort                      : {frozen_path}")
    log(f"Final summary                      : {summary_path}")
    log("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
