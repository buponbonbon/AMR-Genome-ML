#!/usr/bin/env python3
"""
FILE14 — freeze the development-derived E3 sequence-compatibility envelope.

Version 1.0.1

SCIENTIFIC SEPARATION
=====================
Historical development QC is preserved exactly as documented:

    4,270 original genomes
    BV-BRC genome_quality == "Good" -> Keep
    4,233 retained, 37 excluded

That historical QC is NOT redefined here and must not be described as an
N50/contig/genome-size/QUAST rule.

The final common development/modeling cohort contains 4,227 genomes. The six
QC-qualified IDs absent from that common cohort are audited explicitly. The
sequence-compatibility reference used for E3 is therefore derived from the
4,227 common-development FASTAs, i.e. the sequence cohort actually represented
in the frozen modeling pipeline.

No phenotype, model prediction, E3 outcome, or validation metric is used.

Hard E3 compatibility envelope
===============================
Hard bounds are empirical extremes observed across all 4,227 common-development
FASTAs:

  total assembly length : within [development min, development max]
  contig count          : <= development max
  N50                   : >= development min
  largest contig        : >= development min
  ambiguous fraction    : <= development max

By construction, zero of the 4,227 reference FASTAs can fail these bounds.

A Tukey outer-fence summary (Q1 +/- 3*IQR) is also recorded for diagnostics
only. It does not alter pass/fail status.

This is a compatibility/reference envelope for heterogeneous E3 sequence
inputs. It is not a retrospective replacement for the historical BV-BRC
quality label.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VERSION = "1.0.1"

EXPECTED_ORIGINAL = 4270
EXPECTED_GOOD = 4233
EXPECTED_EXCLUDED = 37
EXPECTED_COMMON = 4227
EXPECTED_QC_ONLY = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def find_id_col(df: pd.DataFrame) -> str:
    aliases = [
        "Genome ID", "Genome ID String", "genome_id", "GenomeID",
        "genome id", "genome",
    ]
    by_lower = {str(c).strip().casefold(): c for c in df.columns}
    for a in aliases:
        if a.casefold() in by_lower:
            return by_lower[a.casefold()]
    raise RuntimeError(f"Cannot identify genome-ID column; columns={list(df.columns)[:50]}")


def load_ids(path: Path, expected_n: int, label: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    col = find_id_col(df)
    ids = [norm(x) for x in df[col] if norm(x)]
    if len(ids) != expected_n:
        raise RuntimeError(
            f"{label}: expected {expected_n} nonblank IDs, found {len(ids)} in {path}"
        )
    if len(set(ids)) != expected_n:
        raise RuntimeError(f"{label}: duplicate genome IDs detected in {path}")
    return df, col, ids


def find_fasta(root: Path, gid: str) -> Path | None:
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
            return p
    return None


def fasta_metrics(path: Path):
    opener = gzip.open if path.name.lower().endswith(".gz") else open

    lengths = []
    total = 0
    ambiguous = 0
    gc = 0

    seq_len = 0
    seq_ambig = 0
    seq_gc = 0

    def flush():
        nonlocal seq_len, seq_ambig, seq_gc, total, ambiguous, gc
        if seq_len:
            lengths.append(seq_len)
            total += seq_len
            ambiguous += seq_ambig
            gc += seq_gc
        seq_len = 0
        seq_ambig = 0
        seq_gc = 0

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                flush()
                continue
            s = "".join(line.split()).upper()
            if not s:
                continue
            seq_len += len(s)
            seq_gc += s.count("G") + s.count("C")
            seq_ambig += sum(ch not in {"A", "C", "G", "T"} for ch in s)
        flush()

    if not lengths or total <= 0:
        raise RuntimeError(f"Empty/invalid FASTA: {path}")

    lens = sorted(lengths, reverse=True)
    half = total / 2.0
    running = 0
    n50 = 0
    l50 = 0
    for i, L in enumerate(lens, start=1):
        running += L
        if running >= half:
            n50 = L
            l50 = i
            break

    return {
        "assembly_length_bp": int(total),
        "contig_count": int(len(lens)),
        "n50_bp": int(n50),
        "l50": int(l50),
        "largest_contig_bp": int(max(lens)),
        "ambiguous_bases": int(ambiguous),
        "ambiguous_fraction": float(ambiguous / total),
        "gc_fraction": float(gc / total),
        "contigs_ge_500": int(sum(L >= 500 for L in lens)),
        "contigs_ge_1000": int(sum(L >= 1000 for L in lens)),
    }


def outer_fence(series: pd.Series):
    q1 = float(series.quantile(0.25, interpolation="linear"))
    median = float(series.quantile(0.50, interpolation="linear"))
    q3 = float(series.quantile(0.75, interpolation="linear"))
    iqr = q3 - q1
    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "outer_lower": q1 - 3.0 * iqr,
        "outer_upper": q3 + 3.0 * iqr,
        "observed_min": float(series.min()),
        "observed_max": float(series.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path("/mnt/c/Users/ASUS/Desktop/AMR-Genome-ML"),
    )
    ap.add_argument(
        "--historical-good-membership",
        type=Path,
        default=None,
        help="4233-ID source. Default: data/features/known_amr/X_known_amr_final.csv",
    )
    ap.add_argument(
        "--common-cohort",
        type=Path,
        default=None,
        help="4227-ID common development cohort. Default: data/processed/file07_common_cohort.csv",
    )
    args = ap.parse_args()

    root = args.project_root.resolve()

    good_p = (
        args.historical_good_membership.resolve()
        if args.historical_good_membership
        else root / "data/features/known_amr/X_known_amr_final.csv"
    )
    common_p = (
        args.common_cohort.resolve()
        if args.common_cohort
        else root / "data/processed/file07_common_cohort.csv"
    )

    _, good_col, good_ids = load_ids(good_p, EXPECTED_GOOD, "Historical Good membership")
    _, common_col, common_ids = load_ids(common_p, EXPECTED_COMMON, "Common development cohort")

    good_set = set(good_ids)
    common_set = set(common_ids)

    qc_only = sorted(good_set - common_set)
    common_not_good = sorted(common_set - good_set)

    if len(qc_only) != EXPECTED_QC_ONLY:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_QC_ONLY} historical-Good IDs outside common cohort; "
            f"found {len(qc_only)}: {qc_only}"
        )
    if common_not_good:
        raise RuntimeError(
            "Common development cohort contains IDs absent from the 4,233 historical-Good set: "
            + ", ".join(common_not_good)
        )

    # Fail-fast FASTA audit BEFORE any expensive metric pass.
    fasta_map = {}
    missing_common = []
    for gid in common_ids:
        fp = find_fasta(root, gid)
        if fp is None:
            missing_common.append(gid)
        else:
            fasta_map[gid] = fp

    if missing_common:
        raise RuntimeError(
            f"{len(missing_common)} of {EXPECTED_COMMON} common-development FASTAs are missing:\n"
            + "\n".join(missing_common)
        )

    # Audit current local FASTA availability for the six QC-only IDs. Their
    # absence/presence is logged, but it is NOT used to infer why FILE07 omitted them.
    qc_only_fasta = {
        gid: (str(find_fasta(root, gid).resolve()) if find_fasta(root, gid) else "")
        for gid in qc_only
    }

    outdir = root / "data/external_validation/e3/qc_reference"
    resdir = root / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)

    metrics_p = outdir / "file14_development_common_4227_assembly_metrics.csv"
    env_p = outdir / "file14_E3_sequence_qc_envelope.json"
    log_p = outdir / "file14_E3_sequence_qc_reference.log"
    flag_p = outdir / "FILE14_QC_ENVELOPE_FROZEN.flag"
    status_p = resdir / "FILE14_STATUS_SUMMARY.log"

    # Avoid silently replacing a previously frozen reference.
    if flag_p.exists():
        raise RuntimeError(
            f"Freeze flag already exists: {flag_p}\n"
            "Refusing to overwrite a frozen QC reference silently. "
            "Inspect/archive the existing qc_reference directory first."
        )

    print("=" * 116)
    print("FILE14 — FREEZE DEVELOPMENT-DERIVED E3 SEQUENCE-COMPATIBILITY REFERENCE")
    print("=" * 116)
    print("Version                     :", VERSION)
    print("Historical Good source      :", good_p)
    print("Historical Good ID column   :", good_col)
    print("Historical Good n           :", len(good_ids))
    print("Common cohort source        :", common_p)
    print("Common cohort ID column     :", common_col)
    print("Common sequence reference n :", len(common_ids))
    print("Historical Good - Common    :", len(qc_only))
    print("Historical QC rule          : BV-BRC genome_quality == 'Good'")
    print("Phenotype used              : NO")
    print("E3 data used                : NO")
    print()
    print("Six historical-Good IDs outside common cohort:")
    for gid in qc_only:
        local = qc_only_fasta[gid] or "NO CURRENT LOCAL FASTA"
        print(f"  - {gid}: {local}")
    print()
    print(f"[PASS] Fail-fast audit found FASTA for all {EXPECTED_COMMON} common-development genomes.")
    print()

    rows = []
    for i, gid in enumerate(common_ids, start=1):
        fp = fasta_map[gid]
        m = fasta_metrics(fp)
        m["Genome ID"] = gid
        m["fasta_path"] = str(fp.resolve())
        m["fasta_sha256"] = sha256(fp)
        rows.append(m)

        if i % 250 == 0 or i == len(common_ids):
            print(f"[PROGRESS] {i}/{len(common_ids)} common-development FASTAs")

    d = pd.DataFrame(rows)
    if len(d) != EXPECTED_COMMON or d["Genome ID"].nunique() != EXPECTED_COMMON:
        raise RuntimeError("Metric table does not contain exactly 4,227 unique common-development genomes")

    d.to_csv(metrics_p, index=False)

    metric_names = [
        "assembly_length_bp",
        "contig_count",
        "n50_bp",
        "largest_contig_bp",
        "ambiguous_fraction",
    ]
    warnings = {m: outer_fence(d[m]) for m in metric_names}

    hard = {
        "assembly_length_bp": {
            "min": int(d["assembly_length_bp"].min()),
            "max": int(d["assembly_length_bp"].max()),
        },
        "contig_count": {"max": int(d["contig_count"].max())},
        "n50_bp": {"min": int(d["n50_bp"].min())},
        "largest_contig_bp": {"min": int(d["largest_contig_bp"].min())},
        "ambiguous_fraction": {"max": float(d["ambiguous_fraction"].max())},
    }

    passmask = (
        d["assembly_length_bp"].between(
            hard["assembly_length_bp"]["min"],
            hard["assembly_length_bp"]["max"],
            inclusive="both",
        )
        & d["contig_count"].le(hard["contig_count"]["max"])
        & d["n50_bp"].ge(hard["n50_bp"]["min"])
        & d["largest_contig_bp"].ge(hard["largest_contig_bp"]["min"])
        & d["ambiguous_fraction"].le(hard["ambiguous_fraction"]["max"])
    )
    ref_fail = int((~passmask).sum())
    if ref_fail != 0:
        raise RuntimeError(
            f"Internal invariant failed: {ref_fail} reference genomes fail empirical-extreme envelope"
        )

    envelope = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "E3 sequence-compatibility reference for heterogeneous public/local assemblies; "
            "does not redefine historical development QC"
        ),
        "historical_development_qc": {
            "original_n": EXPECTED_ORIGINAL,
            "retained_good_n": EXPECTED_GOOD,
            "excluded_n": EXPECTED_EXCLUDED,
            "rule": "BV-BRC genome_quality == 'Good' -> Keep; otherwise Exclude",
            "phenotype_used_for_qc": False,
        },
        "common_development_sequence_reference": {
            "n": EXPECTED_COMMON,
            "source": str(common_p),
            "source_sha256": sha256(common_p),
            "all_ids_subset_of_historical_good": True,
            "historical_good_minus_common_n": len(qc_only),
            "historical_good_minus_common_ids": qc_only,
            "note": (
                "The exact six-ID difference is audited here. This identity match does not, "
                "by itself, prove the original causal reason those IDs were absent from FILE07."
            ),
        },
        "historical_good_membership_source": str(good_p),
        "historical_good_membership_sha256": sha256(good_p),
        "current_local_fasta_status_for_six_qc_only_ids": qc_only_fasta,
        "development_metrics_csv": str(metrics_p),
        "development_metrics_csv_sha256": sha256(metrics_p),
        "hard_pass_fail_envelope": hard,
        "hard_rule": (
            "PASS iff assembly length is within observed common-development min/max, "
            "contig_count <= observed max, N50 >= observed min, largest_contig >= "
            "observed min, and ambiguous_fraction <= observed max."
        ),
        "reference_failures_under_hard_rule": ref_fail,
        "robust_warning_outer_fences": warnings,
        "warning_rule": (
            "Tukey outer fences (Q1 +/- 3*IQR) are diagnostic warnings only and do not "
            "override hard pass/fail status."
        ),
        "phenotype_used": False,
        "model_predictions_used": False,
        "e3_outcomes_used": False,
    }
    env_p.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    six_lines = "\n".join(
        f"  - {gid}: {qc_only_fasta[gid] or 'NO CURRENT LOCAL FASTA'}"
        for gid in qc_only
    )

    qc_log = f"""FILE14 E3 SEQUENCE-COMPATIBILITY REFERENCE FREEZE
{'='*116}
Version                           : {VERSION}

HISTORICAL DEVELOPMENT QC — PRESERVED
Original genomes                  : {EXPECTED_ORIGINAL}
BV-BRC Good retained              : {EXPECTED_GOOD}
Excluded                          : {EXPECTED_EXCLUDED}
Historical rule                   : BV-BRC genome_quality == "Good"
Phenotype used                    : NO

COMMON DEVELOPMENT SEQUENCE REFERENCE
Common/model cohort               : {EXPECTED_COMMON}
Historical Good minus Common      : {len(qc_only)}
Common minus Historical Good      : {len(common_not_good)}
All {EXPECTED_COMMON} FASTAs resolved        : YES

Exact six Historical-Good IDs outside common cohort:
{six_lines}

IMPORTANT INTERPRETATION
- The six-ID identity difference is exact.
- The current local FASTA audit and FILE07 membership agree numerically and by ID.
- This does NOT by itself establish the original causal reason those six IDs were
  absent from FILE07; therefore the log does not claim that missing FASTA caused
  their historical exclusion.
- Historical QC remains BV-BRC genome_quality == "Good" on 4,270 -> 4,233.
- The E3 sequence reference uses the 4,227 common-development FASTAs represented
  in the frozen modeling pipeline.
- No historical N50/contig/genome-size rule is being invented retrospectively.

REFERENCE OUTPUTS
Historical Good source            : {good_p}
Historical Good SHA256            : {sha256(good_p)}
Common cohort source              : {common_p}
Common cohort SHA256              : {sha256(common_p)}
Metrics CSV                       : {metrics_p}
Metrics SHA256                    : {sha256(metrics_p)}
Envelope JSON                     : {env_p}
Envelope SHA256                   : {sha256(env_p)}
Reference failures under hard rule: {ref_fail}

NEXT
1. Assemble the 55 E3 SRA-queue samples.
2. Obtain/localize FASTA for all 306 strict pre-assembly E3 candidates.
3. Apply this frozen compatibility envelope identically to every E3 FASTA.
4. Freeze the final E3 manifest only after sequence QC.
5. Require final gate n>=300, R>=100, S>=100 without relaxing QC.
6. Run blind FILE14 scoring with frozen FILE13 model/schema.
7. No tuning, recalibration, feature reselection, or performance-based sample selection.
{'='*116}
"""
    log_p.write_text(qc_log, encoding="utf-8")

    status = f"""FILE14 STATUS SUMMARY
{'='*116}
CURRENT STATE
- FILE13 harmonized development/model freeze: COMPLETE and FROZEN.
- Historical source-level development QC: 4,270 -> 4,233 BV-BRC Good; 37 excluded.
- Common development/model cohort: 4,227.
- Exact Historical-Good minus Common difference: 6 IDs.
- All 4,227 common-development FASTAs are locally resolved for the E3 sequence reference.
- Initial NCBI E3 candidate preparation: 714 (302 R / 412 S).
- Frozen E2 firewall correctly restricted to MEROPENEM: 525 (163 R / 362 S).
- Missing SNP/ERD cluster treated as unclustered rather than automatically invalid.
- Stable PDS-base firewall adopted.
- Strict NCBI-v2 survivors: 244 (96 R / 148 S).
- Missing-assembly/SRA recovery strict survivors: 54.
- BV-BRC supplementary acquisition: 3,944 phenotype-eligible candidates before firewall.
- BV-BRC strict supplementary survivors: 8 (0 R / 8 S).
- Strict PRE-ASSEMBLY E3 pool: 306 (122 R / 184 S).
- Assembly-ready: 251 (96 R / 155 S).
- SRA assembly queue: 55 (26 R / 29 S).
- Pre-assembly gate n>=300, R>=100, S>=100: PASS.

HISTORICAL DEVELOPMENT QC — DO NOT REWRITE
- Rule: BV-BRC genome_quality == "Good" -> Keep.
- Retained 4,233 from 4,270.
- Phenotype was not used.
- Do NOT claim historical development QC used QUAST, N50, genome-size,
  completeness, contamination, or contig thresholds.

E3 SEQUENCE-COMPATIBILITY REFERENCE
- Reference cohort: 4,227 common-development FASTAs.
- These 4,227 IDs are an exact subset of the 4,233 Historical-Good IDs.
- Historical-Good minus Common = 6; exact IDs recorded in the QC reference log.
- Empirical-extreme bounds are frozen before E3 QC/performance inspection.
- Tukey outer fences are diagnostics only.
- No phenotype or model performance is used.
- The six-ID identity match does not by itself prove why those six were originally
  absent from FILE07; no unsupported causal claim should be made.

WHAT MUST HAPPEN NEXT
1. Assemble the 55 SRA queue samples.
2. Collect FASTA for all 306 strict pre-assembly E3 candidates.
3. Apply the frozen sequence-compatibility envelope identically to all E3 FASTAs.
4. Do not relax the envelope to rescue sample count.
5. Freeze final E3 manifest only if final n>=300, R>=100, S>=100.
6. Run FILE14 blind validation with frozen FILE13 model and 702-feature schema.
7. No tuning/recalibration/reselection/performance-based E3 selection.
8. After FILE14, run FILE15 integrity/reporting only.

PROVENANCE
- Earlier FILE14 diagnostic helpers remain audit/history.
- Forward production pipeline: strict cohort builder -> sequence reference freeze
  -> final FILE14 assembly/QC/freeze/blind-validation script.
- Existing frozen manifests/checkpoints must not be overwritten silently.
{'='*116}
"""
    status_p.write_text(status, encoding="utf-8")

    flag = {
        "status": "PASS_FILE14_QC_REFERENCE_FROZEN",
        "version": VERSION,
        "historical_original_n": EXPECTED_ORIGINAL,
        "historical_good_n": EXPECTED_GOOD,
        "common_reference_n": EXPECTED_COMMON,
        "historical_good_minus_common_n": len(qc_only),
        "historical_good_minus_common_ids": qc_only,
        "reference_failures_under_hard_rule": ref_fail,
        "historical_good_source_sha256": sha256(good_p),
        "common_cohort_sha256": sha256(common_p),
        "metrics_sha256": sha256(metrics_p),
        "envelope_sha256": sha256(env_p),
        "qc_log_sha256": sha256(log_p),
        "status_summary_sha256": sha256(status_p),
    }
    flag_p.write_text(json.dumps(flag, indent=2), encoding="utf-8")

    print()
    print("[PASS] Historical development QC preserved: 4,270 -> 4,233 BV-BRC Good.")
    print("[PASS] Common development sequence reference verified: 4,227/4,227 FASTAs.")
    print("[PASS] Historical Good - Common verified: exactly 6 IDs.")
    print("[PASS] Reference failures under hard compatibility envelope: 0")
    print("[PASS] E3 sequence-compatibility reference frozen before E3 QC/scoring.")
    print()
    print("Metrics CSV    :", metrics_p)
    print("Envelope JSON  :", env_p)
    print("QC log         :", log_p)
    print("Status summary :", status_p)
    print("Freeze flag    :", flag_p)
    print("Envelope SHA256:", sha256(env_p))
    print("=" * 116)
    print("STATUS         : PASS_FILE14_QC_REFERENCE_FROZEN")


if __name__ == "__main__":
    main()
