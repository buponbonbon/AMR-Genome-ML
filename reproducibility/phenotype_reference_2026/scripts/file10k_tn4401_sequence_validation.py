#!/usr/bin/env python3
"""
FILE10K — blaKPC / Tn4401 sequence-level validation
Version: 1.0.0

Purpose
-------
Validate the blaKPC-associated mobile-element context discovered by FILE10J.

Evidence layers:
1) exact locus geometry from FILE10J
2) BV-BRC contig sequence retrieval + local cache
3) core Tn4401-like PLFam architecture
4) direct 99-bp upstream-deletion sequence comparison
5) conservative candidate 39-bp IR / 5-bp TSD scan when contig boundaries permit

Guardrail
---------
This script reports "Tn4401-like" and subtype-like calls from the available
sequence/context evidence. It does not establish causality or a novel resistance
mechanism. A definitive named transposon call should ideally be confirmed by
reference alignment / dedicated mobile-element annotation.

Expected input
--------------
<data root>/explainability_biology/external_annotation/
    file10j_unitig_locus_annotations.csv

Outputs
-------
data/explainability_biology/tn4401_validation/
    file10k_carrier_validation.csv
    file10k_intergenic_sequences.fasta
    file10k_locus_windows.fasta
    file10k_candidate_ir_tsd.csv

checkpoints/explainability_biology_tn4401_validation/
    file10k_final_summary.json
    sequence_cache/*.fasta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


VERSION = "1.0.0"
DESIGN_ID = "file10k_tn4401_sequence_validation_v1"
API_BASE = "https://www.bv-brc.org/api"

# PLFams inferred from the FILE10J locus evidence.
# Names are intentionally descriptive/guarded rather than asserted as exact IS names.
CORE_PLFAMS = {
    "iskpn6_like_transposase": "PLF_570_00004900",
    "blaKPC":                  "PLF_570_00004895",
    "iskpn7_istB_like":        "PLF_570_00004958",
    "iskpn7_istA_like":        "PLF_570_00004952",
    "tnpA_like":               "PLF_570_00004891",
}

IR_LEN = 39
TSD_LEN = 5
EXPECTED_A_DELETION_BP = 99


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def revcomp(seq: str) -> str:
    table = str.maketrans(
        "ACGTRYMKBDHVNacgtrymkbdhvn",
        "TGCAYRKMVHDBNtgcayrkmvhd bn".replace(" ", "")
    )
    return seq.translate(table)[::-1]


def parse_fasta(text: str) -> Tuple[str, str]:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError("BV-BRC response is not FASTA")
    header = lines[0][1:]
    seq = "".join(lines[1:]).replace(" ", "").upper()
    if not seq:
        raise ValueError("Empty FASTA sequence")
    return header, seq


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def fetch_contig(
    sequence_id: str,
    cache_dir: Path,
    timeout: int = 60,
) -> Tuple[str, str, bool]:
    """
    Fetch one genome_sequence record as DNA FASTA.
    Returns (header, sequence, cache_reused).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_name(sequence_id)}.fasta"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        header, seq = parse_fasta(cache_path.read_text())
        return header, seq, True

    url = (
        f"{API_BASE}/genome_sequence/"
        f"?eq(sequence_id,{sequence_id})"
        f"&http_accept=application/dna+fasta"
    )
    r = requests.get(
        url,
        timeout=timeout,
        headers={"Accept": "application/dna+fasta", "User-Agent": f"FILE10K/{VERSION}"},
    )
    r.raise_for_status()
    header, seq = parse_fasta(r.text)
    cache_path.write_text(f">{header}\n" + "\n".join(seq[i:i+80] for i in range(0, len(seq), 80)) + "\n")
    return header, seq, False


def fasta_record(header: str, seq: str, width: int = 80) -> str:
    return ">" + header + "\n" + "\n".join(seq[i:i+width] for i in range(0, len(seq), width)) + "\n"


def normalize_gid(v) -> str:
    # CSV may parse 573.29863 as numeric but its string representation is stable.
    if pd.isna(v):
        return ""
    return str(v)


def dedupe_patric_cds(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x[(x["annotation"] == "PATRIC") & (x["feature_type"] == "CDS")]
    return x.drop_duplicates(
        subset=[
            "Genome ID", "feature_sequence_id", "feature_start_1based",
            "feature_end_1based", "plfam_id", "product"
        ]
    )


def one_feature_by_plfam(df: pd.DataFrame, plfam: str) -> Optional[pd.Series]:
    x = df[df["plfam_id"] == plfam]
    if x.empty:
        return None
    # Prefer the longest span if duplicate annotation rows exist.
    spans = (x["feature_end_1based"] - x["feature_start_1based"] + 1).abs()
    return x.loc[spans.idxmax()]


def oriented_interval(seq: str, start1: int, end1: int, strand: str) -> str:
    """
    1-based inclusive genomic interval, oriented to `strand`.
    """
    s = seq[start1 - 1:end1]
    return revcomp(s) if strand == "-" else s


def intergenic_between(a: pd.Series, b: pd.Series) -> Tuple[int, int]:
    """
    Return genomic intergenic interval between non-overlapping features a and b.
    Empty interval encoded by start > end.
    """
    a1, a2 = int(a["feature_start_1based"]), int(a["feature_end_1based"])
    b1, b2 = int(b["feature_start_1based"]), int(b["feature_end_1based"])
    if a2 < b1:
        return a2 + 1, b1 - 1
    if b2 < a1:
        return b2 + 1, a1 - 1
    return 1, 0


def best_exact_length_deletion(long_seq: str, short_seq: str, deletion_len: int = 99) -> dict:
    """
    If lengths differ by exactly deletion_len, find the single deletion from long_seq
    that best reconstructs short_seq. Small SNP counts are tolerated and reported.
    """
    out = {
        "tested": False,
        "length_difference_bp": len(long_seq) - len(short_seq),
        "deletion_len_bp": deletion_len,
        "best_deletion_start_0based": None,
        "best_mismatch_count": None,
        "best_identity": None,
        "sequence_supported": False,
    }
    if len(long_seq) - len(short_seq) != deletion_len:
        return out

    out["tested"] = True
    best = None
    for i in range(0, len(long_seq) - deletion_len + 1):
        candidate = long_seq[:i] + long_seq[i + deletion_len:]
        if len(candidate) != len(short_seq):
            continue
        mism = sum(c1 != c2 for c1, c2 in zip(candidate, short_seq))
        if best is None or mism < best[0]:
            best = (mism, i)

    if best is None:
        return out

    mism, pos = best
    ident = 1.0 - (mism / max(1, len(short_seq)))
    out["best_deletion_start_0based"] = int(pos)
    out["best_mismatch_count"] = int(mism)
    out["best_identity"] = float(ident)
    # Conservative threshold: >=98% identity after one 99-bp deletion.
    out["sequence_supported"] = bool(ident >= 0.98)
    return out


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def scan_ir_tsd(
    oriented_window: str,
    kpc_start0: int,
    *,
    ir_len: int = IR_LEN,
    tsd_len: int = TSD_LEN,
    min_element_len: int = 8000,
    max_element_len: int = 12000,
    max_ir_mismatches: int = 8,
) -> Optional[dict]:
    """
    Conservative candidate scan only.
    Looks for two ~39-bp inverted repeats around blaKPC with separation consistent
    with a ~10-kb element, then checks whether the 5 bp immediately outside match.

    This is NOT a reference-alignment-based definitive boundary caller.
    """
    n = len(oriented_window)
    if n < min_element_len + 2 * tsd_len:
        return None

    # Search regions around KPC, avoiding enormous O(n^2) scans.
    left_lo = max(tsd_len, kpc_start0 - max_element_len)
    left_hi = max(left_lo, kpc_start0 - 1000)
    right_lo = min(n - ir_len - tsd_len, kpc_start0 + 1000)
    right_hi = min(n - ir_len - tsd_len, kpc_start0 + max_element_len)

    best = None
    for i in range(left_lo, left_hi + 1):
        irl = oriented_window[i:i+ir_len]
        if len(irl) != ir_len or "N" in irl:
            continue
        rc = revcomp(irl)

        # Restrict j by allowed element size.
        j0 = max(right_lo, i + min_element_len - ir_len)
        j1 = min(right_hi, i + max_element_len - ir_len)
        for j in range(j0, j1 + 1):
            irr = oriented_window[j:j+ir_len]
            if len(irr) != ir_len or "N" in irr:
                continue
            mism = hamming(rc, irr)
            if mism > max_ir_mismatches:
                continue

            left_tsd = oriented_window[i-tsd_len:i] if i >= tsd_len else ""
            right_tsd = oriented_window[j+ir_len:j+ir_len+tsd_len]
            tsd_match = (
                len(left_tsd) == tsd_len
                and len(right_tsd) == tsd_len
                and left_tsd == right_tsd
            )
            score = (0 if tsd_match else 1, mism, abs((j + ir_len - i) - 10000))
            rec = {
                "irl_start0": i,
                "irr_start0": j,
                "candidate_element_len_bp": j + ir_len - i,
                "ir_mismatches": mism,
                "left_tsd": left_tsd,
                "right_tsd": right_tsd,
                "tsd_match": tsd_match,
            }
            if best is None or score < best[0]:
                best = (score, rec)

    return None if best is None else best[1]


def architecture_signature(features: Dict[str, Optional[pd.Series]], strand: str) -> Tuple[str, bool]:
    """
    Build a strand-normalized order using feature midpoints.
    """
    entries = []
    for role, row in features.items():
        if row is None:
            continue
        mid = (int(row["feature_start_1based"]) + int(row["feature_end_1based"])) / 2
        entries.append((mid, role))

    entries.sort(reverse=(strand == "-"))
    sig = " > ".join(role for _, role in entries)
    complete = all(features.get(k) is not None for k in CORE_PLFAMS.keys())
    return sig, complete


def parse_args():
    p = argparse.ArgumentParser(description="FILE10K — blaKPC/Tn4401 sequence validation")
    p.add_argument(
        "--project-root",
        default="/mnt/c/Users/ASUS/Desktop/AMR-Genome-ML",
        help="AMR-Genome-ML project root",
    )
    p.add_argument(
        "--window-bp",
        type=int,
        default=12000,
        help="Flank on each side of blaKPC for sequence window extraction",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root)

    input_csv = (
        root / "data/explainability_biology/external_annotation/"
        "file10j_unitig_locus_annotations.csv"
    )
    out_dir = root / "data/explainability_biology/tn4401_validation"
    ckpt_dir = root / "checkpoints/explainability_biology_tn4401_validation"
    cache_dir = ckpt_dir / "sequence_cache"

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 96)
    log("FILE10K — blaKPC / Tn4401 SEQUENCE-LEVEL VALIDATION")
    log("=" * 96)
    log(f"Version       : {VERSION}")
    log(f"Project root  : {root}")
    log(f"Window        : ±{args.window_bp} bp around blaKPC")
    log("Guardrail     : subtype calls require locus + sequence evidence")

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing FILE10J locus file: {input_csv}")

    raw = pd.read_csv(input_csv)
    raw["Genome ID"] = raw["Genome ID"].map(normalize_gid)
    cds = dedupe_patric_cds(raw)

    kpc_rows = cds[cds["plfam_id"] == CORE_PLFAMS["blaKPC"]]
    if kpc_rows.empty:
        raise RuntimeError("No PLF_570_00004895 / blaKPC rows found")

    # One record per genome + contig.
    carriers = (
        kpc_rows.sort_values(["Genome ID", "feature_sequence_id"])
        .drop_duplicates(["Genome ID", "feature_sequence_id"])
    )

    log(f"Carrier loci  : {len(carriers)}")

    records: List[dict] = []
    intergenic_by_key: Dict[str, str] = {}
    windows_fasta: List[str] = []
    intergenic_fasta: List[str] = []
    ir_rows: List[dict] = []

    for idx, (_, kpc) in enumerate(carriers.iterrows(), start=1):
        gid = str(kpc["Genome ID"])
        seqid = str(kpc["feature_sequence_id"])
        strand = str(kpc["feature_strand"])
        mlst = kpc.get("MLST", "")
        local = cds[(cds["Genome ID"] == gid) & (cds["feature_sequence_id"] == seqid)]

        features: Dict[str, Optional[pd.Series]] = {
            role: one_feature_by_plfam(local, plfam)
            for role, plfam in CORE_PLFAMS.items()
        }

        log(f"[PROGRESS] carrier {idx}/{len(carriers)} | {gid} | {seqid}")

        header, contig, cache_reused = fetch_contig(seqid, cache_dir)
        contig_len = len(contig)

        kpc_start = int(kpc["feature_start_1based"])
        kpc_end = int(kpc["feature_end_1based"])
        win_start = max(1, kpc_start - args.window_bp)
        win_end = min(contig_len, kpc_end + args.window_bp)

        win_seq = contig[win_start - 1:win_end]
        if strand == "-":
            win_oriented = revcomp(win_seq)
            # 0-based start of KPC in oriented window
            kpc_start0 = win_end - kpc_end
        else:
            win_oriented = win_seq
            kpc_start0 = kpc_start - win_start

        windows_fasta.append(
            fasta_record(
                f"{gid}|{seqid}|KPC_window|{win_start}-{win_end}|KPC_strand={strand}",
                win_oriented,
            )
        )

        sig, complete_core = architecture_signature(features, strand)

        # Direct interval between blaKPC and the istB-like PLFam.
        istb = features["iskpn7_istB_like"]
        intergenic_seq = ""
        intergenic_len = None
        intergenic_start = None
        intergenic_end = None
        if istb is not None:
            ig_s, ig_e = intergenic_between(kpc, istb)
            intergenic_start, intergenic_end = ig_s, ig_e
            if ig_s <= ig_e:
                intergenic_seq = oriented_interval(contig, ig_s, ig_e, strand)
                intergenic_len = len(intergenic_seq)
                key = f"{gid}|{seqid}"
                intergenic_by_key[key] = intergenic_seq
                intergenic_fasta.append(
                    fasta_record(
                        f"{key}|blaKPC_to_istB_like|len={len(intergenic_seq)}|strand={strand}",
                        intergenic_seq,
                    )
                )

        # Conservative IR/TSD candidate scan.
        ir = scan_ir_tsd(win_oriented, kpc_start0)
        if ir is not None:
            ir_rows.append({
                "Genome ID": gid,
                "MLST": mlst,
                "feature_sequence_id": seqid,
                **ir,
                "note": "candidate scan only; confirm against a Tn4401 reference",
            })

        records.append({
            "Genome ID": gid,
            "MLST": mlst,
            "feature_sequence_id": seqid,
            "feature_accession": kpc.get("feature_accession", ""),
            "contig_length_bp": contig_len,
            "kpc_start_1based": kpc_start,
            "kpc_end_1based": kpc_end,
            "kpc_strand": strand,
            "kpc_product": kpc.get("product", ""),
            "core_architecture_signature": sig,
            "core_tn4401_like_complete": complete_core,
            "blaKPC_to_istB_intergenic_start_1based": intergenic_start,
            "blaKPC_to_istB_intergenic_end_1based": intergenic_end,
            "blaKPC_to_istB_intergenic_length_bp": intergenic_len,
            "sequence_cache_reused": cache_reused,
            "window_start_1based": win_start,
            "window_end_1based": win_end,
            "window_truncated_left": win_start == 1,
            "window_truncated_right": win_end == contig_len,
            "ir_tsd_candidate_found": ir is not None,
            "ir_candidate_mismatches": None if ir is None else ir["ir_mismatches"],
            "tsd_5bp_candidate_match": None if ir is None else ir["tsd_match"],
            **{
                f"has_{role}": features[role] is not None
                for role in CORE_PLFAMS
            },
        })

    result = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Direct 99-bp deletion validation across observed intergenic regions.
    # ------------------------------------------------------------------
    valid_lengths = result["blaKPC_to_istB_intergenic_length_bp"].dropna()
    longest_len = int(valid_lengths.max()) if not valid_lengths.empty else None

    result["reference_longest_observed_intergenic_bp"] = longest_len
    result["relative_deletion_from_longest_bp"] = None
    result["deletion_99_geometry_supported"] = False
    result["deletion_99_sequence_supported"] = False
    result["deletion_99_best_identity"] = None
    result["deletion_99_best_mismatch_count"] = None
    result["tn4401_subtype_like_call"] = "Tn4401-like; subtype unresolved"

    if longest_len is not None:
        long_rows = result[
            result["blaKPC_to_istB_intergenic_length_bp"] == longest_len
        ]
        # Prefer first longest observed carrier as internal comparison reference.
        long_ref = long_rows.iloc[0]
        long_key = f"{long_ref['Genome ID']}|{long_ref['feature_sequence_id']}"
        long_seq = intergenic_by_key.get(long_key, "")

        for i, row in result.iterrows():
            L = row["blaKPC_to_istB_intergenic_length_bp"]
            if pd.isna(L):
                continue
            L = int(L)
            delta = longest_len - L
            result.at[i, "relative_deletion_from_longest_bp"] = delta

            complete = bool(row["core_tn4401_like_complete"])

            if delta == EXPECTED_A_DELETION_BP and complete:
                result.at[i, "deletion_99_geometry_supported"] = True
                short_key = f"{row['Genome ID']}|{row['feature_sequence_id']}"
                short_seq = intergenic_by_key.get(short_key, "")
                ev = best_exact_length_deletion(
                    long_seq, short_seq, EXPECTED_A_DELETION_BP
                )
                result.at[i, "deletion_99_sequence_supported"] = ev["sequence_supported"]
                result.at[i, "deletion_99_best_identity"] = ev["best_identity"]
                result.at[i, "deletion_99_best_mismatch_count"] = ev["best_mismatch_count"]

                if ev["sequence_supported"]:
                    result.at[i, "tn4401_subtype_like_call"] = (
                        "Tn4401a-like; 99-bp upstream deletion sequence-supported"
                    )
                else:
                    result.at[i, "tn4401_subtype_like_call"] = (
                        "Tn4401a-like candidate; 99-bp geometry only"
                    )

            elif delta == 0 and complete:
                # Only call b-like if the dataset contains a sequence-supported
                # 99-bp-deleted counterpart. This prevents "largest observed = b"
                # from becoming a circular assumption.
                result.at[i, "tn4401_subtype_like_call"] = (
                    "Tn4401-like; longest observed upstream interval"
                )

        any_seq_supported_a = bool(result["deletion_99_sequence_supported"].fillna(False).any())
        if any_seq_supported_a:
            mask = (
                (result["relative_deletion_from_longest_bp"] == 0)
                & result["core_tn4401_like_complete"]
            )
            result.loc[mask, "tn4401_subtype_like_call"] = (
                "Tn4401b-like candidate; no 99-bp deletion relative to "
                "sequence-supported Tn4401a-like carrier(s)"
            )

    # Write outputs.
    validation_csv = out_dir / "file10k_carrier_validation.csv"
    intergenic_fa = out_dir / "file10k_intergenic_sequences.fasta"
    windows_fa = out_dir / "file10k_locus_windows.fasta"
    ir_csv = out_dir / "file10k_candidate_ir_tsd.csv"
    summary_json = ckpt_dir / "file10k_final_summary.json"

    result.to_csv(validation_csv, index=False)
    intergenic_fa.write_text("".join(intergenic_fasta))
    windows_fa.write_text("".join(windows_fasta))
    pd.DataFrame(ir_rows).to_csv(ir_csv, index=False)

    n_core = int(result["core_tn4401_like_complete"].sum())
    n_a_seq = int(result["deletion_99_sequence_supported"].fillna(False).sum())
    n_ir = int(result["ir_tsd_candidate_found"].fillna(False).sum())
    n_tsd = int(result["tsd_5bp_candidate_match"].fillna(False).sum())

    summary = {
        "script_version": VERSION,
        "design_id": DESIGN_ID,
        "status": "PASS",
        "carrier_loci": int(len(result)),
        "core_tn4401_like_complete": n_core,
        "sequence_supported_99bp_deletion_carriers": n_a_seq,
        "candidate_ir_pairs": n_ir,
        "candidate_5bp_tsd_matches": n_tsd,
        "guardrail": (
            "Tn4401-like/subtype-like calls are sequence/context interpretations. "
            "Definitive element naming should be confirmed by reference alignment "
            "or a dedicated mobile-element annotation method."
        ),
        "outputs": {
            "carrier_validation": str(validation_csv),
            "intergenic_sequences": str(intergenic_fa),
            "locus_windows": str(windows_fa),
            "candidate_ir_tsd": str(ir_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    log("-" * 96)
    log("FILE10K STATUS : PASS")
    log(f"Carrier loci                  : {len(result)}")
    log(f"Complete Tn4401-like core     : {n_core}")
    log(f"99-bp deletion seq-supported  : {n_a_seq}")
    log(f"Candidate IR pairs            : {n_ir}")
    log(f"Candidate matching 5-bp TSDs  : {n_tsd}")
    log(f"Validation table              : {validation_csv}")
    log(f"Final summary                 : {summary_json}")
    log("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
