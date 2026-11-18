#!/usr/bin/env python3
"""
FILE14 — build the strict E3 pre-assembly cohort (consolidated helper)

This consolidates the exploratory FILE14 firewall/recovery logic into one
reproducible cohort-building step. It NEVER reads model predictions and NEVER
selects genomes by model performance.

Two-pass workflow
-----------------
PASS 1:
  If the BV-BRC -> NCBI Pathogen Detection map is absent, emit one BigQuery SQL
  file containing ALL unique BV-BRC BioSamples and exit with
  STATUS: NEED_BVBRC_NCBI_MAP.

PASS 2:
  After that SQL is run and exported as:
    data/external_validation/e3/source_bvbrc/bvbrc_ncbi_pathogen_map.csv
  rerun the SAME command.

The script then:
  A) reconstructs the NCBI-derived E3-v2 pool:
       714 prepared candidates + 269 candidates rejected ONLY for missing SNP/ERD
  B) applies strict E2-meropenem + development firewalls using:
       Genome/target identity, Assembly accession base, BioSample, stable PDS base
  C) reconstructs the 213 missing-assembly NCBI candidates, uses recovered
     target/SRA/PDS metadata, and applies the same strict firewall
  D) maps the complete BV-BRC supplementary tranche to NCBI Pathogen Detection,
     applies the same strict E2/development firewall, excludes overlap with the
     ENTIRE original NCBI E3 source, then deterministically collapses duplicate
     biological identities without using phenotype prevalence or model output
  E) outputs a strict PRE-ASSEMBLY pool plus an SRA assembly queue.

Important:
  - blank SNP/ERD means unclustered and never overlaps another blank;
  - PDS/ERD comparison uses the stable accession base (e.g. PDS000012345);
  - this script does NOT freeze the final E3 cohort because SRA-only candidates
    still need assembly/QC;
  - final freeze happens only after assembly/QC is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

VERSION = "1.0.0"


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def accession_base(x):
    s = norm(x)
    if not s:
        return ""
    return s.rsplit(".", 1)[0] if "." in s else s


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require_cols(df, cols, label):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"{label} missing columns: {miss}; columns={list(df.columns)}")


def nonblank_set(series, base=False):
    f = accession_base if base else norm
    return {f(x) for x in series if f(x)}


def emit_bvbrc_sql(bv: pd.DataFrame, out: Path, source_path: Path):
    bios = sorted({norm(x) for x in bv["BioSample"] if norm(x)})
    if not bios:
        raise RuntimeError("BV-BRC tranche contains no BioSamples; cannot build NCBI map")

    quoted = ",\n    ".join("'" + x.replace("'", "''") + "'" for x in bios)
    sql = f"""-- FILE14 consolidated strict-E3 builder: BV-BRC -> NCBI Pathogen Detection map
-- BV-BRC source: {source_path}
-- BV-BRC source SHA256: {sha256(source_path)}
-- Unique BioSamples requested: {len(bios)}
--
-- Run this query in Google BigQuery and export the complete result as:
--   data/external_validation/e3/source_bvbrc/bvbrc_ncbi_pathogen_map.csv

WITH requested AS (
  SELECT biosample_acc
  FROM UNNEST([
    {quoted}
  ]) AS biosample_acc
)
SELECT DISTINCT
  i.target_acc,
  i.biosample_acc,
  i.asm_acc,
  i.erd_group,
  i.taxgroup_name,
  i.scientific_name
FROM `ncbi-pathogen-detect.pdbrowser.isolates` AS i
JOIN requested AS r
  ON UPPER(i.biosample_acc) = r.biosample_acc
ORDER BY i.biosample_acc, i.target_acc, i.asm_acc, i.erd_group;
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sql, encoding="utf-8")
    return len(bios)


def build_reference_sets(e2, devx, devbq):
    # Frozen E2 MUST be meropenem only.
    e2m = e2[e2["antibiotic"].str.strip().str.casefold().eq("meropenem")].copy()
    if len(e2m) != 525 or e2m["target_acc"].map(norm).nunique() != 525:
        raise RuntimeError(
            f"E2 meropenem invariant failed: rows={len(e2m)}, "
            f"unique IDs={e2m['target_acc'].map(norm).nunique()} (expected 525/525)"
        )

    refs = {
        "e2_id": nonblank_set(e2m["target_acc"]),
        "e2_id_base": nonblank_set(e2m["target_acc"], base=True),
        "e2_asm": nonblank_set(e2m["asm_acc"]),
        "e2_asm_base": nonblank_set(e2m["asm_acc"], base=True),
        "e2_bio": nonblank_set(e2m["biosample_acc"]),
        "e2_erd_base": nonblank_set(e2m["erd_group"], base=True),
        "dev_id": nonblank_set(devx["genome_id"]),
        "dev_id_base": nonblank_set(devx["genome_id"], base=True),
        "dev_target": nonblank_set(devbq["target_acc"]),
        "dev_target_base": nonblank_set(devbq["target_acc"], base=True),
        "dev_asm": nonblank_set(devx["asm_acc"]) | nonblank_set(devbq["asm_acc"]),
        "dev_asm_base": (
            nonblank_set(devx["asm_acc"], base=True)
            | nonblank_set(devbq["asm_acc"], base=True)
        ),
        "dev_bio": nonblank_set(devx["biosample_acc"]) | nonblank_set(devbq["biosample_acc"]),
        "dev_erd_base": nonblank_set(devbq["erd_group"], base=True),
    }
    return e2m, refs


def apply_strict_firewall(df, refs, *, gid_col, asm_col, bio_col, erd_col, prefix=""):
    gid = df[gid_col].map(norm)
    gidb = df[gid_col].map(accession_base)
    asm = df[asm_col].map(norm)
    asmb = df[asm_col].map(accession_base)
    bio = df[bio_col].map(norm)
    erdb = df[erd_col].map(accession_base)

    e2 = (
        gid.isin(refs["e2_id"])
        | (gidb.ne("") & gidb.isin(refs["e2_id_base"]))
        | (asm.ne("") & asm.isin(refs["e2_asm"]))
        | (asmb.ne("") & asmb.isin(refs["e2_asm_base"]))
        | bio.isin(refs["e2_bio"])
        | (erdb.ne("") & erdb.isin(refs["e2_erd_base"]))
    )
    dev = (
        gid.isin(refs["dev_id"] | refs["dev_target"])
        | (gidb.ne("") & gidb.isin(refs["dev_id_base"] | refs["dev_target_base"]))
        | (asm.ne("") & asm.isin(refs["dev_asm"]))
        | (asmb.ne("") & asmb.isin(refs["dev_asm_base"]))
        | bio.isin(refs["dev_bio"])
        | (erdb.ne("") & erdb.isin(refs["dev_erd_base"]))
    )

    df[f"{prefix}E2_OVERLAP"] = e2
    df[f"{prefix}DEV_OVERLAP"] = dev
    df[f"{prefix}STRICT_OVERLAP"] = e2 | dev
    return df


def original_ncbi_source_sets(src):
    return {
        "id": nonblank_set(src["Isolate"]),
        "id_base": nonblank_set(src["Isolate"], base=True),
        "asm": nonblank_set(src["Assembly"]),
        "asm_base": nonblank_set(src["Assembly"], base=True),
        "bio": nonblank_set(src["BioSample"]),
        "erd_base": nonblank_set(src["SNP cluster"], base=True),
    }


def flag_original_ncbi_overlap(df, source_sets):
    gid = df["Genome ID"].map(norm)
    gidb = df["Genome ID"].map(accession_base)
    asm = df["Assembly Accession"].map(norm)
    asmb = df["Assembly Accession"].map(accession_base)
    bio = df["BioSample"].map(norm)
    erdb = df["NCBI SNP/ERD Group"].map(accession_base)

    return (
        gid.isin(source_sets["id"])
        | (gidb.ne("") & gidb.isin(source_sets["id_base"]))
        | (asm.ne("") & asm.isin(source_sets["asm"]))
        | (asmb.ne("") & asmb.isin(source_sets["asm_base"]))
        | bio.isin(source_sets["bio"])
        | (erdb.ne("") & erdb.isin(source_sets["erd_base"]))
    )


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def collapse_biological_duplicates(df):
    """
    Connect rows sharing any strong biological identity:
      BioSample, Assembly-base, NCBI target accession-base, SRA Run.
    Conflicting-phenotype components are excluded.
    Otherwise choose one deterministic representative:
      has Assembly > has SRA > lexicographically smallest Genome ID.
    """
    d = df.reset_index(drop=True).copy()
    n = len(d)
    dsu = DSU(n)

    identity_specs = [
        ("BioSample", norm),
        ("Assembly Accession", accession_base),
        ("NCBI Target", accession_base),
        ("SRA Run", norm),
    ]
    for col, f in identity_specs:
        seen = {}
        for i, x in enumerate(d[col]):
            k = f(x)
            if not k:
                continue
            if k in seen:
                dsu.union(i, seen[k])
            else:
                seen[k] = i

    comps = {}
    for i in range(n):
        comps.setdefault(dsu.find(i), []).append(i)

    keep_idx = []
    conflict_idx = []
    duplicate_nonrep_idx = []

    for idxs in comps.values():
        g = d.loc[idxs].copy()
        ph = sorted(set(g["Phenotype"].map(norm)))
        if len(ph) != 1:
            conflict_idx.extend(idxs)
            continue
        g["_has_asm"] = g["Assembly Accession"].map(norm).ne("").astype(int)
        g["_has_sra"] = g["SRA Run"].map(norm).ne("").astype(int)
        g["_gid_sort"] = g["Genome ID"].map(norm)
        g = g.sort_values(
            ["_has_asm", "_has_sra", "_gid_sort"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        rep = int(g.index[0])
        keep_idx.append(rep)
        duplicate_nonrep_idx.extend([i for i in idxs if i != rep])

    status = pd.Series("KEEP", index=d.index, dtype="object")
    status.loc[conflict_idx] = "EXCLUDE_PHENOTYPE_CONFLICT_COMPONENT"
    status.loc[duplicate_nonrep_idx] = "EXCLUDE_DUPLICATE_NONREPRESENTATIVE"
    d["DEDUP_STATUS"] = status
    keep = d.loc[sorted(keep_idx)].copy()
    return d, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path("/mnt/c/Users/ASUS/Desktop/AMR-Genome-ML"),
    )
    args = ap.parse_args()
    root = args.project_root.resolve()

    # Inputs
    e3_manifest_p = root/"data/external_validation/e3/e3_candidate_manifest.csv"
    rejected_p = root/"data/external_validation/e3/file14_prepare/file14a_rejected_rows.csv"
    ncbi_source_p = root/"data/external_validation/e3/source/ncbi_kp_meropenem_candidates.csv"
    sra_recovery_p = root/"data/external_validation/e3/e3_missing_assembly_sra_recovery.csv"
    e2_p = root/"data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv"
    devx_p = root/"data/external_validation/ncbi_pathogen_detection/intermediate/file11_internal_identifier_crosswalk.csv"
    devbq_p = root/"data/external_validation/e3/development_ncbi_firewall.csv"
    bv_p = root/"data/external_validation/e3/source_bvbrc/bvbrc_kp_meropenem_candidate_tranche.csv"
    bvmap_p = root/"data/external_validation/e3/source_bvbrc/bvbrc_ncbi_pathogen_map.csv"
    bvsql_p = root/"results/file14_bvbrc_ncbi_pathogen_map.sql"

    for p in [e3_manifest_p, rejected_p, ncbi_source_p, sra_recovery_p,
              e2_p, devx_p, devbq_p, bv_p]:
        if not p.is_file():
            raise FileNotFoundError(p)

    e3 = pd.read_csv(e3_manifest_p, dtype=str, keep_default_na=False)
    rej = pd.read_csv(rejected_p, dtype=str, keep_default_na=False)
    ncbi_src = pd.read_csv(ncbi_source_p, dtype=str, keep_default_na=False)
    sr = pd.read_csv(sra_recovery_p, dtype=str, keep_default_na=False)
    e2 = pd.read_csv(e2_p, dtype=str, keep_default_na=False)
    devx = pd.read_csv(devx_p, dtype=str, keep_default_na=False)
    devbq = pd.read_csv(devbq_p, dtype=str, keep_default_na=False)
    bv = pd.read_csv(bv_p, dtype=str, keep_default_na=False)

    require_cols(
        bv,
        ["BV-BRC Genome ID","Phenotype","Assembly Accession","BioSample","SRA Run"],
        "BV-BRC tranche",
    )

    print("="*116)
    print("FILE14 — CONSOLIDATED STRICT E3 PRE-ASSEMBLY COHORT BUILDER")
    print("="*116)
    print("Version:", VERSION)

    # Pass 1: emit mapping SQL.
    if not bvmap_p.is_file():
        n = emit_bvbrc_sql(bv, bvsql_p, bv_p)
        print("BV-BRC candidate genomes     :", len(bv))
        print("Unique BV-BRC BioSamples     :", n)
        print("BigQuery SQL                 :", bvsql_p)
        print("SQL SHA256                   :", sha256(bvsql_p))
        print("Expected export              :", bvmap_p)
        print("="*116)
        print("STATUS                       : NEED_BVBRC_NCBI_MAP")
        return

    bvmap = pd.read_csv(bvmap_p, dtype=str, keep_default_na=False)
    require_cols(bvmap, ["target_acc","biosample_acc","asm_acc","erd_group"], "BV-BRC NCBI map")

    require_cols(
        ncbi_src,
        ["Isolate","Assembly","BioSample","SNP cluster","AST phenotypes"],
        "Original NCBI E3 source",
    )
    require_cols(e2, ["target_acc","antibiotic","asm_acc","biosample_acc","erd_group"], "E2")
    require_cols(devx, ["genome_id","asm_acc","biosample_acc"], "Development crosswalk")
    require_cols(devbq, ["target_acc","asm_acc","biosample_acc","erd_group"], "Development NCBI map")

    e2m, refs = build_reference_sets(e2, devx, devbq)
    ncbi_source_sets = original_ncbi_source_sets(ncbi_src)

    common_cols = [
        "Genome ID","Phenotype","Assembly Accession","BioSample",
        "NCBI SNP/ERD Group","SRA Run","Source","Needs Assembly"
    ]

    # ----------------------------------------------------------------------
    # A) NCBI E3-v2 = 714 + 269 unclustered-only rescues
    # ----------------------------------------------------------------------
    base_cols = ["Genome ID","Phenotype","Assembly Accession","BioSample","NCBI SNP/ERD Group"]
    require_cols(e3, base_cols, "Current E3")
    require_cols(rej, base_cols + ["rejection_reason"], "Rejected rows")

    rescue_cluster = rej[
        rej["Phenotype"].isin(["Resistant","Susceptible"])
        & rej["rejection_reason"].eq("missing_ncbi_cluster")
    ][base_cols].copy()

    ncbi_v2 = pd.concat([e3[base_cols], rescue_cluster], ignore_index=True)
    ncbi_v2["SRA Run"] = ""
    ncbi_v2["Source"] = "NCBI_E3_ORIGINAL"
    ncbi_v2["Needs Assembly"] = False
    ncbi_v2 = apply_strict_firewall(
        ncbi_v2, refs,
        gid_col="Genome ID", asm_col="Assembly Accession",
        bio_col="BioSample", erd_col="NCBI SNP/ERD Group",
        prefix="",
    )
    ncbi_keep = ncbi_v2[~ncbi_v2["STRICT_OVERLAP"]].copy()

    # ----------------------------------------------------------------------
    # B) 213 NCBI missing-assembly recovery rows
    # ----------------------------------------------------------------------
    missing = rej[rej["rejection_reason"].str.contains("missing_assembly", regex=False)].copy()
    missing["_bio"] = missing["BioSample"].map(norm)
    sr["_bio"] = sr["biosample_acc"].map(norm)
    if sr["_bio"].duplicated().any():
        raise RuntimeError("SRA recovery has duplicate BioSamples")
    m = missing.merge(
        sr[["_bio","target_acc","sra_run","erd_group"]],
        on="_bio", how="inner", validate="one_to_one"
    )
    if len(m) != len(missing):
        raise RuntimeError(f"Missing-assembly recovery incomplete: {len(m)}/{len(missing)}")

    cluster_recovered = m["erd_group"].map(norm).where(
        m["erd_group"].map(norm).ne(""),
        m["NCBI SNP/ERD Group"].map(norm)
    )
    orig_cl = m["NCBI SNP/ERD Group"].map(accession_base)
    rec_cl = m["erd_group"].map(accession_base)
    conflict = orig_cl.ne("") & rec_cl.ne("") & orig_cl.ne(rec_cl)
    if conflict.any():
        raise RuntimeError("Original vs recovered stable PDS base conflict in missing-assembly rows")

    miss_pool = pd.DataFrame({
        "Genome ID": m["target_acc"].map(norm),
        "Phenotype": m["Phenotype"],
        "Assembly Accession": "",
        "BioSample": m["BioSample"],
        "NCBI SNP/ERD Group": cluster_recovered,
        "SRA Run": m["sra_run"].map(norm),
        "Source": "NCBI_E3_SRA_RECOVERY",
        "Needs Assembly": True,
    })
    miss_pool = apply_strict_firewall(
        miss_pool, refs,
        gid_col="Genome ID", asm_col="Assembly Accession",
        bio_col="BioSample", erd_col="NCBI SNP/ERD Group",
        prefix="",
    )
    miss_keep = miss_pool[~miss_pool["STRICT_OVERLAP"]].copy()

    # ----------------------------------------------------------------------
    # C) BV-BRC supplementary tranche -> NCBI map -> strict firewall
    # ----------------------------------------------------------------------
    # Restrict map to requested BV-BRC BioSamples and collapse only if metadata
    # are identical. Ambiguous BioSamples with multiple distinct NCBI identities
    # are excluded conservatively.
    bv["_bio"] = bv["BioSample"].map(norm)
    bvmap["_bio"] = bvmap["biosample_acc"].map(norm)
    requested = set(bv["_bio"]) - {""}
    bvm = bvmap[bvmap["_bio"].isin(requested)].copy()

    grouped = []
    ambiguous_bio = set()
    for bio, g in bvm.groupby("_bio", sort=True):
        targets = sorted({norm(x) for x in g["target_acc"] if norm(x)})
        asms = sorted({norm(x) for x in g["asm_acc"] if norm(x)})
        erds = sorted({accession_base(x) for x in g["erd_group"] if accession_base(x)})
        # Multiple target rows are acceptable only if they do not disagree on
        # assembly and stable ERD identity. Otherwise exclude that BioSample.
        if len(asms) > 1 or len(erds) > 1:
            ambiguous_bio.add(bio)
            continue
        grouped.append({
            "_bio": bio,
            "NCBI Target": targets[0] if len(targets) == 1 else "",
            "NCBI Assembly": asms[0] if len(asms) == 1 else "",
            "NCBI SNP/ERD Group": erds[0] if len(erds) == 1 else "",
        })
    map1 = pd.DataFrame(grouped)

    bvj = bv.merge(map1, on="_bio", how="left")
    bvj["_ambiguous_ncbi_map"] = bvj["_bio"].isin(ambiguous_bio)

    # Prefer BV-BRC assembly; if absent, recover NCBI assembly from the mapped BioSample.
    assembly = bvj["Assembly Accession"].map(norm)
    if "NCBI Assembly" not in bvj:
        bvj["NCBI Assembly"] = ""
    recovered_asm = bvj["NCBI Assembly"].map(norm)
    final_asm = assembly.where(assembly.ne(""), recovered_asm)

    if "NCBI Target" not in bvj:
        bvj["NCBI Target"] = ""
    if "NCBI SNP/ERD Group" not in bvj:
        bvj["NCBI SNP/ERD Group"] = ""

    # Use mapped NCBI target when available; otherwise retain BV-BRC ID in
    # Genome ID and leave NCBI Target separately for audit.
    target = bvj["NCBI Target"].map(norm)
    genome_id = target.where(target.ne(""), bvj["BV-BRC Genome ID"].map(norm))

    bvp = pd.DataFrame({
        "Genome ID": genome_id,
        "Phenotype": bvj["Phenotype"],
        "Assembly Accession": final_asm,
        "BioSample": bvj["BioSample"],
        "NCBI SNP/ERD Group": bvj["NCBI SNP/ERD Group"].map(norm),
        "SRA Run": bvj["SRA Run"].map(norm),
        "Source": "BVBRC_SUPPLEMENTARY",
        "Needs Assembly": final_asm.eq("") & bvj["SRA Run"].map(norm).ne(""),
        "BV-BRC Genome ID": bvj["BV-BRC Genome ID"],
        "NCBI Target": bvj["NCBI Target"],
        "Ambiguous NCBI Map": bvj["_ambiguous_ncbi_map"],
    })

    # Must have a BioSample, and either an assembly or an SRA run.
    bvp["Metadata Eligible"] = (
        bvp["BioSample"].map(norm).ne("")
        & (
            bvp["Assembly Accession"].map(norm).ne("")
            | bvp["SRA Run"].map(norm).ne("")
        )
        & ~bvp["Ambiguous NCBI Map"]
    )

    bvp = apply_strict_firewall(
        bvp, refs,
        gid_col="Genome ID", asm_col="Assembly Accession",
        bio_col="BioSample", erd_col="NCBI SNP/ERD Group",
        prefix="",
    )
    bvp["OVERLAP_ORIGINAL_NCBI_E3_SOURCE"] = flag_original_ncbi_overlap(bvp, ncbi_source_sets)

    bv_eligible = bvp[
        bvp["Metadata Eligible"]
        & ~bvp["STRICT_OVERLAP"]
        & ~bvp["OVERLAP_ORIGINAL_NCBI_E3_SOURCE"]
    ].copy()

    bv_dedup_audit, bv_keep = collapse_biological_duplicates(bv_eligible)

    # ----------------------------------------------------------------------
    # D) Combine pre-assembly pool. Sources are disjoint by construction:
    #    BV-BRC was explicitly excluded against entire original NCBI E3 source.
    # ----------------------------------------------------------------------
    for d in [ncbi_keep, miss_keep]:
        for c in common_cols:
            if c not in d.columns:
                d[c] = ""

    for c in common_cols:
        if c not in bv_keep.columns:
            bv_keep[c] = ""

    pool = pd.concat(
        [
            ncbi_keep[common_cols],
            miss_keep[common_cols],
            bv_keep[common_cols],
        ],
        ignore_index=True,
    )

    # Final conservative identity-collapse across combined survivors.
    pool["NCBI Target"] = pool["Genome ID"]
    combined_audit, combined_keep = collapse_biological_duplicates(pool)

    # Any cross-source phenotype conflict is fatal for that component but not
    # hidden; collapse_biological_duplicates excludes it and records status.
    final_preassembly = combined_keep[common_cols].copy()

    # SRA queue: only samples without an assembly but with SRA.
    queue = final_preassembly[
        final_preassembly["Assembly Accession"].map(norm).eq("")
        & final_preassembly["SRA Run"].map(norm).ne("")
    ].copy()

    # Samples already possessing an assembly.
    ready = final_preassembly[
        final_preassembly["Assembly Accession"].map(norm).ne("")
    ].copy()

    # ----------------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------------
    outdir = root/"data/external_validation/e3/strict_build"
    resdir = root/"results"
    outdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)

    pre_p = outdir/"file14_strict_E3_preassembly_pool.csv"
    ready_p = outdir/"file14_strict_E3_assembly_ready.csv"
    queue_p = outdir/"file14_strict_E3_sra_assembly_queue.csv"
    bv_surv_p = outdir/"file14_bvbrc_strict_supplementary_survivors.csv"

    final_preassembly.to_csv(pre_p, index=False)
    ready.to_csv(ready_p, index=False)
    queue.to_csv(queue_p, index=False)
    bv_keep[common_cols].to_csv(bv_surv_p, index=False)

    ncbi_v2.to_csv(resdir/"file14_build_audit_ncbi_v2.csv", index=False)
    miss_pool.to_csv(resdir/"file14_build_audit_ncbi_missing_assembly.csv", index=False)
    bvp.to_csv(resdir/"file14_build_audit_bvbrc_before_dedup.csv", index=False)
    bv_dedup_audit.to_csv(resdir/"file14_build_audit_bvbrc_dedup.csv", index=False)
    combined_audit.to_csv(resdir/"file14_build_audit_combined_dedup.csv", index=False)

    def counts(d):
        return (
            len(d),
            int((d["Phenotype"]=="Resistant").sum()),
            int((d["Phenotype"]=="Susceptible").sum()),
        )

    n1,r1,s1 = counts(ncbi_keep)
    n2,r2,s2 = counts(miss_keep)
    n3,r3,s3 = counts(bv_keep)
    nf,rf,sf = counts(final_preassembly)
    nr,rr,sr_ = counts(ready)
    nq,rq,sq = counts(queue)

    audit = {
        "version": VERSION,
        "ncbi_v2_strict": {"n":n1,"R":r1,"S":s1},
        "ncbi_missing_assembly_strict": {"n":n2,"R":r2,"S":s2},
        "bvbrc_strict_supplementary": {"n":n3,"R":r3,"S":s3},
        "final_preassembly": {"n":nf,"R":rf,"S":sf},
        "assembly_ready": {"n":nr,"R":rr,"S":sr_},
        "sra_assembly_queue": {"n":nq,"R":rq,"S":sq},
        "gate_preassembly": bool(nf >= 300 and rf >= 100 and sf >= 100),
        "input_sha256": {
            "e3_manifest": sha256(e3_manifest_p),
            "rejected": sha256(rejected_p),
            "original_ncbi_source": sha256(ncbi_source_p),
            "sra_recovery": sha256(sra_recovery_p),
            "e2": sha256(e2_p),
            "dev_crosswalk": sha256(devx_p),
            "dev_ncbi_map": sha256(devbq_p),
            "bvbrc_tranche": sha256(bv_p),
            "bvbrc_ncbi_map": sha256(bvmap_p),
        },
        "output_sha256": {
            "preassembly_pool": sha256(pre_p),
            "assembly_ready": sha256(ready_p),
            "sra_queue": sha256(queue_p),
            "bvbrc_survivors": sha256(bv_surv_p),
        },
    }
    audit_p = outdir/"file14_strict_E3_build_audit.json"
    audit_p.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print()
    print("NCBI-v2 strict survivors           : n=%d R=%d S=%d" % (n1,r1,s1))
    print("NCBI SRA-recovery strict survivors : n=%d R=%d S=%d" % (n2,r2,s2))
    print("BV-BRC strict supplementary        : n=%d R=%d S=%d" % (n3,r3,s3))
    print("-"*116)
    print("STRICT PRE-ASSEMBLY POOL           : n=%d R=%d S=%d" % (nf,rf,sf))
    print("Already assembly-ready             : n=%d R=%d S=%d" % (nr,rr,sr_))
    print("SRA assembly queue                 : n=%d R=%d S=%d" % (nq,rq,sq))
    print("Pre-specified gate n>=300,R>=100,S>=100:",
          "PASS" if audit["gate_preassembly"] else "FAIL")
    print()
    print("Pre-assembly pool                  :", pre_p)
    print("Assembly-ready                     :", ready_p)
    print("SRA assembly queue                 :", queue_p)
    print("Audit JSON                         :", audit_p)
    print("="*116)
    print("STATUS                             : PASS_STRICT_PREASSEMBLY_BUILD")
    print("NEXT                               : assemble/QC SRA queue, then freeze final E3 manifest")


if __name__ == "__main__":
    main()
