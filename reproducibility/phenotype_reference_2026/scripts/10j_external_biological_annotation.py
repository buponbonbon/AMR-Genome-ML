#!/usr/bin/env python3
"""
File10J — External biological annotation via BV-BRC APIs
Version 1.0.3

Fixes vs 1.0.2
--------------
- Correct BV-BRC AMR specialty classification. Real records in this cohort use
  property values such as "Antibiotic Resistance;Drug Target", not only the
  literal token "AMR". v1.0.2 therefore under-called nearby AMR features.
- Treat a specialty record as AMR when property contains either
  "ANTIBIOTIC RESISTANCE" or the standalone token "AMR".
- Add a fallback AMR label from the annotated feature product when the specialty
  gene-name field is empty.
- Reuse valid v1.0.1/v1.0.2 API cache entries when the exact URL is unchanged.
- Preserve deterministic pagination and all earlier URL/completeness fixes.

Purpose
-------
Annotate the non-redundant File10I representatives using official BV-BRC data:

1) PLFam representatives:
   - protein_family_ref: family_product, family_type, provenance dates
   - a small sample of genome_feature members for product/gene/protein-ID consensus

2) Unitig representatives:
   - fetch genome_feature + genome_sequence + specialty-gene records for the
     carrier genomes used by File10I
   - map each exact local unitig occurrence to nearby genes/features
   - join specialty-gene annotations (AMR/virulence/etc.) by feature_id
   - summarize whether the same locus/context is reproduced across carriers

This stage does NOT infer causality. It records external database annotations
and genomic proximity/context.

Operational design
------------------
- per-PLFam and per-genome API checkpoints
- retries with exponential backoff
- atomic final outputs
- live progress + ETA
- resume after interruption
- official BV-BRC API only; no scraping

Usage
-----
SMOKE:
  python scripts/10j_external_biological_annotation.py --smoke

FULL:
  python scripts/10j_external_biological_annotation.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.3"
DESIGN_ID = "file10j_bvbrc_annotation_v1"
BASE = "https://www.bv-brc.org/api"
USER_AGENT = "AMR-Genome-ML-File10J/1.0"
API_TIMEOUT = 60
MAX_RETRIES = 5


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def human_seconds(sec):
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    return start


def atomic_write_text(path: Path, text: str):
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


def atomic_write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    t = Path(tmp)
    try:
        df.to_csv(t, index=False)
        os.replace(t, path)
    except Exception:
        t.unlink(missing_ok=True)
        raise


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Progress:
    def __init__(self, total, label):
        self.total = max(int(total), 1)
        self.done = 0
        self.label = label
        self.t0 = time.time()

    def update(self, detail=""):
        self.done += 1
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        log(
            f"[PROGRESS] {self.label}: {self.done}/{self.total} "
            f"({100*self.done/self.total:5.1f}%) | "
            f"elapsed={human_seconds(elapsed)} | "
            f"ETA={human_seconds(eta) if np.isfinite(eta) else '?'}"
            + (f" | {detail}" if detail else "")
        )


def api_get(url: str) -> Any:
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            # Retry transient server/rate-limit errors only. A 400/404-style
            # client error will not improve by waiting and retrying.
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise RuntimeError(
                    f"BV-BRC API client error HTTP {exc.code}: {url} | {exc.reason}"
                ) from exc
            if attempt == MAX_RETRIES:
                break
            wait = min(2 ** (attempt - 1), 16)
            log(f"  API retry {attempt}/{MAX_RETRIES} after HTTP {exc.code}: {exc}; wait={wait}s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt == MAX_RETRIES:
                break
            wait = min(2 ** (attempt - 1), 16)
            log(f"  API retry {attempt}/{MAX_RETRIES} after {type(exc).__name__}: {exc}; wait={wait}s")
            time.sleep(wait)
    raise RuntimeError(f"BV-BRC API failed after {MAX_RETRIES} attempts: {url} | {last}")


def safe_token(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))


def normalize_api_list(obj):
    # BV-BRC JSON endpoints normally return an array. Be tolerant of wrappers.
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("response", "docs", "results", "items"):
            v = obj.get(key)
            if isinstance(v, list):
                return v
        if "response" in obj and isinstance(obj["response"], dict):
            for key in ("docs", "results"):
                v = obj["response"].get(key)
                if isinstance(v, list):
                    return v
        # Direct GET-by-ID may return one object.
        if any(k in obj for k in ("family_id", "feature_id", "genome_id", "sequence_id", "id")):
            return [obj]
    return []


def query_url(endpoint: str, rql: str) -> str:
    # Keep RQL punctuation readable while URL-encoding data safely.
    safe = "(),|:+-_.&="
    return f"{BASE}/{endpoint}/?{urllib.parse.quote(rql, safe=safe)}"


def fetch_cached_json(cache_path: Path, url: str, provenance: dict):
    if cache_path.is_file():
        try:
            d = read_json(cache_path)
            if (
                d.get("script_version") in {"1.0.1", "1.0.2", SCRIPT_VERSION}
                and d.get("design_id") == DESIGN_ID
                and d.get("url") == url
                and d.get("status") == "PASS"
            ):
                return d["data"], True
        except Exception:
            pass

    data = api_get(url)
    payload = {
        "script_version": SCRIPT_VERSION,
        "design_id": DESIGN_ID,
        "status": "PASS",
        "fetched_utc": utc_now(),
        "url": url,
        "provenance": provenance,
        "data": data,
    }
    atomic_write_text(cache_path, json.dumps(payload, indent=2) + "\n")
    return data, False



def fetch_paginated_cached(
    cache_dir: Path,
    cache_prefix: str,
    endpoint: str,
    base_rql: str,
    sort_field: str,
    page_size: int,
    provenance: dict,
):
    """
    Fetch a complete BV-BRC RQL result using documented limit(COUNT,START)
    pagination. Pages are sorted by the endpoint's unique key so resume/page
    boundaries are deterministic.
    """
    if page_size <= 0:
        raise ValueError("page_size must be > 0")

    all_items = []
    page_meta = []
    start = 0
    page_no = 0

    while True:
        rql = (
            f"{base_rql}"
            f"&sort(+{sort_field})"
            f"&limit({int(page_size)},{int(start)})"
        )
        url = query_url(endpoint, rql)
        cp = cache_dir / f"{cache_prefix}__page_{page_no:04d}_start_{start}.json"
        raw, skipped = fetch_cached_json(
            cp,
            url,
            {
                **provenance,
                "pagination": True,
                "page_no": page_no,
                "start": start,
                "page_size": page_size,
                "sort_field": sort_field,
            },
        )
        items = normalize_api_list(raw)
        all_items.extend(items)
        page_meta.append({
            "page_no": page_no,
            "start": start,
            "n": len(items),
            "cache_reused": bool(skipped),
        })

        if len(items) < page_size:
            break

        start += page_size
        page_no += 1
        if page_no > 1000:
            raise RuntimeError(
                f"Pagination safety stop for {endpoint}/{cache_prefix}: >1000 pages"
            )

    # De-duplicate defensively by unique key while preserving sorted page order.
    seen = set()
    deduped = []
    for item in all_items:
        key = str(item.get(sort_field, ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(item)

    return deduped, page_meta


def flatten_value(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return ";".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def first_mode(values: Iterable[str]) -> str:
    vals = [str(x).strip() for x in values if str(x).strip() and str(x).lower() != "nan"]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def unique_join(values: Iterable[str], limit: int = 20) -> str:
    seen = []
    for x in values:
        s = str(x).strip()
        if not s or s.lower() == "nan" or s in seen:
            continue
        seen.append(s)
        if len(seen) >= limit:
            break
    return ";".join(seen)


def contig_aliases(text: str):
    s = str(text).strip()
    first = s.split()[0] if s else ""
    aliases = {s, first}
    # BV-BRC/FASTA headers sometimes include pipe-delimited names.
    if "|" in first:
        aliases.update(x for x in first.split("|") if x)
    # Some headers include trailing punctuation.
    aliases.update(x.rstrip(",;") for x in list(aliases))
    return {x for x in aliases if x}


def feature_matches_contig(feature: dict, contig_header: str, seq_records: List[dict]) -> bool:
    aliases = contig_aliases(contig_header)
    fvals = {
        str(feature.get("sequence_id", "")).strip(),
        str(feature.get("accession", "")).strip(),
    }
    fvals = {x for x in fvals if x}

    if aliases & fvals:
        return True

    # Use genome_sequence crosswalk if local FASTA header is accession while
    # feature uses sequence_id (or vice versa).
    matching_seq_ids = set()
    for rec in seq_records:
        recvals = {
            str(rec.get("sequence_id", "")).strip(),
            str(rec.get("accession", "")).strip(),
        }
        recvals = {x for x in recvals if x}
        if aliases & recvals:
            matching_seq_ids |= recvals
    return bool(fvals & matching_seq_ids)


def interval_distance(a_start, a_end, b_start, b_end):
    # Inclusive 1-based intervals.
    if a_end < b_start:
        return b_start - a_end - 1
    if b_end < a_start:
        return a_start - b_end - 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--member-sample", type=int, default=5)
    ap.add_argument("--nearby-bp", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else find_project_root(Path.cwd())

    prep = root / "data/explainability_biology/annotation_prep"
    reps_path = prep / "file10i_nonredundant_representatives.csv"
    occ_path = prep / "file10i_unitig_context_occurrences.csv"

    if not reps_path.is_file() or not occ_path.is_file():
        raise SystemExit(
            "Missing File10I FULL outputs. Required:\n"
            f"  {reps_path}\n  {occ_path}"
        )

    if args.smoke:
        out_dir = root / "data/explainability_biology/external_annotation/smoke"
        ckpt = root / "checkpoints/explainability_biology_external_annotation_smoke"
    else:
        out_dir = root / "data/explainability_biology/external_annotation"
        ckpt = root / "checkpoints/explainability_biology_external_annotation"

    cache = ckpt / "api_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    start = time.time()
    stage = "startup"

    try:
        log("=" * 96)
        log("FILE10J — BV-BRC EXTERNAL BIOLOGICAL ANNOTATION")
        log("=" * 96)
        log(f"Version       : {SCRIPT_VERSION}")
        log(f"Mode          : {'SMOKE' if args.smoke else 'FULL'}")
        log(f"Project root  : {root}")
        log(f"Member sample : {args.member_sample} features / PLFam")
        log(f"Nearby window : ±{args.nearby_bp} bp")
        log("Checkpointing : PER PLFam + PER CARRIER GENOME API QUERY")
        log("Live progress : ENABLED")

        reps = pd.read_csv(reps_path)
        occ = pd.read_csv(occ_path, dtype={"Genome ID": str, "MLST": str, "Genomic Cluster": str})
        occ = occ[occ["status"] == "EXACT_HIT"].copy()

        pl = reps[reps["representation"] == "pangenome"].copy()
        un = reps[reps["representation"] == "unitig"].copy()
        if args.smoke:
            pl = pl.head(3)
            un_ids = set(un.head(1)["feature_index"].astype(int))
            occ = occ[occ["feature_index"].astype(int).isin(un_ids)].head(1).copy()

        # ------------------------------------------------------------------
        # Stage 1: PLFam reference + member annotations
        # ------------------------------------------------------------------
        stage = "plfam_annotation"
        log(f"Stage 1/4 — PLFam annotation ({len(pl)} representatives)")
        prog = Progress(len(pl), "10J PLFam")

        pl_rows = []
        for r in pl.itertuples(index=False):
            family_id = str(r.feature_id)
            key = safe_token(family_id)

            url_ref = f"{BASE}/protein_family_ref/{urllib.parse.quote(family_id, safe='_-.')}"
            ref_raw, ref_skip = fetch_cached_json(
                cache / f"plfam_{key}__ref.json",
                url_ref,
                {"kind": "protein_family_ref", "family_id": family_id},
            )
            ref_list = normalize_api_list(ref_raw)
            ref = ref_list[0] if ref_list else {}

            rql = f"eq(plfam_id,{family_id})&limit({int(args.member_sample)})"
            url_members = query_url("genome_feature", rql)
            mem_raw, mem_skip = fetch_cached_json(
                cache / f"plfam_{key}__members.json",
                url_members,
                {"kind": "genome_feature_members", "family_id": family_id},
            )
            members = normalize_api_list(mem_raw)

            products = [m.get("product", "") for m in members]
            genes = [m.get("gene", "") for m in members]
            protein_ids = [m.get("protein_id", "") for m in members]
            pgfams = [m.get("pgfam_id", "") for m in members]
            loci = [m.get("refseq_locus_tag", "") for m in members]

            pl_rows.append({
                "feature_index": int(r.feature_index),
                "PLFam_ID": family_id,
                "redundancy_cluster": int(r.redundancy_cluster),
                "representative_role": str(r.representative_role),
                "annotation_priority_rank": int(r.annotation_priority_rank),
                "selected_tasks": int(r.selected_tasks),
                "mean_abs_shap": float(r.mean_abs_shap) if pd.notna(r.mean_abs_shap) else np.nan,
                "family_product": flatten_value(ref.get("family_product")),
                "family_type": flatten_value(ref.get("family_type")),
                "family_date_modified": flatten_value(ref.get("date_modified")),
                "member_count_sampled": len(members),
                "member_product_mode": first_mode(products),
                "member_products": unique_join(products),
                "member_genes": unique_join(genes),
                "member_protein_ids": unique_join(protein_ids),
                "member_pgfams": unique_join(pgfams),
                "member_refseq_locus_tags": unique_join(loci),
                "ref_cache_reused": ref_skip,
                "member_cache_reused": mem_skip,
            })
            prog.update(f"{family_id} | product={flatten_value(ref.get('family_product'))[:55]}")

        plfam_df = pd.DataFrame(pl_rows)
        out_plfam = out_dir / "file10j_plfam_annotations.csv"
        atomic_write_csv(out_plfam, plfam_df)

        # ------------------------------------------------------------------
        # Stage 2: fetch carrier genome sequences/features/specialty genes
        # ------------------------------------------------------------------
        stage = "carrier_genomes"
        genomes = sorted(set(occ["Genome ID"].dropna().astype(str)))
        log(f"Stage 2/4 — carrier genome annotations ({len(genomes)} genomes)")
        prog = Progress(max(len(genomes), 1), "10J carrier genomes")

        genome_data = {}
        for gid in genomes:
            gkey = safe_token(gid)

            gd = {}
            skips = []

            # Sequence records are few; the existing exact URL is retained so
            # valid v1.0.1 cache can be reused.
            seq_url = query_url("genome_sequence", f"eq(genome_id,{gid})&limit(10000)")
            seq_raw, seq_skipped = fetch_cached_json(
                cache / f"genome_{gkey}__sequence.json",
                seq_url,
                {"kind": "sequence", "genome_id": gid},
            )
            gd["sequence"] = normalize_api_list(seq_raw)
            skips.append(seq_skipped)

            # Genome features can exceed 10,000 because multiple annotation
            # sources may coexist. Fetch every page deterministically.
            gd["feature"], feature_pages = fetch_paginated_cached(
                cache_dir=cache,
                cache_prefix=f"genome_{gkey}__feature_v102",
                endpoint="genome_feature",
                base_rql=f"eq(genome_id,{gid})",
                sort_field="feature_id",
                page_size=5000,
                provenance={"kind": "feature", "genome_id": gid},
            )

            # Specialty genes are also paginated so completeness is explicit.
            gd["specialty"], specialty_pages = fetch_paginated_cached(
                cache_dir=cache,
                cache_prefix=f"genome_{gkey}__specialty_v102",
                endpoint="sp_gene",
                base_rql=f"eq(genome_id,{gid})",
                sort_field="id",
                page_size=5000,
                provenance={"kind": "specialty", "genome_id": gid},
            )

            genome_data[gid] = gd
            f_cached = all(x["cache_reused"] for x in feature_pages)
            s_cached = all(x["cache_reused"] for x in specialty_pages)
            prog.update(
                f"{gid} | seq={len(gd['sequence'])} "
                f"features={len(gd['feature'])} ({len(feature_pages)} pages) "
                f"specialty={len(gd['specialty'])} ({len(specialty_pages)} pages) "
                f"| cached={seq_skipped and f_cached and s_cached}"
            )

        # ------------------------------------------------------------------
        # Stage 3: map unitig occurrences into local annotated loci
        # ------------------------------------------------------------------
        stage = "unitig_locus_mapping"
        log(f"Stage 3/4 — unitig locus mapping ({len(occ)} exact occurrences)")
        prog = Progress(max(len(occ), 1), "10J unitig occurrences")

        locus_rows = []
        occ_summary_rows = []

        for occ_no, o in enumerate(occ.itertuples(index=False), 1):
            gid = str(getattr(o, "_asdict")().get("Genome ID", getattr(o, "Genome_ID", "")))
            # pandas itertuples sanitizes "Genome ID" to positional field; use Series below if needed.
            row = occ.iloc[occ_no - 1]
            gid = str(row["Genome ID"])
            fi = int(row["feature_index"])
            contig = str(row["contig"])

            unitig_start1 = int(row["start_0based"]) + 1
            unitig_end1 = int(row["end_0based_exclusive"])
            context_start1 = int(row["context_start_0based"]) + 1
            context_end1 = int(row["context_end_0based_exclusive"])

            gd = genome_data[gid]
            seqs = gd["sequence"]
            features = gd["feature"]
            specialty = gd["specialty"]

            sp_by_feature = defaultdict(list)
            for s in specialty:
                fid = str(s.get("feature_id", ""))
                if fid:
                    sp_by_feature[fid].append(s)

            nearby = []
            contig_matched_features = 0
            for f in features:
                if not feature_matches_contig(f, contig, seqs):
                    continue
                contig_matched_features += 1

                try:
                    fs = int(f.get("start"))
                    fe = int(f.get("end"))
                except Exception:
                    continue
                if fs > fe:
                    fs, fe = fe, fs

                dist = interval_distance(unitig_start1, unitig_end1, fs, fe)
                # Keep anything overlapping extracted context OR within requested distance.
                overlaps_context = not (fe < context_start1 or fs > context_end1)
                if not overlaps_context and dist > args.nearby_bp:
                    continue

                fid = str(f.get("feature_id", ""))
                specials = sp_by_feature.get(fid, [])
                if specials:
                    special_gene = unique_join([s.get("gene", "") for s in specials])
                    special_property = unique_join([s.get("property", "") for s in specials])
                    special_source = unique_join([s.get("source", "") for s in specials])
                    special_class = unique_join(
                        [flatten_value(s.get("classification", "")) for s in specials]
                    )
                    special_antibiotics = unique_join(
                        [flatten_value(s.get("antibiotics", "")) for s in specials]
                    )
                    special_function = unique_join(
                        [s.get("function", s.get("product", "")) for s in specials]
                    )
                    special_pmids = unique_join(
                        [flatten_value(s.get("pmid", "")) for s in specials]
                    )
                else:
                    special_gene = special_property = special_source = ""
                    special_class = special_antibiotics = special_function = special_pmids = ""

                rec = {
                    "unitig_feature_index": fi,
                    "Genome ID": gid,
                    "MLST": row.get("MLST", ""),
                    "Genomic Cluster": row.get("Genomic Cluster", ""),
                    "phenotype_y": row.get("y", np.nan),
                    "local_contig_header": contig,
                    "unitig_strand": row.get("strand", ""),
                    "unitig_start_1based": unitig_start1,
                    "unitig_end_1based": unitig_end1,
                    "context_start_1based": context_start1,
                    "context_end_1based": context_end1,
                    "feature_id": fid,
                    "feature_type": flatten_value(f.get("feature_type")),
                    "annotation": flatten_value(f.get("annotation")),
                    "feature_sequence_id": flatten_value(f.get("sequence_id")),
                    "feature_accession": flatten_value(f.get("accession")),
                    "feature_start_1based": fs,
                    "feature_end_1based": fe,
                    "feature_strand": flatten_value(f.get("strand")),
                    "distance_to_unitig_bp": dist,
                    "overlaps_unitig": dist == 0,
                    "gene": flatten_value(f.get("gene")),
                    "product": flatten_value(f.get("product")),
                    "plfam_id": flatten_value(f.get("plfam_id")),
                    "pgfam_id": flatten_value(f.get("pgfam_id")),
                    "refseq_locus_tag": flatten_value(f.get("refseq_locus_tag")),
                    "protein_id": flatten_value(f.get("protein_id")),
                    "uniprotkb_accession": flatten_value(f.get("uniprotkb_accession")),
                    "specialty_gene": special_gene,
                    "specialty_property": special_property,
                    "specialty_source": special_source,
                    "specialty_classification": special_class,
                    "specialty_antibiotics": special_antibiotics,
                    "specialty_function": special_function,
                    "specialty_pmids": special_pmids,
                }
                nearby.append(rec)
                locus_rows.append(rec)

            def is_amr_specialty(rec):
                prop = str(rec.get("specialty_property", "")).upper()
                tokens = {x.strip() for x in prop.replace("|", ";").split(";") if x.strip()}
                return (
                    "ANTIBIOTIC RESISTANCE" in prop
                    or "AMR" in tokens
                )

            amr_nearby = [r for r in nearby if is_amr_specialty(r)]
            nearest = min(nearby, key=lambda r: r["distance_to_unitig_bp"]) if nearby else None
            nearest_amr = min(amr_nearby, key=lambda r: r["distance_to_unitig_bp"]) if amr_nearby else None

            occ_summary_rows.append({
                "unitig_feature_index": fi,
                "Genome ID": gid,
                "MLST": row.get("MLST", ""),
                "Genomic Cluster": row.get("Genomic Cluster", ""),
                "contig_header": contig,
                "contig_matched_feature_count": contig_matched_features,
                "nearby_feature_count": len(nearby),
                "overlapping_feature_count": sum(r["overlaps_unitig"] for r in nearby),
                "nearest_feature_id": nearest["feature_id"] if nearest else "",
                "nearest_gene": nearest["gene"] if nearest else "",
                "nearest_product": nearest["product"] if nearest else "",
                "nearest_distance_bp": nearest["distance_to_unitig_bp"] if nearest else np.nan,
                "nearest_amr_feature_id": nearest_amr["feature_id"] if nearest_amr else "",
                "nearest_amr_gene": nearest_amr["specialty_gene"] if nearest_amr else "",
                "nearest_amr_product": nearest_amr["product"] if nearest_amr else "",
                "nearest_amr_label": (
                    (nearest_amr["specialty_gene"] or nearest_amr["gene"] or nearest_amr["product"])
                    if nearest_amr else ""
                ),
                "nearest_amr_function": (
                    (nearest_amr["specialty_function"] or nearest_amr["product"])
                    if nearest_amr else ""
                ),
                "nearest_amr_distance_bp": nearest_amr["distance_to_unitig_bp"] if nearest_amr else np.nan,
            })
            display_amr = (
                (nearest_amr["specialty_gene"] or nearest_amr["gene"] or nearest_amr["product"])
                if nearest_amr else "none"
            )
            prog.update(
                f"unitig_{fi} / {gid} | nearby_features={len(nearby)} "
                f"| nearest_AMR={display_amr}"
            )

        locus_df = pd.DataFrame(locus_rows)
        occ_summary = pd.DataFrame(occ_summary_rows)
        out_locus = out_dir / "file10j_unitig_locus_annotations.csv"
        out_occ = out_dir / "file10j_unitig_occurrence_summary.csv"
        atomic_write_csv(out_locus, locus_df)
        atomic_write_csv(out_occ, occ_summary)

        # ------------------------------------------------------------------
        # Stage 4: representative-level unitig summary + integrated shortlist
        # ------------------------------------------------------------------
        stage = "integrated_summary"
        log("Stage 4/4 — integrated annotation summary")

        unitig_summary_rows = []
        for fi, g in occ_summary.groupby("unitig_feature_index", sort=True):
            amr_genes = unique_join(g["nearest_amr_gene"].fillna(""))
            amr_labels = unique_join(g["nearest_amr_label"].fillna(""))
            nearest_genes = unique_join(g["nearest_gene"].fillna(""))
            nearest_products = unique_join(g["nearest_product"].fillna(""))
            exact_locus_mapped = int((g["nearby_feature_count"] > 0).sum())

            # Mode of nearest AMR annotation across carriers. Use a label that
            # falls back to feature product when sp_gene lacks a gene-name field.
            amr_mode = first_mode(g["nearest_amr_gene"].fillna(""))
            amr_label_mode = first_mode(g["nearest_amr_label"].fillna(""))
            gene_mode = first_mode(g["nearest_gene"].fillna(""))
            prod_mode = first_mode(g["nearest_product"].fillna(""))

            unitig_summary_rows.append({
                "feature_index": int(fi),
                "carrier_occurrences": len(g),
                "carrier_occurrences_with_feature_mapping": exact_locus_mapped,
                "nearest_gene_mode": gene_mode,
                "nearest_product_mode": prod_mode,
                "nearest_genes_all": nearest_genes,
                "nearest_products_all": nearest_products,
                "nearest_amr_gene_mode": amr_mode,
                "nearest_amr_genes_all": amr_genes,
                "nearest_amr_label_mode": amr_label_mode,
                "nearest_amr_labels_all": amr_labels,
                "amr_context_carriers": int(g["nearest_amr_label"].fillna("").astype(str).str.len().gt(0).sum()),
                "all_carriers_same_nearest_gene": bool(
                    len(set(x for x in g["nearest_gene"].fillna("").astype(str) if x)) <= 1
                ),
                "all_carriers_same_nearest_amr_gene": bool(
                    len(set(x for x in g["nearest_amr_label"].fillna("").astype(str) if x)) <= 1
                ),
            })

        unitig_summary = pd.DataFrame(unitig_summary_rows)
        out_unitig_summary = out_dir / "file10j_unitig_annotation_summary.csv"
        atomic_write_csv(out_unitig_summary, unitig_summary)

        # Merge annotations back to the representative evidence table.
        integrated = reps.copy()
        integrated = integrated.merge(
            plfam_df[
                [
                    "feature_index", "family_product", "family_type",
                    "member_product_mode", "member_products", "member_genes",
                    "member_protein_ids", "member_pgfams",
                ]
            ],
            on="feature_index",
            how="left",
        )
        if len(unitig_summary):
            integrated = integrated.merge(
                unitig_summary,
                on="feature_index",
                how="left",
            )

        # Conservative annotation-status field only.
        integrated["external_annotation_status"] = np.where(
            integrated["representation"].eq("pangenome"),
            np.where(
                integrated["family_product"].fillna("").astype(str).str.len() > 0,
                "PLFAM_PRODUCT_AVAILABLE",
                "PLFAM_PRODUCT_MISSING",
            ),
            np.where(
                integrated.get("carrier_occurrences_with_feature_mapping", pd.Series(index=integrated.index, dtype=float))
                .fillna(0).gt(0),
                "UNITIG_LOCUS_MAPPED",
                "UNITIG_LOCUS_NOT_MAPPED",
            ),
        )

        out_integrated = out_dir / "file10j_integrated_annotation_shortlist.csv"
        atomic_write_csv(out_integrated, integrated)

        manifest = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "PASS",
            "mode": "SMOKE" if args.smoke else "FULL",
            "completed_utc": utc_now(),
            "elapsed_seconds": time.time() - start,
            "plfam_representatives_annotated": len(plfam_df),
            "carrier_genomes_queried": len(genomes),
            "unitig_exact_occurrences_mapped": len(occ_summary),
            "unitig_locus_annotation_rows": len(locus_df),
            "unitig_representatives_summarized": len(unitig_summary),
            "api_base": BASE,
            "carrier_feature_queries_paginated": True,
            "carrier_specialty_queries_paginated": True,
            "pagination_page_size": 5000,
            "amr_specialty_classifier": "property contains ANTIBIOTIC RESISTANCE or token AMR",
            "causal_claims": False,
            "interpretation_guardrail": (
                "BV-BRC family products, genome-feature annotations, specialty-gene records, "
                "and local proximity are external annotations/context. They do not establish "
                "causality or a novel resistance mechanism."
            ),
            "next_stage": (
                "Review the non-redundant annotated shortlist, select reportable biological "
                "themes/candidates, and perform targeted literature validation."
            ),
            "outputs": {
                "plfam_annotations": str(out_plfam),
                "unitig_locus_annotations": str(out_locus),
                "unitig_occurrence_summary": str(out_occ),
                "unitig_annotation_summary": str(out_unitig_summary),
                "integrated_shortlist": str(out_integrated),
            },
        }
        final = ckpt / "file10j_final_summary.json"
        atomic_write_text(final, json.dumps(manifest, indent=2) + "\n")

        log("=" * 96)
        log("FILE10J STATUS : PASS")
        log(f"PLFam annotated        : {len(plfam_df)}")
        log(f"Carrier genomes queried: {len(genomes)}")
        log(f"Unitig occurrences     : {len(occ_summary)}")
        log(f"Locus annotation rows  : {len(locus_df)}")
        log(f"Unitig reps summarized : {len(unitig_summary)}")
        log(f"Elapsed                : {human_seconds(time.time()-start)}")
        log(f"Integrated shortlist   : {out_integrated}")
        log(f"Final summary          : {final}")
        log("=" * 96)

    except KeyboardInterrupt:
        failure = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "INTERRUPTED",
            "stage": stage,
            "time_utc": utc_now(),
            "elapsed_seconds": time.time() - start,
            "message": "Per-family/per-genome API cache is preserved; re-run to resume.",
        }
        atomic_write_text(ckpt / "file10j_last_failure.json", json.dumps(failure, indent=2) + "\n")
        log(f"FILE10J INTERRUPTED at {stage}. API checkpoints preserved.")
        raise SystemExit(130)

    except Exception as exc:
        failure = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "FAILED",
            "stage": stage,
            "time_utc": utc_now(),
            "elapsed_seconds": time.time() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_text(ckpt / "file10j_last_failure.json", json.dumps(failure, indent=2) + "\n")
        log(f"FILE10J FAILED at {stage}: {type(exc).__name__}: {exc}")
        log(f"Crash record: {ckpt / 'file10j_last_failure.json'}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
