#!/usr/bin/env python3
"""
File10I — Biological candidate collapse and local sequence-context preparation
Version 1.0.0

Purpose
-------
Collapse the 200 File10 broader-feature candidates into non-redundant biological
representatives, export unitig sequences / PLFam IDs, and extract exact local
sequence context for representative unitigs from carrier genome assemblies.

This stage does NOT claim gene identity or mechanism. It prepares a defensible,
non-redundant shortlist for subsequent AMRFinder/BLAST/BV-BRC/literature
annotation.

Operational guarantees
----------------------
- atomic outputs
- per-unitig context checkpoint
- live progress + ETA
- resume without repeating finished unitigs
- no full 4,227-genome scan for each unitig: carriers are chosen from the
  already-decoded File10 unitig cache, then only selected carrier FASTAs are read

Usage
-----
SMOKE:
  python scripts/10i_biological_annotation_prep.py --smoke

FULL:
  python scripts/10i_biological_annotation_prep.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"
DESIGN_ID = "file10i_annotation_prep_v1"
EXPECTED_N = 4227
RANDOM_SEED = 20260920


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
        self.label = label
        self.done = 0
        self.t0 = time.time()

    def update(self, detail=""):
        self.done += 1
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0
        remain = self.total - self.done
        eta = remain / rate if rate > 0 else float("nan")
        pct = 100 * self.done / self.total
        log(
            f"[PROGRESS] {self.label}: {self.done}/{self.total} ({pct:5.1f}%) "
            f"| elapsed={human_seconds(elapsed)} "
            f"| ETA={human_seconds(eta) if np.isfinite(eta) else '?'}"
            + (f" | {detail}" if detail else "")
        )


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]


def fasta_records_gz(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = None
        chunks = []
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks).upper()


def find_exact_context(genome_path: Path, query: str, flank: int):
    q = query.upper()
    rc = reverse_complement(q)
    hits = []
    for header, seq in fasta_records_gz(genome_path):
        start = 0
        while True:
            i = seq.find(q, start)
            if i < 0:
                break
            lo = max(0, i - flank)
            hi = min(len(seq), i + len(q) + flank)
            hits.append({
                "contig": header,
                "strand": "+",
                "start_0based": i,
                "end_0based_exclusive": i + len(q),
                "context_start_0based": lo,
                "context_end_0based_exclusive": hi,
                "context_sequence": seq[lo:hi],
            })
            start = i + 1

        # Avoid double counting palindromic queries.
        if rc != q:
            start = 0
            while True:
                i = seq.find(rc, start)
                if i < 0:
                    break
                lo = max(0, i - flank)
                hi = min(len(seq), i + len(q) + flank)
                hits.append({
                    "contig": header,
                    "strand": "-",
                    "start_0based": i,
                    "end_0based_exclusive": i + len(q),
                    "context_start_0based": lo,
                    "context_end_0based_exclusive": hi,
                    "context_sequence": seq[lo:hi],
                })
                start = i + 1
    return hits


def choose_representatives(cand: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    c = cand.copy()

    numeric_defaults = {
        "mean_abs_shap": 0.0,
        "selected_tasks": 0.0,
        "median_rank": 1e9,
        "cluster_size": 1.0,
        "max_raw_abs_phi": 0.0,
        "genomic_cluster__cmh_p_fdr": 1.0,
        "mlst__cmh_p_fdr": 1.0,
    }
    for col, default in numeric_defaults.items():
        if col not in c.columns:
            c[col] = default
        c[col] = pd.to_numeric(c[col], errors="coerce").fillna(default)

    # Deterministic evidence priority. This is NOT a predictive-model score.
    c["adj_sig_count"] = (
        (c["genomic_cluster__cmh_p_fdr"] < 0.05).astype(int)
        + (c["mlst__cmh_p_fdr"] < 0.05).astype(int)
    )
    c["priority_tuple_text"] = c.apply(
        lambda r: (
            f"adj_sig={int(r['adj_sig_count'])};"
            f"selected_tasks={int(r['selected_tasks'])};"
            f"mean_abs_shap={r['mean_abs_shap']:.8g};"
            f"median_rank={r['median_rank']:.3g}"
        ),
        axis=1,
    )

    # Primary representative per redundancy cluster.
    c = c.sort_values(
        [
            "redundancy_cluster",
            "adj_sig_count",
            "selected_tasks",
            "mean_abs_shap",
            "median_rank",
            "feature_index",
        ],
        ascending=[True, False, False, False, True, True],
    )
    primary = c.groupby("redundancy_cluster", as_index=False, group_keys=False).head(1).copy()
    primary["representative_role"] = "primary_cluster_representative"

    # Preserve one representation-specific alternate when a cluster mixes PLFam + unitig.
    alternates = []
    for cluster, g in c.groupby("redundancy_cluster"):
        reps = set(g["representation"])
        if len(reps) < 2:
            continue
        p = primary[primary["redundancy_cluster"] == cluster].iloc[0]
        other_rep = "unitig" if p["representation"] == "pangenome" else "pangenome"
        h = g[g["representation"] == other_rep].head(1).copy()
        if len(h):
            h["representative_role"] = "cross_representation_alternate"
            alternates.append(h)

    out = pd.concat([primary, *alternates], ignore_index=True) if alternates else primary

    out = out.sort_values(
        [
            "adj_sig_count",
            "selected_tasks",
            "mean_abs_shap",
            "median_rank",
            "feature_index",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    if smoke:
        # Exercise both feature representations if possible.
        pieces = []
        for rep in ("unitig", "pangenome"):
            h = out[out["representation"] == rep].head(3)
            pieces.append(h)
        out = pd.concat(pieces, ignore_index=True)

    out["annotation_priority_rank"] = np.arange(1, len(out) + 1)
    return out


def load_unitig_cache(cache_path: Path):
    with np.load(cache_path, allow_pickle=False) as z:
        idx = z["feature_index"].astype(int)
        X = z["X"].astype(np.uint8)
    return idx, X


def carrier_genomes(
    feature_index: int,
    cache_idx: np.ndarray,
    cache_X: np.ndarray,
    folds: pd.DataFrame,
    n_carriers: int,
):
    pos = {int(fi): j for j, fi in enumerate(cache_idx)}
    if int(feature_index) not in pos:
        raise RuntimeError(f"Unitig {feature_index} not found in File10 selected cache.")
    present = np.flatnonzero(cache_X[:, pos[int(feature_index)]] == 1)
    if len(present) == 0:
        return []

    g = folds.iloc[present][["Genome ID", "MLST", "Genomic Cluster", "y"]].copy()
    g["MLST"] = g["MLST"].fillna("UNRESOLVED").astype(str)
    g["Genomic Cluster"] = g["Genomic Cluster"].fillna("UNRESOLVED").astype(str)

    # Prefer diverse MLSTs and include both phenotypes when available.
    chosen = []
    used_mlst = set()
    for _, r in g.sort_values(["MLST", "Genome ID"]).iterrows():
        if r["MLST"] not in used_mlst:
            chosen.append(r)
            used_mlst.add(r["MLST"])
        if len(chosen) >= n_carriers:
            break

    if len(chosen) < n_carriers:
        chosen_ids = {x["Genome ID"] for x in chosen}
        for _, r in g.sort_values(["Genome ID"]).iterrows():
            if r["Genome ID"] not in chosen_ids:
                chosen.append(r)
                chosen_ids.add(r["Genome ID"])
            if len(chosen) >= n_carriers:
                break

    return [dict(x) for x in chosen]


def resolve_genome_path(genomes_dir: Path, genome_id: str):
    candidates = [
        genomes_dir / f"{genome_id}.fna.gz",
        genomes_dir / f"{genome_id}.fna",
        genomes_dir / f"{genome_id}.fa.gz",
        genomes_dir / f"{genome_id}.fa",
        genomes_dir / f"{genome_id}.fasta.gz",
        genomes_dir / f"{genome_id}.fasta",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def task_valid(marker: Path, task_id: str, outputs):
    if not marker.is_file():
        return False
    try:
        d = read_json(marker)
    except Exception:
        return False
    return (
        d.get("script_version") == SCRIPT_VERSION
        and d.get("design_id") == DESIGN_ID
        and d.get("task_id") == task_id
        and d.get("status") == "PASS"
        and all(Path(p).is_file() for p in outputs)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--flank", type=int, default=5000)
    ap.add_argument("--carriers-per-unitig", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else find_project_root(Path.cwd())

    if args.smoke:
        out_dir = root / "data/explainability_biology/annotation_prep/smoke"
        ckpt = root / "checkpoints/explainability_biology_annotation_prep_smoke"
        n_carriers = min(args.carriers_per_unitig, 1)
        flank = min(args.flank, 1000)
    else:
        out_dir = root / "data/explainability_biology/annotation_prep"
        ckpt = root / "checkpoints/explainability_biology_annotation_prep"
        n_carriers = args.carriers_per_unitig
        flank = args.flank

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    task_dir = ckpt / "unitig_context_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    stage = "startup"

    try:
        log("=" * 92)
        log("FILE10I — BIOLOGICAL ANNOTATION PREPARATION")
        log("=" * 92)
        log(f"Version        : {SCRIPT_VERSION}")
        log(f"Mode           : {'SMOKE' if args.smoke else 'FULL'}")
        log(f"Project root   : {root}")
        log(f"Context flank  : {flank} bp")
        log(f"Carriers/unitig: {n_carriers}")
        log("Checkpointing  : PER UNITIG REPRESENTATIVE")
        log("Live progress  : ENABLED")

        cand_path = root / "data/explainability_biology/file10_biological_candidate_summary.csv"
        folds_path = root / "data/splits/file07_outer_folds.csv"
        unitig_cache_path = root / "checkpoints/explainability_biology/cache/file10_unitig_selected_cache.npz"

        for p in (cand_path, folds_path, unitig_cache_path):
            if not p.is_file():
                raise RuntimeError(f"Missing required File10 input: {p}")

        stage = "collapse"
        cand = pd.read_csv(cand_path)
        required = {
            "representation", "feature_index", "feature_id",
            "redundancy_cluster", "cluster_size", "selected_tasks",
            "median_rank", "mean_abs_shap",
        }
        missing = required - set(cand.columns)
        if missing:
            raise RuntimeError(f"Candidate table missing columns: {sorted(missing)}")

        reps = choose_representatives(cand, args.smoke)
        reps_path = out_dir / "file10i_nonredundant_representatives.csv"
        atomic_write_csv(reps_path, reps)

        n_clusters = int(cand["redundancy_cluster"].nunique())
        log(
            f"Collapsed {len(cand)} candidates across {n_clusters} redundancy clusters "
            f"to {len(reps)} annotation representatives"
        )

        # Export unitig FASTA and PLFam list.
        unitig_reps = reps[reps["representation"] == "unitig"].copy()
        pang_reps = reps[reps["representation"] == "pangenome"].copy()

        unitig_fasta = out_dir / "file10i_unitig_representatives.fasta"
        fasta_lines = []
        for r in unitig_reps.itertuples(index=False):
            seq = str(getattr(r, "sequence", "")).upper()
            if not seq or seq == "NAN":
                raise RuntimeError(f"Missing sequence for representative unitig {r.feature_index}")
            header = (
                f"unitig_{int(r.feature_index)}"
                f"|cluster={int(r.redundancy_cluster)}"
                f"|rank={int(r.annotation_priority_rank)}"
                f"|selected_tasks={int(r.selected_tasks)}"
            )
            fasta_lines.extend([f">{header}", seq])
        atomic_write_text(unitig_fasta, "\n".join(fasta_lines) + ("\n" if fasta_lines else ""))

        plfam_path = out_dir / "file10i_plfam_representatives.csv"
        atomic_write_csv(plfam_path, pang_reps)

        # Context extraction.
        stage = "unitig_context"
        folds = pd.read_csv(
            folds_path,
            dtype={"Genome ID": str, "MLST": str, "Genomic Cluster": str},
        )
        if len(folds) != EXPECTED_N:
            raise RuntimeError(f"Unexpected cohort size in File07: {len(folds)}")

        cache_idx, cache_X = load_unitig_cache(unitig_cache_path)
        genomes_dir = root / "data/genomes"

        prog = Progress(len(unitig_reps), "10I unitig contexts")
        context_csvs = []
        context_fastas = []

        for r in unitig_reps.itertuples(index=False):
            fi = int(r.feature_index)
            task_id = f"unitig_{fi}"
            marker = task_dir / f"{task_id}.json"
            out_csv = task_dir / f"{task_id}__occurrences.csv"
            out_fa = task_dir / f"{task_id}__contexts.fasta"
            context_csvs.append(out_csv)
            context_fastas.append(out_fa)

            if task_valid(marker, task_id, [out_csv, out_fa]):
                prog.update(f"{task_id} SKIP")
                continue

            tt = time.time()
            seq = str(r.sequence).upper()
            carriers = carrier_genomes(fi, cache_idx, cache_X, folds, n_carriers)
            rows = []
            fa = []

            for cnum, carrier in enumerate(carriers, 1):
                gid = str(carrier["Genome ID"])
                gp = resolve_genome_path(genomes_dir, gid)
                if gp is None:
                    rows.append({
                        "feature_index": fi,
                        "carrier_no": cnum,
                        "Genome ID": gid,
                        "MLST": carrier["MLST"],
                        "Genomic Cluster": carrier["Genomic Cluster"],
                        "y": int(carrier["y"]),
                        "genome_path": "",
                        "status": "GENOME_FILE_MISSING",
                    })
                    continue

                hits = find_exact_context(gp, seq, flank)
                if not hits:
                    rows.append({
                        "feature_index": fi,
                        "carrier_no": cnum,
                        "Genome ID": gid,
                        "MLST": carrier["MLST"],
                        "Genomic Cluster": carrier["Genomic Cluster"],
                        "y": int(carrier["y"]),
                        "genome_path": str(gp),
                        "status": "NO_EXACT_HIT",
                    })
                    continue

                for hnum, h in enumerate(hits, 1):
                    row = {
                        "feature_index": fi,
                        "carrier_no": cnum,
                        "hit_no": hnum,
                        "Genome ID": gid,
                        "MLST": carrier["MLST"],
                        "Genomic Cluster": carrier["Genomic Cluster"],
                        "y": int(carrier["y"]),
                        "genome_path": str(gp),
                        "status": "EXACT_HIT",
                        **{k: v for k, v in h.items() if k != "context_sequence"},
                    }
                    rows.append(row)
                    fa_header = (
                        f"unitig_{fi}|genome={gid}|mlst={carrier['MLST']}"
                        f"|contig={h['contig'].replace(' ', '_')}"
                        f"|strand={h['strand']}|start={h['start_0based']}"
                    )
                    fa.extend([f">{fa_header}", h["context_sequence"]])

            atomic_write_csv(out_csv, pd.DataFrame(rows))
            atomic_write_text(out_fa, "\n".join(fa) + ("\n" if fa else ""))
            atomic_write_text(
                marker,
                json.dumps(
                    {
                        "script_version": SCRIPT_VERSION,
                        "design_id": DESIGN_ID,
                        "task_id": task_id,
                        "status": "PASS",
                        "completed_utc": utc_now(),
                        "elapsed_seconds": time.time() - tt,
                        "n_requested_carriers": len(carriers),
                        "n_rows": len(rows),
                        "n_exact_hits": int(sum(x.get("status") == "EXACT_HIT" for x in rows)),
                        "outputs": [str(out_csv), str(out_fa)],
                    },
                    indent=2,
                ) + "\n",
            )
            prog.update(
                f"{task_id} | carriers={len(carriers)} "
                f"| exact_hits={sum(x.get('status') == 'EXACT_HIT' for x in rows)}"
            )

        # Aggregate context tables and FASTA.
        all_occ = []
        for p in context_csvs:
            if p.is_file():
                x = pd.read_csv(p)
                if len(x):
                    all_occ.append(x)
        occ = pd.concat(all_occ, ignore_index=True) if all_occ else pd.DataFrame()
        occ_path = out_dir / "file10i_unitig_context_occurrences.csv"
        atomic_write_csv(occ_path, occ)

        combined_fa = out_dir / "file10i_unitig_contexts.fasta"
        chunks = []
        for p in context_fastas:
            if p.is_file() and p.stat().st_size:
                chunks.append(p.read_text(encoding="utf-8"))
        atomic_write_text(combined_fa, "".join(chunks))

        # Annotation manifest for the next stage.
        stage = "manifest"
        exact_hits = int((occ["status"] == "EXACT_HIT").sum()) if len(occ) and "status" in occ.columns else 0
        missing_files = int((occ["status"] == "GENOME_FILE_MISSING").sum()) if len(occ) and "status" in occ.columns else 0
        no_hits = int((occ["status"] == "NO_EXACT_HIT").sum()) if len(occ) and "status" in occ.columns else 0

        manifest = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "PASS",
            "mode": "SMOKE" if args.smoke else "FULL",
            "completed_utc": utc_now(),
            "elapsed_seconds": time.time() - start,
            "input_candidate_rows": len(cand),
            "input_redundancy_clusters": n_clusters,
            "annotation_representatives": len(reps),
            "unitig_representatives": len(unitig_reps),
            "pangenome_representatives": len(pang_reps),
            "unitig_context_exact_hits": exact_hits,
            "unitig_context_missing_genome_files": missing_files,
            "unitig_context_no_exact_hits": no_hits,
            "context_flank_bp": flank,
            "carriers_per_unitig": n_carriers,
            "interpretation_guardrail": (
                "This stage collapses redundant signals and extracts exact local sequence context. "
                "It does not assign gene identity or causal mechanism."
            ),
            "next_stage": (
                "Annotate unitig contexts with AMRFinder/sequence alignment and query PLFam IDs "
                "against BV-BRC; then perform targeted literature validation."
            ),
            "outputs": {
                "representatives": str(reps_path),
                "unitig_representatives_fasta": str(unitig_fasta),
                "plfam_representatives": str(plfam_path),
                "unitig_context_occurrences": str(occ_path),
                "unitig_contexts_fasta": str(combined_fa),
            },
        }
        final = ckpt / "file10i_final_summary.json"
        atomic_write_text(final, json.dumps(manifest, indent=2) + "\n")

        log("=" * 92)
        log("FILE10I STATUS : PASS")
        log(f"Input candidates       : {len(cand)}")
        log(f"Redundancy clusters    : {n_clusters}")
        log(f"Annotation reps        : {len(reps)}")
        log(f"Unitig reps            : {len(unitig_reps)}")
        log(f"PLFam reps             : {len(pang_reps)}")
        log(f"Exact context hits     : {exact_hits}")
        log(f"Missing genome files   : {missing_files}")
        log(f"No-exact-hit carriers  : {no_hits}")
        log(f"Elapsed                : {human_seconds(time.time()-start)}")
        log(f"Final summary          : {final}")
        log("=" * 92)

    except KeyboardInterrupt:
        failure = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "INTERRUPTED",
            "stage": stage,
            "time_utc": utc_now(),
            "elapsed_seconds": time.time() - start,
            "message": "Completed per-unitig context checkpoints are preserved; re-run to resume.",
        }
        atomic_write_text(ckpt / "file10i_last_failure.json", json.dumps(failure, indent=2) + "\n")
        log(f"FILE10I INTERRUPTED at {stage}. Checkpoints preserved.")
        raise SystemExit(130)

    except Exception as exc:
        failure = {
            "script_version": SCRIPT_VERSION,
            "design_id": DESIGN_ID,
            "status": "FAILED",
            "stage": stage,
            "time_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_text(ckpt / "file10i_last_failure.json", json.dumps(failure, indent=2) + "\n")
        log(f"FILE10I FAILED at {stage}: {type(exc).__name__}: {exc}")
        log(f"Crash record: {ckpt / 'file10i_last_failure.json'}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
