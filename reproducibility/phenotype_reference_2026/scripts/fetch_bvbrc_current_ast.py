#!/usr/bin/env python3
"""Archive current public BV-BRC AMR rows for an explicit candidate panel.

Preserves laboratory and computational evidence separately; never trains models,
assigns clinical truth, or silently truncates query results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.bv-brc.org/api/genome_amr/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def fetch_batch(ids, antibiotic, number, cache, limit):
    query = f"in(genome_id,({','.join(ids)}))&eq(antibiotic,{antibiotic})&limit({limit})"
    url = API + "?" + urllib.parse.quote(query, safe="(),._-&=")
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "AMRProvenanceResearch/1.0"})
    for attempt in range(3):
        started = datetime.now(timezone.utc).isoformat()
        try:
            with urllib.request.urlopen(req, timeout=50) as response:
                payload = response.read(12_000_001)
                content_range = response.headers.get("Content-Range", "")
                final_url = response.url
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 502, 503, 504} or attempt == 2:
                raise
            retry = exc.headers.get("Retry-After", "")
            pause = min(30, max(1, int(retry))) if retry.isdigit() else (attempt + 1) * 2
            time.sleep(pause)
    if len(payload) > 12_000_000:
        raise ValueError("Response exceeded the explicit size guard")
    records = json.loads(payload)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("Unexpected API schema; raw archive is not a validated panel")
    total = re.search(r"/(\d+)$", content_range)
    if total is None or int(total[1]) != len(records) or len(records) >= limit:
        raise ValueError("Cannot establish complete query retrieval from Content-Range")
    for row in records:
        if str(row.get("genome_id", "")) not in ids or str(row.get("antibiotic", "")).lower() != antibiotic:
            raise ValueError("Provider returned a record outside the requested panel")
        if not row.get("id"):
            raise ValueError("Current record lacks a native provider ID")
    name = f"batch_{number:04d}.json"
    (cache / name).write_bytes(payload)
    return records, {"batch": number, "request_url": url, "final_url": final_url,
                     "requested_genome_ids": ids, "record_count": len(records),
                     "content_range": content_range, "response_sha256": digest(payload),
                     "response_file": name, "request_started_utc": started,
                     "completed_utc": datetime.now(timezone.utc).isoformat()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--antibiotic", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.batch_size <= 100 or not 1 <= args.workers <= 3:
        p.error("Use batch size 1..100 and workers 1..3")
    if not re.fullmatch(r"[a-z][a-z_-]+", args.antibiotic):
        p.error("Antibiotic must be an explicit simple provider value")
    original = args.panel.read_bytes()
    with args.panel.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["species_annotation"] == args.species and r["antibiotic"] == args.antibiotic]
    ids = sorted({r["genome_id"] for r in rows})
    if not ids or any(not re.fullmatch(r"\d+\.\d+", x) for x in ids):
        p.error("Panel is empty or has an unexpected public genome ID")
    args.out.mkdir(parents=True, exist_ok=False)
    cache = args.out / "responses"
    cache.mkdir()
    protocol = {"species_source_annotation": args.species, "antibiotic": args.antibiotic,
                "panel_sha256": digest(original), "panel_path": str(args.panel.resolve()),
                "candidate_genome_n": len(ids), "candidate_genome_ids": ids,
                "api": API, "fetch_all_evidence_categories": True,
                "lab_evidence_for_analysis": "Laboratory Method",
                "computational_predictions_are_AST_truth": False,
                "per_batch_limit": 5000, "workers": args.workers,
                "frozen_before_full_fetch_utc": datetime.now(timezone.utc).isoformat(),
                "prior_exposure": "Connectivity probe inspected genome 287.1250 before this full-panel protocol. This is exploratory source screening, not a new blinded clinical validation."}
    write_json(args.out / "protocol.json", protocol)
    batches = [ids[i:i + args.batch_size] for i in range(0, len(ids), args.batch_size)]
    all_records, manifest = [], []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_batch, batch, args.antibiotic, i + 1, cache, 5000)
                       for i, batch in enumerate(batches)]
            for future in as_completed(futures):
                records, meta = future.result()
                all_records.extend(records)
                manifest.append(meta)
                print(f"Completed batch {meta['batch']}/{len(batches)}: {len(records)} records", flush=True)
        unique = {}
        for row in all_records:
            key = row["id"]
            if key in unique and unique[key] != row:
                raise ValueError("Conflicting duplicate native record IDs")
            unique[key] = row
        if digest(args.panel.read_bytes()) != digest(original):
            raise ValueError("Candidate panel changed during retrieval")
        records = sorted(unique.values(), key=lambda r: (r["genome_id"], r["id"]))
        write_json(args.out / "current_all_evidence.json", records)
        write_json(args.out / "current_laboratory_only.json", [r for r in records if r.get("evidence") == "Laboratory Method"])
        write_json(args.out / "retrieval_manifest.json", {
            "status": "COMPLETE_EXPLICIT_PANEL_RETRIEVAL", "requested_genome_n": len(ids),
            "records": len(records), "batches": sorted(manifest, key=lambda r: r["batch"]),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "clinical_adjudication": False})
        print(f"Retrieved {len(records)} current records for {len(ids)} requested genome IDs.")
    except Exception as exc:
        write_json(args.out / "FAILED.json", {"status": "INCOMPLETE_NOT_VALIDATED", "error": str(exc),
                   "completed_batches": sorted(manifest, key=lambda r: r["batch"])})
        raise


if __name__ == "__main__":
    main()
