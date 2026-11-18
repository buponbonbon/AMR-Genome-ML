#!/usr/bin/env python3
"""Archive explicitly selected, public NCBI retained-build metadata TSVs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def fetch_one(directory, build, out, limit):
    url = f"https://ftp.ncbi.nlm.nih.gov/pathogen/Results/{directory}/{build}/Metadata/{build}.metadata.tsv"
    partial = out / (build + ".metadata.tsv.partial")
    meta = {"build": build, "url": url, "started_utc": datetime.now(timezone.utc).isoformat()}
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AMRProvenanceAudit/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                meta.update(http_status=response.status, final_url=response.url,
                            http_last_modified=response.headers.get("Last-Modified"),
                            content_type=response.headers.get("Content-Type"))
                length = response.headers.get("Content-Length", "")
                if length.isdigit() and int(length) > limit:
                    raise ValueError("Declared file size exceeds fixed guard")
                sha, size = hashlib.sha256(), 0
                with partial.open("wb") as handle:
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > limit:
                            raise ValueError("Downloaded file size exceeds fixed guard")
                        handle.write(block)
                        sha.update(block)
                if length.isdigit() and size != int(length):
                    raise ValueError("Content-Length mismatch")
            with partial.open("r", encoding="utf-8-sig") as handle:
                header = handle.readline().rstrip("\r\n").split("\t")
            if not {"target_acc", "biosample_acc", "AST_phenotypes", "scientific_name"} <= set(header):
                raise ValueError("Required AST metadata schema is absent")
            final = out / (build + ".metadata.tsv")
            partial.rename(final)
            meta.update(status="COMPLETE", file=final.name, bytes=size, sha256=sha.hexdigest(),
                        completed_utc=datetime.now(timezone.utc).isoformat(), column_count=len(header))
            return meta
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 502, 503, 504} and attempt < 2:
                retry = exc.headers.get("Retry-After", "")
                time.sleep(min(20, max(2, int(retry))) if retry.isdigit() else 3 * (attempt + 1))
                continue
            meta.update(status="FAILED", error=str(exc))
            return meta
        except Exception as exc:
            meta.update(status="FAILED", error=str(exc))
            return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = args.protocol.read_bytes()
    protocol = json.loads(data)
    args.out.mkdir(parents=True, exist_ok=False)
    tasks = [(row["directory"], row[phase + "_build"]) for row in protocol["targets"] for phase in ["old", "new"]]
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch_one, directory, build, args.out, protocol["file_size_guard_bytes"])
                   for directory, build in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    manifest = {"protocol_sha256": hashlib.sha256(data).hexdigest(),
                "responses": sorted(results, key=lambda r: r["build"]),
                "all_complete": all(r["status"] == "COMPLETE" for r in results)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["all_complete"]:
        raise SystemExit("Some retained metadata files are unavailable; no complete-panel inference")


if __name__ == "__main__":
    main()
