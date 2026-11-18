"""Verify the curated archive against its size and SHA-256 manifest."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    failures = []
    checked = 0
    seen = set()
    with (root / "FILE_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relative = row["relative_path"]
            path = root / relative
            if relative in seen or not path.resolve().is_relative_to(root):
                failures.append({"file": relative, "problem": "duplicate or invalid path"})
                continue
            seen.add(relative)
            if not path.is_file():
                failures.append({"file": relative, "problem": "missing"})
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
                failures.append({"file": relative, "problem": "size or checksum mismatch"})
            checked += 1
    if checked == 0:
        failures.append({"problem": "empty manifest"})
    print(json.dumps({"status": "FAIL" if failures else "PASS",
                      "files_checked": checked, "failures": failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
