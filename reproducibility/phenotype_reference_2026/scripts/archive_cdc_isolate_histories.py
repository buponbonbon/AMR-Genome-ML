#!/usr/bin/env python3
"""Archive public CDC isolate change logs; updates are not automatically errors."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.stack, self.text, self.ignore = [], [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignore += 1
        if tag == "table":
            self.stack.append({"rows": [], "row": None, "cell": None})
        if self.stack:
            t = self.stack[-1]
            if tag == "tr":
                t["row"] = []
            elif tag in {"td", "th"}:
                t["cell"] = []

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.ignore = max(0, self.ignore - 1)
        if self.stack:
            t = self.stack[-1]
            if tag in {"td", "th"} and t["cell"] is not None:
                if t["row"] is not None:
                    t["row"].append(" ".join(" ".join(t["cell"]).split()))
                t["cell"] = None
            elif tag == "tr" and t["row"] is not None:
                t["rows"].append(t["row"])
                t["row"] = None
            elif tag == "table":
                self.tables.append(self.stack.pop()["rows"])

    def handle_data(self, data):
        if not self.ignore and data.strip():
            self.text.append(data.strip())
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"].append(data)


def get(case, out):
    url, panel = case
    isolate = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["IsolateID"][0]
    record = {"url": url, "panel_id": panel, "isolate_id": isolate}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AMRReferenceAudit/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("Isolate HTML exceeded 2 MiB guard")
            record.update(http_status=response.status, retrieved_utc=datetime.now(timezone.utc).isoformat())
        path = out / f"isolate_{isolate}_panel_{panel}.html"
        path.write_bytes(raw)
        record.update(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        parser = Tables()
        parser.feed(raw.decode("utf-8-sig"))
        histories = [t for t in parser.tables if any("Isolate History" in " ".join(row) for row in t)]
        if len(histories) > 1:
            raise ValueError("Ambiguous history tables")
        rows = []
        if histories:
            for row in histories[0]:
                if row and re.fullmatch(r"\d\d/\d\d/\d\d", row[0]):
                    if len(row) != 2:
                        raise ValueError("History schema differs from inspected two-column rows")
                    rows.append({"date_raw": row[0], "description": row[1]})
        record["history_rows"] = rows
        record["history_table_present"] = bool(histories)
        text = " ".join(parser.text)
        record["CLSI_editions_stated"] = sorted(set(re.findall(r"CLSI\s+20\d\d\s+M100\s+S\d+", text)))
        record["biosamples"] = sorted(set(re.findall(r"SAM[NED][A-Z]?\d+", text)))
        record["BMD_stated"] = "broth microdilution" in text.lower()
        record["status"] = "COMPLETE"
    except Exception as exc:
        record.update(status="FAILED", error=str(exc))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    panel_manifest = json.loads((args.panel_dir / "manifest.json").read_text())
    tasks = [(url, case["panel"]["id"]) for case in panel_manifest["results"] for url in case["isolate_detail_urls"]]
    unique_isolates = {urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["IsolateID"][0] for url, _ in tasks}
    if len(unique_isolates) != len(tasks):
        raise ValueError("Repeated isolates require explicit deduplication before download")
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "EXPLORATORY_AMENDMENT_AFTER_FOUR_EXAMPLE_HISTORY_PAGES_AND_CDC_EXPORTS",
                "scope": "All isolates in four previously selected panels, without filtering on disagreement or history outcome",
                "public_pages": len(tasks), "no_author_contact": True,
                "limitations": ["A log entry is evidence of a documented update, not evidence of measurement or labeling error.",
                                "Absence of an MIC entry does not prove MIC remained unchanged.",
                                "This archive is a feasibility dataset; it is not an untouched evaluation holdout."]}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(get, case, args.out) for case in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 25 == 0:
                print(json.dumps({"complete_requests": len(records), "planned": len(tasks)}), flush=True)
    events = []
    pattern = re.compile(r"^(MIC|INT) was updated with a new value for (.+): from '(.*)' to '(.*)'$")
    for case in records:
        for row in case.get("history_rows", []):
            match = pattern.fullmatch(row["description"])
            base = {"isolate_id": case["isolate_id"], "panel_id": case["panel_id"], "biosamples": "|".join(case.get("biosamples", [])),
                    "url": case["url"], "html_sha256": case.get("sha256"), **row}
            if match:
                kind, drug, old, new = match.groups()
                base.update(field=kind, antibiotic=drug.lower(), old_value=old, new_value=new,
                            classification="VALUE_UPDATE" if old not in {"", "---"} and new not in {"", "---"} else "MISSINGNESS_OR_NEW_TEST_UPDATE")
            else:
                base.update(field="OTHER", antibiotic="", old_value="", new_value="", classification="OTHER_DOCUMENTED_CHANGE")
            events.append(base)
    fields = ["isolate_id", "panel_id", "biosamples", "url", "html_sha256", "date_raw", "description", "field", "antibiotic", "old_value", "new_value", "classification"]
    with (args.out / "documented_history_events.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    (args.out / "manifest.json").write_text(json.dumps({"protocol": protocol, "all_complete": all(r["status"] == "COMPLETE" for r in records),
                                                        "results": sorted(records, key=lambda r: int(r["isolate_id"]))}, indent=2) + "\n")
    print(json.dumps({"requests": len(records), "failed": sum(r["status"] != "COMPLETE" for r in records), "history_rows": len(events)}))


if __name__ == "__main__":
    main()
