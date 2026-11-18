#!/usr/bin/env python3
"""Verify raw CDC/XML joins and interval comparisons independently of pandas parsing."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
from datetime import datetime
from html import unescape
from io import BytesIO
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from openpyxl import load_workbook


def measurement(raw):
    raw = "".join(raw.split()).replace("≤", "<=").replace("≥", ">=")
    m = re.fullmatch(r"([<>=]*)([0-9.]+)", raw)
    op = m.group(1)
    return ("=" if op in {"", "=", "=="} else op), Decimal(m.group(2))


def status(first, second):
    a, b = measurement(first), measurement(second)
    if a == b:
        return "IDENTICAL_REPORTED_MIC_INTERVAL"
    def accepts(point, condition):
        op, val = condition
        return {"=": point == val, "<": point < val, "<=": point <= val, ">": point > val, ">=": point >= val}[op]
    # Both predicates change truth only at these two numeric boundaries.
    points = {a[1], b[1], min(a[1], b[1])/2, max(a[1], b[1])*2, (a[1]+b[1])/2}
    return "DIFFERENT_REPORTING_INTERVALS_OVERLAP" if any(accepts(x,a) and accepts(x,b) for x in points) else "DISJOINT_REPORTED_MIC_INTERVALS"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    boundary_cases = [("<=1", "1", "DIFFERENT_REPORTING_INTERVALS_OVERLAP"),
                      ("<1", "1", "DISJOINT_REPORTED_MIC_INTERVALS"),
                      (">8", "8", "DISJOINT_REPORTED_MIC_INTERVALS"),
                      (">=8", "8", "DIFFERENT_REPORTING_INTERVALS_OVERLAP"),
                      ("==4", "4", "IDENTICAL_REPORTED_MIC_INTERVAL"),
                      ("<=0.5", "8", "DISJOINT_REPORTED_MIC_INTERVALS")]
    for a, b, expected_status in boundary_cases:
        if status(a, b) != expected_status:
            raise ValueError("Independent censoring-boundary check failed")
    cdc, ncbi = {}, {}
    manifest = json.loads((args.base / "cdc_reference/manifest.json").read_text())
    for source in manifest["results"]:
        book = load_workbook(BytesIO((args.base / "cdc_reference" / source["export"]["file"]).read_bytes()), read_only=True, data_only=True)
        records = book.active.values
        headers = next(records)
        for values in records:
            row = {k: str(v).strip() if v is not None else "" for k,v in zip(headers,values)}
            for drug in ["Meropenem", "Imipenem", "Ertapenem"]:
                if row.get(drug + " MIC"):
                    key = row["BioSampleAccessionNum"], drug.lower()
                    if key in cdc:
                        raise ValueError("Unexpected repeated CDC BioSample/drug")
                    cdc[key] = row[drug + " MIC"], row.get(drug + " INT", "")
        book.close()
    for path in (args.base / "cdc_linked_biosamples").glob("batch_*.xml"):
        for sample in ET.parse(path).getroot().findall("BioSample"):
            for table in sample.findall(".//Table"):
                if not table.attrib.get("class", "").startswith("Antibiogram"):
                    continue
                headers = [x.text or "" for x in table.findall("Header/Cell")]
                for cells in table.findall("Body/Row"):
                    row = dict(zip(headers, [(x.text or "") for x in cells.findall("Cell")]))
                    key = sample.attrib["accession"], row["Antibiotic"].lower()
                    if key in ncbi:
                        raise ValueError("Unexpected repeated NCBI BioSample/drug")
                    ncbi[key] = row["Measurement sign"] + row["Measurement"], row["Resistance phenotype"]
    produced = list(csv.DictReader((args.audit_dir / "primary_MIC_observation_comparisons.csv").open()))
    expected = set(cdc) & set(ncbi)
    if expected != {(r["biosample_acc"],r["antibiotic"]) for r in produced} or len(expected) != len(produced):
        raise ValueError("Raw source join mismatch")
    counts = Counter()
    for row in produced:
        key = row["biosample_acc"],row["antibiotic"]
        if cdc[key] != (row["CDC_MIC_raw"],row["CDC_INT_raw"]) or ncbi[key] != (row["NCBI_MIC_raw"],row["NCBI_INT_raw"]):
            raise ValueError("Raw MIC or phenotype differs from output")
        observed = status(ncbi[key][0],cdc[key][0])
        if observed != row["MIC_comparison"]:
            raise ValueError("Independent decimal interval check differs")
        counts[observed] += 1
    raw_histories = set()
    for path in (args.base / "cdc_isolate_histories").glob("isolate_*.html"):
        isolate = path.name.split("_")[1]
        for date, description in re.findall(r">(\d\d/\d\d/\d\d)</td>\s*<td[^>]*>(.*?)</td>",path.read_text(),flags=re.S):
            raw_histories.add((isolate,date," ".join(unescape(re.sub(r"<[^>]+>","",description)).split())))
    csv_histories = {(r["isolate_id"],r["date_raw"],r["description"]) for r in csv.DictReader((args.base / "cdc_isolate_histories/documented_history_events.csv").open())}
    # HTMLParser splits malformed '< =value' text into fragments; compare
    # after normalizing whitespace around inequality operators only.
    normalize = lambda item: (item[0], item[1], re.sub(r"\s*([<>])\s*(=?)\s*", r"\1\2", item[2]))
    if {normalize(x) for x in raw_histories} != {normalize(x) for x in csv_histories}:
        raise ValueError("Independent raw history extraction differs")
    history_manifest = json.loads((args.base / "cdc_isolate_histories/manifest.json").read_text())
    sample_by_isolate = {r["isolate_id"]: r.get("biosamples", []) for r in history_manifest["results"]}
    independently_linked_updates = set()
    for isolate, date, description in raw_histories:
        match = re.fullmatch(r"MIC was updated with a new value for (.+): from '(.*)' to '(.*)'", description)
        if not match:
            continue
        drug, old, new = match.groups()
        for sample in sample_by_isolate.get(isolate, []):
            key = sample, drug.lower()
            if key not in cdc or key not in ncbi:
                continue
            try:
                if measurement(old) != measurement(new) and measurement(old) == measurement(ncbi[key][0]) and measurement(new) == measurement(cdc[key][0]):
                    independently_linked_updates.add((sample,drug.lower(),datetime.strptime(date,"%m/%d/%y").date().isoformat()))
            except (AttributeError, ValueError):
                continue
    produced_updates = {(r["biosample_acc"], r["antibiotic"], r["CDC_documented_update_date"]) for r in csv.DictReader((args.audit_dir / "documented_CDC_update_NCBI_retains_old_MIC.csv").open())}
    if independently_linked_updates != produced_updates:
        raise ValueError("Independent documented-value update linkage differs")
    report = {"status":"PASS","method":"Openpyxl + XML raw-source join, independent decimal predicate intersection, regex HTML history extraction (inequality whitespace normalized)",
              "censoring_boundary_cases_passed": len(boundary_cases),
              "verified_documented_update_old_value_cases": len(independently_linked_updates),
              "verified_unique_MIC_pairs":len(produced),"interval_counts":dict(counts),"verified_unique_history_entries":len(raw_histories)}
    with args.out.open("x") as handle:
        json.dump(report,handle,indent=2);handle.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
