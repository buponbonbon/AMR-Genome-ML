#!/usr/bin/env python3
"""Secondary conditional rule checks on three drugs in the fixed 175 samples."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from audit_meropenem_mic_reference import digest, label, measurement, possible_categories, source_labels


RULES = {"meropenem": [("2", "8"), ("2", "2")],
         "imipenem": [("2", "4")], "ertapenem": [("0.5", "0.5")]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--october", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = args.reference_dir
    config = json.loads((base / "meropenem_breakpoints.json").read_text())
    for version in config["versions"]:
        if digest(base / version["pdf_file"]) != version["pdf_sha256"]:
            raise ValueError("Breakpoint PDF hash mismatch")
        page = (base / (Path(version["pdf_file"]).stem + ".txt")).read_text().split("\f")[version["pdf_page"] - 1]
        if not (re.search(r"Ertapenem\s+0\.5\s+0\.5\s", page)
                and re.search(r"Imipenem, Enterobacterales except\s+2\s+4\s", page)):
            raise ValueError("Additional drug breakpoints not verified in specified source page")
    with args.crosswalk.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 525 or len({r["biosample_acc"] for r in rows}) != 175:
        raise ValueError("Unexpected fixed-panel size")
    old, october = source_labels(args.historical), source_labels(args.october)
    fresh, source_types = {}, {}
    paths = sorted(base.glob("fresh_focal_biosample_*.xml"))
    for path in paths:
        for sample in ET.parse(path).getroot().findall(".//BioSample"):
            accession = sample.attrib["accession"]
            attrs = {e.get("attribute_name"): (e.text or "").strip() for e in sample.findall("./Attributes/Attribute")}
            source_types[accession] = attrs.get("source_type", "")
            for table in sample.findall("./Description/Comment/Table"):
                if table.findtext("Caption") != "Antibiogram":
                    continue
                fields = ["".join(e.itertext()).strip() for e in table.findall("./Header/Cell")]
                for row in table.findall("./Body/Row"):
                    values = ["".join(e.itertext()).strip() for e in row.findall("Cell")]
                    if len(fields) != len(values):
                        raise ValueError("Table width mismatch")
                    record = dict(zip(fields, values))
                    if record.get("Antibiotic") in RULES:
                        key = accession, record["Antibiotic"]
                        if key in fresh:
                            raise ValueError("Ambiguous observation")
                        fresh[key] = record
    if set(fresh) != {(r["biosample_acc"], r["antibiotic"]) for r in rows}:
        raise ValueError("Live observations differ from fixed source panel")
    results, summary = [], defaultdict(Counter)
    for row in rows:
        bio, drug = row["biosample_acc"], row["antibiotic"]
        record = fresh[bio, drug]
        a = old[row["historical_target_acc"], bio, drug]
        b = october[row["current_target_acc"], bio, drug]
        if a != label(row["historical_file11_raw_label"]) or b != label(row["current_bigquery_label"]) or b != label(record["Resistance phenotype"]):
            raise ValueError("Label/source mismatch")
        for field, xmlfield in [("mic_sign", "Measurement sign"), ("mic_value", "Measurement"),
                                ("mic_units", "Measurement units"), ("testing_standard", "Testing standard")]:
            if row[field] != record[xmlfield]:
                raise ValueError("Current MIC metadata changed")
        if record["Testing standard"] != "EUCAST" or record["Measurement units"] != "mg/L":
            raise ValueError("Unexpected standard/units")
        sets = [possible_categories(measurement(row["mic_sign"], row["mic_value"]), s, r) for s, r in RULES[drug]]
        categories = set().union(*sets)
        agood, bgood = all(a in s for s in sets), all(b in s for s in sets)
        results.append({"biosample_acc": bio, "antibiotic": drug, "source_type_annotation": source_types[bio],
                        "historical_label": a, "october_and_live_label": b, "current_mic_sign": row["mic_sign"],
                        "current_mic_value": row["mic_value"], "unit": "mg/L",
                        "possible_categories_across_checked_rules": "|".join(sorted(categories)),
                        "historical_compatible_with_current_MIC": agood, "current_compatible_with_current_MIC": bgood})
        for key, value in {"records": 1, "historical_compatible": agood, "current_compatible": bgood,
                           "current_incompatible": not bgood}.items():
            summary[drug][key] += value
    flags = [r for r in results if not r["current_compatible_with_current_MIC"]]
    args.out.mkdir(parents=True, exist_ok=False)
    for name, records in [("three_drug_MIC_consistency.csv", results), ("current_MIC_label_flags.csv", flags)]:
        with (args.out / name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(records)
    report = {"status": "SECONDARY_CONDITIONAL_RULE_CHECK_NOT_AST_ERROR_ADJUDICATION",
              "rule_years": [2021, 2022, 2023, 2024], "by_drug": {k: dict(v) for k, v in summary.items()},
              "current_flag_records": len(flags), "flag_unique_biosamples": len({r["biosample_acc"] for r in flags}),
              "flag_source_type_annotations": dict(Counter(r["source_type_annotation"] for r in flags)),
              "input_sha256": {str(f.resolve()): digest(f) for f in [args.crosswalk, args.historical, args.october, *paths]},
              "interpretation": "Flags concern current label/MIC/standard combinations under named rules. They do not identify which field, actual rule, assay or software is responsible. No historical MIC or clinical truth is established."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
