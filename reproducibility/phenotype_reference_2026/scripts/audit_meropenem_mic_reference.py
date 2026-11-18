#!/usr/bin/env python3
"""Check reference labels against deposited MIC intervals under explicit rules.

This is a conditional consistency audit, not laboratory AST adjudication.
Neither unchanged historical MIC nor the actual laboratory breakpoint version
is inferred. No predictions or phenotype source files are modified.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(value):
    return {"resistant": "R", "susceptible": "S", "intermediate": "I",
            "r": "R", "s": "S", "i": "I"}[(value or "").strip().lower()]


def measurement(sign, value):
    """Return lower/upper limits and whether each endpoint is included."""
    value = Decimal(value)
    if not value.is_finite() or value < 0:
        raise ValueError("Invalid MIC")
    zero, infinity = Decimal(0), Decimal("Infinity")
    if sign in {"=", "=="}:
        return value, True, value, True
    if sign == "<":
        return zero, True, value, False
    if sign in {"<=", "≤"}:
        return zero, True, value, True
    if sign == ">":
        return value, False, infinity, False
    if sign in {">=", "≥"}:
        return value, True, infinity, False
    raise ValueError("Missing or unsupported MIC sign; equality is not imputed")


def intersects(a, b):
    lo, hi = max(a[0], b[0]), min(a[2], b[2])
    if lo < hi:
        return True
    if lo != hi or not lo.is_finite():
        return False
    return all((lo > v[0] or v[1]) and (lo < v[2] or v[3]) for v in (a, b))


def possible_categories(interval, susceptible_max, resistant_gt):
    s, r = Decimal(susceptible_max), Decimal(resistant_gt)
    if s < 0 or r < s:
        raise ValueError("Invalid breakpoint ordering")
    categories = {"S": (Decimal(0), True, s, True),
                  "R": (r, False, Decimal("Infinity"), False)}
    if s < r:
        categories["I"] = (s, False, r, True)
    return {name for name, region in categories.items() if intersects(interval, region)}


def source_labels(path):
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["target_acc"], row["biosample_acc"], row["antibiotic"]
            value = label(row["phenotype"])
            if key in result and result[key] != value:
                raise ValueError("Conflicting same-key source labels")
            result[key] = value
    return result


def live_records(paths):
    result = {}
    for path in paths:
        root = ET.parse(path).getroot()
        for sample in root.findall(".//BioSample"):
            accession = sample.attrib["accession"]
            if accession in result:
                raise ValueError("Duplicate live BioSample accession")
            matches = []
            for table in sample.findall("./Description/Comment/Table"):
                if table.findtext("Caption") != "Antibiogram":
                    continue
                fields = ["".join(x.itertext()).strip() for x in table.findall("./Header/Cell")]
                for row in table.findall("./Body/Row"):
                    values = ["".join(x.itertext()).strip() for x in row.findall("Cell")]
                    if len(values) != len(fields):
                        raise ValueError("Antibiogram header/row width mismatch")
                    record = dict(zip(fields, values))
                    if record.get("Antibiotic", "").lower() == "meropenem":
                        matches.append(record)
            if len(matches) != 1:
                raise ValueError("Missing or ambiguous meropenem observation")
            attrs = {x.get("attribute_name"): (x.text or "").strip()
                     for x in sample.findall("./Attributes/Attribute")}
            result[accession] = {"ast": matches[0], "source_type": attrs.get("source_type", ""),
                                 "host": attrs.get("host", ""),
                                 "isolate_name_alias": attrs.get("isolate_name_alias", ""),
                                 "strain": attrs.get("strain", ""),
                                 "organism_name": sample.findtext("./Description/Organism/OrganismName", ""),
                                 "last_update": sample.get("last_update", "")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--october", type=Path, required=True)
    parser.add_argument("--live-xml", type=Path, nargs="+", required=True)
    parser.add_argument("--breakpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.crosswalk, args.historical, args.october, args.breakpoints, *args.live_xml]
    hashes = {str(p.resolve()): digest(p) for p in inputs}
    config = json.loads(args.breakpoints.read_text())
    if (config["organism_group"], config["drug"], config["unit"]) != ("Enterobacterales", "meropenem", "mg/L"):
        raise ValueError("Unexpected breakpoint scope")
    for version in config["versions"]:
        pdf = args.breakpoints.parent / version["pdf_file"]
        if digest(pdf) != version["pdf_sha256"]:
            raise ValueError("Official breakpoint PDF hash mismatch")
    with args.crosswalk.open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["antibiotic"] == "meropenem"]
    ids = {r["biosample_acc"] for r in rows}
    if len(rows) != 175 or len(ids) != 175:
        raise ValueError("This protocol is restricted to the 175 focal meropenem records")
    old, october, fresh = source_labels(args.historical), source_labels(args.october), live_records(args.live_xml)
    if set(fresh) != ids:
        raise ValueError("Live BioSample response does not exactly cover the fixed focal panel")
    results, distribution, source_types = [], Counter(), Counter()
    for row in rows:
        accession = row["biosample_acc"]
        a = old[row["historical_target_acc"], accession, "meropenem"]
        b = october[row["current_target_acc"], accession, "meropenem"]
        live = fresh[accession]
        record = live["ast"]
        c = label(record["Resistance phenotype"])
        if a != label(row["historical_file11_raw_label"]) or b != label(row["current_bigquery_label"]):
            raise ValueError("Crosswalk/source CSV mismatch")
        projected = {"mic_sign": "Measurement sign", "mic_value": "Measurement",
                     "mic_units": "Measurement units", "mic_method": "Laboratory typing method",
                     "mic_platform": "Laboratory typing platform", "testing_standard": "Testing standard"}
        if any(row[field] != record[xmlfield] for field, xmlfield in projected.items()):
            raise ValueError("Live MIC metadata changed since supplied October crosswalk")
        if record["Measurement units"] != "mg/L" or record["Testing standard"] != "EUCAST":
            raise ValueError("Unexpected unit/standard; not silently harmonized")
        if not live["organism_name"].startswith("Klebsiella pneumoniae"):
            raise ValueError("Unexpected source organism annotation")
        interval = measurement(record["Measurement sign"], record["Measurement"])
        possibilities = {}
        for version in config["versions"]:
            for rule in version["rules"]:
                possibilities[version["version"] + "/" + rule["context"]] = possible_categories(interval, rule["S_le"], rule["R_gt"])
        union = set().union(*possibilities.values())
        robust = next(iter(union)) if len(union) == 1 else "UNRESOLVED"
        results.append({"biosample_acc": accession, "isolate_name_alias": live["isolate_name_alias"],
                        "strain": live["strain"], "source_type": live["source_type"],
                        "historical_label": a, "october_label": b, "november_live_label": c,
                        "deposited_mic_sign": record["Measurement sign"],
                        "deposited_mic_value": record["Measurement"], "unit": "mg/L",
                        "EUCAST_2021_2024_both_contexts_category": robust,
                        "historical_compatible_with_deposited_MIC_all_rules": all(a in x for x in possibilities.values()),
                        "october_compatible_with_deposited_MIC_all_rules": all(b in x for x in possibilities.values()),
                        "live_compatible_with_deposited_MIC_all_rules": all(c in x for x in possibilities.values())})
        distribution[(record["Measurement sign"], record["Measurement"], robust)] += 1
        source_types[live["source_type"]] += 1
    for path in inputs:
        if digest(path) != hashes[str(path.resolve())]:
            raise ValueError("Input changed during audit")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "meropenem_MIC_reference_consistency.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = {"status": "CONDITIONAL_DEPOSITED_MIC_RULE_CONSISTENCY_NOT_BMD_ADJUDICATION",
              "records": len(results), "input_sha256": hashes,
              "historical_compatible_n": sum(r["historical_compatible_with_deposited_MIC_all_rules"] for r in results),
              "october_compatible_n": sum(r["october_compatible_with_deposited_MIC_all_rules"] for r in results),
              "november_live_compatible_n": sum(r["live_compatible_with_deposited_MIC_all_rules"] for r in results),
              "october_and_november_labels_equal_n": sum(r["october_label"] == r["november_live_label"] for r in results),
              "unresolved_n": sum(r["EUCAST_2021_2024_both_contexts_category"] == "UNRESOLVED" for r in results),
              "mic_distribution": [{"sign": k[0], "value": k[1], "category": k[2], "n": n} for k, n in sorted(distribution.items())],
              "source_type_annotations": dict(source_types),
              "limitations": ["Historical MIC is unavailable; unchanged MIC across historical snapshots is not established.",
                              "Agreement with the same deposited MIC is not independent laboratory validation.",
                              "Actual breakpoint version and assay validity remain unverified.",
                              "Source metadata includes nonhuman entries; this audit does not validate clinical use.",
                              "No model scores enter category determination."]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"input_sha256", "limitations"}}, indent=2))


if __name__ == "__main__":
    main()
