#!/usr/bin/env python3
"""Compare public primary MIC records as censored intervals; never relabel observations."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd


def interval(raw):
    text = re.sub(r"\s+", "", raw).replace("≤", "<=").replace("≥", ">=")
    m = re.fullmatch(r"(<=|>=|==|<|>|=)?([0-9]+(?:\.[0-9]+)?)", text)
    if not m:
        return None
    op, value = m.group(1) or "==", float(m.group(2))
    if value <= 0:
        return None
    if op in {"==", "="}:
        return value, value, True, True
    if op in {"<", "<="}:
        return 0., value, False, op == "<="
    return value, math.inf, op == ">=", False


def overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if lo < hi:
        return True
    if lo > hi:
        return False
    def contains(x, t):
        return (x > t[0] or x == t[0] and t[2]) and (x < t[1] or x == t[1] and t[3])
    return contains(lo, a) and contains(lo, b)


def categories(mic, s, r):
    ranges = {"S": (0., s, False, True), "I": (s, r, False, False), "R": (r, math.inf, True, False)}
    return {name for name, span in ranges.items() if overlap(mic, span)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--breakpoints", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.base / "cdc_linked_biosamples"
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest["all_complete"]:
        raise ValueError("BioSample retrieval incomplete")
    observations = defaultdict(list)
    mapping = {"susceptible": "S", "intermediate": "I", "resistant": "R", "not defined": "ND"}
    for entry in manifest["responses"]:
        raw = (source / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("BioSample hash mismatch")
        for sample in ET.fromstring(raw).findall("BioSample"):
            for ti, table in enumerate(sample.findall(".//Table")):
                if not table.attrib.get("class", "").startswith("Antibiogram"):
                    continue
                headers = [(x.text or "").strip() for x in table.findall("./Header/Cell")]
                for ri, row in enumerate(table.findall("./Body/Row")):
                    cells = [(x.text or "").strip() for x in row.findall("Cell")]
                    if len(cells) != len(headers):
                        raise ValueError("BioSample AST schema mismatch")
                    obs = dict(zip(headers, cells))
                    key = sample.attrib["accession"], obs["Antibiotic"].lower()
                    observations[key].append({"NCBI_MIC_raw": obs["Measurement sign"] + obs["Measurement"],
                                              "NCBI_units_raw": obs["Measurement units"],
                                              "NCBI_INT_raw": obs["Resistance phenotype"],
                                              "NCBI_SIR": mapping.get(obs["Resistance phenotype"].lower(), ""),
                                              "NCBI_testing_standard_raw": obs["Testing standard"],
                                              "NCBI_last_update": sample.attrib.get("last_update", ""),
                                              "XML_table_index": ti, "XML_row_index": ri})
    cdc = pd.read_csv(args.base / "cdc_reference_linkage_final/CDC_primary_MIC_rows.csv", dtype=str, keep_default_na=False)
    history = json.loads((args.base / "cdc_isolate_histories/manifest.json").read_text())
    editions = {b: r["CLSI_editions_stated"] for r in history["results"] for b in r.get("biosamples", [])}
    bp = pd.read_csv(args.breakpoints, sep="\t", dtype=str, keep_default_na=False)
    rows = []
    for row in cdc.to_dict("records"):
        key = row["biosample_acc"], row["antibiotic"]
        for obs in observations.get(key, []):
            a, b = interval(obs["NCBI_MIC_raw"]), interval(row["CDC_MIC_raw"])
            units_ok = obs["NCBI_units_raw"].lower() in {"mg/l", "ug/ml", "μg/ml", "µg/ml"}
            if not a or not b or not units_ok:
                mic_status = "NOT_COMPARABLE"
            elif a == b:
                mic_status = "IDENTICAL_REPORTED_MIC_INTERVAL"
            elif overlap(a, b):
                mic_status = "DIFFERENT_REPORTING_INTERVALS_OVERLAP"
            else:
                mic_status = "DISJOINT_REPORTED_MIC_INTERVALS"
            result = {**row, **obs, "MIC_comparison": mic_status,
                      "same_SIR": row["CDC_INT_raw"] == obs["NCBI_SIR"] if row["CDC_INT_raw"] in {"S", "I", "R"} and obs["NCBI_SIR"] in {"S", "I", "R"} else "NOT_COMPARABLE"}
            edition_list = editions.get(row["biosample_acc"], [])
            result["CDC_stated_edition"] = "|".join(edition_list)
            result["retrospective_NCBI_label_MIC_compatibility"] = "NOT_ASSESSED"
            result["CDC_label_MIC_compatibility"] = "NOT_ASSESSED"
            if len(edition_list) == 1 and a and b and units_ok:
                guideline = " ".join(edition_list[0].split()[:2])
                mo = "B_ACNTB" if row["panel_id"] == "1" else "B_PSDMN_AERG" if row["panel_id"] == "12" else "B_[ORD]_ENTRBCTR"
                ab = {"meropenem": "MEM", "imipenem": "IPM", "ertapenem": "ETP"}[row["antibiotic"]]
                selected = bp[(bp.guideline == guideline) & (bp.type == "human") & (bp.host == "human") & (bp.method == "MIC") & (bp.site == "") & (bp.mo == mo) & (bp.ab == ab)]
                if len(selected) == 1:
                    s, r = float(selected.iloc[0].breakpoint_S), float(selected.iloc[0].breakpoint_R)
                    result["retrospective_NCBI_label_MIC_compatibility"] = obs["NCBI_SIR"] in categories(a, s, r) if obs["NCBI_SIR"] in {"S", "I", "R"} else "NOT_SIR"
                    result["CDC_label_MIC_compatibility"] = row["CDC_INT_raw"] in categories(b, s, r) if row["CDC_INT_raw"] in {"S", "I", "R"} else "NOT_SIR"
            rows.append(result)
    data = pd.DataFrame(rows)
    data.to_csv(args.out / "primary_MIC_observation_comparisons.csv", index=False)
    disagree = data[data.same_SIR.eq(False)]
    disagree.to_csv(args.out / "SIR_disagreements_with_MIC_and_version_context.csv", index=False)
    hist = pd.read_csv(args.base / "cdc_isolate_histories/documented_history_events.csv", dtype=str, keep_default_na=False)
    hist_unique = hist.drop_duplicates(["isolate_id", "date_raw", "description"])
    lag_cases = []
    for event in hist_unique.loc[(hist_unique.field == "MIC") & (hist_unique.classification == "VALUE_UPDATE")].to_dict("records"):
        old, new = interval(event["old_value"]), interval(event["new_value"])
        if old is None or new is None or old == new:
            continue
        selected = data.loc[data.biosample_acc.eq(event["biosamples"]) & data.antibiotic.eq(event["antibiotic"])]
        for row in selected.to_dict("records"):
            if interval(row["NCBI_MIC_raw"]) == old and interval(row["CDC_MIC_raw"]) == new:
                lag_cases.append({**row, "CDC_documented_update_date": datetime.strptime(event["date_raw"], "%m/%d/%y").date().isoformat(),
                                  "CDC_logged_old_MIC": event["old_value"], "CDC_logged_new_MIC": event["new_value"],
                                  "history_html_sha256": event["html_sha256"], "history_url": event["url"]})
    pd.DataFrame(lag_cases).to_csv(args.out / "documented_CDC_update_NCBI_retains_old_MIC.csv", index=False)
    report = {"status": "EXPLORATORY_REFERENCE_MEASUREMENT_AUDIT_NOT_CLINICAL_ADJUDICATION", "paired_observation_rows": len(data),
              "paired_unique_BioSample_drug_keys": len(data[["biosample_acc", "antibiotic"]].drop_duplicates()),
              "paired_unique_BioSamples": data.biosample_acc.nunique(), "MIC_comparison_counts": dict(Counter(data.MIC_comparison)),
              "SIR_disagreement_observation_rows": len(disagree),
              "SIR_disagreement_MIC_status": dict(Counter(disagree.MIC_comparison)),
              "SIR_disagreement_NCBI_retrospective_compatibility": {str(k): v for k, v in Counter(disagree.retrospective_NCBI_label_MIC_compatibility).items()},
              "SIR_disagreement_CDC_compatibility": {str(k): v for k, v in Counter(disagree.CDC_label_MIC_compatibility).items()},
              "same_SIR_but_disjoint_MIC_observation_rows": int((data.same_SIR.eq(True) & data.MIC_comparison.eq("DISJOINT_REPORTED_MIC_INTERVALS")).sum()),
              "history_rows_raw": len(hist), "history_rows_after_exact_deduplication": len(hist_unique),
              "distinct_nonmissing_MIC_value_update_logs": int(((hist_unique.field == "MIC") & (hist_unique.classification == "VALUE_UPDATE")).sum()),
              "distinct_nonmissing_INT_value_update_logs": int(((hist_unique.field == "INT") & (hist_unique.classification == "VALUE_UPDATE")).sum()),
              "documented_MIC_update_with_NCBI_old_value_cases": len(lag_cases),
              "documented_MIC_update_with_NCBI_old_value_BioSamples": len({r["biosample_acc"] for r in lag_cases}),
              "breakpoint_source_sha256": hashlib.sha256(args.breakpoints.read_bytes()).hexdigest(),
              "breakpoint_source_type": "Secondary curated AMR package pinned dataset; not an official CLSI primary document",
              "limitations": ["Different MIC intervals can arise from different assays, retesting, rounding, curation or error; cause is not assumed.",
                              "NCBI source assay standard edition remains unknown; compatibility uses the CDC-stated 2024/2025 edition retrospectively.",
                              "A reported modal BMD MIC remains subject to technical variability.",
                              "Current BioSample data and September target-level AST exports need separate temporal checks.",
                              "No new model predictions, clinical error rates or independent patient counts are inferred."]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
