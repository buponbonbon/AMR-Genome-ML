#!/usr/bin/env python3
"""Inventory CDC BMD reference exports and exact BioSample links, not assay adjudication."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import pandas as pd


def write_csv(path, rows, fields):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdc-dir", type=Path, required=True)
    parser.add_argument("--snapshot-protocol", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.cdc_dir / "manifest.json").read_text())
    protocol = json.loads(args.snapshot_protocol.read_text())
    primary = ["Meropenem", "Imipenem", "Ertapenem"]
    long, inventory = [], []
    for case in manifest["results"]:
        data = pd.read_excel(args.cdc_dir / case["export"]["file"], dtype=str, keep_default_na=False)
        required = {"ARBankNum", "OrganismName", "BioSampleAccessionNum"}
        if not required <= set(data.columns):
            raise ValueError("CDC export schema differs from inspected schema")
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
        inventory.append({"panel_id": case["panel"]["id"], "panel_name": case["panel"]["name"],
                          "export_rows": len(data), "unique_BioSamples": data.BioSampleAccessionNum.nunique(),
                          "BMD_stated_on_page": case["BMD_stated_on_panel_page"],
                          "organism_names": dict(Counter(data.OrganismName))})
        for row in data.to_dict("records"):
            if not re.fullmatch(r"SAM[NED][A-Z]?\d+", row["BioSampleAccessionNum"]):
                raise ValueError("CDC row lacks one exact BioSample accession")
            for drug in primary:
                mic = row.get(drug + " MIC", "")
                category = row.get(drug + " INT", "")
                if not mic and not category:
                    continue
                long.append({"panel_id": case["panel"]["id"], "ARBankNum": row["ARBankNum"],
                             "biosample_acc": row["BioSampleAccessionNum"], "CDC_organism": row["OrganismName"],
                             "antibiotic": drug.lower(), "CDC_MIC_raw": mic, "CDC_INT_raw": category,
                             "export_sha256": case["export"]["sha256"]})
    samples = {r["biosample_acc"] for r in long}
    linked = []
    seen = set()
    for case in protocol["targets"]:
        path = args.archive_dir / (case["new_build"] + ".metadata.tsv")
        for chunk in pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
                                 chunksize=100000, usecols=["target_acc", "biosample_acc", "scientific_name", "AST_phenotypes", "bioproject_acc", "asm_acc"]):
            for row in chunk.loc[chunk.biosample_acc.isin(samples)].to_dict("records"):
                key = row["target_acc"], row["biosample_acc"]
                if key in seen:
                    raise ValueError("Multiple metadata rows for one target/BioSample")
                seen.add(key)
                row["metadata_build"] = case["new_build"]
                linked.append(row)
    paired = []
    for cdc in long:
        matches = [r for r in linked if r["biosample_acc"] == cdc["biosample_acc"]]
        for ncbi in matches:
            categories = set()
            ast = ncbi["AST_phenotypes"]
            if ast.startswith('"') and ast.endswith('"'):
                ast = ast[1:-1]
            for token in ast.split(","):
                if "=" in token:
                    drug, category = token.rsplit("=", 1)
                    if drug.strip().lower() == cdc["antibiotic"]:
                        categories.add(category.strip().upper())
            c = cdc["CDC_INT_raw"]
            if c not in {"S", "I", "R"} or not categories & {"S", "I", "R"}:
                status = "NOT_SIR_COMPARABLE"
            elif categories & {"S", "I", "R"} == {c}:
                status = "SAME_SINGLE_SIR_SET"
            elif c in categories:
                status = "CDC_INT_INCLUDED_IN_MULTICATEGORY_SET"
            else:
                status = "CATEGORY_DISAGREEMENT_NOT_ADJUDICATED_ERROR"
            paired.append({**cdc, "target_acc": ncbi["target_acc"], "asm_acc": ncbi["asm_acc"],
                           "NCBI_scientific_name": ncbi["scientific_name"], "metadata_build": ncbi["metadata_build"],
                           "NCBI_bioproject": ncbi["bioproject_acc"], "NCBI_category_set": "|".join(sorted(categories)),
                           "comparison": status})
    write_csv(args.out / "CDC_primary_MIC_rows.csv", long, list(long[0]))
    write_csv(args.out / "CDC_linked_NCBI_metadata.csv", linked, list(linked[0]) if linked else ["biosample_acc"])
    write_csv(args.out / "CDC_NCBI_category_comparison_NOT_error_adjudication.csv", paired,
              list(paired[0]) if paired else ["biosample_acc"])
    unique_pairs = {(r["biosample_acc"], r["antibiotic"]) for r in long}
    report = {"status": "PRIMARY_MIC_REFERENCE_FEASIBILITY_NOT_MODEL_VALIDATION", "panel_inventory": inventory,
              "total_panel_rows": sum(r["export_rows"] for r in inventory),
              "unique_BioSamples_with_primary_MIC": len(samples), "primary_MIC_rows_including_panel_repeats": len(long),
              "unique_BioSample_primary_drug_pairs": len(unique_pairs),
              "linked_unique_BioSamples": len({r["biosample_acc"] for r in linked}), "linked_metadata_target_rows": len(linked),
              "linked_category_counts_including_panel_repeats": dict(Counter(r["comparison"] for r in paired)),
              "known_focal_project_overlap_BioSamples": sorted({r["biosample_acc"] for r in linked if "PRJNA1322038" in r["bioproject_acc"]}),
              "primary_MIC_raw_distributions": {drug.lower(): dict(Counter(r["CDC_MIC_raw"] for r in long if r["antibiotic"] == drug.lower())) for drug in primary},
              "limitations": ["Panels were selected for assay and mechanism diversity, not clinical representativeness.",
                              "An isolate can occur in several panels; unique BioSample/drug counts are reported separately.",
                              "Exact BioSample matching does not establish that two entries are the same assay observation.",
                              "NCBI metadata build is September 2026, CDC export was retrieved November 2026.",
                              "No historical MIC, actual breakpoint edition or new genomic predictions are inferred."]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"primary_MIC_raw_distributions", "panel_inventory"}}, indent=2))


if __name__ == "__main__":
    main()
