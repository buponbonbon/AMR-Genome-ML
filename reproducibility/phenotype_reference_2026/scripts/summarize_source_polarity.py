#!/usr/bin/env python3
"""Summarize a fixed source-specific label mapping from paired AST artifacts."""
import argparse
from collections import Counter
import csv
import itertools
import json
from pathlib import Path
import re


FOCAL = "PRJNA1322038"
CHANGED = {"UNOPPOSED_R_S_REVERSAL_CANDIDATE", "OTHER_SIR_SET_CHANGE"}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields=None):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--known-event-crosswalk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for directory in args.comparison_dirs for r in read(directory / "all_shared_AST_comparisons.csv")]
    keys = [(r["target_acc"], r["biosample_acc"], r["antibiotic"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate shared key across analysis panels")
    focal = [r for r in rows if r["old_bioproject"] == FOCAL]
    if any(r["new_bioproject"] != FOCAL for r in focal):
        raise ValueError("Source membership changed; revise the grouping protocol")
    outside = [r for r in rows if r["old_bioproject"] != FOCAL]
    differences = [r for r in rows if r["comparison"] in CHANGED]
    species_rows, permutations = [], []
    for species in sorted({r["species_source_annotation"] for r in rows}):
        cases = [r for r in focal if r["species_source_annotation"] == species]
        controls = [r for r in outside if r["species_source_annotation"] == species]
        changed = [r for r in cases if r["comparison"] in CHANGED]
        species_rows.append({"species_source_annotation": species,
                             "focal_shared_raw_AST_keys": len(cases),
                             "focal_shared_SIR_keys": sum(r["comparison"] != "NON_SIR_OR_MISSING_INVOLVED" for r in cases),
                             "R_S_reversal_keys": sum(r["comparison"] == "UNOPPOSED_R_S_REVERSAL_CANDIDATE" for r in cases),
                             "S_to_R": sum(r["old_categories"] == "S" and r["new_categories"] == "R" for r in cases),
                             "R_to_S": sum(r["old_categories"] == "R" and r["new_categories"] == "S" for r in cases),
                             "stable_I_keys": sum(r["old_categories"] == r["new_categories"] == "I" for r in cases),
                             "changed_unique_BioSamples": len({r["biosample_acc"] for r in changed}),
                             "changed_drug_names": len({r["antibiotic"] for r in changed}),
                             "outside_shared_SIR_keys": sum(r["comparison"] != "NON_SIR_OR_MISSING_INVOLVED" for r in controls),
                             "outside_SIR_changes": sum(r["comparison"] in CHANGED for r in controls)})
        for name, subset in [("focal", cases), ("outside", controls)]:
            for perm in itertools.permutations("SIR"):
                mapping = dict(zip("SIR", perm))
                fits = sum({mapping.get(x, x) for x in r["old_categories"].split("|")} == set(r["new_categories"].split("|")) for r in subset)
                permutations.append({"species": species, "source_group": name,
                                     "mapping": ";".join(k + "->" + mapping[k] for k in "SIR"),
                                     "matched_raw_keys": len(subset), "mapping_fits_n": fits,
                                     "mapping_mismatches_n": len(subset) - fits})
    indexed = dict(zip(keys, rows))
    reference_labels = {"resistant": "R", "susceptible": "S", "intermediate": "I"}
    anchor = []
    for record in read(args.known_event_crosswalk):
        for phase, target_field, source_field, category_field in [
                ("old", "historical_target_acc", "historical_file11_raw_label", "old_categories"),
                ("new", "current_target_acc", "current_bigquery_label", "new_categories")]:
            key = record[target_field], record["biosample_acc"], record["antibiotic"]
            native = indexed.get(key)
            value = native[category_field] if native else ""
            expected = reference_labels[record[source_field]]
            anchor.append({"phase": phase, "target_acc": key[0], "biosample_acc": key[1], "antibiotic": key[2],
                           "export_label": expected, "native_release_label": value,
                           "comparison": "MISSING" if not native else "AGREE" if value == expected else "DISAGREE"})
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "species_source_summary.csv", species_rows)
    write(args.out / "source_polarity_permutation_fits.csv", permutations)
    write(args.out / "all_R_S_reversal_records.csv", [r for r in differences if r["comparison"] == "UNOPPOSED_R_S_REVERSAL_CANDIDATE"], list(rows[0]))
    write(args.out / "known_event_native_archive_validation.csv", anchor)
    counts = Counter(r["old_categories"] + "->" + r["new_categories"] for r in focal)
    comparable_outside = [r for r in outside if r["comparison"] != "NON_SIR_OR_MISSING_INVOLVED"]
    report = {"status": "DESCRIPTIVE_SOURCE_EVENT_SUMMARY_NOT_CLINICAL_ADJUDICATION",
              "all_shared_raw_AST_keys": len(rows), "all_shared_SIR_keys": sum(r["comparison"] != "NON_SIR_OR_MISSING_INVOLVED" for r in rows),
              "focal_raw_AST_keys": len(focal), "focal_category_transitions": dict(counts),
              "R_S_reversal_keys": sum(r["comparison"] == "UNOPPOSED_R_S_REVERSAL_CANDIDATE" for r in differences),
              "changed_unique_BioSamples": len({r["biosample_acc"] for r in differences}),
              "changed_drug_names": sorted({r["antibiotic"] for r in differences}),
              "changed_project_ids": sorted({r["old_bioproject"] for r in differences}),
              "outside_shared_SIR_keys": len(comparable_outside),
              "outside_changed_SIR_keys": sum(r["comparison"] in CHANGED for r in outside),
              "outside_unique_BioSamples_with_shared_SIR": len({r["biosample_acc"] for r in comparable_outside}),
              "outside_BioProject_ids_with_shared_SIR": len(set().union(*(set(re.findall(r"PRJ(?:NA|EB|DB)\d+", r["old_bioproject"] + " " + r["new_bioproject"])) for r in comparable_outside))),
              "known_event_anchor_counts": dict(Counter(r["phase"] + "/" + r["comparison"] for r in anchor)),
              "species": species_rows,
              "limitations": ["One observed source event, not four independent events or 11087 independent patients.",
                              "Full-antibiogram expansion is exploratory; meropenem/imipenem were the primary screening drugs.",
                              "Source taxonomy annotations are not genomic species adjudication.",
                              "AST correctness and root cause are unresolved; do not equate the newest category with truth."]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"species", "limitations", "changed_drug_names"}}, indent=2))


if __name__ == "__main__":
    main()
