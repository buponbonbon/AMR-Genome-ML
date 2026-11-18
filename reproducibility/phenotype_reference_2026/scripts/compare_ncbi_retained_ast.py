#!/usr/bin/env python3
"""Compare AST phenotype sets in retained NCBI build metadata, never clinical truth."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


MISSING = {"", "NULL", "NA", "N/A", "-"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def parse_ast(value):
    # NCBI encloses list-valued fields in a single pair of double quotes.
    # Strip that wrapper after splitting the literal TSV columns.
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if value.strip().upper() in MISSING:
        return {}
    result = defaultdict(set)
    for token in value.split(","):
        if token.count("=") != 1:
            raise ValueError(f"Unsupported AST token: {token!r}")
        drug, category = (x.strip() for x in token.split("="))
        if not drug or not category:
            raise ValueError("Empty drug/category in nonmissing AST field")
        result[drug.lower()].add(category.upper())
    return {k: frozenset(v) for k, v in result.items()}


def load(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("Retained metadata hash mismatch")
    records, counts, categories = {}, Counter(), Counter()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        # Split literal tabs: an unmatched quote in unrelated free text must
        # not consume the remainder of the row or subsequent records.
        # List-valued AST quote wrappers are handled in parse_ast().
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        required = {"target_acc", "biosample_acc", "scientific_name", "AST_phenotypes", "bioproject_acc"}
        if not required <= set(reader.fieldnames):
            raise ValueError("Required metadata schema absent")
        for row in reader:
            counts["all_genome_rows"] += 1
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Metadata row width differs from header")
            ast = parse_ast(row["AST_phenotypes"])
            if not ast:
                continue
            counts["rows_with_AST_tokens"] += 1
            key = row["target_acc"], row["biosample_acc"]
            if any(x.strip().upper() in MISSING for x in key):
                counts["AST_rows_missing_primary_identifiers"] += 1
                continue
            record = {"ast": ast, "scientific_name": row["scientific_name"],
                      "bioproject_acc": row["bioproject_acc"], "asm_acc": row["asm_acc"]}
            if key in records:
                if records[key] != record:
                    raise ValueError("Conflicting duplicate genome/BioSample metadata")
                counts["exact_projected_duplicates"] += 1
            records[key] = record
            for values in ast.values():
                categories.update(values)
    return records, {"counts": dict(counts), "category_strings": dict(categories)}


def selected(name, species):
    return name == species or name.startswith(species + " ")


def comparison(a, b):
    aa, bb = set(a) & {"S", "I", "R"}, set(b) & {"S", "I", "R"}
    if not aa or not bb:
        return "NON_SIR_OR_MISSING_INVOLVED"
    if aa == bb:
        return "SAME_SIR_SET"
    if (aa, bb) in [({"R"}, {"S"}), ({"S"}, {"R"})]:
        return "UNOPPOSED_R_S_REVERSAL_CANDIDATE"
    return "OTHER_SIR_SET_CHANGE"


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(protocol, archive_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    if not manifest["all_complete"]:
        raise ValueError("Archive retrieval incomplete")
    entries = {r["build"]: r for r in manifest["responses"]}
    results, differences, inventories, summary, coverage = [], [], {}, [], []
    for target in protocol["targets"]:
        species = target["species"]
        old, old_stats = load(archive_dir / entries[target["old_build"]]["file"], entries[target["old_build"]]["sha256"])
        new, new_stats = load(archive_dir / entries[target["new_build"]]["file"], entries[target["new_build"]]["sha256"])
        inventories[species] = {"old": old_stats, "new": new_stats}
        counts, bios, projects = defaultdict(Counter), defaultdict(set), defaultdict(set)
        excluded_taxonomy = 0
        for key in sorted(set(old) & set(new)):
            a, b = old[key], new[key]
            if not (selected(a["scientific_name"], species) and selected(b["scientific_name"], species)):
                excluded_taxonomy += 1
                continue
            for drug in sorted(set(a["ast"]) | set(b["ast"])):
                av, bv = a["ast"].get(drug, frozenset()), b["ast"].get(drug, frozenset())
                if not av:
                    counts[drug]["drug_only_in_new_AST"] += 1
                    continue
                if not bv:
                    counts[drug]["drug_only_in_old_AST"] += 1
                    continue
                status = comparison(av, bv)
                counts[drug][status] += 1
                counts[drug]["shared_raw_AST_keys"] += 1
                bios[drug].add(key[1])
                projects[drug].update(re.findall(r"PRJ(?:NA|EB|DB)\d+", a["bioproject_acc"] + " " + b["bioproject_acc"]))
                row = {"species_source_annotation": species, "target_acc": key[0], "biosample_acc": key[1],
                       "antibiotic": drug, "old_categories": "|".join(sorted(av)), "new_categories": "|".join(sorted(bv)),
                       "old_bioproject": a["bioproject_acc"], "new_bioproject": b["bioproject_acc"],
                       "comparison": status, "raw_category_set_changed": av != bv}
                results.append(row)
                if status not in {"SAME_SIR_SET", "NON_SIR_OR_MISSING_INVOLVED"}:
                    differences.append(row)
        for drug, c in sorted(counts.items()):
            summary.append({"species_source_annotation": species, "antibiotic": drug,
                            "is_primary_drug": drug in protocol["primary_drugs"],
                            "shared_raw_AST_keys": c["shared_raw_AST_keys"],
                            "same_SIR_set": c["SAME_SIR_SET"],
                            "unopposed_R_S_reversal_candidates": c["UNOPPOSED_R_S_REVERSAL_CANDIDATE"],
                            "other_SIR_set_changes": c["OTHER_SIR_SET_CHANGE"],
                            "non_SIR_or_missing_involved": c["NON_SIR_OR_MISSING_INVOLVED"],
                            "drug_only_in_old_AST": c["drug_only_in_old_AST"],
                            "drug_only_in_new_AST": c["drug_only_in_new_AST"],
                            "unique_BioSamples_shared_keys": len(bios[drug]),
                            "BioProject_ids_shared_keys": len(projects[drug])})
        coverage.append({"species": species, "old_build": target["old_build"], "new_build": target["new_build"],
                         "AST_genome_keys_old": len(old), "AST_genome_keys_new": len(new),
                         "shared_AST_genome_keys_before_taxonomy_filter": len(set(old) & set(new)),
                         "shared_AST_genome_keys_excluded_by_source_taxonomy": excluded_taxonomy,
                         "AST_genome_keys_old_only": len(set(old) - set(new)),
                         "AST_genome_keys_new_only": len(set(new) - set(old))})
        print(json.dumps({"species": species, "shared_AST_genome_keys": len(set(old) & set(new)),
                          "SIR_changed_keys": sum(r["species_source_annotation"] == species for r in differences)}), flush=True)
    fields = ["species_source_annotation", "target_acc", "biosample_acc", "antibiotic", "old_categories", "new_categories", "old_bioproject", "new_bioproject", "comparison", "raw_category_set_changed"]
    write_csv(out / "all_shared_AST_comparisons.csv", results, fields)
    write_csv(out / "SIR_change_candidates.csv", differences, fields)
    write_csv(out / "per_drug_summary.csv", summary, list(summary[0]))
    write_csv(out / "primary_drug_summary.csv", [r for r in summary if r["is_primary_drug"]], list(summary[0]))
    report = {"status": "RETAINED_RELEASE_PHENOTYPE_SET_COMPARISON_NOT_AST_ADJUDICATION",
              "primary_key": ["target_acc", "biosample_acc", "antibiotic"],
              "inventory": inventories, "coverage": coverage, "per_drug": summary,
              "SIR_change_candidate_keys": len(differences),
              "limitations": ["AST entries lack observation-level native IDs in this export.",
                              "Different dates/release paths do not prove metadata immutability.",
                              "Labels ND and unknown categories are never mapped to S/R.",
                              "Different drugs/targets from the same BioSample are not independent patients.",
                              "No changes or correctness are inferred for records without paired S/I/R."]}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.archive_dir, args.out)


if __name__ == "__main__":
    main()
