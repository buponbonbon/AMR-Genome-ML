#!/usr/bin/env python3
"""Offline census of candidate AMR source metadata; never adjudicates AST or trains models."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

SPECIES = ("Escherichia coli", "Pseudomonas aeruginosa", "Acinetobacter baumannii")
DRUGS = ("meropenem", "imipenem")
FIELDS = ("measurement_value", "measurement_sign", "measurement_unit",
          "laboratory_typing_method", "laboratory_typing_method_version",
          "laboratory_typing_platform", "vendor", "testing_standard", "testing_standard_year")
LABELS = {"resistant": "R", "r": "R", "susceptible": "S", "s": "S",
          "intermediate": "I", "i": "I"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def populated(value):
    return value.strip().lower() not in {"", "-", "na", "n/a", "nan", "none", "null", "missing"}


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Invalid TSV header")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("TSV row width differs from header")
            yield {k: v.strip() for k, v in row.items()}


def organism(name):
    for species in SPECIES:
        if name == species or name.startswith(species + " "):
            return species
    return None


def write_csv(path, records, fields):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def screen(raw_path, mapping_path):
    selected, seen = [], set()
    all_drugs = Counter()
    raw_total = selected_before_dedup = 0
    for row in rows(raw_path):
        raw_total += 1
        species = organism(row["genome_name"])
        if species is None:
            continue
        drug = row["antibiotic"].lower()
        all_drugs[(species, drug)] += 1
        if drug not in DRUGS:
            continue
        selected_before_dedup += 1
        key = tuple(row.items())
        if key in seen:
            continue
        seen.add(key)
        selected.append({**row, "screen_species_annotation": species,
                         "screen_drug": drug})

    maps = defaultdict(list)
    for row in rows(mapping_path):
        key = row["genome_id"]
        if row not in maps[key]:
            maps[key].append(row)

    summaries, details, links = [], {}, []
    for species in SPECIES:
        for drug in DRUGS:
            records = [r for r in selected if r["screen_species_annotation"] == species
                       and r["screen_drug"] == drug]
            genomes = defaultdict(list)
            for record in records:
                genomes[record["genome_id"]].append(record)
            strict = Counter()
            rs_conflicts = multicategory = 0
            pubs, projects = Counter(), Counter()
            mapping_ambiguous = mapping_missing = mapped = bio = assembly = 0
            for genome_id, observations in genomes.items():
                labels = {LABELS.get(r["resistant_phenotype"].lower(),
                                    "OTHER:" + r["resistant_phenotype"])
                          for r in observations if populated(r["resistant_phenotype"])}
                if len(labels) == 1:
                    strict[next(iter(labels))] += 1
                elif len(labels) > 1:
                    multicategory += 1
                if {"R", "S"} <= labels:
                    rs_conflicts += 1
                candidates = maps.get(genome_id, [])
                link = {"species_annotation": species, "antibiotic": drug,
                        "genome_id": genome_id, "raw_record_count": len(observations),
                        "phenotype_categories": "|".join(sorted(labels))}
                if len(candidates) == 1:
                    m = candidates[0]
                    mapped += 1
                    bio += populated(m.get("biosample_accession", ""))
                    assembly += populated(m.get("assembly_accession", ""))
                    if populated(m.get("publication", "")):
                        pubs[m["publication"]] += 1
                    for project in set(re.findall(r"PRJ(?:NA|EB|DB)\d+", m.get("bioproject_accession", ""))):
                        projects[project] += 1
                    link.update(m)
                    link["mapping_status"] = "UNIQUE_SOURCE_MAPPING"
                elif candidates:
                    mapping_ambiguous += 1
                    link["mapping_status"] = "AMBIGUOUS_NOT_RESOLVED"
                else:
                    mapping_missing += 1
                    link["mapping_status"] = "MISSING"
                links.append(link)
            complete = {field: sum(populated(r.get(field, "")) for r in records) for field in FIELDS}
            valid_year = lambda r: bool(re.fullmatch(r"(?:19|20)\d{2}", r.get("testing_standard_year", "")))
            mic = lambda r: r.get("laboratory_typing_method", "").upper() == "MIC"
            mic_family = lambda r: r.get("laboratory_typing_method", "").lower().replace("_", " ") in {
                "mic", "broth microdilution", "agar dilution", "etest"}
            mic_value_unit_year = sum(mic(r) and all(populated(r.get(f, "")) for f in
                ("measurement_value", "measurement_sign", "measurement_unit", "testing_standard"))
                and valid_year(r) for r in records)
            # A blank sign is not silently converted to equality. Report field
            # availability separately so literal numeric measurement strings and
            # BMD/agar-dilution methods are not falsely described as absent MIC data.
            mic_family_value_unit_year = sum(mic_family(r) and all(populated(r.get(f, "")) for f in
                ("measurement_value", "measurement_unit", "testing_standard"))
                and valid_year(r) for r in records)
            mic_family_value_unit_year_genomes = len({r["genome_id"] for r in records if
                mic_family(r) and all(populated(r.get(f, "")) for f in
                ("measurement_value", "measurement_unit", "testing_standard")) and valid_year(r)})
            summary = {"species_annotation": species, "antibiotic": drug,
                       "unique_source_records": len(records), "unique_genome_ids": len(genomes),
                       "single_category_R_genomes": strict["R"],
                       "single_category_S_genomes": strict["S"],
                       "single_category_I_genomes": strict["I"],
                       "genomes_R_S_conflict": rs_conflicts,
                       "genomes_multiple_nonmissing_categories": multicategory,
                       "genomes_no_phenotype": sum(not any(populated(r["resistant_phenotype"]) for r in rs) for rs in genomes.values()),
                       "unique_mapping_genomes": mapped, "mapping_missing_genomes": mapping_missing,
                       "mapping_ambiguous_genomes": mapping_ambiguous,
                       "genomes_with_biosample_field": bio, "genomes_with_assembly_field": assembly,
                       "bioproject_ids_in_mapping": len(projects),
                       "publication_field_groups": len(pubs),
                       "largest_publication_field_group_genomes": max(pubs.values(), default=0),
                       "records_MIC_value_sign_unit_standard_validyear": mic_value_unit_year,
                       "records_MICfamily_value_unit_standard_validyear": mic_family_value_unit_year,
                       "genomes_MICfamily_value_unit_standard_validyear": mic_family_value_unit_year_genomes,
                       "records_nonempty_standard_year": complete["testing_standard_year"],
                       "records_valid_four_digit_standard_year": sum(valid_year(r) for r in records),
                       "screening_status": "CURRENT_COUNTERPART_AND_AST_CONTEXT_NOT_VERIFIED"}
            summaries.append(summary)
            details[species + " / " + drug] = {
                "summary": summary, "raw_phenotype_strings": dict(Counter(r["resistant_phenotype"] for r in records)),
                "single_nonmissing_genome_categories": dict(strict), "nonempty_fields": complete,
                "method_strings": dict(Counter(r["laboratory_typing_method"] for r in records)),
                "standard_strings": dict(Counter(r["testing_standard"] for r in records)),
                "standard_year_strings": dict(Counter(r["testing_standard_year"] for r in records)),
                "measurement_sign_strings": dict(Counter(r["measurement_sign"] for r in records)),
                "unit_strings": dict(Counter(r["measurement_unit"] for r in records)),
                "publication_field_groups": dict(pubs), "bioproject_groups": dict(projects)}
    report = {"status": "SCREENING_ONLY_NOT_AST_ADJUDICATION", "raw_rows": raw_total,
              "selected_rows_before_dedup": selected_before_dedup,
              "selected_unique_full_records": len(selected),
              "selected_exact_duplicates": selected_before_dedup - len(selected),
              "candidate_pairs": details,
              "limitations": ["Species assigned by source genome_name prefix, not sequence adjudication.",
                 "Counts are observed source records, not independent laboratory tests or patients.",
                 "R/S conflicts can reflect repeat measurements; they are not established data errors.",
                 "Non-susceptible, RS and other source labels are retained, not mapped to resistant.",
                 "Publication fields and BioProjects do not prove laboratory independence.",
                 "Completeness does not verify metadata correctness or breakpoint applicability.",
                 "No current-reference comparison, model training, AST adjudication or MIC interpretation performed."]}
    drug_catalog = [{"species_annotation": s, "antibiotic_raw_lower": d, "raw_rows": n}
                    for (s, d), n in sorted(all_drugs.items())]
    return report, summaries, selected, links, drug_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    inputs = {"raw": args.raw.resolve(), "mapping": args.mapping.resolve()}
    before = {k: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for k, p in inputs.items()}
    if args.out.resolve() in {args.raw.resolve(), args.mapping.resolve()}:
        raise ValueError("Output conflicts with an input")
    args.out.mkdir(parents=True, exist_ok=False)
    report, summaries, selected, links, catalog = screen(args.raw, args.mapping)
    for key, path in inputs.items():
        if sha256(path) != before[key]["sha256"]:
            raise ValueError("Input changed during census")
    report.update({"inputs": before, "completed_utc": datetime.now(timezone.utc).isoformat()})
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out / "candidate_pair_counts.csv", summaries, list(summaries[0]))
    if selected:
        write_csv(args.out / "candidate_raw_records.csv", selected, list(selected[0]))
    if links:
        fields = list(dict.fromkeys(k for row in links for k in row))
        write_csv(args.out / "candidate_genome_links.csv", links, fields)
    write_csv(args.out / "all_antibiotic_counts.csv", catalog, list(catalog[0]))
    print(json.dumps({"out": str(args.out), "pairs": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
