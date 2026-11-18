#!/usr/bin/env python3
"""Independently reconstruct a retained metadata screen using pandas explode/join."""
import argparse
import json
from pathlib import Path

import pandas as pd

from verify_ncbi_retained_ast_independent import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--archive-dir", type=Path, required=True)
    p.add_argument("--comparison-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    production = pd.read_csv(args.comparison_dir / "all_shared_AST_comparisons.csv", dtype=str, keep_default_na=False)
    keys = ["target_acc", "biosample_acc", "antibiotic"]
    sir = frozenset(["S", "I", "R"])
    records, all_samples, sir_samples, projects = [], set(), set(), set()
    for case in protocol["targets"]:
        species = case["species"]
        a = load(args.archive_dir / (case["old_build"] + ".metadata.tsv"), species)
        b = load(args.archive_dir / (case["new_build"] + ".metadata.tsv"), species)
        joined = a.join(b, how="inner", lsuffix="_old", rsuffix="_new")
        # Match the explicit identifier eligibility in production.
        mask = pd.Series(True, index=joined.index)
        for name in ["target_acc", "biosample_acc"]:
            mask &= ~joined.index.get_level_values(name).str.strip().str.upper().isin(["", "NULL", "NA", "N/A", "-"])
        joined = joined.loc[mask]
        old, new = joined.category_old.map(lambda s: s & sir), joined.category_new.map(lambda s: s & sir)
        valid = old.map(bool) & new.map(bool)
        changed = valid & (old != new)
        outside = ~joined.project_old.str.contains("PRJNA1322038", regex=False) & ~joined.project_new.str.contains("PRJNA1322038", regex=False)
        expected = production.loc[production.species_source_annotation.eq(species)].set_index(keys)
        if not joined.index.is_unique or not expected.index.is_unique or set(joined.index) != set(expected.index):
            raise ValueError("Independent primary-key join differs from production")
        expected = expected.loc[joined.index]
        for side in ["old", "new"]:
            observed = joined["category_" + side].map(lambda s: "|".join(sorted(s)))
            if not observed.eq(expected[side + "_categories"]).all():
                raise ValueError("Independent category reconstruction differs from production")
            if not joined["project_" + side].eq(expected[side + "_bioproject"]).all():
                raise ValueError("Independent source project differs from production")
        expected_changes = expected.comparison.isin(["UNOPPOSED_R_S_REVERSAL_CANDIDATE", "OTHER_SIR_SET_CHANGE"])
        if not changed.eq(expected_changes).all():
            raise ValueError("Independent change classification differs from production")
        bios = set(joined.index.get_level_values("biosample_acc"))
        valid_bios = set(joined.loc[valid].index.get_level_values("biosample_acc"))
        all_samples.update(bios)
        sir_samples.update(valid_bios)
        for value in pd.concat([joined.project_old, joined.project_new]):
            projects.update(x.strip() for x in value.split(",") if x.strip())
        record = {"species": species, "shared_raw_AST_keys": len(joined), "shared_SIR_keys": int(valid.sum()),
                  "SIR_changed_keys": int(changed.sum()), "outside_known_source_SIR_keys": int((valid & outside).sum()),
                  "outside_known_source_SIR_changes": int((changed & outside).sum()),
                  "unique_BioSamples_shared_AST": len(bios), "unique_BioSamples_shared_SIR": len(valid_bios)}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {"status": "PASS", "independent_method": "Pandas literal-TSV chunk/explode/groupby/index join versus all production key/category/project records",
              "results": records, "total_shared_raw_AST_keys": sum(r["shared_raw_AST_keys"] for r in records),
              "total_shared_SIR_keys": sum(r["shared_SIR_keys"] for r in records),
              "total_SIR_changes": sum(r["SIR_changed_keys"] for r in records),
              "unique_BioSamples_shared_AST": len(all_samples), "unique_BioSamples_shared_SIR": len(sir_samples),
              "BioProject_strings": sorted(projects),
              "limitations": ["Paired stable target/BioSample identifiers only; unpaired or accession-version-changed records are outside this endpoint.",
                              "The source event is not replicated by counting related species, drugs or BioSamples as independent events."]}
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
