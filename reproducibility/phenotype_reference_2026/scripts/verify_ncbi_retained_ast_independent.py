#!/usr/bin/env python3
"""Independent pandas verification of the retained-release CSV implementation."""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def load(path, species):
    frames = []
    for chunk in pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, usecols=["target_acc", "biosample_acc", "scientific_name", "AST_phenotypes", "bioproject_acc"],
                             dtype=str, keep_default_na=False, chunksize=100000):
        keep = (chunk.scientific_name.eq(species) | chunk.scientific_name.str.startswith(species + " ")) & ~chunk.AST_phenotypes.str.upper().isin(["", "NULL", "NA", "N/A", "-"])
        frames.append(chunk.loc[keep])
    data = pd.concat(frames, ignore_index=True)
    data["AST_phenotypes"] = data.AST_phenotypes.map(lambda x: x[1:-1] if x.startswith('"') and x.endswith('"') else x)
    exploded = data.assign(token=data.AST_phenotypes.str.split(",")).explode("token")
    pieces = exploded.token.str.rsplit("=", n=1, expand=True)
    if pieces.shape[1] != 2:
        raise ValueError("Unexpected token schema")
    exploded["antibiotic"] = pieces[0].str.strip().str.lower()
    exploded["category"] = pieces[1].str.strip().str.upper()
    keys = ["target_acc", "biosample_acc", "antibiotic"]
    exploded = exploded.drop_duplicates(keys + ["category"])
    return exploded.groupby(keys).agg(category=("category", lambda s: frozenset(s)), project=("bioproject_acc", "first"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for protocol_name, folder in [("protocol.json", "retained_metadata"), ("additional_archive_protocol.json", "retained_metadata_additional")]:
        protocol = json.loads((args.base / protocol_name).read_text())
        for case in protocol["targets"]:
            species = case["species"]
            a = load(args.base / folder / (case["old_build"] + ".metadata.tsv"), species)
            b = load(args.base / folder / (case["new_build"] + ".metadata.tsv"), species)
            joined = a.join(b, how="inner", lsuffix="_old", rsuffix="_new")
            allowed = frozenset(["S", "I", "R"])
            old, new = joined.category_old.map(lambda x: x & allowed), joined.category_new.map(lambda x: x & allowed)
            valid = old.map(bool) & new.map(bool)
            changed = valid & (old != new)
            reversed_rs = changed & (((old == frozenset(["S"])) & (new == frozenset(["R"]))) | ((old == frozenset(["R"])) & (new == frozenset(["S"]))))
            outside = valid & joined.project_old.ne("PRJNA1322038")
            record = {"species": species, "shared_raw_AST_keys": len(joined), "SIR_comparable_keys": int(valid.sum()),
                      "SIR_changed_keys": int(changed.sum()), "unopposed_R_S_flips": int(reversed_rs.sum()),
                      "outside_focal_SIR_keys": int(outside.sum()), "outside_focal_changes": int((outside & changed).sum())}
            results.append(record)
            print(json.dumps(record), flush=True)
    expected = json.loads((args.base / "source_event_summary/report.json").read_text())
    for observed in results:
        original = next(r for r in expected["species"] if r["species_source_annotation"] == observed["species"])
        if observed["unopposed_R_S_flips"] != original["R_S_reversal_keys"] or observed["outside_focal_SIR_keys"] != original["outside_shared_SIR_keys"] or observed["outside_focal_changes"] != original["outside_SIR_changes"]:
            raise ValueError("Independent verification disagrees with production summary")
    if sum(r["shared_raw_AST_keys"] for r in results) != expected["all_shared_raw_AST_keys"] or sum(r["SIR_comparable_keys"] for r in results) != expected["all_shared_SIR_keys"]:
        raise ValueError("Independent denominator mismatch")
    with args.out.open("x") as handle:
        json.dump({"status": "PASS", "method": "Independent pandas chunk/explode/index join", "results": results}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
