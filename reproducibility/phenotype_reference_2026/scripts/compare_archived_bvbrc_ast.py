#!/usr/bin/env python3
"""Compare archived/current laboratory phenotype sets, without adjudicating AST.

Revalidates cached HTTP response hashes and native record identities before use.
The comparison key is genome ID plus antibiotic, not an individual AST observation.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


MISSING = {"", "-", "na", "n/a", "nan", "none", "null", "missing"}
LABELS = {"r": "R", "resistant": "R", "s": "S", "susceptible": "S",
          "i": "I", "intermediate": "I"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def phenotype(value):
    value = (value or "").strip().lower()
    if value in MISSING:
        return None
    return LABELS.get(value, "OTHER:" + value)


def label_set(records):
    return {v for r in records if (v := phenotype(r.get("resistant_phenotype")))}


def classify(old, current, current_record_n):
    if not current_record_n:
        return "MISSING_CURRENT_LAB"
    if old == current:
        return "SAME_LABEL_SET" if old else "BOTH_LABEL_SETS_EMPTY"
    if (old, current) in [({"R"}, {"S"}), ({"S"}, {"R"})]:
        return "UNOPPOSED_R_S_REVERSAL_CANDIDATE"
    return "OTHER_LABEL_SET_CHANGE"


def validated_current(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    manifest = json.loads((folder / "retrieval_manifest.json").read_text())
    if manifest["status"] != "COMPLETE_EXPLICIT_PANEL_RETRIEVAL" or (folder / "FAILED.json").exists():
        raise ValueError(f"Incomplete retrieval: {folder}")
    by_id = {}
    for batch in manifest["batches"]:
        filename = batch["response_file"]
        if Path(filename).name != filename:
            raise ValueError("Unexpected response path")
        payload = (folder / "responses" / filename).read_bytes()
        if sha(payload) != batch["response_sha256"]:
            raise ValueError("Cached response hash mismatch")
        records = json.loads(payload)
        if len(records) != batch["record_count"]:
            raise ValueError("Cached response count mismatch")
        for row in records:
            if row["genome_id"] not in batch["requested_genome_ids"] or row["antibiotic"] != protocol["antibiotic"]:
                raise ValueError("Cached response outside requested panel")
            if row["id"] in by_id and by_id[row["id"]] != row:
                raise ValueError("Conflicting native record identity")
            by_id[row["id"]] = row
    combined = json.loads((folder / "current_all_evidence.json").read_text())
    if len(combined) != len(by_id) or len(combined) != manifest["records"]:
        raise ValueError("Combined archive count mismatch")
    if {r["id"]: r for r in combined} != by_id:
        raise ValueError("Combined archive differs from raw responses")
    lab = [r for r in combined if r.get("evidence") == "Laboratory Method"]
    if lab != json.loads((folder / "current_laboratory_only.json").read_text()):
        raise ValueError("Laboratory-only archive differs from explicit evidence filter")
    # A byte-identical local copy makes archived bundles portable while the
    # original protocol path and its pinned hash remain unchanged.
    panel = folder / "panel_input.csv"
    if not panel.is_file():
        panel = Path(protocol["panel_path"])
    if sha(panel.read_bytes()) != protocol["panel_sha256"]:
        raise ValueError("Panel input hash changed since retrieval")
    if len(protocol["candidate_genome_ids"]) != manifest["requested_genome_n"]:
        raise ValueError("Panel size mismatch")
    return protocol, combined, lab


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.archived.read_bytes()
    old = defaultdict(list)
    with args.archived.open(newline="", encoding="utf-8-sig") as handle:
        for r in csv.DictReader(handle):
            old[r["screen_species_annotation"], r["screen_drug"], r["genome_id"]].append(r)
    details, summary, covered = [], [], set()
    folders = sorted(args.current_root.glob("*_current"))
    if not folders:
        raise ValueError("No current panels")
    for folder in folders:
        protocol, combined, lab = validated_current(folder)
        species, drug = protocol["species_source_annotation"], protocol["antibiotic"]
        ids = set(protocol["candidate_genome_ids"])
        expected = {gid for s, d, gid in old if (s, d) == (species, drug)}
        if ids != expected or (species, drug) in covered:
            raise ValueError("Current and archived panels differ, or panel duplicated")
        covered.add((species, drug))
        current = defaultdict(list)
        for r in lab:
            current[r["genome_id"]].append(r)
        counts = Counter()
        for gid in sorted(ids):
            historical, latest = old[species, drug, gid], current[gid]
            a, b = label_set(historical), label_set(latest)
            status = classify(a, b, len(latest))
            counts[status] += 1
            details.append({"species_annotation": species, "antibiotic": drug,
                            "genome_id": gid, "archived_label_set": "|".join(sorted(a)),
                            "current_laboratory_label_set": "|".join(sorted(b)),
                            "archived_record_n": len(historical),
                            "current_laboratory_record_n": len(latest), "comparison": status})
        evidence = Counter(r.get("evidence", "MISSING") for r in combined)
        summary.append({"species_annotation": species, "antibiotic": drug,
                        "genome_drug_pairs": len(ids),
                        "same_nonempty_label_set": counts["SAME_LABEL_SET"],
                        "both_label_sets_empty": counts["BOTH_LABEL_SETS_EMPTY"],
                        "missing_current_lab": counts["MISSING_CURRENT_LAB"],
                        "unopposed_R_S_reversal_candidates": counts["UNOPPOSED_R_S_REVERSAL_CANDIDATE"],
                        "other_label_set_changes": counts["OTHER_LABEL_SET_CHANGE"],
                        "archived_record_n": sum(len(old[species, drug, gid]) for gid in ids),
                        "current_laboratory_record_n": len(lab),
                        "computational_records_excluded": evidence["Computational Method"],
                        "other_evidence_records_excluded": sum(n for k, n in evidence.items()
                                                               if k not in {"Computational Method", "Laboratory Method"})})
    if covered != {(s, d) for s, d, _ in old}:
        raise ValueError("Some archived panels have no current comparison")
    if sha(args.archived.read_bytes()) != sha(source):
        raise ValueError("Archived input changed during comparison")
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "panel_summary.csv", summary, list(summary[0]))
    write_csv(args.out / "genome_drug_comparisons.csv", details, list(details[0]))
    differences = [r for r in details if r["comparison"] not in {"SAME_LABEL_SET", "BOTH_LABEL_SETS_EMPTY"}]
    write_csv(args.out / "label_set_difference_candidates.csv", differences, list(details[0]))
    report = {"status": "EXPLORATORY_LABEL_SET_SCREEN_NOT_CLINICAL_ADJUDICATION",
              "archived_input_sha256": sha(source), "completed_utc": datetime.now(timezone.utc).isoformat(),
              "comparison_unit": "genome_id plus antibiotic, not native AST observation identity",
              "panels": summary, "cached_response_hashes_and_combined_records_verified": True,
              "limitations": ["Same label sets can conceal changes in individual observations or metadata.",
                              "Empty sets do not provide evidence of phenotype stability.",
                              "A shared repository can preserve shared source errors.",
                              "No claims about NCBI cross-repository agreement or clinical correctness."]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"panels": len(summary), "genome_drug_pairs": len(details),
                      "difference_candidates": len(differences), "output": str(args.out)}))


if __name__ == "__main__":
    main()
