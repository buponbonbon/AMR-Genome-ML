#!/usr/bin/env python3
"""Deterministic evidence-state audit of paired AST references; no clinical truth inference."""
import argparse
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata

VERSION = "1.0.0"
SIR = {"S", "I", "R"}
UNITS = {"mg/l", "ug/ml", "μg/ml"}


def parse_mic(raw):
    value = re.sub(r"\s+", "", str(raw)).replace("≤", "<=").replace("≥", ">=")
    m = re.fullmatch(r"(<=|>=|==|<|>|=)?([0-9]+(?:\.[0-9]+)?)", value)
    if not m:
        return None
    x = Decimal(m.group(2))
    if x <= 0:
        return None
    op = m.group(1) or "="
    if op in {"=", "=="}:
        return x, x, True, True
    if op in {"<", "<="}:
        return Decimal(0), x, False, op == "<="
    return x, Decimal("Infinity"), op == ">=", False


def intervals_overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if lo != hi:
        return lo < hi
    def contains(t):
        return (lo > t[0] or lo == t[0] and t[2]) and (lo < t[1] or lo == t[1] and t[3])
    return contains(a) and contains(b)


def normalize_unit(value):
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def unique_states(observations):
    states = {}
    for obs in observations:
        mic = parse_mic(obs.get("MIC_raw", ""))
        unit = normalize_unit(obs.get("units_raw", ""))
        category = str(obs.get("category_raw", "")).strip().upper()
        canonical_unit = "mg/L-equivalent" if unit in UNITS else unit
        key = (mic, canonical_unit, category)
        states.setdefault(key, []).append(obs)
    return states


def valid_history(record, case):
    if record.get("biosample_acc") != case["biosample_acc"] or record.get("antibiotic", "").strip().casefold() != case["antibiotic"].strip().casefold():
        return False
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("artifact_sha256", "")) or not record.get("source_url", "").startswith("https://"):
        return False
    try:
        when = datetime.strptime(record["date_raw"], "%m/%d/%y").date()
        as_of = date.fromisoformat(case["as_of_date"])
    except (ValueError, KeyError):
        return False
    old, new = parse_mic(record.get("old_value", "")), parse_mic(record.get("new_value", ""))
    return when <= as_of and old is not None and new is not None and old != new


def audit_case(case):
    old_obs, new_obs = case.get("NCBI_observations", []), case.get("CDC_observations", [])
    old_states, new_states = unique_states(old_obs), unique_states(new_obs)
    base = {"biosample_acc": case["biosample_acc"], "antibiotic": case["antibiotic"],
            "NCBI_observation_rows": len(old_obs), "CDC_observation_rows": len(new_obs),
            "NCBI_distinct_states": len(old_states), "CDC_distinct_states": len(new_states),
            "category_difference": None, "MIC_interval_difference": None, "MIC_disjointness": None,
            "native_history_only": None, "exact_history_link": None,
            "mechanism_resolved": False, "clinical_correctness": "NOT_ADJUDICATED", "matched_history": []}
    if not old_states or not new_states:
        return {**base, "evidence_state": "INSUFFICIENT_PAIRED_REFERENCE"}
    if len(old_states) != 1 or len(new_states) != 1:
        return {**base, "evidence_state": "AMBIGUOUS_MULTIPLE_REFERENCE_STATES"}
    (a, au, ac), (b, bu, bc) = next(iter(old_states)), next(iter(new_states))
    if a is None or b is None or au != "mg/L-equivalent" or bu != "mg/L-equivalent":
        return {**base, "evidence_state": "UNSUPPORTED_OR_MISSING_MIC_OR_UNIT"}
    category_changed = ac != bc if ac in SIR and bc in SIR else None
    different, disjoint = a != b, not intervals_overlap(a, b)
    updates = [r for r in case.get("MIC_history", []) if valid_history(r, case)]
    linked = [r for r in updates if parse_mic(r["old_value"]) == a and parse_mic(r["new_value"]) == b]
    base.update(category_difference=category_changed, MIC_interval_difference=different, MIC_disjointness=disjoint,
                native_history_only=bool(updates), exact_history_link=bool(linked), matched_history=linked)
    if linked:
        return {**base, "evidence_state": "DOCUMENTED_MIC_UPDATE_LINK", "mechanism_resolved": True}
    if category_changed is None:
        return {**base, "evidence_state": "CATEGORY_UNAVAILABLE_CAUSE_UNRESOLVED"}
    if disjoint:
        state = "DISJOINT_MIC_WITH_CATEGORY_CHANGE" if category_changed else "DISJOINT_MIC_WITH_SAME_CATEGORY"
    elif category_changed:
        state = "CATEGORY_ONLY_REFERENCE_DIFFERENCE"
    elif different:
        state = "OVERLAPPING_REPORTING_INTERVALS"
    else:
        state = "SAME_REPORTED_REFERENCE"
    return {**base, "evidence_state": state}


def summarize(results):
    positive = [r for r in results if r["exact_history_link"]]
    baseline_names = ["category_difference", "MIC_interval_difference", "MIC_disjointness", "native_history_only", "exact_history_link"]
    return {"auditor_version": VERSION, "unique_case_keys": len(results),
            "evidence_state_counts": dict(Counter(r["evidence_state"] for r in results)),
            "documented_update_link_case_keys": len(positive),
            "baseline_counts": {name: {"flagged_case_keys": sum(r[name] is True for r in results),
                                         "eligible_case_keys": sum(r[name] is not None for r in results),
                                         "retrieved_documented_links": sum(r[name] is True for r in positive),
                                         "documented_link_retrieval_fraction": sum(r[name] is True for r in positive)/len(positive) if positive else None}
                                for name in baseline_names},
            "limitations": ["Unmatched history is not a negative update label; specificity and clinical-error precision are not estimated.",
                            "Exact-history-link is a direct evidence lookup baseline; this auditor has no claimed detection advantage over it.",
                            "The mechanism-resolved flag means a recorded value update is linked, not that the reason for the update or biological truth is established.",
                            "Same reported reference does not certify clinical correctness.",
                            "Native breakpoint editions and assay identity are required to establish a breakpoint-only mechanism; this version only reports candidates."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    raw = args.cases.read_bytes()
    cases = json.loads(raw)["cases"]
    keys = [(x["biosample_acc"], x["antibiotic"]) for x in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated case keys must be combined before audit")
    results = [audit_case(c) for c in cases]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "case_results.json").write_text(json.dumps(results, indent=2) + "\n")
    report = {**summarize(results), "input_sha256": hashlib.sha256(raw).hexdigest(),
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
