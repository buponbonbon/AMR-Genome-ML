#!/usr/bin/env python3
"""FILE19 v1.0.3: immutable cross-source, cross-time AST comparison.

No training, phenotype correction, recalculation of model metrics, or overwrite.
Outputs are forensic audit artifacts, not independently adjudicated AST.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

VERSION = "1.0.3"
DRUGS = ("meropenem", "imipenem", "ertapenem")
FOCUS = "PRJNA1322038"
EXPECTED = 175


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def load_csv(p: Path, required: set[str]) -> list[dict]:
    if not p.is_file():
        raise RuntimeError(f"Missing input: {p}")
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        absent = sorted(required - columns)
        if absent:
            raise RuntimeError(f"{p.name} missing columns: {absent}; found: {sorted(columns)}")
        return list(reader)


def norm(x: object) -> str:
    s = str(x or "").strip().lower()
    return {"r": "resistant", "s": "susceptible", "i": "intermediate", "non-susceptible": "nonsusceptible"}.get(s, s)


def parse_xml(xml_path: Path, target: set[str]) -> dict[tuple[str, str], dict]:
    if not xml_path.is_file():
        raise RuntimeError(f"Missing XML: {xml_path}")
    root = ET.parse(xml_path).getroot()
    out: dict[tuple[str, str], dict] = {}
    sample_seen: set[str] = set()
    for sample in root.findall(".//BioSample"):
        acc = sample.attrib.get("accession", "").strip()
        if acc not in target:
            continue
        if acc in sample_seen:
            raise RuntimeError(f"Duplicate BioSample in XML: {acc}")
        sample_seen.add(acc)
        for table in sample.findall(".//Comment/Table"):
            if not table.attrib.get("class", "").lower().startswith("antibiogram"):
                continue
            headers = [(c.text or "").strip().lower() for c in table.findall("./Header/Cell")]
            must = {"antibiotic", "resistance phenotype", "measurement sign", "measurement", "measurement units"}
            if not must.issubset(headers):
                raise RuntimeError(f"Unexpected NCBI antibiogram columns for {acc}: {headers}")
            for row in table.findall("./Body/Row"):
                values = [(c.text or "").strip() for c in row.findall("Cell")]
                if len(values) != len(headers):
                    raise RuntimeError(f"Antibiogram cells/header mismatch for {acc}")
                d = dict(zip(headers, values))
                drug = d["antibiotic"].lower().strip()
                if drug not in DRUGS:
                    continue
                key = (acc, drug)
                if key in out:
                    raise RuntimeError(f"Multiple BioSample AST rows for {key}; do not auto-choose")
                out[key] = {
                    "xml_label": norm(d["resistance phenotype"]),
                    "mic_sign": d["measurement sign"],
                    "mic_value": d["measurement"],
                    "mic_units": d["measurement units"],
                    "method": d.get("laboratory typing method", ""),
                    "platform": d.get("laboratory typing platform", ""),
                    "standard": d.get("testing standard", ""),
                }
    missing = target - sample_seen
    if missing:
        raise RuntimeError(f"Missing {len(missing)} target BioSamples in XML, e.g. {sorted(missing)[:5]}")
    return out


def unique_index(rows: list[dict], target: set[str], name: str, *, is_bq=False) -> dict[tuple[str, str], dict]:
    out = {}
    for row in rows:
        acc = str(row.get("biosample_acc", "")).strip()
        drug = str(row.get("antibiotic", "")).strip().lower()
        if acc not in target or drug not in DRUGS:
            continue
        key = (acc, drug)
        if key in out:
            raise RuntimeError(f"Duplicate {name} accession/drug {key}; examine source manually")
        out[key] = row
    return out


def relation(a: str, b: str) -> str:
    if not a or not b:
        return "MISSING"
    return "AGREE" if a == b else "DISAGREE"


def stat_row(rows: list[dict], drug: str, col: str) -> dict:
    counts = Counter(row[col] for row in rows if row["antibiotic"] == drug)
    return {"agree": counts["AGREE"], "disagree": counts["DISAGREE"], "missing": counts["MISSING"]}


def run(args) -> dict:
    root = Path(args.project_root).resolve()
    def path(x: str) -> Path:
        p = Path(x)
        return (p if p.is_absolute() else root / p).resolve()

    focus_path = path(args.focus_csv)
    old_path = path(args.old_raw_csv)
    xml_path = path(args.biosample_xml)
    bq_path = path(args.bigquery_csv)
    out_dir = path(args.out_dir)
    inputs = {"focus_file19_v1_0_2": focus_path, "historical_file11_raw": old_path,
              "ncbi_biosample_xml": xml_path, "current_bigquery_csv": bq_path}
    focus = load_csv(focus_path, {"biosample_acc", "asm_acc", "bioproject_acc", "frozen_label", "ncbi_phenotype"})
    if len(focus) != EXPECTED:
        raise RuntimeError(f"Focus CSV must have {EXPECTED} rows; got {len(focus)}")
    if any(row["bioproject_acc"].strip() != FOCUS for row in focus):
        raise RuntimeError("Focus CSV contains another BioProject")
    by_sample = {}
    for row in focus:
        acc = row["biosample_acc"].strip()
        if acc in by_sample:
            raise RuntimeError(f"Duplicate BioSample in focus CSV: {acc}")
        if row["frozen_label"] not in ("0", "1"):
            raise RuntimeError(f"Nonbinary frozen meropenem outcome for {acc}")
        by_sample[acc] = row
    target = set(by_sample)
    xml = parse_xml(xml_path, target)
    old = unique_index(load_csv(old_path, {"target_acc", "biosample_acc", "asm_acc", "antibiotic", "phenotype"}), target, "historical FILE11 raw")
    bq = unique_index(load_csv(bq_path, {"queried_at_utc", "target_acc", "biosample_acc", "asm_acc", "antibiotic", "phenotype"}), target, "current BigQuery", is_bq=True)
    if not bq:
        raise RuntimeError("Current BigQuery CSV contains zero records from the target 175 BioSamples")
    bq_times = set()
    bq_extra_samples = set()
    for k, row in bq.items():
        if not row["queried_at_utc"].strip():
            raise RuntimeError(f"Current BQ row {k} has no query timestamp")
        bq_times.add(row["queried_at_utc"].strip())
        if row["asm_acc"].strip() != by_sample[k[0]]["asm_acc"].strip():
            raise RuntimeError(f"Current BQ assembly accession mismatch: {k}: {row['asm_acc']} vs {by_sample[k[0]]['asm_acc']}")
    if len(bq_times) != 1:
        raise RuntimeError(f"BigQuery CSV has {len(bq_times)} query timestamps; require a single-query snapshot")
    data = []
    for acc in sorted(target):
        fi = by_sample[acc]
        for drug in DRUGS:
            key = (acc, drug)
            orow = old.get(key, {})
            brow = bq.get(key, {})
            xrow = xml.get(key, {})
            if orow and orow["asm_acc"].strip() != fi["asm_acc"].strip():
                raise RuntimeError(f"Historical FILE11 raw assembly mismatch {key}")
            if orow and brow and orow["target_acc"].strip() != brow["target_acc"].strip():
                raise RuntimeError(f"target_acc changed between FILE11 and current BQ for {key}; investigate")
            oldlabel = norm(orow.get("phenotype", ""))
            newlabel = norm(brow.get("phenotype", ""))
            xmllabel = norm(xrow.get("xml_label", ""))
            # Frozen phenotype is only the original MEROPENEM label (never impute other drugs).
            frozen = ("resistant" if fi["frozen_label"] == "1" else "susceptible") if drug == "meropenem" else ""
            row = {
                "biosample_acc": acc,
                "asm_acc": fi["asm_acc"],
                "antibiotic": drug,
                "historical_target_acc": orow.get("target_acc", ""),
                "current_target_acc": brow.get("target_acc", ""),
                "frozen_meropenem_label": frozen,
                "historical_file11_raw_label": oldlabel,
                "current_bigquery_label": newlabel,
                "current_biosample_xml_label": xmllabel,
                "mic_sign": xrow.get("mic_sign", ""),
                "mic_value": xrow.get("mic_value", ""),
                "mic_units": xrow.get("mic_units", ""),
                "mic_method": xrow.get("method", ""),
                "mic_platform": xrow.get("platform", ""),
                "testing_standard": xrow.get("standard", ""),
                "old_vs_xml": relation(oldlabel, xmllabel),
                "current_vs_xml": relation(newlabel, xmllabel),
                "old_vs_current": relation(oldlabel, newlabel),
                "frozen_vs_xml": relation(frozen, xmllabel) if drug == "meropenem" else "NOT_APPLICABLE",
                "queried_at_utc": brow.get("queried_at_utc", ""),
            }
            data.append(row)

    mer = [r for r in data if r["antibiotic"] == "meropenem"]
    mer_current = stat_row(data, "meropenem", "current_vs_xml")
    mer_old = stat_row(data, "meropenem", "old_vs_xml")
    mer_old_current = stat_row(data, "meropenem", "old_vs_current")
    if mer_current["missing"]:
        verdict = "EVIDENCE_INCOMPLETE__BIGQUERY_CURRENT_MEROPENEM_MISSING"
    elif mer_current["disagree"]:
        verdict = "CURRENT_BIGQUERY_AND_BIOSAMPLE_DISCORDANT__HOLD"
    elif mer_old["disagree"] == EXPECTED and mer_old_current["disagree"] == EXPECTED:
        verdict = "CURRENT_BQ_BIOSAMPLE_CONSISTENT__HISTORICAL_RAW_DISCORDANT__CAUSE_UNRESOLVED"
    else:
        verdict = "CURRENT_BQ_BIOSAMPLE_CONSISTENT__HISTORICAL_PATTERN_MIXED__REVIEW"

    summaries = {
        drug: {
            "historical_file11_raw_vs_xml": stat_row(data, drug, "old_vs_xml"),
            "current_bigquery_vs_xml": stat_row(data, drug, "current_vs_xml"),
            "historical_vs_current_bigquery": stat_row(data, drug, "old_vs_current"),
        } for drug in DRUGS
    }
    if any(r["frozen_vs_xml"] != "DISAGREE" for r in mer):
        raise RuntimeError("Frozen-vs-XML pattern no longer 175/175 opposite: investigate input provenance")
    if any(not old.get((acc, "meropenem")) for acc in target):
        raise RuntimeError("Historical FILE11 raw lacks at least one target meropenem record")
    if any(not xml.get((acc, "meropenem")) for acc in target):
        raise RuntimeError("NCBI XML lacks at least one target meropenem record")
    summary = {
        "file": "FILE19_TEMPORAL_AST_SOURCE_RECONCILIATION", "version": VERSION,
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_E2_modified": False, "training_performed": False, "primary_metrics_recomputed": False,
        "focus_bioproject": FOCUS, "focus_biosample_count": EXPECTED,
        "query_timestamp_utc_unmodified": next(iter(bq_times)),
        "query_timestamp_is_not_table_update_timestamp": True,
        "input_sha256": {name: sha256(p) for name, p in inputs.items()},
        "meropenem_current_bq_row_count": sum(bool(bq.get((acc, "meropenem"))) for acc in target),
        "per_drug": summaries,
        "verdict": verdict,
        "interpretation": "A dated discrepancy between historical source CSV and current sources is not sufficient to prove NCBI edited phenotype records on any specific day. Alternative queries, joins, caching and source/version differences need provenance verification.",
        "evidence_type": "SOURCE_RECORD_COMPARISON__NOT_INDEPENDENT_LABORATORY_ADJUDICATION",
    }
    if out_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    csv_path = out_dir / "file19_temporal_175_three_drug_crosswalk.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)
    summary_path = out_dir / "file19_temporal_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# FILE19 v1.0.3 — Cross-source temporal AST audit", "",
        f"- Query timestamp (from BigQuery result): {next(iter(bq_times))}",
        f"- Verdict: **{verdict}**", "",
        "Historical FILE11 raw and current NCBI BigQuery/BioSample are compared as **source records**, not independent validation of laboratory AST.",
        "No outcome labels, predictions, training files or primary frozen results were modified.", "",
        "| Drug | Old vs XML (agree/disagree/missing) | Current vs XML (agree/disagree/missing) | Old vs current (agree/disagree/missing) |",
        "|---|---:|---:|---:|",
    ]
    for drug in DRUGS:
        ss = summaries[drug]
        def fmt(d: dict) -> str: return f"{d['agree']}/{d['disagree']}/{d['missing']}"
        report.append(f"| {drug} | {fmt(ss['historical_file11_raw_vs_xml'])} | {fmt(ss['current_bigquery_vs_xml'])} | {fmt(ss['historical_vs_current_bigquery'])} |")
    report.extend(["", "## Causal interpretation boundary", "",
                  "A matching current BigQuery/BioSample snapshot and a discordant older CSV establish a *between-snapshot/source difference*, not when, why, or by whom labels changed.",
                  "The BigQuery table is a mutable public resource; obtaining an historical query job ID or NCBI explanation would increase confidence in the cause.",
                  "Do not overwrite E2_FROZEN or substitute these phenotypes into the primary study.", ""])
    report_path = out_dir / "file19_temporal_reporting_guardrails.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    output_hashes = {p.name: sha256(p) for p in [csv_path, summary_path, report_path]}
    (out_dir / "file19_temporal_output_sha256.json").write_text(json.dumps({"version": VERSION, "sha256": output_hashes}, indent=2) + "\n", encoding="utf-8")
    print("[FILE19 v1.0.3] BigQuery records:",len(bq),"| unique target samples:",len({a for a, _ in bq}))
    print("[FILE19 v1.0.3] Meropenem current BQ vs XML:",mer_current)
    print("[FILE19 v1.0.3] Old raw vs current BQ:",mer_old_current)
    print("[FILE19 v1.0.3] Verdict:",verdict)
    print("[FILE19 v1.0.3] Outputs:",out_dir)
    return summary


def self_test():
    assert norm("R") == "resistant"
    assert norm("S") == "susceptible"
    assert relation("resistant", "susceptible") == "DISAGREE"
    assert relation("", "susceptible") == "MISSING"
    sample_xml = '''<BioSampleSet><BioSample accession="SAMN1"><Description><Comment>
    <Table class="Antibiogram.1.0"><Header><Cell>Antibiotic</Cell><Cell>Resistance phenotype</Cell><Cell>Measurement sign</Cell><Cell>Measurement</Cell><Cell>Measurement units</Cell><Cell>Testing standard</Cell></Header><Body>
    <Row><Cell>meropenem</Cell><Cell>resistant</Cell><Cell>&gt;</Cell><Cell>16</Cell><Cell>mg/L</Cell><Cell>EUCAST</Cell></Row>
    </Body></Table></Comment></Description></BioSample></BioSampleSet>'''
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "tiny.xml"; p.write_text(sample_xml)
        result = parse_xml(p, {"SAMN1"})
        assert result[("SAMN1", "meropenem")]["xml_label"] == "resistant"
        assert result[("SAMN1", "meropenem")]["mic_sign"] == ">"
    print("[FILE19 v1.0.3 SELFTEST] PASS — typed antibiogram parser and fail-closed comparison semantics")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--project-root", default=".")
    p.add_argument("--bigquery-csv", help="CSV exported from FILE19_v1_0_3_BIGQUERY_175.sql")
    p.add_argument("--focus-csv", default="results/file19_ast_provenance_audit_v1_0_2/file19_focus_PRJNA1322038_ast_triage.csv")
    p.add_argument("--old-raw-csv", default="data/external_validation/ncbi_pathogen_detection/raw/file11_ncbi_ast_isolates_joined_raw.csv")
    p.add_argument("--biosample-xml", default="results/file19_ncbi_evidence_cache_v1_0_1/PRJNA1322038_biosample_efetch.xml")
    p.add_argument("--out-dir", default="results/file19_temporal_ast_source_reconciliation_v1_0_3")
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    if not args.bigquery_csv:
        p.error("--bigquery-csv is required for real audit")
    try:
        run(args)
    except Exception as e:
        raise SystemExit(f"[FILE19 v1.0.3 FAIL-CLOSED] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
