#!/usr/bin/env python3
"""Offline reference-provenance audits. Never fits models or changes input files."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

VERSION = "1.0.1"
KEY = ("target_acc", "biosample", "drug")
LABELS = {"r": "R", "resistant": "R", "s": "S", "susceptible": "S",
          "i": "I", "intermediate": "I"}
DISCLAIMER = "Reference-record audit only; not clinical adjudication, causal attribution, or independent validation."


class AuditError(ValueError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_base"] = str(path.parent)
    cfg["_config"] = str(path)
    return cfg


def input_path(cfg, value):
    p = Path(value).expanduser()
    return (Path(cfg["_base"]) / p).resolve() if not p.is_absolute() else p.resolve()


def file_specs(cfg):
    for name, spec in cfg.items():
        if isinstance(spec, dict) and isinstance(spec.get("path"), str):
            yield name, input_path(cfg, spec["path"])


def new_output(path, inputs=()):
    out = Path(path).resolve()
    for src in inputs:
        src = Path(src).resolve()
        if out == src or out in src.parents or src in out.parents:
            raise AuditError("Output must be separate from every input file/directory")
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise AuditError("Missing/duplicate CSV header")
        rows = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise AuditError("CSV row width differs from header")
            rows.append({k: v.strip() for k, v in row.items()})
    return rows, fields


def label(value, binary=False):
    v = value.strip().lower()
    if binary and v in {"0", "1"}:
        return "R" if v == "1" else "S"
    if v not in LABELS:
        raise AuditError("Unknown phenotype; do not silently map missing/I to S")
    return LABELS[v]


def table(cfg, name, required):
    if name not in cfg:
        raise AuditError(f"Missing configuration section: {name}")
    spec = cfg[name]
    rows, headers = read_csv(input_path(cfg, spec["path"]))
    columns = spec.get("columns", {})
    if not set(required) <= set(columns):
        raise AuditError(f"{name}: incomplete explicit column mapping")
    if not set(columns.values()) <= set(headers):
        raise AuditError(f"{name}: mapped columns absent from input")
    result = []
    for raw in rows:
        r = {dest: raw[src] for dest, src in columns.items()}
        if any(not r.get(k) for k in required):
            raise AuditError(f"{name}: empty required field")
        if "drug" in r:
            aliases = {k.lower(): v.lower() for k, v in cfg.get("drug_aliases", {}).items()}
            v = r["drug"].lower()
            r["drug"] = aliases.get(v, v)
        if "label" in r:
            r["label"] = label(r["label"], bool(spec.get("binary_label")))
        result.append(r)
    if not result:
        raise AuditError(f"{name}: empty input table")
    return result


def unique(rows, keys, name, collapse=False):
    out = {}
    duplicates = 0
    for r in rows:
        k = tuple(r.get(x, "") for x in keys)
        if any(not x for x in k):
            raise AuditError(f"{name}: missing exact join key")
        if k in out:
            if not collapse or r != out[k]:
                raise AuditError(f"{name}: duplicate/conflicting join keys; resolve explicitly")
            duplicates += 1
        else:
            out[k] = r
    return out, duplicates


def metrics(labels, scores):
    if len(labels) != len(scores) or not labels:
        raise AuditError("Metric vectors empty or have unequal lengths")
    if any(y not in (0, 1) for y in labels):
        raise AuditError("Binary metric requires binary labels")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise AuditError("Probabilities must be finite and in [0,1]")
    n, pos = len(labels), sum(labels)
    neg = n - pos
    tp = sum(y == 1 and p >= .5 for y, p in zip(labels, scores))
    fp = sum(y == 0 and p >= .5 for y, p in zip(labels, scores))
    fn, tn = pos - tp, neg - fp
    ordered = sorted(zip(scores, labels), reverse=True)
    cumulative_pos = cumulative_n = 0
    ap_num = wins = 0.0
    j = 0
    while j < n:
        end = j + 1
        while end < n and ordered[end][0] == ordered[j][0]:
            end += 1
        block_pos = sum(y for _, y in ordered[j:end])
        block_neg = end - j - block_pos
        lower_neg = neg - (cumulative_n - cumulative_pos) - block_neg
        wins += block_pos * (lower_neg + .5 * block_neg)
        cumulative_pos += block_pos
        cumulative_n += end - j
        ap_num += block_pos * cumulative_pos / cumulative_n
        j = end
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    sensitivity = tp / pos if pos else None
    specificity = tn / neg if neg else None
    return {"n": n, "R": pos, "S": neg, "threshold": .5,
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "roc_auc": wins / (pos * neg) if pos and neg else None,
            "average_precision": ap_num / pos if pos else None,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2 if pos and neg else None,
            "mcc": (tp * tn - fp * fn) / math.sqrt(den) if den else 0.0,
            "brier": sum((p - y) ** 2 for y, p in zip(labels, scores)) / n,
            "undefined_reason": "single-class reference" if not (pos and neg) else None}


def expected(cfg, name, observed):
    if name in cfg.get("expected", {}) and observed != cfg["expected"][name]:
        raise AuditError(f"Expected-count guard failed: {name}; check selection/mapping")


def reconcile(cfg):
    old_rows = table(cfg, "historical", (*KEY, "label"))
    new_rows = table(cfg, "current", (*KEY, "label"))
    old, od = unique(old_rows, KEY, "historical", True)
    new, nd = unique(new_rows, KEY, "current", True)
    membership = {}
    if "membership" in cfg:
        rows = table(cfg, "membership", ("biosample", "project"))
        membership, _ = unique(rows, ("biosample",), "membership", True)
    for source in (old, new):
        for r in source.values():
            member = membership.get((r["biosample"],), {}).get("project", "")
            if member and r.get("project") and member != r["project"]:
                raise AuditError("BioProject metadata disagree with explicit membership")
            r["project"] = r.get("project") or member
    common = old.keys() & new.keys()
    expected(cfg, "matched_records", len(common))
    if not common:
        raise AuditError("No exact historical/current matches")
    focal = cfg.get("focal_project")
    drug = cfg.get("target_drug", "meropenem").lower()
    if not focal:
        raise AuditError("Declare focal_project explicitly; do not select by model errors")
    crosswalk = []
    counts = Counter()
    for k in sorted(common):
        a, b = old[k], new[k]
        if a["project"] and b["project"] and a["project"] != b["project"]:
            raise AuditError("Source project changed across versions; review mapping")
        project = a["project"] or b["project"]
        group = "focal" if project == focal else "other" if project else "unknown"
        counts[(group, a["drug"], a["label"], b["label"])]+=1
        crosswalk.append({**dict(zip(KEY, k)), "project": project,
                          "historical_label": a["label"], "current_label": b["label"]})
    report = {"historical_rows": len(old_rows), "historical_unique": len(old),
              "historical_duplicate_projected_rows": od, "current_rows": len(new_rows),
              "current_unique": len(new), "current_duplicate_projected_rows": nd,
              "matched_records": len(common), "historical_only": len(old.keys()-new.keys()),
              "current_only": len(new.keys()-old.keys()),
              "source_chain_status": "MANUAL_REVIEW_REQUIRED: matching hashes do not establish the full SQL/export chain",
              "clinical_adjudication": False, "independent_validation": False,
              "confidence_intervals": "NOT_ESTIMATED", "disclaimer": DISCLAIMER}
    unmatched = [{**dict(zip(KEY,k)), "available_in": side}
                 for side, ks in [("historical_only",old.keys()-new.keys()),("current_only",new.keys()-old.keys())]
                 for k in sorted(ks)]
    if "biosample_table" in cfg:
        xml_rows = table(cfg, "biosample_table", ("biosample", "drug", "label"))
        xml, _ = unique(xml_rows, ("biosample", "drug"), "biosample_table", True)
        comparisons = Counter()
        focal_comparisons = Counter()
        for r in new.values():
            x = xml.get((r["biosample"], r["drug"]))
            status="missing" if x is None else "agree" if x["label"] == r["label"] else "disagree"
            comparisons[status] += 1
            if r["project"] == focal:
                focal_comparisons[status] += 1
        report["current_vs_biosample"] = dict(comparisons)
        report["focal_current_vs_biosample"] = dict(focal_comparisons)
        report["xml_review_required"] = bool(comparisons["disagree"] or focal_comparisons["missing"] or not focal_comparisons)
        report["biosample_is_independent_ast"] = False
    if "predictions" in cfg:
        ps = cfg["predictions"]
        preds = table(cfg, "predictions", ("target_acc", "biosample", "label", "probability"))
        for r in preds:
            r["drug"] = drug
        unique(preds, KEY, "predictions")
        if "cohort" in ps:
            raise AuditError("Use a dedicated frozen E2 predictions file, not a silently filtered mixed-cohort file")
        expected(cfg, "prediction_n", len(preds))
        ys, scenario, scores, selected = [], [], [], []
        substituted = []
        for r in preds:
            k = tuple(r[x] for x in KEY)
            if k not in old:
                raise AuditError("A frozen prediction has no exact historical accession/BioSample/drug match")
            a = old[k]
            if r["label"] not in {"R", "S"} or r["label"] != a["label"]:
                raise AuditError("Frozen historical labels differ from archived source")
            project = a["project"]
            if not project:
                raise AuditError("Every prediction requires explicit BioProject membership")
            is_focal = project == focal
            if is_focal and k not in new:
                raise AuditError("Focal prediction missing its current counterpart")
            s = new[k]["label"] if is_focal else a["label"]
            if s not in {"R", "S"}:
                raise AuditError("Scenario contains I; binary endpoint must be resolved explicitly")
            try:
                score = float(r["probability"])
            except ValueError as e:
                raise AuditError("Invalid probability") from e
            y = int(r["label"] == "R")
            if "prediction" in r:
                try:
                    predicted = int(r["prediction"])
                except ValueError as e:
                    raise AuditError("Frozen prediction must be 0 or 1") from e
                if predicted not in (0,1) or predicted != int(score >= .5):
                    raise AuditError("Stored prediction disagrees with fixed >=0.5 threshold")
            ys.append(y); scenario.append(int(s == "R")); scores.append(score); selected.append(is_focal)
            substituted.append({**dict(zip(KEY,k)), "project": project, "probability": r["probability"],
                                "historical_label": r["label"], "scenario_label": s,
                                "focal": is_focal, "label_changed": s != r["label"]})
        expected(cfg, "focal_prediction_n", sum(selected))
        expected(cfg, "historical_R", sum(ys))
        expected(cfg, "scenario_R", sum(scenario))
        report["historical_E2"] = metrics(ys, scores)
        report["partial_source_substitution_POSTHOC"] = metrics(scenario, scores)
        report["substituted_records"] = sum(selected)
        report["changed_labels"] = sum(a != b for a,b in zip(ys,scenario))
        report["frozen_score_vector_sha256"] = hashlib.sha256(json.dumps(scores,allow_nan=False).encode()).hexdigest()
        report["subgroups_descriptive"] = {}
        for name, flag in [("focal",True),("other",False)]:
            ix = [i for i,v in enumerate(selected) if v == flag]
            if ix:
                report["subgroups_descriptive"][name] = {
                    "historical":metrics([ys[i] for i in ix],[scores[i] for i in ix]),
                    "scenario":metrics([scenario[i] for i in ix],[scores[i] for i in ix])}
    else:
        substituted = []
        report["prediction_metrics"] = "NOT_RUN: frozen predictions not supplied"
    transitions = [{"group":g,"drug":d,"historical_label":a,"current_label":b,"n":n}
                   for (g,d,a,b),n in sorted(counts.items())]
    return report, crosswalk, unmatched, transitions, substituted


def xml_extract(cfg):
    spec = cfg["xml"]
    path = input_path(cfg, spec["path"])
    # Do not accept entity declarations from supplied XML.
    with path.open("rb") as f:
        tail = b""
        for b in iter(lambda:f.read(1024*1024),b""):
            data = (tail+b).upper()
            if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
                raise AuditError("XML DTD/entity declarations are not supported")
            tail = data[-32:]
    root = ET.parse(path).getroot()
    if "record_path" not in spec or "columns" not in spec:
        schema = Counter()
        for node in root.iter():
            schema[(node.tag, tuple(sorted(node.attrib)))] += 1
        return {"status":"SCHEMA_ONLY_NOT_RECONCILED", "nodes":[
            {"tag":tag,"attributes":list(attrs),"n":n} for (tag,attrs),n in sorted(schema.items())]}, []
    def select(node, selector):
        if selector.startswith("@"):
            return node.get(selector[1:], "").strip()
        if "/@" in selector:
            xp, attr = selector.rsplit("/@",1)
            child = node.find(xp)
            return child.get(attr, "").strip() if child is not None else ""
        return node.findtext(selector, "").strip()
    rows = []
    for sample in root.findall(spec.get("sample_path", ".//BioSample")):
        acc = select(sample,spec.get("accession_selector","@accession"))
        for node in sample.findall(spec["record_path"]):
            rows.append({"biosample":acc, **{k:select(node,v) for k,v in spec["columns"].items()}})
    if not rows or not {"drug","label"} <= set(spec["columns"]):
        raise AuditError("XML mapping returned no records or lacks drug/label selectors")
    unique(rows,("biosample","drug"),"XML",True)
    return {"status":"EXTRACTED_WITH_EXPLICIT_SELECTORS", "records":len(rows), "disclaimer":DISCLAIMER}, rows


def mic_interpret(row, breakpoints):
    needed = ("mic_value","mic_sign","unit","method","standard","version")
    if any(not row.get(k) for k in needed):
        return {"status":"INCOMPLETE_METADATA", "possible_categories":""}
    if row["method"].upper() != "MIC":
        return {"status":"NOT_MIC", "possible_categories":""}
    try:
        m = Decimal(row["mic_value"])
    except InvalidOperation:
        return {"status":"INVALID_MIC", "possible_categories":""}
    if not m.is_finite() or m <= 0:
        return {"status":"INVALID_MIC", "possible_categories":""}
    sign = row["mic_sign"].replace("≤","<=").replace("≥",">=")
    # The supplied NCBI XML uses == for exact MICs; keep the raw input intact.
    if sign == "==":
        sign = "="
    if sign not in {"=","<","<=",">",">="}:
        return {"status":"UNKNOWN_SIGN", "possible_categories":""}
    keys = ("organism","drug","standard","version","context","unit")
    if any(not row.get(k) for k in keys):
        return {"status":"INCOMPLETE_INTERPRETIVE_CONTEXT", "possible_categories":""}
    candidates = [b for b in breakpoints if all(b.get(k,"").lower() == row[k].lower() for k in keys)]
    if len(candidates) != 1:
        return {"status":"NO_UNIQUE_DOCUMENTED_BREAKPOINT", "possible_categories":""}
    b = candidates[0]
    if not b.get("source_url") or not b.get("source_sha256") or not re.fullmatch(r"[0-9a-fA-F]{64}",b["source_sha256"]):
        return {"status":"BREAKPOINT_CITATION_INCOMPLETE", "possible_categories":""}
    try:
        s, r = Decimal(b["s_max"]), Decimal(b["r_above"])
    except (InvalidOperation,KeyError):
        return {"status":"INVALID_BREAKPOINT", "possible_categories":""}
    if not s.is_finite() or not r.is_finite() or not 0 < s <= r:
        return {"status":"UNSUPPORTED_BREAKPOINT", "possible_categories":""}
    # Only supported convention: S <= s_max, R > r_above, otherwise I.
    if b.get("convention") != "S_LE_R_GT":
        return {"status":"UNSUPPORTED_BREAKPOINT_CONVENTION", "possible_categories":""}
    low, high = (m,m) if sign == "=" else (Decimal(0),m) if sign in {"<","<="} else (m,Decimal("Infinity"))
    categories = []
    if low < s or (low == s and sign in {"=",">="}):
        categories.append("S")
    if s < r and high > s and (low < r or (low == r and sign in {"=",">="})):
        categories.append("I")
    if high > r:
        categories.append("R")
    # A strict upper limit equal to a boundary belongs only below it.
    if sign == "<" and m == r and "I" not in categories and s < r:
        categories.append("I")
    return {"status":"VERSION_CONDITIONAL_INTERPRETATION" if len(categories)==1 else "CENSORED_AMBIGUOUS",
            "possible_categories":"|".join(categories), "breakpoint_source_url":b["source_url"],
            "breakpoint_source_sha256":b["source_sha256"]}


def mic_audit(cfg):
    rows = table(cfg,"mic",("biosample","drug"))
    raw_count=len(rows)
    index,duplicates=unique(rows,("biosample","drug"),"MIC table",True)
    rows=list(index.values())
    expected(cfg,"mic_rows",len(rows))
    fields = ("mic_value","mic_sign","unit","method","platform","standard","version","organism","context","testing_date")
    completeness = {k:sum(bool(r.get(k)) for r in rows) for k in fields}
    bps = []
    if "breakpoints" in cfg:
        bps,_ = read_csv(input_path(cfg,cfg["breakpoints"]["path"]))
    results = [{**r, **mic_interpret(r,bps)} for r in rows]
    return {"rows":len(rows),"raw_rows":raw_count,"duplicate_projected_rows":duplicates,"nonempty_fields":completeness,
            "statuses":dict(Counter(r["status"] for r in results)),
            "clinical_adjudication":False,"disclaimer":DISCLAIMER,
            "breakpoint_document_contents_verified":False}, results


def inventory(root):
    root = Path(root).resolve()
    wanted = {"file19_temporal_175_three_drug_crosswalk.csv", "file19_temporal_summary.json",
              "file19_summary.json", "file18_summary.json", "file17_summary.json",
              "file11c_blind_predictions.csv", "PRJNA1322038_biosample_efetch.xml",
              "bq-results-20260922-062428-1790058310007.csv",
              "bq-results-20261022-175414-1792691665677.csv",
              "main.tex","supplementary.tex","references_restored_50.bib"}
    skip = {".git",".venv","venv","conda-env","__pycache__","node_modules"}
    found = {n:[] for n in sorted(wanted)}
    errors = []
    def walk(folder):
        try:
            for p in folder.iterdir():
                if p.is_symlink():
                    continue
                if p.is_dir() and p.name not in skip:
                    walk(p)
                elif p.is_file() and p.name in found:
                    found[p.name].append({"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":digest(p)})
        except OSError as e:
            errors.append(type(e).__name__)
    walk(root)
    return {"root":str(root),"found":found,"missing":[n for n,v in found.items() if not v],
            "ambiguous_basenames":[n for n,v in found.items() if len(v)>1],"traversal_errors":errors,
            "status":"INVENTORY_ONLY_NOT_SCIENTIFIC_VERIFICATION", "disclaimer":DISCLAIMER}


def manuscript(cfg):
    main = input_path(cfg,cfg["main_tex"]["path"]).read_text(encoding="utf-8")
    supp = input_path(cfg,cfg["supplementary_tex"]["path"]).read_text(encoding="utf-8")
    bib = input_path(cfg,cfg["bibliography"]["path"]).read_text(encoding="utf-8")
    def strip_comments(t):
        return re.sub(r"(?<!\\)%[^\n]*","",t)
    citations = set()
    for t in (main,supp):
        t = strip_comments(t)
        for m in re.finditer(r"\\(?:cite\w*|nocite)\*?(?:\s*\[[^]]*\])*\s*\{([^}]+)\}",t):
            citations.update(k.strip() for k in m[1].split(",") if k.strip() != "*")
    entries = re.findall(r"@(?!(?:string|comment|preamble)\b)\w+\s*[{(]\s*([^,\s]+)\s*,",bib,re.I)
    missing = sorted(citations-set(entries))
    duplicates = sorted(k for k,v in Counter(entries).items() if v>1)
    warnings = []
    if re.search(r"S1\s*[–—-]+\s*S5",main):
        warnings.append("Main refers to S1-S5: check against actual supplementary figures")
    if "AUTHOR VERIFICATION REQUIRED" in main:
        warnings.append("Author declaration placeholders remain")
    if re.search(r"confusion.{0,200}main manuscript",supp,re.I|re.S):
        warnings.append("Manually review confusion-matrix reference to main manuscript")
    return {"unique_citation_keys":len(citations),"bib_entries":len(entries),
            "missing_keys":missing,"duplicate_keys":duplicates,"warnings":warnings,
            "status":"STATIC_SOURCE_CHECK_ONLY", "compilation":"NOT_RUN", "doi_metadata":"NOT_VERIFIED",
            "limitations":"Does not expand TeX macros, included files, or validate journal formatting."
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["inventory","reconcile","xml","mic","manuscript"])
    parser.add_argument("--config",type=Path)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--out",type=Path,required=True,help="NEW output directory, never an existing folder")
    args=parser.parse_args(argv)
    out = None
    try:
        cfg = load_config(args.config) if args.config else None
        if args.command != "inventory" and cfg is None:
            raise AuditError("This command requires --config")
        relevant = {
            "reconcile":{"historical","current","membership","predictions","biosample_table","historical_sql","export_chain"},
            "xml":{"xml"}, "mic":{"mic","breakpoints"},
            "manuscript":{"main_tex","supplementary_tex","bibliography"}, "inventory":set()}
        inputs = [(name,p) for name,p in file_specs(cfg) if name in relevant[args.command]] if cfg else []
        if cfg:
            inputs.append(("config",Path(cfg["_config"])))
        if args.command == "inventory":
            if args.root is None:
                raise AuditError("inventory requires --root")
            # Output may be under the project but never overwrites existing content.
            out = new_output(args.out)
            report=inventory(args.root)
            code=2 if report["missing"] or report["traversal_errors"] or report["ambiguous_basenames"] else 0
        else:
            hashes={name:{"path":str(p),"sha256":digest(p)} for name,p in inputs}
            for name,p in inputs:
                want=cfg.get(name,{}).get("sha256") if name!="config" else None
                if want and hashes[name]["sha256"].lower()!=want.lower():
                    raise AuditError(f"Expected input SHA256 mismatch: {name}")
            out=new_output(args.out,[p for _,p in inputs])
            code=0
            if args.command=="reconcile":
                report,cross,unmatched,trans,sub=reconcile(cfg)
                write_csv(out/"reference_crosswalk.csv",cross,[*KEY,"project","historical_label","current_label"])
                write_csv(out/"unmatched_records.csv",unmatched,[*KEY,"available_in"])
                write_csv(out/"transitions.csv",trans,["group","drug","historical_label","current_label","n"])
                if sub:
                    write_csv(out/"E2_POSTHOC_reference_scenario.csv",sub,[*KEY,"project","probability","historical_label","scenario_label","focal","label_changed"])
                if not sub or report.get("xml_review_required"):
                    code=2
            elif args.command=="xml":
                report,rows=xml_extract(cfg)
                if rows:
                    write_csv(out/"biosample_records.csv",rows,sorted(set().union(*(r.keys() for r in rows))))
                else:
                    code=2
            elif args.command=="mic":
                report,rows=mic_audit(cfg)
                write_csv(out/"mic_audit.csv",rows,sorted(set().union(*(r.keys() for r in rows))))
                if any(r["status"]!="VERSION_CONDITIONAL_INTERPRETATION" for r in rows):
                    code=2
            else:
                report=manuscript(cfg)
                if report["missing_keys"] or report["duplicate_keys"] or report["warnings"]:
                    code=2
            for name,p in inputs:
                if digest(p)!=hashes[name]["sha256"]:
                    raise AuditError("Input changed during audit; output must not be trusted")
            report["inputs"]=hashes
        report.update({"audit_version":VERSION,"completed_utc":datetime.now(timezone.utc).isoformat()})
        write_json(out/"report.json",report)
        print(f"{args.command}: report written; exit={code}. {DISCLAIMER}")
        return code
    except (AuditError,OSError,ValueError,KeyError,ET.ParseError) as e:
        if out is not None:
            write_json(out/"FAILED.json",{"status":"FAILED_NOT_VALIDATED","error":str(e),"audit_version":VERSION})
        print(f"Audit stopped ({type(e).__name__}): {e}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
