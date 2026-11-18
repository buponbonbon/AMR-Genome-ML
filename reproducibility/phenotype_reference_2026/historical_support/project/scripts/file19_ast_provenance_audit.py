#!/usr/bin/env python3
"""FILE19 v1.0.2 | E2 PRJNA1322038 AST provenance audit.

READ ONLY: original FILE01--FILE18 inputs, phenotypes and models.
Output: new directories under results/. No training, no AST relabeling.

Modes:
  discover --project-root .
  fetch --project-root . --email YOUR_EMAIL   # online NCBI BioSample XML
  audit --project-root . [--biosample-xml ...] [--external-ast-csv ...]
  self-test

Rationale: public NCBI BioSample AST is provenance evidence, NOT an independent
laboratory reference. Numeric MIC and reported phenotype may be mutually
surprising but cannot be adjudicated without drug/species/standard/year/method.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

VERSION = "1.0.2"
FOCUS = "PRJNA1322038"
EXPECTED = {"e2_n": 525, "e2_resistant": 163, "focus_n": 175,
            "focus_resistant": 36, "focus_errors": 163, "full_errors": 195}
FROZEN_CANDIDATES = [
    "data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv",
    "results/file16d_source_audit_v1/file16d_manual_review_priority.csv",
    "file11_external_strict_E2_FROZEN.csv",
]
PRED_CANDIDATES = [
    "results/file16_audit_mechanisms_v1/file16_per_genome_prediction_error_audit.csv",
    "results/file16_audit_v1/file16_per_genome_prediction_error_audit.csv",
]
DEFAULT_CACHE = "results/file19_ncbi_evidence_cache_v1_0_1/PRJNA1322038_biosample_efetch.xml"
DEFAULT_OUTPUT = "results/file19_ast_provenance_audit_v1_0_2"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(x):
    if x is None or pd.isna(x):
        return ""
    return str(x).strip()


def cn(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def pick(df, *candidates):
    by = {cn(col): col for col in df.columns}
    for candidate in candidates:
        if cn(candidate) in by:
            return by[cn(candidate)]
    return None


def choose(root, optional, candidates, required=True):
    if optional:
        p = Path(optional)
        p = p if p.is_absolute() else root / p
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p.resolve()
    for c in candidates:
        p = root / c
        if p.is_file():
            return p.resolve()
    if required:
        raise FileNotFoundError("Cannot find input. Expected one of: " + "; ".join(candidates))
    return None


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def outdir_new(path):
    if path.exists():
        raise FileExistsError(f"FAIL-CLOSED existing output directory: {path}. Keep immutable; change --outdir for a new run.")
    path.mkdir(parents=True, exist_ok=False)


def flag_binary(v):
    s = norm(v).lower()
    if s in ("1", "resistant", "r"): return 1
    if s in ("0", "susceptible", "s"): return 0
    raise ValueError(f"Unexpected frozen binary phenotype: {v!r}")


def load_e2(root: Path, frozen_file, pred_file, test_mode=False):
    frozen_path = choose(root, frozen_file, FROZEN_CANDIDATES)
    pred_path = choose(root, pred_file, PRED_CANDIDATES)
    meta = pd.read_csv(frozen_path, dtype=str, keep_default_na=False, low_memory=False)
    pred = pd.read_csv(pred_path, dtype=str, keep_default_na=False, low_memory=False)
    # FILE16D review table already contains probabilities and all accessions.
    af = pick(meta, "asm_acc", "genome_id", "assembly_accession")
    bf = pick(meta, "biosample_acc", "BioSample", "biosample_accession")
    cf = pick(meta, "bioproject_acc", "BioProject")
    yf = pick(meta, "y_true", "label_binary")
    pf = pick(meta, "phenotype")
    drugf = pick(meta, "antibiotic")
    if not all([af,bf,cf]) or not (yf or pf):
        raise RuntimeError("Frozen/FILE16D table needs asm_acc, biosample_acc, bioproject_acc and label")
    if drugf:
        meta=meta.loc[meta[drugf].map(lambda x: norm(x).lower()=="meropenem")].copy()
    meta = pd.DataFrame({"asm_acc":meta[af].map(norm),
                         "biosample_acc":meta[bf].map(norm).str.upper(),
                         "bioproject_acc":meta[cf].map(norm),
                         "frozen_label":meta[yf if yf else pf].map(flag_binary)})
    if meta.asm_acc.eq("").any() or meta.asm_acc.duplicated().any():
        raise RuntimeError("Duplicate/empty assembly identifier in E2 source")
    if meta.biosample_acc.eq("").any() or meta.biosample_acc.duplicated().any():
        raise RuntimeError("Missing/duplicate BioSample: cannot safely join AST")
    if not meta.biosample_acc.str.fullmatch(r"SAM[NED][0-9]+", case=False).all():
        raise RuntimeError("Unexpected BioSample accession format")
    co = pick(pred, "cohort")
    if co: pred = pred.loc[pred[co].map(lambda x: norm(x).upper()=="E2")].copy()
    ap = pick(pred, "genome_id", "asm_acc")
    yp = pick(pred, "y_true")
    pp = pick(pred, "prob_resistance", "probability_R_reproduced_FILE11C_model")
    if not all([ap,yp,pp]):
        raise RuntimeError("Missing E2 prediction genome_id/y_true/prob_resistance")
    pred=pd.DataFrame({"asm_acc":pred[ap].map(norm),"pred_label":pred[yp].map(flag_binary),
                       "prob_R":pd.to_numeric(pred[pp],errors="raise")})
    if pred.asm_acc.duplicated().any() or pred.asm_acc.eq("").any():
        raise RuntimeError("Duplicate/empty E2 prediction assembly")
    if not pred.prob_R.between(0,1).all() or not pred.prob_R.map(math.isfinite).all():
        raise RuntimeError("Invalid non-finite or out-of-range prediction probability")
    merged=meta.merge(pred, on="asm_acc", how="left",validate="one_to_one",indicator=True)
    if not merged._merge.eq("both").all() or len(merged)!=len(pred):
        raise RuntimeError("E2 frozen/prediction row identity mismatch")
    if not merged.frozen_label.eq(merged.pred_label).all():
        raise RuntimeError("Frozen label differs from FILE16 prediction label")
    merged=merged.drop(columns="_merge")
    merged["predicted_R"]=(merged.prob_R >= 0.5).astype(int)
    merged["wrong"]=(merged.predicted_R!=merged.frozen_label)
    if not test_mode:
        actual={"e2_n":len(merged),"e2_resistant":int(merged.frozen_label.sum()),
                "focus_n":int(merged.bioproject_acc.eq(FOCUS).sum()),
                "focus_resistant":int(merged.loc[merged.bioproject_acc.eq(FOCUS),"frozen_label"].sum()),
                "focus_errors":int(merged.loc[merged.bioproject_acc.eq(FOCUS),"wrong"].sum()),
                "full_errors":int(merged.wrong.sum())}
        if actual!=EXPECTED:
            raise RuntimeError(f"FAIL-CLOSED frozen E2 invariants: {actual} != {EXPECTED}")
    return merged, {"frozen":str(frozen_path), "frozen_sha256":sha(frozen_path),
                    "predictions":str(pred_path),"predictions_sha256":sha(pred_path)}


def tagname(tag):
    return tag.split("}")[-1].lower().replace("-","").replace("_","")


def child_text(row, *names):
    wanted={cn(x) for x in names}
    for el in row:
        if cn(tagname(el.tag)) in wanted:
            return " ".join(el.itertext()).strip()
    # Some NCBI layouts encode fields as attributes.
    for key,val in row.attrib.items():
        if cn(key) in wanted: return norm(val)
    return ""


def parse_biosample_xml(path):
    """Extract meropenem AST from both NCBI layouts.

    NCBI production EFetch uses Description/Comment/Table[@class="Antibiogram.1.0"]
    with Header/Cell and Body/Row/Cell; historic test fixtures may use
    Antibiogram/AntibiogramRow with field-name children.  Every row is
    linked ONLY by its parent BioSample accession; never fuzzy joined.
    """
    if path is None:
        return pd.DataFrame(), set()
    root = ET.parse(path).getroot()
    samples = [e for e in root.iter() if tagname(e.tag) == "biosample"]
    if not samples:
        raise RuntimeError("Not a BioSample XML file: no BioSample nodes")
    rows = []
    found = set()

    def append_row(acc, d):
        drug = norm(d.get("antibiotic", "")).lower()
        if drug not in ("meropenem", "mem"):
            return
        rows.append({
            "biosample_acc": acc,
            "ncbi_phenotype": norm(d.get("resistance phenotype", d.get("resistancephenotype", d.get("phenotype", "")))).lower(),
            "mic_sign": norm(d.get("measurement sign", d.get("measurementsign", d.get("sign", "")))),
            "mic_value": norm(d.get("measurement", d.get("measurementvalue", d.get("mic", "")))),
            "mic_units": norm(d.get("measurement units", d.get("measurementunits", d.get("units", "")))),
            "typing_method": norm(d.get("laboratory typing method", d.get("laboratorytypingmethod", d.get("method", "")))),
            "platform": norm(d.get("laboratory typing platform", d.get("laboratorytypingplatform", d.get("platform", "")))),
            "testing_standard": norm(d.get("testing standard", d.get("testingstandard", d.get("standard", "")))),
            "standard_year": norm(d.get("testing standard year", d.get("testingstandardyear", d.get("standardyear", "")))),
            "source": "NCBI_BIOSAMPLE_ANTIBIOGRAM"
        })

    for sample in samples:
        accession = norm(sample.get("accession") or sample.get("Accession")).upper()
        if not accession:
            for item in sample.iter():
                if tagname(item.tag) == "id" and "biosample" in norm(item.attrib.get("db")).lower():
                    accession = norm(item.text).upper()
                    break
        if not re.fullmatch(r"SAM[NED]\d+", accession):
            continue
        if accession in found:
            raise RuntimeError(f"Duplicate BioSample record in XML: {accession}")
        found.add(accession)

        # Production NCBI 'One Health Enteric' layout: Table/Caption/Header/Body.
        for table in sample.iter():
            if tagname(table.tag) != "table":
                continue
            klass = norm(table.attrib.get("class", "")).lower()
            caption = norm(table.findtext("./Caption", default="")).lower()
            if not (klass.startswith("antibiogram") or caption == "antibiogram"):
                continue
            header = table.find("./Header")
            body = table.find("./Body")
            if header is None or body is None:
                continue
            columns = [norm("".join(c.itertext())).lower() for c in header if tagname(c.tag)=="cell"]
            if not columns or len(set(columns)) != len(columns) or "antibiotic" not in columns:
                raise RuntimeError(f"Bad antibiogram header for {accession}")
            for row in body:
                if tagname(row.tag) != "row":
                    continue
                cells = [norm("".join(c.itertext())) for c in row if tagname(c.tag) == "cell"]
                if len(cells) != len(columns):
                    raise RuntimeError(f"BioSample {accession} antibiogram column mismatch: {len(cells)} != {len(columns)}")
                append_row(accession, dict(zip(columns, cells)))

        # Legacy explicit XML nodes supported by earlier versions/tests.
        for block in sample.iter():
            if tagname(block.tag) != "antibiogram":
                continue
            for row in block:
                if "row" not in tagname(row.tag):
                    continue
                d = {cn(tagname(field.tag)): norm("".join(field.itertext())) for field in row}
                append_row(accession, d)
    return pd.DataFrame(rows), found

def load_external_ast(path):
    """Optional external data with explicit accession, method and drug.

    Could represent authors' supplementary study data, BUT is not independent
    adjudication unless its collection and source are established externally.
    """
    if not path: return pd.DataFrame()
    src=pd.read_csv(path,dtype=str,keep_default_na=False)
    bio=pick(src,"biosample_acc","biosample","biosample_accession", "accession")
    drug=pick(src,"antibiotic","drug","antibiotic_name")
    phen=pick(src,"resistance_phenotype","phenotype","reported_phenotype","ast_phenotype")
    if not bio or not phen or not drug:
        raise RuntimeError("--external-ast-csv must contain exact BioSample, antibiotic, phenotype; no fuzzy joining")
    ss=src.loc[src[drug].map(lambda x:norm(x).lower() in ("meropenem","mem"))].copy()
    if not len(ss): raise RuntimeError("External AST CSV has no meropenem rows")
    select={"biosample_acc":bio,"external_phenotype":phen}
    for dest,names in {"external_mic_value":["measurement","mic","mic_value"],
                       "external_mic_sign":["measurement_sign","sign","mic_sign"],
                       "external_units":["measurement_units","units"],
                       "external_method":["laboratory_typing_method","method","ast_method"],
                       "external_standard":["testing_standard","standard"],
                       "external_standard_year":["testing_standard_year","standard_year"],
                       "external_evidence_reference":["evidence_reference","source_url","paper_supplement","source"]}.items():
        col=pick(ss,*names)
        if col:select[dest]=col
    out=ss[[*dict.fromkeys(select.values())]].copy().rename(columns={v:k for k,v in select.items()})
    out["biosample_acc"]=out.biosample_acc.map(lambda x:norm(x).upper())
    if out.biosample_acc.duplicated().any():
        raise RuntimeError("Multiple meropenem rows per BioSample in external AST; require disambiguation first")
    if not out.biosample_acc.str.fullmatch(r"SAM[NED][0-9]+",case=False).all():
        raise RuntimeError("Invalid external AST BioSample accession")
    return out


def mic_review_flag(row):
    sign=norm(row.get("mic_sign"))
    category=norm(row.get("ncbi_phenotype")).lower()
    units=norm(row.get("mic_units")).lower()
    method=norm(row.get("typing_method")).lower()
    if not norm(row.get("mic_value")) or not category: return "MISSING_DATA"
    if units not in ("mg/l","mg/litre","mg/liter","ug/ml","µg/ml") or "mic" not in method:
        return "REVIEW_UNIT_OR_METHOD"
    if sign in ("<","<=","≤") and category in ("resistant","r"):
        return "LOWER_CENSORED_MIC_REPORTED_RESISTANT_REVIEW"
    if sign in (">",">=","≥") and category in ("susceptible","s","sensitive"):
        return "UPPER_CENSORED_MIC_REPORTED_SUSCEPTIBLE_REVIEW"
    return "NO_SIMPLE_DIRECTION_FLAG__NOT_AST_ADJUDICATION"


def audit(root, frozen, predictions, xml, external, out, test_mode=False):
    e2,inputs=load_e2(root,frozen,predictions,test_mode)
    xfile=(choose(root,xml,[],required=True) if xml else root/DEFAULT_CACHE)
    if not xfile.is_file(): xfile=None
    src, xml_samples=parse_biosample_xml(xfile)
    if xfile: inputs["biosample_xml_sha256"]=sha(xfile);inputs["biosample_xml"]=str(xfile)
    ext_file=choose(root,external,[],required=True) if external else None
    ext=load_external_ast(ext_file)
    if ext_file:inputs["external_ast_sha256"]=sha(ext_file);inputs["external_ast"]=str(ext_file)

    # Preserve all NCBI meropenem observations, including ambiguous duplicate rows.
    if not src.empty:
        src=src.sort_values(["biosample_acc","ncbi_phenotype","mic_value"],kind="stable")
        counts=src.groupby("biosample_acc").size().rename("ncbi_row_count")
        first=src.drop_duplicates("biosample_acc",keep="first").set_index("biosample_acc")
        joined=e2.join(counts,on="biosample_acc").join(first,on="biosample_acc")
    else:
        joined=e2.copy();joined["ncbi_row_count"]=0
    if not ext.empty:
        joined=joined.merge(ext,on="biosample_acc",how="left",validate="one_to_one")
    for col in ["ncbi_phenotype","mic_sign","mic_value","mic_units","typing_method","platform","testing_standard","standard_year", "external_phenotype"]:
        if col not in joined:joined[col]=""
        joined[col]=joined[col].fillna("").map(norm)
    joined["ncbi_row_count"]=joined.ncbi_row_count.fillna(0).astype(int)
    joined["ncbi_sample_present_in_xml"]=joined.biosample_acc.isin(xml_samples)
    joined["ncbi_phenotype_binary"] = joined.ncbi_phenotype.str.lower().map({"resistant":1,"susceptible":0,"sensitive":0})
    joined["evidence_label_relation"] = "NOT_COMPARABLE_OR_MISSING"
    eligible=joined.ncbi_row_count.eq(1) & joined.ncbi_phenotype_binary.notna()
    joined.loc[eligible & joined.ncbi_phenotype_binary.eq(joined.frozen_label),"evidence_label_relation"]="NCBI_SUBMISSION_AGREES_WITH_FROZEN_NOT_INDEPENDENT"
    joined.loc[eligible & joined.ncbi_phenotype_binary.ne(joined.frozen_label),"evidence_label_relation"]="NCBI_SUBMISSION_CONFLICT_REVIEW"
    joined.loc[joined.ncbi_row_count.gt(1),"evidence_label_relation"]="MULTIPLE_MEROPENEM_AST_ROWS_REVIEW"
    joined["mic_direction_review_flag"]=joined.apply(mic_review_flag,axis=1)
    joined["external_evidence_relation"]="NOT_PROVIDED_OR_UNMATCHED"
    if "external_phenotype" in joined:
        extlabel=joined.external_phenotype.str.lower().map({"resistant":1,"susceptible":0,"sensitive":0})
        joined.loc[extlabel.notna() & extlabel.eq(joined.frozen_label),"external_evidence_relation"]="EXTERNAL_RECORD_AGREES__NOT_YET_INDEPENDENT_ADJUDICATION"
        joined.loc[extlabel.notna() & extlabel.ne(joined.frozen_label),"external_evidence_relation"]="EXTERNAL_RECORD_CONFLICT_REVIEW"
    joined["audit_priority"]=(joined.bioproject_acc.eq(FOCUS).astype(int)*100+
                             joined.wrong.astype(int)*30+
                             joined.evidence_label_relation.str.contains("CONFLICT|MULTIPLE").astype(int)*35+
                             joined.mic_direction_review_flag.str.contains("CENSORED").astype(int)*25+
                             joined.external_evidence_relation.str.contains("CONFLICT").astype(int)*40+
                             (~joined.ncbi_sample_present_in_xml).astype(int)*5)
    joined["biosample_url"]="https://www.ncbi.nlm.nih.gov/biosample/"+joined.biosample_acc
    focus=joined.loc[joined.bioproject_acc.eq(FOCUS)].copy()
    summary={
        "file":"FILE19", "version":VERSION,"generated_utc":utc(),
        "status":"POST_HOC_AST_PROVENANCE_TRIAGE__NOT_LABORATORY_ADJUDICATION",
        "primary_e2_unchanged":True,"focus_bioproject":FOCUS,
        "input_files":inputs,
        "frozen_guardrails":{"e2_n":len(joined),"e2_R":int(joined.frozen_label.sum()),
            "e2_model_errors":int(joined.wrong.sum()),"focus_n":len(focus),
            "focus_R":int(focus.frozen_label.sum()),"focus_errors":int(focus.wrong.sum())},
        "evidence":{"biosample_xml_provided":bool(xfile),"xml_biosamples_total":len(xml_samples),
            "focus_biosamples_seen_in_xml":int(focus.ncbi_sample_present_in_xml.sum()),
            "focus_records_with_one_meropenem_AST_row":int(focus.ncbi_row_count.eq(1).sum()),
            "focus_records_with_multiple_AST_rows":int(focus.ncbi_row_count.gt(1).sum()),
            "focus_missing_meropenem_AST":int(focus.ncbi_row_count.eq(0).sum()),
            "focus_ncbi_phenotype_agrees":int(focus.evidence_label_relation.eq("NCBI_SUBMISSION_AGREES_WITH_FROZEN_NOT_INDEPENDENT").sum()),
            "focus_ncbi_phenotype_conflicts":int(focus.evidence_label_relation.eq("NCBI_SUBMISSION_CONFLICT_REVIEW").sum()),
            "focus_mic_review_flags":dict(Counter(focus.mic_direction_review_flag)),
            "external_ast_csv_provided":bool(ext_file),
            "focus_external_ast_label_agrees":int(focus.external_evidence_relation.eq("EXTERNAL_RECORD_AGREES__NOT_YET_INDEPENDENT_ADJUDICATION").sum()),
            "focus_external_ast_label_conflicts":int(focus.external_evidence_relation.eq("EXTERNAL_RECORD_CONFLICT_REVIEW").sum())},
        "constraints":{"no_training":True,"no_label_changes":True,"no_breakpoint_inference":True,
            "no_primary_cohort_changes":True,"no_automatic_AST_validation":True,
            "never_infer_phenotype_from_low_or_high_MIC_without_standards":True},
        "scientific_interpretation":"This audit measures record-level concordance and flags potentially suspicious source entries. NCBI BioSample values may originate from the same submitted AST as the frozen labels. Agreements are not independent laboratory confirmation."
    }
    # Dataset-level polarity diagnostics. DO NOT use for automatic label correction.
    comparable = focus.loc[focus.ncbi_row_count.eq(1) & focus.ncbi_phenotype_binary.notna()].copy()
    cross = {
        f"frozen_{int(y)}_ncbi_{int(z)}": int(n)
        for (y,z),n in comparable.groupby(["frozen_label","ncbi_phenotype_binary"]).size().items()
    }
    summary["evidence"]["focus_ncbi_resistant_count"] = int(comparable.ncbi_phenotype_binary.eq(1).sum())
    summary["evidence"]["focus_ncbi_susceptible_count"] = int(comparable.ncbi_phenotype_binary.eq(0).sum())
    summary["evidence"]["focus_frozen_vs_ncbi_cross_table"] = cross
    summary["evidence"]["focus_all_comparable_labels_opposite"] = bool(
        len(comparable) == len(focus) and
        comparable.ncbi_phenotype_binary.ne(comparable.frozen_label).all()
    )
    summary["evidence"]["focus_model_predictions_match_ncbi_count"] = int(
        comparable.predicted_R.eq(comparable.ncbi_phenotype_binary).sum()
    )
    summary["evidence"]["focus_model_predictions_disagree_ncbi_count"] = int(
        comparable.predicted_R.ne(comparable.ncbi_phenotype_binary).sum()
    )
    summary["decision"] = ("EVIDENCE_INCOMPLETE__FETCH_OR_PROVIDE_MEROPENEM_AST"
        if not xfile or summary["evidence"]["focus_records_with_one_meropenem_AST_row"] < len(focus)
        else "REVIEW_DOCUMENTED_AST_PROVENANCE__NO_AUTOMATIC_LABEL_ADJUDICATION")
    if summary["evidence"]["focus_all_comparable_labels_opposite"]:
        summary["decision"] = "CRITICAL_COMPLETE_LABEL_POLARITY_DISCORDANCE__HALT_MANUSCRIPT_CLAIMS__VERIFY_SOURCE_MAPPING"
    elif summary["evidence"]["focus_ncbi_phenotype_conflicts"]:
        summary["decision"] = "LABEL_DISCORDANCE_REQUIRES_ACCESSION_REVIEW__NO_AUTOMATIC_RELABEL"
    outdir_new(out)
    joined=joined.sort_values(["audit_priority","bioproject_acc","biosample_acc"],ascending=[False,True,True])
    focus=joined.loc[joined.bioproject_acc.eq(FOCUS)].copy()
    outputs={
        "file19_all_E2_ast_crosswalk.csv":joined,
        "file19_focus_PRJNA1322038_ast_triage.csv":focus,
    }
    if not src.empty: outputs["file19_ncbi_meropenem_rows.csv"]=src
    for name,d in outputs.items():d.to_csv(out/name,index=False)
    atomic_json(out/"file19_summary.json",summary)
    (out/"file19_reporting_guardrails.md").write_text(
        "# FILE19 — AST provenance guardrails\n\n"
        "- E2 full-cohort (525 genomes), frozen labels and scores are unchanged.\n"
        "- NCBI BioSample antibiogram can be the source of the original deposited AST, NOT independent confirmation.\n"
        "- Low censored MIC labeled resistant or high censored MIC labeled susceptible is a **review flag**, NOT a relabel.\n"
        "- No numeric breakpoint is assumed without species, test year, standard, dose and laboratory method.\n"
        "- Paper supplementary VITEK 2 / BMD, if available, must be accession matched and its measurement provenance established.\n"
        "- Missing AST cannot be reported as verified phenotype. No conclusions about AST mislabelling are supported yet.\n"
        "- Post-hoc source selection cannot redefine primary E2 results or prove cause of external AUROC decline.\n",
        encoding="utf-8")
    manifests={name:sha(out/name) for name in [*outputs,"file19_summary.json","file19_reporting_guardrails.md"]}
    atomic_json(out/"file19_output_manifest_sha256.json",{"version":VERSION,"outputs_sha256":manifests})
    print(json.dumps({"status":summary["decision"],"focus_n":len(focus),
                      "focus_AST_unique":summary["evidence"]["focus_records_with_one_meropenem_AST_row"],
                      "summary":str(out/"file19_summary.json")},indent=2))
    return summary


def ncbi_request(endpoint, params, retries=3):
    qs=urllib.parse.urlencode({**params,"tool":"AMR_Genome_File19_Provenance"})
    url=f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}?{qs}"
    last=None
    for attempt in range(retries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"AMR-Genome-ML FILE19 provenance research/1.0 (NCBI EUtils)"})
            with urllib.request.urlopen(req, timeout=60) as rsp: return rsp.read()
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as err:
            last=err;time.sleep(2**attempt)
    raise RuntimeError(f"NCBI request failed: {endpoint}: {last}")


def ncbi_esearch_ids(database, term, email, retmax=3000):
    """Retrieve UIDs; distinguish genuine zero hits from malformed/error responses."""
    obj = json.loads(ncbi_request("esearch.fcgi", {
        "db":database, "term":term, "retmode":"json", "retmax":retmax,
        "email":email}))
    result = obj.get("esearchresult")
    if not isinstance(result, dict) or "idlist" not in result:
        raise RuntimeError("Malformed NCBI ESearch JSON: no esearchresult.idlist")
    if result.get("errorlist") or result.get("ERROR"):
        raise RuntimeError(f"NCBI ESearch reported an error: {result.get('errorlist',result.get('ERROR'))}")
    try:
        total = int(result.get("count",len(result["idlist"])))
    except (ValueError, TypeError):
        raise RuntimeError("Malformed NCBI ESearch count")
    if total > retmax:
        raise RuntimeError(f"NCBI ESearch returned {total} UIDs (>{retmax}); refuse truncated retrieval")
    return [str(v) for v in result["idlist"]]


def ncbi_bioproject_to_biosample_ids(email):
    """Link BioProject to BioSamples as documented by NCBI ELink.

    Bioproject ID 1322038 is a *known* integer identifier, but resolving the
    accession first is preferred to avoid assuming numeric-ID identity.
    """
    pids = ncbi_esearch_ids("bioproject", FOCUS, email, 50)
    if not pids:
        pids = ncbi_esearch_ids("bioproject", f"{FOCUS}[All Fields]", email, 50)
    if not pids:
        return [], "BIOPROJECT_ESearch_NO_UID"
    result = ncbi_request("elink.fcgi", {
        "dbfrom":"bioproject", "db":"biosample", "id":",".join(pids),
        "retmode":"xml", "email":email})
    root = ET.fromstring(result)
    ids=[]
    for ls in root.iter():
        if tagname(ls.tag) != "linksetdb":
            continue
        linked_db = next((norm(child.text).lower() for child in ls
                          if tagname(child.tag)=="linkname"), "")
        if linked_db and "biosample" not in linked_db:
            continue
        for child in ls.iter():
            if tagname(child.tag)=="link":
                for sub in child:
                    if tagname(sub.tag)=="id" and norm(sub.text).isdigit():
                        ids.append(norm(sub.text))
    return list(dict.fromkeys(ids)), "BIOPROJECT_ELINK"


def fetch_ncbi(root, frozen, predictions, email, cache_dir, test_mode=False):
    if not email or "@" not in email:
        raise RuntimeError("NCBI requests require --email YOUR_CONTACT_EMAIL")
    e2,inputs=load_e2(root,frozen,predictions,test_mode)
    targets=set(e2.loc[e2.bioproject_acc.eq(FOCUS),"biosample_acc"])
    if not targets: raise RuntimeError("Focus project absent in frozen E2")
    if cache_dir.exists():
        raise FileExistsError("NCBI cache exists; do not overwrite: "+str(cache_dir))

    # v1.0.0 searched the project accession as a BioSample *field*, which need
    # not be indexed the same way as links between the two NCBI databases.
    # Direct BioSample accession queries remove that fragile dependency.
    target_list=sorted(targets)
    candidate_uids=set()
    searches=[]
    for i in range(0,len(target_list),15):
        batch=target_list[i:i+15]
        # Exact accession terms, never a fuzzy isolate-name match.
        query=" OR ".join(f"{acc}[Accession]" for acc in batch)
        try:
            batch_ids=ncbi_esearch_ids("biosample",query,email,retmax=300)
            search_error=""
        except (RuntimeError, ValueError) as exc:
            batch_ids=[]
            search_error=f"{type(exc).__name__}: {str(exc)[:200]}"
        candidate_uids.update(batch_ids)
        searches.append({"method":"TARGET_BIOSAMPLE_ACCESSION_OR",
                         "batch_index":i//15, "target_accessions":len(batch),
                         "uids_found":len(batch_ids), "error":search_error})
        time.sleep(.38)

    # A direct one-accession lookup distinguishes unsupported OR/field parsing
    # from truly absent accession records, without silently changing IDs.
    if not candidate_uids:
        query=target_list[0]
        try:
            ids=ncbi_esearch_ids("biosample", query, email,retmax=10)
            probe_error=""
        except (RuntimeError, ValueError) as exc:
            ids=[]
            probe_error=f"{type(exc).__name__}: {str(exc)[:200]}"
        candidate_uids.update(ids)
        searches.append({"method":"SINGLE_ACCESSION_UNFIELDED_PROBE",
                         "accession":query,"uids_found":len(ids),"error":probe_error})
        time.sleep(.38)

    # Secondary route via the stable BioProject->BioSample NCBI ELink.
    # In the absence of complete direct IDs, ELink recovers any BioSamples
    # without relying on the '[BioProject]' BioSample search index.
    linked_uids=[]
    linked_status="SKIPPED_DIRECT_UID_COUNT_AT_LEAST_TARGET_COUNT"
    if len(candidate_uids)<len(targets):
        try:
            linked_uids,linked_status=ncbi_bioproject_to_biosample_ids(email)
            candidate_uids.update(linked_uids)
        except (RuntimeError, ET.ParseError, ValueError) as exc:
            linked_status=f"UNAVAILABLE_{type(exc).__name__}: {str(exc)[:300]}"
    if not candidate_uids:
        raise RuntimeError(
            f"NCBI returned no UIDs for {len(targets)} exact BioSample accession targets; "
            f"BioProject link status={linked_status}. No frozen inputs or prior outputs modified")

    # UIDs identify download candidates; acceptance is by exact BioSample
    # accession. Linked BioProject records that are not among the 175 targets
    # are never incorporated into the source audit.
    chunks=[];seen=set();other_count=0; batches_count=0
    if any(not x.isdigit() for x in candidate_uids):
        raise RuntimeError("NCBI returned non-numeric UIDs; refuse ambiguous EFetch")
    uidlist=sorted(candidate_uids,key=int)
    for i in range(0,len(uidlist),60):
        blob=ncbi_request("efetch.fcgi",{
            "db":"biosample", "id":",".join(uidlist[i:i+60]),
            "retmode":"xml", "email":email})
        node=ET.fromstring(blob)
        batch=[e for e in node.iter() if tagname(e.tag)=="biosample"]
        if not batch:
            raise RuntimeError("NCBI EFetch returned no BioSample records for nonempty UID batch")
        for b in batch:
            key=norm(b.get("accession") or b.get("Accession")).upper()
            if key in targets and key not in seen:
                seen.add(key);chunks.append(b)
            elif key not in targets:
                other_count+=1
        batches_count+=1
        time.sleep(.38)

    if not chunks:
        raise RuntimeError("NCBI EFetch returned BioSamples but no exact frozen focus accessions; no cache committed")
    xmlroot=ET.Element("BioSampleSet")
    for b in chunks:xmlroot.append(b)
    # Cache is committed only after successful parsing and exact matching.
    with tempfile.TemporaryDirectory(prefix="file19_ncbi_xml_") as temp:
        tpath=Path(temp)/"BioSampleSet.xml"
        ET.ElementTree(xmlroot).write(tpath,encoding="utf-8",xml_declaration=True)
        rows,found=parse_biosample_xml(tpath)
        if found != seen:
            raise RuntimeError("Internal XML parser did not preserve exact BioSample identities")
        cache_dir.mkdir(parents=True,exist_ok=False)
        xml_path=cache_dir/f"{FOCUS}_biosample_efetch.xml"
        xml_path.write_bytes(tpath.read_bytes())
    summary={
        "file":"FILE19_NCBI_CACHE", "version":VERSION,
        "timestamp_utc":utc(),
        "retrieval_strategy":"EXACT_FROZEN_BIOSAMPLE_ACCESSIONS__BIOPROJECT_ELINK_FALLBACK",
        "accession_batch_queries":searches,
        "bioproject_elink_status":linked_status,
        "bioproject_elink_uid_count":len(linked_uids),
        "unique_candidate_uids":len(candidate_uids),
        "efetch_batches":batches_count,
        "nonfocus_records_ignored":other_count,
        "fetched_unique_target_biosamples":len(found),
        "focus_biosamples_in_frozen_E2":len(targets),
        "target_biosamples_matched":len(targets & found),
        "target_biosamples_unmatched":sorted(targets-found),
        "fetched_meropenem_antibiogram_rows":len(rows),
        "complete_biosample_identity_coverage":found==targets,
        "cache_xml_sha256":sha(xml_path),
        "frozen_inputs":inputs,
        "note":"Exact accession match proves record linkage, not independent AST reliability. Missing AST remains incomplete evidence."
    }
    atomic_json(cache_dir/"file19_ncbi_fetch_manifest.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in
          ("target_biosamples_unmatched","accession_batch_queries")},indent=2))
    print(f"[FILE19] Exact accession coverage: {len(found)}/{len(targets)}; meropenem AST rows: {len(rows)}")


def discover(root,frozen,predictions):
    result={"file":"FILE19","version":VERSION,"project_root":str(root),"timestamp":utc(),
        "inputs":{},"suggested_command":"python scripts/file19_ast_provenance_audit.py fetch --project-root . --email your@email.com"}
    for key,opt,cands in [("E2_frozen",frozen,FROZEN_CANDIDATES),
                           ("FILE16_predictions",predictions,PRED_CANDIDATES),
                           ("NCBI_cached_XML",None,[DEFAULT_CACHE]),
                           ("FILE16D_triage",None,[FROZEN_CANDIDATES[1]])]:
        p=choose(root,opt,cands,required=False)
        result["inputs"][key]={"exists":p is not None,"path":str(p) if p else "",
                                  "bytes":p.stat().st_size if p else None}
    return result


def self_test():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="amr_file19_test_") as temp:
        root=Path(temp)
        data=root/"data";data.mkdir()
        meta=pd.DataFrame([
            {"asm_acc":"GCA_000000001.1","biosample_acc":"SAMN123451","bioproject_acc":FOCUS,"label_binary":"resistant"},
            {"asm_acc":"GCA_000000002.1","biosample_acc":"SAMN123452","bioproject_acc":FOCUS,"label_binary":"susceptible"},
            {"asm_acc":"GCA_000000003.1","biosample_acc":"SAMN123453","bioproject_acc":"PRJNAOTHER","label_binary":"susceptible"}])
        meta.to_csv(data/"frozen.csv",index=False)
        pp=pd.DataFrame([{"genome_id":a,"y_true":b,"prob_resistance":c,"cohort":"E2"}
             for a,b,c in zip(meta.asm_acc,[1,0,0],[.1,.9,.15])]);pp.to_csv(data/"pred.csv",index=False)
        xml='''<?xml version="1.0"?><BioSampleSet>
        <BioSample accession="SAMN123451"><Antibiogram><AntibiogramRow>
        <Antibiotic>meropenem</Antibiotic><ResistancePhenotype>resistant</ResistancePhenotype>
        <MeasurementSign>&lt;</MeasurementSign><Measurement>0.25</Measurement>
        <MeasurementUnits>mg/L</MeasurementUnits><LaboratoryTypingMethod>MIC</LaboratoryTypingMethod>
        <LaboratoryTypingPlatform>Vitek</LaboratoryTypingPlatform>
        <TestingStandard>EUCAST</TestingStandard></AntibiogramRow></Antibiogram></BioSample>
        <BioSample accession="SAMN123452"><Antibiogram><AntibiogramRow>
        <Antibiotic>meropenem</Antibiotic><ResistancePhenotype>resistant</ResistancePhenotype>
        <MeasurementSign>==</MeasurementSign><Measurement>4</Measurement>
        <MeasurementUnits>mg/L</MeasurementUnits><LaboratoryTypingMethod>MIC</LaboratoryTypingMethod>
        <TestingStandard>EUCAST</TestingStandard></AntibiogramRow></Antibiogram></BioSample>
        </BioSampleSet>'''
        (data/"bio.xml").write_text(xml,encoding="utf-8")
        smry=audit(root,"data/frozen.csv","data/pred.csv",str(data/"bio.xml"),None,
                   root/"results/file19_ast_provenance_audit_v1",test_mode=True)
        assert smry["evidence"]["focus_ncbi_phenotype_conflicts"]==1
        assert smry["evidence"]["focus_ncbi_phenotype_agrees"]==1
        assert smry["evidence"]["focus_mic_review_flags"].get("LOWER_CENSORED_MIC_REPORTED_RESISTANT_REVIEW")==1
        assert smry["decision"]=="LABEL_DISCORDANCE_REQUIRES_ACCESSION_REVIEW__NO_AUTOMATIC_RELABEL"
        try:
            audit(root,"data/frozen.csv","data/pred.csv",str(data/"bio.xml"),None,
                  root/"results/file19_ast_provenance_audit_v1",test_mode=True)
            raise AssertionError("Overwriting output was not blocked")
        except FileExistsError:pass
        # Regression: NCBI actual BioSample antibiogram is an inline Table (not an Antibiogram tag).
        nested = (data/"real_layout.xml")
        nested.write_text('''''''''<?xml version="1.0"?><BioSampleSet><BioSample accession="SAMN99999999"><Description><Comment>
        <Table class="Antibiogram.1.0"><Caption>Antibiogram</Caption>
        <Header><Cell>Antibiotic</Cell><Cell>Resistance phenotype</Cell><Cell>Measurement sign</Cell>
        <Cell>Measurement</Cell><Cell>Measurement units</Cell><Cell>Laboratory typing method</Cell>
        <Cell>Laboratory typing platform</Cell><Cell>Testing standard</Cell></Header>
        <Body><Row><Cell>meropenem</Cell><Cell>susceptible</Cell><Cell>&lt;</Cell>
        <Cell>0.25</Cell><Cell>mg/L</Cell><Cell>MIC</Cell><Cell>Vitek</Cell><Cell>EUCAST</Cell></Row></Body>
        </Table></Comment></Description></BioSample></BioSampleSet>''''''''',encoding="utf-8")
        nested_rows, nested_found = parse_biosample_xml(nested)
        assert nested_found == {"SAMN99999999"} and len(nested_rows) == 1
        assert nested_rows.iloc[0].ncbi_phenotype == "susceptible"
        # Incomplete evidence must never be reported as verified.
        s2=audit(root,"data/frozen.csv","data/pred.csv",None,None,
             root/"results/file19_no_ast",test_mode=True)
        assert s2["decision"].startswith("EVIDENCE_INCOMPLETE")
        # Duplicate labels cannot silently change the frozen analysis.
        bad=meta.copy();bad.loc[1,"asm_acc"]=bad.loc[0,"asm_acc"]
        bad.to_csv(data/"bad.csv",index=False)
        try:
            load_e2(root,"data/bad.csv","data/pred.csv",test_mode=True)
            raise AssertionError("Duplicate identifiers accepted")
        except RuntimeError:pass
        print("[FILE19 SELF-TEST] PASS: actual NCBI nested-table antibiogram, legacy XML, exact joins, label conflict, immutable outputs")


def main():
    p=argparse.ArgumentParser(description="FILE19 post-hoc AST provenance audit. Never recodes phenotypes.")
    sp=p.add_subparsers(dest="command",required=True)
    for command in ("discover","fetch","audit","self-test"):
        sub=sp.add_parser(command)
        if command!="self-test":
            sub.add_argument("--project-root",default=".")
            sub.add_argument("--frozen-e2",default="")
            sub.add_argument("--predictions",default="")
        if command=="fetch":
            sub.add_argument("--email",required=True)
            sub.add_argument("--cache-dir",default="results/file19_ncbi_evidence_cache_v1_0_1")
        if command=="audit":
            sub.add_argument("--biosample-xml",default="")
            sub.add_argument("--external-ast-csv",default="",help="optional exact-BioSample paper AST table, with provenance")
            sub.add_argument("--outdir",default=DEFAULT_OUTPUT)
    args=p.parse_args()
    if args.command=="self-test":return self_test()
    root=Path(args.project_root).resolve()
    if args.command=="discover":
        report=discover(root,args.frozen_e2,args.predictions)
        dst=root/"results/file19_discovery_v1_0_1"
        if not dst.exists():
            dst.mkdir(parents=True,exist_ok=False)
            atomic_json(dst/"file19_discovered_inputs.json",report)
        print(json.dumps({**report,"saved_initial_discovery_at":str(dst/"file19_discovered_inputs.json")},indent=2));return
    if args.command=="fetch":
        fetch_ncbi(root,args.frozen_e2,args.predictions,args.email,root/args.cache_dir);return
    if args.command=="audit":
        audit(root,args.frozen_e2,args.predictions,args.biosample_xml,args.external_ast_csv,root/args.outdir)

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print(f"[FILE19 FAIL-CLOSED] {type(exc).__name__}: {exc}",file=sys.stderr)
        raise SystemExit(2)
