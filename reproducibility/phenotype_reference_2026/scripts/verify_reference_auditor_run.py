#!/usr/bin/env python3
"""Independently check an auditor run against raw XML, Excel and CDC history HTML."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from html import unescape
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

from openpyxl import load_workbook
from verify_cdc_reference_audit import measurement, status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-dir",type=Path,required=True)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--code-lock",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    cases=json.loads((args.inputs/"auditor_cases.json").read_text())["cases"]
    results=json.loads((args.audit/"case_results.json").read_text());report=json.loads((args.audit/"report.json").read_text())
    lock=json.loads(args.code_lock.read_text())
    if report["auditor_sha256"]!=lock["auditor_sha256"]:
        raise ValueError("Auditor changed from the pre-download lock")
    if hashlib.sha256((args.inputs/"auditor_cases.json").read_bytes()).hexdigest()!=report["input_sha256"]:
        raise ValueError("Input changed since audit")
    included={c["biosample_acc"] for c in cases}
    cdc=defaultdict(list);ncbi=defaultdict(list)
    for src in json.loads((args.panel_dir/"manifest.json").read_text())["results"]:
        raw=(args.panel_dir/src["export"]["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=src["export"]["sha256"]:raise ValueError("Export hash mismatch")
        book=load_workbook(BytesIO(raw),read_only=True,data_only=True);values=book.active.values;header=next(values)
        for values in values:
            row={k:str(v).strip() if v is not None else "" for k,v in zip(header,values)}
            sample=row["BioSampleAccessionNum"]
            if sample not in included:continue
            for drug in ["Meropenem","Imipenem","Ertapenem"]:
                if row.get(drug+" MIC","") or row.get(drug+" INT",""):
                    cdc[(sample,drug.lower())].append((row.get(drug+" MIC",""),row.get(drug+" INT","")))
        book.close()
    label={"susceptible":"S","intermediate":"I","resistant":"R","not defined":"ND"}
    manifest=json.loads((args.inputs/"biosamples/manifest.json").read_text())
    for src in manifest["responses"]:
        raw=(args.inputs/"biosamples"/src["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=src["sha256"]:raise ValueError("XML hash mismatch")
        for sample in ET.fromstring(raw).findall("BioSample"):
            for table in sample.findall(".//Table"):
                if not table.attrib.get("class","").startswith("Antibiogram"):continue
                header=[(x.text or "").strip() for x in table.findall("Header/Cell")]
                for row in table.findall("Body/Row"):
                    d=dict(zip(header,[(x.text or "").strip() for x in row.findall("Cell")]))
                    if d["Antibiotic"].casefold() in {"meropenem","imipenem","ertapenem"}:
                        ncbi[(sample.attrib["accession"],d["Antibiotic"].casefold())].append((d["Measurement sign"]+d["Measurement"],d["Measurement units"],label.get(d["Resistance phenotype"].casefold(),d["Resistance phenotype"])))
    histories=defaultdict(list)
    manifest=json.loads((args.inputs/"histories/manifest.json").read_text())
    for src in manifest["results"]:
        raw=(args.inputs/"histories"/src["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=src["sha256"]:raise ValueError("History HTML hash mismatch")
        html=raw.decode();anchor=re.search(r'<span[^>]*id="MainContent_lblSeqAcc"[^>]*>(.*?)</span>',html,re.S)
        samples=set(re.findall(r"SAM[NED][A-Z]?\d+",anchor.group(1))) if anchor else set()
        if len(samples)!=1:raise ValueError("Primary current BioSample anchor ambiguous")
        sample=next(iter(samples))
        for date,description in re.findall(r">(\d\d/\d\d/\d\d)</td>\s*<td[^>]*>(.*?)</td>",html,re.S):
            description=" ".join(unescape(re.sub(r"<[^>]+>","",description)).split())
            match=re.fullmatch(r"MIC was updated with a new value for (.+): from '(.*)' to '(.*)'",description)
            if match:
                histories[(sample,match.group(1).casefold())].append((date,match.group(2),match.group(3)))
    expected_keys=set(cdc)
    if expected_keys!={(c["biosample_acc"],c["antibiotic"]) for c in cases} or len(cases)!=len(expected_keys):
        raise ValueError("Case eligibility/key mismatch")
    by_key={(r["biosample_acc"],r["antibiotic"]):r for r in results};counts=Counter();links=0
    for c in cases:
        key=c["biosample_acc"],c["antibiotic"]
        if Counter(cdc[key])!=Counter((o["MIC_raw"],o["category_raw"]) for o in c["CDC_observations"]):raise ValueError("CDC case values differ from Excel")
        if Counter(ncbi[key])!=Counter((o["MIC_raw"],o["units_raw"],o["category_raw"]) for o in c["NCBI_observations"]):raise ValueError("NCBI case values differ from XML")
        r=by_key[key]
        if not cdc[key] or not ncbi[key]:
            expected="INSUFFICIENT_PAIRED_REFERENCE"
        else:
            current=set(cdc[key]);old=set(ncbi[key])
            if len(current)!=1 or len(old)!=1:
                expected="AMBIGUOUS_MULTIPLE_REFERENCE_STATES"
            else:
                b,bc=next(iter(current));a,unit,ac=next(iter(old))
                try:
                    mic_relation=status(a,b)
                except (AttributeError,ValueError):
                    expected="UNSUPPORTED_OR_MISSING_MIC_OR_UNIT"
                else:
                    if unicodedata.normalize('NFKC',unit).casefold() not in {'mg/l','ug/ml','μg/ml'}:
                        expected="UNSUPPORTED_OR_MISSING_MIC_OR_UNIT"
                    else:
                        linked=False
                        for when,x,y in histories[key]:
                            try:
                                if datetime.strptime(when,'%m/%d/%y').date()<=datetime.fromisoformat(c['as_of_date']).date() and measurement(x)!=measurement(y) and measurement(x)==measurement(a) and measurement(y)==measurement(b):linked=True
                            except (ValueError,AttributeError):pass
                        if linked:expected="DOCUMENTED_MIC_UPDATE_LINK";links+=1
                        elif ac not in {'S','I','R'} or bc not in {'S','I','R'}:expected="CATEGORY_UNAVAILABLE_CAUSE_UNRESOLVED"
                        elif mic_relation=="DISJOINT_REPORTED_MIC_INTERVALS":expected="DISJOINT_MIC_WITH_CATEGORY_CHANGE" if ac!=bc else "DISJOINT_MIC_WITH_SAME_CATEGORY"
                        elif ac!=bc:expected="CATEGORY_ONLY_REFERENCE_DIFFERENCE"
                        elif mic_relation=="DIFFERENT_REPORTING_INTERVALS_OVERLAP":expected="OVERLAPPING_REPORTING_INTERVALS"
                        else:expected="SAME_REPORTED_REFERENCE"
        if expected!=r['evidence_state']:raise ValueError(f"Independent evidence state differs for {key}: {expected} vs {r['evidence_state']}")
        counts[expected]+=1
    result={"status":"PASS","method":"Raw openpyxl/XML join and independent Decimal predicate/regex-history reconstruction","case_keys":len(cases),"unique_BioSamples":len(included),"documented_update_links":links,"evidence_states":dict(counts),"pre_holdout_auditor_hash_unchanged":True}
    with args.out.open('x') as h:json.dump(result,h,indent=2);h.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
