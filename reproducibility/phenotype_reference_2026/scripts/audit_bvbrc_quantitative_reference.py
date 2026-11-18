#!/usr/bin/env python3
"""Screen archived/current MIC state sets and version-conditional compatibility."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from compare_archived_bvbrc_ast import validated_current, phenotype
from reference_evidence_auditor import parse_mic, intervals_overlap


def read_measurement(row):
    unit=unicodedata.normalize("NFKC",str(row.get("measurement_unit", ""))).strip().casefold()
    method=str(row.get("laboratory_typing_method", "")).casefold()
    if unit not in {"mg/l","ug/ml","μg/ml"} or "disk" in method or "kirby" in method:
        return None,"UNSUPPORTED_UNIT_OR_DISK_METHOD"
    raw=str(row.get("measurement", "")).strip()
    sign=str(row.get("measurement_sign", "")).strip()
    value=str(row.get("measurement_value", "")).strip()
    a=parse_mic(raw)
    explicit_parts=parse_mic(sign+value) if sign in {"=","==",">",">=","<","<="} else None
    bare=re.fullmatch(r"[0-9]+(?:\.[0-9]+)?",raw)
    if a is not None and explicit_parts is not None:
        if bare and value and parse_mic(value)==a:
            return explicit_parts,"MATCHING_VALUE_WITH_EXPLICIT_SIGN"
        if a!=explicit_parts:
            return None,"CONFLICTING_EXPLICIT_MIC_ENCODINGS"
    if a is not None:
        if value and parse_mic(value) is not None:
            va=parse_mic(value)[0]
            if (a[0] if a[0]>0 else a[1])!=va:
                return None,"CONFLICTING_RAW_AND_NUMERIC_VALUE"
        return a,"RAW_MEASUREMENT"
    if explicit_parts is not None:
        return explicit_parts,"EXPLICIT_SIGN_AND_VALUE"
    return None,"MISSING_OR_UNSUPPORTED_MIC_ENCODING"


def serial(interval):
    return "|".join(str(x.normalize()) if hasattr(x,"normalize") else str(x) for x in interval)


def breakpoint_categories(mic,s,r,guideline):
    from decimal import Decimal
    s,r=Decimal(s),Decimal(r)
    infinity=Decimal("Infinity")
    # CLSI R >= cutoff; EUCAST R > cutoff, as documented by pinned AMR rules.
    regions={"S":(Decimal(0),s,False,True),
             "I":(s,r,False,guideline=="EUCAST"),"R":(r,infinity,guideline!="EUCAST",False)}
    return {name for name,region in regions.items() if intervals_overlap(mic,region)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archived",type=Path,required=True)
    p.add_argument("--current-root",type=Path,required=True)
    p.add_argument("--breakpoints",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    old=defaultdict(list)
    with args.archived.open(newline="") as h:
        for row in csv.DictReader(h):
            old[(row["screen_species_annotation"],row["screen_drug"],row["genome_id"])].append(row)
    bp=pd.read_csv(args.breakpoints,sep="\t",dtype=str,keep_default_na=False)
    selected=bp[(bp.type=="human")&(bp.host=="human")&(bp.method=="MIC")&(bp.site=="")&(bp.breakpoint_S!="")&(bp.breakpoint_R!="")]
    bp_lookup=defaultdict(list)
    for row in selected.to_dict("records"):
        bp_lookup[(row["guideline"],row["mo"],row["ab"])].append(row)
    pair_rows,conditional_rows,encoding_counts=[],[],Counter()
    def states(records,source,species,drug,gid):
        result=set()
        for row in records:
            mic,status=read_measurement(row)
            encoding_counts[(source,status)]+=1
            if mic is None:
                continue
            category=phenotype(row.get("resistant_phenotype"))
            result.add((serial(mic),category or "MISSING_CATEGORY"))
            standard=str(row.get("testing_standard","")).strip().upper()
            year=str(row.get("testing_standard_year","")).strip()
            if standard not in {"CLSI","EUCAST"} or not re.fullmatch(r"20\d\d",year) or category not in {"S","I","R"}:
                continue
            mo={"Acinetobacter baumannii":"B_ACNTB","Pseudomonas aeruginosa":"B_PSDMN_AERG","Escherichia coli":"B_[ORD]_ENTRBCTR"}[species]
            ab={"meropenem":"MEM","imipenem":"IPM"}[drug]
            candidates=bp_lookup.get((standard+" "+year,mo,ab),[])
            if len(candidates)!=1:
                continue
            match=candidates[0]
            allowed=breakpoint_categories(mic,match["breakpoint_S"],match["breakpoint_R"],standard)
            conditional_rows.append({"source":source,"species":species,"antibiotic":drug,"genome_id":gid,
                                     "native_AST_id":row.get("id",""),"category":category,"MIC_interval":serial(mic),
                                     "measurement_verbatim":row.get("measurement",""),"sign_verbatim":row.get("measurement_sign",""),
                                     "value_verbatim":row.get("measurement_value",""),"unit_verbatim":row.get("measurement_unit",""),
                                     "method_verbatim":row.get("laboratory_typing_method",""),"standard_verbatim":row.get("testing_standard",""),
                                     "year_verbatim":year,"conditional_human_clinical_table":match["guideline"],
                                     "conditional_category_set":"|".join(sorted(allowed)),"conditionally_compatible":category in allowed,
                                     "pmid_verbatim":json.dumps(row.get("pmid",[])),"breakpoint_table_reference":match["ref_tbl"]})
        return result
    summaries=[]
    for folder in sorted(args.current_root.glob("*_current")):
        if not (folder/"current_laboratory_only.json").exists():
            continue
        protocol,_,lab=validated_current(folder)
        species,drug=protocol["species_source_annotation"],protocol["antibiotic"]
        current=defaultdict(list)
        for row in lab:current[row["genome_id"]].append(row)
        counts=Counter()
        for gid in protocol["candidate_genome_ids"]:
            a=states(old[(species,drug,gid)],"archived",species,drug,gid)
            b=states(current[gid],"current_laboratory",species,drug,gid)
            if not a or not b:status="MIC_STATE_SET_UNPAIRED_OR_UNAVAILABLE"
            elif a==b:status="SAME_MIC_AND_CATEGORY_STATE_SET"
            else:status="MIC_OR_CATEGORY_STATE_SET_DIFFERENCE_CANDIDATE"
            counts[status]+=1
            pair_rows.append({"species":species,"antibiotic":drug,"genome_id":gid,"archived_state_set":json.dumps(sorted(a)),
                              "current_state_set":json.dumps(sorted(b)),"archived_unique_MIC_states":len(a),"current_unique_MIC_states":len(b),"comparison":status})
        summaries.append({"species":species,"antibiotic":drug,**dict(counts)})
    for filename,rows in [("quantitative_state_set_comparisons.csv",pair_rows),("conditional_version_specific_MIC_label_checks.csv",conditional_rows)]:
        with (args.out/filename).open("x",newline="") as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    flags=[r for r in conditional_rows if not r["conditionally_compatible"]]
    with (args.out/"conditional_incompatibility_candidates_NOT_errors.csv").open("x",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(conditional_rows[0]));writer.writeheader();writer.writerows(flags)
    report={"status":"EXPLORATORY_QUANTITATIVE_SETS_AND_CONDITIONAL_BREAKPOINT_SCENARIOS_NOT_ERROR_ADJUDICATION", "panels":summaries,
            "genome_drug_pairs":len(pair_rows),"pair_status_counts":dict(Counter(r["comparison"] for r in pair_rows)),
            "measurement_encoding_counts":{source:dict((status,n) for (s,status),n in encoding_counts.items() if s==source) for source in ["archived","current_laboratory"]},
            "conditional_checked_record_rows":len(conditional_rows),"conditional_incompatible_record_rows":len(flags),
            "conditional_incompatible_unique_genome_drug_keys":len({(r["species"],r["genome_id"],r["antibiotic"]) for r in flags}),
            "conditional_flag_counts_by_source":dict(Counter(r["source"] for r in flags)),
            "archived_sha256":hashlib.sha256(args.archived.read_bytes()).hexdigest(),"breakpoints_sha256":hashlib.sha256(args.breakpoints.read_bytes()).hexdigest(),
            "limitations":["Archived data lack native assay IDs; set differences may reflect additions/removals or encoding changes, not identified edits to an assay.",
                           "CLSI/EUCAST plus year does not establish M100 versus veterinary, infection site, dosing or exact edition.",
                           "Breakpoint compatibility uses a secondary curated dataset as an explicit human-clinical scenario; flags are not proven label errors.",
                           "Same source rows, multiple antibiotic observations, repeated publications and repeat genomes are not independent patients or studies.",
                           "Neither the original 2020 labels nor current labels are replaced."]}
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
