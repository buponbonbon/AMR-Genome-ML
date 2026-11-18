#!/usr/bin/env python3
"""Finite-cohort error-count certificates for fixed model/rule under observed references."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def confusion(labels, predictions):
    return {"tp":int(((labels==1)&(predictions==1)).sum()),"fn":int(((labels==1)&(predictions==0)).sum()),
            "tn":int(((labels==0)&(predictions==0)).sum()),"fp":int(((labels==0)&(predictions==1)).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crosswalk",type=Path,required=True)
    p.add_argument("--exact-metrics",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    data=pd.read_csv(args.crosswalk,dtype={"target_acc":str,"biosample":str})
    labels={"R":1,"S":0,"1":1,"0":0}
    old=data.historical_label.astype(str).str.upper().map(labels)
    new=data.scenario_label.astype(str).str.upper().map(labels)
    if old.isna().any() or new.isna().any():raise ValueError("Unexpected nonbinary reference category")
    old=old.astype(int);new=new.astype(int);pred=(data.probability.astype(float)>=0.5).astype(int)
    focal=data.project.eq("PRJNA1322038")
    if not ((old[focal]+new[focal])==1).all() or not old[~focal].eq(new[~focal]).all():
        raise ValueError("The known source-level complement scenario does not hold")
    metrics=pd.read_csv(args.exact_metrics)
    exact={}
    for method,reference in [("frozen_ML","historical"),("frozen_ML","october"),("frozen_rule","historical")]:
        selected=metrics[(metrics.method==method)&(metrics.reference==reference)]
        if len(selected)!=1:raise ValueError("Missing or ambiguous archived method metrics")
        exact[(method,reference)]={k:int(selected.iloc[0][k]) for k in ["tp","fn","tn","fp"]}
    for reference,labels in [("historical",old),("october",new)]:
        if confusion(labels,pred)!=exact[("frozen_ML",reference)]:raise ValueError("Frozen predictions do not reproduce archived ML counts")
    rule=exact[("frozen_rule","historical")]
    fr=int(old[focal].sum());fs=int(focal.sum())-fr
    nr=int(old[~focal].sum());ns=int((~focal).sum())-nr
    feasible=set()
    for focal_tp in range(fr+1):
        for focal_fp in range(fs+1):
            outside=(rule["tp"]-focal_tp,rule["fn"]-(fr-focal_tp),rule["tn"]-(fs-focal_fp),rule["fp"]-focal_fp)
            if min(outside)<0 or outside[0]+outside[1]!=nr or outside[2]+outside[3]!=ns:continue
            feasible.add((outside[0]+focal_fp,outside[1]+fs-focal_fp,outside[2]+fr-focal_tp,outside[3]+focal_tp))
    current_R=int(new.sum());current_S=len(new)-current_R;positive_calls=rule["tp"]+rule["fp"]
    assert all(tp+fn==current_R and tn+fp==current_S and tp+fp==positive_calls for tp,fn,tn,fp in feasible)
    ranges={key:{"min":min(row[i] for row in feasible),"max":max(row[i] for row in feasible)} for i,key in enumerate(["tp","fn","tn","fp"])}
    rows=[]
    for (method,reference),cm in exact.items():
        r=cm["tp"]+cm["fn"];s=cm["tn"]+cm["fp"]
        rows.append({"method":method,"reference":reference,"status":"EXACT_FIXED_PREDICTION_COUNTS",
                     "reference_R":r,"reference_S":s,"FN_min":cm["fn"],"FN_max":cm["fn"],"FP_min":cm["fp"],"FP_max":cm["fp"],
                     "apparent_false_susceptible_rate_min":cm["fn"]/r,"apparent_false_susceptible_rate_max":cm["fn"]/r,
                     "apparent_false_resistant_rate_min":cm["fp"]/s,"apparent_false_resistant_rate_max":cm["fp"]/s})
    rows.append({"method":"frozen_rule","reference":"october","status":"FEASIBLE_RANGE_NOT_ACTUAL_REPLAY_OR_CI",
                 "reference_R":current_R,"reference_S":current_S,"FN_min":ranges["fn"]["min"],"FN_max":ranges["fn"]["max"],
                 "FP_min":ranges["fp"]["min"],"FP_max":ranges["fp"]["max"],
                 "apparent_false_susceptible_rate_min":ranges["fn"]["min"]/current_R,"apparent_false_susceptible_rate_max":ranges["fn"]["max"]/current_R,
                 "apparent_false_resistant_rate_min":ranges["fp"]["min"]/current_S,"apparent_false_resistant_rate_max":ranges["fp"]["max"]/current_S})
    with (args.out/"error_counts_and_feasible_bounds.csv").open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    ml=exact[("frozen_ML","october")]
    weighted=[]
    for ratio in [1,2,5,10,20,50]:
        differences=[ratio*fn+fp-(ratio*ml["fn"]+ml["fp"]) for tp,fn,tn,fp in feasible]
        weighted.append({"hypothetical_FN_to_FP_weight_ratio":ratio,"rule_minus_ML_weighted_error_min":min(differences),
                         "rule_minus_ML_weighted_error_max":max(differences),"ML_weighted_error_advantage_certified":min(differences)>0,
                         "rule_weighted_error_advantage_certified":max(differences)<0})
    with (args.out/"hypothetical_weighted_error_bounds_NOT_clinical_utility.csv").open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(weighted[0]));w.writeheader();w.writerows(weighted)
    report={"status":"FINITE_COHORT_REFERENCE_RELATIVE_ERROR_CERTIFICATE_NOT_CLINICAL_SUPERIORITY",
            "threshold":0.5,"n":len(data),"source_focal_old_R":fr,"source_focal_old_S":fs,"feasible_current_rule_aggregate_matrices":len(feasible),
            "rule_fixed_positive_calls":positive_calls,"current_reference_R":current_R,"current_reference_S":current_S,
            "current_rule_FP_lower_bound_from_margins":max(0,positive_calls-current_R),"current_ML_FP":ml["fp"],
            "ML_fewer_false_resistant_calls_under_both_observed_references":rule["fp"]>exact[("frozen_ML","historical")]["fp"] and ranges["fp"]["min"]>ml["fp"],
            "minimum_fewer_false_resistant_calls_current_reference":ranges["fp"]["min"]-ml["fp"],
            "current_reference_FP_rate_advantage_lower_bound_percentage_points":100*(ranges["fp"]["min"]-ml["fp"])/current_S,
            "current_ML_FN":ml["fn"],"current_rule_FN_feasible_range":ranges["fn"],
            "current_false_susceptible_order_identified":ranges["fn"]["max"]<ml["fn"] or ranges["fn"]["min"]>ml["fn"],
            "hypothetical_weighted_error_order_identified_for_any_tested_ratio":any(x["ML_weighted_error_advantage_certified"] or x["rule_weighted_error_advantage_certified"] for x in weighted),
            "minimum_missing_input_for_exact_rule_counts":"results/file16i_evidence_lock_v1/file16i_E2_paired_by_bioproject.csv",
            "input_sha256":{args.crosswalk.name:hashlib.sha256(args.crosswalk.read_bytes()).hexdigest(),args.exact_metrics.name:hashlib.sha256(args.exact_metrics.read_bytes()).hexdigest()},
            "limitations":["Errors are relative to the selected recorded reference labels, not adjudicated true clinical susceptibility.",
                           "The same locked thresholds imply different positive-call totals; lower FP does not establish a better sensitivity/specificity tradeoff.",
                           "Ranges enumerate compatible missing rule calls; they are not confidence intervals or probability distributions.",
                           "Weighted error ratios are illustrative mathematical scenarios, not patient utilities or calibrated costs.",
                           "No novel mathematical identity, new prediction model or clinical safety certification is claimed."]}
    (args.out/"certificate.json").write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
