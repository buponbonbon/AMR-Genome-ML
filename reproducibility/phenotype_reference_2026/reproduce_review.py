"""Recompute the main fixed-prediction comparisons from the review archive."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, matthews_corrcoef, roc_auc_score

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("review_reproduced"))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    args.out.mkdir(parents=True,exist_ok=True)
    base=root/"results"
    factorial=base/"database_reference_factorial_20261111/final_analysis"
    e2=pd.read_csv(factorial/"E2/paired_case_audit.csv",keep_default_na=False)
    ccre=pd.read_csv(factorial/"CCRE/paired_case_audit.csv",keep_default_na=False)
    eu=pd.read_csv(base/"independent_reference_extension_20261112/EuSCAPE_analysis/paired_case_audit.csv",keep_default_na=False)
    results=[]
    def add(cohort,reference,database,method,labels,scores):
        y=np.asarray(labels=="R",int);scores=np.asarray(scores,float)
        tn,fp,fn,tp=confusion_matrix(y,scores>=.5,labels=[0,1]).ravel()
        row=dict(cohort=cohort,reference=reference,database=database,method=method,n=len(y),
                 TP=int(tp),FN=int(fn),TN=int(tn),FP=int(fp),
                 balanced_accuracy=float(balanced_accuracy_score(y,scores>=.5)),
                 MCC=float(matthews_corrcoef(y,scores>=.5)),
                 AUROC=float(roc_auc_score(y,scores)))
        results.append(row)
    for cohort,frame,refs in [
        ("E2",e2,{"historical":"historical_label","October_partial":"scenario_label"}),
        ("CCRE",ccre,{"PHAS_original":"S/I/R (PHAS interpretation)_Meropenem"}),
    ]:
        for reference,label in refs.items():
            for database in ["Jan","Aug"]:
                add(cohort,reference,database,"Frozen ML",frame[label],frame["model_probability_"+database])
                add(cohort,reference,database,"Broad rule",frame[label],frame["broad_annotation_rule_"+database+"_R"])
                add(cohort,reference,database,"KPC rule",frame[label],frame["KPC_rule_"+database+"_R"])
        if cohort=="CCRE":
            add(cohort,"PHAS_original","author","Source CP presence",frame["S/I/R (PHAS interpretation)_Meropenem"],
                frame["Bla_Carb_acquired (with new allele designations)"].ne("-").astype(int))
    for method,column in [
        ("Frozen ML","frozen_probability_R"),("Broad rule","native_broad_annotation_rule_R"),
        ("KPC rule","native_KPC_rule_R"),("Source CP presence","author_carbapenemase_presence_R"),
    ]:
        add("EuSCAPE","EUCAST14_derived","Aug" if method!="Source CP presence" else "author",
            method,eu["analysis_label_EUCAST14_nonmeningitis"],eu[column])
    table=pd.DataFrame(results)
    expected={
        ("E2","historical","Frozen ML"):(109,54,221,141),
        ("E2","October_partial","Frozen ML"):(236,30,245,14),
        ("CCRE","PHAS_original","Frozen ML"):(149,15,176,14),
        ("EuSCAPE","EUCAST14_derived","Frozen ML"):(35,5,114,16),
    }
    for key,counts in expected.items():
        cohort,reference,method=key
        r=table[(table.cohort==cohort)&(table.reference==reference)&(table.method==method)&(table.database=="Aug")].iloc[0]
        assert tuple(int(r[k]) for k in ["TP","FN","TN","FP"])==counts
    table.to_csv(args.out/"recomputed_main_metrics.csv",index=False)
    coeff_dir=base/"frozen_model_challenge_20261110/frozen_reproduction_verified"
    coefficients=pd.read_csv(coeff_dir/"frozen_linear_coefficients.csv")
    spec=json.loads((coeff_dir/"frozen_linear_score_specification.json").read_text())
    coefficient_series=coefficients.set_index("feature").coefficient
    checks=[]
    arms=[
        "database_reference_factorial_20261111/E2_Jan",
        "database_reference_factorial_20261111/E2_Aug",
        "database_reference_factorial_20261111/CCRE_Jan",
        "strict_CCRE_followup_20261111/native355",
        "independent_reference_extension_20261112/EuSCAPE_native170",
    ]
    for arm in arms:
        folder=base/arm
        z=np.load(folder/"native_features.npz",allow_pickle=False)
        data=pd.read_csv(folder/"frozen_challenge_predictions.csv",keep_default_na=False).set_index("sample_accession")
        probability=expit(z["X"]@coefficient_series.loc[z["features"]].to_numpy()+spec["intercept"])
        recorded=data.loc[z["biosamples"],"frozen_probability_R"].to_numpy(float)
        difference=float(np.max(np.abs(probability-recorded)))
        assert difference<1e-12
        checks.append(dict(arm=arm,n=len(recorded),maximum_probability_difference=difference))
    mic=pd.read_csv(base/"rsos_readiness_extension_20261113/MIC_boundary_analysis/per_case_MIC_boundary_audit.csv")
    combined=[]
    for cohort,frame,label in [("CCRE",ccre,"S/I/R (PHAS interpretation)_Meropenem"),
                              ("EuSCAPE",eu,"analysis_label_EUCAST14_nonmeningitis")]:
        status=mic[mic.cohort==cohort].set_index("sample_accession").boundary_status
        robust=frame.sample_accession.map(status).eq("BOUNDARY_ROBUST").to_numpy()
        yy=frame[label].eq("R").to_numpy(int)
        score=frame["model_probability_Aug" if cohort=="CCRE" else "frozen_probability_R"].to_numpy(float)
        rule=frame["broad_annotation_rule_Aug_R" if cohort=="CCRE" else "native_broad_annotation_rule_R"].to_numpy(float)
        for name,mask in [("All",np.ones(len(frame),bool)),("Robust",robust)]:
            delta=balanced_accuracy_score(yy[mask],score[mask]>=.5)-balanced_accuracy_score(yy[mask],rule[mask]>=.5)
            combined.append(dict(cohort=cohort,stratum=name,n=int(mask.sum()),ML_minus_rule_BA=float(delta)))
    pd.DataFrame(combined).to_csv(args.out/"recomputed_MIC_contrasts.csv",index=False)
    report={"status":"PASS","metric_rows":len(table),"frozen_score_checks":checks,
            "reference_event_replications":1,
            "clinical_AST_adjudication_claimed":False}
    (args.out/"verification.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
