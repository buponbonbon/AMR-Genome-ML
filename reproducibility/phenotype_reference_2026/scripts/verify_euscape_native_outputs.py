#!/usr/bin/env python3
"""Independent CSV, math.fsum and sklearn checks of the new frozen challenge."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,matthews_corrcoef,roc_auc_score,average_precision_score
from verify_factorial_completed_outputs import read_tsv

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    run=root/'results/independent_reference_extension_20261112';analysis=run/'EuSCAPE_analysis';native=run/'EuSCAPE_native170'
    report=json.loads((analysis/'report.json').read_text());case=pd.read_csv(analysis/'paired_case_audit.csv',keep_default_na=False)
    coefdir=root/'results/frozen_model_challenge_20261110/frozen_reproduction_verified';coef=pd.read_csv(coefdir/'frozen_linear_coefficients.csv');spec=json.loads((coefdir/'frozen_linear_score_specification.json').read_text())
    z=np.load(native/'native_features.npz',allow_pickle=False);assert z['biosamples'].tolist()==case.sample_accession.tolist() and z['features'].tolist()==coef.feature.tolist()
    signature=json.loads((native/'RUN_SIGNATURE.json').read_text());maxerror=0
    for ix,row in enumerate(case.to_dict('records')):
        sample=row['sample_accession'];p=native/'amrfinder'/(sample+'.tsv');r=json.loads((native/'receipts'/(sample+'.json')).read_text())
        acquired=json.loads((run/'EuSCAPE_assemblies/receipts'/(sample+'.json')).read_text())
        assert r['status']=='PASS' and sha(p)==r['output_sha256'] and r['fasta_sha256']==acquired['fasta_sha256']
        assert r['database_manifest_sha256']==signature['database_manifest_sha256'] and r['caller_exe_sha256']==signature['caller_exe_sha256']
        vector,calls=read_tsv(p,coef.feature.tolist());assert np.array_equal(vector,z['X'][ix])
        score=spec['intercept']+math.fsum(float(x)*float(y) for x,y in zip(vector,coef.coefficient));prob=1/(1+math.exp(-score))
        error=abs(prob-float(row['frozen_probability_R']));maxerror=max(maxerror,error);assert error<1e-12
        assert int(float(row['frozen_predicted_R']))==int(prob>=.5)
        assert calls['broad_annotation_rule']==int(row['native_broad_annotation_rule_R']) and calls['KPC_rule']==int(row['native_KPC_rule_R'])
        assert int(row['author_carbapenemase_presence_R'])==int(row['Presence of one or more carbapenemases (in silico)']=='Yes')
    y=case.analysis_label_EUCAST14_nonmeningitis.eq('R').to_numpy(int);methods={m:case['frozen_probability_R' if m=='frozen_ML' else m+'_R'].to_numpy(float) for m in report['methods']}
    for m,s in methods.items():
        tn,fp,fn,tp=confusion_matrix(y,s>=.5,labels=[0,1]).ravel();expected=dict(TP=tp,FN=fn,TN=tn,FP=fp,balanced_accuracy=balanced_accuracy_score(y,s>=.5),MCC=matthews_corrcoef(y,s>=.5),AUROC=roc_auc_score(y,s),average_precision=average_precision_score(y,s))
        assert all(abs(report['methods'][m][k]-v)<1e-12 for k,v in expected.items())
    keys=['TP','FN','TN','FP','sensitivity','specificity','balanced_accuracy','MCC','AUROC','average_precision']
    for scheme,data in report['source_composition_sensitivity'].items():
        arrays=np.load(analysis/(scheme+'_bootstrap_metrics.npz'),allow_pickle=False)
        for comparator,bounds in data['ML_minus_comparator_percentiles'].items():
            delta=arrays['frozen_ML']-arrays[comparator]
            for i,k in enumerate(keys[4:],4):
                vals=delta[:,i];vals=vals[np.isfinite(vals)];q=np.quantile(vals,[.025,.975]);saved=bounds[k]
                assert len(vals)==saved['valid_draws'] and np.allclose(q,[saved['percentile_2_5'],saved['percentile_97_5']],rtol=0,atol=1e-12)
    witnesses=pd.read_csv(analysis/'COUNTERFACTUAL_WITNESSES_NOT_CORRECTIONS.csv',keep_default_na=False)
    for r in report['exact_counterfactual_ranking_budgets']:
        if r['minimum_label_reassignments_for_strict_reversal'] is None:continue
        selected=witnesses[witnesses.comparator.eq(r['comparator'])&witnesses.metric.eq(r['metric'])&witnesses.class_totals_fixed.eq(r['class_totals_fixed'])]
        assert len(selected)==r['minimum_label_reassignments_for_strict_reversal'];yy=y.copy();ix=case.sample_accession.isin(selected.sample_accession).to_numpy();yy[ix]=1-yy[ix]
        a,b=methods['frozen_ML'],methods[r['comparator']]
        delta=balanced_accuracy_score(yy,a>=.5)-balanced_accuracy_score(yy,b>=.5) if r['metric']=='balanced_accuracy' else roc_auc_score(yy,a)-roc_auc_score(yy,b)
        assert delta*r['point_difference']<0 and abs(delta-r['witness_point_difference'])<1e-12
    result=dict(status='PASS_INDEPENDENT_NATIVE_TSV_VECTOR_RULE_LOGISTIC_METRIC_INTERVAL_WITNESS_AUDIT',cases=len(case),maximum_math_fsum_probability_difference=maxerror,
        all_saved_bootstrap_contrasts_verified=True,all_counterfactual_witnesses_recomputed=True,report_sha256=sha(analysis/'report.json'))
    (analysis/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
