#!/usr/bin/env python3
"""Independently audit completed factorial outputs using CSV rows and sklearn."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,matthews_corrcoef,roc_auc_score,average_precision_score

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read_tsv(p,features):
    with p.open() as f:raw=list(csv.DictReader(f,delimiter='\t'))
    rows=[r for r in raw if r['Type'].upper()=='AMR' and r['Subtype'].upper() in {'AMR','POINT','POINT_DISRUPT'}]
    keys={r['Element symbol'].strip().lower() for r in rows if r['Element symbol'].strip()}
    vector=np.array([int(s.lower().strip() in keys) for s in features],np.uint8)
    kpc=any(re.match(r'^(?:bla)?kpc(?:[-_].*|$)',r['Element symbol'].strip(),re.I) for r in rows)
    broad=False
    for r in rows:
        semantic=' | '.join(r.get(k,'').strip() for k in ['Element name','Name','Class','Subclass','Element symbol'] if r.get(k,''))
        broad|=bool(re.search(r'carbapenem(?:ase|[- ]?hydroly\w*|[- ]?resistan\w*)?|carbapenem',semantic,re.I) or
            re.match(r'^(?:bla)?(?:kpc|ndm|vim|imp|sme|imi|nmc|spm|gim|sim|dim|aim|tmb|fri|bic)(?:[-_].*|$)',r['Element symbol'].strip(),re.I))
    return vector,{'KPC_rule':int(kpc),'broad_annotation_rule':int(broad)}

def contrast(cells):
    result={};refs=sorted({k.split('|')[1] for k in cells})
    methods=['ML','broad_annotation_rule','KPC_rule']
    for ref in refs:
        for method in methods:result[f'DB_Aug_minus_Jan|{ref}|{method}']=cells[f'Aug|{ref}|{method}']-cells[f'Jan|{ref}|{method}']
        for db in ['Jan','Aug']:
            for rule in methods[1:]:result[f'ML_minus_{rule}|{db}|{ref}']=cells[f'{db}|{ref}|ML']-cells[f'{db}|{ref}|{rule}']
    if len(refs)==2:
        for db in ['Jan','Aug']:
            for method in methods:result[f'Reference_Oct_minus_historical|{db}|{method}']=cells[f'{db}|October_partial|{method}']-cells[f'{db}|historical|{method}']
        for method in methods:result['DB_by_reference_interaction|'+method]=result[f'Reference_Oct_minus_historical|Aug|{method}']-result[f'Reference_Oct_minus_historical|Jan|{method}']
        for rule in methods[1:]:result['Method_by_DB_by_reference|ML_minus_'+rule]=result['DB_by_reference_interaction|ML']-result['DB_by_reference_interaction|'+rule]
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--verify-dna',action='store_true');a=p.parse_args()
    run=a.root/'results/database_reference_factorial_20261111';analysis=run/'final_analysis'
    report=json.loads((analysis/'report.json').read_text());specdir=a.root/'results/frozen_model_challenge_20261110/frozen_reproduction_verified'
    coef=pd.read_csv(specdir/'frozen_linear_coefficients.csv');spec=json.loads((specdir/'frozen_linear_score_specification.json').read_text())
    features=coef.feature.tolist();betas=coef.coefficient.to_numpy();intercept=spec['intercept']
    metrics=['TP','FN','TN','FP','sensitivity','specificity','balanced_accuracy','MCC','AUROC','average_precision']
    paths={'CCRE':{'Jan':run/'CCRE_Jan','Aug':a.root/'results/strict_CCRE_followup_20261111/native355'},'E2':{'Jan':run/'E2_Jan','Aug':run/'E2_Aug'}}
    counts={};checks=0;maxerror=0;dna_checked=set();metadata={}
    for cohort,arms in paths.items():
        r=report['cohorts'][cohort];case=pd.read_csv(analysis/cohort/'paired_case_audit.csv',keep_default_na=False)
        for db,path in arms.items():
            pred=pd.read_csv(path/'frozen_challenge_predictions.csv',keep_default_na=False).set_index('sample_accession');z=np.load(path/'native_features.npz',allow_pickle=False)
            matrix={s:x for s,x in zip(z['biosamples'],z['X'])};signature=json.loads((path/'RUN_SIGNATURE.json').read_text())
            for sample in case.sample_accession:
                receipt=json.loads((path/'receipts'/(sample+'.json')).read_text());tsv=path/'amrfinder'/(sample+'.tsv')
                assert sha(tsv)==receipt['output_sha256'] and receipt['status']=='PASS'
                assert receipt['caller_exe_sha256']==signature['caller_exe_sha256']
                assert receipt['database_manifest_sha256']==signature['database_manifest_sha256']
                key=(cohort,sample)
                if key in metadata:assert metadata[key]==receipt['fasta_sha256']
                metadata[key]=receipt['fasta_sha256']
                if a.verify_dna and key not in dna_checked:
                    assert sha(Path(receipt['fasta_path']))==receipt['fasta_sha256'];dna_checked.add(key)
                vector,rules=read_tsv(tsv,features);assert np.array_equal(vector,matrix[sample])
                logit=intercept+math.fsum(float(x)*float(y) for x,y in zip(vector,betas));prob=1/(1+math.exp(-logit))
                error=abs(prob-float(pred.loc[sample,'frozen_probability_R']));maxerror=max(maxerror,error);assert error<1e-12
                for method,col in [('KPC_rule','kpc_family_rule'),('broad_annotation_rule','amrfinder_carbapenemase_rule')]:assert rules[method]==int(float(pred.loc[sample,col]))
                checks+=1
            for cell in [v for v in r['cell_metrics'] if v['database']==db]:
                ref=cell['reference'];method=cell['method'];y=case['S/I/R (PHAS interpretation)_Meropenem' if cohort=='CCRE' else 'historical_label' if ref=='historical' else 'scenario_label'].eq('R').to_numpy(int)
                col='model_probability_'+db if method=='ML' else method+'_'+db+'_R';s=case[col].to_numpy(float);q=s>=.5
                tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
                expected=dict(TP=tp,FN=fn,TN=tn,FP=fp,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),balanced_accuracy=balanced_accuracy_score(y,q),MCC=matthews_corrcoef(y,q),AUROC=roc_auc_score(y,s),average_precision=average_precision_score(y,s))
                assert all(abs(float(cell[k])-float(v))<1e-12 for k,v in expected.items())
            counts[cohort+'_'+db]=len(case)
        for scheme,b in r['source_composition_sensitivity'].items():
            z=np.load(analysis/cohort/(scheme+'_bootstrap_metrics.npz'),allow_pickle=False);c=contrast({k:z[k] for k in z.files})
            for name,v in c.items():
                saved=b['paired_contrast_percentiles'][name]['metrics']
                for i,k in enumerate(metrics[4:],4):
                    vals=v[:,i];vals=vals[np.isfinite(vals)];bounds=np.quantile(vals,[.025,.975])
                    assert len(vals)==saved[k]['valid_draws']
                    assert np.allclose(bounds,[saved[k]['percentile_2_5'],saved[k]['percentile_97_5']],rtol=0,atol=1e-12)
    e=pd.read_csv(run/'E2_execution_manifest.csv');original=pd.read_csv(a.root/'results/received_bundle_audit_20261024/historical_raw_verification_v2/inputs/file11c_blind_predictions.csv').set_index('biosample_acc')
    scenario=pd.read_csv(a.root/'results/received_bundle_audit_20261024/full_source_verification_20261024/reconciliation/E2_POSTHOC_reference_scenario.csv').set_index('biosample')
    assert original.index.is_unique and scenario.index.is_unique
    # The original source columns are inspected explicitly, rather than inferred from the new result.
    for row in e.to_dict('records'):
        old=original.loc[row['sample_accession']]
        assert row['historical_label']==('R' if int(old.label_binary) else 'S')
        assert abs(row['probability_R']-old.probability_R)<1e-12
        assert row['scenario_label']==scenario.loc[row['sample_accession'],'scenario_label']
        assert row['project']==scenario.loc[row['sample_accession'],'project']
        assert row['fasta_sha256']==old.fasta_sha256
    changed=e.historical_label.ne(e.scenario_label)
    assert changed.sum()==175 and e.loc[changed,'project'].eq('PRJNA1322038').all()
    result=dict(status='PASS_INDEPENDENT_CSV_LOGISTIC_RULE_METRIC_AND_INTERVAL_AUDIT',arms=counts,
        raw_TSV_feature_rule_score_checks=checks,maximum_math_fsum_probability_difference=maxerror,
        distinct_DNA_files_rehashed=len(dna_checked),DNA_hash_recheck_requested=a.verify_dna,
        preserved_E2_historical_probabilities_and_175_focal_reference_differences_verified=True,
        sklearn_point_metrics_and_all_saved_bootstrap_contrast_percentiles_verified=True,
        analysis_sha256=sha(analysis/'report.json'),verifier_sha256=sha(Path(__file__)))
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
