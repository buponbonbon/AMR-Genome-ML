#!/usr/bin/env python3
"""Verify and analyze the fixed EuSCAPE primary-BMD-derived challenge."""
import argparse,hashlib,itertools,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_frozen_database_reference_factorial import arm
from analyze_native_ccre_challenge import weighted,point,METRICS
from certify_pairwise_reference_ranking import certificate,direct_oracle
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,roc_auc_score,average_precision_score

LABEL='analysis_label_EUCAST14_nonmeningitis'
CP='Presence of one or more carbapenemases (in silico)'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')

def intervals(x):
    r={}
    for i,k in enumerate(METRICS[4:],4):
        v=x[:,i];v=v[np.isfinite(v)];r[k]=dict(valid_draws=len(v),percentile_2_5=float(np.quantile(v,.025)) if len(v) else None,percentile_97_5=float(np.quantile(v,.975)) if len(v) else None)
    return r

def normalize_ST(x):
    m=re.fullmatch(r'(?:ST)?([1-9]\d*)\*?',str(x).strip());return m.group(1) if m else ''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root
    run=root/'results/independent_reference_extension_20261112';manifest=run/'EuSCAPE_primary_BMD_execution_manifest_NOT_SCORED.csv'
    protocol=json.loads((run/'EUScAPE_NATIVE_PROTOCOL.json').read_text());assert protocol['manifest_sha256']==sha(manifest)
    assert json.loads((run/'EuSCAPE_independent_input_verification.json').read_text())['status'].startswith('PASS_')
    model=root/'results/frozen_model_challenge_20261110/received/project/checkpoints/external_validation_file11c/file11c_final_model.joblib';assert sha(model)==protocol['model_sha256']
    specdir=root/'results/frozen_model_challenge_20261110/frozen_reproduction_verified'
    coef=pd.read_csv(specdir/'frozen_linear_coefficients.csv');spec=json.loads((specdir/'frozen_linear_score_specification.json').read_text())
    native=arm(run/'EuSCAPE_native170',manifest,coef.feature.tolist(),coef.coefficient.to_numpy(),spec['intercept'],'2026-08-07.1')
    d=native['scored'].copy();assert set(d[CP])<={'Yes','No'};y=d[LABEL].map({'S':0,'R':1}).to_numpy(int)
    methods={'frozen_ML':d.frozen_probability_R.to_numpy(float),
        'native_broad_annotation_rule':d.amrfinder_carbapenemase_rule.to_numpy(float),
        'native_KPC_rule':d.kpc_family_rule.to_numpy(float),
        'author_carbapenemase_presence':d[CP].eq('Yes').to_numpy(float)}
    a.out.mkdir(parents=True,exist_ok=False);metrics={m:point(y,s) for m,s in methods.items()}
    pd.DataFrame([dict(method=m,**v) for m,v in metrics.items()]).to_csv(a.out/'method_metrics.csv',index=False)
    d['analysis_reference_R']=y
    for m,s in methods.items():d[m+'_R']=(s>=.5).astype(int)
    d['author_carbapenemase_presence_R']=methods['author_carbapenemase_presence'].astype(int)
    fitting=pd.read_csv(root/'results/frozen_model_challenge_20261110/received/project/data/splits/file07_outer_folds.csv',dtype=str)
    known={normalize_ST(x) for x in fitting.MLST};known.discard('')
    d['reported_ST_normalized']=d['Sequence type (ST)2'].map(normalize_ST)
    d['reported_ST_stratum']=d.reported_ST_normalized.map(lambda s:'unknown' if not s else 'seen_in_fitting' if s in known else 'not_seen_in_fitting')
    d.reset_index().to_csv(a.out/'paired_case_audit.csv',index=False)
    source_groups=[]
    for col in [CP,'reported_ST_stratum','Country']:
        for value,group in d.groupby(col):
            yy=group[LABEL].eq('R').to_numpy(int);ix=d.index.get_indexer(group.index)
            for m,s in methods.items():source_groups.append(dict(variable=col,stratum=value,method=m,**point(yy,s[ix])))
    pd.DataFrame(source_groups).to_csv(a.out/'descriptive_source_strata.csv',index=False)
    sensitivity={};checks=0
    for gi,col in enumerate(['PDS_bases','Country']):
        values=d[col].to_numpy();unique=np.unique(values);membership=(values[None,:]==unique[:,None]).astype(int)
        rng=np.random.default_rng(20261112+gi);counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=10000);weights=counts@membership
        arrays={m:weighted(y,s,weights) for m,s in methods.items()}
        for m,s in methods.items():
            for ix in range(5):
                w=weights[ix]
                if w@y==0 or w@(1-y)==0:continue
                expected=[balanced_accuracy_score(y,s>=.5,sample_weight=w),matthews_corrcoef(y,s>=.5,sample_weight=w),roc_auc_score(y,s,sample_weight=w),average_precision_score(y,s,sample_weight=w)]
                assert np.allclose(arrays[m][ix,[6,7,8,9]],expected,atol=1e-12);checks+=1
        contrasts={m:arrays['frozen_ML']-v for m,v in arrays.items() if m!='frozen_ML'}
        sensitivity[col]=dict(groups=len(unique),seed=20261112+gi,draws=10000,method_percentiles={m:intervals(v) for m,v in arrays.items()},ML_minus_comparator_percentiles={m:intervals(v) for m,v in contrasts.items()})
        np.savez_compressed(a.out/(col+'_bootstrap_metrics.npz'),**arrays)
    budgets=[];witnesses=[]
    for comparator,metric,fixed in itertools.product([m for m in methods if m!='frozen_ML'],['balanced_accuracy','AUROC'],[False,True]):
        r=certificate(y,methods['frozen_ML'],methods[comparator],np.ones(len(y),bool),metric,fixed);ix=r.pop('witness_indices');r['comparator']=comparator;budgets.append(r)
        assert abs(r['point_difference']-float(direct_oracle(metric,y,methods['frozen_ML'],methods[comparator])))<1e-12
        for i in ix:witnesses.append(dict(comparator=comparator,metric=metric,class_totals_fixed=fixed,sample_accession=d.index[i],source_reference='R' if y[i] else 'S',hypothetical_reference='S' if y[i] else 'R',hypothetical_only_not_a_correction=True))
    pd.DataFrame(budgets).to_csv(a.out/'exact_counterfactual_ranking_budgets.csv',index=False)
    pd.DataFrame(witnesses).to_csv(a.out/'COUNTERFACTUAL_WITNESSES_NOT_CORRECTIONS.csv',index=False)
    missing=native['frame'].loc[~native['frame'].index.isin(d.index)].copy();missing.reset_index().to_csv(a.out/'unscored_fixed_cases.csv',index=False)
    bounds={}
    if len(missing)<=10:
        ym=missing[LABEL].map({'S':0,'R':1}).to_numpy(int)
        for m,s in methods.items():
            values=[point(np.r_[y,ym],np.r_[s,extra]) for extra in itertools.product([0.,1.],repeat=len(missing))]
            bounds[m]={k:dict(min=min(v[k] for v in values),max=max(v[k] for v in values)) for k in METRICS if all(v[k] is not None for v in values)}
    report=dict(status='COMPLETE_FROZEN_PRIMARY_BMD_DERIVED_EUSCAPE_CHALLENGE',fixed_target=170,scored=len(d),unscored=len(missing),
        reference_definition='Explicit EUCAST14 Enterobacterales non-meningitis MIC scenario: S<=2mg/L,R>8mg/L; I/unsupported excluded at input gate. Not author-reported S/I/R.',
        source_doi='10.1038/s41564-019-0492-8',model_sha256=protocol['model_sha256'],no_training_tuning_or_recalibration=True,
        native_arm_verification=native['verification'],methods=metrics,source_composition_sensitivity=sensitivity,
        exact_counterfactual_ranking_budgets=budgets,full_target_missing_input_bounds=bounds,independent_weighted_metric_checks=checks,
        input_sha256=dict(manifest=sha(manifest),native_signature=sha(run/'EuSCAPE_native170/RUN_SIGNATURE.json'),primary_workbook=sha(run/'supplementary/PMC7244338/EMS83227-supplement-Supplementary_Table_4.xlsx'),analysis_script=sha(Path(__file__))),
        country_count=d.Country.nunique(),selected_PDS_count=d.PDS_bases.nunique(),reported_ST_counts=d.reported_ST_stratum.value_counts().to_dict(),
        limits=protocol['analysis_plan'][-1:]+[
            'Selected retrospective surveillance collection; not representative prospective patients and not a new laboratory study.',
            'Primary MICs are published central BMD; the source author S/I/R edition and infection-specific clinical truth are not invented.',
            'PDS filtering excludes sharing only against mapped fitting/E2/CCRE; fitting coverage2944/4218, not absolute clonal independence.',
            'EuSCAPE and CCRE are related European surveillance programmes; distinct specimen/cluster screens do not prove all collection-level effects independent.',
            'Author carbapenemase presence is a separate assembly/annotation control, never phenotypic truth.',
            'Composition percentiles and counterfactual budgets are not clinical error probabilities, prospective accuracy guarantees or patient outcomes.',
            'AUROC of continuous ML and binary rules has different output resolution; paired BA and error tradeoffs are primary.'])
    save(a.out/'report.json',report);print(json.dumps(dict(status=report['status'],scored=len(d),methods=metrics,country_count=report['country_count'],verification=native['verification']),indent=2))

if __name__=='__main__':main()
