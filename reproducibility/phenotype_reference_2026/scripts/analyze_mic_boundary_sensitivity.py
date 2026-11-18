#!/usr/bin/env python3
"""Post-hoc fixed-prediction MIC-boundary sensitivity; never edits source labels."""
import argparse, hashlib, json, re
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from analyze_native_ccre_challenge import point, weighted, METRICS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def interval(raw):
    s=str(raw).strip().replace('≤','<=').replace('≥','>=')
    m=re.fullmatch(r'(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)',s)
    if not m or Decimal(m[2])<=0:return None
    op=m[1] or '=';v=Decimal(m[2]);zero=Decimal(0);inf=Decimal('Infinity')
    return (v,v,True,True) if op=='=' else (zero,v,False,op=='<=') if op in {'<','<='} else (v,inf,op=='>=',False)

def possible(bounds):
    lo,hi,lc,hc=bounds;labels=[]
    if lo<2 or lo==2 and lc:labels.append('S')
    if hi>2 and (lo<8 or lo==8 and lc):labels.append('I')
    if hi>8:labels.append('R')
    return labels

def classify(raw,label):
    bounds=interval(raw)
    if bounds is None:return dict(observation_categories='',dilution_envelope_categories='',boundary_status='UNSUPPORTED_MIC',source_category_vs_scenario='NOT_DETERMINABLE')
    lo,hi,lc,hc=bounds;obs=possible(bounds);envelope=possible((lo/2,hi*2,lc,hc))
    relation='UNIQUE_AGREEMENT' if obs==[label] else 'INTERVAL_COMPATIBLE_NOT_UNIQUE' if label in obs else 'INCOMPATIBLE_WITH_DECLARED_SCENARIO'
    status='BOUNDARY_ROBUST' if envelope==[label] else 'BOUNDARY_SENSITIVE' if obs==[label] else 'OBSERVED_INTERVAL_OR_CATEGORY_UNCERTAIN'
    return dict(observation_categories='|'.join(obs),dilution_envelope_categories='|'.join(envelope),boundary_status=status,source_category_vs_scenario=relation)

def oracle(raw,label):
    # Separate direct inequality implementation; no interval overlap routine.
    s=str(raw).strip().replace('≤','<=').replace('≥','>=');m=re.fullmatch(r'(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)',s)
    if not m or Decimal(m[2])<=0:return 'UNSUPPORTED_MIC'
    op=m[1] or '=';v=Decimal(m[2])
    if op=='=':category='S' if v<=2 else 'R' if v>8 else 'I';core='S' if v<=1 else 'R' if v>16 else None
    elif op in {'<','<='}:category='S' if v<=2 else None;core='S' if v<=1 else None
    else:category='R' if (v>=8 if op=='>' else v>8) else None;core='R' if (v>=16 if op=='>' else v>16) else None
    return 'BOUNDARY_ROBUST' if core==label else 'BOUNDARY_SENSITIVE' if category==label else 'OBSERVED_INTERVAL_OR_CATEGORY_UNCERTAIN'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root
    run=root/'results/rsos_readiness_extension_20261113';protocol=run/'MIC_BOUNDARY_PROTOCOL.json'
    assert sha(protocol)==(run/'MIC_BOUNDARY_PROTOCOL.json.sha256').read_text().strip()
    a.out.mkdir(parents=True,exist_ok=False);all_metrics=[];rows=[];reports={};arrays={};checks=0
    definitions=[('CCRE','results/database_reference_factorial_20261111/final_analysis/CCRE/paired_case_audit.csv','MIC_Meropenem','S/I/R (PHAS interpretation)_Meropenem','Country of origin'),
        ('EuSCAPE','results/independent_reference_extension_20261112/EuSCAPE_analysis/paired_case_audit.csv','Meropenem MIC (centralised re-testing)','analysis_label_EUCAST14_nonmeningitis','Country')]
    for name,path,mic,label,country in definitions:
        f=root/path;d=pd.read_csv(f,keep_default_na=False);assert d.sample_accession.is_unique
        y=d[label].eq('R').to_numpy(int);assert set(d[label])=={'S','R'}
        methods={'frozen_ML':d['model_probability_Aug' if name=='CCRE' else 'frozen_probability_R'].to_numpy(float),
            'broad_rule':d['broad_annotation_rule_Aug_R' if name=='CCRE' else 'native_broad_annotation_rule_R'].to_numpy(float),
            'KPC_rule':d['KPC_rule_Aug_R' if name=='CCRE' else 'native_KPC_rule_R'].to_numpy(float)}
        if name=='EuSCAPE':methods['source_CP_presence']=d.author_carbapenemase_presence_R.to_numpy(float)
        else:
            source=d['Bla_Carb_acquired (with new allele designations)'].str.strip()
            assert not source.isin(['','Missing','NULL','NA','Not tested']).any()
            methods['source_CP_presence']=source.ne('-').to_numpy(float)
        annotations=[]
        for i,row in d.iterrows():
            c=classify(row[mic],row[label]);assert c['boundary_status']==oracle(row[mic],row[label]);checks+=1;annotations.append(c)
            rows.append(dict(cohort=name,sample_accession=row.sample_accession,source_MIC_raw=row[mic],original_reference=row[label],**c,
                **{method+'_predicted_R':int(s[i]>=.5) for method,s in methods.items()}))
        ann=pd.DataFrame(annotations)
        masks={'all_fixed_cases':np.ones(len(d),bool),'boundary_robust':ann.boundary_status.eq('BOUNDARY_ROBUST').to_numpy(),
            'boundary_sensitive':ann.boundary_status.eq('BOUNDARY_SENSITIVE').to_numpy(),
            'uncertain_or_unsupported':~ann.boundary_status.isin(['BOUNDARY_ROBUST','BOUNDARY_SENSITIVE']).to_numpy()}
        assert sum(int(v.sum()) for k,v in masks.items() if k!='all_fixed_cases')==len(d)
        groups=d[country].to_numpy();unique=np.unique(groups);membership=(groups[None,:]==unique[:,None]).astype(int)
        rng=np.random.default_rng(20261114);counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=10000);w=counts@membership
        strata={}
        for stratum,mask in masks.items():
            if not mask.any():strata[stratum]={'n':0};continue
            yy=y[mask];values={};boots={}
            for method,scores in methods.items():
                s=scores[mask];v=point(yy,s);all_metrics.append(dict(cohort=name,stratum=stratum,method=method,**v));values[method]=v
                tn,fp,fn,tp=confusion_matrix(yy,s>=.5,labels=[0,1]).ravel();assert [v[k] for k in ['TP','FN','TN','FP']]==[tp,fn,tn,fp]
                if len(np.unique(yy))==2:
                    assert abs(v['balanced_accuracy']-balanced_accuracy_score(yy,s>=.5))<1e-12
                    assert abs(v['AUROC']-roc_auc_score(yy,s))<1e-12
                checks+=1;boots[method]=weighted(yy,s,w[:,mask]);arrays[name+'|'+stratum+'|'+method]=boots[method]
            delta=boots['frozen_ML'][:,METRICS.index('balanced_accuracy')]-boots['broad_rule'][:,METRICS.index('balanced_accuracy')]
            valid=delta[np.isfinite(delta)];q=np.quantile(valid,[.025,.975]) if len(valid) else [None,None]
            for draw in range(3):
                ww=w[draw,mask]
                if ww@yy==0 or ww@(1-yy)==0:continue
                expected=balanced_accuracy_score(yy,methods['frozen_ML'][mask]>=.5,sample_weight=ww)-balanced_accuracy_score(yy,methods['broad_rule'][mask]>=.5,sample_weight=ww)
                assert abs(delta[draw]-expected)<1e-12;checks+=1
            strata[stratum]=dict(n=int(mask.sum()),methods=values,paired_country_composition=dict(seed=20261114,draws=10000,source_countries=len(unique),valid_draws=len(valid),ML_minus_broad_BA=float(values['frozen_ML']['balanced_accuracy']-values['broad_rule']['balanced_accuracy']) if len(np.unique(yy))==2 else None,
                percentile_2_5=float(q[0]) if q[0] is not None else None,percentile_97_5=float(q[1]) if q[1] is not None else None))
        reports[name]=dict(input_sha256=sha(f),fixed_cases=len(d),MIC_category_checks=ann.source_category_vs_scenario.value_counts().to_dict(),boundary_status_counts=ann.boundary_status.value_counts().to_dict(),strata=strata)
    # Boundary/open/censored examples independently specify expected behavior.
    edges=[('1','S','BOUNDARY_ROBUST'),('2','S','BOUNDARY_SENSITIVE'),('8','I','BOUNDARY_SENSITIVE'),('16','R','BOUNDARY_SENSITIVE'),('32','R','BOUNDARY_ROBUST'),('>16','R','BOUNDARY_ROBUST'),('>=16','R','BOUNDARY_SENSITIVE'),('>8','R','BOUNDARY_SENSITIVE'),('<64','R','OBSERVED_INTERVAL_OR_CATEGORY_UNCERTAIN'),('≤0.06','S','BOUNDARY_ROBUST'),('</=0.125','S','UNSUPPORTED_MIC')]
    for raw,label,status in edges:assert classify(raw,label)['boundary_status']==status and oracle(raw,label)==status;checks+=1
    pd.DataFrame(rows).to_csv(a.out/'per_case_MIC_boundary_audit.csv',index=False);pd.DataFrame(all_metrics).to_csv(a.out/'stratum_metrics.csv',index=False)
    np.savez_compressed(a.out/'paired_country_bootstrap_metrics.npz',**arrays)
    report=dict(status='COMPLETE_POSTHOC_MIC_BOUNDARY_FIXED_PREDICTION_SENSITIVITY',protocol_sha256=sha(protocol),cohorts=reports,independent_interval_and_metric_checks=checks,
        limits=json.loads(protocol.read_text())['limits'],no_labels_or_predictions_changed=True)
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,r in reports.items():
        print(name,r['MIC_category_checks'],r['boundary_status_counts'])
        for stratum,info in r['strata'].items():
            if info['n']:print(stratum,info['n'],info['methods']['frozen_ML'],info['paired_country_composition'])

if __name__=='__main__':main()
