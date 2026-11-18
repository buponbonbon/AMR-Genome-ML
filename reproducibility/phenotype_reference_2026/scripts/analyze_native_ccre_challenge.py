#!/usr/bin/env python3
"""Audit frozen native CCRE scores against unchanged publisher phenotype cells."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score)

from file11c_blind_external_validation import parse_amrfinder_tsv, feature_vector_from_keys
from file11e_external_simple_baseline_comparators import call_fixed_rules

LABEL = 'S/I/R (PHAS interpretation)_Meropenem'
BLA = 'Bla_Carb_acquired (with new allele designations)'
METRICS = ['TP','FN','TN','FP','sensitivity','specificity','balanced_accuracy','MCC','AUROC','average_precision']


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj): p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def weighted(y, scores, w):
    """Vectorized weighted confusion, tied AUROC and stepwise average precision."""
    w = np.atleast_2d(w).astype(float); y = np.asarray(y, int); s = np.asarray(scores, float)
    pred = s >= .5; positive = w @ y; negative = w @ (1-y)
    tp = w @ (y * pred); fp = w @ ((1-y)*pred); tn = negative-fp; fn = positive-tp
    sensitivity = np.divide(tp, positive, out=np.full_like(tp,np.nan), where=positive>0)
    specificity = np.divide(tn, negative, out=np.full_like(tp,np.nan), where=negative>0)
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = np.divide(tp*tn-fp*fn, denom, out=np.zeros_like(tp), where=denom>0)
    ix = np.argsort(s, kind='stable'); starts = np.r_[0, np.flatnonzero(np.diff(s[ix]))+1]
    pw = np.add.reduceat(w[:,ix]*y[ix], starts, axis=1)
    nw = np.add.reduceat(w[:,ix]*(1-y[ix]), starts, axis=1)
    below = np.cumsum(nw, axis=1)-nw
    auc = np.divide((pw*(below+.5*nw)).sum(axis=1), positive*negative,
                    out=np.full_like(tp,np.nan), where=positive*negative>0)
    pdsc, ndsc = pw[:,::-1], nw[:,::-1]
    pp, total = np.cumsum(pdsc,axis=1), np.cumsum(pdsc+ndsc,axis=1)
    precision = np.divide(pp,total,out=np.zeros_like(pp),where=total>0)
    ap = np.divide((precision*pdsc).sum(axis=1),positive,out=np.full_like(tp,np.nan),where=positive>0)
    return np.column_stack([tp,fn,tn,fp,sensitivity,specificity,(sensitivity+specificity)/2,mcc,auc,ap])


def point(y, score):
    v = weighted(y,score,np.ones(len(y)))[0]
    return dict(n=len(y), R=int(np.sum(y)), S=int(len(y)-np.sum(y)),
                **{k:int(v[i]) if i<4 else (float(v[i]) if np.isfinite(v[i]) else None) for i,k in enumerate(METRICS)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['native','manifest','workbook','coefficients','specification','legacy','out']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--draws',type=int,default=10000)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    status=json.loads((a.native/'status.json').read_text())
    if not status['status'].startswith('COMPLETE_'): raise ValueError('Native caller incomplete')
    m=pd.read_csv(a.manifest,dtype=str,keep_default_na=False)
    d=pd.read_csv(a.native/'frozen_challenge_predictions.csv',dtype=str,keep_default_na=False)
    if d.sample_accession.tolist()!=m.sample_accession.tolist() or not d.sample_accession.is_unique:
        raise ValueError('Fixed manifest membership/order changed')
    for col in m:
        if not d[col].equals(m[col]): raise ValueError('Manifest source cell changed: '+col)
    if sha(a.workbook)!='a486d288842447fed0d50d97ed9e78471b8a2c671fc6015f95bee230d5c2443a':
        raise ValueError('Publisher workbook hash differs')
    raw=pd.read_excel(a.workbook,dtype=str,keep_default_na=False).set_index('Run accession')
    for _,row in d.iterrows():
        original=raw.loc[row['Run accession']]
        for col in [LABEL,'MIC_Meropenem',BLA,'Country of origin','Sample name']:
            if row[col]!=original[col]: raise ValueError('Publisher source cell differs: '+col)
    if not d[LABEL].isin(['S','R']).all(): raise ValueError('Nonbinary reference cannot be recoded')
    scorecol='frozen_probability_R'; available=d[scorecol].ne('')
    complete=d.loc[available].copy(); missing=d.loc[~available].copy()
    if len(complete)!=status['scored']: raise ValueError('Scored count differs')
    coef=pd.read_csv(a.coefficients); spec=json.loads(a.specification.read_text())
    if spec['model_sha256']!='1f55620cbd4b1dfd946ff37977e9a851b6aeaad8a5466eb553323331613cdcdd':
        raise ValueError('Frozen model specification differs')
    features=coef.feature.tolist()
    if hashlib.sha256(('\n'.join(features)+'\n').encode()).hexdigest()!=spec['feature_schema_sha256']:
        raise ValueError('Schema mismatch')
    z=np.load(a.native/'native_features.npz',allow_pickle=False)
    if z['biosamples'].tolist()!=complete.sample_accession.tolist() or z['features'].tolist()!=features:
        raise ValueError('Native matrix identities differ')
    rebuilt=[]; differences=[]
    # Independently parse every raw TSV and recompute scores and both original rules.
    for _,row in complete.iterrows():
        path=a.native/'amrfinder'/(row.sample_accession+'.tsv')
        x=feature_vector_from_keys(parse_amrfinder_tsv(path),features).astype(np.uint8)
        calls,_=call_fixed_rules(path)
        for name in ['kpc_family_rule','amrfinder_carbapenemase_rule']:
            if int(float(row[name]))!=calls[name]: raise ValueError('Recomputed frozen rule differs')
        receipt=a.native/'receipts'/(row.sample_accession+'.json')
        if receipt.exists() and json.loads(receipt.read_text())['output_sha256']!=sha(path):
            raise ValueError('Caller receipt hash mismatch')
        rebuilt.append(x)
    x=np.asarray(rebuilt)
    if not np.array_equal(x,z['X']): raise ValueError('Rebuilt matrix differs')
    recomputed=expit(x@coef.coefficient.to_numpy()+spec['intercept'])
    score=complete[scorecol].astype(float).to_numpy()
    maxdiff=float(np.max(np.abs(score-recomputed)))
    if maxdiff>1e-12: raise ValueError('Independent frozen linear score differs')
    y=complete[LABEL].map({'S':0,'R':1}).to_numpy(int)
    if not np.array_equal((score>=.5).astype(int),complete.frozen_predicted_R.astype(float).to_numpy(int)):
        raise ValueError('Frozen threshold differs')
    methods={'frozen_ML':score,'frozen_broad_annotation_rule':complete.amrfinder_carbapenemase_rule.astype(float).to_numpy(),
             'frozen_KPC_rule':complete.kpc_family_rule.astype(float).to_numpy()}
    source=complete[BLA].str.strip()
    if source.isin(['','Missing','NULL','NA','Not tested']).any():
        raise ValueError('Source comparator missingness requires separate denominator')
    methods['author_Kleborate_carbapenemase_presence']=(source!='-').to_numpy(float)
    points={name:point(y,s) for name,s in methods.items()}
    pd.DataFrame([dict(method=k,**v) for k,v in points.items()]).to_csv(a.out/'paired_point_metrics.csv',index=False)
    for k,s in methods.items(): complete[k+'_predicted_R']=(s>=.5).astype(int)
    broad=methods['frozen_broad_annotation_rule'].astype(int); pred=(score>=.5).astype(int)
    joint=[]
    for ref,ml,rule in itertools.product([0,1],repeat=3):
        joint.append({'reference':ref,'model':ml,'broad_rule':rule,'n':int(((y==ref)&(pred==ml)&(broad==rule)).sum())})
    pd.DataFrame(joint).to_csv(a.out/'joint_decision_counts.csv',index=False)
    complete.to_csv(a.out/'available_case_audit.csv',index=False)
    complete[(pred!=y)|(broad!=y)].to_csv(a.out/'model_or_rule_discordant_cases_original_MIC.csv',index=False)
    missing.to_csv(a.out/'unscored_fixed_cases.csv',index=False)
    # Paired source-composition sensitivity; no winner selection or threshold fitting.
    rng=np.random.default_rng(20261111); bootstrap={}; verification=[]
    for grouping,col in [('PDS','PDS_bases'),('country','Country of origin')]:
        values=complete[col].to_numpy(); unique=np.unique(values); membership=(values[None,:]==unique[:,None]).astype(int)
        counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=a.draws)
        weights=counts@membership
        valid=(weights@y>0)&(weights@(1-y)>0); weights=weights[valid]
        arrays={k:weighted(y,s,weights) for k,s in methods.items()}
        for j in range(min(10,len(weights))):
            w=weights[j]
            for k,s in methods.items():
                v=arrays[k][j]; h=s>=.5
                expected=[balanced_accuracy_score(y,h,sample_weight=w),matthews_corrcoef(y,h,sample_weight=w),
                          roc_auc_score(y,s,sample_weight=w),average_precision_score(y,s,sample_weight=w)]
                if not np.allclose(v[[6,7,8,9]],expected,atol=1e-12): raise ValueError('Independent weighted metric check failed')
                verification.append({'group':grouping,'draw':j,'method':k,'passed':True})
        results={}
        for k,v in arrays.items():
            results[k]={name:{'percentile_2_5':float(np.quantile(v[:,i],.025)),
                              'percentile_97_5':float(np.quantile(v[:,i],.975))} for i,name in enumerate(METRICS[4:],4)}
        contrasts={}
        for k in methods:
            if k=='frozen_ML': continue
            delta=arrays['frozen_ML']-arrays[k]
            contrasts['ML_minus_'+k]={name:{'point':points['frozen_ML'][name]-points[k][name],
                 'percentile_2_5':float(np.quantile(delta[:,i],.025)),
                 'percentile_97_5':float(np.quantile(delta[:,i],.975))} for i,name in enumerate(METRICS[4:],4)}
        bootstrap[grouping]={'groups':len(unique),'requested_draws':a.draws,'valid_draws':len(weights),
                            'point_percentiles':results,'paired_contrasts':contrasts}
        np.savez_compressed(a.out/(grouping+'_bootstrap_metrics.npz'),**arrays)
    st=[]
    for value,part in complete.groupby('ST_seen_in_fitting_File07'):
        ix=complete.index.get_indexer(part.index)
        for k,s in methods.items(): st.append(dict(ST_seen=value,method=k,**point(y[ix],s[ix])))
    pd.DataFrame(st).to_csv(a.out/'reported_ST_sensitivity.csv',index=False)
    # Fixed-target bounds retain missing cases rather than treating NA as negative.
    bounds={}; ym=missing[LABEL].map({'S':0,'R':1}).to_numpy(int); full_y=np.r_[y,ym]
    if len(missing)>12: raise ValueError('Too many missing cases for enumerated bounds')
    for k,s in methods.items():
        if k=='author_Kleborate_carbapenemase_presence':
            extra=(missing[BLA].str.strip()!='-').to_numpy(float)
            bounds[k]={'source_call_available_for_full_target':point(full_y,np.r_[s,extra])}; continue
        possibilities=[point(full_y,np.r_[s,extra]) for extra in itertools.product([0,1],repeat=len(missing))]
        bounds[k]={'n':len(full_y),'unscored':len(missing),'ranges_over_missing_predictions_or_extreme_scores':
                   {name:{'min':min(v[name] for v in possibilities),'max':max(v[name] for v in possibilities)} for name in METRICS}}
    legacy=pd.read_csv(a.legacy,dtype=str,keep_default_na=False).set_index('sample_accession')
    older=legacy.loc[complete.sample_accession,'exploratory_frozen_probability_R_from_Jan2026_features'].astype(float).to_numpy()
    npz=a.legacy.parent/'legacy_projected_features.npz'; oldx=np.load(npz,allow_pickle=False)
    lookup={s:i for i,s in enumerate(oldx['biosamples'])}; ox=oldx['X'][[lookup[s] for s in complete.sample_accession]]
    if oldx['features'].tolist()!=features: raise ValueError('Legacy schema differs')
    comparison={'matched_n':len(complete),'feature_cells_changed':int((x!=ox).sum()),
        'samples_with_feature_changes':int((x!=ox).any(axis=1).sum()),
        'maximum_absolute_probability_difference':float(np.abs(score-older).max()),
        'hard_call_changes':int(((score>=.5)!=(older>=.5)).sum()),
        'older_projection_metrics_same_cases':point(y,older),
        'interpretation':'Joint assembly-source and feature-database difference; not an isolated database effect.'}
    pd.DataFrame({'sample_accession':complete.sample_accession,'older_probability_R':older,
                  'native_probability_R':score,'feature_cells_changed':(x!=ox).sum(axis=1),
                  'hard_call_changed':(score>=.5)!=(older>=.5)}).to_csv(a.out/'same_case_processing_comparison.csv',index=False)
    report={'status':'COMPLETE_INDEPENDENT_RECOMPUTATION_AND_PAIRED_SOURCE_REFERENCE_ANALYSIS',
        'fixed_target':len(d),'scored':len(complete),'unscored':len(missing), 'threshold':.5,
        'reference':'Original publisher PHAS meropenem S/R with original MIC cells, heterogeneous AST; not repeat reference AST.',
        'methods':points,'joint_decisions':joint,'source_composition_sensitivity':bootstrap,
        'missing_input_bounds':bounds,'processing_comparison':comparison,
        'verification':{'original_publisher_cells_match':True,'all_native_matrices_and_rules_recomputed':True,
          'maximum_manual_probability_difference':maxdiff,'weighted_metric_sklearn_checks':len(verification)},
        'input_sha256':{name:sha(path) for name,path in [('manifest',a.manifest),('workbook',a.workbook),
           ('native_predictions',a.native/'frozen_challenge_predictions.csv'),('coefficients',a.coefficients),
           ('score_specification',a.specification),('legacy_predictions',a.legacy)]},
        'limitations':['Post-hoc selected source; not prospective validation or random clinical population.',
          'One-per-PDS filter only excludes sharing with mapped development isolates (2944/4218 BioSamples).',
          '40-case staging subset is included and is not an independent replication.',
          'BioSample/Run-linked AllTheBacteria Shovill assemblies differ from the original author assembly source.',
          'No exact fitting/E2 BioSample overlap detected; universal genomic independence is not established.',
          'Country and PDS percentile ranges describe selected source composition, not future-patient confidence intervals.',
          'Frozen broad rule includes carbapenem-associated porin disruptions, not only carbapenemases.',
          'Author Kleborate presence control has different annotation/assembly provenance; keep separate from native rules.',
          'No classifier, threshold, frozen rule, eligibility or source phenotype was optimized or changed.']}
    save(a.out/'report.json',report)
    print(json.dumps({k:report[k] for k in ['status','fixed_target','scored','unscored','methods','verification','processing_comparison']},indent=2))


if __name__=='__main__': main()
