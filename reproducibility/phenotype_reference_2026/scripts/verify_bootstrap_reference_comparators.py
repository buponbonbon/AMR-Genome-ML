#!/usr/bin/env python3
"""Independently verify project bootstrap weights, scalar metrics and reported ranges."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,roc_auc_score,average_precision_score


def close(a,b):
    if not np.isclose(a,b,atol=1e-12,rtol=0,equal_nan=True):raise ValueError(f'Independent metric mismatch {a} vs {b}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-counts',type=Path,required=True);p.add_argument('--crosswalk',type=Path,required=True)
    p.add_argument('--bootstrap-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();source=list(csv.DictReader(args.source_counts.open()));by={r['bioproject_acc']:r for r in source}
    data=pd.read_csv(args.crosswalk);z=np.load(args.bootstrap_dir/'bootstrap_replicates.npz',allow_pickle=False)
    projects=z['projects'].tolist();pid={p:i for i,p in enumerate(projects)};focal=projects.index('PRJNA1322038')
    source_index=np.array([pid[p] for p in data.project]);sizes=np.array([int(by[p]['n']) for p in projects])
    n_checks=0;records=[]
    for scheme in ['all_projects','focal_retained']:
        weights=z['weights__'+scheme].astype(int)
        assert weights.shape==(10000,32) and (weights>=0).all() and np.all(weights.sum(axis=1)==32)
        assert np.array_equal(weights@sizes,z['n__'+scheme])
        if scheme=='focal_retained':assert np.all(weights[:,focal]==1)
        selected=sorted({0,1,2,len(weights)//2,len(weights)-1,*np.random.default_rng(197709).choice(len(weights),30,replace=False).tolist()})
        for j in selected:
            cluster_weights=weights[j];sample_weights=cluster_weights[source_index].astype(float)
            for ref,col in [('historical','historical_label'),('october','scenario_label')]:
                truth=data[col].eq('R').astype(int).to_numpy();pred=data.probability.ge(.5).astype(int).to_numpy()
                for method in ['ML','rule']:
                    if method=='ML':
                        y,q,score,w=truth,pred,data.probability.to_numpy(),sample_weights
                        tp=float(w[(y==1)&(q==1)].sum());fn=float(w[(y==1)&(q==0)].sum())
                        tn=float(w[(y==0)&(q==0)].sum());fp=float(w[(y==0)&(q==1)].sum())
                    else:
                        cm=dict(tp=0,fn=0,tn=0,fp=0)
                        for project,copies in zip(projects,cluster_weights):
                            r=by[project];mapping={'tp':'fp','fn':'tn','tn':'fn','fp':'tp'} if ref=='october' and project=='PRJNA1322038' else {k:k for k in cm}
                            for k in cm:cm[k]+=copies*int(r['rule_'+mapping[k]])
                        tp,fn,tn,fp=(cm[k] for k in ['tp','fn','tn','fp'])
                        y=np.array([1,1,0,0]);q=np.array([1,0,0,1]);score=q;w=np.array([tp,fn,tn,fp],dtype=float)
                    metrics={}
                    if tp+fn>0 and tn+fp>0:
                        metrics['balanced_accuracy']=balanced_accuracy_score(y,q,sample_weight=w)
                        metrics['auroc']=roc_auc_score(y,score,sample_weight=w)
                    else:metrics['balanced_accuracy']=metrics['auroc']=np.nan
                    metrics['mcc']=matthews_corrcoef(y,q,sample_weight=w) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)>0 else np.nan
                    metrics['average_precision']=average_precision_score(y,score,sample_weight=w) if tp+fn>0 else np.nan
                    metrics['false_susceptible_rate']=fn/(tp+fn) if tp+fn else np.nan
                    metrics['false_resistant_rate']=fp/(tn+fp) if tn+fp else np.nan
                    metrics['false_susceptible_fraction_called_S']=fn/(tn+fn) if tn+fn else np.nan
                    metrics['accuracy']=(tp+tn)/(tp+fn+tn+fp)
                    for metric,value in metrics.items():close(value,z[f'{scheme}__{ref}__{method}__{metric}'][j]);n_checks+=1
        absent=weights[:,focal]==0
        for metric in ['balanced_accuracy','mcc','auroc','average_precision','false_susceptible_rate','false_resistant_rate','false_susceptible_fraction_called_S','accuracy']:
            for ref in ['historical','october']:
                delta=z[f'{scheme}__{ref}__ML__{metric}']-z[f'{scheme}__{ref}__rule__{metric}']
                assert np.allclose(delta,z[f'{scheme}__{ref}__delta__{metric}'],atol=1e-12,rtol=0,equal_nan=True)
            a=z[f'{scheme}__historical__delta__{metric}'];b=z[f'{scheme}__october__delta__{metric}']
            assert np.allclose(b-a,z[f'{scheme}__interaction__{metric}'],atol=1e-12,rtol=0,equal_nan=True)
            assert np.allclose(a[absent],b[absent],atol=1e-12,rtol=0,equal_nan=True)
        records.append({'scheme':scheme,'draws_checked_for_weights':len(weights),'independent_scalar_draws':len(selected),'focal_absent_draws':int(absent.sum())})
    summary=pd.read_csv(args.bootstrap_dir/'paired_contrast_resampling_ranges.csv')
    for r in summary.itertuples():
        key=f'{r.scheme}__interaction__{r.metric}' if r.reference=='interaction_october_minus_historical' else f'{r.scheme}__{r.reference}__delta__{r.metric}'
        a=z[key];valid=a[np.isfinite(a)];q=np.quantile(valid,[.025,.5,.975])
        assert r.valid_draws==len(valid)
        for x,y in zip([r.q025,r.median,r.q975],q):close(x,y)
    report={'status':'PASS','method':'Independent sklearn weighted scalar checks without importing production bootstrap functions, full resampling-weight and contrast/range reconstruction',
            'schemes':records,'scalar_metric_checks':n_checks,'resampling_summary_rows_verified':len(summary),
            'limitation':'Numerical verification of source-composition resampling; it does not validate biological ground truth or population inference.'}
    with args.out.open('x') as h:json.dump(report,h,indent=2);h.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
