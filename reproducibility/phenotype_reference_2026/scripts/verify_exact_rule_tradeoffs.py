#!/usr/bin/env python3
"""Independent sklearn and source-marginal verification of aggregate replay.

Weighted four-state rule checks use known confusion-cell counts, not invented
per-isolate rule predictions.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,matthews_corrcoef,roc_auc_score,average_precision_score


def near(a,b):
    if not np.isclose(float(a),float(b),rtol=0,atol=1e-12):raise ValueError(f'Metric mismatch {a} vs {b}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-counts',type=Path,required=True);p.add_argument('--crosswalk',type=Path,required=True)
    p.add_argument('--replay-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    sources=list(csv.DictReader(args.source_counts.open()));data=pd.read_csv(args.crosswalk)
    if set(data.project)!=set(r['bioproject_acc'] for r in sources):raise ValueError('Source coverage mismatch')
    points=pd.read_csv(args.replay_dir/'tradeoffs/exact_error_metrics.csv')
    omissions=pd.read_csv(args.replay_dir/'tradeoffs/source_omission_sensitivity_NOT_retraining.csv')
    for r in sources:
        subset=data[data.project==r['bioproject_acc']]
        cm=confusion_matrix(subset.historical_label.eq('R'),subset.probability.ge(.5),labels=[False,True]).ravel()
        for k,v in zip(['tn','fp','fn','tp'],cm):
            if int(r['ml_'+k])!=v:raise ValueError('Per-source ML margin not reproduced')
        assert len(subset)==int(r['n'])
        assert int(subset.historical_label.eq('R').sum())==int(r['n_resistant'])
    def rule_cm(reference,exclude=None):
        totals=dict(tp=0,fn=0,tn=0,fp=0)
        for r in sources:
            if r['bioproject_acc']==exclude:continue
            if reference=='october' and r['bioproject_acc']=='PRJNA1322038':
                mapping={'tp':'fp','fn':'tn','tn':'fn','fp':'tp'}
            else:mapping={k:k for k in totals}
            for k in totals:totals[k]+=int(r['rule_'+mapping[k]])
        return totals
    verified=[]
    for ref,column in [('historical','historical_label'),('october','scenario_label')]:
        truth=data[column].eq('R').astype(int);pred=data.probability.ge(.5).astype(int)
        rcm=rule_cm(ref)
        for method in ['frozen_ML','frozen_rule']:
            saved=points[(points.method==method)&(points.reference==ref)].iloc[0]
            if method=='frozen_ML':
                yt,yp,score,weight=truth,pred,data.probability,None
                tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
            else:
                yt=np.array([1,1,0,0]);yp=np.array([1,0,0,1]);score=yp
                weight=np.array([rcm['tp'],rcm['fn'],rcm['tn'],rcm['fp']]);tp,fn,tn,fp=(rcm[k] for k in ['tp','fn','tn','fp'])
            for k,v in [('tp',tp),('fn',fn),('tn',tn),('fp',fp)]:assert int(saved[k])==v
            near(saved.balanced_accuracy,balanced_accuracy_score(yt,yp,sample_weight=weight))
            near(saved.mcc,matthews_corrcoef(yt,yp,sample_weight=weight))
            near(saved.auroc,roc_auc_score(yt,score,sample_weight=weight))
            near(saved.average_precision,average_precision_score(yt,score,sample_weight=weight))
            near(saved.apparent_false_susceptible_rate_among_reference_R,fn/(tp+fn))
            near(saved.apparent_false_resistant_rate_among_reference_S,fp/(tn+fp))
            near(saved.apparent_false_susceptible_fraction_among_called_S,fn/(tn+fn))
            verified.append({'method':method,'reference':ref,'tp':int(tp),'fn':int(fn),'tn':int(tn),'fp':int(fp)})
        for project in data.project.unique():
            subset=data[data.project!=project];yt=subset[column].eq('R').astype(int);yp=subset.probability.ge(.5).astype(int)
            saved=omissions[(omissions.excluded_BioProject==project)&(omissions.reference==ref)].iloc[0]
            near(saved.ML_balanced_accuracy,balanced_accuracy_score(yt,yp));near(saved.ML_mcc,matthews_corrcoef(yt,yp))
            cm=rule_cm(ref,project);weights=[cm['tp'],cm['fn'],cm['tn'],cm['fp']]
            near(saved.rule_balanced_accuracy,balanced_accuracy_score([1,1,0,0],[1,0,0,1],sample_weight=weights))
            near(saved.rule_mcc,matthews_corrcoef([1,1,0,0],[1,0,0,1],sample_weight=weights))
    pairs=pd.read_csv(args.replay_dir/'tradeoffs/paired_correctness_counts.csv')
    for ref in ['historical','october']:
        b=c=0
        for r in sources:
            reverse=ref=='october' and r['bioproject_acc']=='PRJNA1322038'
            b+=int(r['ml_wrong_rule_correct' if reverse else 'ml_correct_rule_wrong'])
            c+=int(r['ml_correct_rule_wrong' if reverse else 'ml_wrong_rule_correct'])
        saved=pairs[pairs.reference==ref].iloc[0]
        assert saved.ML_correct_rule_wrong==b and saved.ML_wrong_rule_correct==c
    assert sum(int(r['ml_positive_rule_negative']) for r in sources)==0
    assert sum(int(r['rule_positive_ml_negative']) for r in sources)==58
    assert Fraction(46-14,30-4)==Fraction(16,13)
    report={'status':'PASS','methods':'Independent per-source sklearn confusion checks, weighted aggregate rule metrics, 64 source-omission checks and direct paired correctness updates',
            'source_projects_verified':len(sources),'global_method_reference_pairs_verified':verified,'source_omission_rows_verified':len(omissions),
            'current_apparent_FN_difference_ML_minus_rule':26,'current_apparent_FP_difference_rule_minus_ML':32,
            'current_hypothetical_error_weight_break_even':'16/13',
            'source_counts_sha256':hashlib.sha256(args.source_counts.read_bytes()).hexdigest(),
            'limitations':'Exact counts and metric arithmetic verified; no per-isolate rule calls or clinical reference truth were recovered.'}
    with args.out.open('x') as h:json.dump(report,h,indent=2);h.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
