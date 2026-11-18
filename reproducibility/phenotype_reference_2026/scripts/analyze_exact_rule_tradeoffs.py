#!/usr/bin/env python3
"""Exact aggregate fixed-decision tradeoffs from preserved source-level joint counts.

Does not reconstruct per-isolate rule predictions or adjudicate clinical truth.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import csv

FOCAL="PRJNA1322038"


def joint_cells(row):
    c={key:0 for key in itertools.product([0,1],repeat=3)}
    c[1,0,0]=int(row['both_false_susceptible'])
    c[1,0,1]=int(row['rule_extra_resistant_true'])
    c[1,1,1]=int(row['rule_tp'])-c[1,0,1]
    c[1,1,0]=int(row['ml_tp'])-c[1,1,1]
    c[0,0,1]=int(row['rule_extra_resistant_false'])
    c[0,0,0]=int(row['ml_tn'])-c[0,0,1]
    c[0,1,0]=int(row['rule_tn'])-c[0,0,0]
    c[0,1,1]=int(row['ml_fp'])-c[0,1,0]
    if min(c.values())<0 or sum(c.values())!=int(row['n']):
        raise ValueError('Inconsistent joint cells or source sample count')
    if sum(v for (y,m,r),v in c.items() if y==1)!=int(row['n_resistant']):raise ValueError('Source R margin differs')
    if sum(v for (y,m,r),v in c.items() if y==0)!=int(row['n_susceptible']):raise ValueError('Source S margin differs')
    for method,axis in [('ml',1),('rule',2)]:
        cm=confusion(c,axis)
        if any(cm[k]!=int(row[method+'_'+k]) for k in ['tp','fn','tn','fp']):raise ValueError('Joint cells differ from method margins')
    checks={
        'ml_correct_rule_wrong':sum(v for (y,m,r),v in c.items() if m==y and r!=y),
        'ml_wrong_rule_correct':sum(v for (y,m,r),v in c.items() if m!=y and r==y),
        'rule_positive_ml_negative':sum(v for (y,m,r),v in c.items() if r==1 and m==0),
        'ml_positive_rule_negative':sum(v for (y,m,r),v in c.items() if m==1 and r==0)}
    if any(v!=int(row[k]) for k,v in checks.items()):raise ValueError('Paired decision/correctness count mismatch')
    return c


def confusion(cells,axis):
    c=Counter(tp=0,fn=0,tn=0,fp=0)
    for key,n in cells.items():
        y,pred=key[0],key[axis]
        c['tp' if y and pred else 'fn' if y else 'fp' if pred else 'tn']+=n
    return dict(c)


def summarize(c):
    tp,fn,tn,fp=(c[k] for k in ['tp','fn','tn','fp'])
    R,S=tp+fn,tn+fp
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {**c,'n':R+S,'reference_R':R,'reference_S':S,'predicted_R':tp+fp,'predicted_S':tn+fn,
            'sensitivity':tp/R if R else None,'specificity':tn/S if S else None,
            'apparent_false_susceptible_rate_among_reference_R':fn/R if R else None,
            'apparent_false_resistant_rate_among_reference_S':fp/S if S else None,
            'apparent_false_susceptible_fraction_among_called_S':fn/(tn+fn) if tn+fn else None,
            'balanced_accuracy':(tp/R+tn/S)/2 if R and S else None,
            'mcc':(tp*tn-fp*fn)/den if den else None,'error_n':fn+fp,'accuracy':(tp+tn)/(R+S)}


def combine(sources,reference,exclude=None):
    out=Counter({key:0 for key in itertools.product([0,1],repeat=3)})
    for project,cells in sources.items():
        if project==exclude:continue
        for (y,m,r),n in cells.items():
            out[(1-y if reference=='october' and project==FOCAL else y,m,r)]+=n
    return out


def write_csv(path,rows):
    with path.open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-counts',type=Path,required=True)
    p.add_argument('--exact-metrics',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    rows=list(csv.DictReader(args.source_counts.open(encoding='utf-8-sig')))
    if len({r['bioproject_acc'] for r in rows})!=len(rows):raise ValueError('Duplicate sources')
    sources={r['bioproject_acc']:joint_cells(r) for r in rows}
    point_metrics=list(csv.DictReader(args.exact_metrics.open()))
    points=[];paired=[];joint_output=[]
    for reference in ['historical','october']:
        cells=combine(sources,reference)
        for method,axis in [('frozen_ML',1),('frozen_rule',2)]:
            metric=summarize(confusion(cells,axis))
            saved=next(r for r in point_metrics if r['method']==method and r['reference']==reference)
            if any(metric[k]!=int(saved[k]) for k in ['tp','fn','tn','fp']):raise ValueError('Exact replay differs from aggregate joints')
            points.append({'method':method,'reference':reference,**metric,'auroc':float(saved['auroc']),'average_precision':float(saved['average_precision'])})
        paired.append({'reference':reference,
                       'ML_correct_rule_wrong':sum(n for (y,m,r),n in cells.items() if m==y and r!=y),
                       'ML_wrong_rule_correct':sum(n for (y,m,r),n in cells.items() if m!=y and r==y),
                       'both_correct':sum(n for (y,m,r),n in cells.items() if m==y and r==y),
                       'both_wrong':sum(n for (y,m,r),n in cells.items() if m!=y and r!=y),
                       'rule_R_ML_S':sum(n for (y,m,r),n in cells.items() if r==1 and m==0),
                       'ML_R_rule_S':sum(n for (y,m,r),n in cells.items() if m==1 and r==0)})
        for (y,m,r),n in cells.items():joint_output.append({'reference':reference,'reference_label':y,'ML_prediction':m,'rule_prediction':r,'count':n})
    write_csv(args.out/'exact_error_metrics.csv',points)
    write_csv(args.out/'paired_correctness_counts.csv',paired)
    write_csv(args.out/'aggregate_joint_cells_NOT_per_isolate_predictions.csv',joint_output)
    sensitivity=[]
    for project in sources:
        for reference in ['historical','october']:
            cells=combine(sources,reference,exclude=project)
            ml=summarize(confusion(cells,1));rule=summarize(confusion(cells,2))
            sensitivity.append({'excluded_BioProject':project,'reference':reference,'n_remaining':ml['n'],
                                'ML_balanced_accuracy':ml['balanced_accuracy'],'rule_balanced_accuracy':rule['balanced_accuracy'],
                                'ML_minus_rule_balanced_accuracy':ml['balanced_accuracy']-rule['balanced_accuracy'],
                                'ML_mcc':ml['mcc'],'rule_mcc':rule['mcc'],'ML_minus_rule_mcc':ml['mcc']-rule['mcc'],
                                'ML_FN':ml['fn'],'rule_FN':rule['fn'],'ML_FP':ml['fp'],'rule_FP':rule['fp']})
    write_csv(args.out/'source_omission_sensitivity_NOT_retraining.csv',sensitivity)
    weighted=[];thresholds={}
    for reference in ['historical','october']:
        ml=next(r for r in points if r['method']=='frozen_ML' and r['reference']==reference)
        rule=next(r for r in points if r['method']=='frozen_rule' and r['reference']==reference)
        fn_difference=rule['fn']-ml['fn'];fp_difference=rule['fp']-ml['fp']
        thresholds[reference]={'rule_minus_ML_FN_coefficient':fn_difference,'rule_minus_ML_FP_constant':fp_difference,
                                'break_even_FN_to_FP_weight_ratio':str(Fraction(-fp_difference,fn_difference)) if fn_difference else None}
        for weight in [Fraction(1,2),Fraction(1),Fraction(11,10),Fraction(16,13),Fraction(2),Fraction(5),Fraction(10)]:
            difference=weight*fn_difference+fp_difference
            weighted.append({'reference':reference,'hypothetical_FN_to_FP_weight_ratio':float(weight),
                             'rule_minus_ML_weighted_error':float(difference),
                             'lower_weighted_error':'rule' if difference<0 else 'ML' if difference>0 else 'tie'})
    write_csv(args.out/'hypothetical_error_weight_scenarios_NOT_clinical_utilities.csv',weighted)
    focal_omission=[r for r in sensitivity if r['excluded_BioProject']==FOCAL]
    report={'status':'EXACT_AGGREGATE_TRADEOFF_AND_SOURCE_SENSITIVITY_COMPLETE_NOT_CLINICAL_VALIDATION',
            'source_BioProjects':len(sources),'cohort_n':525,'paired_correctness':paired,'error_weight_thresholds':thresholds,
            'focal_source_omission':focal_omission,
            'balanced_accuracy_ML_minus_rule_by_reference':{ref:next(r['balanced_accuracy'] for r in points if r['method']=='frozen_ML' and r['reference']==ref)-next(r['balanced_accuracy'] for r in points if r['method']=='frozen_rule' and r['reference']==ref) for ref in ['historical','october']},
            'current_ML_minus_rule_FN':next(r['fn'] for r in points if r['method']=='frozen_ML' and r['reference']=='october')-next(r['fn'] for r in points if r['method']=='frozen_rule' and r['reference']=='october'),
            'current_rule_minus_ML_FP':next(r['fp'] for r in points if r['method']=='frozen_rule' and r['reference']=='october')-next(r['fp'] for r in points if r['method']=='frozen_ML' and r['reference']=='october'),
            'missing_source_count_input_status':'RESOLVED_BY_UPLOADED_ORIGINAL_4030_BYTE_TABLE',
            'input_sha256':{args.source_counts.name:hashlib.sha256(args.source_counts.read_bytes()).hexdigest(),args.exact_metrics.name:hashlib.sha256(args.exact_metrics.read_bytes()).hexdigest()},
            'limitations':['Counts are exact with respect to the supplied frozen aggregate table and the two recorded references, not adjudicated clinical truth.',
                           'Joint cells represent aggregate counts; per-isolate rule calls have not been recovered or fabricated.',
                           'Source-omission analysis changes cohort composition with fixed predictions; it is not retraining, cross-validation or an independent validation.',
                           'FN:FP weights are illustrative decision scenarios; no clinical utilities, treatment benefits or harms were elicited.',
                           'One source event and post-hoc source selection do not support universal clinical superiority or independent replication.',
                           'Score-order metrics compare continuous ML probabilities with the documented binary 0/1 rule score.']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
