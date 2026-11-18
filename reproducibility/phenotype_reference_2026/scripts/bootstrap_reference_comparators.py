#!/usr/bin/env python3
"""Paired source-cluster resampling of frozen method comparisons, not clinical validation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_exact_rule_tradeoffs import joint_cells, confusion

METRICS=['balanced_accuracy','mcc','auroc','average_precision','false_susceptible_rate',
         'false_resistant_rate','false_susceptible_fraction_called_S','accuracy']
FOCAL='PRJNA1322038'


def divide(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)!=0)


def cm_metrics(cm):
    tp,fn,tn,fp=cm.T.astype(float)
    R,S=tp+fn,tn+fp
    tpr,tnr=divide(tp,R),divide(tn,S)
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {'balanced_accuracy':(tpr+tnr)/2,'mcc':divide(tp*tn-fp*fn,den),
            'false_susceptible_rate':divide(fn,R),'false_resistant_rate':divide(fp,S),
            'false_susceptible_fraction_called_S':divide(fn,tn+fn),'accuracy':divide(tp+tn,R+S)}


def score_metrics(weights,positive_groups,total_groups,batch_size=500):
    auc,ap=[],[]
    for start in range(0,len(weights),batch_size):
        w=weights[start:start+batch_size].astype(float)
        positive=w@positive_groups;total=w@total_groups;negative=total-positive
        P,N=positive.sum(axis=1),negative.sum(axis=1)
        negative_before=np.cumsum(negative,axis=1)-negative
        auc.append(divide(np.sum(positive*(negative_before+.5*negative),axis=1),P*N))
        p_desc=positive[:,::-1];t_desc=total[:,::-1]
        precision=divide(np.cumsum(p_desc,axis=1),np.cumsum(t_desc,axis=1))
        # Unobserved score groups have zero contribution, including leading empty groups.
        ap.append(divide(np.sum(np.where(p_desc>0,p_desc*np.nan_to_num(precision,nan=0.),0.),axis=1),P))
    return np.concatenate(auc),np.concatenate(ap)


def prepare(crosswalk,source_counts):
    data=pd.read_csv(crosswalk)
    source=list(csv.DictReader(source_counts.open()))
    projects=sorted(r['bioproject_acc'] for r in source)
    if len(projects)!=32 or len(set(projects))!=32 or len(data)!=525 or data.biosample.nunique()!=525:
        raise ValueError('Unexpected preserved cohort/source coverage')
    if set(projects)!=set(data.project):raise ValueError('Source IDs differ')
    by={r['bioproject_acc']:r for r in source};indices={p:i for i,p in enumerate(projects)}
    n=np.array([int(by[p]['n']) for p in projects]);cm={}
    for reference,label_column in [('historical','historical_label'),('october','scenario_label')]:
        ml=[];rule=[]
        for project in projects:
            row=by[project];cells=joint_cells(row)
            if reference=='october' and project==FOCAL:cells={(1-y,m,r):v for (y,m,r),v in cells.items()}
            for axis,target in [(1,ml),(2,rule)]:target.append([confusion(cells,axis)[k] for k in ['tp','fn','tn','fp']])
            subset=data[data.project==project];truth=subset[label_column].eq('R').to_numpy();pred=subset.probability.ge(.5).to_numpy()
            native=np.array([(truth&pred).sum(),(truth&~pred).sum(),(~truth&~pred).sum(),(~truth&pred).sum()])
            if not np.array_equal(native,ml[-1]):raise ValueError('Source ML counts differ from frozen predictions')
        cm[reference]={'ML':np.array(ml),'rule':np.array(rule)}
    _,inverse=np.unique(data.probability.to_numpy(float),return_inverse=True)
    groups=len(set(inverse));total=np.zeros((32,groups));positive={}
    project_index=np.array([indices[p] for p in data.project])
    np.add.at(total,(project_index,inverse),1)
    for ref,col in [('historical','historical_label'),('october','scenario_label')]:
        positive[ref]=np.zeros_like(total)
        np.add.at(positive[ref],(project_index,inverse),data[col].eq('R').to_numpy(int))
    return data,projects,n,cm,positive,total


def evaluate(weights,cm,positive,total):
    result={}
    for reference in ['historical','october']:
        for method in ['ML','rule']:
            matrix=weights@cm[reference][method]
            metrics=cm_metrics(matrix)
            if method=='ML':
                metrics['auroc'],metrics['average_precision']=score_metrics(weights,positive[reference],total)
            else:
                tp,fn,tn,fp=matrix.T.astype(float);P,N=tp+fn,tn+fp
                recall=divide(tp,P);precision=divide(tp,tp+fp)
                metrics['auroc']=(recall+divide(tn,N))/2
                # Zero positive predictions contribute no retrieved positives.
                metrics['average_precision']=recall*np.nan_to_num(precision,nan=0)+(1-recall)*divide(P,P+N)
            result[(reference,method)]=metrics
    return result


def summarize(values):
    valid=np.isfinite(values);a=values[valid]
    if not len(a):return {'valid_draws':0,'invalid_draws':len(values),'q025':None,'median':None,'q975':None,'fraction_positive':None,'fraction_negative':None}
    q=np.percentile(a,[2.5,50,97.5])
    return {'valid_draws':int(valid.sum()),'invalid_draws':int((~valid).sum()),'q025':float(q[0]),'median':float(q[1]),'q975':float(q[2]),
            'fraction_positive':float(np.mean(a>1e-12)),'fraction_negative':float(np.mean(a< -1e-12))}


def write_csv(path,rows):
    with path.open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--source-counts',type=Path,required=True)
    p.add_argument('--crosswalk',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();protocol=json.loads(args.protocol.read_text());args.out.mkdir(parents=True,exist_ok=False)
    for name,path in [('source_counts',args.source_counts),('crosswalk',args.crosswalk)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['inputs'][name]['sha256']:raise ValueError('Predeclared input hash mismatch')
    data,projects,n,cm,positive,total=prepare(args.crosswalk,args.source_counts)
    B=protocol['replicates_per_scheme'];rng=np.random.default_rng(protocol['seed']);focal_index=projects.index(FOCAL)
    weights={};weights['all_projects']=rng.multinomial(32,np.full(32,1/32),size=B)
    background=[i for i in range(32) if i!=focal_index]
    conditional=np.zeros((B,32),dtype=int);conditional[:,focal_index]=1
    conditional[:,background]=rng.multinomial(31,np.full(31,1/31),size=B);weights['focal_retained']=conditional
    point=evaluate(np.ones((1,32),dtype=int),cm,positive,total)
    arrays={'projects':np.array(projects),'source_sizes':n,'focal_index':np.array(focal_index)}
    summaries=[];flips=[];points=[];method_ranges=[];scheme_info={}
    for ref in ['historical','october']:
        for method in ['ML','rule']:
            points.append({'reference':ref,'method':method,**{m:float(point[(ref,method)][m][0]) for m in METRICS}})
    for scheme,w in weights.items():
        assert np.all(w.sum(axis=1)==32)
        if scheme=='focal_retained':assert np.all(w[:,focal_index]==1)
        arrays['weights__'+scheme]=w.astype(np.uint16);arrays['n__'+scheme]=w@n
        values=evaluate(w,cm,positive,total)
        deltas={}
        for ref in ['historical','october']:
            for metric in METRICS:
                for method in ['ML','rule']:
                    a=values[(ref,method)][metric];arrays[f'{scheme}__{ref}__{method}__{metric}']=a
                    method_ranges.append({'scheme':scheme,'reference':ref,'method':method,'metric':metric,'point':float(point[(ref,method)][metric][0]),**summarize(a)})
                delta=values[(ref,'ML')][metric]-values[(ref,'rule')][metric];deltas[(ref,metric)]=delta
                arrays[f'{scheme}__{ref}__delta__{metric}']=delta
                summaries.append({'scheme':scheme,'reference':ref,'metric':metric,
                                  'point_ML_minus_rule':float(point[(ref,'ML')][metric][0]-point[(ref,'rule')][metric][0]),**summarize(delta)})
        for metric in METRICS:
            a,b=deltas[('historical',metric)],deltas[('october',metric)]
            interaction=b-a;arrays[f'{scheme}__interaction__{metric}']=interaction
            summaries.append({'scheme':scheme,'reference':'interaction_october_minus_historical','metric':metric,
                              'point_ML_minus_rule':float((point[('october','ML')][metric][0]-point[('october','rule')][metric][0])-(point[('historical','ML')][metric][0]-point[('historical','rule')][metric][0])),**summarize(interaction)})
            valid=np.isfinite(a)&np.isfinite(b)
            flips.append({'scheme':scheme,'metric':metric,'valid_draws':int(valid.sum()),
                          'fraction_historical_rule_higher_october_ML_higher':float(np.mean((a[valid]< -1e-12)&(b[valid]>1e-12))),
                          'fraction_any_strict_order_reversal':float(np.mean(a[valid]*b[valid]< -1e-24))})
        omitted=w[:,focal_index]==0
        if omitted.any():
            for metric in METRICS:
                valid=omitted&np.isfinite(deltas[('historical',metric)])&np.isfinite(deltas[('october',metric)])
                if not np.allclose(deltas[('historical',metric)][valid],deltas[('october',metric)][valid],rtol=0,atol=1e-12):raise ValueError('References differ despite omitting the only changed source')
        scheme_info[scheme]={'draws':B,'focal_absent_draws':int(omitted.sum()),'focal_absent_fraction':float(omitted.mean()),
                             'resampled_total_n_min':int((w@n).min()),'resampled_total_n_max':int((w@n).max())}
    write_csv(args.out/'point_metrics.csv',points);write_csv(args.out/'paired_contrast_resampling_ranges.csv',summaries)
    write_csv(args.out/'method_resampling_ranges.csv',method_ranges);write_csv(args.out/'metric_order_reversal_fractions.csv',flips)
    np.savez_compressed(args.out/'bootstrap_replicates.npz',**arrays)
    report={'status':'PAIRED_PROJECT_BOOTSTRAP_SOURCE_COMPOSITION_SENSITIVITY_NOT_POPULATION_CERTIFICATION',
            'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seed':protocol['seed'],'schemes':scheme_info,'expected_focal_absent_probability_primary':(31/32)**32,
            'interpretation':'Percentile ranges and order fractions describe the specified project-resampling experiment; they are not posterior probabilities, clinical error-rate confidence guarantees or independent replications.',
            'limitations':protocol['limitations'],'input_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('source_counts',args.source_counts),('crosswalk',args.crosswalk)]}}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
