#!/usr/bin/env python3
"""Exact worst-case label-reassignment budgets for fixed BA/AUROC comparisons.

These are counterfactual finite-cohort stress tests, not estimates of label error.
No first-in-field algorithmic novelty is claimed.
"""
import argparse,hashlib,itertools,json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score,roc_auc_score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def linear_weights(metric,a,b):
    n=len(a)
    if metric=='balanced_accuracy':
        d=(np.asarray(a)>=.5).astype(int)-(np.asarray(b)>=.5).astype(int)
        return n*d-int(d.sum())
    if metric=='AUROC':
        return np.rint(2*(rankdata(a,method='average')-rankdata(b,method='average'))).astype(int)
    raise ValueError(metric)

def certificate(y,a,b,allowed,metric,fixed_totals):
    y=np.asarray(y,int);allowed=np.asarray(allowed,bool);w=linear_weights(metric,a,b)
    P=int(y.sum());N=len(y)-P
    if not P or not N:raise ValueError('Both reference classes required')
    L=int(y@w);direction=int(np.sign(L));flips=(1-2*y)*w
    result=dict(metric=metric,n=len(y),reference_R=P,reference_S=N,eligible_flip_cases=int(allowed.sum()),
        class_totals_fixed=fixed_totals,point_difference=Fraction(L,2*P*N).__float__(),
        integer_margin=L,initial_order='method1_above_method2' if L>0 else 'method1_below_method2' if L<0 else 'tie',
        minimum_label_reassignments_for_strict_reversal=None,witness_indices=[],certificate_status='INITIAL_TIE_NO_STRICT_ORDER')
    if direction==0:return result
    indices=np.flatnonzero(allowed)
    harmful=-direction*flips
    if fixed_totals:
        positive=sorted(indices[y[indices]==1],key=lambda i:(-int(harmful[i]),int(i)))
        negative=sorted(indices[y[indices]==0],key=lambda i:(-int(harmful[i]),int(i)))
        units=[(int(i),int(j)) for i,j in zip(positive,negative)]
    else:units=[(int(i),) for i in sorted(indices,key=lambda i:(-int(harmful[i]),int(i)))]
    margin=abs(L);chosen=[]
    for unit in units:
        margin-=sum(int(harmful[i]) for i in unit);chosen.extend(unit)
        if margin<0:
            yy=y.copy();yy[chosen]=1-yy[chosen]
            assert 0<yy.sum()<len(yy)
            if fixed_totals:assert yy.sum()==y.sum()
            after=int(yy@w)
            assert after*L<0
            result.update(minimum_label_reassignments_for_strict_reversal=len(chosen),
                witness_indices=chosen,witness_integer_margin=after,
                witness_point_difference=float(Fraction(after,2*int(yy.sum())*(len(y)-int(yy.sum())))),
                all_fewer_reassignments_cannot_strictly_reverse=True,
                certificate_status='EXACT_MINIMUM_WITH_ATTAINING_COUNTERFACTUAL_WITNESS')
            return result
    result['certificate_status']='NO_STRICT_REVERSAL_POSSIBLE_WITHIN_ALLOWED_MASK'
    return result

def direct_oracle(metric,y,a,b):
    y=list(map(int,y));P=sum(y);N=len(y)-P
    def score(s):
        if metric=='balanced_accuracy':
            tp=sum(z and q>=.5 for z,q in zip(y,s));tn=sum(not z and q<.5 for z,q in zip(y,s))
            return (Fraction(tp,P)+Fraction(tn,N))/2
        total=sum(2*(s[i]>s[j])+int(s[i]==s[j]) for i in range(len(y)) if y[i] for j in range(len(y)) if not y[j])
        return Fraction(total,2*P*N)
    return score(a)-score(b)

def verify_math():
    rng=np.random.default_rng(20261112);cases=0;assignments=0;library_checks=0
    for n in range(3,9):
        for trial in range(8):
            y=rng.integers(0,2,size=n);y[0]=0;y[1]=1
            a=rng.choice([0,.25,.5,.75,1],size=n);b=rng.choice([0,.25,.5,.75,1],size=n)
            allowed=rng.random(n)>.25
            for metric in ['balanced_accuracy','AUROC']:
                original=direct_oracle(metric,y,a,b);w=linear_weights(metric,a,b)
                assert original==Fraction(int(y@w),2*int(y.sum())*(n-int(y.sum())))
                library=balanced_accuracy_score(y,a>=.5)-balanced_accuracy_score(y,b>=.5) if metric=='balanced_accuracy' else roc_auc_score(y,a)-roc_auc_score(y,b)
                assert abs(float(original)-library)<1e-12;library_checks+=1
                for fixed in [False,True]:
                    oracle_min=None
                    for bits in itertools.product([0,1],repeat=int(allowed.sum())):
                        ix=np.flatnonzero(allowed)[np.asarray(bits,bool)];yy=y.copy();yy[ix]=1-yy[ix]
                        if not 0<yy.sum()<n or fixed and yy.sum()!=y.sum():continue
                        value=direct_oracle(metric,yy,a,b);assignments+=1
                        if value*original<0:oracle_min=len(ix) if oracle_min is None else min(oracle_min,len(ix))
                    r=certificate(y,a,b,allowed,metric,fixed)
                    if original:assert r['minimum_label_reassignments_for_strict_reversal']==oracle_min
                    else:assert r['initial_order']=='tie'
                    cases+=1
    # Same decisions / same score ordering are unchanged for every reference.
    y=np.array([0,0,1,1]);s=np.array([.1,.3,.7,.9])
    assert np.all(linear_weights('balanced_accuracy',s,s)==0) and np.all(linear_weights('AUROC',s,s)==0)
    return dict(status='PASS_EXHAUSTIVE_DIRECT_CONFUSION_PAIRWISE_RANK_ORACLES',certificate_scenarios=cases,
        enumerated_valid_assignments=assignments,independent_sklearn_point_checks=library_checks,
        hypothetical_controls_are_not_observed_label_changes=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    verification=verify_math();a.out.mkdir(parents=True,exist_ok=False);records=[];witnesses=[]
    analysis=a.root/'results/database_reference_factorial_20261111/final_analysis'
    for cohort in ['E2','CCRE']:
        path=analysis/cohort/'paired_case_audit.csv';d=pd.read_csv(path,keep_default_na=False)
        refs=['historical_label','scenario_label'] if cohort=='E2' else ['S/I/R (PHAS interpretation)_Meropenem']
        masks={'all_available_cases':np.ones(len(d),bool)}
        if cohort=='E2':masks.update(focal_project_only=d.project.eq('PRJNA1322038').to_numpy(),background_only=d.project.ne('PRJNA1322038').to_numpy())
        for db,ref,comparator,metric,maskname,fixed in itertools.product(['Jan','Aug'],refs,['broad_annotation_rule','KPC_rule'],['balanced_accuracy','AUROC'],masks,[False,True]):
            y=d[ref].eq('R').to_numpy(int);s=d['model_probability_'+db].to_numpy(float);t=d[comparator+'_'+db+'_R'].to_numpy(float)
            r=certificate(y,s,t,masks[maskname],metric,fixed)
            label=dict(cohort=cohort,database=db,reference=ref,method1='frozen_ML',method2=comparator,allowed_mask=maskname)
            case_indices=r.pop('witness_indices');record=dict(**label,**r);records.append(record)
            actual=direct_oracle(metric,y,s,t)
            assert abs(r['point_difference']-float(actual))<1e-12
            if case_indices:
                yy=y.copy();yy[case_indices]=1-yy[case_indices];after=direct_oracle(metric,yy,s,t)
                assert after*actual<0 and abs(float(after)-r['witness_point_difference'])<1e-12
                for i in case_indices:witnesses.append(dict(**label,metric=metric,class_totals_fixed=fixed,
                    sample_accession=d.iloc[i].sample_accession,original_reference='R' if y[i] else 'S',
                    hypothetical_reference='S' if y[i] else 'R',hypothetical_only_not_a_source_correction=True))
    pd.DataFrame(records).to_csv(a.out/'exact_ranking_reassignment_budgets.csv',index=False)
    pd.DataFrame(witnesses).to_csv(a.out/'COUNTERFACTUAL_WITNESSES_NOT_CORRECTIONS.csv',index=False)
    r=dict(status='COMPLETE_EXACT_COUNTERFACTUAL_FINITE_COHORT_RANKING_BUDGETS',verification=verification,scenarios=records,
        input_sha256={c:sha(analysis/c/'paired_case_audit.csv') for c in ['E2','CCRE']},script_sha256=sha(Path(__file__)),
        limits=['No reference labels, model scores, features, thresholds or source records are edited.',
            'The budget is an optimal worst-case counterfactual, not an error-rate estimate or evidence that the named witness labels are wrong.',
            'Independent per-label reassignment is a mathematical stress scenario; source-correlated errors may behave differently.',
            'The source-restricted masks and fixed-class-total version state their assumptions explicitly.',
            'Finite selected-cohort guarantees do not transfer automatically to patients or unobserved sources.',
            'AUROC comparisons of continuous ML scores and binary rules have different output resolution; BA is the primary operational contrast.',
            'This robustness analysis does not establish a second temporal source event or new clinical AST ground truth.',
            'Exact algebra and software verification do not establish first-in-field methodological novelty.'])
    (a.out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(verification,indent=2))
    keep=[x for x in records if x['database']=='Aug' and x['method2']=='broad_annotation_rule' and x['metric']=='balanced_accuracy' and x['allowed_mask']=='all_available_cases']
    for x in keep:print(json.dumps(x))

if __name__=='__main__':main()
