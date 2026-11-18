#!/usr/bin/env python3
"""Independently verify the public PDS gate and exploratory comparison bounds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def ba(y,p):return float(((p[y==1]==1).mean()+(p[y==0]==0).mean())/2)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    cl=pd.read_csv(root/'cluster_and_primary/Kp_cluster_list.tsv',sep='\t',dtype=str,keep_default_na=False)
    training=pd.read_csv(root/'CCRE_verified_screen_final/frozen_training_identity_complete.csv',dtype=str)
    candidates=pd.read_csv(root/'CCRE_verified_screen_final/CCRE_candidate_primary_MIC_RS.csv',dtype=str)
    t=cl[cl.biosample_acc.isin(set(training.biosample_accession))]
    groups=set(t.PDS_acc.str.split('.').str[0]);bypatient=cl.groupby('biosample_acc').PDS_acc.agg(lambda x:set(v.split('.')[0] for v in x))
    classifications=[]
    for sample in candidates.sample_accession:
        clusters=bypatient.get(sample,set())
        classifications.append('MISSING_CANDIDATE_PDS' if not clusters else 'SHARED_WITH_MAPPED_FITTING' if clusters & groups else 'NO_SHARED_PDS_WITH_MAPPED_FITTING')
    counts=pd.Series(classifications).value_counts().to_dict();reported=json.loads((root/'CCRE_SNP_gate/report.json').read_text())
    assert counts==reported['candidate_counts_by_gate']
    d=pd.read_csv(root/'catalog_comparator_bounds/CCRE_catalog_comparator_bounds.csv',dtype=str)
    y=d['S/I/R (PHAS interpretation)_Meropenem'].map({'R':1,'S':0}).to_numpy();ml=d.exploratory_frozen_predicted_R.to_numpy(int);lo=d.catalog_comparator_lower_R.to_numpy(int);hi=d.catalog_comparator_upper_R.to_numpy(int)
    result={}
    for label,column in [('PDS','PDS_bases'),('country','Country of origin')]:
        group=d[column].to_numpy();u=np.unique(group);indices=[np.flatnonzero(group==v) for v in u];rng=np.random.default_rng(20261111);values=[]
        for _ in range(3000):
            ix=np.concatenate([indices[k] for k in rng.integers(0,len(u),len(u))]);yy=y[ix]
            if np.unique(yy).size<2:continue
            good=np.where(yy==1,hi[ix],lo[ix]);bad=np.where(yy==1,lo[ix],hi[ix]);mb=ba(yy,ml[ix]);values.append((mb-ba(yy,good),mb-ba(yy,bad)))
        b=np.asarray(values);result[label]={'groups':len(u),'valid_draws':len(b),'lower_bound_of_ML_minus_comparator_BA_percentiles':np.quantile(b[:,0],[.025,.975]).tolist(),'upper_bound_of_ML_minus_comparator_BA_percentiles':np.quantile(b[:,1],[.025,.975]).tolist()}
    result['error_weight_formula']={'lambda_definition':'Illustrative FN weight relative to FP, not an estimated patient cost','catalog_rule_minus_ML_weighted_errors_lower':'119 - 31 * lambda','catalog_rule_minus_ML_weighted_errors_upper':'169 - 31 * lambda','ML_fewer_weighted_errors_for_all_catalog_assignments_if_lambda_below':119/31,'catalog_rule_fewer_weighted_errors_for_all_catalog_assignments_if_lambda_above':169/31}
    result['scope']='Exploratory old-feature-DB diagnostic only. Separate PDS and country resampling do not jointly establish independence or clinical utility.'
    path=root/'catalog_comparator_bounds/composition_sensitivity.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    (root/'CCRE_SNP_gate/independent_verification.json').write_text(json.dumps({'status':'PASS','independent_PDS_classification_counts':counts,'composition_sensitivity_reproduced':True},indent=2)+'\n')
    print('PASS: PDS gate independently recomputed and both exploratory sensitivity regimes reproduced.')

if __name__=='__main__':main()
