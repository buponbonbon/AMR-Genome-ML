#!/usr/bin/env python3
"""Independent closed-form controls for the database/reference interaction axes."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,roc_auc_score,average_precision_score
from analyze_frozen_database_reference_factorial import contrasts,NAMES
from analyze_native_ccre_challenge import weighted


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    refs={'historical':np.array([0,0,0,1,1,1]),'October_partial':np.array([0,1,0,1,1,0])}
    scores={'Jan':{'ML':np.array([.1,.6,.2,.7,.9,.8]),'broad_annotation_rule':np.array([0,0,0,1,1,1]),'KPC_rule':np.array([0,0,0,0,1,0])},
            'Aug':{'ML':np.array([.1,.4,.2,.7,.9,.8]),'broad_annotation_rule':np.array([0,0,0,1,1,1]),'KPC_rule':np.array([0,0,0,0,1,0])}}
    cells={(db,ref,m):weighted(y,scores[db][m],np.ones(6)) for db in scores for ref,y in refs.items() for m in NAMES}
    c=contrasts(cells,refs)
    # Hand-enumerated balanced confusion matrices: historical ML .8333 ->1,
    # October ML .8333 ->.6667. The unchanged broad rule's DB contrast is0.
    assert abs(c['DB_Aug_minus_Jan|historical|ML'][0,6]-1/6)<1e-12
    assert abs(c['DB_Aug_minus_Jan|October_partial|ML'][0,6]+1/6)<1e-12
    assert abs(c['Method_by_DB_by_reference|ML_minus_broad_annotation_rule'][0,6]+1/3)<1e-12
    assert abs(c['Method_by_DB_by_reference|ML_minus_broad_annotation_rule'][0,7]+2/3)<1e-12
    identical={(db,ref,m):weighted(y,scores['Jan'][m],np.ones(6)) for db in scores for ref,y in refs.items() for m in NAMES}
    null=contrasts(identical,refs)
    for name,values in null.items():
        if name.startswith(('DB_Aug_minus_Jan','DB_by_reference','Method_by_DB_by_reference')):assert np.allclose(values,0)
    rng=np.random.default_rng(20261111);checks=0
    for _ in range(20):
        weights=rng.integers(0,4,size=6)
        for db in scores:
            for ref,y in refs.items():
                if weights@y==0 or weights@(1-y)==0:continue
                for method,s in scores[db].items():
                    v=weighted(y,s,weights)[0,[6,7,8,9]]
                    expected=[balanced_accuracy_score(y,s>=.5,sample_weight=weights),matthews_corrcoef(y,s>=.5,sample_weight=weights),roc_auc_score(y,s,sample_weight=weights),average_precision_score(y,s,sample_weight=weights)]
                    assert np.allclose(v,expected,atol=1e-12);checks+=1
    result={'status':'PASS_CLOSED_FORM_AXES_AND_UNCHANGED_DATABASE_NEGATIVE_CONTROLS','independent_sklearn_weighted_scenarios':checks,
      'controls_are_analytical_examples_not_observed_AMR_results':True,'hand_computed_triple_interaction_BA':-1/3,'hand_computed_triple_interaction_MCC':-2/3}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
