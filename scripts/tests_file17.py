"""Synthetic FILE17 tests; by construction NO real AMR scientific outputs."""
from __future__ import annotations
import tempfile
from pathlib import Path
from argparse import Namespace
import pandas as pd

def build_data(root):
    pred=[]
    for cohort in ['E2','E3']:
        for i,(y,p) in enumerate([(1,.90),(1,.8),(1,.05),(1,.2),(0,.1),(0,.4),(0,.6),(0,.95)],1):
            pred.append(dict(cohort=cohort,genome_id=f'{cohort}_G{i}',y_true=y,prob_resistance=p,
                             predicted_resistant_at_frozen_threshold_05=int(p>=.5)))
    df=pd.DataFrame(pred)
    predpath=root/'pred.csv';df.to_csv(predpath,index=False)
    rows=[]
    for cohort in ('E2','E3'):
        for j in (1,5):
            rows.append(dict(cohort=cohort,external_genome_id=f'{cohort}_G{j}',development_genome_id=f'DEV_{cohort}_{j}',
                        mash_distance=.0004,ani_percent=99.96,aligned_fraction_ref=.96,aligned_fraction_query=.96))
        rows.append(dict(cohort=cohort,external_genome_id=f'{cohort}_G2',development_genome_id=f'DEV_{cohort}_2',
                    mash_distance=.005,ani_percent=99.8,aligned_fraction_ref=.96,aligned_fraction_query=.96))
    nearpath=root/'near.csv';pd.DataFrame(rows).to_csv(nearpath,index=False)
    return predpath,nearpath

def raises(func, msg):
    try:func()
    except (ValueError,FileNotFoundError) as exc:
        assert msg.lower() in str(exc).lower(),(msg,str(exc))
    else:raise AssertionError('Expected rejection: '+msg)

def run_tests():
    from file17_frozen_neighbour_sensitivity_audit import read_predictions,read_ani,calculate,audit
    with tempfile.TemporaryDirectory(prefix='file17_synthetic_') as td:
        root=Path(td);pred,ani=build_data(root)
        data=read_predictions(pred,testing=True)
        selected,meta=read_ani(ani,data,testing=True)
        assert meta['confirmed_neighbour_counts']=={'E2':2,'E3':2}
        _,metrics,sumry=calculate(data,selected,testing=True)
        assert sumry['E2']['full_unchanged']['n']==8
        assert sumry['E2']['remainder_sensitivity_posthoc_not_independent']['n']==6
        assert sumry['E3']['ani_confirmed_targeted_neighbours']['n']==2
        assert set(metrics.panel)=={'FROZEN_FULL_PRIMARY','TARGETED_ANI_CONFIRMED_NEAR','REMAINDER_AFTER_REMOVING_CONFIRMED_NEAR_POSTHOC'}
        p2=root/'pred_bad.csv';d=pd.read_csv(pred);d.loc[0,'predicted_resistant_at_frozen_threshold_05']=0;d.to_csv(p2,index=False)
        raises(lambda:read_predictions(p2,testing=True),'frozen >=0.5')
        a2=root/'near_bad.csv';d=pd.read_csv(ani);d.loc[0,'external_genome_id']='WRONG_ID';d.to_csv(a2,index=False)
        raises(lambda:read_ani(a2,data,testing=True),'not in FILE16')
        d=pd.read_csv(ani);d.loc[0,'ani_percent']=.9996;d.to_csv(a2,index=False)
        raises(lambda:read_ani(a2,data,testing=True),'percentages')
        d=pd.read_csv(ani);d.loc[0,'aligned_fraction_ref']=96;d.to_csv(a2,index=False)
        raises(lambda:read_ani(a2,data,testing=True),'0..1')
        d=pd.read_csv(ani);pd.concat([d,d.iloc[:1]],ignore_index=True).to_csv(a2,index=False)
        raises(lambda:read_ani(a2,data,testing=True),'repeated')
        raises(lambda:read_predictions(pred,testing=False),'frozen n mismatch')
        raises(lambda:read_ani(ani,data,testing=False),'31/25')
        a=Namespace(project_root=str(root),predictions=str(pred),ani_pairs=str(ani),outdir=str(root/'out'))
        result=audit(a,testing=True)
        assert (root/'out'/'file17_summary.json').is_file()
        assert (root/'out'/'file17_output_manifest_sha256.json').is_file()
        assert result['no_training_or_threshold_selection'] and result['no_primary_cohort_redefinition']
        raises(lambda:audit(a,testing=True),'already exists')
    print('[FILE17 SELF-TEST] PASS: synthetic happy path, guards, hashes, no-overwrite, case mapping')

if __name__=='__main__': run_tests()
