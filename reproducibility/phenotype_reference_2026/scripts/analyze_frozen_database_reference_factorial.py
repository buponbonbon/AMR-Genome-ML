#!/usr/bin/env python3
"""Analyze controlled database/reference cells without changing the frozen methods."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,roc_auc_score,average_precision_score
from analyze_native_ccre_challenge import weighted,point,METRICS
from file11c_blind_external_validation import parse_amrfinder_tsv,feature_vector_from_keys
from file11e_external_simple_baseline_comparators import call_fixed_rules

MODEL='1f55620cbd4b1dfd946ff37977e9a851b6aeaad8a5466eb553323331613cdcdd'
NAMES={'ML':'frozen_probability_R','broad_annotation_rule':'amrfinder_carbapenemase_rule','KPC_rule':'kpc_family_rule'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def arm(path,manifest,features,coef,intercept,version):
    status=json.loads((path/'status.json').read_text());signature=json.loads((path/'RUN_SIGNATURE.json').read_text())
    if not status['status'].startswith('COMPLETE_'):raise ValueError('Caller arm incomplete')
    if signature['model_sha256']!=MODEL or signature['database_version']!=version:raise ValueError('Factor/model differs')
    if signature['manifest_sha256']!=sha(manifest):raise ValueError('Selection manifest changed')
    source=pd.read_csv(manifest,dtype=str,keep_default_na=False)
    d=pd.read_csv(path/'frozen_challenge_predictions.csv',dtype=str,keep_default_na=False)
    if d.sample_accession.tolist()!=source.sample_accession.tolist():raise ValueError('Case order/membership differs')
    for col in source:
        if not d[col].equals(source[col]):raise ValueError('Source column changed: '+col)
    mask=d.frozen_probability_R.ne('');scored=d.loc[mask].copy();z=np.load(path/'native_features.npz',allow_pickle=False)
    if z['biosamples'].tolist()!=scored.sample_accession.tolist() or z['features'].tolist()!=features:raise ValueError('Matrix identities differ')
    rows=[];evidence={}
    for row in scored.to_dict('records'):
        sample=row['sample_accession'];p=path/'amrfinder'/(sample+'.tsv');receipt=json.loads((path/'receipts'/(sample+'.json')).read_text())
        if receipt['status']!='PASS' or receipt['output_sha256']!=sha(p):raise ValueError('Native output receipt differs')
        x=feature_vector_from_keys(parse_amrfinder_tsv(p),features).astype(np.uint8);rows.append(x)
        calls,det=call_fixed_rules(p);evidence[sample]=det
        for name,value in calls.items():
            if int(float(row[name]))!=value:raise ValueError('Original rule recomputation differs')
    x=np.asarray(rows)
    if not np.array_equal(x,z['X']):raise ValueError('Native vector reconstruction differs')
    manual=expit(x@coef+intercept);p=scored.frozen_probability_R.to_numpy(float)
    difference=float(np.max(np.abs(manual-p),initial=0))
    if difference>1e-12:raise ValueError('Manual frozen score differs')
    if not np.array_equal((p>=.5).astype(int),scored.frozen_predicted_R.to_numpy(float).astype(int)):raise ValueError('Threshold differs')
    matrix=pd.DataFrame(x,index=scored.sample_accession,columns=features)
    inventory=pd.read_csv(path/'input_inventory.csv',dtype=str,keep_default_na=False).set_index('sample_accession')
    return {'frame':d.set_index('sample_accession'),'scored':scored.set_index('sample_accession'),
            'X':matrix,'inventory':inventory,'signature':signature,'evidence':evidence,
            'verification':{'scored':len(scored),'target':len(d),'maximum_manual_probability_difference':difference,
                            'all_feature_vectors_rules_and_receipt_hashes_verified':True}}


def intervals(array):
    result={}
    for i,k in enumerate(METRICS[4:],4):
        v=array[:,i];v=v[np.isfinite(v)]
        result[k]={'valid_draws':len(v),'percentile_2_5':float(np.quantile(v,.025)) if len(v) else None,
                   'percentile_97_5':float(np.quantile(v,.975)) if len(v) else None}
    return result


def contrasts(cells,refs):
    result={}
    for ref in refs:
        for method in NAMES:
            result['DB_Aug_minus_Jan|'+ref+'|'+method]=cells[('Aug',ref,method)]-cells[('Jan',ref,method)]
        for db in ['Jan','Aug']:
            for comparator in ['broad_annotation_rule','KPC_rule']:
                result['ML_minus_'+comparator+'|'+db+'|'+ref]=cells[(db,ref,'ML')]-cells[(db,ref,comparator)]
    if len(refs)==2:
        for db in ['Jan','Aug']:
            for method in NAMES:result['Reference_Oct_minus_historical|'+db+'|'+method]=cells[(db,'October_partial',method)]-cells[(db,'historical',method)]
        for method in NAMES:
            result['DB_by_reference_interaction|'+method]=result['Reference_Oct_minus_historical|Aug|'+method]-result['Reference_Oct_minus_historical|Jan|'+method]
        for comparator in ['broad_annotation_rule','KPC_rule']:
            result['Method_by_DB_by_reference|ML_minus_'+comparator]=result['DB_by_reference_interaction|ML']-result['DB_by_reference_interaction|'+comparator]
    return result


def cohort(root,name,jan_path,aug_path,manifest,features,coef,intercept,out,draws):
    out.mkdir(parents=True,exist_ok=False)
    a=arm(jan_path,manifest,features,coef,intercept,'2026-01-21.1');b=arm(aug_path,manifest,features,coef,intercept,'2026-08-07.1')
    if a['signature']['caller_exe_sha256']!=b['signature']['caller_exe_sha256']:raise ValueError('Caller executable differs')
    ids=a['frame'].index;paired=ids[ids.isin(a['scored'].index)&ids.isin(b['scored'].index)]
    if not a['inventory'].loc[paired,'fasta_sha256'].equals(b['inventory'].loc[paired,'fasta_sha256']):raise ValueError('DNA bytes differ between databases')
    frame=a['frame'].loc[paired].copy();xa=a['X'].loc[paired].to_numpy();xb=b['X'].loc[paired].to_numpy()
    references={'PHAS_original':'S/I/R (PHAS interpretation)_Meropenem'} if name=='CCRE' else {'historical':'historical_label','October_partial':'scenario_label'}
    ys={ref:frame[col].map({'S':0,'R':1}).to_numpy(int) for ref,col in references.items()}
    scores={db:{method:record['scored'].loc[paired,col].to_numpy(float) for method,col in NAMES.items()} for db,record in [('Jan',a),('Aug',b)]}
    cells={(db,ref,method):weighted(y,scores[db][method],np.ones(len(paired))) for db,ref,method in itertools.product(['Jan','Aug'],references,NAMES) for y in [ys[ref]]}
    points=[]
    for db,ref,method in cells:points.append(dict(database=db,reference=ref,method=method,**point(ys[ref],scores[db][method])))
    pd.DataFrame(points).to_csv(out/'cell_metrics.csv',index=False)
    deltas=contrasts(cells,references);point_deltas={k:{metric:float(v[0,i]) for i,metric in enumerate(METRICS[4:],4)} for k,v in deltas.items()}
    boot={};metric_checks=0
    groups=[('PDS','PDS_bases'),('country','Country of origin')] if name=='CCRE' else [('BioProject','project'),('focal_retained','project')]
    for gi,(scheme,col) in enumerate(groups):
        values=frame[col].to_numpy();unique=np.unique(values);membership=(values[None,:]==unique[:,None]).astype(int)
        seed=20261111+(0 if name=='CCRE' else 100)+gi;rng=np.random.default_rng(seed)
        if scheme=='focal_retained':
            pos=np.flatnonzero(unique=='PRJNA1322038')
            if len(pos)!=1:raise ValueError('Known source missing')
            bg=np.delete(np.arange(len(unique)),pos[0]);counts=np.zeros((draws,len(unique)),int);counts[:,pos[0]]=1
            counts[:,bg]=rng.multinomial(len(bg),np.full(len(bg),1/len(bg)),size=draws)
        else:counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=draws)
        weights=counts@membership
        arrays={(db,ref,method):weighted(ys[ref],scores[db][method],weights) for db,ref,method in cells}
        for key,v in arrays.items():
            db,ref,method=key;y=ys[ref];s=scores[db][method]
            for j in range(3):
                w=weights[j]
                if w@y<=0 or w@(1-y)<=0:continue
                expected=[balanced_accuracy_score(y,s>=.5,sample_weight=w),matthews_corrcoef(y,s>=.5,sample_weight=w),roc_auc_score(y,s,sample_weight=w),average_precision_score(y,s,sample_weight=w)]
                if not np.allclose(v[j,[6,7,8,9]],expected,atol=1e-12):raise ValueError('Independent weighted metric check differs')
                metric_checks+=1
        cd=contrasts(arrays,references)
        boot[scheme]={'groups':len(unique),'draws':draws,'seed':seed,
                      'cell_percentiles':{'|'.join(k):intervals(v) for k,v in arrays.items()},
                      'paired_contrast_percentiles':{k:dict(point=point_deltas[k],**{'metrics':intervals(v)}) for k,v in cd.items()}}
        if scheme=='BioProject':boot[scheme]['draws_omitting_focal']=int((counts[:,np.flatnonzero(unique=='PRJNA1322038')[0]]==0).sum())
        np.savez_compressed(out/(scheme+'_bootstrap_metrics.npz'),**{'|'.join(k):v for k,v in arrays.items()})
    changed=xa!=xb;ix,jx=np.where(changed)
    pd.DataFrame({'sample_accession':paired.to_numpy()[ix],'feature':np.asarray(features)[jx],
      'January':xa[ix,jx],'August':xb[ix,jx],'coefficient':coef[jx],
      'August_minus_January_logit_contribution':(xb[ix,jx].astype(float)-xa[ix,jx])*coef[jx]}).to_csv(out/'changed_model_features.csv',index=False)
    case=frame.copy();case['changed_feature_cells']=changed.sum(axis=1);case['August_minus_January_logit']=(xb.astype(float)-xa)@coef
    case['model_probability_Jan']=scores['Jan']['ML'];case['model_probability_Aug']=scores['Aug']['ML']
    decisions={};det=[]
    for method in NAMES:
        p=scores['Jan'][method]>=.5;q=scores['Aug'][method]>=.5
        case[method+'_Jan_R']=p.astype(int);case[method+'_Aug_R']=q.astype(int)
        decisions[method]={'changed':int((p!=q).sum()),'S_to_R':int((~p&q).sum()),'R_to_S':int((p&~q).sum())}
        if method!='ML':
            for sample in paired[p!=q]:
                for db,record in [('Jan',a),('Aug',b)]:
                    for evidence in record['evidence'][sample]:det.append(dict(sample_accession=sample,database=db,changed_rule=method,**evidence))
    case.reset_index().to_csv(out/'paired_case_audit.csv',index=False)
    if det:pd.DataFrame(det).to_csv(out/'changed_rule_annotation_evidence.csv',index=False)
    missing=ids[~ids.isin(paired)];a['frame'].loc[missing].reset_index().to_csv(out/'unpaired_fixed_cases.csv',index=False)
    bounds={}
    if len(missing)>10:raise ValueError('Too many technical failures for enumerated bounds')
    for ref,col in references.items():
        ym=a['frame'].loc[missing,col].map({'S':0,'R':1}).to_numpy(int);full_y=np.r_[ys[ref],ym]
        for db,method in itertools.product(['Jan','Aug'],NAMES):
            possible=[point(full_y,np.r_[scores[db][method],extra]) for extra in itertools.product([0,1],repeat=len(missing))]
            bounds['|'.join([db,ref,method])]={metric:{'min':min(x[metric] for x in possible),'max':max(x[metric] for x in possible)} for metric in METRICS}
    reproduction=None
    if name=='E2':
        old=pd.read_csv(root/'results/frozen_model_challenge_20261110/received/project/data/external_validation/ncbi_pathogen_detection/file11c/file11c_external_known_amr_668.csv').set_index('asm_acc')
        expected=old.loc[frame.asm_acc,features].to_numpy(np.uint8)
        original=frame.probability_R.to_numpy(float)
        reproduction={'n':len(paired),'feature_cells_different':int((expected!=xb).sum()),'samples_with_feature_differences':int((expected!=xb).any(axis=1).sum()),
          'maximum_probability_difference':float(np.abs(original-scores['Aug']['ML']).max()),
          'hard_calls_different':int(((original>=.5)!=(scores['Aug']['ML']>=.5)).sum()),
          'original_historical_results_not_replaced':True}
        expected_rules={'historical':{'TP':138,'FN':25,'TN':192,'FP':170},'October_partial':{'TP':262,'FN':4,'TN':213,'FP':46}}
        reproduction['native_August_broad_rule_counts_match_original']={ref:all(next(x for x in points if x['database']=='Aug' and x['reference']==ref and x['method']=='broad_annotation_rule')[k]==v for k,v in values.items()) for ref,values in expected_rules.items()}
    result={'status':'COMPLETE_CONTROLLED_DATABASE_REFERENCE_ANALYSIS','cohort':name,'fixed_target':len(ids),'paired':len(paired),'unpaired':len(missing),
      'controlled_DNA_bytes_and_caller_executable_match':True,'input_sha256':{'manifest':sha(manifest),'January_run_signature':sha(jan_path/'RUN_SIGNATURE.json'),'August_run_signature':sha(aug_path/'RUN_SIGNATURE.json'),'analysis_script':sha(Path(__file__))},'arm_verification':{'Jan':a['verification'],'Aug':b['verification']},
      'cell_metrics':points,'point_contrasts':point_deltas,'source_composition_sensitivity':boot,
      'feature_changes':{'cells':int(changed.sum()),'samples':int(changed.any(axis=1).sum()),'features':int(changed.any(axis=0).sum()),
                        'maximum_probability_difference':float(np.abs(scores['Aug']['ML']-scores['Jan']['ML']).max())},
      'decision_changes':decisions,'full_target_missing_bounds':bounds,'original_August_reproduction':reproduction,
      'independent_weighted_metric_checks':metric_checks,'phenotype_labels_unchanged_within_each_reference':True,
      'limits':['Post-hoc selected sources; source-composition percentiles, not patient-population confidence guarantees.',
        'Database effects concern only these two catalog versions with the frozen model/schema/rules.',
        'Reference effects are relative to dated records, not adjudication of clinical truth.',
        'Only E2 has two dated reference conditions; CCRE is a database-factor extension with stable originalPHAS.',
        'No new independent temporal phenotype-reversal event is established.',
        'Annotation changes and model coefficients do not establish biological resistance mechanisms.']}
    save(out/'report.json',result);print(json.dumps({'cohort':name,'paired':len(paired),'feature_changes':result['feature_changes'],'decision_changes':decisions,'original_August_reproduction':reproduction},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--draws',type=int,default=10000);p.add_argument('--cohort',choices=['CCRE','E2','both'],default='both');a=p.parse_args()
    root=a.root;run=root/'results/database_reference_factorial_20261111';a.out.mkdir(parents=True,exist_ok=False)
    specdir=root/'results/frozen_model_challenge_20261110/frozen_reproduction_verified'
    coef=pd.read_csv(specdir/'frozen_linear_coefficients.csv');spec=json.loads((specdir/'frozen_linear_score_specification.json').read_text())
    if spec['model_sha256']!=MODEL or hashlib.sha256(('\n'.join(coef.feature)+'\n').encode()).hexdigest()!=spec['feature_schema_sha256']:raise ValueError('Frozen specification differs')
    reports={}
    if a.cohort in ['CCRE','both']:
        reports['CCRE']=cohort(root,'CCRE',run/'CCRE_Jan',root/'results/strict_CCRE_followup_20261111/native355',root/'results/strict_CCRE_followup_20261111/native355_execution_manifest.csv',coef.feature.tolist(),coef.coefficient.to_numpy(),spec['intercept'],a.out/'CCRE',a.draws)
    if a.cohort in ['E2','both']:
        reports['E2']=cohort(root,'E2',run/'E2_Jan',run/'E2_Aug',run/'E2_execution_manifest.csv',coef.feature.tolist(),coef.coefficient.to_numpy(),spec['intercept'],a.out/'E2',a.draws)
    save(a.out/'report.json',{'status':'COMPLETE_REQUESTED_CONTROLLED_COMPARISONS','cohorts':reports,'frozen_model_sha256':MODEL,'no_training_tuning_or_label_redefinition':True})


if __name__=='__main__':main()
