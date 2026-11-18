#!/usr/bin/env python3
"""Exploratory frozen-model diagnostic under the documented older NCBI feature DB.

This deliberately does NOT certify a pipeline-equivalent external validation.
It preserves the locked model, probabilities, threshold, schema and source labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from file11e_external_simple_baseline_comparators import KPC_RE, norm_symbol

LABEL = 'S/I/R (PHAS interpretation)_Meropenem'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, scores):
    pred = (scores >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {'n':len(y),'R':int(y.sum()),'S':int((1-y).sum()),
            'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp),
            'sensitivity':float(tp/(tp+fn)) if tp+fn else None,
            'specificity':float(tn/(tn+fp)) if tn+fp else None,
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'MCC':float(matthews_corrcoef(y,pred)),
            'AUROC':float(roc_auc_score(y,scores)),
            'average_precision':float(average_precision_score(y,scores))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--reproduction',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--draws',type=int,default=3000)
    args=p.parse_args()
    protocol=args.root/'legacy_feature_diagnostic/protocol.json'
    if not protocol.is_file():raise ValueError('Exploratory protocol must precede outcomes')
    source=pd.read_csv(args.root/'CCRE_SNP_gate/CCRE_operational_SNP_filtered_candidates.csv',dtype=str,keep_default_na=False)
    columns=['Run','biosample_acc','target_acc','asm_acc','AMR_genotypes','amrfinder_version','refgene_db_version','amrfinder_analysis_type']
    metadata=pd.read_csv(args.metadata,sep='\t',usecols=columns,dtype=str,keep_default_na=False)
    subset=metadata[metadata.biosample_acc.isin(source.sample_accession)].copy()
    rows=source.merge(subset,left_on=['sample_accession','Run accession'],right_on=['biosample_acc','Run'],how='left',validate='one_to_one',indicator=True)
    if not rows['_merge'].eq('both').all():raise ValueError('Missing or ambiguous exact sample/run feature linkage')
    if not (rows.amrfinder_version.eq('4.2.7') & rows.amrfinder_analysis_type.eq('NUCLEOTIDE') & rows.refgene_db_version.eq('2026-01-21.1')).all():raise ValueError('Unexpected or mixed legacy feature configuration')
    if rows.AMR_genotypes.isin(['','NULL','NA']).any():raise ValueError('Missing gene summaries must not be encoded as negative calls')
    coef=pd.read_csv(args.reproduction/'frozen_linear_coefficients.csv')
    spec=json.loads((args.reproduction/'frozen_linear_score_specification.json').read_text())
    features=coef.feature.tolist()
    if hashlib.sha256(('\n'.join(features)+'\n').encode()).hexdigest()!=spec['feature_schema_sha256']:raise ValueError('Locked schema mismatch')
    expected_spec={'model_sha256':'1f55620cbd4b1dfd946ff37977e9a851b6aeaad8a5466eb553323331613cdcdd','threshold':.5,'feature_count':668}
    if any(spec.get(k)!=v for k,v in expected_spec.items()):raise ValueError('Locked model specification mismatch')
    lookup={f.lower():i for i,f in enumerate(features)}
    X=np.zeros((len(rows),len(features)),dtype=np.uint8)
    kpc=[]
    for i,text in enumerate(rows.AMR_genotypes):
        genes=[item.split('=',1)[0].strip() for item in text.split(',')]
        for gene in genes:
            j=lookup.get(gene.lower())
            if j is not None:X[i,j]=1
        kpc.append(int(any(KPC_RE.match(norm_symbol(g)) for g in genes)))
    score=expit(X@coef.coefficient.to_numpy()+spec['intercept'])
    rows['exploratory_frozen_probability_R_from_Jan2026_features']=score
    rows['exploratory_frozen_predicted_R']=(score>=spec['threshold']).astype(int)
    rows['KPC_family_presence_rule_from_summary']=kpc
    rows['analysis_status']='EXPLORATORY_OLDER_FEATURE_DB_NOT_PIPELINE_EQUIVALENT_VALIDATION'
    if not rows[LABEL].isin(['S','R']).all():raise ValueError('Nonbinary original categories cannot be recoded')
    y=rows[LABEL].map({'S':0,'R':1}).to_numpy(int)
    kpc=np.asarray(kpc,float)
    point={'locked_ML_older_DB_diagnostic':metrics(y,score),'KPC_family_presence_older_DB':metrics(y,kpc)}
    # One deterministic representative per retained PDS for a descriptive check.
    if rows.PDS_bases.str.contains(';').any():raise ValueError('Multicluster records require connected-component grouping')
    ordering=rows.sample_accession.map(lambda s:hashlib.sha256(('AMR_CCRE_20261111|'+s).encode()).hexdigest())
    representative=rows.assign(_selection=ordering).sort_values('_selection').drop_duplicates('PDS_bases').index.to_numpy()
    point['ML_one_representative_per_PDS']=metrics(y[representative],score[representative])
    point['KPC_one_representative_per_PDS']=metrics(y[representative],kpc[representative])
    # Multinomial PDS resampling keeps each group's full paired reference/model rows.
    groups=rows.PDS_bases.to_numpy();unique=np.unique(groups)
    indices=[np.flatnonzero(groups==group) for group in unique]
    rng=np.random.default_rng(20261111);boot=[]
    for _ in range(args.draws):
        ix=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        if len(np.unique(y[ix]))!=2:continue
        a=metrics(y[ix],score[ix]);b=metrics(y[ix],kpc[ix])
        boot.append([a['AUROC'],a['balanced_accuracy'],a['MCC'],a['balanced_accuracy']-b['balanced_accuracy'],a['MCC']-b['MCC']])
    boot=np.asarray(boot)
    names=['ML_AUROC','ML_balanced_accuracy','ML_MCC','ML_minus_KPC_balanced_accuracy','ML_minus_KPC_MCC']
    ranges={name:{'percentile_2_5':float(np.quantile(boot[:,i],.025)),'percentile_97_5':float(np.quantile(boot[:,i],.975))} for i,name in enumerate(names)}
    report={'status':'EXPLORATORY_ONLY_OLDER_FEATURE_DATABASE_AND_PARTIAL_INTERNAL_CLUSTER_COVERAGE',
            'strict_pipeline_equivalent_new_cohort_validation_complete':False,
            'n':len(rows),'PDS_groups':len(unique),'legacy_feature_caller':'AMRFinderPlus 4.2.7 / nucleotide / DB 2026-01-21.1',
            'locked_original_feature_database':'2026-08-07.1','threshold':spec['threshold'],
            'original_primary_reference':'Publication PHAS meropenem S/I/R, with explicit S/R only; original MIC inequalities retained',
            'point_metrics':point,'paired_PDS_bootstrap':{'seed':20261111,'draws':len(boot),'percentile_ranges':ranges},
            'comparators':'KPC-family baseline uses the unchanged File11E KPC regex on retained AMR symbols; this is distinct from the broader carbapenem-associated rule.',
            'no_original_carbapenem_associated_per_isolate_calls_fabricated':True,
            'input_sha256':{'protocol':sha(protocol),'NCBI_metadata':sha(args.metadata),'coefficient_readout':sha(args.reproduction/'frozen_linear_coefficients.csv'),'score_spec':sha(args.reproduction/'frozen_linear_score_specification.json')},
            'limitations':[
                'An older feature database is a material change in input extraction, so these numbers are not the pre-specified Aug-2026 clinical validation.',
                'Do not attribute a difference from E2 entirely to phenotype provenance: source composition, assay and feature DB also differ.',
                'Absence of a shared PDS only concerns mapped fitting isolates; internal PDS coverage is incomplete.',
                'The selected collection is enriched and was screened post hoc. These numbers do not identify population predictive values or clinical utility.',
                'KPC-family presence is a narrow comparator; exceeding it does not establish superiority over the broader frozen carbapenem-associated annotation rule.',
                'PDS percentile ranges describe the selected source composition, not guaranteed population confidence intervals.',
            ]}
    args.out.mkdir(parents=True,exist_ok=False)
    rows.to_csv(args.out/'CCRE_legacy_feature_diagnostic_predictions.csv',index=False)
    np.savez_compressed(args.out/'legacy_projected_features.npz',X=X,features=np.asarray(features),biosamples=rows.sample_accession.to_numpy(dtype=str))
    pd.DataFrame([dict(method=k,**v) for k,v in point.items()]).to_csv(args.out/'point_metrics.csv',index=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['input_sha256']},indent=2))


if __name__=='__main__':main()
