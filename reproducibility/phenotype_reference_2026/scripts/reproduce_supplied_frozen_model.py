#!/usr/bin/env python3
"""Validate supplied frozen model/feature artifacts and reproduce original E2 probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score

MODEL_SHA='1f55620cbd4b1dfd946ff37977e9a851b6aeaad8a5466eb553323331613cdcdd'
SCHEMA_SHA='9ca9c8c89a6bb2c1a97d85b9b5906efbf9aee0fca7eedb0328915063212c132f'
PRED_SHA='6b56e1eeb60421261d6933bf89897d0a10faae4ba7cea8399787b76008db9ef1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--original-predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    model_path=args.project/'checkpoints/external_validation_file11c/file11c_final_model.joblib'
    lock_path=args.project/'checkpoints/external_validation_file11c/file11c_model_lock.json'
    known_path=args.project/'data/features/known_amr/X_known_amr_final.csv'
    fold_path=args.project/'data/splits/file07_outer_folds.csv'
    external_path=args.project/'data/external_validation/ncbi_pathogen_detection/file11c/file11c_external_known_amr_668.csv'
    metadata_path=args.project/'data/features/known_amr/known_amr_feature_metadata_final.csv'
    if sha(model_path)!=MODEL_SHA or sha(args.original_predictions)!=PRED_SHA:raise ValueError('Frozen model/original prediction hash mismatch')
    lock=json.loads(lock_path.read_text())
    if lock['model_artifact_sha256']!=MODEL_SHA or lock['feature_schema_sha256']!=SCHEMA_SHA:raise ValueError('Frozen lock mismatch')
    if sha(known_path)!=lock['source_hashes']['known'] or sha(fold_path)!=lock['source_hashes']['folds']:raise ValueError('Training input hashes differ from model lock')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');artifact=joblib.load(model_path)
    model=artifact['model'];features=artifact['features']
    if type(model).__name__!='LogisticRegression' or model.classes_.tolist()!=[0,1] or len(features)!=668:raise ValueError('Unexpected estimator/schema')
    if hashlib.sha256(('\n'.join(features)+'\n').encode()).hexdigest()!=SCHEMA_SHA:raise ValueError('Model feature-order hash mismatch')
    if artifact['specification']!=lock['specification']:raise ValueError('Specification differs from lock')
    known=pd.read_csv(known_path,dtype={'Genome ID':str});folds=pd.read_csv(fold_path,dtype={'Genome ID':str})
    external=pd.read_csv(external_path,dtype={'asm_acc':str});metadata=pd.read_csv(metadata_path)
    original=pd.read_csv(args.original_predictions,dtype={'asm_acc':str,'biosample_acc':str,'target_acc':str})
    if known.columns[1:].tolist()!=features or external.columns[1:].tolist()!=features or set(metadata.Feature)!=set(features):raise ValueError('Supplied matrices or metadata differ from model schema')
    # Original FILE11C selects the fitting cohort by File07 IDs; the final
    # feature table can also contain rows not used to fit the estimator.
    if not known['Genome ID'].is_unique or len(folds)!=4227 or folds['Genome ID'].nunique()!=4227 or not set(folds['Genome ID'])<=set(known['Genome ID']):raise ValueError('Training cohort mismatch')
    if int(folds.y.sum())!=1601 or len(folds)-int(folds.y.sum())!=2626:raise ValueError('Training class counts mismatch')
    if external.asm_acc.nunique()!=525 or original.asm_acc.nunique()!=525 or set(external.asm_acc)!=set(original.asm_acc):raise ValueError('Original E2 assembly set mismatch')
    for frame in [known,external]:
        if not np.isin(frame[features].to_numpy(),[0,1]).all():raise ValueError('Nonbinary feature value')
    prevalence=known[features].sum().astype(int)
    if not np.array_equal(prevalence.loc[metadata.Feature].to_numpy(),metadata['Genomes Present'].to_numpy(int)):raise ValueError('Feature metadata prevalence mismatch')
    X=external[features].to_numpy(dtype=np.float32)
    probability=model.predict_proba(X)[:,1]
    manual=expit(X.astype(np.float64)@model.coef_[0]+float(model.intercept_[0]))
    expected=original.set_index('asm_acc').loc[external.asm_acc]
    prior=expected.probability_R.to_numpy(float);pred=(probability>=.5).astype(int)
    max_difference=float(np.max(np.abs(probability-prior)));manual_difference=float(np.max(np.abs(manual-probability)))
    if max_difference>1e-8 or manual_difference>1e-12 or not np.array_equal(pred,expected.predicted_R.to_numpy(int)):raise ValueError('Frozen probability or hard-prediction reproduction failed')
    tn,fp,fn,tp=confusion_matrix(expected.label_binary.to_numpy(int),pred,labels=[0,1]).ravel()
    if [int(tp),int(fn),int(tn),int(fp)]!=[109,54,221,141]:raise ValueError('Historical confusion counts differ')
    output=pd.DataFrame({'asm_acc':external.asm_acc,'biosample_acc':expected.biosample_acc.to_numpy(),
                         'original_probability_R':prior,'replayed_probability_R':probability,'manual_linear_probability_R':manual,
                         'absolute_difference':np.abs(probability-prior),'original_predicted_R':expected.predicted_R.to_numpy(int),'replayed_predicted_R':pred})
    output.to_csv(args.out/'E2_probability_reproduction.csv',index=False)
    pd.DataFrame({'feature':features,'coefficient':model.coef_[0]}).to_csv(args.out/'frozen_linear_coefficients.csv',index=False)
    score_spec={'intercept':float(model.intercept_[0]),'threshold':.5,'feature_schema_sha256':SCHEMA_SHA,'model_sha256':MODEL_SHA,'classes':[0,1],'feature_count':668}
    (args.out/'frozen_linear_score_specification.json').write_text(json.dumps(score_spec,indent=2)+'\n')
    report={'status':'PASS_FROZEN_MODEL_AND_ALL_525_ORIGINAL_E2_PROBABILITIES_REPRODUCED',
            'n':len(external),'features':len(features),'model_sha256':MODEL_SHA,'feature_schema_sha256':SCHEMA_SHA,
            'training_X_and_folds_hashes_match_original_lock':True,'feature_metadata_prevalence_matches_training_matrix':True,
            'feature_table_rows':len(known),'fitted_training_rows_from_File07':len(folds),'feature_table_rows_not_in_fitting_cohort':len(set(known['Genome ID'])-set(folds['Genome ID'])),
            'maximum_absolute_original_probability_difference':max_difference,'maximum_absolute_manual_sigmoid_difference':manual_difference,
            'hard_predictions_identical':True,'historical_confusion':{'tp':int(tp),'fn':int(fn),'tn':int(tn),'fp':int(fp)},
            'historical_auroc':roc_auc_score(expected.label_binary,probability),'historical_average_precision':average_precision_score(expected.label_binary,probability),
            'load_warnings':[str(w.message) for w in captured],
            'no_training_or_tuning':True,'limitations':['This is frozen artifact/prediction reproduction, not a new external clinical validation.','A serialization-version warning, if present, is recorded; all native and independent linear-score probabilities were numerically checked.']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
