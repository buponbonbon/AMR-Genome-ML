#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FILE18 v1.0.1 -- post-hoc, immutable FILE11C model transfer from E2 to E3.

Usage:
  python scripts/file18_same_frozen_model_cross_cohort_audit.py self-test
  python scripts/file18_same_frozen_model_cross_cohort_audit.py discover --project-root .
  python scripts/file18_same_frozen_model_cross_cohort_audit.py audit --project-root .

NO training, threshold selection, cohort changes, re-annotation, bootstrap significance,
or modification of any FILE01--17 artifact.  The E3 harmonized 668 projection is NOT
proven to use exactly the same AMRFinder annotation protocol as the historical E2
matrix.  E3 input columns can be PERMUTED to the exact original FILE11C model order,
after checking identical feature names and the two independent schema locks.
The scientific claim is limited to SAME MODEL WEIGHTS + SAME NAMED FEATURES,
NOT a fully identical preprocessing or prospectively planned external validation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    matthews_corrcoef, roc_auc_score,
)

VERSION = '1.0.1'
OUTPUT_REL = 'results/file18_same_model_audit_v1_0_1'
PATHS = {
    'e2_model': 'checkpoints/external_validation_file11c/file11c_final_model.joblib',
    'e2_lock': 'checkpoints/external_validation_file11c/file11c_model_lock.json',
    'e2_matrix': 'data/external_validation/ncbi_pathogen_detection/file11c/file11c_external_known_amr_668.csv',
    'e2_predictions': 'data/external_validation/ncbi_pathogen_detection/file11c/file11c_blind_predictions.csv',
    'file13_manifest': 'models/file13/file13_freeze_manifest.json',
    'file13_668_schema': 'models/file13/file13_harmonized_original668_sensitivity_feature_schema.csv',
    'file13_freeze_flag': 'checkpoints/file13_harmonized_amrfinder/FREEZE_COMPLETE.flag',
    'e3_668_matrix': 'data/external_validation/e3/file14_final/file14_e3_sensitivity_668_matrix.csv.gz',
    'e3_primary_predictions': 'data/evaluation/file14/file14_E3_primary_predictions_labeled.csv',
    'file14_summary': 'checkpoints/file14_final_e3_validation/file14_final_summary.json',
}


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            d.update(block)
    return d.hexdigest()


def require(condition: bool, why: str) -> None:
    if not condition:
        raise RuntimeError(why)


def load_json(path: Path) -> dict:
    x = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(x, dict), f'Not JSON object: {path}')
    return x


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def input_files(root: Path) -> dict[str, Path]:
    return {k: root / v for k, v in PATHS.items()}


def preflight(root: Path) -> dict:
    paths = input_files(root)
    status = {k: {'relative_path': PATHS[k], 'exists': p.is_file(),
                  'bytes': p.stat().st_size if p.is_file() else None}
              for k,p in paths.items()}
    return {'file': 'FILE18', 'version': VERSION, 'project_root': str(root),
            'all_required_present': all(x['exists'] for x in status.values()),
            'required_inputs': status,
            'scope': 'ONE original 668-feature FILE11C model, independently scored on E2 and E3',
            'preprocessing_limitation': 'E3 FILE14 harmonized 668 projection and E2 FILE11C original annotations may differ',
            'note': 'Discover only inventories paths, does NOT score the model or modify frozen inputs.'}


def schema_668(path: Path) -> list[str]:
    x = pd.read_csv(path, dtype=str, keep_default_na=False)
    require('feature' in x.columns, f'FILE13 schema lacks feature column: {path}')
    f = x['feature'].astype(str).tolist()
    require(len(f) == 668 and all(q.strip() for q in f), 'FILE13 schema must contain 668 nonblank features')
    require(len({q.casefold() for q in f}) == 668, 'Duplicated feature name (case-insensitive)')
    return f


def matrix_header_features(path: Path, id_col: str) -> list[str]:
    """Read true disk-order feature schema (not FILE13 metadata order)."""
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    require(cols and cols[0] == id_col,
            f'Unexpected ID/header in matrix {path}: {cols[:3]}')
    feats = cols[1:]
    require(len(feats) == 668 and all(x.strip() for x in feats),
            f'Expected exactly 668 nonblank feature columns in {path}, got {len(feats)}')
    require(len(set(feats)) == 668 and len({f.casefold() for f in feats}) == 668,
            f'Duplicate feature names in matrix: {path}')
    return feats


def feature_set_gate(original: list[str], harmonized: list[str]) -> dict:
    """Exact-name equality only; never lowercase, rename, fill absent columns."""
    original_set, harmonized_set = set(original), set(harmonized)
    missing = sorted(original_set - harmonized_set)
    unexpected = sorted(harmonized_set - original_set)
    require(not missing and not unexpected,
            'FILE18 FEATURE SET MISMATCH (not an order-only issue): '
            f'missing_from_FILE13={missing[:15]}; unexpected_in_FILE13={unexpected[:15]}. '
            'Refusing automatic mapping or synthetic zeros.')
    idx = {name: i for i, name in enumerate(harmonized)}
    permutation = [idx[name] for name in original]
    require(len(set(permutation)) == 668, 'Non-bijective feature permutation')
    return {'feature_count': 668,
            'exact_feature_set_equal': True,
            'on_disk_order_equal': original == harmonized,
            'e3_columns_reordered_in_memory_for_file11c_model': original != harmonized,
            'number_different_positions': sum(a != b for a,b in zip(original,harmonized)),
            'mapping_file11c_position_to_file13_position': permutation}


def schema_diagnostic(root: Path, output: bool = False) -> dict:
    """Read-only preflight; always report real set/order disagreements explicitly."""
    paths = input_files(root)
    for key in ('e2_lock','e2_matrix','file13_manifest','file13_668_schema','e3_668_matrix'):
        require(paths[key].is_file(), f'Diagnostic missing input: {PATHS[key]}')
    lock=load_json(paths['e2_lock'])
    f13=load_json(paths['file13_manifest'])
    original=matrix_header_features(paths['e2_matrix'], 'asm_acc')
    harmonized=schema_668(paths['file13_668_schema'])
    e3_features=matrix_header_features(paths['e3_668_matrix'], 'Genome ID')
    sha_original=hashlib.sha256(('\n'.join(original)+'\n').encode()).hexdigest()
    sha_file13text=hashlib.sha256(('\n'.join(harmonized)+'\n').encode()).hexdigest()
    original_set, harmonized_set = set(original), set(harmonized)
    missing, unexpected = sorted(original_set-harmonized_set), sorted(harmonized_set-original_set)
    equal_sets=(not missing and not unexpected)
    mapping=feature_set_gate(original,harmonized) if equal_sets else None
    correct_11c_hash=lock.get('feature_schema_sha256') == sha_original
    correct_13_file_hash=f13['sensitivity_feature_schema']['sha256']==sha256(paths['file13_668_schema'])
    correct_e3_order=e3_features==harmonized
    permitted=correct_11c_hash and correct_13_file_hash and correct_e3_order and equal_sets
    report={'file':'FILE18', 'version': VERSION,
            'FILE11C_lock_vs_E2_matrix_order_hash_matches': correct_11c_hash,
            'FILE11C_locked_order_sha256': lock.get('feature_schema_sha256'),
            'E2_matrix_order_sha256': sha_original,
            'FILE13_668_manifest_file_sha_matches': correct_13_file_hash,
            'FILE13_668_text_order_sha256':sha_file13text,
            'E3_matrix_matches_FILE13_frozen_order':correct_e3_order,
            'exact_feature_name_set_equal': equal_sets,
            'missing_from_FILE13_count':len(missing),'missing_from_FILE13_examples':missing[:20],
            'unexpected_in_FILE13_count':len(unexpected),'unexpected_in_FILE13_examples':unexpected[:20],
            'original_vs_harmonized':mapping,
            'decision':'AUDIT_ALLOWED_WITH_EXACT_NAME_MAPPING' if permitted else 'STOP_REVIEW_PROVENANCE'}
    if output:
        dest=output_location(root,'results/file18_schema_diagnostic_v1_0_1')
        dest.mkdir(parents=True,exist_ok=False)
        write_json(dest/'file18_schema_diagnostic.json',report)
        print('[FILE18] Schema diagnostic written to:', dest)
    return report


def load_matrix(path: Path, id_col: str, features: list[str], expected: int) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={id_col: str}, low_memory=False)
    require(list(d.columns) == [id_col] + features,
            f'Exact feature order/schema mismatch in {path}. Refusing silent reordering, zeros or feature imputation.')
    d[id_col] = d[id_col].astype(str).str.strip()
    require(len(d) == expected and d[id_col].nunique() == expected and d[id_col].ne('').all(),
            f'ID count/uniqueness mismatch in {path}, got {len(d)}')
    x = d[features].to_numpy()
    require(np.isfinite(x.astype(float)).all(), f'Nonfinite values in {path}')
    require(np.isin(x.astype(float), [0, 1]).all(), f'Matrix is not binary: {path}')
    return d


def labels_and_preds(path: Path, id_col: str, y_col: str, n: int, r: int,
                     expected_confusion: tuple[int,int,int,int]) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={id_col: str}, low_memory=False)
    for c in [id_col,y_col,'probability_R','predicted_R']:
        require(c in d.columns, f'{path} missing {c}')
    d[id_col] = d[id_col].astype(str).str.strip()
    require(len(d) == n and d[id_col].nunique() == n, f'Frozen prediction count/uniqueness mismatch: {path}')
    y = pd.to_numeric(d[y_col], errors='raise').to_numpy(dtype=int)
    p = pd.to_numeric(d['probability_R'], errors='raise').to_numpy(dtype=float)
    z = pd.to_numeric(d['predicted_R'], errors='raise').to_numpy(dtype=int)
    require(np.isin(y, [0,1]).all() and int(y.sum()) == r, f'Frozen label guardrail failed: {path}')
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), f'Invalid frozen scores: {path}')
    require(np.array_equal(z, (p >= 0.5).astype(int)), f'Frozen threshold guardrail failed: {path}')
    c = conf(y,z)
    require(c == expected_confusion, f'Frozen confusion matrix mismatch at {path}: {c} != {expected_confusion}')
    return d


def conf(y, z) -> tuple[int,int,int,int]:
    return (int(((y==1)&(z==1)).sum()), int(((y==1)&(z==0)).sum()),
            int(((y==0)&(z==0)).sum()), int(((y==0)&(z==1)).sum()))


def metrics(y, p, threshold=0.5) -> dict:
    y, p = np.asarray(y,dtype=int), np.asarray(p,dtype=float)
    z = (p >= threshold).astype(int)
    tp,fn,tn,fp = conf(y,z)
    require(y.size >= 2 and set(y.tolist()) == {0,1}, 'Both classes required for AUROC')
    return dict(n=int(len(y)), R=int(y.sum()), S=int((1-y).sum()),
                tp=tp, fn=fn, tn=tn, fp=fp,
                auroc=float(roc_auc_score(y,p)),
                average_precision=float(average_precision_score(y,p)),
                sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
                balanced_accuracy=float(balanced_accuracy_score(y,z)),
                mcc=float(matthews_corrcoef(y,z)),
                brier=float(brier_score_loss(y,p)),
                false_susceptible_among_called_susceptible=fn/(tn+fn) if tn+fn else None)


def output_location(root: Path, out_rel: str) -> Path:
    results = (root/'results').resolve()
    o = (root/out_rel).resolve()
    require(o != results and results in o.parents, 'Output must be a new child directory of PROJECT_ROOT/results/')
    return o


def safe_save(out: Path, frames: dict[str,pd.DataFrame], objects: dict[str,dict], notes: dict[str,str], in_hashes:dict) -> None:
    # Atomic reservation, fail closed if this exact version was run already.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=False, exist_ok=False)
    for name,frame in frames.items():
        frame.to_csv(out/name, index=False, float_format='%.17g')
    for name,obj in objects.items():
        write_json(out/name,obj)
    for name,s in notes.items():
        (out/name).write_text(s,encoding='utf-8')
    write_json(out/'file18_input_sha256.json', in_hashes)
    produced=sorted(p for p in out.iterdir() if p.is_file())
    write_json(out/'file18_output_manifest_sha256.json',
               {'file':'FILE18','version':VERSION,
                'outputs_sha256':{p.name:sha256(p) for p in produced},
                'source_inputs_verified_unchanged':True})


def audit(root: Path, out_rel: str=OUTPUT_REL, quiet=False) -> dict:
    paths = input_files(root)
    missing = [PATHS[k] for k,p in paths.items() if not p.is_file()]
    require(not missing, 'Missing required input(s), run discover:\n  ' + '\n  '.join(missing))
    out = output_location(root,out_rel)
    require(not out.exists(), f'Output already exists (no overwrite): {out}')
    hashes = {k:sha256(p) for k,p in paths.items()}
    lock=load_json(paths['e2_lock'])
    require(hashes['e2_model'] == lock.get('model_artifact_sha256'), 'FILE11C model SHA mismatch vs model_lock')
    f13=load_json(paths['file13_manifest'])
    flag=load_json(paths['file13_freeze_flag'])
    f14=load_json(paths['file14_summary'])
    require(f13.get('status')=='PASS_HARMONIZED_DEVELOPMENT_FREEZE' and
            flag.get('status')=='PASS_HARMONIZED_DEVELOPMENT_FREEZE', 'FILE13 freeze manifest/flag not PASS')
    require(f14.get('status')=='PASS_E3_BLIND_EXTERNAL_VALIDATION', 'FILE14 confirmatory record not PASS')
    require(f13['sensitivity_feature_schema']['sha256'] == hashes['file13_668_schema'], 'FILE13 668 schema SHA mismatch')
    require(f14.get('FILE13',{}).get('primary_model_sha256')==f13['primary_model']['model_sha256'],
            'FILE14 record and FILE13 primary model SHA disagree')
    require(hashes['e2_model'] != f13['primary_model']['model_sha256'],
            'Unusual FILE11C vs FILE13 primary SHA equality; verify artifact identities')
    file13_features = schema_668(paths['file13_668_schema'])
    features = matrix_header_features(paths['e2_matrix'], 'asm_acc')  # ORIGINAL FILE11C ORDER
    want_schema_hash = hashlib.sha256(('\n'.join(features)+'\n').encode()).hexdigest()
    require(lock.get('feature_schema_sha256') == want_schema_hash,
            'FILE11C locked feature-order hash differs from its OWN E2 input matrix; stop.')
    feature_mapping = feature_set_gate(features, file13_features)
    e3_disk_header = matrix_header_features(paths['e3_668_matrix'], 'Genome ID')
    require(e3_disk_header == file13_features,
            'E3 668 matrix must match exact FILE13 frozen order BEFORE in-memory mapping')
    bundle=joblib.load(paths['e2_model'])  # TRUSTED USER-CREATED ARTIFACTS ONLY; joblib deserialization runs code.
    require(isinstance(bundle,dict) and set(['model','features']).issubset(bundle), 'FILE11C bundle schema mismatch')
    require(bundle['features']==features, 'FILE11C model-bundle feature order differs from its frozen E2 matrix')
    m=bundle['model']
    require(type(m).__name__=='LogisticRegression', 'Unexpected estimator, refuse same-model claim')
    require(getattr(m,'n_features_in_',None)==668, 'Frozen model not fitted on 668 features')
    require(np.array_equal(m.classes_,[0,1]), 'Unexpected model class ordering')
    spec=bundle.get('specification',{})
    require(spec.get('decision_threshold')==0.5 and spec.get('feature_count')==668,
            'FILE11C model frozen spec/threshold mismatch')

    d2=load_matrix(paths['e2_matrix'],'asm_acc',features,525)
    d3=load_matrix(paths['e3_668_matrix'],'Genome ID',file13_features,306)
    p2=labels_and_preds(paths['e2_predictions'],'asm_acc','label_binary',525,163,(109,54,221,141))
    p3=labels_and_preds(paths['e3_primary_predictions'],'Genome ID','y',306,122,(67,55,157,27))
    require(set(p2.asm_acc)==set(d2.asm_acc), 'E2 IDs not 1:1 between matrix and frozen predictions')
    require(set(p3['Genome ID'])==set(d3['Genome ID']), 'E3 IDs not 1:1 between matrix and frozen predictions')
    d2=d2.set_index('asm_acc').loc[p2['asm_acc']]
    d3=d3.set_index('Genome ID').loc[p3['Genome ID']]
    x2=d2[features].to_numpy(dtype=np.float32)
    x3=d3[features].to_numpy(dtype=np.float32)  # exact-name, bijective, in-memory E3 reorder to FILE11C model order
    s2=m.predict_proba(x2)[:,1]
    s3=m.predict_proba(x3)[:,1]
    require(np.isfinite(s2).all() and np.isfinite(s3).all(), 'Nonfinite cross-scores')
    error=np.abs(s2-p2['probability_R'].to_numpy(dtype=float))
    max_err=float(error.max())
    require(max_err <= 1e-10 and np.array_equal((s2>=0.5).astype(int),p2['predicted_R'].to_numpy(int)),
            f'MODEL IDENTITY GATE FAILED: FILE11C E2 probabilities do not reproduce original. Max difference={max_err:.3g}')
    # Entire input is read-only and must still be byte-for-byte unchanged.
    changed=[k for k,p in paths.items() if sha256(p)!=hashes[k]]
    require(not changed, f'Input changed during run: {changed}')
    met2=metrics(p2['label_binary'].to_numpy(dtype=int),s2)
    met3=metrics(p3['y'].to_numpy(dtype=int),s3)
    met3_original=metrics(p3['y'].to_numpy(dtype=int),p3['probability_R'].to_numpy(float))
    e2tab=pd.DataFrame({'cohort':'E2','genome_id':p2.asm_acc,'y_true':p2.label_binary.astype(int),
                        'probability_R_reproduced_FILE11C_model':s2,
                        'predicted_R':(s2>=.5).astype(int),
                        'same_model_sha256':hashes['e2_model']})
    e3tab=pd.DataFrame({'cohort':'E3','genome_id':p3['Genome ID'],'y_true':p3['y'].astype(int),
                        'probability_R_FILE11C_cross_score':s3,
                        'predicted_R':(s3>=.5).astype(int),
                        'same_model_sha256':hashes['e2_model'],
                        'original_FILE13_702_primary_probability_R':p3['probability_R'].astype(float)})
    metrows=[]
    for cohort,name,met,modelsha in [('E2','FILE11C_668_ORIGINAL_REPRODUCED',met2,hashes['e2_model']),
                                     ('E3','FILE11C_668_CROSS_SCORE_POSTHOC',met3,hashes['e2_model']),
                                     ('E3','FILE13_702_ORIGINAL_PRIMARY_REFERENCE',met3_original,f13['primary_model']['model_sha256'])]:
        metrows.append({'cohort':cohort,'analysis':name,'model_sha256':modelsha,**met})
    summary={
        'file':'FILE18','version':VERSION,'status':'PASS_POSTHOC_FILE11C_MODEL_CROSS_SCORE',
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'model_identity':{'file11c_original_model_sha256':hashes['e2_model'],
                          'frozen_model_unchanged':True,'feature_count':668,
                          'exact_feature_names_equal_after_verified_bijective_mapping':True,
                          'on_disk_original_feature_order_equal':feature_mapping['on_disk_order_equal'],
                          'e3_in_memory_column_permutation':feature_mapping['e3_columns_reordered_in_memory_for_file11c_model'],
                          'e2_reproduction_max_abs_prob_difference':max_err,
                          'e2_reproduced_predictions_identical':True},
        'frozen_original_primary_unchanged':True,
        'E2_same_file11c_model':met2,'E3_same_file11c_model_posthoc':met3,
        'E3_original_file13_primary_702_different_model_reference':met3_original,
        'E3_original_primary_model_sha256':f13['primary_model']['model_sha256'],
        'E2_vs_E3_same_model_weight_identity':True,
        'E3_feature_order_mapping':feature_mapping,
        'E2_vs_E3_annotation_protocol_identity':'NOT_ESTABLISHED: FILE11C historic preprocessing versus FILE14 harmonized E3 projection',
        'E2_E3_cohort_and_source_exchangeability':'NOT_ESTABLISHED',
        'secondary_analysis_only':True, 'no_training':True,'no_threshold_tuning':True,
        'no_recalibration':True,'no_cohort_redefinition':True,'no_primary_result_replacement':True,
        'no_confidence_intervals_or_p_values':True,'not_for_clinical_use':True,
        'interpretation': 'Use the same frozen FILE11C weight artifact on E2 and E3 668-column matrices. E3 annotation and processing are not assured identical to E2. Treat comparisons as post-hoc descriptive only.'}
    guardrails='''# FILE18 reporting guardrails\n\n1. The exact frozen FILE11C model is used for E2 and E3. FILE13/E3 on-disk order is matched to FILE11C via exact-name, bijective in-memory column permutation if needed; no renaming or imputation. No training or recalibration.\n2. **NOT** proof of exactly the same end-to-end preprocessing or feature semantics: original E2 FILE11C annotations and E3 FILE14 harmonized calls may differ even if names match.\n3. Do not call FILE18 a prospectively prespecified independent E3 validation of FILE11C.\n4. Do not replace E3 primary 702-feature result, and do not compare AUROCs causally across cohorts.\n5. No AST phenotype errors are adjudicated; no clinical predictions may be generated for use in care.\n6. Interpret with FILE17 near-neighbour selection caveat and FILE13 original-vs-harmonized development bridge audit.\n7. The SHA-locked model is loaded with joblib. Run only on your own trusted checkpoint; never deserialize untrusted model files.\n'''
    identity={'original_FILE11C_model_sha256':hashes['e2_model'],
              'FILE11C_model_lock_sha256':hashes['e2_lock'],
              'FILE13_668_schema_sha256':hashes['file13_668_schema'],
              'FILE11C_668_features_text_sha256':want_schema_hash,
              'FILE13_668_features_text_sha256':hashlib.sha256(('\n'.join(file13_features)+'\n').encode()).hexdigest(),
              'schema_alignment':feature_mapping,
              'E3_matched_FILE13_schema_exactly':True,
              'FILE13_primary_702_model_sha256':f13['primary_model']['model_sha256'],
              'E2_reproduction_max_abs_probability_difference':max_err,
              'E2_confusion':conf(p2.label_binary.to_numpy(int),(s2>=.5).astype(int)),
              'E3_original_primary_confusion':conf(p3.y.to_numpy(int),p3.predicted_R.to_numpy(int)),
              'feature_generation_protocol_identity':'NOT_VERIFIED'}
    safe_save(out,{
        'file18_cross_scored_E2_reproduction.csv':e2tab,
        'file18_cross_scored_E3_posthoc.csv':e3tab,
        'file18_same_model_comparison.csv':pd.DataFrame(metrows),
    },{
        'file18_summary.json':summary,
        'file18_model_identity_and_schema_audit.json':identity,
    },{'file18_reporting_guardrails.md':guardrails},hashes)
    if not quiet:
        print('[FILE18 PASS] Original E2 model reproduced exactly (tolerance <=1e-10); no refit.')
        print('[FILE18] E2 same model AUROC=',round(met2['auroc'],6), '| E3 same model post-hoc AUROC=',round(met3['auroc'],6))
        print('[FILE18] Original E3 702 model AUROC=',round(met3_original['auroc'],6),'(different model, descriptive reference)')
        print('[FILE18] Outputs:', out)
    return summary


def self_test() -> None:
    # End-to-end synthetic fixtures matching the original E2/E3 count and confusion guardrails.
    from sklearn.linear_model import LogisticRegression
    with tempfile.TemporaryDirectory(prefix='file18_test_') as tmp:
        root=Path(tmp)
        paths=input_files(root)
        for p in paths.values(): p.parent.mkdir(parents=True,exist_ok=True)
        f=[f'AMR_TEST_{i:03d}' for i in range(668)]
        pd.DataFrame({'feature':f}).to_csv(paths['file13_668_schema'],index=False)
        train=np.zeros((400,668),dtype=np.float32); train[200:,0]=1
        target=np.array([0]*200+[1]*200)
        model=LogisticRegression(solver='liblinear',C=1.0,max_iter=3000).fit(train,target)
        bundle={'model':model,'features':f,'specification':{'decision_threshold':0.5,'feature_count':668}}
        joblib.dump(bundle,paths['e2_model'])
        schema_hash=hashlib.sha256(('\n'.join(f)+'\n').encode()).hexdigest()
        write_json(paths['e2_lock'],{'model_artifact_sha256':sha256(paths['e2_model']), 'feature_schema_sha256':schema_hash})
        fake_primary='1'*64
        write_json(paths['file13_manifest'],{'status':'PASS_HARMONIZED_DEVELOPMENT_FREEZE',
                   'sensitivity_feature_schema':{'sha256':sha256(paths['file13_668_schema'])},
                   'primary_model':{'model_sha256':fake_primary}})
        write_json(paths['file13_freeze_flag'],{'status':'PASS_HARMONIZED_DEVELOPMENT_FREEZE'})
        write_json(paths['file14_summary'],{'status':'PASS_E3_BLIND_EXTERNAL_VALIDATION',
                   'FILE13':{'primary_model_sha256':fake_primary}})
        def mk(n,tag,confusion,idcol):
            tp,fn,tn,fp=confusion
            y=np.array([1]*(tp+fn)+[0]*(tn+fp))
            z=np.array([1]*tp+[0]*fn+[0]*tn+[1]*fp)
            x=np.zeros((n,668),dtype=np.float32);x[:,0]=z
            ids=[f'{tag}{i:04d}' for i in range(n)]
            mat=pd.DataFrame(x,columns=f);mat.insert(0,idcol,ids)
            return ids,y,z,mat,model.predict_proba(x)[:,1]
        ids2,y2,z2,d2,p2=mk(525,'E2_', (109,54,221,141),'asm_acc')
        ids3,y3,z3,d3,p3=mk(306,'E3_', (67,55,157,27),'Genome ID')
        d2.to_csv(paths['e2_matrix'],index=False);d3.to_csv(paths['e3_668_matrix'],index=False,compression='gzip')
        pd.DataFrame({'asm_acc':ids2,'label_binary':y2,'probability_R':p2,'predicted_R':z2}).to_csv(paths['e2_predictions'],index=False)
        # E3 702 original reference deliberately not the same model as FILE11C
        pp3=np.where(z3==1,.93,.08).astype(float)
        pd.DataFrame({'Genome ID':ids3,'y':y3,'probability_R':pp3,'predicted_R':z3}).to_csv(paths['e3_primary_predictions'],index=False)
        out_rel='results/file18_test_valid'
        summary=audit(root,out_rel,quiet=True)
        require(summary['E2_same_file11c_model']['n']==525 and summary['E3_same_file11c_model_posthoc']['n']==306,'self-test audit counts')
        require(summary['model_identity']['e2_reproduced_predictions_identical'],'self-test E2 reproduction')
        # Realistic FILE11C vs FILE13 semantic equality with DIFFERENT on-disk feature order.
        # FILE11C derives order from known-AMR matrix header; FILE13 from metadata.
        reordered=[f[1],f[0]]+f[2:]
        pd.DataFrame({'feature':reordered}).to_csv(paths['file13_668_schema'],index=False)
        j=load_json(paths['file13_manifest']); j['sensitivity_feature_schema']['sha256']=sha256(paths['file13_668_schema']);write_json(paths['file13_manifest'],j)
        d3[['Genome ID']+reordered].to_csv(paths['e3_668_matrix'],index=False,compression='gzip')
        diagnostic=schema_diagnostic(root)
        require(diagnostic['original_vs_harmonized']['exact_feature_set_equal'] and not diagnostic['original_vs_harmonized']['on_disk_order_equal'], 'self-test set equal, order differs')
        reordered_summary=audit(root,'results/file18_test_reordered',quiet=True)
        require(reordered_summary['E3_feature_order_mapping']['e3_columns_reordered_in_memory_for_file11c_model'], 'self-test E3 correct permutation')
        require(abs(reordered_summary['E3_same_file11c_model_posthoc']['auroc'] - summary['E3_same_file11c_model_posthoc']['auroc'])<1e-12,'self-test reorder changes E3 AUROC')
        # TRUE feature mismatch must STOP, without imputing zero or changing labels.
        malformed=['TRULY_MISSING_FEATURE']+reordered[1:]
        pd.DataFrame({'feature':malformed}).to_csv(paths['file13_668_schema'],index=False)
        j=load_json(paths['file13_manifest']); j['sensitivity_feature_schema']['sha256']=sha256(paths['file13_668_schema']);write_json(paths['file13_manifest'],j)
        try:
            audit(root,'results/file18_test_feature_set_mismatch',quiet=True)
        except RuntimeError as e:
            require('FEATURE SET MISMATCH' in str(e),'self-test actual feature mismatch fail-closed')
        else: raise RuntimeError('Self-test failed: different feature set accepted')
        pd.DataFrame({'feature':reordered}).to_csv(paths['file13_668_schema'],index=False)
        j=load_json(paths['file13_manifest']); j['sensitivity_feature_schema']['sha256']=sha256(paths['file13_668_schema']);write_json(paths['file13_manifest'],j)
        try:
            audit(root,out_rel,quiet=True)
        except RuntimeError as e:
            require('already exists' in str(e),'self-test no-overwrite')
        else: raise RuntimeError('Self-test failed: output overwrite was allowed')
        # Mutation of the model bundle must trigger model SHA guardrail.
        orig_bundle_bytes=paths['e2_model'].read_bytes()
        with paths['e2_model'].open('ab') as fh: fh.write(b'X')
        try:
            audit(root,'results/file18_test_mutation',quiet=True)
        except RuntimeError as e:
            require('model SHA mismatch' in str(e),'self-test model SHA lock')
        else: raise RuntimeError('Self-test failed: tampered model accepted')
        paths['e2_model'].write_bytes(orig_bundle_bytes)
        # Swapping two feature columns must fail instead of silently reordering.
        original_e3=paths['e3_668_matrix'].read_bytes()
        swapped=d3[['Genome ID']+f]
        swapped.to_csv(paths['e3_668_matrix'],index=False,compression='gzip')
        try:
            audit(root,'results/file18_test_order',quiet=True)
        except RuntimeError as e:
            require('E3 668 matrix must match exact FILE13 frozen order' in str(e),'self-test feature order')
        else: raise RuntimeError('Self-test failed: shuffled features accepted')
        paths['e3_668_matrix'].write_bytes(original_e3)
        # Reproduced FILE11C E2 scores must match exactly up to frozen numeric tolerance.
        original_e2_preds=paths['e2_predictions'].read_bytes()
        bad=pd.read_csv(paths['e2_predictions'])
        bad.loc[0,'probability_R']+=0.01
        bad.to_csv(paths['e2_predictions'],index=False)
        try:
            audit(root,'results/file18_test_scores',quiet=True)
        except RuntimeError as e:
            require('MODEL IDENTITY GATE FAILED' in str(e) or 'Frozen threshold guardrail failed' in str(e),
                    'self-test E2 prediction reproduction')
        else: raise RuntimeError('Self-test failed: altered E2 scores accepted')
        paths['e2_predictions'].write_bytes(original_e2_preds)
        # No production outputs were created by this self-test.
        print('[FILE18 SELF-TEST] PASS | 525/306 end-to-end | order-only permutation verified | TRUE feature mismatches fail-closed | no-overwrite | SHA tamper | E2 reproduction')


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('self-test')
    for action in ['discover','diagnose','audit']:
        x=sub.add_parser(action)
        x.add_argument('--project-root',type=Path,default=Path('.'))
    a=ap.parse_args()
    if a.action=='self-test': self_test();return
    root=a.project_root.expanduser().resolve()
    require(root.is_dir(),'project-root is not an existing directory')
    if a.action=='discover':
        status=preflight(root)
        out=output_location(root,'results/file18_same_model_discovery_v1')
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'file18_discovered_inputs.json',status)
        print(json.dumps(status,indent=2,ensure_ascii=False))
        print('[FILE18] Discovery saved:',out/'file18_discovered_inputs.json')
    elif a.action=='diagnose':
        report=schema_diagnostic(root,output=True)
        print(json.dumps({k:v for k,v in report.items() if k!='original_vs_harmonized'},indent=2,ensure_ascii=False))
        if report['original_vs_harmonized'] is not None:
            print('[FILE18] Difference positions:',report['original_vs_harmonized']['number_different_positions'])
    else: audit(root)

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'[FILE18 FAIL-CLOSED] {type(e).__name__}: {e}',file=__import__('sys').stderr)
        raise SystemExit(1)
