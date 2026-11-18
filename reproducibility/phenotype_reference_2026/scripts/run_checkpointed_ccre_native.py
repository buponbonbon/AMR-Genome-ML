#!/usr/bin/env python3
"""Run the unchanged frozen caller/model/rules with audited, resumable case checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from file11c_blind_external_validation import (verify_amrfinder,run_one_amrfinder,
    parse_amrfinder_tsv,feature_vector_from_keys)
from file11e_external_simple_baseline_comparators import call_fixed_rules


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','fasta-dir','model','amrfinder','database','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed-out',type=Path);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--allow-reported-missing',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();manifest=pd.read_csv(a.manifest,dtype=str,keep_default_na=False)
    if not manifest.sample_accession.is_unique or not manifest['S/I/R (PHAS interpretation)_Meropenem'].isin(['S','R']).all():raise ValueError('Invalid fixed manifest')
    model_sha=sha(a.model)
    if model_sha!='1f55620cbd4b1dfd946ff37977e9a851b6aeaad8a5466eb553323331613cdcdd':raise ValueError('Frozen artifact hash mismatch')
    a.out.mkdir(parents=True,exist_ok=a.resume)
    for name in ['amrfinder','receipts']:(a.out/name).mkdir(exist_ok=True)
    signature={'manifest_sha256':sha(a.manifest),'model_sha256':model_sha,'caller_exe_sha256':sha(a.amrfinder),'database_version':(a.database/'version.txt').read_text().strip(),'database_manifest_sha256':sha(a.database/'manifest.json')}
    signature_path=a.out/'RUN_SIGNATURE.json'
    if signature_path.exists() and json.loads(signature_path.read_text())!=signature:raise ValueError('Resume signature changed')
    save(signature_path,signature);preflight=verify_amrfinder(str(a.amrfinder.resolve()),a.database.resolve());save(a.out/'caller_preflight.json',preflight)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');artifact=joblib.load(a.model)
    features=artifact['features'];model=artifact['model']
    if hashlib.sha256(('\n'.join(features)+'\n').encode()).hexdigest()!='9ca9c8c89a6bb2c1a97d85b9b5906efbf9aee0fca7eedb0328915063212c132f':raise ValueError('Frozen schema mismatch')
    seed={}
    if a.seed_out:
        status=json.loads((a.seed_out/'status.json').read_text())
        if not status.get('status','').startswith('COMPLETE_FROZEN_CALLER'):raise ValueError('Seed checkpoint is not complete')
        seed_preflight=json.loads((a.seed_out/'caller_preflight.json').read_text())
        if seed_preflight!=preflight:raise ValueError('Seed caller or database preflight differs')
        inv=pd.read_csv(a.seed_out/'input_inventory.csv',dtype=str,keep_default_na=False).set_index('biosample')
        for sample,row in inv.iterrows():seed[sample]=row['fasta_sha256']
    jobs=[];inventory=[];missing=[]
    for row in manifest.to_dict('records'):
        sample=row['sample_accession'];fasta=a.fasta_dir/(sample+'.fna')
        item={'sample_accession':sample,'fasta_path':str(fasta.resolve()),'fasta_sha256':sha(fasta) if fasta.is_file() else ''}
        inventory.append(item)
        if not fasta.is_file():missing.append(dict(item,status='MISSING_FASTA_NO_SCORE'));continue
        jobs.append(item)
    pd.DataFrame(inventory).to_csv(a.out/'input_inventory.csv',index=False)
    if missing and not a.allow_reported_missing:raise SystemExit('Missing fixed-manifest FASTAs; no zero-feature substitution permitted')
    def call(item):
        sample=item['sample_accession'];output=a.out/'amrfinder'/(sample+'.tsv');receipt=a.out/'receipts'/(sample+'.json')
        expected=dict(item,caller_exe_sha256=signature['caller_exe_sha256'],database_manifest_sha256=signature['database_manifest_sha256'])
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old.get('status')=='PASS' and all(old.get(k)==v for k,v in expected.items()) and output.is_file() and sha(output)==old.get('output_sha256'):
                parse_amrfinder_tsv(output);return dict(old,execution='REUSED_VERIFIED_CHECKPOINT')
            if old.get('status')=='PASS':raise ValueError('Existing successful case no longer matches verified provenance')
        if output.exists():
            history=a.out/'unverified_or_failed_outputs';history.mkdir(exist_ok=True)
            output.rename(history/(sample+'.'+str(time.time_ns())+'.tsv'))
        started=time.monotonic();result=dict(expected,started_utc=datetime.now(timezone.utc).isoformat())
        try:
            if sample in seed:
                if seed[sample]!=item['fasta_sha256']:raise ValueError('Seed FASTA differs from current immutable input')
                source=a.seed_out/'amrfinder'/(sample+'.tsv');parse_amrfinder_tsv(source);shutil.copy2(source,output);ok=True;detail='REUSED_COMPLETE_INITIAL40'
            else:ok,detail=run_one_amrfinder(str(a.amrfinder.resolve()),a.database.resolve(),Path(item['fasta_path']),output)
            if not ok:raise RuntimeError(detail)
            if sha(Path(item['fasta_path']))!=item['fasta_sha256']:raise ValueError('Caller input changed during execution')
            result.update(status='PASS',execution=detail,output_sha256=sha(output))
        except Exception as error:result.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        result['seconds']=time.monotonic()-started;save(receipt,result);return result
    started=time.monotonic();completed=list(missing)
    def progress():
        status={'status':'RUNNING_NATIVE_CALLER','target':len(manifest),'completed':len(completed),'passed':sum(r['status']=='PASS' for r in completed),'failed_or_missing':sum(r['status']!='PASS' for r in completed),'elapsed_seconds':time.monotonic()-started}
        save(a.out/'progress.json',status);pd.DataFrame(completed).to_csv(a.out/'caller_status.csv',index=False);print(json.dumps(status),flush=True)
    progress()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(call,item) for item in jobs]
        for future in as_completed(futures):completed.append(future.result());progress()
    bysample={r['sample_accession']:r for r in completed};predictions=[];matrix=[];matrix_samples=[];native_manual_differences=[]
    for row in manifest.to_dict('records'):
        sample=row['sample_accession'];receipt=bysample[sample];row['native_input_status']=receipt['status']
        if receipt['status']=='PASS':
            path=a.out/'amrfinder'/(sample+'.tsv');keys=parse_amrfinder_tsv(path);x=feature_vector_from_keys(keys,features).astype(np.float32)
            probability=float(model.predict_proba(x.reshape(1,-1))[0,1]);manual=float(expit(x.astype(float)@model.coef_[0]+model.intercept_[0]));native_manual_differences.append(abs(probability-manual))
            if abs(probability-manual)>1e-12:raise ValueError('Native model and independent linear score disagree')
            calls,_=call_fixed_rules(path);row.update(frozen_probability_R=probability,frozen_predicted_R=int(probability>=.5),**calls);matrix.append(x.astype(np.uint8));matrix_samples.append(sample)
        predictions.append(row)
    pd.DataFrame(predictions).to_csv(a.out/'frozen_challenge_predictions.csv',index=False)
    np.savez_compressed(a.out/'native_features.npz',X=np.asarray(matrix),features=np.asarray(features),biosamples=np.asarray(matrix_samples))
    final={'status':'COMPLETE_ALL_FIXED_CASES' if len(matrix)==len(manifest) else 'COMPLETE_AVAILABLE_CASES_WITH_REPORTED_MISSING_OR_FAILED_INPUTS','target':len(manifest),'scored':len(matrix),'missing_or_failed':len(manifest)-len(matrix),'maximum_native_manual_probability_difference':max(native_manual_differences,default=None),'model_sha256':model_sha,'threshold':.5,'caller':'AMRFinderPlus4.2.7 / database2026-08-07.1 / nucleotide','load_warnings':[str(w.message) for w in caught],'model_fitting_or_tuning_performed':False,'clinical_utility_or_absolute_clonal_independence_claimed':False,'elapsed_seconds':time.monotonic()-started}
    save(a.out/'status.json',final);print(json.dumps(final,indent=2),flush=True)


if __name__=='__main__':main()
