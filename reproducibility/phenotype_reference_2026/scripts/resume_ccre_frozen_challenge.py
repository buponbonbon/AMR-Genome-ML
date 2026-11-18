#!/usr/bin/env python3
"""Check or run the preselected CCRE challenge from authorized local FASTAs.

Missing inputs stop scoring; no archived older-DB features are substituted.
No model fitting, threshold tuning, label changes, or author contact.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from file11c_blind_external_validation import verify_amrfinder,run_one_amrfinder,parse_amrfinder_tsv
from file11e_external_simple_baseline_comparators import call_fixed_rules


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--fasta-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--amrfinder',type=Path)
    p.add_argument('--database',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();manifest=pd.read_csv(args.manifest,dtype=str,keep_default_na=False)
    if not manifest.sample_accession.is_unique or not manifest['S/I/R (PHAS interpretation)_Meropenem'].isin(['S','R']).all():raise ValueError('Manifest identifiers or original categories invalid')
    mappings=[]
    for row in manifest.to_dict('records'):
        names=[row['sample_accession'],row['genome_uuid'],row['provider_name']]
        candidates={args.fasta_dir/(n+ext) for n in names for ext in ['.fna','.fasta','.fa']}
        found=[f for f in candidates if f.is_file()]
        if len(found)>1:raise ValueError('Ambiguous FASTA filename mapping; provide one file per manifest sample')
        mappings.append({'biosample':row['sample_accession'],'fasta_path':str(found[0].resolve()) if found else '',
                         'fasta_sha256':sha(found[0]) if found else '',
                         'input_status':'PRESENT' if found else 'MISSING'})
    args.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(mappings).to_csv(args.out/'input_inventory.csv',index=False)
    status={'status':'READY' if all(x['input_status']=='PRESENT' for x in mappings) else 'BLOCKED_MISSING_AUTHORIZED_FASTAS',
            'target':len(manifest),'FASTA_present':sum(x['input_status']=='PRESENT' for x in mappings),
            'manifest_sha256':sha(args.manifest),'check_only':args.check_only,'new_model_training':False}
    (args.out/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2),flush=True)
    if args.check_only:return
    if status['status']!='READY':raise SystemExit('Scoring stopped. Missing FASTAs are never replaced with old-DB summaries.')
    if not args.amrfinder or not args.database:raise ValueError('Exact caller and database required for scoring')
    if sha(args.model)!='1f55620cbd4b1dfd946ff37977e9a851b6aeaad8a5466eb553323331613cdcdd':raise ValueError('Frozen model hash mismatch')
    caller=verify_amrfinder(str(args.amrfinder.resolve()),args.database.resolve())
    (args.out/'caller_preflight.json').write_text(json.dumps(caller,indent=2)+'\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');artifact=joblib.load(args.model)
    features=artifact['features'];model=artifact['model']
    if hashlib.sha256(('\n'.join(features)+'\n').encode()).hexdigest()!='9ca9c8c89a6bb2c1a97d85b9b5906efbf9aee0fca7eedb0328915063212c132f':raise ValueError('Feature schema changed')
    def call(item):
        output=args.out/'amrfinder'/(item['biosample']+'.tsv')
        ok,detail=run_one_amrfinder(str(args.amrfinder.resolve()),args.database.resolve(),Path(item['fasta_path']),output)
        return dict(item,caller_ok=ok,caller_detail=detail,amr_path=str(output))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:called=list(pool.map(call,mappings))
    pd.DataFrame(called).to_csv(args.out/'caller_status.csv',index=False)
    if not all(x['caller_ok'] for x in called):raise SystemExit('A caller failure prevents reporting a completed selected challenge; inspect caller_status.csv')
    X=[];rules=[]
    for item in called:
        path=Path(item['amr_path']);genes=parse_amrfinder_tsv(path)
        X.append([int(f.lower() in genes) for f in features]);r,_=call_fixed_rules(path);rules.append(r)
    X=np.asarray(X,dtype=np.float32);scores=model.predict_proba(X)[:,1]
    manual=expit(X.astype(float)@model.coef_[0]+model.intercept_[0]);assert np.max(np.abs(scores-manual))<1e-12
    output=manifest.copy();output['frozen_probability_R']=scores;output['frozen_predicted_R']=(scores>=.5).astype(int)
    for key in rules[0]:output[key]=[r[key] for r in rules]
    output.to_csv(args.out/'frozen_challenge_predictions.csv',index=False)
    status.update(status='COMPLETE_FROZEN_CALLER_CHALLENGE_WITH_REPORTED_SOURCE_LIMITATIONS',scored=len(output),warnings=[str(w.message) for w in caught])
    (args.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()
