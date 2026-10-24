#!/usr/bin/env python3
"""FILE16H - Exact frozen FILE11E genotype-rule transfer from E2 to E3.

Research use only. Does not retrain, tune thresholds, change phenotype,
replace original E2/E3 metrics, or issue clinical decisions.

The exact function `call_fixed_rules` is IMPORTED from scripts/
file11e_external_simple_baseline_comparators.py, rather than re-implemented.
E2 calls MUST reproduce its previously saved file11e_rule_calls.csv for all 525
samples before any E3 result is reported. All 831 TSV SHA-256 must match FILE16B.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED={"E2":(525,163), "E3":(306,122)}


def load_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def relpath(root,s):
    p=Path(s)
    return p if p.is_absolute() else root/p

def unique(rows, field, label):
    keys=[str(r[field]).strip() for r in rows]
    if not all(keys) or len(keys)!=len(set(keys)):
        raise ValueError(f'{label}: blank or duplicated {field}')
    return {k:r for k,r in zip(keys,rows)}

def metrics(rows, method):
    c=Counter()
    for r in rows:
        y=int(r['y_true']); p=int(r[method])
        c[('TP' if y==1 and p==1 else 'FN' if y==1 else 'FP' if p==1 else 'TN')]+=1
    n=len(rows); r=c['TP']+c['FN']; s=c['TN']+c['FP']
    return dict(n=n,n_resistant=r,n_susceptible=s,tp=c['TP'],fn=c['FN'],tn=c['TN'],fp=c['FP'],
                sensitivity=(c['TP']/r if r else None),specificity=(c['TN']/s if s else None),
                balanced_accuracy=((c['TP']/r+c['TN']/s)/2 if r and s else None),
                accuracy=(c['TP']+c['TN'])/n if n else None)

def paired(rows, rule, group_filter=None):
    counts=Counter()
    for r in rows:
        if group_filter is not None and int(r['y_true'])!=group_filter: continue
        a=int(r['ml_call'])==int(r['y_true'])
        b=int(r[rule])==int(r['y_true'])
        counts[(a,b)]+=1
    a=counts[(True,False)]; b=counts[(False,True)]
    return {'ml_correct_rule_wrong':a,'ml_wrong_rule_correct':b,
            'discordant':a+b,
            'difference_correct_rule_minus_ml':b-a,
            'notes':'paired descriptive counts; not independence-adjusted inferential p-values'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',default='.')
    ap.add_argument('--evidence',default='results/file16b_annotation_v1/file16b_family_annotation_evidence.csv')
    ap.add_argument('--predictions',default='results/file16_audit_mechanisms_v1/file16_per_genome_prediction_error_audit.csv')
    ap.add_argument('--frozen-e2-rules',default='data/external_validation/ncbi_pathogen_detection/file11e/file11e_rule_calls.csv')
    ap.add_argument('--rule-definition-script',default='scripts/file11e_external_simple_baseline_comparators.py')
    ap.add_argument('--outdir',default='results/file16h_exact_rule_transfer_v1')
    args=ap.parse_args()
    root=Path(args.project_root).resolve()
    inputs={k:relpath(root,v) for k,v in [
        ('evidence',args.evidence),('predictions',args.predictions),
        ('frozen_e2_rules',args.frozen_e2_rules),('rule_definition',args.rule_definition_script)]}
    for k,p in inputs.items():
        if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
    out=relpath(root,args.outdir)
    if out.exists():raise FileExistsError(f'Output exists; choose a new --outdir: {out}')
    evid=load_csv(inputs['evidence']); pred=load_csv(inputs['predictions']); baseline=load_csv(inputs['frozen_e2_rules'])
    if len(evid)!=831 or len(pred)!=831 or len(baseline)!=525:
        raise ValueError(f'Unexpected counts: evidence={len(evid)}, predictions={len(pred)}, E2 saved rules={len(baseline)}')
    evidence=unique(evid,'genome_id','evidence')
    prediction=unique(pred,'genome_id','predictions')
    e2calls=unique(baseline,'asm_acc','saved E2 rule calls')
    if set(evidence)!=set(prediction):
        raise ValueError('Evidence and prediction genome IDs do not match exactly')
    for cohort, (n,nr) in EXPECTED.items():
        q=[r for r in pred if r['cohort']==cohort]
        if len(q)!=n or sum(int(r['y_true']) for r in q)!=nr: raise ValueError(f'Unexpected {cohort} cohort labels/counts')
    if {r['genome_id'] for r in pred if r['cohort']=='E2'}!=set(e2calls):
        raise ValueError('Saved E2 rule calls not keyed to the same 525 E2 genomes')
    for r in pred:
        if r['cohort']!=evidence[r['genome_id']]['cohort']:
            raise ValueError(f"Cohort mismatch for {r['genome_id']}")
    spec=importlib.util.spec_from_file_location('file11e_rules_frozen_for_file16h', inputs['rule_definition'])
    if not spec or not spec.loader:raise RuntimeError('Cannot load original FILE11E rule definition')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if not hasattr(mod,'call_fixed_rules'):raise ValueError('FILE11E has no call_fixed_rules')
    records=[]
    for ix, r in enumerate(pred,1):
        id=r['genome_id']; e=evidence[id]; tsv=Path(e['raw_tsv'])
        if not tsv.is_file(): raise FileNotFoundError(f'Missing raw annotation for {id}: {tsv}')
        digest=sha256(tsv)
        if digest!=e['raw_tsv_sha256']:
            raise ValueError(f'Frozen raw TSV hash mismatch for {id}: {digest}')
        calls,_=mod.call_fixed_rules(tsv)
        kpc=int(calls['kpc_family_rule']); broad=int(calls['amrfinder_carbapenemase_rule'])
        if r['cohort']=='E2':
            old=e2calls[id]
            if kpc!=int(old['kpc_family_rule']) or broad!=int(old['amrfinder_carbapenemase_rule']):
                raise ValueError(f'CRITICAL E2 RULE PARITY FAILURE for {id}; stop, do not report E3')
        y=int(r['y_true']); ml=int(r['predicted_resistant_at_frozen_threshold_05'])
        if y not in (0,1) or ml not in (0,1) or not 0<=float(r['prob_resistance'])<=1:
            raise ValueError(f'Invalid frozen prediction for {id}')
        records.append(dict(cohort=r['cohort'],genome_id=id,y_true=y,ml_score=r['prob_resistance'],
                            ml_call=ml,kpc_family_rule=kpc,amrfinder_carbapenemase_rule=broad,
                            ml_error=int(ml!=y),kpc_error=int(kpc!=y),broad_rule_error=int(broad!=y),
                            mechanism_group=e['mechanism_group'],erd_group=r.get('erd_group',''),
                            tsv_sha256=digest))
        if ix%100==0 or ix==831: print(f'[FILE16H] E2/E3 exact rule transfer, verified {ix}/831 SHA-256 TSV',flush=True)
    results=[]; comparisons=[]
    methods=('ml_call','kpc_family_rule','amrfinder_carbapenemase_rule')
    for cohort in ('E2','E3'):
        sub=[r for r in records if r['cohort']==cohort]
        for method in methods: results.append(dict(cohort=cohort,method=method,**metrics(sub,method)))
        for method in methods[1:]:
            for strat,yfilter in [('all',None),('resistant',1),('susceptible',0)]:
                comparisons.append(dict(cohort=cohort,rule_method=method,stratum=strat,
                                        **paired(sub,method,yfilter)))
    out.mkdir(parents=True)
    write_csv(out/'file16h_per_genome_fixed_rule_calls.csv',records,list(records[0]))
    write_csv(out/'file16h_method_metrics.csv',results,list(results[0]))
    write_csv(out/'file16h_paired_error_contrasts.csv',comparisons,list(comparisons[0]))
    summary={
        'status':'POST_HOC_APPLIED_RULE_TRANSFER_NOT_CLINICAL',
        'date_utc':datetime.now(timezone.utc).isoformat(),
        'frozen_cohorts':{'E2':525,'E3':306},'TSV_SHA256_verified':831,
        'E2_rule_parity':'PASS_525_OF_525_EXACT' ,'rule_policy':'Imported exact FILE11E call_fixed_rules without tuning',
        'method_metrics':results,'paired_error_contrasts':comparisons,
        'inputs_sha256':{k:sha256(p) for k,p in inputs.items()},
        'caveats':[
            'Both cohorts assessed by the same fixed biological rule definition, but ML models are different (668 vs 702 features).',
            'E3 AMRFinder input protocol may not be identical to E2; protocol metadata require separate verification.',
            'Comparisons are post-hoc; E2 unusual source labels have not been independently confirmed with AST/MIC.',
            'Paired contrasts are descriptive, not clonal or source adjusted statistical evidence.',
            'E3 BioProject assignment and true sequence independence have not been established.',
            'Positive rule call is genomic annotation, not an AST laboratory measurement or clinical decision.',
            'Original frozen E2 and E3 summary metrics remain unchanged.'
        ]}
    (out/'file16h_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('[FILE16H] E2 rule parity PASS (525/525), E3 exact rule calls complete (306).')
    print(json.dumps({'methods':results},indent=2))
    print(f'[FILE16H] COMPLETE: {out}')

if __name__=='__main__':main()
