#!/usr/bin/env python3
"""FILE16I — applied AMR paired reliability + metadata evidence lock.

Post-hoc research audit ONLY. No model training, threshold selection, label correction,
model recalibration, cohort exclusion, or clinical decision support.

Python 3.10+, numpy. All output goes into a NEW directory; never overwrites inputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MISSING = {'', 'na', 'n/a', 'nan', 'none', 'null', 'unknown', 'unresolved', '-', '?'}
METHODS = ('ml_call', 'amrfinder_carbapenemase_rule')
EXPECTED = {'E2': (525, 163), 'E3': (306, 122)}


def norm(x):
    return str(x or '').strip()


def present(x):
    return norm(x).lower() not in MISSING


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f'No CSV header: {path}')
        return list(r), r.fieldnames


def select_field(fields, names, required=True):
    lc = {i.strip().lower(): i for i in fields}
    for n in names:
        if n.strip().lower() in lc:
            return lc[n.strip().lower()]
    if required:
        raise ValueError(f'Missing one of {names}; available: {fields}')
    return None


def bool01(value, desc):
    s = norm(value).lower()
    if s in {'0', 'false', 's', 'susceptible'}: return 0
    if s in {'1', 'true', 'r', 'resistant'}: return 1
    raise ValueError(f'Expected 0/1 for {desc}; received {value!r}')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fieldnames):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
        w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in fieldnames})


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def percentiles(values):
    vals=np.asarray([x for x in values if np.isfinite(x)],dtype=float)
    if len(vals)==0: return {'lower':None,'upper':None,'median':None,'valid_replicates':0}
    a,b,c=np.quantile(vals,[0.025,0.975,0.5])
    return {'lower':float(a),'upper':float(b),'median':float(c),'valid_replicates':int(len(vals))}


def stats(arr, ix=None):
    y=arr['y']; m=arr['ml']; r=arr['rule']
    if ix is not None:
        y=y[ix];m=m[ix];r=r[ix]
    rmask=y==1; smask=~rmask
    out={'n':int(len(y)),'n_resistant':int(rmask.sum()),'n_susceptible':int(smask.sum())}
    for name,pred in [('ml',m),('rule',r)]:
        tp=int(np.sum((pred==1)&rmask));fn=int(np.sum((pred==0)&rmask))
        tn=int(np.sum((pred==0)&smask));fp=int(np.sum((pred==1)&smask))
        out[name+'_tp']=tp;out[name+'_fn']=fn;out[name+'_tn']=tn;out[name+'_fp']=fp
        out[name+'_sensitivity']=(tp/(tp+fn)) if tp+fn else None
        out[name+'_specificity']=(tn/(tn+fp)) if tn+fp else None
        out[name+'_balanced_accuracy']=(out[name+'_sensitivity']+out[name+'_specificity'])/2 if rmask.any() and smask.any() else None
        out[name+'_false_susceptible_among_called_susceptible']=(fn/(fn+tn)) if fn+tn else None
    for metric in ['sensitivity','specificity','balanced_accuracy']:
        a=out['rule_'+metric];b=out['ml_'+metric]
        out['delta_rule_minus_ml_'+metric]=a-b if a is not None and b is not None else None
    out['ml_correct_rule_wrong']=int(np.sum((m==y)&(r!=y)))
    out['ml_wrong_rule_correct']=int(np.sum((m!=y)&(r==y)))
    out['rule_positive_ml_negative']=int(np.sum((m==0)&(r==1)))
    out['ml_positive_rule_negative']=int(np.sum((m==1)&(r==0)))
    out['both_false_susceptible']=int(np.sum(rmask&(m==0)&(r==0)))
    out['rule_extra_resistant_true']=int(np.sum(rmask&(m==0)&(r==1)))
    out['rule_extra_resistant_false']=int(np.sum(smask&(m==0)&(r==1)))
    return out


def array_data(rows):
    return {'y':np.asarray([r['y'] for r in rows],dtype=np.int8),
            'ml':np.asarray([r['ml'] for r in rows],dtype=np.int8),
            'rule':np.asarray([r['rule'] for r in rows],dtype=np.int8)}


def bootstrap(arr, groups, nrep, seed, group_label):
    rng=np.random.default_rng(seed)
    m=defaultdict(list)
    for i,g in enumerate(groups): m[g].append(i)
    ix=[np.asarray(v,dtype=int) for v in m.values()]
    deltas=defaultdict(list)
    no_both=0
    for _ in range(nrep):
        choice=rng.integers(0,len(ix),len(ix))
        samples=np.concatenate([ix[j] for j in choice])
        s=stats(arr,samples)
        if not (s['n_resistant'] and s['n_susceptible']): no_both+=1
        for metric in ['sensitivity','specificity','balanced_accuracy']:
            value=s['delta_rule_minus_ml_'+metric]
            if value is not None: deltas[metric].append(value)
    return {'grouping':group_label,'n_clusters':len(ix),'n_replicates':nrep,
            'resamples_without_both_classes':no_both,
            'delta_rule_minus_ml':{m:percentiles(v) for m,v in deltas.items()}}


def id_normalized(x):
    return norm(x).upper()


def acc_no_version(x):
    return id_normalized(x).split('.')[0] if present(x) else ''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default='.')
    p.add_argument('--rule-predictions',help='FILE16H per-genome rule calls CSV; defaults to results/...')
    p.add_argument('--e2-source',help='FILE16D manual review priority CSV; defaults to results/...')
    p.add_argument('--file16h-summary',help='FILE16H summary JSON; defaults to results/...')
    p.add_argument('--e3-frozen',default='',help='Optional FILE14 E3 FINAL FROZEN CSV; verifies IDs, labels, metadata completeness')
    p.add_argument('--e3-bioproject-map',default='',help='Optional independent CSV mapping genome ID / BioSample to BioProject; NOT E3 Source field')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--outdir',required=True)
    args=p.parse_args()
    if args.bootstrap<200: p.error('bootstrap must be at least 200')
    root=Path(args.project_root).expanduser().resolve()
    pred_path=Path(args.rule_predictions or root/'results/file16h_exact_rule_transfer_v1/file16h_per_genome_fixed_rule_calls.csv').resolve()
    meta_path=Path(args.e2_source or root/'results/file16d_source_audit_v1/file16d_manual_review_priority.csv').resolve()
    summary_path=Path(args.file16h_summary or root/'results/file16h_exact_rule_transfer_v1/file16h_summary.json').resolve()
    outdir=Path(args.outdir).expanduser().resolve()
    if outdir.exists(): raise SystemExit(f'REFUSING TO OVERWRITE output dir: {outdir} (use new _v2)')
    for required in [pred_path,meta_path,summary_path]:
        if not required.is_file():raise FileNotFoundError(str(required))
    prior=json.loads(summary_path.read_text(encoding='utf-8'))
    if prior.get('E2_rule_parity')!='PASS_525_OF_525_EXACT' or prior.get('TSV_SHA256_verified')!=831:
        raise ValueError('FILE16H E2 rule parity or 831 SHA-256 verification NOT confirmed')
    data,fields=read_csv(pred_path)
    needed=['cohort','genome_id','y_true','ml_call','amrfinder_carbapenemase_rule','ml_score']
    for n in needed:
        if n not in fields: raise ValueError(f'Missing field {n} from FILE16H')
    bycohort=defaultdict(list);seen=set()
    for d in data:
        c=norm(d['cohort']);gid=norm(d['genome_id']);key=(c,gid)
        if c not in EXPECTED or not gid or key in seen: raise ValueError(f'Bad cohort or duplicate genome ID: {key}')
        seen.add(key)
        row={'genome_id':gid,'y':bool01(d['y_true'],'y_true'),
             'ml':bool01(d['ml_call'],'ml_call'),
             'rule':bool01(d['amrfinder_carbapenemase_rule'],'rule'),
             'erd':norm(d.get('erd_group')),'score':float(d['ml_score'])}
        if not 0<=row['score']<=1: raise ValueError(f'Out of range ML score for {key}')
        # Model decision threshold expected to be fixed at 0.5 (validate, never alter).
        if row['ml']!=int(row['score']>=0.5):raise ValueError(f'Frozen ML label-score mismatch: {key}')
        bycohort[c].append(row)
    for c,(n,r) in EXPECTED.items():
        if len(bycohort[c])!=n or sum(x['y'] for x in bycohort[c])!=r:
            raise ValueError(f'Unexpected {c} n or R count; do not proceed: {len(bycohort[c])}/{sum(x["y"] for x in bycohort[c])}')
    source,srcfields=read_csv(meta_path)
    for k in ['asm_acc','bioproject_acc','biosample_acc','erd_group','y_true']:
        if k not in srcfields:raise ValueError(f'E2 source audit missing {k}')
    smap={}
    for d in source:
        key=norm(d['asm_acc'])
        if not key or key in smap: raise ValueError(f'Missing/duplicate E2 assembly in source: {key}')
        smap[key]=d
    if len(smap)!=525:raise ValueError(f'E2 metadata must include 525 rows, got {len(smap)}')
    for d in bycohort['E2']:
        src=smap.get(d['genome_id'])
        if src is None:raise ValueError(f'No E2 metadata match: {d["genome_id"]}')
        if d['y']!=bool01(src['y_true'],'E2 metadata y_true'):raise ValueError(f'E2 source label mismatch {d["genome_id"]}')
        if not present(src['bioproject_acc']):raise ValueError(f'No BioProject for E2 {d["genome_id"]}')
        d['bioproject']=norm(src['bioproject_acc'])
        d['assembly']=norm(src['asm_acc']); d['biosample']=norm(src['biosample_acc'])
        if present(src['erd_group']) and present(d['erd']) and src['erd_group']!=d['erd']:
            raise ValueError(f'E2 ERD mismatch {d["genome_id"]}')
    for d in bycohort['E3']:
        d.update({'bioproject':'','assembly':'','biosample':''})
    e3frozen_meta={'status':'NOT_PROVIDED','n_rows':None,'n_unmatched':None,'n_label_mismatch':None,'n_assembly':0,'n_biosample':0,'n_erd':0,'n_source_field':0,'n_bioproject':0,'notes':'E3 Source field is NOT a BioProject accession.'}
    if args.e3_frozen:
        ep=Path(args.e3_frozen).resolve();rows,hdr=read_csv(ep)
        idf=select_field(hdr,['Genome ID','genome_id'])
        yfield=select_field(hdr,['Phenotype','y','label_binary'],required=False)
        af=select_field(hdr,['Assembly Accession','assembly_accession'],required=False)
        bf=select_field(hdr,['BioSample','biosample'],required=False)
        gf=select_field(hdr,['NCBI SNP/ERD Group','erd_group'],required=False)
        sf=select_field(hdr,['Source'],required=False)
        if len(rows)!=306: raise ValueError(f'E3 frozen manifest should have 306, got {len(rows)}')
        lookup={}
        for d in rows:
            gid=norm(d[idf]);
            if not gid or gid in lookup: raise ValueError(f'Duplicate E3 frozen ID: {gid}')
            lookup[gid]=d
        mismatch=0;nomatch=0
        for d in bycohort['E3']:
            m=lookup.get(d['genome_id'])
            if m is None:nomatch+=1;continue
            if yfield and present(m[yfield]) and d['y']!=bool01(m[yfield],'E3 frozen phenotype'): mismatch+=1
            if af: d['assembly']=norm(m[af])
            if bf: d['biosample']=norm(m[bf])
            if gf:
                g=norm(m[gf]);
                if present(g) and present(d['erd']) and g!=d['erd']: raise ValueError(f'E3 ERD mismatch {d["genome_id"]}')
                if present(g):d['erd']=g
        if nomatch or mismatch:raise ValueError(f'E3 frozen mismatch: ids_missing={nomatch}, phenotype={mismatch}')
        e3frozen_meta={'status':'MATCHED_FROZEN_IDS','n_rows':len(rows),'n_unmatched':nomatch,'n_label_mismatch':mismatch,
                       'n_assembly':sum(present(d['assembly']) for d in bycohort['E3']),
                       'n_biosample':sum(present(d['biosample']) for d in bycohort['E3']),
                       'n_erd':sum(present(d['erd']) for d in bycohort['E3']),
                       'n_source_field':sum(present(m[sf]) for m in rows) if sf else 0,
                       'n_bioproject':0,'notes':'The optional E3 Source field was not used as BioProject accession.'}
    if args.e3_bioproject_map:
        mp,hdr=read_csv(args.e3_bioproject_map)
        idf=select_field(hdr,['genome_id','Genome ID','biosample','BioSample'])
        pf=select_field(hdr,['bioproject_acc','BioProject Accession','BioProject'])
        lut={}
        for row in mp:
            k=norm(row[idf]); v=norm(row[pf]);
            if k in lut and lut[k]!=v: raise ValueError(f'Conflicting E3 BioProject map for {k}')
            lut[k]=v
        nmap=0
        for row in bycohort['E3']:
            k=row['biosample'] if idf.lower()=='biosample' else row['genome_id']
            v=lut.get(k,'')
            if present(v): row['bioproject']=v; nmap+=1
        e3frozen_meta['n_bioproject']=nmap
        e3frozen_meta['bioproject_map_match_count']=nmap
        e3frozen_meta['bioproject_map_complete']=nmap==306
    outdir.mkdir(parents=True,exist_ok=False)
    fingerprints={str(pred_path):sha(pred_path),str(meta_path):sha(meta_path),str(summary_path):sha(summary_path)}
    if args.e3_frozen: fingerprints[str(Path(args.e3_frozen).resolve())]=sha(args.e3_frozen)
    if args.e3_bioproject_map: fingerprints[str(Path(args.e3_bioproject_map).resolve())]=sha(args.e3_bioproject_map)
    cohort_arr={c:array_data(rows) for c,rows in bycohort.items()}
    evidence={c:stats(a) for c,a in cohort_arr.items()}
    for c in EXPECTED:
        if evidence[c]['ml_positive_rule_negative']!=0:
            print(f'NOTICE: {c} not nested; comparison remains valid, but prior nested claim does not transfer.')
    bootstrap_result={}; ci_rows=[]
    for c in ['E2','E3']:
        arr=cohort_arr[c]; rows=bycohort[c]
        modes=[('genome_iid', [f'genome_{i}' for i in range(len(rows))]),
               ('erd_or_singleton', [r['erd'] if present(r['erd']) else f'unresolved_{i}' for i,r in enumerate(rows)])]
        if c=='E2': modes.append(('bioproject_source_mix',[r['bioproject'] for r in rows]))
        if c=='E3' and all(present(r['bioproject']) for r in rows):
            modes.append(('bioproject_source_mix',[r['bioproject'] for r in rows]))
        for j,(mode, groups) in enumerate(modes):
            b=bootstrap(arr, groups,args.bootstrap,args.seed+(0 if c=='E2' else 100)+j,mode)
            b['n_missing_erd']=sum(not present(x['erd']) for x in rows)
            b['warning']='E3 genomic group completeness is inadequate' if c=='E3' and mode=='erd_or_singleton' and b['n_missing_erd']>len(rows)*.5 else ''
            bootstrap_result[c+'_'+mode]=b
            for metric, ci in b['delta_rule_minus_ml'].items():
                ci_rows.append({'cohort':c,'resampling':mode,'metric':'delta_rule_minus_ml_'+metric,
                                'point':evidence[c]['delta_rule_minus_ml_'+metric],**ci,'n_clusters':b['n_clusters'],
                                'n_missing_erd':b['n_missing_erd'],'caveat':b['warning']})
    write_csv(outdir/'file16i_paired_bootstrap_contrasts.csv',ci_rows,
              ['cohort','resampling','metric','point','lower','upper','median','valid_replicates','n_clusters','n_missing_erd','caveat'])
    per_source=[]
    groups=defaultdict(list)
    for row in bycohort['E2']:groups[row['bioproject']].append(row)
    for source_name in sorted(groups):
        rr=groups[source_name]
        result=stats(array_data(rr)); result={'bioproject_acc':source_name,'source_selected_posthoc':source_name=='PRJNA1322038',**result}
        per_source.append(result)
    write_csv(outdir/'file16i_E2_paired_by_bioproject.csv',per_source,list(per_source[0]))
    focus=[]
    for name,rows in [('E2_ALL',bycohort['E2']),('E2_FOCUS_POSTHOC',[r for r in bycohort['E2'] if r['bioproject']=='PRJNA1322038']),
                      ('E2_OTHER_SOURCES_POSTHOC',[r for r in bycohort['E2'] if r['bioproject']!='PRJNA1322038']),('E3_ALL',bycohort['E3'])]:
        focus.append({'stratum':name,'post_hoc_source_split':'POSTHOC' in name,**stats(array_data(rows))})
    write_csv(outdir/'file16i_paired_source_strata.csv',focus,list(focus[0]))
    e3metadata=[]
    for row in bycohort['E3']:
        e3metadata.append({'genome_id':row['genome_id'],
                           'has_assembly_accession':present(row['assembly']),
                           'has_biosample':present(row['biosample']),
                           'has_erd_group':present(row['erd']),
                           'has_bioproject':present(row['bioproject']),
                           'erd_group':row['erd'] if present(row['erd']) else '',
                           'bioproject_acc':row['bioproject'] if present(row['bioproject']) else '',
                           'metadata_note':'Do not infer BioProject from source/phenotype'})
    write_csv(outdir/'file16i_E3_metadata_completeness.csv',e3metadata,list(e3metadata[0]))
    overlap=[]
    for field in ['genome_id','assembly','biosample']:
        a={id_normalized(row[field]) for row in bycohort['E2'] if present(row.get(field,''))}
        b={id_normalized(row[field]) for row in bycohort['E3'] if present(row.get(field,''))}
        ov=a&b
        overlap.append({'field':field,'id_set':'full','E2_available':len(a),'E3_available':len(b),'overlap_n':len(ov),
                        'example_overlap':';'.join(sorted(ov)[:5]),'interpretation':'Identifier equality only; NOT nucleotide relatedness'})
        if field=='assembly':
            aa={acc_no_version(row[field]) for row in bycohort['E2'] if present(row.get(field,''))}
            bb={acc_no_version(row[field]) for row in bycohort['E3'] if present(row.get(field,''))}
            overlap.append({'field':field,'id_set':'version_removed','E2_available':len(aa),'E3_available':len(bb),
                            'overlap_n':len(aa&bb),'example_overlap':';'.join(sorted(aa&bb)[:5]),
                            'interpretation':'Candidate assembly aliases only; must audit collisions'})
    write_csv(outdir/'file16i_E2_E3_identifier_checks.csv',overlap,list(overlap[0]))
    citation_gaps=[]
    if not args.e3_frozen: citation_gaps.append('E3 final frozen manifest not provided: assembly/BioSample metadata not audited here.')
    if not args.e3_bioproject_map: citation_gaps.append('E3 BioProject metadata not provided; E3 source clustering not possible.')
    citation_gaps.extend(['E2 abnormal source phenotype cannot be laboratory-verified without AST/MIC records.',
                          'E2 and E3 use different frozen ML model artifacts.',
                          'Genomic sequence independence not established by identifier checks.',
                          'Detected gene determinants do not establish phenotype/causality.',
                          'E2 unusual source found post hoc; leave-one-source-out is not primary performance.'])
    ledger={
      'ready_to_report':[
         {'claim':'E2 and E3 paired rule vs ML sensitivity/specificity trade-offs, fixed FILE11E rule','level':'exploratory applied comparison','basis':'FILE16H + FILE16I paired table'},
         {'claim':'E2 performance strongly depends on source mixture','level':'post-hoc source sensitivity','basis':'FILE16D/G and FILE16I BioProject audit'},
         {'claim':'ML/rule disagreements warrant research AST review; not treatment decisions','level':'research-only operational hypothesis','basis':'paired discordance table'}],
      'not_supported':[
        'New validated classifier, improved algorithm, or clinical safety guarantee',
        'Independent confirmation across identical ML models (E2/E3 use different artifacts)',
        'Phenotype label corruption or causal gene mechanism',
        'E3 source-level transportability without BioProject metadata',
        'New web server is algorithmic novelty or clinically approved software',
        'Replacing frozen E2 AUROC with post-hoc source-excluded AUROC'],
      'must_be_resolved_if_claiming':citation_gaps,
      'primary_endpoint_proposal':'Frozen E2/E3 false-susceptibility vs phenotypic AST with sensitivity, specificity, and paired contrasts; E2 source-mix sensitivity analysis; no patient-level outcome claim',
      'interpretation':'Exploratory post-hoc original research, not prospective independent or clinical validation'}
    write_json(outdir/'file16i_publication_claim_ledger.json',ledger)
    result={'status':'FILE16I_POSTHOC_EVIDENCE_LOCK_NOT_CLINICAL',
            'run_utc':datetime.now(timezone.utc).isoformat(),
            'no_training':True,'no_threshold_tuning':True,'no_frozen_model_modified':True,
            'file16h_E2_parity_confirmed':True,'tsv_hashes_already_confirmed_in_FILE16H':831,
            'input_sha256':fingerprints,'cohorts':evidence,'n_E2_bioprojects':len(groups),
            'E3_metadata':e3frozen_meta,
            'E3_ERD_group_known':sum(present(r['erd']) for r in bycohort['E3']),
            'bootstrap_derived_CIs':bootstrap_result,
            'identifier_overlap':overlap,
            'evidence_gaps':citation_gaps,
            'publication_status':'Research-only post-hoc; NOT validated clinical tool'}
    write_json(outdir/'file16i_summary.json',result)
    readme=(f'# FILE16I Evidence Lock — {_utc_date()}\n\n'
      'This is a post-hoc applied audit, not a prospective study, causal gene finding, or a new classifier.\n\n'
      f'E2 full: N={evidence["E2"]["n"]}, rule minus ML sensitivity = {evidence["E2"]["delta_rule_minus_ml_sensitivity"]:+.4f}; '
      f'specificity = {evidence["E2"]["delta_rule_minus_ml_specificity"]:+.4f}.\n\n'
      f'E3 full: N={evidence["E3"]["n"]}, rule minus ML sensitivity = {evidence["E3"]["delta_rule_minus_ml_sensitivity"]:+.4f}; '
      f'specificity = {evidence["E3"]["delta_rule_minus_ml_specificity"]:+.4f}.\n\n'
      'Bootstrap intervals are within-cohort descriptive sensitivity analyses. E2 BioProject resampling varies SOURCE MIX and cannot establish a clinical population CI. '
      'E3 ERD grouping mostly unresolved; an individual-level bootstrap is not a robust replacement for a source-clustered external validation.\n\n'
      'For the manuscript: separate pre-existing original analyses from the FILE16 series of explicitly post-hoc exploratory analyses. '
      'Report frozen full E2 as primary external evidence; do not present removal of PRJNA1322038 as a corrected score. '
      'Original AST and genomic relatedness remain not verified. See file16i_publication_claim_ledger.json for forbidden assertions.\n')
    (outdir/'file16i_manuscript_guardrails.md').write_text(readme,encoding='utf-8')
    manifest={'outputs':{p.name:sha(p) for p in sorted(outdir.iterdir()) if p.is_file()},'input_unchanged':True}
    write_json(outdir/'file16i_outputs_sha256.json',manifest)
    print('[FILE16I] COMPLETE:',outdir)
    print(json.dumps({'E2_delta_sensitivity_rule_minus_ml':evidence['E2']['delta_rule_minus_ml_sensitivity'],
                      'E3_delta_sensitivity_rule_minus_ml':evidence['E3']['delta_rule_minus_ml_sensitivity'],
                      'E2_bioprojects':len(groups), 'E3_bioproject_coverage':e3frozen_meta['n_bioproject'],
                      'E3_ERD_known':sum(present(r['erd']) for r in bycohort['E3'])},indent=2))


def _utc_date(): return datetime.now(timezone.utc).strftime('%Y-%m-%d')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[FILE16I] ABORTED: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(2)
