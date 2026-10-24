#!/usr/bin/env python3
"""FILE16 — immutable, post-hoc external reliability audit.

NOT an algorithmic novelty claim. This script never trains/recalibrates a model,
never selects thresholds from outcomes, and never changes FILE13–FILE15 artifacts.

Commands:
  python file16_external_reliability_audit.py inventory --project-root /path/to/project --outdir /path/to/new/output
  python file16_external_reliability_audit.py discover --project-root /path/to/project --outdir /path/to/new/output
  python file16_external_reliability_audit.py audit --e2 /path/E2.csv --e3 /path/E3.csv --outdir /path/to/new/output [options]

The audit expects standardized CSVs with genome_id, y_true (0/1), prob_resistance.
Optional columns: assembly_accession, biosample, erd_group, mlst_group,
mechanism_group, broad_rule (0/1); these can also be merged from annotation CSVs.

All added analyses are exploratory/post-hoc, not confirmatory validation.

FILE16 adds a *replication-oriented, mechanism-stratified failure analysis*:
it looks for resistant->susceptible errors in independently sampled cohorts.
Any novelty belongs to reproducible scientific findings, not to an invented formula.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.stats import fisher_exact

DEFAULT_DEV_R = 1601
DEFAULT_DEV_N = 4227
SEED = 20261020
HIGH_CONFIDENCE = 0.90
MISSING_GROUPS = {"", "na", "nan", "none", "null", "unknown", "unresolved", "-", "not available", "n/a"}
IDENTIFIERS = ("genome_id", "assembly_accession", "biosample")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_new_dir(path: Path) -> Path:
    if path.exists():
        raise ValueError(f"Output path ALREADY EXISTS, refusing to overwrite: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=to_builtin) + '\n', encoding='utf-8')


def to_builtin(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, Path): return str(obj)
    raise TypeError(type(obj).__name__)


def inventory(args) -> None:
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Project root does not exist on THIS machine: {root}")
    out = ensure_new_dir(Path(args.outdir).expanduser().resolve())
    # Traverse only source-like trees by default; add extra paths as needed.
    search_roots = [root] + [root / p for p in ('scripts', 'notebooks', 'results', 'data/external_validation', 'checkpoints') if (root/p).exists()]
    candidates = {}
    skipped = {'__pycache__','.git','.venv','venv','node_modules','.ipynb_checkpoints','raw','assemblies','reads','fastq','fasta'}
    for parent in search_roots:
        if not parent.is_dir(): continue
        for cur, ds, fs in os.walk(parent):
            ds[:] = [d for d in ds if d not in skipped and not d.startswith('.') and not d.lower().startswith('file16')]
            for fn in fs:
                low = fn.lower()
                if re.search(r'(?:^|[_\-])file[_\-]?(?:0?[1-9]|1[0-5])(?:[_\-.]|$)', low) or (re.match(r'^(?:0?[1-9]|1[0-5])[_\-]',low) and low.endswith(('.py','.ipynb'))):
                    p = (Path(cur)/fn).resolve()
                    candidates[str(p)] = p
    rows = []
    for p in sorted(candidates.values(),key=lambda x:str(x).lower()):
        rel = str(p.relative_to(root)) if root in p.parents else str(p)
        match = re.search(r'file[_\-]?(0?[1-9]|1[0-5])(?:[_\-.]|$)',p.name,re.I)
        if match is None:
            match = re.match(r'^(0?[1-9]|1[0-5])[_\-]',p.name,re.I)
        number = int(match.group(1)) if match else None
        rows.append({'file_no':number,'relative_path':rel,'absolute_path':str(p),'size_bytes':p.stat().st_size,
                     'sha256':sha256(p) if p.stat().st_size <= args.hash_max_mb*1024**2 else 'SKIPPED_SIZE_LIMIT'})
    pd.DataFrame(rows,columns=['file_no','relative_path','absolute_path','size_bytes','sha256']).to_csv(out/'file01_to_file15_inventory.csv',index=False)
    by_number = {x['file_no'] for x in rows if x['file_no'] is not None}
    report = {'project_root':str(root), 'files_detected':len(rows), 'file_numbers_detected': sorted(by_number),
              'file_numbers_not_detected':sorted(set(range(1,16))-by_number),
              'meaning':'File numbers not detected by naming search are NOT proof they do not exist.',
              'run_utc':datetime.now(timezone.utc).isoformat()}
    save_json(out/'inventory_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Inventory:',out/'file01_to_file15_inventory.csv')



def discover(args) -> None:
    """Find candidates but NEVER auto-select an input from a filename alone."""
    root=Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Project root does not exist on this machine: {root}')
    out=ensure_new_dir(Path(args.outdir).expanduser().resolve())
    hit=[]
    # Focus on generated artifacts, NOT expensive input assemblies / sequence banks.
    skip={'__pycache__','.git','.venv','venv','node_modules','.ipynb_checkpoints',
          'raw','assemblies','reads','fastq','fasta','unitigs','sequence_variation'}
    supported={'.csv','.tsv','.parquet','.json','.joblib','.pkl','.pickle'}
    for cur,ds,fs in os.walk(root):
        ds[:]=[d for d in ds if d not in skip and not d.startswith('.') and not d.lower().startswith('file16')]
        for fn in fs:
            p=Path(cur)/fn
            if p.suffix.lower() not in supported:continue
            rel=str(p.relative_to(root))
            low=rel.lower()
            label=''
            if 'e2' in low or 'external' in low: label='E2 / external (verify)'
            if 'e3' in low or 'file14' in low: label='E3 / FILE14 (verify)'
            if 'file15' in low: label='FILE15 reporting (verify)'
            if any(t in low for t in ('prediction','predictions','oof','probability','proba')): label=(label+'; predicted-probability candidate').strip('; ')
            if any(t in low for t in ('manifest','cohort','freeze','schema','model')): label=(label+'; provenance candidate').strip('; ')
            if not label:continue
            header=''
            if p.suffix.lower() in ('.csv','.tsv'):
                try:
                    sep='\t' if p.suffix.lower()=='.tsv' else ','
                    header=' | '.join(pd.read_csv(p,nrows=0,sep=sep).columns[:35].tolist())
                except Exception as e:
                    header=f'HEADER_READ_ERROR: {type(e).__name__}'
            size=p.stat().st_size
            hit.append({'role_hint':label,'relative_path':rel,'absolute_path':str(p.resolve()),
                        'size_bytes':size,'header_first_35_fields':header})
    cols=['role_hint','relative_path','absolute_path','size_bytes','header_first_35_fields']
    pd.DataFrame(hit,columns=cols).to_csv(out/'file16_artifact_candidates.csv',index=False)
    report={'project_root':str(root),'candidate_count':len(hit),
            'warning':'Candidates not automatically selected: verify frozen E2/E3, schema, accession IDs, class labels and columns manually.',
            'file':str(out/'file16_artifact_candidates.csv')}
    save_json(out/'discovery_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

def read_predictions(path: Path, name: str, id_col, label_col, prob_col) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(path)
    d = pd.read_csv(path, low_memory=False, dtype={id_col:'string'})
    mandatory = [id_col,label_col,prob_col]
    missing = [x for x in mandatory if x not in d]
    if missing: raise ValueError(f'{name} missing columns {missing}. Available: {list(d.columns)}')
    ren = {id_col:'genome_id',label_col:'y_true',prob_col:'prob_resistance'}
    d = d.rename(columns={k:v for k,v in ren.items() if k!=v})
    if not d['genome_id'].notna().all(): raise ValueError(f'{name}: missing genome IDs')
    d['genome_id'] = d['genome_id'].astype('string').str.strip()
    if (d['genome_id']=='').any() or d['genome_id'].duplicated().any():
        raise ValueError(f'{name}: blank/duplicate genome IDs; resolve without using outcomes')
    d['y_true'] = pd.to_numeric(d['y_true'],errors='coerce')
    d['prob_resistance'] = pd.to_numeric(d['prob_resistance'],errors='coerce')
    if not d['y_true'].isin([0,1]).all(): raise ValueError(f'{name}: y_true must be only 0/1')
    if not d['prob_resistance'].between(0,1).all(): raise ValueError(f'{name}: probability must be within [0,1]')
    d['y_true'] = d['y_true'].astype(int)
    for c in ('assembly_accession','biosample','erd_group','mlst_group','mechanism_group'):
        if c in d.columns: d[c] = d[c].astype('string').str.strip()
    if 'broad_rule' in d.columns:
        d['broad_rule'] = pd.to_numeric(d['broad_rule'],errors='coerce')
        if not d['broad_rule'].isin([0,1]).all(): raise ValueError(f'{name} broad_rule must be 0/1')
        d['broad_rule'] = d['broad_rule'].astype(int)
    return d


def merge_annotations(d: pd.DataFrame, path: Path | None, name: str) -> pd.DataFrame:
    if path is None: return d
    a = pd.read_csv(path,dtype={'genome_id':'string'},low_memory=False)
    if 'genome_id' not in a: raise ValueError(f'{name} annotations must contain genome_id')
    a['genome_id'] = a['genome_id'].astype('string').str.strip()
    if a['genome_id'].duplicated().any(): raise ValueError(f'{name} annotation duplicate genome_id')
    cols = [c for c in ('genome_id','assembly_accession','biosample','erd_group','mlst_group','mechanism_group','broad_rule') if c in a]
    overlapping = set(cols).intersection(set(d.columns)) - {'genome_id'}
    if overlapping: raise ValueError(f'{name} overlapping annotation/prediction columns: {overlapping}; reconcile explicitly')
    d = d.merge(a[cols],on='genome_id',how='left',validate='one_to_one',indicator=True)
    if (d['_merge']!='both').any(): raise ValueError(f'{name}: annotations missing {(d["_merge"]!="both").sum()} genomes')
    d = d.drop(columns='_merge')
    if 'broad_rule' in d:
        d['broad_rule'] = pd.to_numeric(d['broad_rule'],errors='coerce')
        if not d['broad_rule'].isin([0,1]).all(): raise ValueError(f'{name}: broad_rule must be 0/1')
        d['broad_rule'] = d['broad_rule'].astype(int)
    return d


def normids(d: pd.DataFrame, col: str) -> set:
    if col not in d: return set()
    s=d[col].astype('string').str.strip().str.lower()
    return {x for x in s.dropna().tolist() if x not in MISSING_GROUPS}


def overlap_report(datasets: dict[str,pd.DataFrame]) -> list:
    reports=[]
    names=list(datasets)
    for a in range(len(names)):
        for b in range(a+1,len(names)):
            n1,n2=names[a],names[b]
            for col in IDENTIFIERS:
                x,y=normids(datasets[n1],col),normids(datasets[n2],col)
                reports.append({'cohort_a':n1,'cohort_b':n2,'field':col,'available_a':len(x),'available_b':len(y),
                                'overlap_n':len(x&y),'example_overlaps':' | '.join(sorted(x&y)[:5])})
    return reports


def binary_stats(y: np.ndarray, scores: np.ndarray, thr: float=.5) -> dict:
    p=np.asarray(scores,dtype=float)
    y=np.asarray(y,dtype=int)
    pred=(p>=thr).astype(int)
    tp=int(np.sum((y==1)&(pred==1))); fn=int(np.sum((y==1)&(pred==0)))
    tn=int(np.sum((y==0)&(pred==0))); fp=int(np.sum((y==0)&(pred==1)))
    sens=tp/(tp+fn) if tp+fn else None
    spec=tn/(tn+fp) if tn+fp else None
    return dict(n=len(y),n_resistant=int(y.sum()),resistant_prevalence=float(y.mean()),
        auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
        brier=float(np.mean((p-y)**2)), sensitivity=sens,specificity=spec,
        balanced_accuracy=(sens+spec)/2 if sens is not None and spec is not None else None,
        mcc=float(matthews_corrcoef(y,pred)) if len(np.unique(y))==2 else None,
        tp=tp,fn=fn,tn=tn,fp=fp,
        high_conf_wrong=int(np.sum(((y==1)&(p<=1-HIGH_CONFIDENCE))|((y==0)&(p>=HIGH_CONFIDENCE)))),
        high_conf_false_susceptible=int(np.sum((y==1)&(p<=1-HIGH_CONFIDENCE))),
        false_susceptible_risk=fn/(tp+fn) if tp+fn else None)


def get_groups(d: pd.DataFrame, col: str | None) -> np.ndarray:
    # If no genomic group metadata is available, IID resampling is only a sensitivity proxy.
    if col is None or col not in d: return np.array([f'IID_{i}' for i in range(len(d))],dtype=object)
    groups=[]
    for i,x in enumerate(d[col].astype('string')):
        val='' if pd.isna(x) else str(x).strip()
        groups.append(f'SINGLETON_{i}' if val.lower() in MISSING_GROUPS else 'GROUP_'+val)
    return np.array(groups,dtype=object)


def bootstrap_contrasts(d: pd.DataFrame, null_p: float, group_col: str | None, iters: int, seed: int) -> dict:
    y=d['y_true'].to_numpy(dtype=int); p=d['prob_resistance'].to_numpy(dtype=float)
    g=get_groups(d,group_col)
    ix=defaultdict(list)
    for i,k in enumerate(g): ix[k].append(i)
    blocks=[np.asarray(x,dtype=int) for x in ix.values()]
    rng=np.random.default_rng(seed)
    arrays={'delta_brier_ml_minus_null':[], 'ml_false_susceptible_risk':[], 'ml_high_conf_false_susceptible_count_per_resistant':[]}
    has_rule='broad_rule' in d
    if has_rule:
        arrays.update({'delta_sensitivity_ml_minus_rule':[], 'delta_balanced_accuracy_ml_minus_rule':[],
                       'delta_brier_ml_minus_rule':[]})
        rule=d['broad_rule'].to_numpy(dtype=float)
    for _ in range(iters):
        chosen=rng.integers(0,len(blocks),size=len(blocks))
        sample=np.concatenate([blocks[i] for i in chosen])
        yy=y[sample]; pp=p[sample]
        if yy.min()==yy.max(): continue
        m=binary_stats(yy,pp)
        arrays['delta_brier_ml_minus_null'].append(float(np.mean((pp-yy)**2)-np.mean((null_p-yy)**2)))
        arrays['ml_false_susceptible_risk'].append(m['false_susceptible_risk'])
        arrays['ml_high_conf_false_susceptible_count_per_resistant'].append(m['high_conf_false_susceptible']/sum(yy))
        if has_rule:
            rr=binary_stats(yy,rule[sample]);
            arrays['delta_sensitivity_ml_minus_rule'].append(m['sensitivity']-rr['sensitivity'])
            arrays['delta_balanced_accuracy_ml_minus_rule'].append(m['balanced_accuracy']-rr['balanced_accuracy'])
            arrays['delta_brier_ml_minus_rule'].append(m['brier']-rr['brier'])
    out={}
    for k,arr in arrays.items():
        valid=np.asarray([x for x in arr if x is not None and np.isfinite(x)],dtype=float)
        out[k]={'valid_replicates':len(valid),'mean':float(valid.mean()) if len(valid) else None,
                'ci95_low':float(np.quantile(valid,.025)) if len(valid) else None,
                'ci95_high':float(np.quantile(valid,.975)) if len(valid) else None}
    out['_grouping']={'column':group_col or 'IID_ONLY','n_resampling_blocks':len(blocks),
                     'n_singleton_blocks':sum(len(b)==1 for b in blocks),
                     'warning':'Unresolved genomic groups treated as singleton; possible overly narrow uncertainty.' if group_col else 'IID-only proxy: not a genomic-cluster confidence interval.'}
    return out


def wilson(n_pos: int,n_tot: int,z: float=1.95996398454):
    if n_tot==0:return (None,None)
    p=n_pos/n_tot;den=1+z*z/n_tot
    mid=(p+z*z/(2*n_tot))/den
    half=z*math.sqrt(p*(1-p)/n_tot+z*z/(4*n_tot*n_tot))/den
    return float(mid-half),float(mid+half)


def mechanistic_failure_table(d: pd.DataFrame,cohort: str,min_resistant: int) -> list:
    if 'mechanism_group' not in d:return []
    rows=[]
    for group, dd in d.groupby('mechanism_group',dropna=False):
        y=dd['y_true'].to_numpy(dtype=int);p=dd['prob_resistance'].to_numpy(dtype=float)
        nr=int(y.sum());fn=int(np.sum((y==1)&(p<.5)))
        hc=int(np.sum((y==1)&(p<=.1)))
        lo,hi=wilson(fn,nr)
        rows.append({'cohort':cohort,'mechanism_group':str(group),'n':len(dd),'n_resistant':nr,
                     'false_susceptible_n':fn,'false_susceptible_rate':fn/nr if nr else None,
                     'false_susceptible_wilson95_low':lo,'false_susceptible_wilson95_high':hi,
                     'high_conf_false_susceptible_n':hc,'brier_within_mechanism':float(np.mean((p-y)**2)),
                     'predicted_resistant_probability_mean':float(p.mean()),
                     'observed_resistant_prevalence':float(y.mean()),
                     'exploratory':True,'low_n_warning': nr<min_resistant})
    return rows



def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Deterministic BH-FDR for a family of exploratory tests."""
    n=len(pvals)
    if not n: return []
    order=np.argsort(pvals,kind='stable')
    corrected=[1.0]*n
    best=1.0
    for rank in range(n,0,-1):
        j=int(order[rank-1])
        best=min(best, float(pvals[j])*n/rank)
        corrected[j]=min(1.0,best)
    return corrected


def cross_cohort_mechanism_signatures(d2: pd.DataFrame,d3: pd.DataFrame,min_resistant:int) -> list:
    """Resistant-only false-susceptible enrichment vs other mechanism groups.

    A same-sign finding is *suggestive replication* only. The groups may be
    chosen post-hoc and individual isolates may be related or source-confounded.
    A group must appear as an exclusive single category in both cohorts.
    """
    if not ('mechanism_group' in d2 and 'mechanism_group' in d3): return []
    ds={'E2':d2,'E3':d3}
    by_cohort={}
    all_names=set()
    for cohort,d in ds.items():
        resistant=d.loc[d.y_true.eq(1)].copy()
        resistant['is_fn']=resistant.prob_resistance.lt(.5)
        resistant['mechanism_group']=resistant.mechanism_group.astype('string').fillna('UNASSIGNED')
        rows=[]
        for g,dd in resistant.groupby('mechanism_group',dropna=False):
            nr=len(dd)
            if nr<min_resistant: continue
            rest=resistant.loc[resistant.mechanism_group.ne(g)]
            if len(rest)<min_resistant:continue
            fn=int(dd.is_fn.sum());fnr=int(rest.is_fn.sum())
            risk_group=fn/nr
            risk_other=fnr/len(rest)
            p=float(fisher_exact([[fn,nr-fn],[fnr,len(rest)-fnr]],alternative='two-sided').pvalue)
            rows.append({'cohort':cohort,'mechanism_group':str(g),
                         'resistant_in_group':nr,'false_susceptible_in_group':fn,
                         'resistant_outside_group':len(rest),'false_susceptible_outside_group':fnr,
                         'group_fn_rate':risk_group,'others_fn_rate':risk_other,
                         'risk_difference':risk_group-risk_other,'fisher_p_uncorrected':p})
        adj=benjamini_hochberg([r['fisher_p_uncorrected'] for r in rows])
        for row,q in zip(rows,adj):
            row['bh_q_within_cohort']=q
            all_names.add(row['mechanism_group'])
        by_cohort[cohort]={r['mechanism_group']:r for r in rows}
    output=[]
    for group in sorted(all_names):
        a=by_cohort['E2'].get(group)
        b=by_cohort['E3'].get(group)
        if a and b:
            # Same direction in two independent cohorts: weak screening signal,
            # not a proof of validated biological causality or significance.
            samedir=(a['risk_difference']>0 and b['risk_difference']>0) or (a['risk_difference']<0 and b['risk_difference']<0)
            status='SAME_DIRECTION_EXPLORATORY' if samedir else 'INCONSISTENT_DIRECTION'
            # A numerical lower-envelope effect is more conservative than the
            # strongest observed cohort effect (still descriptive, not an estimand).
            envelope=min(abs(a['risk_difference']),abs(b['risk_difference'])) if samedir else None
        else:
            samedir=False;status='INSUFFICIENT_RESISTANT_COUNT_IN_ONE_COHORT';envelope=None
        output.append({'mechanism_group':group,'status':status,
                       'same_sign_across_cohorts':samedir,
                       'conservative_absolute_risk_difference_envelope':envelope,
                       'e2_n_resistant':a['resistant_in_group'] if a else None,
                       'e2_group_fn_rate':a['group_fn_rate'] if a else None,
                       'e2_others_fn_rate':a['others_fn_rate'] if a else None,
                       'e2_risk_difference':a['risk_difference'] if a else None,
                       'e2_fisher_p':a['fisher_p_uncorrected'] if a else None,
                       'e2_bh_q':a['bh_q_within_cohort'] if a else None,
                       'e3_n_resistant':b['resistant_in_group'] if b else None,
                       'e3_group_fn_rate':b['group_fn_rate'] if b else None,
                       'e3_others_fn_rate':b['others_fn_rate'] if b else None,
                       'e3_risk_difference':b['risk_difference'] if b else None,
                       'e3_fisher_p':b['fisher_p_uncorrected'] if b else None,
                       'e3_bh_q':b['bh_q_within_cohort'] if b else None,
                       'warning':'All strata/outcomes exploratory. Same sign is NOT confirmatory replication or proof of a causal mechanism.'})
    return output

def selective_safety(d: pd.DataFrame,name: str) -> list:
    # Diagnostic of confident susceptible calls, never used to choose a deployed threshold.
    rows=[]
    y=d['y_true'].to_numpy(dtype=int);p=d['prob_resistance'].to_numpy(dtype=float)
    for conf in (.7,.8,.9,.95):
        selected=p<=1-conf
        n=int(selected.sum()); positives=int(y[selected].sum())
        lo,hi=wilson(positives,n)
        rows.append({'cohort':name,'predicted_susceptible_confidence_at_least':conf,
                     'n_predicted_susceptible':n,'n_truly_resistant_among_them':positives,
                     'observed_false_susceptible_fraction':positives/n if n else None,
                     'wilson95_low':lo,'wilson95_high':hi,'all_genomes_coverage':n/len(d),
                     'exploratory':True})
    return rows


def audit(args):
    paths={'e2_predictions':Path(args.e2).expanduser().resolve(),
           'e3_predictions':Path(args.e3).expanduser().resolve()}
    if args.e2_annotations: paths['e2_annotations']=Path(args.e2_annotations).expanduser().resolve()
    if args.e3_annotations: paths['e3_annotations']=Path(args.e3_annotations).expanduser().resolve()
    if args.dev_manifest: paths['dev_manifest']=Path(args.dev_manifest).expanduser().resolve()
    if args.rule_spec: paths['rule_spec']=Path(args.rule_spec).expanduser().resolve()
    if args.e2_model: paths['e2_model']=Path(args.e2_model).expanduser().resolve()
    if args.e3_model: paths['e3_model']=Path(args.e3_model).expanduser().resolve()
    for k,path in paths.items():
        if not path.is_file():raise FileNotFoundError(f'{k}: {path}')
    outpath=Path(args.outdir).expanduser().resolve()
    if outpath.exists():raise ValueError(f'Refusing to overwrite prior analysis: {outpath}')
    d2=merge_annotations(read_predictions(paths['e2_predictions'],'E2',args.id_col,args.label_col,args.prob_col),paths.get('e2_annotations'),'E2')
    d3=merge_annotations(read_predictions(paths['e3_predictions'],'E3',args.id_col,args.label_col,args.prob_col),paths.get('e3_annotations'),'E3')
    if len(d2)!=525 or len(d3)!=306:
        raise ValueError(f'Expected frozen E2=525 and E3=306, got {len(d2)} and {len(d3)}. Not the frozen cohorts?')
    if int(d2.y_true.sum())!=163 or int(d3.y_true.sum())!=122:
        raise ValueError('Phenotype counts do not match locked E2=163 R and E3=122 R. Aborting.')
    # Known primary E2 / E3 model discrimination sanity checks; not exact authentication.
    auc2=roc_auc_score(d2.y_true,d2.prob_resistance)
    auc3=roc_auc_score(d3.y_true,d3.prob_resistance)
    if abs(auc2-0.667)>.012 or abs(auc3-0.778)>.012:
        raise ValueError(f'Frozen AUROC sanity checks differ from paper: {auc2:.4f} and {auc3:.4f}. Aborting.')
    # Check the other published locked metrics to prevent mixing models/cohorts.
    check2 = binary_stats(d2.y_true.to_numpy(), d2.prob_resistance.to_numpy())
    check3 = binary_stats(d3.y_true.to_numpy(), d3.prob_resistance.to_numpy())
    expected_e2 = {'tp':109,'fn':54,'tn':221,'fp':141,'high_conf_wrong':128}
    bad2 = {k:(check2[k],v) for k,v in expected_e2.items() if check2[k] != v}
    if bad2 or abs(check2['average_precision']-.411)>.012 or abs(check2['brier']-.316)>.015:
        raise ValueError(f'E2 does not reproduce the published frozen prediction results (confusion/high-confidence errors/AP/Brier): {bad2}; metrics={check2}')
    if (check3['fn']+check3['fp'] != 82 or check3['high_conf_wrong'] != 51
        or abs(check3['average_precision']-.664)>.012 or abs(check3['brier']-.222)>.015):
        raise ValueError(f'E3 does not reproduce published 702-feature primary results (82 errors, 51 high confidence wrong, AP, Brier): {check3}')
    datasets={'E2':d2,'E3':d3}
    if args.dev_manifest:
        dv=pd.read_csv(paths['dev_manifest'],low_memory=False,dtype='string')
        if args.id_col in dv and args.id_col!='genome_id': dv=dv.rename(columns={args.id_col:'genome_id'})
        datasets['Development']=dv
    overlaps=overlap_report(datasets)
    if any(x['overlap_n']>0 for x in overlaps):
        raise ValueError('Identifier overlap detected! See stdout: '+json.dumps([x for x in overlaps if x['overlap_n']>0],ensure_ascii=False))
    any_rule=any('broad_rule' in d for d in (d2,d3))
    if any_rule and not ('broad_rule' in d2 and 'broad_rule' in d3):
        raise ValueError('Need the SAME fixed broad_rule for BOTH E2 and E3, or omit from both.')
    if any_rule and not args.rule_spec:
        raise ValueError('Rule analysis requires --rule-spec written from frozen E2 rule definition.')
    if any_rule:
        # E2 comparators should reproduce published sensitivity/specificity before E3 application.
        r2=binary_stats(d2.y_true.to_numpy(),d2.broad_rule.to_numpy())
        if abs(r2['sensitivity']-.847)>.015 or abs(r2['specificity']-.530)>.015:
            raise ValueError('Provided broad_rule does not reproduce E2 published reference: sensitivity 0.847, specificity 0.530. STOP.')
    null_p=args.dev_resistant/args.dev_total
    if not 0<null_p<1:raise ValueError('Development prevalence must be (0,1)')
    results={}
    for name, d in (('E2',d2),('E3',d3)):
        st=binary_stats(d.y_true.to_numpy(),d.prob_resistance.to_numpy())
        y=d.y_true.to_numpy(dtype=int)
        st['brier_null_dev_prevalence']=float(np.mean((null_p-y)**2))
        st['brier_skill_vs_dev_prevalence_null']=1-st['brier']/st['brier_null_dev_prevalence']
        if 'broad_rule' in d: st['fixed_broad_rule']=binary_stats(y,d.broad_rule.to_numpy())
        grouping='erd_group' if 'erd_group' in d else None
        boot={'erd_or_iid':bootstrap_contrasts(d,null_p,grouping,args.bootstrap,args.seed+(0 if name=='E2' else 100))}
        if 'mlst_group' in d:
            boot['mlst_group_sensitivity']=bootstrap_contrasts(d,null_p,'mlst_group',args.bootstrap,args.seed+500+(0 if name=='E2' else 100))
        results[name]={'metrics':st,'bootstrap':boot}
    run_log={'run_utc':datetime.now(timezone.utc).isoformat(),'analysis_type':'post-hoc/exploratory',
             'no_training':True,'no_recalibration':True,'no_threshold_optimization':True,
             'inputs':[{'role':k,'path':str(v),'sha256':sha256(v)} for k,v in paths.items()],
             'cohort_overlap_checks':overlaps,
             'critical_caveats':['Identifier matching is not equivalent to genomic relatedness/sequence-distance validation.',
                                 'E2 primary model has 668 features; E3 primary has 702, so performance differences confound cohort and model configuration unless matching model artifacts are verified.',
                                 'E3 ERD groups are mostly unresolved; singleton-based CI may be too narrow.',
                                 'Post-hoc analyses cannot be relabeled as prespecified primary hypotheses.']}
    if args.e2_model and args.e3_model:
        run_log['identical_model_sha256']=sha256(paths['e2_model'])==sha256(paths['e3_model'])
    ensure_new_dir(outpath)
    save_json(outpath/'file16_input_provenance.json',run_log)
    save_json(outpath/'file16_summary_metrics_and_bootstrap.json',results)
    pd.DataFrame(overlaps).to_csv(outpath/'file16_identifier_overlaps.csv',index=False)
    mech=[];coverage=[]
    for name,d in (('E2',d2),('E3',d3)):
        mech += mechanistic_failure_table(d,name,args.min_resistant)
        coverage += selective_safety(d,name)
    pd.DataFrame(mech,columns=['cohort','mechanism_group','n','n_resistant','false_susceptible_n','false_susceptible_rate','false_susceptible_wilson95_low','false_susceptible_wilson95_high','high_conf_false_susceptible_n','brier_within_mechanism','predicted_resistant_probability_mean',
                                 'observed_resistant_prevalence','exploratory','low_n_warning']).to_csv(outpath/'file16_mechanism_failure_map.csv',index=False)
    pd.DataFrame(coverage).to_csv(outpath/'file16_confident_susceptibility_risks.csv',index=False)
    error_rows=[]
    for name,d in (('E2',d2),('E3',d3)):
        u=d[['genome_id','y_true','prob_resistance']].copy()
        u['cohort']=name
        u['predicted_resistant_at_frozen_threshold_05']=(u.prob_resistance>=0.5).astype(int)
        u['error_type']=np.select([
            u.y_true.eq(1)&u.prob_resistance.lt(.5),
            u.y_true.eq(0)&u.prob_resistance.ge(.5)],
            ['FALSE_SUSCEPTIBLE','FALSE_RESISTANT'],default='CORRECT')
        u['high_conf_wrong']=((u.y_true.eq(1)&u.prob_resistance.le(.1))|
                              (u.y_true.eq(0)&u.prob_resistance.ge(.9)))
        for field in ('mechanism_group','mlst_group','erd_group'):
            if field in d:u[field]=d[field].values
        error_rows.append(u)
    pd.concat(error_rows,ignore_index=True).to_csv(outpath/'file16_per_genome_prediction_error_audit.csv',index=False)

    cross=cross_cohort_mechanism_signatures(d2,d3,args.min_resistant)
    cross_cols=['mechanism_group','status','same_sign_across_cohorts','conservative_absolute_risk_difference_envelope',
                'e2_n_resistant','e2_group_fn_rate','e2_others_fn_rate','e2_risk_difference','e2_fisher_p','e2_bh_q',
                'e3_n_resistant','e3_group_fn_rate','e3_others_fn_rate','e3_risk_difference','e3_fisher_p','e3_bh_q','warning']
    pd.DataFrame(cross,columns=cross_cols).to_csv(outpath/'file16_cross_cohort_mechanism_signatures.csv',index=False)
    with (outpath/'file16_caution_and_interpretation.txt').open('w',encoding='utf-8') as f:
        f.write('POST-HOC analysis. NO new classifier, threshold tuning, or recalibration.\n')
        f.write('What may be novel is an observed cross-cohort, mechanism-conditioned failure signature IF replicated, not this analysis formula.\n')
        f.write('DO NOT claim molecular mechanism causality based on this observational association.\n')
        f.write('DO NOT claim E2/E3 are fully sequence-independent merely from identifier checks.\n')
        f.write('DO NOT claim grouped E3 uncertainty is fully resolved by singleton grouping.\n')
        f.write('DO NOT claim two cohorts compare one identical classifier if model hashes differ.\n')
        f.write('Mechanism-signature same-direction flags are descriptive exploratory screening only.\n')
        f.write('Novelty would require a defensible new finding, biological explanation, and external independent verification.\n')
    print('FILE16 FINISHED (exploratory only):',outpath)
    print(json.dumps({x:results[x]['metrics'] for x in ('E2','E3')},indent=2,ensure_ascii=False))


def build_parser():
    ap=argparse.ArgumentParser(description='FILE16: immutable post-hoc external reliability audit')
    sub=ap.add_subparsers(dest='cmd',required=True)
    inv=sub.add_parser('inventory',help='Locate previously saved FILE01–FILE15 paths locally')
    inv.add_argument('--project-root',required=True)
    inv.add_argument('--outdir',required=True,help='NEW output folder, must not exist')
    inv.add_argument('--hash-max-mb',type=float,default=100.,help='Skip SHA256 on very large files')
    dc=sub.add_parser('discover',help='Catalog possible E2/E3 data, model and schema artifacts (never auto-select)')
    dc.add_argument('--project-root',required=True)
    dc.add_argument('--outdir',required=True,help='NEW output folder, must not exist')
    an=sub.add_parser('audit',help='Analyze the existing frozen E2/E3 predictions')
    an.add_argument('--e2',required=True,help='E2 frozen genome-level prediction CSV')
    an.add_argument('--e3',required=True,help='E3 frozen genome-level prediction CSV')
    an.add_argument('--outdir',required=True,help='NEW output folder, must not exist')
    an.add_argument('--e2-annotations',help='Optional E2 annotation CSV keyed genome_id')
    an.add_argument('--e3-annotations',help='Optional E3 annotation CSV keyed genome_id')
    an.add_argument('--dev-manifest',help='Strongly recommended development manifest for identifier overlap audit')
    an.add_argument('--rule-spec',help='Frozen original E2 broad-carbapenemase rule specification file')
    an.add_argument('--e2-model',help='Optional existing E2 frozen model binary for SHA audit')
    an.add_argument('--e3-model',help='Optional existing E3 frozen model binary for SHA audit')
    an.add_argument('--id-col',default='genome_id')
    an.add_argument('--label-col',default='y_true')
    an.add_argument('--prob-col',default='prob_resistance')
    an.add_argument('--dev-resistant',type=int,default=DEFAULT_DEV_R)
    an.add_argument('--dev-total',type=int,default=DEFAULT_DEV_N)
    an.add_argument('--bootstrap',type=int,default=2000)
    an.add_argument('--seed',type=int,default=SEED)
    an.add_argument('--min-resistant',type=int,default=15)
    return ap


def main():
    args=build_parser().parse_args()
    try:
        {'inventory':inventory,'discover':discover,'audit':audit}[args.cmd](args)
    except Exception as exc:
        print('FILE16 STOPPED WITHOUT MODIFYING PRIOR FILES:',str(exc),file=sys.stderr)
        sys.exit(2)
if __name__=='__main__':main()
