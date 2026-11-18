#!/usr/bin/env python3
"""FILE17 — post-hoc, frozen genomic-neighbour sensitivity audit (E2/E3).

No training, no model refit, no modification to frozen prediction vectors, labels,
thresholds, cohorts, or primary metrics. Targeted, nonrandom ANI-confirmed Mash
neighbours are *excluded only in an explicitly labeled descriptive sensitivity
panel*. The remainder is NOT certified genomically independent.

Commands
--------
  python scripts/file17_frozen_neighbour_sensitivity_audit.py discover --project-root .
  python scripts/file17_frozen_neighbour_sensitivity_audit.py audit --project-root .
  python scripts/file17_frozen_neighbour_sensitivity_audit.py self-test

Expected standardized predictions (from FILE16):
  results/file16_audit_mechanisms_v1/file16_per_genome_prediction_error_audit.csv
  cohort,genome_id,y_true,prob_resistance,predicted_resistant_at_frozen_threshold_05

ANI defaults to the native FILE16J file16j_ani_56_pairs.csv (8 columns);
  also accepts the FILE17 standardized 7-column schema. For native FILE16J,
  ANI and aligned fractions are percentage units (0..100). For standardized
  FILE17, aligned fractions are 0..1. External IDs must join with FILE16 IDs.
Outputs are always under project_root/results/ by default.

Python 3.10+, pandas, numpy, scikit-learn.
"""
from __future__ import annotations

import argparse
import tempfile
from argparse import Namespace
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EXPECTED = {'E2': {'n': 525, 'r': 163, 'tp': 109, 'fn': 54, 'tn': 221, 'fp': 141, 'auroc': 0.667, 'ap':0.411, 'brier':0.316, 'high_conf':128},
            'E3': {'n': 306, 'r': 122, 'tp': 67, 'fn': 55, 'tn': 157, 'fp': 27, 'auroc': 0.778, 'ap':0.664, 'brier':0.222, 'high_conf':51}}
DEFAULT_PRED = 'results/file16_audit_mechanisms_v1/file16_per_genome_prediction_error_audit.csv'
DEFAULT_OUT = 'results/file17_frozen_neighbour_sensitivity_v1'
DEFAULT_ANI = 'results/file16j_mash_verified_v2/file16j_ani_56_pairs.csv'
NATIVE_ANI = ('cohort','external_genome','nearest_dev_genome','min_mash_distance',
              'ani_percent','aligned_fraction_dev_percent','aligned_fraction_external_percent','status')
REQUIRED_ANI = ('cohort','external_genome_id','development_genome_id','mash_distance',
                'ani_percent','aligned_fraction_ref','aligned_fraction_query')
PRED_REQ = ('cohort','genome_id','y_true','prob_resistance',
            'predicted_resistant_at_frozen_threshold_05')
VERSION = '1.0.2'


def utc():
    return datetime.now(timezone.utc).isoformat()


def fullpath(root: Path, value: str) -> Path:
    p=Path(value).expanduser()
    return (p if p.is_absolute() else root/p).resolve()


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def need(condition:bool, message:str)->None:
    if not condition:
        raise ValueError(message)


def json_ready(x):
    if isinstance(x, dict): return {str(k):json_ready(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [json_ready(v) for v in x]
    if isinstance(x, (np.integer,)):return int(x)
    if isinstance(x, (np.floating,)): return json_ready(float(x))
    if isinstance(x, (np.bool_,)):return bool(x)
    if isinstance(x, float) and not math.isfinite(x):return None
    if isinstance(x, Path):return str(x)
    return x


def write_json(path:Path, obj)->None:
    path.write_text(json.dumps(json_ready(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def checked_id(series:pd.Series, label:str)->pd.Series:
    need(series.notna().all(),f'{label}: missing IDs')
    s=series.astype(str).str.strip()
    need(s.ne('').all() and (~s.str.lower().isin(['nan','none','null'])).all(),f'{label}: blank IDs')
    return s


def require_header(df:pd.DataFrame, req, label):
    missing=set(req)-set(df.columns)
    need(not missing,f'{label}: missing required columns {sorted(missing)}; got {list(df.columns)}')


def metric_panel(frame:pd.DataFrame)->dict:
    n=len(frame)
    y=frame.y_true.to_numpy(dtype=int)
    p=frame.prob_resistance.to_numpy(dtype=float)
    call=(p>=.5).astype(int)
    nr=int((y==1).sum());ns=n-nr
    tp=int(((y==1)&(call==1)).sum());fn=nr-tp
    tn=int(((y==0)&(call==0)).sum());fp=ns-tn
    auroc=float(roc_auc_score(y,p)) if nr>0 and ns>0 else None
    ap=float(average_precision_score(y,p)) if nr>0 else None
    return {
        'n':n,'n_resistant':nr,'n_susceptible':ns,'tp':tp,'fn':fn,'tn':tn,'fp':fp,
        'auroc':auroc,'average_precision':ap,
        'sensitivity':tp/nr if nr else None,'specificity':tn/ns if ns else None,
        'balanced_accuracy':((tp/nr+tn/ns)/2) if nr and ns else None,
        'mcc':_mcc(tp,tn,fp,fn),
        'brier':float(np.mean((p-y)**2)) if n else None,
        'error_n':fp+fn,
        'high_confidence_wrong_n':int((((y==1)&(p<=.10))|((y==0)&(p>=.90))).sum()),
        'false_susceptible_among_called_susceptible':fn/(fn+tn) if fn+tn else None,
    }


def _mcc(tp,tn,fp,fn):
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d else None


def read_predictions(path:Path, *, testing=False)->pd.DataFrame:
    if not path.is_file():raise FileNotFoundError(f'FILE16 standardized prediction file missing: {path}')
    df=pd.read_csv(path,dtype={'genome_id':'string'},low_memory=False,encoding='utf-8-sig')
    require_header(df,PRED_REQ,'FILE16 frozen predictions')
    need(set(df.cohort)=={'E2','E3'},'FILE16 predictions: expected only cohorts E2 and E3')
    df['genome_id']=checked_id(df.genome_id,'FILE16 genome_id')
    need(not df.duplicated(['cohort','genome_id']).any(), 'FILE16: duplicate cohort+genome_id')
    need(set(df[df.cohort=='E2'].genome_id).isdisjoint(set(df[df.cohort=='E3'].genome_id)),
         'FILE16: E2/E3 genome IDs overlap; STOP')
    for col in ['y_true','prob_resistance','predicted_resistant_at_frozen_threshold_05']:
        df[col]=pd.to_numeric(df[col],errors='raise')
        need(np.isfinite(df[col].to_numpy(dtype=float)).all(),f'FILE16: invalid numeric {col}')
    need(df.y_true.isin([0,1]).all(),'FILE16: phenotype labels must be binary 0/1')
    need(df.prob_resistance.between(0,1).all(),'FILE16: probability outside 0..1')
    need(df.predicted_resistant_at_frozen_threshold_05.isin([0,1]).all(),
         'FILE16: predicted classes must be binary 0/1')
    pred=(df.prob_resistance.to_numpy(dtype=float)>=.5).astype(int)
    need(np.array_equal(pred,df.predicted_resistant_at_frozen_threshold_05.to_numpy(dtype=int)),
         'FILE16 prediction class differs from frozen >=0.5 threshold')
    df['y_true']=df.y_true.astype(int)
    for cohort in ('E2','E3'):
        m=metric_panel(df.loc[df.cohort==cohort])
        if not testing:
            ref=EXPECTED[cohort]
            for k in ('n','n_resistant','tp','fn','tn','fp'):
                target=ref['r'] if k=='n_resistant' else ref[k]
                need(m[k]==target,f'{cohort}: frozen {k} mismatch ({m[k]} != {target}); do not continue')
            for observed,stored in (('auroc','auroc'),('average_precision','ap'),('brier','brier')):
                need(m[observed] is not None and abs(m[observed]-ref[stored])<0.003,
                     f'{cohort}: frozen {observed} mismatch ({m[observed]} vs ~{ref[stored]})')
            need(m['high_confidence_wrong_n']==ref['high_conf'],
                 f'{cohort}: frozen high-confidence wrong count mismatch ({m["high_confidence_wrong_n"]} != {ref["high_conf"]})')
    return df


def read_ani(path:Path,pred:pd.DataFrame, *, testing=False)->tuple[pd.DataFrame,dict]:
    if not path.is_file():raise FileNotFoundError(f'ANI/skani paired evidence file missing: {path}')
    d=pd.read_csv(path,low_memory=False,encoding='utf-8-sig',dtype='string')
    # FILE16J (actual project output) has 8 native columns and alignment
    # fractions in percentage units, unlike FILE17's original template.
    if set(NATIVE_ANI).issubset(d.columns):
        native_statuses=d['status'].fillna('').astype(str).str.strip()
        need(native_statuses.ne('').all(), 'FILE16J ANI: missing verification status; inspect source')
        d=d.rename(columns={
            'external_genome':'external_genome_id',
            'nearest_dev_genome':'development_genome_id',
            'min_mash_distance':'mash_distance',
            'aligned_fraction_dev_percent':'aligned_fraction_ref',
            'aligned_fraction_external_percent':'aligned_fraction_query',
        })
        # Percentages from native FILE16J: divide 0..100 by 100 only here.
        for c in ('aligned_fraction_ref','aligned_fraction_query'):
            raw=pd.to_numeric(d[c],errors='raise')
            need(raw.notna().all() and raw.between(0,100).all(),
                 f'FILE16J {c} must be percentages in 0..100 (e.g. 95.1)')
            d[c]=raw/100.0
        d['ani_evidence_schema']='FILE16J_NATIVE_PERCENTAGES'
    else:
        require_header(d,REQUIRED_ANI,'ANI evidence (FILE17 standardized or FILE16J native)')
        d['ani_evidence_schema']='FILE17_STANDARDIZED_FRACTIONS'
    require_header(d,REQUIRED_ANI,'normalized ANI evidence')
    need(len(d)>0,'ANI evidence is empty; cannot assume no related genomes')
    need(set(d.cohort).issubset({'E2','E3'}), 'ANI evidence: only E2/E3 values allowed')
    d['external_genome_id']=checked_id(d.external_genome_id,'ANI external_genome_id')
    d['development_genome_id']=checked_id(d.development_genome_id,'ANI development_genome_id')
    # Require candidate row identity uniqueness. Multiple matches per external genome
    # are allowed; report per-genome minimum Mash and maximum ANI with AND on pair.
    need(not d.duplicated(['cohort','external_genome_id','development_genome_id']).any(),
         'ANI evidence: repeated external-to-development pair; reconcile raw skani rows')
    available=set(zip(pred.cohort,pred.genome_id))
    absent=[key for key in zip(d.cohort,d.external_genome_id) if key not in available]
    need(not absent,f'ANI external cohort/ID not in FILE16 frozen predictions: {absent[:7]}; '
         f'check whether file16j external_genome uses a different identifier than FILE16 genome_id; '
         f'do not join on row order')
    for col in ('mash_distance','ani_percent','aligned_fraction_ref','aligned_fraction_query'):
        d[col]=pd.to_numeric(d[col],errors='raise')
        need(np.isfinite(d[col].to_numpy(dtype=float)).all(),f'ANI: missing/nonfinite {col}')
    need(d.mash_distance.between(0,1).all(),'mash_distance needs unit 0..1')
    need(d.ani_percent.between(0,100).all() and (d.ani_percent>50).all(),
         'ani_percent must be percentages (e.g., 99.93, NOT 0.9993)')
    for col in ('aligned_fraction_ref','aligned_fraction_query'):
        need(d[col].between(0,1).all(),f'{col} must be 0..1, not percentages')
    # Closest pairs selected BY MINHASH, followed by validated skani ANI.
    d['qualifying_pair']=(d.mash_distance<=0.0005+1e-12)&(d.ani_percent>=99.93-1e-9)
    d['both_aligned_ge_95pct']=(d.aligned_fraction_ref>=.95)&(d.aligned_fraction_query>=.95)
    sel=d.loc[d.qualifying_pair,['cohort','external_genome_id']].drop_duplicates()
    ids_by_cohort=sel.groupby('cohort').size().to_dict()
    actual={c:int(ids_by_cohort.get(c,0)) for c in ('E2','E3')}
    # In production, enforce published targeted-audit counts as an integrity gate.
    if not testing:
        need(actual=={'E2':31,'E3':25},
             f'ANI-confirmed selected ID counts {actual} do not reproduce recorded 31/25; STOP. '
             'Do not substitute Mash-only candidates or silently redefine cutoff.')
        need(int(d.qualifying_pair.sum())==56,
             'ANI evidence expected 56 unique selected nearest-development PAIRS; reconcile duplicated/extra partner rows')
        need(int((d.qualifying_pair & d.both_aligned_ge_95pct).sum())==55,
             'ANI aligned fraction check did not reproduce recorded 55/56 qualifying pairs; investigate units/provenance')
    qc={'ani_total_pairs':len(d),'ani_total_external_genomes':int(d[['cohort','external_genome_id']].drop_duplicates().shape[0]),
        'confirmed_neighbour_counts':actual,
        'confirmed_pair_count':int(d.qualifying_pair.sum()),
        'confirmed_pair_both_af_ge95_count':int((d.qualifying_pair & d.both_aligned_ge_95pct).sum()),
        'confirm_neighbour_definition': 'Mash<=0.0005 AND skani ANI>=99.93%; aligned fractions descriptive, not an additional exclusion gate',
        'ANI_is_targeted_not_exhaustive': True,
        'input_schemas':sorted(d['ani_evidence_schema'].unique().tolist()),
        'native_verification_statuses':sorted(d['status'].dropna().unique().tolist()) if 'status' in d else []}
    return d,qc


def calculate(pred:pd.DataFrame,ani:pd.DataFrame,testing:bool)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    qualifying=ani.loc[ani.qualifying_pair,['cohort','external_genome_id']].drop_duplicates().copy()
    qualifying['ani_confirmed_near_dev']=True
    keyed=pred.merge(qualifying,how='left',left_on=['cohort','genome_id'],
                     right_on=['cohort','external_genome_id'],validate='one_to_one')
    keyed['ani_confirmed_near_dev']=keyed.ani_confirmed_near_dev.eq(True)
    keyed=keyed.drop(columns=['external_genome_id'])
    full=[]; summary={}
    keys=['cohort','panel','n','n_resistant','n_susceptible','tp','fn','tn','fp',
          'auroc','average_precision','balanced_accuracy','sensitivity','specificity','mcc','brier',
          'error_n','high_confidence_wrong_n','false_susceptible_among_called_susceptible']
    for cohort in ('E2','E3'):
        q=keyed[keyed.cohort==cohort].copy()
        panels={'FROZEN_FULL_PRIMARY':q,
                'TARGETED_ANI_CONFIRMED_NEAR':q[q.ani_confirmed_near_dev],
                'REMAINDER_AFTER_REMOVING_CONFIRMED_NEAR_POSTHOC':q[~q.ani_confirmed_near_dev]}
        ref=metric_panel(q)
        near=metric_panel(panels['TARGETED_ANI_CONFIRMED_NEAR'])
        rem=metric_panel(panels['REMAINDER_AFTER_REMOVING_CONFIRMED_NEAR_POSTHOC'])
        for panel,panel_df in panels.items():
            full.append({'cohort':cohort,'panel':panel,**metric_panel(panel_df)})
        summary[cohort]={'full_unchanged':ref,'ani_confirmed_targeted_neighbours':near,
                         'remainder_sensitivity_posthoc_not_independent':rem,
                         'delta_rem_minus_full_AUROC':_diff(rem['auroc'],ref['auroc']),
                         'delta_rem_minus_full_sensitivity':_diff(rem['sensitivity'],ref['sensitivity']),
                         'delta_rem_minus_full_specificity':_diff(rem['specificity'],ref['specificity']),
                         'n_removed':near['n'],'proportion_removed':near['n']/ref['n']}
    keyed['frozen_prediction_class']=(keyed.prob_resistance>=.5).astype(int)
    keyed['frozen_error']=(keyed.frozen_prediction_class!=keyed.y_true).astype(int)
    keyed['frozen_false_susceptible']=((keyed.y_true==1)&(keyed.frozen_prediction_class==0)).astype(int)
    keyed['analysis_scope']='POSTHOC_TARGETED_ANI_AUDIT_NOT_REPLACEMENT_PRIMARY'
    return keyed,pd.DataFrame(full,columns=keys),summary


def _diff(x,y):return (x-y) if x is not None and y is not None else None


def discover(root:Path):
    candidates=[]
    for area in ('results','data/external_validation'):
        base=root/area
        if not base.is_dir():continue
        for dpath, dirs, files in os.walk(base):
            dirs[:]=[x for x in dirs if x not in {'.git','__pycache__','raw','fastq','fasta','assemblies','unitigs','reads'}]
            for name in files:
                if not name.lower().endswith(('.csv','.tsv','.json')): continue
                p=Path(dpath)/name
                low=str(p.relative_to(root)).lower()
                if not any(w in low for w in ('ani','skani','mash','neigh','pair','file16j','file16_per_genome_prediction')):continue
                header=''
                if p.suffix.lower() in ('.csv','.tsv'):
                    try:
                        header=' | '.join(pd.read_csv(p,sep='\t' if p.suffix.lower()=='.tsv' else ',',nrows=0).columns.tolist()[:24])
                    except Exception as exc:header='UNREADABLE: '+str(exc)[:120]
                candidates.append({'relative_path':str(p.relative_to(root)),'size_bytes':p.stat().st_size,'header':header})
    return candidates


def audit(args, testing=False):
    root=Path(args.project_root).expanduser().resolve()
    need(root.is_dir(),f'Project root missing: {root}')
    predpath=fullpath(root,args.predictions)
    anipath=fullpath(root,args.ani_pairs)
    out=fullpath(root,args.outdir)
    need(not out.exists(),f'Output already exists, refusal to overwrite: {out}. Choose a new --outdir')
    # Fail closed BEFORE writing the new output directory.
    preds=read_predictions(predpath,testing=testing)
    near,ani_qc=read_ani(anipath,preds,testing=testing)
    pergenome,panels,summary=calculate(preds,near,testing)
    hashes_before={'FILE16_predictions':sha256(predpath),'ANI_pair_evidence':sha256(anipath)}
    out.mkdir(parents=True,exist_ok=False)
    pergenome.to_csv(out/'file17_per_genome_neighbour_flags.csv',index=False)
    panels.to_csv(out/'file17_descriptive_sensitivity_panels.csv',index=False)
    near.to_csv(out/'file17_ANI_evidence_qualified_pairs.csv',index=False)
    report={
        'file17_version':VERSION,'generated_utc':utc(),'status':'POSTHOC_FROZEN_TARGETED_SENSITIVITY_ONLY',
        'production_full_count_guardrails_applied':not testing,
        'input_sha256':hashes_before,'ANI_evidence_quality':ani_qc,
        'frozen_and_sensitivity_results':summary,
        'method':'Compare unchanged, frozen model scores across full cohort, selected targeted ANI-confirmed near neighbours, and remainder after removing only that selected subset. Threshold fixed at 0.5.',
        'e2_vs_e3_are_different_frozen_models':True,
        'not_validated_as_genomically_independent':'The remainder was NOT exhaustively subjected to ANI testing, hence should never be described as clonally independent.',
        'no_primary_cohort_redefinition':True,'no_training_or_threshold_selection':True,
        'no_AST_label_adjudication':True,'no_p_values_or_unjustified_CIs':True,
        'clinical_use':'NOT_FOR_CLINICAL_SUSCEPTIBILITY_REPORTING',
        'interpretation_guardrails':[
           'Targeted post-hoc selection cannot give an unbiased prevalence of clonal relatedness.',
           'The 56 detected near-neighbours do not imply identical clinical isolates or same-patient origin.',
           'Remainder metrics are secondary sensitivities, NOT new external primary results.',
           'E2 and E3 use different frozen model artifacts (668 vs 702 features).',
           'Group-relatedness and source-dependent inference remain unresolved for many E3 records.',
           'No adjustment of thresholds, phenotypes or selection based on prediction accuracy.',
           'AF>=95% in both directions is reported separately; it is NOT used as an undocumented gate.'
        ]
    }
    write_json(out/'file17_summary.json',report)
    body=['# FILE17 — Frozen external genomic near-neighbour sensitivity audit','',
          '**Post-hoc / descriptive only.** Full-cohort results are unchanged. The remaining set is not ANI-certified independent.','',
          f'Input FILE16 SHA-256: `{hashes_before["FILE16_predictions"]}`',
          f'Input ANI SHA-256: `{hashes_before["ANI_pair_evidence"]}`','']
    for cohort in ('E2','E3'):
        z=summary[cohort]; f=z['full_unchanged']; rem=z['remainder_sensitivity_posthoc_not_independent'];near_m=z['ani_confirmed_targeted_neighbours']
        body+= [f'## {cohort}',
                f'- Original frozen primary n={f["n"]}, AUROC={_fmt(f["auroc"])}, TP/FN/TN/FP={f["tp"]}/{f["fn"]}/{f["tn"]}/{f["fp"]}.',
                f'- Targeted ANI-confirmed selected subset n={near_m["n"]} (R={near_m["n_resistant"]}, S={near_m["n_susceptible"]}).',
                f'- Remainder (not certified independent) n={rem["n"]}, AUROC={_fmt(rem["auroc"])}, TP/FN/TN/FP={rem["tp"]}/{rem["fn"]}/{rem["tn"]}/{rem["fp"]}.',
                f'- Descriptive AUROC difference remainder minus full={_fmt(z["delta_rem_minus_full_AUROC"])}.','']
    body += ['**Do not replace published primary E2/E3 estimates with remainder values.**',
             'Close-ANI pairs are derived from selective Mash screening; unknown relatedness in the remainder is unquantified.',
             'No independent laboratory MIC or breakpoint verification; label errors not adjudicated.','']
    (out/'file17_reporting_guardrails.md').write_text('\n'.join(body),encoding='utf-8')
    for name,p in [('FILE16_predictions',predpath),('ANI_pair_evidence',anipath)]:
        need(sha256(p)==hashes_before[name],f'INPUT CHANGED DURING FILE17 EXECUTION: {name}')
    manifest={'generated_utc':utc(),'version':VERSION,'outputs_sha256':{
        p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
        'input_sha256':hashes_before,'verified_inputs_unchanged':True}
    write_json(out/'file17_output_manifest_sha256.json',manifest)
    print('[FILE17] AUDIT COMPLETED — research only (not primary validation)')
    print(json.dumps({'output':str(out),'confirmed_counts':ani_qc['confirmed_neighbour_counts'],
                      'full_E2':summary['E2']['full_unchanged']['n'],
                      'full_E3':summary['E3']['full_unchanged']['n'],
                      'frozen_inputs_unchanged':True},indent=2))
    return report


def _fmt(x):return f'{x:.6f}' if x is not None else 'not estimable'


# Self-contained synthetic tests: no adjacent tests_file17.py needed.
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
    with tempfile.TemporaryDirectory(prefix='file17_synthetic_') as td:
        root=Path(td);pred,ani=build_data(root)
        data=read_predictions(pred,testing=True)
        selected,meta=read_ani(ani,data,testing=True)
        assert meta['confirmed_neighbour_counts']=={'E2':2,'E3':2}
        # Native FILE16J schema: aligned fraction columns are percentages, not decimals.
        native=pd.read_csv(ani).rename(columns={
            'external_genome_id':'external_genome',
            'development_genome_id':'nearest_dev_genome',
            'mash_distance':'min_mash_distance',
            'aligned_fraction_ref':'aligned_fraction_dev_percent',
            'aligned_fraction_query':'aligned_fraction_external_percent'})
        native['aligned_fraction_dev_percent']*=100
        native['aligned_fraction_external_percent']*=100
        native['status']='ANI_VERIFIED_TEST_ONLY'
        nativepath=root/'ani_native.csv';native.to_csv(nativepath,index=False)
        native_pairs,native_meta=read_ani(nativepath,data,testing=True)
        assert native_meta['confirmed_neighbour_counts']==meta['confirmed_neighbour_counts']
        assert native_meta['input_schemas']==['FILE16J_NATIVE_PERCENTAGES']
        assert int((native_pairs.qualifying_pair & native_pairs.both_aligned_ge_95pct).sum())==4
        native.loc[0,'aligned_fraction_dev_percent']=200
        native.to_csv(root/'native_bad.csv',index=False)
        raises(lambda:read_ani(root/'native_bad.csv',data,testing=True),'0..100')
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


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    s=p.add_subparsers(dest='cmd',required=True)
    d=s.add_parser('discover',help='Find possible result artifacts; do not auto-select ANI evidence')
    d.add_argument('--project-root',default='.')
    a=s.add_parser('audit',help='Post-hoc analysis from verified FILE16 predictions and ANI/skani pairs')
    a.add_argument('--project-root',default='.')
    a.add_argument('--predictions',default=DEFAULT_PRED)
    a.add_argument('--ani-pairs',default=DEFAULT_ANI,help='Defaults to native FILE16J ANI-verified 56-pair CSV')
    a.add_argument('--outdir',default=DEFAULT_OUT)
    s.add_parser('self-test',help='Run isolated synthetic test suite; never produces scientific estimates')
    return p


def main():
    a=parser().parse_args()
    if a.cmd=='discover':
        root=Path(a.project_root).expanduser().resolve()
        if not root.is_dir():raise FileNotFoundError(root)
        print(json.dumps({'project_root':str(root),'candidate_files':discover(root),
                          'note':'Candidates not chosen automatically; Mash-only is not ANI confirmation'},indent=2))
    elif a.cmd=='audit':audit(a)
    else:
        run_tests()


if __name__=='__main__':
    try:main()
    except (FileNotFoundError,ValueError,KeyError) as exc:
        print('[FILE17][STOP] '+str(exc),file=sys.stderr)
        sys.exit(2)
