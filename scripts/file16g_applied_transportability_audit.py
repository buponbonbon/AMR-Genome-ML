#!/usr/bin/env python3
"""FILE16G: post-hoc source-aware transportability audit for frozen E2/E3.

No training, threshold optimization, phenotype modification, primary-cohort exclusion,
or clinical recommendation. Run from AMR-Genome-ML project root. Python >=3.9.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FOCUS = 'PRJNA1322038'   # Discovered post hoc in FILE16D; never a training target.
SCHEMA = 'FILE16G_APPLIED_POSTHOC_1.0'


def digest(p):
    sha = hashlib.sha256()
    with open(p, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def assert_ok(condition, msg):
    if not condition:
        raise ValueError(msg)


def ratio(a, b):
    return float(a / b) if b else None


def clean_json(val):
    if isinstance(val, dict):
        return {str(k): clean_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [clean_json(v) for v in val]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val) if np.isfinite(val) else None
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float) and not math.isfinite(val):
        return None
    return val


def dump_json(path, val):
    path.write_text(json.dumps(clean_json(val), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def validate_pred(d, cohort, expected_n, expected_r):
    req = {'cohort', 'genome_id', 'y_true', 'prob_resistance', 'mechanism_group',
           'predicted_resistant_at_frozen_threshold_05'}
    assert_ok(req.issubset(d.columns), f'{cohort}: missing columns {req - set(d.columns)}')
    d = d.loc[d.cohort.eq(cohort)].copy()
    assert_ok(len(d) == expected_n, f'{cohort}: expected {expected_n} rows, got {len(d)}')
    assert_ok(d.genome_id.notna().all() and d.genome_id.astype(str).str.strip().ne('').all(), f'{cohort}: blank genome IDs')
    assert_ok(d.genome_id.is_unique, f'{cohort}: duplicate genome IDs')
    d['y_true'] = pd.to_numeric(d['y_true'], errors='raise').astype(int)
    d['prob_resistance'] = pd.to_numeric(d['prob_resistance'], errors='raise').astype(float)
    assert_ok(d.y_true.isin([0, 1]).all(), f'{cohort}: nonbinary phenotype')
    assert_ok(int(d.y_true.sum()) == expected_r, f'{cohort}: expected {expected_r} resistant')
    assert_ok(d.prob_resistance.notna().all() and d.prob_resistance.between(0, 1).all(), f'{cohort}: invalid scores')
    assert_ok(d.mechanism_group.notna().all(), f'{cohort}: missing mechanisms (need FILE16B audit)')
    d['pred'] = (d.prob_resistance >= 0.5).astype(int)
    old = pd.to_numeric(d.predicted_resistant_at_frozen_threshold_05, errors='raise').astype(int)
    assert_ok((old == d.pred).all(), f'{cohort}: disagreement with frozen 0.5 predictions')
    d['wrong'] = (d.pred != d.y_true).astype(int)
    d['fn'] = ((d.y_true == 1) & (d.pred == 0)).astype(int)
    d['fp'] = ((d.y_true == 0) & (d.pred == 1)).astype(int)
    d['high_conf_wrong'] = ((d.wrong == 1) & ((d.prob_resistance <= .1) | (d.prob_resistance >= .9))).astype(int)
    d['no_major'] = d.mechanism_group.eq('No_detected_major_family')
    return d


def metrics(d):
    n = len(d)
    y = d.y_true.to_numpy(dtype=int)
    p = d.prob_resistance.to_numpy(dtype=float)
    pred = d.pred.to_numpy(dtype=int)
    nr = int(np.sum(y)); ns = n-nr
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = nr - tp; fp = ns - tn
    return {
        'n': n, 'n_resistant': nr, 'n_susceptible': ns,
        'auroc': float(roc_auc_score(y,p)) if nr and ns else None,
        'average_precision': float(average_precision_score(y,p)) if nr else None,
        'brier': float(np.mean((p-y)**2)) if n else None,
        'sensitivity': ratio(tp,nr), 'specificity': ratio(tn,ns),
        'balanced_accuracy': (tp/nr+tn/ns)/2 if nr and ns else None,
        'false_negative_rate': ratio(fn,nr), 'error_rate': ratio(fn+fp,n),
        'tp':tp,'tn':tn,'fp':fp,'fn':fn,'n_wrong':fn+fp,
        'n_high_conf_wrong':int(d.high_conf_wrong.sum()),
    }


def wilson(k, n):
    if n == 0: return (None,None)
    z=1.959963984540054
    p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    wid=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(max(0,mid-wid)),float(min(1,mid+wid))


def mechanism_row(d,cohort,source,panel):
    q = d[d.no_major.eq(panel)]
    nr = int(q.y_true.sum()); ns = len(q)-nr
    fn = int(q.fn.sum()); fp = int(q.fp.sum())
    low,hi=wilson(fn,nr)
    mlst = q.mlst_group.dropna().astype(str).str.strip() if 'mlst_group' in q else pd.Series(dtype=str)
    erd = q.erd_group.dropna().astype(str).str.strip() if 'erd_group' in q else pd.Series(dtype=str)
    return dict(cohort=cohort,source_stratum=source, panel_group='No_detected_five_family_panel' if panel else 'At_least_one_detected_family',
                n=len(q),n_resistant=nr,n_susceptible=ns,
                fn=fn,fn_rate=ratio(fn,nr),fn_wilson_low=low,fn_wilson_high=hi,
                fp=fp,fp_rate=ratio(fp,ns),high_conf_fn=int((q.fn*q.high_conf_wrong).sum()),
                n_known_erd=int(erd.ne('').sum()),n_distinct_known_erd=int(erd[erd.ne('')].nunique()),
                n_known_mlst=int(mlst.ne('').sum()),n_distinct_known_mlst=int(mlst[mlst.ne('')].nunique()),
                caveat='5-family restricted screen; observational, not causal or all-carbapenemase-negative')


def source_sensitivity(df, full):
    records=[]
    for source,q in df.groupby('bioproject_acc',sort=True,dropna=False):
        ms=metrics(q)
        leave=metrics(df[df.bioproject_acc.ne(source)])
        records.append(dict(bioproject_acc=source,post_hoc=True,
                            n=ms['n'],n_resistant=ms['n_resistant'],n_susceptible=ms['n_susceptible'],
                            n_wrong=ms['n_wrong'],n_high_conf_wrong=ms['n_high_conf_wrong'],
                            error_rate=ms['error_rate'],brier=ms['brier'],auroc=ms['auroc'],
                            fn=ms['fn'],fn_rate=ms['false_negative_rate'],
                            n_erd_known=int(q.erd_group.notna().sum()) if 'erd_group' in q else None,
                            share_of_all_E2_errors=ratio(ms['n_wrong'],full['n_wrong']),
                            n_after_leave_one_out=leave['n'],
                            auroc_leave_one_out=leave['auroc'],
                            delta_auroc_after_removal=leave['auroc']-full['auroc'] if leave['auroc'] is not None else None,
                            brier_leave_one_out=leave['brier'],
                            delta_brier_after_removal=leave['brier']-full['brier']))
    return pd.DataFrame(records).sort_values(['n_wrong','n'],ascending=False)


def bootstrap_source_mixture(df, reps, seed, unit):
    """Entire-source resampling is a descriptive SOURCE-MIX sensitivity, NOT a clinical CI."""
    if unit=='bioproject':
        keys=df.bioproject_acc.astype(str).to_numpy()
    elif unit=='erd':
        if 'erd_group' not in df:
            raise ValueError('ERD group column absent')
        raw=df.erd_group.fillna('').astype(str).str.strip().to_numpy()
        keys=np.array([v if v and v.lower() not in ('nan','unknown','none','na','unresolved') else f'UNRESOLVED_{i}' for i,v in enumerate(raw)])
    elif unit=='genome_iid':
        keys=np.array([f'genome_{i}' for i in range(len(df))])
    else:
        raise ValueError(unit)
    unique,indices=np.unique(keys,return_inverse=True)
    blocks=[np.flatnonzero(indices==i) for i in range(len(unique))]
    rng=np.random.default_rng(seed)
    vals={'auroc':[],'brier':[],'false_negative_rate':[],'error_rate':[]}
    invalid=0
    for _ in range(reps):
        ids=rng.integers(0,len(blocks),size=len(blocks))
        rowids=np.concatenate([blocks[i] for i in ids])
        t=df.iloc[rowids]
        met=metrics(t)
        if met['auroc'] is None:
            invalid+=1
        for key in vals:
            x=met[key]
            if x is not None: vals[key].append(x)
    return {
        'resampling_unit':unit, 'n_resampling_blocks':len(blocks), 'n_unresolved_singletons':int(np.sum(np.char.startswith(unique.astype(str),'UNRESOLVED_'))),
        'n_replicates':reps,'reps_without_both_classes':invalid,
        'descriptive_distribution_2_5_to_97_5':{k:{'median':float(np.median(v)) if v else None,
            'lower':float(np.percentile(v,2.5)) if v else None,'upper':float(np.percentile(v,97.5)) if v else None,
            'valid_reps':len(v)} for k,v in vals.items()},
        'caution':('BioProject resampling varies the observed source mixture; not a confidence interval for clinical deployment.' if unit=='bioproject' else 'Unresolved genomic groups are singletons; clustering may remain incomplete.')
    }


def parse_rule(x):
    val=str(x).strip().lower()
    if val in ('1','1.0','true','t','yes','y','r','resistant','positive'):return 1
    if val in ('0','0.0','false','f','no','n','s','susceptible','negative'):return 0
    raise ValueError(f'Unsupported/blank rule call value: {x!r}. Verify file11e rule schema.')


def rule_metrics(df,name):
    y=df.y_true.to_numpy(dtype=int);p=df[name].to_numpy(dtype=int)
    n_r=int(sum(y));n_s=len(y)-n_r
    tp=int(np.sum((y==1)&(p==1)));tn=int(np.sum((y==0)&(p==0)))
    fn=n_r-tp;fp=n_s-tn
    return dict(method=name,n=len(df),n_resistant=n_r,n_susceptible=n_s,
                tp=tp,fn=fn,tn=tn,fp=fp, sensitivity=ratio(tp,n_r),specificity=ratio(tn,n_s),
                balanced_accuracy=(tp/n_r+tn/n_s)/2 if n_r and n_s else None,
                accuracy=ratio(tp+tn,len(df)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('.'))
    parser.add_argument('--predictions',type=Path,default=Path('results/file16_audit_mechanisms_v1/file16_per_genome_prediction_error_audit.csv'))
    parser.add_argument('--frozen-e2',type=Path,default=Path('data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv'))
    parser.add_argument('--rule-calls',type=Path,default=Path('data/external_validation/ncbi_pathogen_detection/file11e/file11e_rule_calls.csv'))
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--outdir',type=Path,default=Path('results/file16g_applied_v1'))
    args=parser.parse_args()
    root=args.project_root.resolve()
    def absolute(p): return p if p.is_absolute() else root/p
    pred_file=absolute(args.predictions);frozen_file=absolute(args.frozen_e2);rule_file=absolute(args.rule_calls);out=absolute(args.outdir)
    assert_ok(100<=args.bootstrap<=20000,'--bootstrap must be in [100,20000]')
    assert_ok(pred_file.is_file() and frozen_file.is_file(),'Missing FILE16 predictions or E2 frozen inputs')
    assert_ok(not out.exists(),f'Output directory already exists: {out}; use a new _vN')
    # Read/validate BEFORE creating outputs.
    raw=pd.read_csv(pred_file,low_memory=False)
    assert_ok(len(raw)==831,'FILE16 predictions must contain E2=525 and E3=306 only')
    assert_ok(set(raw.cohort)=={'E2','E3'},'Unexpected cohort values')
    e2=validate_pred(raw,'E2',525,163); e3=validate_pred(raw,'E3',306,122)
    rawf=pd.read_csv(frozen_file,low_memory=False)
    assert_ok({'asm_acc','bioproject_acc','label_binary','antibiotic'}.issubset(rawf.columns),'Missing frozen E2 keys')
    f=rawf[rawf.antibiotic.astype(str).str.lower().eq('meropenem')].copy()
    assert_ok(len(f)==525 and f.asm_acc.is_unique,'E2 frozen meropenem must have 525 unique assemblies')
    e2=e2.merge(f[['asm_acc','bioproject_acc','label_binary']].rename(columns={'asm_acc':'genome_id'}),how='left',on='genome_id',validate='one_to_one',indicator=True)
    assert_ok(e2._merge.eq('both').all() and e2.bioproject_acc.notna().all(),'Could not match E2 frozen phenotype + BioProject for all genomes')
    assert_ok((e2.y_true==pd.to_numeric(e2.label_binary,errors='raise').astype(int)).all(),'E2 labels disagree with frozen phenotype')
    e2=e2.drop(columns=['_merge','label_binary'])
    assert_ok(e2.bioproject_acc.nunique()==32,'Expected 32 E2 BioProjects')
    focus=e2[e2.bioproject_acc.eq(FOCUS)];other=e2[e2.bioproject_acc.ne(FOCUS)]
    assert_ok(len(focus)==175 and len(other)==350,'Post-hoc source size unexpectedly changed; inspect provenance')
    m_e2=metrics(e2);m_e3=metrics(e3)
    assert_ok((m_e2['tp'],m_e2['fn'],m_e2['tn'],m_e2['fp'])==(109,54,221,141),'E2 predictions not frozen reference')
    assert_ok((m_e3['tp'],m_e3['fn'],m_e3['tn'],m_e3['fp'])==(67,55,157,27),'E3 predictions not frozen reference')
    strata=[]
    for cohort, source, d in [('E2','ALL_32_BIOPROJECTS',e2),('E2','FOCUS_POSTHOC',focus),('E2','OTHER_31_SOURCES_POSTHOC',other),('E3','ALL_E3_SOURCE_UNKNOWN',e3)]:
        for is_no_major in [True,False]:
            strata.append(mechanism_row(d,cohort,source,is_no_major))
    mechanisms=pd.DataFrame(strata)
    sources=source_sensitivity(e2,m_e2)
    # Rule predictions are optional only to permit audit of attached CSVs; they should exist on user's project.
    rule_comparison=[]
    rules_present=rule_file.is_file()
    if rules_present:
        r=pd.read_csv(rule_file,low_memory=False)
        assert_ok({'asm_acc','kpc_family_rule','amrfinder_carbapenemase_rule'}.issubset(r.columns),'E2 rule column mismatch')
        assert_ok(r.asm_acc.is_unique,'Duplicate rule call assembly accession')
        tmp=e2.merge(r[['asm_acc','kpc_family_rule','amrfinder_carbapenemase_rule']].rename(columns={'asm_acc':'genome_id'}),on='genome_id',how='left',validate='one_to_one',indicator=True)
        assert_ok(tmp._merge.eq('both').all(),'E2 rule calls do not cover all 525 assemblies')
        for method in ['kpc_family_rule','amrfinder_carbapenemase_rule']:
            tmp[method]=tmp[method].map(parse_rule).astype(int)
        for group,d in [('ALL_E2',tmp),('FOCUS_POSTHOC',tmp[tmp.bioproject_acc.eq(FOCUS)]),('OTHER_31_SOURCES_POSTHOC',tmp[tmp.bioproject_acc.ne(FOCUS)])]:
            for method in ['pred','kpc_family_rule','amrfinder_carbapenemase_rule']:
                x=rule_metrics(d,method)
                x.update(source_stratum=group,post_hoc=True)
                rule_comparison.append(x)
    source_robustness={k:bootstrap_source_mixture(e2,args.bootstrap,args.seed+i,k) for i,k in enumerate(['bioproject','erd','genome_iid'])}
    # E3 ERD is mostly unresolved; its bootstrap in FILE16 already documented, do not claim precision.
    out.mkdir(parents=True,exist_ok=False)
    sources.to_csv(out/'file16g_all_sources_leave_one_out.csv',index=False)
    mechanisms.to_csv(out/'file16g_mechanism_by_source.csv',index=False)
    if rule_comparison:
        pd.DataFrame(rule_comparison).to_csv(out/'file16g_rule_vs_ml_source_strata.csv',index=False)
    dump_json(out/'file16g_source_mix_bootstrap.json',source_robustness)
    important={
        'file16g_schema':SCHEMA,'generated_utc':datetime.now(timezone.utc).isoformat(),
        'status':'POST_HOC_APPLIED_TRANSPORTABILITY_NOT_CAUSAL_NOT_CLINICAL',
        'reference_E2_all_525':m_e2,'reference_E3_all_306_DIFFERENT_MODEL_ARTIFACT':m_e3,
        'focus_source_discovered_posthoc':FOCUS,'focus_source_metrics':metrics(focus),
        'other_sources_metrics_NOT_ALTERNATIVE_PRIMARY':metrics(other),
        'errors_in_focus_source':int(focus.wrong.sum()),'all_E2_errors':int(e2.wrong.sum()),
        'share_of_errors_focus':ratio(int(focus.wrong.sum()),int(e2.wrong.sum())),
        'restricted_panel_source_stratification':strata,
        'n_e2_sources':int(e2.bioproject_acc.nunique()),
        'e2_rule_comparison_available':rules_present,
        'input_sha256':{'predictions':digest(pred_file),'frozen_e2':digest(frozen_file),
                        'rule_calls':digest(rule_file) if rules_present else None},
        'no_training':True,'no_threshold_selection':True,'no_primary_E2_exclusion':True,
        'research_limits':[
            'The anomalous source was discovered using E2 outcomes: all subgroup analyses are exploratory.',
            'E2 and E3 predictions come from different frozen model artifacts (668 vs 702 features).',
            'No independent AST/MIC verification of PRJNA1322038 phenotype; no assertion that labels are wrong.',
            'The five-family panel does not prove a genome lacks carbapenemases or other AMR mechanisms.',
            'Only nine resistant/no-major-panel E2 genomes remain after separating the focus source.',
            'BioProject bootstrap is SOURCE-MIX sensitivity, not a confidence interval establishing external generalizability.',
            'ERD clustering includes singleton unresolved IDs and cannot establish sequence independence.',
            'Any rule-vs-ML comparisons are exploratory and are not prospective clinical safety evaluations.',
            'E3 BioProject identifiers are not present in the provided prediction table; do not infer E3 source effects.',
            'No predictor is a replacement for isolate-level phenotypic AST or clinical decision making.'
        ],
        'next_research_gate':{
            'positive':'Independent source-level genotype/phenotype harmonization OR independent prospective failure alarm validation; no post-hoc exclusion.',
            'negative':'If independent phenotype quality/lineage cannot be established, report source-stratified sensitivity only and frame work as methodological caution.',
            'applied_manuscript':'Original research: external transportability, source and mechanism failure analysis, clinical-microbiology risk; software is reproducibility companion.'}
    }
    dump_json(out/'file16g_summary.json',important)
    dump_json(out/'file16g_manifest.json',{'outputs':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()},'no_input_modified':True})
    print('[FILE16G] COMPLETE: '+str(out))
    print(json.dumps({'n_e2':len(e2),'n_e3':len(e3),'n_bioproject':int(e2.bioproject_acc.nunique()),
                      'focus_error_n':int(focus.wrong.sum()),'e2_errors':int(e2.wrong.sum()),
                      'rules_evaluated':bool(rule_comparison),'bootstrap':args.bootstrap,
                      'interpretation':'Exploratory applied/source-aware sensitivity ONLY; no new classifier and no independent clinical validation.'},indent=2))

if __name__=='__main__':
    main()
