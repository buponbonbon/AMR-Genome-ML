#!/usr/bin/env python3
"""FILE16D: post-hoc BioProject / source-phenotype audit for frozen E2.

Read-only audit. No training, model reload, sample exclusion, phenotype changes,
threshold optimization or recalibration. Generated source-stratified statistics
are diagnostic and must not replace locked E2 headline results.

Run with python scripts/file16d_source_phenotype_audit.py --project-root .
Optional: --raw-ast /path/to/file11_ncbi_ast_isolates_joined_raw.csv
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score, brier_score_loss

FLAG_PROJECT = 'PRJNA1322038'
EXPECTED_N = 525
EXPECTED_POS = 163
INPUT_NAMES = {
 'raw_ast':'file11_ncbi_ast_isolates_joined_raw.csv',
 'frozen':'file11_external_strict_E2_FROZEN.csv',
 'predictions':'file16_per_genome_prediction_error_audit.csv',
 'genes':'file16c_per_genome_gene_audit.csv',
}
CANDIDATES = {
 'raw_ast':['data/external_validation/ncbi_pathogen_detection/raw/file11_ncbi_ast_isolates_joined_raw.csv',
            'data/external_validation/ncbi_pathogen_detection/file11_ncbi_ast_isolates_joined_raw.csv',
            'data/external_validation/ncbi_pathogen_detection/file11_ncbi_ast_isolates_joined_raw.csv',
            'file11_ncbi_ast_isolates_joined_raw.csv'],
 'frozen':['data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv',
           'file11_external_strict_E2_FROZEN.csv'],
 'predictions':['results/file16_audit_mechanisms_v1/file16_per_genome_prediction_error_audit.csv',
                'results/file16_audit_v1/file16_per_genome_prediction_error_audit.csv',
                'file16_per_genome_prediction_error_audit.csv'],
 'genes':['results/file16c_discordance_v1/file16c_per_genome_gene_audit.csv',
          'file16c_per_genome_gene_audit.csv'],
}


def die(msg):
    raise RuntimeError(msg)


def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):
            h.update(buf)
    return h.hexdigest()


def unique_output(out:Path):
    if out.exists():die(f'Output already exists, refusing overwrite: {out} (choose --outdir with _v2)')
    out.mkdir(parents=True)


def locate(root:Path,kind:str,override):
    if override:
        p=Path(override).expanduser()
        if not p.is_absolute():p=root/p
        if p.is_file():return p.resolve()
        die(f'Missing {kind} file at --{kind.replace("_","-")} {p}')
    for rel in CANDIDATES[kind]:
        p=root/rel
        if p.is_file():return p.resolve()
    # Limited filename discovery: script could live in scripts and user can have a source copy elsewhere.
    paths=[]
    for top in ('results','data','scripts'):
        folder=root/top
        if folder.exists():paths.extend(folder.rglob(INPUT_NAMES[kind]))
    paths=sorted({str(x.resolve()) for x in paths if x.is_file()})
    if len(paths)==1:return Path(paths[0])
    if len(paths)>1:die(f'Multiple possible files for {kind}; pass explicit --{kind.replace("_","-")}: {paths[:8]}')
    die(f'Missing {kind} ({INPUT_NAMES[kind]}); place a copy in project root or pass --{kind.replace("_","-")} with its actual path. No original file is modified.')


def require_cols(df,cols,where):
    missing=sorted(set(cols)-set(df.columns))
    if missing:die(f'{where}: required columns missing: {missing}; actual={list(df.columns)}')


def canonical(s):
    return s.fillna('').astype(str).str.strip()


def norm_pheno(s):
    return canonical(s).str.lower()


def metrics(sub):
    y=sub['y_true'].to_numpy(int)
    pr=sub['prob_resistance'].to_numpy(float)
    yp=(pr>=.5).astype(int)
    n=len(sub)
    return {
        'n':n,'n_resistant':int(y.sum()),'n_susceptible':int(n-y.sum()),
        'auroc':float(roc_auc_score(y,pr)) if np.unique(y).size==2 else None,
        'brier':float(brier_score_loss(y,pr)) if n else None,
        'tp':int(((yp==1)&(y==1)).sum()),
        'fn':int(((yp==0)&(y==1)).sum()),
        'tn':int(((yp==0)&(y==0)).sum()),
        'fp':int(((yp==1)&(y==0)).sum()),
        'n_wrong':int((yp!=y).sum()),
        'n_high_conf_wrong':int(((yp!=y)&(np.maximum(pr,1-pr)>=.9)).sum()),
        'susceptibility_false_negative_rate':float(((yp==0)&(y==1)).sum()/y.sum()) if y.sum() else None,
    }


def safe_group_units(df, col='erd_group'):
    # Group only for resampling uncertainty; empty ERD is a singleton, not biological independence.
    out=[]
    for i,(g,genome_id) in enumerate(zip(df[col],df['genome_id'])):
        if pd.isna(g) or not str(g).strip():out.append('unresolved_singleton:'+str(genome_id))
        else:out.append(str(g).strip())
    return np.asarray(out,dtype=object)


def bootstrap_auc(sub,nboot=2000,seed=42):
    if not len(sub):return {'valid_replicates':0,'ci95_low':None,'ci95_high':None}
    groups=safe_group_units(sub)
    partitions=[np.flatnonzero(groups==g) for g in pd.unique(groups)]
    rng=np.random.default_rng(seed);est=[]
    y=sub.y_true.to_numpy(dtype=int);p=sub.prob_resistance.to_numpy(dtype=float)
    for _ in range(nboot):
        picked=rng.integers(0,len(partitions),len(partitions))
        idx=np.concatenate([partitions[k] for k in picked]); ys=y[idx]
        if ys.min()==ys.max():continue
        est.append(float(roc_auc_score(ys,p[idx])))
    if not est:return {'valid_replicates':0,'ci95_low':None,'ci95_high':None}
    return {'valid_replicates':len(est), 'ci95_low':float(np.quantile(est,.025)), 'ci95_high':float(np.quantile(est,.975)),'resampling_blocks':len(partitions),'note':'ERD grouped when available; unresolved ERD singleton; descriptive post-hoc, not proof of independence'}


def two_by_two(sub):
    # Major-family detection and meropenem phenotype; association, not causal mechanism.
    # rows: major detected / no major detected; cols: resistant / susceptible
    a=int((sub.has_major_panel & sub.y_true.eq(1)).sum())
    b=int((sub.has_major_panel & sub.y_true.eq(0)).sum())
    c=int((~sub.has_major_panel & sub.y_true.eq(1)).sum())
    d=int((~sub.has_major_panel & sub.y_true.eq(0)).sum())
    odds,p=fisher_exact([[a,b],[c,d]],alternative='two-sided')
    or_corr=((a+.5)*(d+.5))/((b+.5)*(c+.5))
    return {'major_R':a,'major_S':b,'no_major_R':c,'no_major_S':d,
       'fisher_exact_odds_ratio':float(odds) if np.isfinite(odds) else None,
       'haldane_corrected_odds_ratio':float(or_corr),
       'fisher_p_exploratory_ignores_relatedness':float(p),
       'warning':'Observations may be clonally related; no inference of causality or labeling error.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--project-root',default='.',help='AMR-Genome-ML project root')
    ap.add_argument('--raw-ast',help='Source AST table path; needed if auto location fails')
    ap.add_argument('--frozen',help='Original E2 frozen cohort CSV')
    ap.add_argument('--predictions',help='FILE16 per-genome prediction audit CSV')
    ap.add_argument('--genes',help='FILE16C per-genome gene audit CSV')
    ap.add_argument('--focus-project',default=FLAG_PROJECT)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--outdir',default='results/file16d_source_audit_v1')
    args=ap.parse_args()
    if args.bootstrap<0 or args.bootstrap>10000:die('bootstrap must be between 0 and 10000')
    root=Path(args.project_root).expanduser().resolve()
    if not root.is_dir():die('Project root not found: '+str(root))
    sources={name:locate(root,name,getattr(args,name)) for name in INPUT_NAMES}
    out=Path(args.outdir).expanduser()
    if not out.is_absolute():out=root/out
    if out.exists():die('Output exists, refusing overwrite: '+str(out))
    print('[FILE16D] Using inputs:')
    for k,p in sources.items():print(' ',k,'=',p)
    raw=pd.read_csv(sources['raw_ast'],low_memory=False)
    frozen=pd.read_csv(sources['frozen'],low_memory=False)
    preds=pd.read_csv(sources['predictions'],low_memory=False)
    genes=pd.read_csv(sources['genes'],low_memory=False)
    require_cols(raw,['target_acc','antibiotic','phenotype','asm_acc','biosample_acc','bioproject_acc'],'raw AST')
    require_cols(frozen,['target_acc','antibiotic','phenotype','asm_acc','biosample_acc','bioproject_acc','label_binary','erd_group'],'frozen cohort')
    require_cols(preds,['cohort','genome_id','y_true','prob_resistance'],'FILE16 predictions')
    require_cols(genes,['cohort','genome_id','mechanism_group'],'FILE16C gene audit')
    if not set(norm_pheno(frozen.phenotype)).issubset({'resistant','susceptible'}):die('Frozen phenotype unexpected category')
    # Exact duplicated records can legitimately arise from raw query. If one key has conflicting content: stop.
    n_raw_dup=int(raw.duplicated().sum())
    raw_unique=raw.drop_duplicates().copy()
    raw_unique['antibiotic']=norm_pheno(raw_unique.antibiotic)
    raw_unique['phenotype']=norm_pheno(raw_unique.phenotype)
    frozen['antibiotic']=norm_pheno(frozen.antibiotic)
    frozen['phenotype']=norm_pheno(frozen.phenotype)
    key=['target_acc','antibiotic']
    if raw_unique.duplicated(key).any():
        ex=raw_unique.loc[raw_unique.duplicated(key,keep=False),key].head(5).to_dict('records')
        die(f'Raw AST has conflicting duplicate target+antibiotic records: {ex}')
    if frozen.duplicated(key).any():die('Frozen table contains duplicate target+antibiotic keys')
    raw_join=frozen.merge(raw_unique,on=key,how='left',suffixes=('_frozen','_raw'),indicator=True,validate='one_to_one')
    if not raw_join['_merge'].eq('both').all():die('Some frozen rows not in raw AST source')
    check_fields=['phenotype','asm_acc','biosample_acc','bioproject_acc']
    checks={}
    for field in check_fields:
        x=canonical(raw_join[f'{field}_frozen']).str.lower()
        y=canonical(raw_join[f'{field}_raw']).str.lower()
        bad=raw_join[x.ne(y)].copy()
        checks[field]={'matched':int(len(raw_join)-len(bad)),'mismatched':int(len(bad))}
        if len(bad):die(f'Frozen/raw {field} mismatch: {len(bad)} records; sample '+str(bad[['target_acc','antibiotic']].head().to_dict('records')))
    m=frozen[frozen.antibiotic.eq('meropenem')].copy()
    if len(m)!=EXPECTED_N or int(m.label_binary.sum())!=EXPECTED_POS:
        die(f'Expected frozen 525 meropenem genomes with 163 resistant. Got n={len(m)}, resistant={m.label_binary.sum()}')
    if not m.asm_acc.notna().all() or m.asm_acc.duplicated().any() or m.target_acc.duplicated().any():die('Frozen meropenem genome keys missing/duplicate')
    if not (m.label_binary.astype(int)==m.phenotype.eq('resistant').astype(int)).all():die('Frozen meropenem label_binary conflicts with phenotype')
    pp=preds[preds.cohort.astype(str).str.upper().eq('E2')].copy()
    gg=genes[genes.cohort.astype(str).str.upper().eq('E2')].copy()
    if pp.genome_id.duplicated().any() or gg.genome_id.duplicated().any():die('Prediction or gene audit has duplicate E2 genome ID')
    if set(pp.genome_id)!=set(m.asm_acc) or set(gg.genome_id)!=set(m.asm_acc):
        die('Frozen E2 asm_acc does not match exactly E2 prediction and gene-audit genome_id')
    if not pd.to_numeric(pp.prob_resistance,errors='coerce').between(0,1).all():die('E2 probabilities invalid')
    predfields=['genome_id','y_true','prob_resistance']
    if 'error_type' in pp:predfields.append('error_type')
    if 'high_conf_wrong' in pp:predfields.append('high_conf_wrong')
    pg=m.merge(pp[predfields],left_on='asm_acc',right_on='genome_id',validate='one_to_one')
    if not (pg.label_binary.astype(int)==pg.y_true.astype(int)).all():die('Prediction label disagrees with frozen cohort')
    gene_cols=['genome_id','mechanism_group']
    for x in ['detected_major_element_symbols','all_element_symbols','has_CTX_M','has_porin_related_amrfinder_symbol']:
        if x in gg:gene_cols.append(x)
    pg=pg.merge(gg[gene_cols],on='genome_id',validate='one_to_one')
    if len(pg)!=EXPECTED_N:die('Unanticipated merge cardinality change')
    pg['has_major_panel']=~pg.mechanism_group.eq('No_detected_major_family')
    pg['pred_resistant']=(pg.prob_resistance>=.5).astype(int)
    pg['wrong']=pg.pred_resistant.ne(pg.y_true)
    pg['high_conf_wrong_recalc']=pg.wrong & np.maximum(pg.prob_resistance,1-pg.prob_resistance).ge(.9)
    if pg.wrong.sum()!=195 or int(pg.high_conf_wrong_recalc.sum())!=128:
        die(f'FILE16 frozen checkpoints mismatch: wrong={pg.wrong.sum()} (expected195), high-conf-wrong={pg.high_conf_wrong_recalc.sum()} (expected128)')
    if 'error_type' in pg:
        derived=np.where(~pg.wrong,'CORRECT',np.where(pg.y_true.eq(1),'FALSE_SUSCEPTIBLE','FALSE_RESISTANT'))
        if not (derived==pg.error_type.to_numpy()).all():die('error_type mismatch with actual frozen predictions')
    # Secondary AST cross-antibiotic, source records: pivot by unique target_acc+antibiotic.
    sub_raw=raw_unique[raw_unique.target_acc.isin(m.target_acc)].copy()
    pivot=sub_raw.pivot(index='target_acc',columns='antibiotic',values='phenotype')
    for ab in ['ertapenem','imipenem','meropenem']:
        if ab not in pivot.columns:pivot[ab]=pd.NA
    pivot=pivot[['ertapenem','imipenem','meropenem']].rename(columns=lambda x:'ast_'+x)
    pg=pg.merge(pivot.reset_index(),on='target_acc',validate='one_to_one')
    if (pg.ast_meropenem != pg.phenotype).any():die('AST meropenem pivot label mismatch')
    pg['source_group']=np.where(pg.bioproject_acc.eq(args.focus_project),args.focus_project,'All_other_BioProjects')
    if not pg.bioproject_acc.eq(args.focus_project).any():die('Focus BioProject not present: '+args.focus_project)
    # Main, focus, and remainder summaries; do not replace main E2 headline.
    rows=[]
    for name,sub in [('E2_all_frozen',pg),(args.focus_project,pg[pg.source_group.eq(args.focus_project)]),('All_other_BioProjects',pg[pg.source_group.eq('All_other_BioProjects')])]:
        row={'source_group':name,'post_hoc':True,**metrics(sub)}
        if args.bootstrap:row['auroc_erd_bootstrap_ci95']=bootstrap_auc(sub,args.bootstrap,seed=42)
        rows.append(row)
    source_summary=pd.DataFrame(rows)
    # Show all projects, no cherry-picked omission; no performance filtering.
    project_rows=[]
    for project,sub in pg.groupby('bioproject_acc',dropna=False):
        project_rows.append({'bioproject_acc':project,'post_hoc':True,**metrics(sub), 'n_erd_groups':int(sub.erd_group.nunique(dropna=True)),
         'n_biosamples':int(sub.biosample_acc.nunique(dropna=True))})
    project_df=pd.DataFrame(project_rows).sort_values(['n','bioproject_acc'],ascending=[False,True])
    # Gene-phenotype association by source, Fisher p is only exploratory, ignores dependence.
    pairs=[]
    for name,sub in [('E2_all_frozen',pg),(args.focus_project,pg[pg.source_group.eq(args.focus_project)]),('All_other_BioProjects',pg[pg.source_group.eq('All_other_BioProjects')])]:
        pairs.append({'source_group':name,'n':len(sub),**two_by_two(sub)})
    gpa=pd.DataFrame(pairs)
    # Sibling-drug AST concordance: list every genome regardless of available siblings.
    ast=pg[['target_acc','asm_acc','biosample_acc','bioproject_acc','erd_group','phenotype','ast_meropenem','ast_imipenem','ast_ertapenem','mechanism_group','source_group']].copy()
    ast['imipenem_disagrees_meropenem_when_binary']=ast.ast_imipenem.isin(['resistant','susceptible']) & ast.ast_imipenem.ne(ast.ast_meropenem)
    ast['ertapenem_disagrees_meropenem_when_binary']=ast.ast_ertapenem.isin(['resistant','susceptible']) & ast.ast_ertapenem.ne(ast.ast_meropenem)
    ast['missing_imipenem']=ast.ast_imipenem.isna();ast['missing_ertapenem']=ast.ast_ertapenem.isna()
    # ERD concentration for discordance and errors.
    concentration=[]
    for source,sub in pg.groupby('source_group'):
        for label,ss in [('KPC_family_susceptible',sub[sub.mechanism_group.eq('KPC_family') & sub.y_true.eq(0)]),
                         ('panel_positive_susceptible',sub[sub.has_major_panel & sub.y_true.eq(0)]),
                         ('panel_negative_resistant',sub[~sub.has_major_panel & sub.y_true.eq(1)]),
                         ('false_susceptible',sub[(sub.y_true.eq(1)) & (sub.pred_resistant.eq(0))]),
                         ('false_resistant',sub[(sub.y_true.eq(0)) & (sub.pred_resistant.eq(1))])]:
            counts=ss.erd_group.dropna().astype(str).str.strip().value_counts()
            counts=counts[counts.index!='']
            concentration.append({'source_group':source,'stratum':label,'n':len(ss),'n_with_known_erd':int(counts.sum()),
              'n_distinct_erd':len(counts),'max_erd_n':int(counts.iloc[0]) if len(counts) else 0,
              'max_erd_group':counts.index[0] if len(counts) else '',
              'top3_erd_count':int(counts.iloc[:3].sum()),
              'warning':'ERD group is not a genome-distance proof; do not treat correlated genomes as independent.'})
    concentration_df=pd.DataFrame(concentration)
    # Missing numeric AST / MIC in supplied raw source, not a fake source validation.
    measurement_keywords=['mic','minimum_inhibitory','zone_diameter','disk_diameter','breakpoint','ast_method','testing_method','ast_standard']
    available_measure_cols=[x for x in raw.columns if any(w in x.lower() for w in measurement_keywords)]
    # triage list prioritizes the suspicious source but preserves other sources as controls.
    triage=pg.copy()
    triage['triage_category']=np.select([
       triage.has_major_panel & triage.y_true.eq(0),
       ~triage.has_major_panel & triage.y_true.eq(1),
       triage.wrong],['major_panel_detected_but_susceptible',
                     'no_major_panel_detected_but_resistant',
                     'other_frozen_prediction_error'],default='control_case')
    triage['triage_priority']=(triage.source_group.eq(args.focus_project).astype(int)*10
                               +triage.wrong.astype(int)*4
                               +triage.high_conf_wrong_recalc.astype(int)*3
                               +triage.triage_category.isin(['major_panel_detected_but_susceptible','no_major_panel_detected_but_resistant']).astype(int)*2)
    triage['biosample_url']='https://www.ncbi.nlm.nih.gov/biosample/?term='+canonical(triage.biosample_acc)
    triage['assembly_url']='https://www.ncbi.nlm.nih.gov/assembly/?term='+canonical(triage.asm_acc)
    triage['bioproject_url']='https://www.ncbi.nlm.nih.gov/bioproject/?term='+canonical(triage.bioproject_acc)
    cols=['triage_priority','triage_category','source_group','target_acc','asm_acc','biosample_acc','bioproject_acc','erd_group',
     'phenotype','ast_meropenem','ast_imipenem','ast_ertapenem','mechanism_group','y_true','prob_resistance','pred_resistant','wrong','high_conf_wrong_recalc',
     'biosample_url','assembly_url','bioproject_url']
    for extra in ['detected_major_element_symbols','all_element_symbols','collection_date','geo_loc_name','isolation_source']:
        if extra in triage.columns:cols.append(extra)
    triage=triage[cols].sort_values(['triage_priority','bioproject_acc','asm_acc'],ascending=[False,True,True])
    # Release outputs only after all consistency checks pass.
    unique_output(out)
    output_paths={
        'file16d_source_performance.csv':source_summary,
        'file16d_all_bioprojects.csv':project_df,
        'file16d_panel_phenotype_inversion.csv':gpa,
        'file16d_multidrug_AST_reconciliation.csv':ast,
        'file16d_group_concentration.csv':concentration_df,
        'file16d_manual_review_priority.csv':triage,
    }
    for filename,df in output_paths.items():df.to_csv(out/filename,index=False)
    result={
       'status':'POST_HOC_SOURCE_AUDIT_NOT_AST_GROUND_TRUTH',
       'generated_utc':datetime.now(timezone.utc).isoformat(),
       'project_root':str(root),'focus_bioproject':args.focus_project,
       'constraints':{'no_model_training':True,'no_threshold_change':True,'no_source_exclusion_in_primary_E2':True,'no_tuning':True},
       'input_files':{k:{'path':str(v),'sha256':sha256_file(v),'bytes':v.stat().st_size} for k,v in sources.items()},
       'raw_ast_rows':len(raw),'raw_exact_duplicate_rows':n_raw_dup,
       'frozen_all_antibiotics_rows':len(frozen),'frozen_raw_field_checks':checks,
       'frozen_meropenem_rows':len(m),'predictions_and_genes_joined':len(pg),
       'all_frozen_e2_metrics':metrics(pg),
       'focus_source_metrics':metrics(pg[pg.bioproject_acc.eq(args.focus_project)]),
       'other_sources_metrics':metrics(pg[pg.bioproject_acc.ne(args.focus_project)]),
       'ast_source_numeric_mic_columns':available_measure_cols,
       'has_primary_ast_mic_breakpoint_data_in_provided_raw':bool(available_measure_cols),
       'warnings':[
           'A source-specific AUROC increase after excluding BioProject is descriptive; do NOT replace published/full frozen E2 AUROC.',
           'Phenotype agreement between two pipeline CSVs does not validate original laboratory AST or MIC.',
           'Strong genotype-phenotype inversion can reflect identifier mapping, source record semantics, testing standards or biology; do not assume label reversal.',
           'Use nucleotide-level relatedness and original AST metadata to establish independence and source validity.',
           'Fisher test ignores population structure; p values are exploratory and are not corrected for ascertainment or clonality.',
           'No inference about E3 or external generalizability is made by this E2 source audit.',
           'This analysis was designed post-hoc in response to an observed anomaly.'
       ]
    }
    (out/'file16d_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'output_sha256':{p.name:sha256_file(p) for p in sorted(out.glob('*'))},'no_existing_input_modified':True}
    (out/'file16d_output_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({
        'status':result['status'],'raw_records':len(raw),'frozen_all_drug_records':len(frozen),
        'matched_E2_genomes':len(pg),'focus_bioproject':args.focus_project,
        'focus_n':result['focus_source_metrics']['n'],'focus_auroc':result['focus_source_metrics']['auroc'],
        'other_n':result['other_sources_metrics']['n'],'other_auroc':result['other_sources_metrics']['auroc'],
        'MIC_columns_in_original_source':available_measure_cols,
        'outdir':str(out)},indent=2))

if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,KeyError,FileNotFoundError) as exc:
        print('[FILE16D ERROR]',str(exc),file=sys.stderr);sys.exit(2)
