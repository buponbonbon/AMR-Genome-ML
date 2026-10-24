#!/usr/bin/env python3
"""FILE16C: READ-ONLY genotype/phenotype-discordance and AMRFinderPlus evidence audit.

Run from the AMR-Genome-ML project root in WSL:
 python scripts/file16c_discordance_audit.py --project-root . \
    --outdir results/file16c_discordance_v1

Uses frozen FILE16 predictions and FILE16B evidence. Checks every raw annotation
TSV against its prior SHA-256; never trains/updates models, calibrates thresholds,
or modifies source data. Post-hoc exploratory observations are NOT causal claims.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

REL_AUDIT = 'results/file16_audit_mechanisms_v1/file16_per_genome_prediction_error_audit.csv'
REL_EVIDENCE = 'results/file16b_annotation_v1/file16b_family_annotation_evidence.csv'
EXPECTED = {'E2': (525, 163), 'E3': (306, 122)}
OXA48_LIKE = {48,162,163,181,204,232,244,245,370,405,438,484,505,517,519,535,537}
# Conservative GENE-LEVEL descriptive categories, not a biochemical mechanism classifier.
AMPC_PREFIX = ('CMY','DHA','FOX','MOX','ACC','ACT','MIR','LAT','CFE','PDC','ADC','EBC')
OTHER_CARBAPENEMASE_CANDIDATES = ('SME','IMI','NMC','FRI','SFC','BIC','GIM','SIM','DIM','AIM','SPM','KHM','TMB')


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def clean_sym(value: object) -> str:
    s=str(value or '').strip()
    if s.lower() in ('nan','none','<na>'):return ''
    return s


def gene_base(symbol: str) -> str:
    s=clean_sym(symbol).upper().replace(' ','')
    return re.sub(r'^BLA[ _-]*','',s)


def starts_gene(symbol: str, gene: str) -> bool:
    s=gene_base(symbol)
    return bool(re.match(r'^'+re.escape(gene)+r'(?:$|[-_0-9])',s))


def family_of(symbol):
    s=gene_base(symbol)
    for name in ('KPC','NDM','VIM','IMP'):
        if re.match(r'^'+name+r'(?:$|[-_0-9])',s):return name
    m=re.match(r'^OXA[-_]?([0-9]+)(?:$|[^0-9])',s)
    if m and int(m.group(1)) in OXA48_LIKE:return 'OXA48_like'
    return None


def grouping_of(fams):
    if not fams:return 'No_detected_major_family'
    if len(fams)>1:return 'Multiple_major_families'
    return {'KPC':'KPC_family','NDM':'NDM_family','OXA48_like':'OXA48_like_family',
            'VIM':'VIM_family','IMP':'IMP_family'}[next(iter(fams))]


def column(d, *names):
    norm=lambda s:re.sub(r'[^a-z0-9]','',str(s).lower())
    lookup={norm(c):c for c in d.columns}
    for name in names:
        if norm(name) in lookup:return lookup[norm(name)]
    return None


def parse_percentage(value):
    if pd.isna(value):return np.nan
    s=str(value).strip().replace('%','')
    try:return float(s)
    except ValueError:return np.nan


def load_tsv(p):
    if p.stat().st_size==0:raise ValueError(f'Empty AMRFinderPlus TSV: {p}')
    try:
        df=pd.read_csv(p,sep='\t',dtype='string',keep_default_na=False,low_memory=False)
    except Exception as exc:
        raise ValueError(f'Could not parse AMRFinderPlus TSV {p}: {exc}') from exc
    symbol=column(df,'Element symbol','Gene symbol')
    if not symbol:
        raise ValueError(f'Missing Element symbol column: {p}; columns={list(df.columns)}')
    return df,symbol


def normalize_input(root: Path, audit_file: Path, evidence_file: Path):
    for p in (audit_file,evidence_file):
        if not p.is_file():raise FileNotFoundError(p)
    audit=pd.read_csv(audit_file,dtype={'genome_id':'string'},low_memory=False)
    evid=pd.read_csv(evidence_file,dtype={'genome_id':'string'},low_memory=False)
    audit_needed={'cohort','genome_id','y_true','prob_resistance','error_type','mechanism_group'}
    evid_needed={'cohort','genome_id','mechanism_group','raw_tsv','raw_tsv_sha256','n_amrfinder_rows'}
    if audit_needed-set(audit):raise ValueError(f'Audit lacks {audit_needed-set(audit)}')
    if evid_needed-set(evid):raise ValueError(f'Evidence lacks {evid_needed-set(evid)}')
    for name,d in [('audit',audit),('evidence',evid)]:
        if d[['cohort','genome_id']].isna().any().any() or d[['cohort','genome_id']].duplicated().any():
            raise ValueError(f'{name}: missing/duplicate (cohort, genome_id)')
    joined=audit.merge(evid,on=['cohort','genome_id'],how='outer',validate='one_to_one',
                       suffixes=('_audit','_evidence'),indicator=True)
    if not (joined['_merge']=='both').all():
        raise ValueError('Prediction/evidence IDs do not match: '+str(joined._merge.value_counts().to_dict()))
    if (joined.mechanism_group_audit!=joined.mechanism_group_evidence).any():
        raise ValueError('FILE16 and FILE16B mechanism groups do not match')
    joined=joined.drop(columns=['_merge'])
    if set(joined.cohort)!=set(EXPECTED):raise ValueError('Cohort labels must be E2 and E3')
    for name,(n,nr) in EXPECTED.items():
        d=joined[joined.cohort==name]
        if len(d)!=n or int(d.y_true.sum())!=nr or not d.y_true.isin([0,1]).all():
            raise ValueError(f'{name} frozen cohort mismatch: got n={len(d)}, R={d.y_true.sum()}')
    if not joined.prob_resistance.between(0,1).all():raise ValueError('Invalid probability outside [0,1]')
    for x in joined.itertuples(index=False):
        expected_label='CORRECT' if (int(x.prob_resistance>=0.5)==int(x.y_true)) else ('FALSE_SUSCEPTIBLE' if int(x.y_true)==1 else 'FALSE_RESISTANT')
        if str(x.error_type)!=expected_label:raise ValueError(f'Prediction error mismatch for {x.genome_id}')
    # Important: 831 evidence rows must refer to existing on-disk TSVs inside project.
    return joined


def inspect_one(row,root):
    rp=Path(str(row.raw_tsv)).expanduser()
    if not rp.is_absolute():rp=root/rp
    rp=rp.resolve()
    if not rp.is_relative_to(root):raise ValueError(f'Annotation path outside project root: {rp}')
    if not rp.is_file():raise FileNotFoundError(f'Raw AMRFinderPlus TSV missing: {rp}')
    computed=sha256(rp)
    if computed.lower()!=str(row.raw_tsv_sha256).lower():
        raise ValueError(f'Annotation TSV SHA256 changed since FILE16B: {rp}')
    tab,symbolcol=load_tsv(rp)
    if len(tab)!=int(row.n_amrfinder_rows):
        raise ValueError(f'Annotation row count changed since FILE16B: {rp}')
    syms=sorted({clean_sym(x) for x in tab[symbolcol].tolist() if clean_sym(x)})
    fams={f for s in syms if (f:=family_of(s))}
    if grouping_of(fams)!=str(row.mechanism_group_evidence):
        raise ValueError(f'Annotation family disagrees with FILE16B: {row.cohort}/{row.genome_id}')
    covers=column(tab,'% Coverage of reference sequence','Coverage of reference sequence','% Coverage')
    idents=column(tab,'% Identity to reference sequence','Identity to reference sequence','% Identity')
    kpc_idx=[i for i,v in enumerate(tab[symbolcol].astype(str).tolist()) if family_of(v)=='KPC']
    kpc_rows=tab.iloc[kpc_idx]
    def vector_max(col):
        if not col or len(kpc_rows)==0:return np.nan
        vals=[parse_percentage(v) for v in kpc_rows[col]]
        return float(np.nanmax(vals)) if any(np.isfinite(v) for v in vals) else np.nan
    has_ctx=any(starts_gene(s,'CTX-M') for s in syms)
    has_ampc=any(any(starts_gene(s,z) for z in AMPC_PREFIX) for s in syms)
    has_shv=any(starts_gene(s,'SHV') for s in syms)
    has_tem=any(starts_gene(s,'TEM') for s in syms)
    has_other=any(any(starts_gene(s,z) for z in OTHER_CARBAPENEMASE_CANDIDATES) for s in syms)
    has_oxa_other=any(gene_base(s).startswith('OXA-') and family_of(s)!='OXA48_like' for s in syms)
    porin=[s for s in syms if re.search(r'(?:OMPK35|OMPK36|OMPK37)',gene_base(s))]
    # These gene-presence flags cannot resolve protein expression, porin loss,
    # breakpoint differences, MIC, AST quality, or causality.
    return {'cohort':row.cohort,'genome_id':row.genome_id,
            'mechanism_group':row.mechanism_group_audit,
            'y_true':int(row.y_true),'prob_resistance':float(row.prob_resistance),
            'error_type':row.error_type,'high_conf_wrong':bool(row.high_conf_wrong),
            'erd_group':str(row.erd_group) if pd.notna(row.erd_group) else '',
            'mlst_group':str(row.mlst_group) if pd.notna(row.mlst_group) else '',
            'detected_major_element_symbols':row.detected_major_element_symbols if pd.notna(row.detected_major_element_symbols) else '',
            'n_amrfinder_rows':int(row.n_amrfinder_rows),'n_distinct_element_symbols':len(syms),
            'all_element_symbols':';'.join(syms),
            'has_CTX_M':has_ctx,'has_candidate_AmpC_gene_family':has_ampc,
            'has_SHV_family':has_shv,'has_TEM_family':has_tem,
            'has_other_candidate_carbapenemase_family':has_other,
            'has_other_OXA_family_symbol':has_oxa_other,
            'has_porin_related_amrfinder_symbol':bool(porin),
            'porin_related_symbols':';'.join(porin),
            'KPC_call_max_reference_coverage_pct':vector_max(covers),
            'KPC_call_max_reference_identity_pct':vector_max(idents),
            'raw_tsv':str(rp),'raw_tsv_sha256':computed}


def compare_enrichment(df):
    """Generate *screening* frequencies only, no significance claims.
    A cohort-conditioned comparison allows biological characterization without
    misleading p-values when groups have tiny effective sample sizes.
    """
    tags=['has_CTX_M','has_candidate_AmpC_gene_family','has_SHV_family','has_TEM_family',
          'has_other_candidate_carbapenemase_family','has_other_OXA_family_symbol',
          'has_porin_related_amrfinder_symbol']
    out=[]
    for cohort in EXPECTED:
        z=df[df.cohort==cohort]
        strata={
          'No_major__resistant_FN':z[(z.mechanism_group=='No_detected_major_family')&(z.y_true==1)&(z.error_type=='FALSE_SUSCEPTIBLE')],
          'No_major__resistant_TP':z[(z.mechanism_group=='No_detected_major_family')&(z.y_true==1)&(z.error_type=='CORRECT')],
          'No_major__susceptible':z[(z.mechanism_group=='No_detected_major_family')&(z.y_true==0)],
          'KPC__susceptible':z[(z.mechanism_group=='KPC_family')&(z.y_true==0)],
          'KPC__resistant':z[(z.mechanism_group=='KPC_family')&(z.y_true==1)],
        }
        for group,x in strata.items():
            for tag in tags:
                k=int(x[tag].astype(bool).sum())
                out.append({'cohort':cohort,'stratum':group,'feature':tag,'n':len(x),'present_n':k,
                            'present_fraction':k/len(x) if len(x) else np.nan,
                            'interpretation':'DESCRIPTIVE ONLY; symbols not causal; subgroup sizes may be small'})
    return pd.DataFrame(out)


def summarize(df):
    out=[]
    for cohort in EXPECTED:
        for group in ['KPC_family','No_detected_major_family']:
            z=df[(df.cohort==cohort)&(df.mechanism_group==group)]
            S=z[z.y_true==0];R=z[z.y_true==1]
            out.append({'cohort':cohort,'mechanism_group':group,'n_total':len(z),
                'n_resistant':len(R),'n_susceptible':len(S),
                'false_susceptible_n':int((R.error_type=='FALSE_SUSCEPTIBLE').sum()),
                'false_resistant_n':int((S.error_type=='FALSE_RESISTANT').sum()),
                'high_conf_false_susceptible_n':int(((R.error_type=='FALSE_SUSCEPTIBLE')&(R.high_conf_wrong)).sum()),
                'high_conf_false_resistant_n':int(((S.error_type=='FALSE_RESISTANT')&(S.high_conf_wrong)).sum()),
                'n_erd_groups_known_susceptible':int(S.erd_group.replace('',pd.NA).nunique(dropna=True)),
                'n_mlst_groups_known_susceptible':int(S.mlst_group.replace('',pd.NA).nunique(dropna=True)),
                'n_erd_groups_known_resistant':int(R.erd_group.replace('',pd.NA).nunique(dropna=True)),
                'n_mlst_groups_known_resistant':int(R.mlst_group.replace('',pd.NA).nunique(dropna=True))})
    return pd.DataFrame(out)


def grouped_summary(df):
    output=[]
    for cohort in EXPECTED:
        x=df[df.cohort==cohort]
        for label,q in [('KPC_susceptible', x[(x.mechanism_group=='KPC_family')&(x.y_true==0)]),
                        ('No_major_resistant',x[(x.mechanism_group=='No_detected_major_family')&(x.y_true==1)])]:
            for column in ['erd_group','mlst_group']:
                sub=q[q[column].astype(str).str.len()>0]
                counts=sub[column].value_counts()
                output.append({'cohort':cohort,'target':label,'grouping':column,'n_cases':len(q),
                    'n_with_resolved_group':len(sub),'n_distinct_groups':len(counts),
                    'largest_group_n':int(counts.iloc[0]) if len(counts) else 0,
                    'largest_group_fraction':float(counts.iloc[0]/len(q)) if len(counts) and len(q) else np.nan,
                    'top_group_id':str(counts.index[0]) if len(counts) else '',
                    'top3_groups':' | '.join(f'{key} ({val})' for key,val in counts.head(3).items()),
                    'warning':'Group ID metadata coverage can be incomplete; no sequence-distance audit performed.'})
    return pd.DataFrame(output)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--audit-csv',default=REL_AUDIT)
    ap.add_argument('--evidence-csv',default=REL_EVIDENCE)
    ap.add_argument('--outdir',required=True,help='New directory, must not pre-exist')
    args=ap.parse_args()
    root=Path(args.project_root).expanduser().resolve()
    def source(v):
        p=Path(v).expanduser()
        return (root/p).resolve() if not p.is_absolute() else p.resolve()
    audit=source(args.audit_csv); evid=source(args.evidence_csv)
    out=source(args.outdir)
    if out.exists():raise ValueError(f'REFUSING TO OVERWRITE: {out}')
    for p in [audit,evid]:
        if not p.is_relative_to(root):raise ValueError(f'Input outside project root: {p}')
    if not out.is_relative_to(root):raise ValueError('Output must stay within project root')
    data=normalize_input(root,audit,evid)
    print(f'[FILE16C] Found {len(data)} matched genomes; checking 831 frozen AMRFinderPlus TSV hashes...',flush=True)
    results=[]
    for i,row in enumerate(data.itertuples(index=False),1):
        results.append(inspect_one(row,root))
        if i%100==0:print(f'[FILE16C] verified {i}/{len(data)} TSVs',flush=True)
    df=pd.DataFrame(results)
    report=summarize(df);freq=compare_enrichment(df);clusters=grouped_summary(df)
    kpcS=df[(df.mechanism_group=='KPC_family')&(df.y_true==0)].copy()
    no_major_R=df[(df.mechanism_group=='No_detected_major_family')&(df.y_true==1)].copy()
    # Columns explicitly include genotype/phenotype status and gene-level checks.
    summary={'status':'POST_HOC_EXPLORATORY_NOT_CAUSAL',
         'date_utc':datetime.now(timezone.utc).isoformat(),'no_training':True,
         'frozen_threshold_changed':False,
         'e2_n':int((df.cohort=='E2').sum()),'e3_n':int((df.cohort=='E3').sum()),
         'raw_amrfinder_tsv_sha256_verified':len(df),
         'input_sha256':{'file16_prediction_audit':sha256(audit), 'file16b_evidence':sha256(evid)},
         'kpc_susceptible_total':len(kpcS),
         'no_major_resistant_total':len(no_major_R),
         'no_major_resistant_false_susceptible_total':int((no_major_R.error_type=='FALSE_SUSCEPTIBLE').sum()),
         'limitations':[
          'All results post-hoc; not evidence of novel causal mechanism.',
          'Five-family panel absence is not absence of all carbapenemases.',
          'AMRFinderPlus gene calls cannot establish gene expression, porin loss, or AST phenotype accuracy.',
          'The provided E2/E3 model artifacts are different; cross-cohort differences can reflect model and cohort.',
          'E3 ERD and MLST metadata sparse; no genomic-distance independence claim.',
          'No AST/MIC/breakpoint source records were supplied or evaluated by this script.',
          'No significance testing of genome-level gene strata is claimed because samples may share ancestry.',
          'Any positive follow-up requires phenotype/AST reconciliation and genomic relatedness checks.'
         ]}
    # ONLY write after all 831 SHA/row-count checks pass.
    out.mkdir(parents=True,exist_ok=False)
    df.to_csv(out/'file16c_per_genome_gene_audit.csv',index=False)
    kpcS.to_csv(out/'file16c_KPC_susceptible_cases.csv',index=False)
    no_major_R.to_csv(out/'file16c_no_major_resistant_cases.csv',index=False)
    report.to_csv(out/'file16c_discordance_summary.csv',index=False)
    freq.to_csv(out/'file16c_gene_family_descriptives.csv',index=False)
    clusters.to_csv(out/'file16c_group_concentration.csv',index=False)
    (out/'file16c_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('[FILE16C] COMPLETED:',out)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[FILE16C ERROR] {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
