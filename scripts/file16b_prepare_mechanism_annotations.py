#!/usr/bin/env python3
"""FILE16B: read-only annotation linking for the locked E2/E3 prediction cohorts.

Produces one-row-per-genome exploratory, MUTUALLY EXCLUSIVE *major-family* labels
from EXISTING AMRFinderPlus TSV outputs; no training, threshold selection, or edits.

Usage from project root (WSL):
  python scripts/file16b_prepare_mechanism_annotations.py \
    --project-root . --outdir results/file16b_annotation_v1

IMPORTANT: These are coarse major carbapenemase-family strata, NOT clinical
phenotype rules and NOT all known biochemical mechanisms of resistance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import pandas as pd

E2_INPUT = 'results/file16_inputs_v1/file16_e2_predictions_standardized.csv'
E3_INPUT = 'results/file16_inputs_v1/file16_e3_predictions_standardized.csv'
E2_TSV = 'checkpoints/external_validation_file11c/amrfinder'
E3_TSV = 'data/external_validation/e3/file14_final/raw_amrfinder'
OXA48_LIKE = {48,162,163,181,204,232,244,245,370,405,438,484,505,517,519,535,537}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def family_of(symbol):
    """Conservative family catalog, anchored to AMRFinderPlus Element symbol."""
    s=str(symbol).strip().upper()
    s=re.sub(r'^BLA[ _-]*','',s)  # blaKPC-2 -> KPC-2
    for name in ('KPC','NDM','VIM','IMP'):
        if re.match(r'^'+name+r'(?:$|[-_0-9])',s):return name
    m=re.match(r'^OXA[-_]?([0-9]+)(?:$|[^0-9])',s)
    if m and int(m.group(1)) in OXA48_LIKE:return 'OXA48_like'
    return None


def indexed_tsvs(directory, cohort):
    if not directory.is_dir():raise FileNotFoundError(f'{cohort}: missing AMRFinder directory: {directory}')
    paths=sorted(directory.glob('*.tsv'))
    if not paths:raise ValueError(f'{cohort}: no TSVs under {directory}')
    out={}
    for p in paths:
        if cohort=='E2':
            if not p.name.endswith('.amrfinder.tsv'):
                raise ValueError(f'E2 unexpected TSV filename: {p.name}')
            gid=p.name[:-len('.amrfinder.tsv')]
        else:
            if '__' not in p.stem:raise ValueError(f'E3 unexpected TSV filename: {p.name}')
            gid=p.stem.split('__',1)[0]
        if not gid or gid in out:raise ValueError(f'{cohort}: empty or duplicated TSV key {gid!r}')
        out[gid]=p
    return out


def group_from_families(fams):
    if len(fams)==0:return 'No_detected_major_family'
    if len(fams)>1:return 'Multiple_major_families'
    return {'KPC':'KPC_family','NDM':'NDM_family','OXA48_like':'OXA48_like_family',
            'VIM':'VIM_family','IMP':'IMP_family'}[next(iter(fams))]


def annotate(cohort, predictions, tsv_dir, expected_n, expected_r):
    d=pd.read_csv(predictions,dtype={'genome_id':'string'},low_memory=False)
    needed=['genome_id','y_true','prob_resistance']
    if any(c not in d for c in needed):raise ValueError(f'{cohort} missing input columns {needed}')
    ids=d.genome_id.astype('string').str.strip()
    if len(d)!=expected_n or int(d.y_true.sum())!=expected_r or ids.isna().any() or ids.duplicated().any():
        raise ValueError(f'{cohort} locked predictions count/labels/IDs failed verification')
    index=indexed_tsvs(tsv_dir,cohort)
    s=set(ids.tolist());t=set(index)
    if s!=t:
        raise ValueError(f'{cohort}: TSV/prediction ID mismatch: prediction IDs without TSV={sorted(s-t)[:8]}; '
                         f'TSV IDs outside frozen cohort={sorted(t-s)[:8]}; totals {len(s)} vs {len(t)}')
    annotations=[]; details=[]; counter=Counter()
    for gid in ids:
        p=index[gid]
        # sep=tab; empty (0-byte) files are rejected rather than treated as negative.
        if p.stat().st_size==0:raise ValueError(f'{cohort}: empty raw TSV {p}')
        tab=pd.read_csv(p,sep='\t',dtype='string',keep_default_na=False,low_memory=False)
        if 'Element symbol' not in tab.columns:raise ValueError(f'{cohort}: missing Element symbol in {p}; fields={tab.columns.tolist()}')
        fams=set()
        symbols=set()
        for v in tab['Element symbol'].tolist():
            fam=family_of(v)
            if fam:
                fams.add(fam)
                symbols.add(str(v).strip())
        g=group_from_families(fams)
        counter[g]+=1
        annotations.append({'genome_id':gid,'mechanism_group':g})
        details.append({'cohort':cohort,'genome_id':gid,'mechanism_group':g,
                        'detected_major_families':';'.join(sorted(fams)),
                        'detected_major_element_symbols':';'.join(sorted(symbols)),
                        'n_amrfinder_rows':len(tab),'raw_tsv':str(p),
                        'raw_tsv_sha256':sha256(p)})
    if sum(v for k,v in counter.items() if k!='No_detected_major_family')==0:
        raise ValueError(f'{cohort}: found zero major carbapenemase families; verify TSV parser and nomenclature')
    return pd.DataFrame(annotations),pd.DataFrame(details),dict(sorted(counter.items()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--outdir',required=True,help='New output directory, must not exist')
    args=ap.parse_args()
    root=Path(args.project_root).expanduser().resolve()
    out=Path(args.outdir).expanduser().resolve()
    if out.exists():raise ValueError(f'Refusing to overwrite pre-existing output: {out}')
    for rel in (E2_INPUT,E3_INPUT):
        if not (root/rel).is_file():raise FileNotFoundError(root/rel)
    a2,d2,c2=annotate('E2',root/E2_INPUT,root/E2_TSV,525,163)
    a3,d3,c3=annotate('E3',root/E3_INPUT,root/E3_TSV,306,122)
    out.mkdir(parents=True,exist_ok=False)
    a2.to_csv(out/'file16b_e2_mechanism_annotations.csv',index=False)
    a3.to_csv(out/'file16b_e3_mechanism_annotations.csv',index=False)
    pd.concat([d2,d3],ignore_index=True).to_csv(out/'file16b_family_annotation_evidence.csv',index=False)
    report={'date_utc':datetime.now(timezone.utc).isoformat(),
            'approach':'post-hoc exploratory; no model training/tuning; biologically defined broad family bins',
            'e2_input_sha256':sha256(root/E2_INPUT),'e3_input_sha256':sha256(root/E3_INPUT),
            'e2_count':len(a2),'e3_count':len(a3),'e2_groups':c2,'e3_groups':c3,
            'caution':['Negative means no member of THIS restricted five-family panel was detected; it does not prove carbapenemase absence.',
                       'Genotype family labels do not prove a causal resistance mechanism.',
                       'Multiple-family group has priority over single-family groups.',
                       'Observed cross-cohort concordance would still be exploratory, not prospective replication.',
                       'Different annotation sources/pipelines may yield differential detection; compare protocol metadata.']}
    (out/'file16b_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print('FILE16B PREPARED:',out)

if __name__=='__main__':
    main()
