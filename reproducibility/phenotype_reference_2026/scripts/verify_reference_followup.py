#!/usr/bin/env python3
"""Independent checks for the follow-up's quantitative screen, run links and error bounds."""
import argparse
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd


def number(raw):
    value=''.join(str(raw).split()).replace('≤','<=').replace('≥','>=')
    m=re.fullmatch(r'([<>=]*)([0-9]+(?:\.[0-9]+)?)',value)
    if not m or Decimal(m[2])<=0:return None
    return ('=' if m[1] in {'','==','='} else m[1]),Decimal(m[2])


def native_mic(row):
    unit=unicodedata.normalize('NFKC',row.get('measurement_unit','')).strip().casefold()
    method=row.get('laboratory_typing_method','').casefold()
    if unit not in {'mg/l','ug/ml','μg/ml'} or 'disk' in method or 'kirby' in method:return None
    raw=number(row.get('measurement',''));sign=row.get('measurement_sign','').strip();value=row.get('measurement_value','').strip()
    parts=number(sign+value) if sign in {'=','==','>','>=','<','<='} else None
    if raw and parts:
        bare=re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',row.get('measurement','').strip())
        if bare and number(value)==raw:return parts
        if raw!=parts:return None
    if raw:
        v=number(value)
        if v is not None and v[1]!=raw[1]:return None
        return raw
    return parts


def output_mic(raw):
    lo,hi,lc,uc=raw.split('|');lo,hi=Decimal(lo),Decimal(hi)
    if lo==hi:return '=',lo
    if lo==0:return ('<=' if uc=='True' else '<'),hi
    if hi.is_infinite():return ('>=' if lc=='True' else '>'),lo
    raise ValueError('Unexpected interval')


def label(value):
    x=value.strip().casefold()
    if x in {'','-','na','n/a','nan','none','null','missing'}:return 'MISSING_CATEGORY'
    return {'s':'S','susceptible':'S','r':'R','resistant':'R','i':'I','intermediate':'I'}.get(x,'OTHER:'+x)


def verify_bvbrc(base,out):
    archived=base/'results/expansion_screening_20261109/raw_census_v2/candidate_raw_records.csv'
    old=defaultdict(set);current=defaultdict(set)
    with archived.open() as h:
        for r in csv.DictReader(h):
            value=native_mic(r)
            if value:old[(r['screen_species_annotation'],r['screen_drug'],r['genome_id'])].add((value,label(r['resistant_phenotype'])))
    root=base/'results/expansion_screening_20261109/phase2_current_reference'
    for folder in root.glob('*_current'):
        if not (folder/'current_laboratory_only.json').exists():continue
        protocol=json.loads((folder/'protocol.json').read_text());species=protocol['species_source_annotation']
        for r in json.loads((folder/'current_laboratory_only.json').read_text()):
            if r.get('evidence')!='Laboratory Method':raise ValueError('Unexpected evidence class')
            value=native_mic(r)
            if value:current[(species,r['antibiotic'],r['genome_id'])].add((value,label(r.get('resistant_phenotype',''))))
    records=list(csv.DictReader((out/'quantitative_state_set_comparisons.csv').open()));counts=Counter()
    for r in records:
        key=r['species'],r['antibiotic'],r['genome_id']
        for source,expected in [('archived',old[key]),('current',current[key])]:
            seen={(output_mic(m),c) for m,c in json.loads(r[source+'_state_set'])}
            if seen!=expected:raise ValueError('Independent quantitative state reconstruction differs')
        expected='MIC_STATE_SET_UNPAIRED_OR_UNAVAILABLE' if not old[key] or not current[key] else 'SAME_MIC_AND_CATEGORY_STATE_SET' if old[key]==current[key] else 'MIC_OR_CATEGORY_STATE_SET_DIFFERENCE_CANDIDATE'
        if expected!=r['comparison']:raise ValueError('Independent quantitative classification differs')
        counts[expected]+=1
    return {'status':'PASS','verified_genome_drug_keys':len(records),'state_set_counts':dict(counts)}


def verify_author(base,out):
    matrix=pd.read_csv(base/'results/reference_validation_20261109/cheon_kp_feature_matrix.tsv.gz',sep='\t',usecols=['sample_id','antibiotic','label'],dtype=str)
    author={(r.sample_id,r.antibiotic.strip().lower()):r.label for r in matrix.itertuples() if re.fullmatch(r'(?:SRR|ERR|DRR)\d+',r.sample_id)}
    runs={k[0] for k in author};by_side={}
    matched={}
    for side,build in [('old','PDG000000012.2530'),('new','PDG000000012.2559')]:
        path=base/'results/novelty_followup_20261110/retained_metadata_additional'/(build+'.metadata.tsv')
        by=defaultdict(set);matched[side]=set()
        for chunk in pd.read_csv(path,sep='\t',quoting=csv.QUOTE_NONE,usecols=['Run','AST_phenotypes','biosample_acc'],dtype=str,keep_default_na=False,chunksize=40000):
            chunk['run_list']=chunk.Run.str.findall(r'(?:SRR|ERR|DRR)\d+')
            expanded=chunk.explode('run_list');expanded=expanded[expanded.run_list.isin(runs)]
            for r in expanded.itertuples():
                matched[side].add(r.run_list)
                raw=r.AST_phenotypes
                if raw.startswith('"') and raw.endswith('"'):raw=raw[1:-1]
                if raw in {'','NULL','NA','N/A','-'}:continue
                for item in raw.split(','):
                    drug,cat=item.rsplit('=',1);by[(r.run_list,drug.strip().lower())].add(cat.strip().upper())
        by_side[side]=by
    rows=list(csv.DictReader((out/'author_run_reference_comparisons.csv').open()))
    comparable=changed=author_difference=0
    if set(author)!={(r['sample_id'],r['antibiotic'].strip().lower()) for r in rows}:raise ValueError('Author matrix key mismatch')
    for r in rows:
        key=r['sample_id'],r['antibiotic'].strip().lower()
        if r['label']!=author[key]:raise ValueError('Author label changed')
        binary={}
        for side in ['old','new']:
            categories=by_side[side][key]
            if '|'.join(sorted(categories))!=r[side+'_native_category_set']:raise ValueError('Native category set mismatch')
            values={0 if x=='S' else 1 for x in categories if x in {'S','I','R'}}
            binary[side]=str(next(iter(values))) if len(values)==1 and categories<={'S','I','R'} else ''
            if binary[side]!=r[side+'_binary_label']:raise ValueError('Binary mapping mismatch')
        if binary['old'] and binary['new'] and r['linkage_status']=='PAIRED_RUN_BIOSAMPLE':
            comparable+=1;changed+=binary['old']!=binary['new'];author_difference+=author[key]!=binary['old']
    return {'status':'PASS','author_run_drug_keys':len(rows),'matched_runs':{k:len(v) for k,v in matched.items()},'comparable_keys':comparable,'old_new_binary_changes':changed,'author_label_differences':author_difference}


def verify_certificate(base,out):
    source=base/'results/received_bundle_audit_20261024/full_source_verification_20261024/reconciliation/E2_POSTHOC_reference_scenario.csv'
    rows=list(csv.DictReader(source.open()));pred=[float(r['probability'])>=.5 for r in rows]
    ml={}
    for name,column in [('old','historical_label'),('new','scenario_label')]:
        resistant=[r[column]=='R' for r in rows]
        ml[name]={'FN':sum(y and not p for y,p in zip(resistant,pred)),'FP':sum(not y and p for y,p in zip(resistant,pred))}
    # Independent closed-form source-allocation bounds, rather than the production enumeration.
    focal=[r for r in rows if r['project']=='PRJNA1322038']
    fr=sum(r['historical_label']=='R' for r in focal);fs=len(focal)-fr
    tp,fn,tn,fp=138,25,192,170
    amin,amax=max(0,fr-fn),min(fr,tp);bmin,bmax=max(0,fs-tn),min(fs,fp)
    fnlo,fnhi=fn-fr+fs+amin-bmax,fn-fr+fs+amax-bmin
    R=sum(r['scenario_label']=='R' for r in rows);S=len(rows)-R
    fpmin=tp+fp-R+fnlo
    report=json.loads((out/'certificate.json').read_text())
    assert report['current_rule_FN_feasible_range']=={'min':fnlo,'max':fnhi}
    assert report['current_rule_FP_lower_bound_from_margins']==fpmin==42
    assert report['current_ML_FN']==ml['new']['FN']==30
    assert report['current_ML_FP']==ml['new']['FP']==14
    assert report['minimum_fewer_false_resistant_calls_current_reference']==fpmin-ml['new']['FP']==28
    assert report['feasible_current_rule_aggregate_matrices']==fnhi-fnlo+1==165
    return {'status':'PASS','method':'Independent closed-form source allocation bounds and stdlib frozen-prediction counts','ML_counts':ml,'rule_new_FN_bounds':[fnlo,fnhi],'rule_new_FP_min':fpmin,'guaranteed_FP_count_difference':28,'reference_S':S}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();report={}
    report['bvbrc']=verify_bvbrc(args.repo,args.results/'bvbrc_quantitative_audit_final')
    report['clinical_error_certificate']=verify_certificate(args.repo,args.results/'clinical_error_certificate_final')
    report['author_run_linkage']=verify_author(args.repo,args.results/'cheon_full_run_linkage')
    with args.out.open('x') as h:json.dump({'status':'PASS','checks':report},h,indent=2);h.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
