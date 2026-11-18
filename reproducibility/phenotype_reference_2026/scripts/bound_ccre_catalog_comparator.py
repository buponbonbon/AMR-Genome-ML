#!/usr/bin/env python3
"""Bound a catalog-derived carbapenem-associated comparator without inventing native hit rows.

This is a distinct exploratory comparator representation. It does not recover
the missing original per-isolate File11E AMRFinder output. Unresolved annotations
remain unresolved; no genotype symbol is used as a clinical reference label.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from file11e_external_simple_baseline_comparators import (CARBAPENEM_SEMANTIC_RE,
    CANONICAL_CARBAPENEMASE_FAMILY_RE)


def semantic(row, mutation=False):
    name=row['mutated_protein_name'].replace('_',' ') if mutation else row['product_name']
    return bool(CARBAPENEM_SEMANTIC_RE.search(' | '.join([name,row['class'],row['subclass']])))


def confusion_bounds(y, low, high, ml):
    R=int(y.sum());S=len(y)-R
    tp=int(((low==1)&(y==1)).sum());fp=int(((low==1)&(y==0)).sum())
    uR=int(((high>low)&(y==1)).sum());uS=int(((high>low)&(y==0)).sum())
    tps=np.arange(tp,tp+uR+1)[:,None];fps=np.arange(fp,fp+uS+1)[None,:]
    fns=R-tps;tns=S-fps
    ba=(tps/R+tns/S)/2
    denom=np.sqrt((tps+fps)*(tps+fns)*(tns+fps)*(tns+fns),dtype=float)
    mcc=np.divide(tps*tns-fps*fns,denom,out=np.zeros_like(denom,dtype=float),where=denom!=0)
    mlba=((ml[y==1]==1).mean()+(ml[y==0]==0).mean())/2
    return {'known_positive':int((low==1).sum()),'known_negative':int((high==0).sum()),
        'unresolved':int((high>low).sum()),'unresolved_R':uR,'unresolved_S':uS,
        'TP_range':[tp,tp+uR],'FP_range':[fp,fp+uS],
        'FN_range':[R-tp-uR,R-tp],'TN_range':[S-fp-uS,S-fp],
        'balanced_accuracy_range':[float(ba.min()),float(ba.max())],
        'MCC_range':[float(mcc.min()),float(mcc.max())],
        'ML_minus_catalog_comparator_BA_range':[float(mlba-ba.max()),float(mlba-ba.min())]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();catpath=args.root/'legacy_catalog/ReferenceGeneCatalog.txt';mutpath=args.root/'legacy_catalog/AMRProt-mutation.tsv'
    catalog=pd.read_csv(catpath,sep='\t',dtype=str,keep_default_na=False)
    catalog=catalog[catalog.type.eq('AMR') & catalog.subtype.isin(['AMR','POINT','POINT_DISRUPT'])]
    mutation=pd.read_csv(mutpath,sep='\t',dtype=str,keep_default_na=False)
    mutation=mutation[mutation['#taxgroup'].eq('Klebsiella_pneumoniae')]
    rows=pd.read_csv(args.root/'legacy_feature_diagnostic/results/CCRE_legacy_feature_diagnostic_predictions.csv',dtype=str,keep_default_na=False)
    evidence={}
    @lru_cache(None)
    def classify(entry):
        gene,*status=entry.split('=',1);gene=gene.strip();tag=status[0] if status else ''
        if CANONICAL_CARBAPENEMASE_FAMILY_RE.match(gene):
            result=(1,1,'UNCHANGED_CANONICAL_FAMILY_FALLBACK',[])
        else:
            m=mutation[mutation.reported_mutation_symbol.eq(gene)] if tag=='POINT' else mutation.iloc[:0]
            if len(m):
                flags=[semantic(r,True) for r in m.to_dict('records')]
                result=(int(all(flags)),int(any(flags)),'EXACT_REPORTED_POINT_SYMBOL',m[['reported_mutation_symbol','class','subclass','mutated_protein_name']].drop_duplicates().to_dict('records'))
            else:
                c=catalog[catalog.allele.eq(gene)]
                route='EXACT_CATALOG_ALLELE'
                if c.empty and tag=='POINT' and re.search(r'(?:Ter|fs)',gene):
                    base=gene.split('_',1)[0]
                    c=catalog[catalog.allele.eq('') & catalog.gene_family.eq(base) & catalog.subtype.eq('POINT_DISRUPT')]
                    route='CATALOG_DISRUPTION_FAMILY'
                if c.empty and tag!='POINT':
                    c=catalog[catalog.allele.eq('') & catalog.gene_family.eq(gene) & catalog.subtype.eq('AMR')]
                    route='UNALLELED_CATALOG_AMR_FAMILY'
                if c.empty:result=(0,1,'UNRESOLVED_DO_NOT_INVENT_ANNOTATION',[])
                else:
                    flags=[semantic(r) for r in c.to_dict('records')]
                    result=(int(all(flags)),int(any(flags)),route,c[['allele','gene_family','product_name','subtype','class','subclass']].drop_duplicates().to_dict('records'))
        evidence[entry]={'entry':entry,'gene':gene,'metadata_status':tag,'definitely_positive':result[0],
                         'possibly_positive':result[1],'mapping_route':result[2],'catalog_annotations':result[3]}
        return result[0],result[1]
    lows=[];highs=[];reasons=[]
    for text in rows.AMR_genotypes:
        entries=[x.strip() for x in text.split(',') if x.strip()]
        c=[classify(e) for e in entries]
        lo=max((v[0] for v in c),default=0);hi=max((v[1] for v in c),default=0)
        lows.append(lo);highs.append(hi)
        reasons.append(';'.join(e for e,v in zip(entries,c) if v[0]))
    low=np.asarray(lows);high=np.asarray(highs)
    rows['catalog_comparator_lower_R']=low;rows['catalog_comparator_upper_R']=high
    rows['catalog_comparator_status']=np.where(low==high,'IDENTIFIED_UNDER_CATALOG_MAPPING','UNRESOLVED')
    rows['definite_carbapenem_associated_entries']=reasons
    y=rows['S/I/R (PHAS interpretation)_Meropenem'].map({'S':0,'R':1}).to_numpy(int)
    ml=rows.exploratory_frozen_predicted_R.to_numpy(int)
    report={'status':'EXPLORATORY_CATALOG_COMPARATOR_PARTIAL_IDENTIFICATION','n':len(rows),
            'feature_database':'2026-01-21.1','bounds':confusion_bounds(y,low,high,ml),
            'unique_gene_entries':len(evidence),
            'unresolved_gene_entries':sum(e['definitely_positive']!=e['possibly_positive'] for e in evidence.values()),
            'native_File11E_per_isolate_replay_claimed':False,
            'unresolved_annotations_coerced_to_negative':False,
            'scope':'Archived Jan-2026 catalog semantics plus the unchanged canonical-family fallback. Bounds cover unresolved gene annotations within this mapping, not every possible source-annotation error.',
            'limitations':['These are catalog-derived diagnostic bounds, not recovered original per-isolate File11E calls.','The older feature DB and partial internal PDS coverage remain material limits.','A carbapenem-associated annotation rule includes porin disruptions and is not restricted to carbapenemase genes.'],
            'input_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [catpath,mutpath]}}
    args.out.mkdir(parents=True,exist_ok=False)
    rows.to_csv(args.out/'CCRE_catalog_comparator_bounds.csv',index=False)
    (args.out/'gene_annotation_evidence.json').write_text(json.dumps(list(evidence.values()),indent=2)+'\n')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
