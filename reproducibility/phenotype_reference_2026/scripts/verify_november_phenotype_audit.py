#!/usr/bin/env python3
"""Independent chunked dataframe reconstruction of BioSample-level S/I/R sets."""
import argparse
from collections import defaultdict, Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import pandas as pd


def load(path, species):
    values=defaultdict(set); focal=set()
    cols=['scientific_name','biosample_acc','bioproject_acc','AST_phenotypes']
    for chunk in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,quoting=csv.QUOTE_NONE,
                             usecols=cols,chunksize=20000):
        focal.update(chunk.loc[chunk.bioproject_acc.str.contains(r'\bPRJNA1322038\b',regex=True),'biosample_acc'])
        chunk=chunk[chunk.scientific_name.eq(species)|chunk.scientific_name.str.startswith(species+' ')]
        for row in chunk.itertuples(index=False,name=None):
            # Address by name, since original file column ordering is preserved by pandas.
            d=dict(zip(chunk.columns,row)); bio=d['biosample_acc']; text=d['AST_phenotypes'].strip('"')
            if not re.fullmatch(r'SAM[NED][A-Z]?\d+',bio) or text.upper() in {'','NULL','NA','N/A','-'}:continue
            tokens=re.findall(r'([^,=]+)=([^,=]+)',text)
            if len(tokens)!=len(text.split(',')):raise ValueError('Independent token parser coverage failed')
            for drug,category in tokens:values[(bio,drug.strip().lower())].add(category.strip().upper())
    return values,focal


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.root/'PROTOCOL.json').read_text());reported=json.loads((a.root/'report.json').read_text())
    results=[]
    for t,expected in zip(protocol['targets'],reported['targets']):
        old,of=load(Path(t['old_file']),t['species']);new,nf=load(a.root/(t['new_build']+'.metadata.tsv'),t['species'])
        focal=of|nf;counts=Counter();primary=Counter();n=0
        for key in old.keys()&new.keys():
            if key[0] in focal:continue
            aa=old[key]&{'S','I','R'};bb=new[key]&{'S','I','R'}
            status='NON_SIR_OR_MISSING_INVOLVED' if not aa or not bb else 'SAME_SIR_SET' if aa==bb else 'UNOPPOSED_R_S_REVERSAL_CANDIDATE' if (aa,bb) in [({'S'},{'R'}),({'R'},{'S'})] else 'OTHER_SIR_SET_CHANGE'
            counts[status]+=1;n+=1
            if key[1]=='meropenem':primary[status]+=1
        if dict(counts)!=expected['aggregated']['counts'] or dict(primary)!=expected['primary_meropenem_counts'] or n!=expected['aggregated']['paired_keys']:
            raise ValueError('Independent reconstruction disagrees')
        results.append({'species':t['species'],'paired_keys':n,'counts':dict(counts),'meropenem':dict(primary)})
        print(t['species'],'PASS',flush=True)
    result={'status':'PASS_INDEPENDENT_CHUNKED_SOURCE_RECONSTRUCTION','results':results,
            'report_sha256':hashlib.sha256((a.root/'report.json').read_bytes()).hexdigest(),
            'independent_parser_and_grouping':True}
    (a.root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
