#!/usr/bin/env python3
"""Recompute a later-cluster sensitivity without changing the selected cohort."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['clusters','receipt','training','manifest','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();receipt=json.loads(a.receipt.read_text())
    if hashlib.sha256(a.clusters.read_bytes()).hexdigest()!=receipt['sha256']:raise ValueError('Source receipt failed')
    c=pd.read_csv(a.clusters,sep='\t',dtype=str,keep_default_na=False)
    f=pd.read_csv(a.training,dtype=str,keep_default_na=False);d=pd.read_csv(a.manifest,dtype=str,keep_default_na=False)
    by=defaultdict(set)
    for row in c.itertuples():by[row.biosample_acc].add(row.PDS_acc.split('.')[0])
    fit=set(f.biosample_accession);fitgroups=set().union(*(by[s] for s in fit));rows=[]
    for row in d.itertuples():
        groups=by[row.sample_accession];shared=groups&fitgroups
        rows.append({'sample_accession':row.sample_accession,'original_PDS_bases':row.PDS_bases,
          'latest_PDS_bases':';'.join(sorted(groups)),'latest_shared_with_fitting':';'.join(sorted(shared)),
          'latest_gate':'MISSING_PDS' if not groups else 'SHARED_WITH_MAPPED_FITTING' if shared else 'NO_SHARED_WITH_MAPPED_FITTING'})
    a.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out/'fixed355_latest_PDS_sensitivity.csv',index=False)
    report={'status':'COMPLETE_FIXED_COHORT_LATEST_COMPLETED_PDS_SENSITIVITY','build':'PDG000000012.2575',
      'fitting_unique_BioSamples':len(fit),'fitting_BioSamples_mapped':sum(bool(by[s]) for s in fit),
      'fitting_PDS_bases':len(fitgroups),'case_gate_counts':dict(Counter(x['latest_gate'] for x in rows)),
      'unique_case_PDS_bases':len(set().union(*(by[s] for s in d.sample_accession))),
      'multi_PDS_cases':sum(len(by[s])>1 for s in d.sample_accession),
      'case_membership_changed_from_original':sum(x['original_PDS_bases']!=x['latest_PDS_bases'] for x in rows),
      'original_manifest_unchanged':True,
      'limit':'Absence of sharing only concerns mapped fitting BioSamples; source-defined PDS is not a universal pairwise genomic-independence certificate.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
