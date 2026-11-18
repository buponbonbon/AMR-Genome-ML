#!/usr/bin/env python3
"""Prepare a public primary-BMD source gate; no genomes are scored here."""
import argparse,hashlib,json,re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def classify(value):
    s=str(value).strip().replace('≤','<=').replace('≥','>=')
    m=re.fullmatch(r'(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)',s)
    if not m:return 'MISSING_OR_UNSUPPORTED'
    sign=m.group(1) or '=';v=Decimal(m.group(2))
    # Explicit analysis scenario: EUCAST2024 Enterobacterales, non-meningitis.
    if sign=='=':return 'S' if v<=2 else 'R' if v>8 else 'I'
    if sign in {'<','<='} and v<=2:return 'S'
    if sign=='>' and v>=8 or sign=='>=' and v>8:return 'R'
    return 'AMBIGUOUS_INTERVAL'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    run=root/'results/independent_reference_extension_20261112'
    table=run/'supplementary/PMC7244338/EMS83227-supplement-Supplementary_Table_4.xlsx'
    assert sha(table)=='947ace74cc2c498d81f31244a1c01e985a92306de616a5a04ced0db6d57e792a'
    d=pd.read_csv(run/'EuSCAPE_primary_isolate_identity.csv',dtype=str,keep_default_na=False)
    training=pd.read_csv(root/'results/independent_MIC_screen_20261111/CCRE_verified_screen_final/frozen_training_identity_complete.csv',dtype=str,keep_default_na=False)
    e2=pd.read_csv(root/'results/database_reference_factorial_20261111/E2_execution_manifest.csv',dtype=str)
    ccre=pd.read_csv(root/'results/strict_CCRE_followup_20261111/native355_execution_manifest.csv',dtype=str)
    sets={'fitting':set(training.biosample_accession),'E2':set(e2.sample_accession),'CCRE':set(ccre.sample_accession)}
    clpath=root/'results/strict_CCRE_followup_20261111/latest_cluster_sensitivity/PDG000000012.2575.cluster_list.tsv'
    receipt=json.loads(clpath.with_name(clpath.name+'.receipt.json').read_text());assert sha(clpath)==receipt['sha256']
    cl=pd.read_csv(clpath,sep='\t',dtype=str,keep_default_na=False,usecols=['biosample_acc','PDS_acc'])
    groups=defaultdict(set)
    for bio,pds in cl[['biosample_acc','PDS_acc']].itertuples(index=False,name=None):
        if re.fullmatch(r'PDS\d+(?:\.\d+)?',pds):groups[bio].add(pds.split('.')[0])
    blocked={role:set().union(*(groups[s] for s in samples)) for role,samples in sets.items()}
    if any(not blocked[role] for role in sets):raise ValueError('Known mapped reference cohorts disappeared; inspect cluster parsing')
    d['sample_accession']=d.sample_accession;d['source_primary_MIC_raw']=d['Meropenem MIC (centralised re-testing)']
    d['analysis_label_EUCAST14_nonmeningitis']=d.source_primary_MIC_raw.map(classify)
    d['source_reported_meropenem_SIR']=''
    d['source_KpI']=d.Species.eq('K. pneumoniae sensu stricto (KpI)')
    d['source_BMD_present']=~d.analysis_label_EUCAST14_nonmeningitis.eq('MISSING_OR_UNSUPPORTED')
    d['source_explicit_assembly_version']=d['Assembly accession'].str.fullmatch(r'GC[AF]_\d+\.\d+')
    classifications=[];cluster_values=[]
    for row in d.to_dict('records'):
        identifiers={row['sample_accession'],row['secondary_sample_accession']};identifiers.discard('')
        exact=[role for role,samples in sets.items() if identifiers & samples]
        g=set().union(*(groups[s] for s in identifiers));cluster_values.append(';'.join(sorted(g)))
        shared=[role for role,pds in blocked.items() if g & pds]
        classifications.append('EXACT_ID_OVERLAP_'+','.join(exact) if exact else 'MISSING_PDS' if not g else 'MULTIPLE_PDS' if len(g)!=1 else 'SHARED_PDS_'+','.join(shared) if shared else 'PASS_OPERATIONAL_SOURCE_GATE')
    d['PDS_bases']=cluster_values;d['operational_independence_gate']=classifications
    d['selection_hash']=d.sample_accession.map(lambda x:hashlib.sha256(('EuSCAPE_primary_BMD|'+x).encode()).hexdigest())
    eligible=d.source_KpI & d.analysis_label_EUCAST14_nonmeningitis.isin(['S','R']) & d.source_explicit_assembly_version & d.operational_independence_gate.eq('PASS_OPERATIONAL_SOURCE_GATE')
    candidates=d.loc[eligible].copy();selected=candidates.sort_values('selection_hash').drop_duplicates('PDS_bases').sort_values('sample_accession')
    d.to_csv(run/'EuSCAPE_primary_BMD_all_source_gate.csv',index=False);candidates.to_csv(run/'EuSCAPE_primary_BMD_operational_candidates.csv',index=False);selected.to_csv(run/'EuSCAPE_primary_BMD_execution_manifest_NOT_SCORED.csv',index=False)
    report=dict(status='PRIMARY_BMD_INPUT_GATE_COMPLETE_NO_NATIVE_PREDICTIONS',source_doi='10.1038/s41564-019-0492-8',primary_table_sha256=sha(table),source_isolates=len(d),source_KpI=int(d.source_KpI.sum()),KpI_BMD_present=int((d.source_KpI&d.source_BMD_present).sum()),
        KpI_analysis_categories=d.loc[d.source_KpI,'analysis_label_EUCAST14_nonmeningitis'].value_counts().to_dict(),
        KpI_independence_gate=d.loc[d.source_KpI,'operational_independence_gate'].value_counts().to_dict(),
        eligible_cases=len(candidates),selected_one_per_PDS=len(selected),selected_analysis_categories=selected.analysis_label_EUCAST14_nonmeningitis.value_counts().to_dict(),
        cluster_build='PDG000000012.2575',mapped_samples_by_role={role:sum(bool(groups[s]) for s in samples) for role,samples in sets.items()},
        no_model_training_tuning_or_scoring=True,
        labels_are_derived_explicit_analysis_scenario_not_author_SIR=True,
        limits=['Quantitative central BMD is published; historical author S/I/R and exact interpretive edition are not inferred.',
            'The proposed labels explicitly use EUCAST14 non-meningitis cutoffs; this is a derived validation scenario, not adjudication of old records.',
            'Assembly accessions are the published versions, but FASTA has not yet been acquired or hashed.',
            'PDS exclusion uses the same completed build for all roles and only mapped genomes; it does not establish universal genetic independence.',
            'This is an additional BMD-validation route; it is not a second temporal record-change event.',
            'Selection and input inspection are post-hoc; no new source model score has been read.'])
    (run/'EuSCAPE_primary_BMD_readiness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
