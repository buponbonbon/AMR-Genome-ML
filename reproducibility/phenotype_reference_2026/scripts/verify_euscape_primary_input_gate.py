#!/usr/bin/env python3
"""Verify the primary workbook/ENA/PDS and exact published assembly inputs."""
import argparse,hashlib,json,re
from collections import Counter
from decimal import Decimal
from pathlib import Path
import openpyxl
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def interval_category(s):
    s=str(s).strip().replace('≤','<=').replace('≥','>=');m=re.fullmatch(r'(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)',s)
    if not m:return 'MISSING_OR_UNSUPPORTED'
    sign=m.group(1) or '=';v=Decimal(m.group(2));lo=Decimal(0);hi=Decimal('Infinity');lo_closed=False
    if sign=='=':lo=hi=v;lo_closed=True
    if sign in ['>','>=']:lo=v;lo_closed=sign=='>='
    if sign in ['<','<=']:hi=v
    if hi<=2:return 'S'
    if lo>8 or lo==8 and not lo_closed:return 'R'
    if lo>2 and hi<=8:return 'I'
    return 'AMBIGUOUS_INTERVAL'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    run=root/'results/independent_reference_extension_20261112';book=run/'supplementary/PMC7244338/EMS83227-supplement-Supplementary_Table_4.xlsx'
    ws=openpyxl.load_workbook(book,read_only=True,data_only=True).active;rows=list(ws.iter_rows(values_only=True));header=[str(x or '') for x in rows[2]]
    source={str(r[0]):dict(zip(header,r)) for r in rows[3:] if str(r[0]).startswith('EuSCAPE_')}
    assert len(source)==1717
    allcases=pd.read_csv(run/'EuSCAPE_primary_BMD_all_source_gate.csv',dtype=str,keep_default_na=False)
    chosen=pd.read_csv(run/'EuSCAPE_primary_BMD_execution_manifest_NOT_SCORED.csv',dtype=str,keep_default_na=False)
    identity=pd.read_csv(run/'EuSCAPE_ENA_primary_identity.tsv',sep='\t',dtype=str,keep_default_na=False).set_index('run_accession')
    for row in allcases.to_dict('records'):
        s=source[row['Sample name']]
        for col in ['Run accession','Assembly accession','Meropenem MIC (centralised re-testing)','Species']:assert str(s[col])==row[col]
        assert row['sample_accession']==identity.loc[row['Run accession'],'sample_accession']
        assert row['secondary_sample_accession']==identity.loc[row['Run accession'],'secondary_sample_accession']
        assert row['analysis_label_EUCAST14_nonmeningitis']==interval_category(s['Meropenem MIC (centralised re-testing)'])
        assert row['source_reported_meropenem_SIR']==''
    fit=pd.read_csv(root/'results/independent_MIC_screen_20261111/CCRE_verified_screen_final/frozen_training_identity_complete.csv',dtype=str)
    e2=pd.read_csv(root/'results/database_reference_factorial_20261111/E2_execution_manifest.csv',dtype=str)
    ccre=pd.read_csv(root/'results/strict_CCRE_followup_20261111/native355_execution_manifest.csv',dtype=str)
    sets={'fitting':set(fit.biosample_accession),'E2':set(e2.sample_accession),'CCRE':set(ccre.sample_accession)}
    cp=root/'results/strict_CCRE_followup_20261111/latest_cluster_sensitivity/PDG000000012.2575.cluster_list.tsv'
    cl=pd.read_csv(cp,sep='\t',dtype=str,keep_default_na=False);cl=cl[cl.PDS_acc.str.match(r'^PDS\d+\.')]
    grouped=cl.assign(base=cl.PDS_acc.str.split('.').str[0]).groupby('biosample_acc').base.agg(frozenset)
    blocked={role:set(cl.loc[cl.biosample_acc.isin(ids),'PDS_acc'].str.split('.').str[0]) for role,ids in sets.items()}
    counts=Counter()
    for row in allcases[allcases.source_KpI.eq('True')].to_dict('records'):
        ids={row['sample_accession'],row['secondary_sample_accession']};ids.discard('')
        pds=set().union(*(grouped.get(s,frozenset()) for s in ids));overlap=[role for role,known in sets.items() if ids & known];shared=[role for role,known in blocked.items() if pds & known]
        gate='EXACT_ID_OVERLAP_'+','.join(overlap) if overlap else 'MISSING_PDS' if not pds else 'MULTIPLE_PDS' if len(pds)!=1 else 'SHARED_PDS_'+','.join(shared) if shared else 'PASS_OPERATIONAL_SOURCE_GATE'
        assert row['operational_independence_gate']==gate;counts[gate]+=1
    eligible=allcases[allcases.source_KpI.eq('True')&allcases.analysis_label_EUCAST14_nonmeningitis.isin(['S','R'])&allcases.source_explicit_assembly_version.eq('True')&allcases.operational_independence_gate.eq('PASS_OPERATIONAL_SOURCE_GATE')]
    expected=eligible.sort_values('selection_hash').drop_duplicates('PDS_bases').sort_values('sample_accession')
    assert expected.sample_accession.tolist()==chosen.sample_accession.tolist() and len(chosen)==170
    for row in chosen.to_dict('records'):
        receipt=json.loads((run/'EuSCAPE_assemblies/receipts'/(row['sample_accession']+'.json')).read_text())
        assert receipt['assembly_accession']==row['Assembly accession'] and receipt['archive_md5_verified'] is True
        assert receipt['fasta_sha256']==sha(run/'EuSCAPE_assemblies/fastas'/(row['sample_accession']+'.fna'))
        assert row['scientific_name']=='Klebsiella pneumoniae'
    result=dict(status='PASS_INDEPENDENT_PRIMARY_XLSX_INTERVAL_ENA_PDS_FASTA_GATE',source_rows=1717,KpI_PDS_counts=dict(counts),eligible=len(eligible),selected=170,selected_labels=chosen.analysis_label_EUCAST14_nonmeningitis.value_counts().to_dict(),FASTA_byte_hashes_checked=170,source_SIR_not_invented=True)
    (run/'EuSCAPE_independent_input_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
