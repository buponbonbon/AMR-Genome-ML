#!/usr/bin/env python3
"""Download public ENA-linked assemblies with checksum and identity receipts.

Only decompresses the provided data. No bases, headers or reference labels are
edited. Assembly QC flags are descriptive and do not select favorable outcomes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
import urllib.error

import pandas as pd


def fasta_qc(data):
    lengths=[];current=0;ambiguous=0;headers=set()
    for line in data.decode('ascii').splitlines():
        line=line.strip()
        if not line:continue
        if line.startswith('>'):
            if current:lengths.append(current)
            header=line[1:].split()[0]
            if not header or header in headers:raise ValueError('Missing or duplicate FASTA contig ID')
            headers.add(header);current=0
        else:
            if not headers:raise ValueError('FASTA sequence precedes its header')
            sequence=line.upper()
            if set(sequence)-set('ACGTNRYWSKMBDHV'):raise ValueError('Non-DNA symbols in assembly')
            current+=len(sequence);ambiguous+=sum(c not in 'ACGT' for c in sequence)
    if current:lengths.append(current)
    if not lengths:raise ValueError('Empty FASTA assembly')
    total=sum(lengths);running=0;n50=0
    for value in sorted(lengths,reverse=True):
        running+=value
        if running>=total/2:n50=value;break
    flags=[]
    if not 4_000_000<=total<=7_000_000:flags.append('LENGTH_OUTSIDE_4_TO_7_MBP')
    if n50<10_000:flags.append('N50_LT_10K')
    if len(lengths)>1500:flags.append('CONTIGS_GT_1500')
    if ambiguous/total>.005:flags.append('AMBIGUOUS_BASES_GT_0_5_PERCENT')
    return {'length_bp':total,'contigs':len(lengths),'N50_bp':n50,'ambiguous_bases':ambiguous,
            'QC_flags':';'.join(flags),'QC_flags_do_not_change_selection':True}


def fetch(row,out,timeout):
    sample=row['sample_accession'];target=out/'fastas'/(sample+'.fna');compressed=out/'compressed'/(sample+'.fa.gz')
    receipt_path=out/'receipts'/(sample+'.json')
    if receipt_path.exists():
        history=out/'receipts/history';history.mkdir(exist_ok=True)
        old=receipt_path.read_bytes();tag=hashlib.sha256(old).hexdigest()[:16]
        (history/(sample+'.'+tag+'.json')).write_bytes(old)
    receipt={key:row[key] for key in ['sample_accession','analysis_accession','study_accession','submitted_ftp','submitted_md5','submitted_bytes']}
    receipt['started_utc']=datetime.now(timezone.utc).isoformat();start=time.monotonic()
    try:
        if not re.fullmatch(r'SAM[NED][A-Z]?[0-9]+',sample):raise ValueError('Noncanonical BioSample')
        if ';' in row['submitted_ftp'] or not row['submitted_ftp'].endswith('.gz'):raise ValueError('Expected one explicitly linked gzip FASTA')
        url='https://'+row['submitted_ftp'].removeprefix('ftp://').removeprefix('https://')
        if not url.startswith('https://ftp.sra.ebi.ac.uk/'):raise ValueError('Unexpected assembly source host')
        if compressed.is_file():data=compressed.read_bytes();receipt['download_status']='REUSED_AFTER_CHECKSUM_VALIDATION'
        else:
            request=urllib.request.Request(url,headers={'User-Agent':'AMRReferenceValidation/1.0 (public data audit)'})
            with urllib.request.urlopen(request,timeout=timeout) as response:
                if response.status!=200:raise ValueError(f'Unexpected HTTP {response.status}')
                data=response.read(30*1024**2+1)
                if len(data)>30*1024**2:raise ValueError('Assembly exceeds per-file size limit')
            receipt['download_status']='DOWNLOADED'
        if len(data)!=int(row['submitted_bytes']) or hashlib.md5(data).hexdigest()!=row['submitted_md5']:raise ValueError('ENA byte count or MD5 mismatch')
        if not compressed.is_file():
            tmp=compressed.with_suffix('.part');tmp.write_bytes(data);tmp.replace(compressed)
        fasta=gzip.decompress(data)
        qc=fasta_qc(fasta)
        fasta_hash=hashlib.sha256(fasta).hexdigest()
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=fasta_hash:
            tmp=target.with_suffix('.part');tmp.write_bytes(fasta);tmp.replace(target)
        receipt.update(status='PASS_ENA_CHECKSUM_AND_FASTA',compressed_sha256=hashlib.sha256(data).hexdigest(),fasta_sha256=hashlib.sha256(fasta).hexdigest(),fasta_path=str(target.resolve()),**qc)
    except Exception as error:
        receipt.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        if isinstance(error,urllib.error.HTTPError):
            receipt['http_status']=error.code
            receipt['retry_after']=error.headers.get('Retry-After')
    receipt['seconds']=time.monotonic()-start;receipt['completed_utc']=datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assembly-table',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--timeout',type=int,default=30)
    p.add_argument('--failed-only',action='store_true',help='Make one bounded retry of previously failed inputs; retain successful receipts unchanged')
    a=p.parse_args();manifest=pd.read_csv(a.manifest,dtype=str,keep_default_na=False);table=pd.read_csv(a.assembly_table,sep='\t',dtype=str,keep_default_na=False)
    if not manifest.sample_accession.is_unique:raise ValueError('Repeated target sample')
    table=table[table.sample_accession.isin(manifest.sample_accession)]
    if not table.sample_accession.is_unique or set(table.sample_accession)!=set(manifest.sample_accession):raise ValueError('Missing or ambiguous assembly linkage; no outcome-driven replacement permitted')
    if not table.analysis_type.eq('SEQUENCE_ASSEMBLY').all():raise ValueError('Non-assembly analysis in source table')
    table=table.set_index('sample_accession').loc[manifest.sample_accession].reset_index()
    for name in ['fastas','compressed','receipts']:(a.out/name).mkdir(parents=True,exist_ok=True)
    jobs=table.to_dict('records')
    if a.failed_only:
        jobs=[r for r in jobs if not (a.out/'receipts'/(r['sample_accession']+'.json')).exists() or json.loads((a.out/'receipts'/(r['sample_accession']+'.json')).read_text()).get('status')=='FAIL']
    completed=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(fetch,row,a.out,a.timeout) for row in jobs]
        for future in as_completed(pending):
            result=future.result();completed.append(result)
            if len(completed)%10==0 or len(completed)==len(pending):print(json.dumps({'completed':len(completed),'target':len(pending),'failures':sum(r['status']=='FAIL' for r in completed)}),flush=True)
    if a.failed_only:
        completed=[json.loads((a.out/'receipts'/(sample+'.json')).read_text()) for sample in manifest.sample_accession]
    pd.DataFrame(completed).to_csv(a.out/'assembly_acquisition.csv',index=False)
    summary={'status':'PASS' if all(r['status'].startswith('PASS') for r in completed) else 'FAILURES_RETAINED','target':len(table),'passed':sum(r['status'].startswith('PASS') for r in completed),'failed':sum(r['status']=='FAIL' for r in completed),'source_projects':table.study_accession.value_counts().to_dict(),'source_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'assembly_table_sha256':hashlib.sha256(a.assembly_table.read_bytes()).hexdigest(),'DNA_or_reference_labels_modified':False}
    (a.out/'report.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
