#!/usr/bin/env python3
"""Acquire exactly the published assembly versions for the fixed EuSCAPE gate."""
import argparse,hashlib,io,json,urllib.request,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,j):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    protocol=json.loads((root/'EUScAPE_NATIVE_PROTOCOL.json').read_text());manifest=root/'EuSCAPE_primary_BMD_execution_manifest_NOT_SCORED.csv'
    assert sha(manifest.read_bytes())==protocol['manifest_sha256'];d=pd.read_csv(manifest,dtype=str,keep_default_na=False)
    assert len(d)==170 and d.sample_accession.is_unique and d['Assembly accession'].is_unique
    dest=root/'EuSCAPE_assemblies'
    for name in ['fastas','receipts','batches']:(dest/name).mkdir(parents=True,exist_ok=True)
    pending=[]
    for row in d.to_dict('records'):
        f=dest/'fastas'/(row['sample_accession']+'.fna');r=dest/'receipts'/(row['sample_accession']+'.json')
        if f.exists() and r.exists():
            receipt=json.loads(r.read_text())
            if receipt['status']=='PASS_PUBLISHED_ASSEMBLY_VERSION' and sha(f.read_bytes())==receipt['fasta_sha256']:continue
        pending.append(row)
    batches=[pending[i:i+20] for i in range(0,len(pending),20)]
    def fetch(rows):
        accessions=','.join(x['Assembly accession'] for x in rows);tag=sha(accessions.encode())[:18]
        u='https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/'+accessions+'/download?include_annotation_type=GENOME_FASTA'
        receipt=dict(url=u,requested_accessions=[r['Assembly accession'] for r in rows],started_utc=datetime.now(timezone.utc).isoformat());path=dest/'batches'/(tag+'.zip')
        try:
            if not path.exists():
                q=path.with_suffix('.part')
                with urllib.request.urlopen(u,timeout=45) as response,q.open('wb') as f:
                    if response.status!=200:raise ValueError('HTTP status is not complete')
                    while b:=response.read(1048576):f.write(b)
                q.replace(path)
            z=zipfile.ZipFile(path);assert z.testzip() is None;receipt.update(zip_sha256=sha(path.read_bytes()),zip_bytes=path.stat().st_size)
            md5={}
            for name in z.namelist():
                if name.endswith('md5sum.txt'):
                    for line in z.read(name).decode().splitlines():
                        parts=line.split(maxsplit=1)
                        if len(parts)==2:md5[parts[1].strip().lstrip('*').removeprefix('./')]=parts[0]
            for row in rows:
                sample=row['sample_accession'];prefix='ncbi_dataset/data/'+row['Assembly accession']+'/'
                members=[n for n in z.namelist() if n.startswith(prefix) and n.endswith('.fna')]
                r=dict(sample_accession=sample,assembly_accession=row['Assembly accession'],url=u,batch_zip=path.name)
                if len(members)!=1:r.update(status='FAIL_MISSING_OR_AMBIGUOUS_VERSIONED_FASTA',members=members)
                else:
                    name=members[0];dna=z.read(name)
                    if name in md5 and hashlib.md5(dna).hexdigest()!=md5[name]:raise ValueError('Archive supplied FASTA MD5 mismatch')
                    if not dna.startswith(b'>'):raise ValueError('FASTA header missing')
                    f=dest/'fastas'/(sample+'.fna');f.write_bytes(dna)
                    r.update(status='PASS_PUBLISHED_ASSEMBLY_VERSION',fasta_sha256=sha(dna),bytes=len(dna),zip_member=name,archive_md5_verified=name in md5)
                save(dest/'receipts'/(sample+'.json'),r)
            receipt['status']='BATCH_COMPLETE'
        except Exception as e:
            receipt.update(status='FAILED_BATCH',error=str(e))
            for row in rows:
                r=dest/'receipts'/(row['sample_accession']+'.json')
                if not r.exists():save(r,dict(sample_accession=row['sample_accession'],assembly_accession=row['Assembly accession'],status='FAILED_BATCH',error=str(e),url=u))
        save(dest/'batches'/(tag+'.json'),receipt);return receipt
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(fetch,b) for b in batches]):print(json.dumps(fut.result()),flush=True)
    records=[json.loads((dest/'receipts'/(s+'.json')).read_text()) for s in d.sample_accession]
    pd.DataFrame(records).to_csv(dest/'acquisition.csv',index=False)
    passed=sum(r['status']=='PASS_PUBLISHED_ASSEMBLY_VERSION' for r in records)
    result=dict(status='COMPLETE_ALL_VERSIONED_ASSEMBLIES' if passed==len(d) else 'COMPLETE_WITH_REPORTED_MISSING',target=len(d),passed=passed,missing=len(d)-passed,manifest_sha256=protocol['manifest_sha256'])
    save(dest/'status.json',result);print(json.dumps(result))

if __name__=='__main__':main()
