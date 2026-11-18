#!/usr/bin/env python3
"""Recover public E2 assemblies only when byte SHA matches the frozen manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,re,urllib.request,urllib.error,zipfile
from pathlib import Path
import pandas as pd


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(2**20):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=25);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    if sha(a.predictions)!='6b56e1eeb60421261d6933bf89897d0a10faae4ba7cea8399787b76008db9ef1':raise ValueError('Frozen E2 source differs')
    d=pd.read_csv(a.predictions,dtype=str,keep_default_na=False)
    if len(d)!=525 or not d.biosample_acc.is_unique or not d.asm_acc.is_unique:raise ValueError('Fixed cohort differs')
    for name in ['fastas','receipts','batch_zips']:(a.out/name).mkdir(parents=True,exist_ok=True)
    pending=[]
    for row in d.to_dict('records'):
        if not re.fullmatch(r'GCA_\d+\.\d+',row['asm_acc']):raise ValueError('Unexpected assembly accession')
        f=a.out/'fastas'/(row['biosample_acc']+'.fna');r=a.out/'receipts'/(row['biosample_acc']+'.json')
        if f.exists() and sha(f)==row['fasta_sha256']:continue
        if r.exists() and not a.retry_failed:continue
        pending.append(row)
    batches=[pending[i:i+a.batch_size] for i in range(0,len(pending),a.batch_size)]
    def job(rows):
        ids=','.join(row['asm_acc'] for row in rows)
        url='https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/'+ids+'/download?include_annotation_type=GENOME_FASTA'
        tag=hashlib.sha256(ids.encode()).hexdigest()[:16];path=a.out/'batch_zips'/(tag+'.zip');tmp=path.with_suffix('.part')
        metadata={'url':url,'started_utc':datetime.now(timezone.utc).isoformat(),'requested_accessions':[r['asm_acc'] for r in rows]}
        try:
            if not path.exists():
                with urllib.request.urlopen(url,timeout=90) as q,tmp.open('wb') as f:
                    if q.status!=200:raise ValueError('Incomplete HTTP response')
                    metadata['http_status']=q.status
                    while block:=q.read(2**20):f.write(block)
                tmp.replace(path)
            metadata['zip_sha256']=sha(path);metadata['zip_bytes']=path.stat().st_size
            with zipfile.ZipFile(path) as z:
                names=z.namelist()
                for row in rows:
                    sample=row['biosample_acc'];matches=[n for n in names if n.startswith('ncbi_dataset/data/'+row['asm_acc']+'/') and n.endswith('.fna')]
                    receipt={'sample_accession':sample,'asm_acc':row['asm_acc'],'expected_fasta_sha256':row['fasta_sha256'],'batch_zip':str(path),'url':url}
                    if len(matches)!=1:receipt.update(status='FAIL_MISSING_OR_AMBIGUOUS_FASTA',matches=matches)
                    else:
                        dna=z.read(matches[0]);actual=hashlib.sha256(dna).hexdigest();receipt.update(actual_fasta_sha256=actual,bytes=len(dna),zip_member=matches[0])
                        if actual!=row['fasta_sha256']:receipt['status']='FAIL_FROZEN_BYTE_HASH_MISMATCH'
                        else:
                            target=a.out/'fastas'/(sample+'.fna');target.write_bytes(dna);receipt['status']='PASS_EXACT_FROZEN_FASTA_BYTES'
                    receipt['completed_utc']=datetime.now(timezone.utc).isoformat();(a.out/'receipts'/(sample+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
            metadata['status']='BATCH_READ_COMPLETE'
        except Exception as error:
            metadata.update(status='FAIL_BATCH_ACQUISITION',error=str(error))
            for row in rows:
                sample=row['biosample_acc'];target=a.out/'receipts'/(sample+'.json')
                if not target.exists() or a.retry_failed:target.write_text(json.dumps({'sample_accession':sample,'asm_acc':row['asm_acc'],'status':'FAIL_BATCH_ACQUISITION','error':str(error),'expected_fasta_sha256':row['fasta_sha256'],'url':url},indent=2)+'\n')
        (a.out/'batch_zips'/(tag+'.receipt.json')).write_text(json.dumps(metadata,indent=2)+'\n');return metadata
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(job,b) for b in batches]
        for future in as_completed(futures):
            r=future.result();print(json.dumps({'batch':len(r['requested_accessions']),'status':r['status'],'error':r.get('error')}),flush=True)
    receipts=[]
    for row in d.to_dict('records'):
        p=a.out/'receipts'/(row['biosample_acc']+'.json')
        receipts.append(json.loads(p.read_text()) if p.exists() else {'sample_accession':row['biosample_acc'],'status':'NO_RECEIPT'})
    pd.DataFrame(receipts).to_csv(a.out/'acquisition.csv',index=False)
    n=sum(r['status']=='PASS_EXACT_FROZEN_FASTA_BYTES' for r in receipts)
    status={'status':'COMPLETE_EXACT_FROZEN_INPUT_RECOVERY' if n==525 else 'INCOMPLETE_INPUTS_RETAINED','target':525,'exact_byte_matches':n,'unverified':525-n,'original_manifest_sha256':sha(a.predictions),'no_sequence_or_label_edited':True}
    (a.out/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
