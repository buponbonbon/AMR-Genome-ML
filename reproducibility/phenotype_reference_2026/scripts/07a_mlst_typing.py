#!/usr/bin/env python3
"""File07a: resumable MLST typing for the 4,227-sample cohort."""
from __future__ import annotations
import argparse, csv, json, os, re, signal, subprocess, sys, tempfile, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

VERSION='1.0.0'; N=4227; SCHEME='klebsiella'
LOCI=('gapA','infB','mdh','pgi','phoE','rpoB','tonB')
STOP=False

def now(): return datetime.now(timezone.utc).isoformat()
def hsec(x): return f'{x:.1f}s' if x<60 else (f'{x/60:.1f}m' if x<3600 else f'{x/3600:.2f}h')
def sig(*_):
    global STOP; STOP=True

def atomic_text(p:Path,s:str):
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            f.write(s); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def atomic_csv(p:Path,rows,fields):
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

class Log:
    def __init__(self,p): self.p=p; p.parent.mkdir(parents=True,exist_ok=True)
    def __call__(self,m):
        line=f'[{datetime.now().strftime("%H:%M:%S")}] {m}'
        print(line,flush=True)
        with self.p.open('a',encoding='utf-8') as f: f.write(line+'\n')

def root_from(p:Path):
    p=p.resolve()
    for q in (p,*p.parents):
        if (q/'data').exists() and (q/'scripts').exists(): return q
    return p

def lock_acquire(p:Path):
    p.parent.mkdir(parents=True,exist_ok=True)
    try: fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'Lock exists: {p}')
    with os.fdopen(fd,'w') as f: json.dump({'pid':os.getpid(),'utc':now(),'version':VERSION},f)

def manifest_ids(p:Path):
    with p.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        if not r.fieldnames or 'Genome ID' not in r.fieldnames: raise RuntimeError("Manifest lacks 'Genome ID'.")
        ids=[str(x['Genome ID']).strip() for x in r]
    if len(ids)!=N: raise RuntimeError(f'Expected {N} cohort rows, found {len(ids)}.')
    if len(set(ids))!=len(ids) or any(not x for x in ids): raise RuntimeError('Invalid/duplicate Genome ID.')
    return ids

def preflight(mlst:Path,log):
    if not mlst.is_file(): raise RuntimeError(f'MLST executable not found: {mlst}')
    env=os.environ.copy(); env['PATH']=str(mlst.parent)+os.pathsep+env.get('PATH','')
    a=subprocess.run([str(mlst),'--version'],capture_output=True,text=True,env=env,timeout=30)
    txt=(a.stdout+'\n'+a.stderr)
    if a.returncode or '2.35.0' not in txt: raise RuntimeError('mlst 2.35.0 preflight failed.')
    b=subprocess.run([str(mlst),'--info'],capture_output=True,text=True,env=env,timeout=60)
    txt=b.stdout+'\n'+b.stderr
    line=next((x.strip() for x in txt.splitlines() if x.startswith(SCHEME+'\t') or x.startswith(SCHEME+' ')),None)
    if b.returncode or not line: raise RuntimeError(f"Scheme '{SCHEME}' not found.")
    log('MLST executable/version OK: 2.35.0'); log(f'MLST scheme OK: {line}')

def checkpoint_load(p:Path,valid:set[str],log):
    d={}; bad=0
    if p.exists():
        with p.open(encoding='utf-8') as f:
            for line in f:
                try: x=json.loads(line)
                except Exception: bad+=1; continue
                gid=str(x.get('Genome ID',''))
                if gid in valid and x.get('parse_ok') is True: d[gid]=x
    log(f'Checkpoint: {len(d):,} valid completed samples loaded.' + (f' Ignored {bad} malformed line(s).' if bad else ''))
    return d

def checkpoint_append(p:Path,recs):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        for x in recs: f.write(json.dumps(x,sort_keys=True)+'\n')
        f.flush(); os.fsync(f.fileno())

line_re=re.compile(r'^(.*?)\t([^\t]+)\t([^\t]+)(?:\t(.*))?$')
allele_re=re.compile(r'([A-Za-z0-9_.-]+)\(([^)]*)\)')
def gid_from_path(s):
    n=Path(s).name
    for z in ('.fna.gz','.fa.gz','.fasta.gz','.fna','.fa','.fasta'):
        if n.endswith(z): return n[:-len(z)]
    return n

def parse(stdout:str, expected:set[str]):
    out={}
    for line in stdout.splitlines():
        m=line_re.match(line.rstrip())
        if not m: continue
        gid=gid_from_path(m.group(1))
        if gid not in expected: continue
        scheme=m.group(2).strip(); st=m.group(3).strip(); raw=(m.group(4) or '').strip(); alleles=dict(allele_re.findall(raw))
        x={'Genome ID':gid,'scheme':scheme,'ST':st,'typed_exact':bool(scheme==SCHEME and re.fullmatch(r'\d+',st)),'parse_ok':True,'raw_alleles':raw,'completed_utc':now()}
        for loc in LOCI: x[loc]=alleles.get(loc,'')
        out[gid]=x
    return out

def run_chunk(chunk,mlst:Path,timeout):
    ids=[g for g,_ in chunk]; paths=[str(p) for _,p in chunk]
    env=os.environ.copy(); env['PATH']=str(mlst.parent)+os.pathsep+env.get('PATH','')
    t=time.time()
    try:
        p=subprocess.run([str(mlst),'--scheme',SCHEME,*paths],capture_output=True,text=True,env=env,timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return [],f'TIMEOUT {timeout}s: {str(e)[:1000]}',124,time.time()-t
    parsed=parse(p.stdout,set(ids)); miss=[x for x in ids if x not in parsed]
    diag=p.stderr.strip()
    if miss: diag+='\nMissing parsed: '+','.join(miss)+'\nstdout tail:\n'+'\n'.join(p.stdout.splitlines()[-12:])
    return list(parsed.values()),diag,p.returncode,time.time()-t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=4); ap.add_argument('--chunk-size',type=int,default=10)
    ap.add_argument('--limit',type=int); ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--mlst-bin',default=str(Path.home()/'micromamba/envs/mlst-typing/bin/mlst'))
    ap.add_argument('--project-root'); a=ap.parse_args()
    if not 1<=a.workers<=8: raise SystemExit('--workers must be 1..8')
    if not 1<=a.chunk_size<=100: raise SystemExit('--chunk-size must be 1..100')
    if a.limit is not None and a.limit<1: raise SystemExit('--limit must be >=1')
    root=Path(a.project_root).resolve() if a.project_root else root_from(Path.cwd())
    manifest=root/'data/processed/unitig_full_k31_input_manifest.csv'; genomes=root/'data/genomes'
    out=root/'data/processed/mlst_typing_results.csv'; ck=root/'checkpoints/mlst_typing'; log=Log(root/'logs/07a_mlst_typing.log')
    partial=ck/'mlst_results.partial.jsonl'; summary=ck/'file07a_final_summary.json'; failure=ck/'07a_last_failure.json'; lock=ck/'07a_mlst_typing.lock'
    stage='startup'; t0=time.time(); lock_acquire(lock)
    try:
        log(f'File07a script version: {VERSION}'); log(f'Project root: {root}'); log(f'Python: {sys.executable}'); log(f'Workers={a.workers}; chunk_size={a.chunk_size}')
        if not manifest.is_file(): raise RuntimeError(f'Missing manifest: {manifest}')
        stage='preflight'; mlst=Path(a.mlst_bin).expanduser().resolve(); preflight(mlst,log)
        ids=manifest_ids(manifest); fmap={g:genomes/f'{g}.fna.gz' for g in ids}; missing=[g for g,p in fmap.items() if not p.is_file()]
        if missing: raise RuntimeError(f'Missing {len(missing)} FASTA files; first: {missing[:10]}')
        log(f'Cohort/FASTA preflight PASS: {len(ids):,}/{N:,}')
        done=checkpoint_load(partial,set(ids),log); pending=[g for g in ids if g not in done]
        if pending:
            stage='typing'; sel=pending[:a.limit] if a.limit else pending; items=[(g,fmap[g]) for g in sel]; chunks=[items[i:i+a.chunk_size] for i in range(0,len(items),a.chunk_size)]
            log(f'Typing {len(sel):,} unfinished sample(s) in {len(chunks):,} chunk(s).')
            new=problems=0
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs={ex.submit(run_chunk,ch,mlst,a.timeout):ch for ch in chunks}
                for i,f in enumerate(as_completed(futs),1):
                    if STOP:
                        for q in futs: q.cancel()
                        break
                    ch=futs[f]
                    try: recs,diag,rc,elapsed=f.result()
                    except Exception as e: recs=[];diag=str(e);rc=1;elapsed=0
                    if recs:
                        checkpoint_append(partial,recs)
                        for x in recs: done[x['Genome ID']]=x
                        new+=len(recs)
                    miss=len(ch)-len(recs)
                    if rc or miss:
                        problems+=1; log(f'Chunk {i}/{len(chunks)} WARNING: parsed={len(recs)}/{len(ch)}, rc={rc}, time={hsec(elapsed)}')
                        if diag: log('Diagnostic tail:\n'+'\n'.join(diag.splitlines()[-8:]))
                    else: log(f'Chunk {i}/{len(chunks)} PASS: {len(recs)} samples; {hsec(elapsed)}; checkpoint={len(done):,}/{N:,}')
            log(f'This run added {new:,}; checkpoint={len(done):,}/{N:,}; problem chunks={problems}.')
        if STOP: raise KeyboardInterrupt
        if a.limit is not None and len(done)<N:
            log(f'FILE07a SMOKE/PARTIAL PASS: {len(done):,}/{N:,} checkpointed. Re-run without --limit to resume.'); return 0
        stage='final_audit'; miss=[g for g in ids if g not in done]
        if miss: raise RuntimeError(f'Typing incomplete: {len(miss)} missing. First: {miss[:20]}. Re-run to retry.')
        rows=[]
        for g in ids:
            x=done[g]; r={'Genome ID':g,'scheme':x.get('scheme',''),'ST':x.get('ST',''),'typed_exact':bool(x.get('typed_exact',False))}
            for loc in LOCI: r[loc]=x.get(loc,'')
            r['raw_alleles']=x.get('raw_alleles',''); rows.append(r)
        bad=sum(r['scheme']!=SCHEME for r in rows)
        if bad: raise RuntimeError(f'{bad} result(s) reported a non-{SCHEME} scheme.')
        fields=['Genome ID','scheme','ST','typed_exact',*LOCI,'raw_alleles']; atomic_csv(out,rows,fields)
        exact=sum(r['typed_exact'] for r in rows); uniq=len({r['ST'] for r in rows if r['typed_exact']})
        s={'script_version':VERSION,'status':'PASS','completed_utc':now(),'cohort_n':N,'scheme':SCHEME,'exact_numeric_ST_n':exact,'exact_numeric_ST_coverage':exact/N,'unresolved_or_novel_n':N-exact,'unique_exact_ST_n':uniq,'phenotype_used':False,'results_csv':str(out.relative_to(root)),'elapsed_seconds':time.time()-t0}
        atomic_text(summary,json.dumps(s,indent=2)+'\n')
        if failure.exists(): failure.unlink()
        log('='*72); log('FILE07a — MLST TYPING'); log('='*72); log(f'Cohort              : {N:,}'); log(f'Exact numeric ST    : {exact:,} ({exact/N:.2%})'); log(f'Unresolved/novel    : {N-exact:,}'); log(f'Unique exact STs    : {uniq:,}'); log('Phenotype used      : NO'); log(f'Final results       : {out}'); log(f'Elapsed             : {hsec(time.time()-t0)}'); log('FILE07a STATUS      : PASS'); log('='*72)
        return 0
    except KeyboardInterrupt:
        atomic_text(failure,json.dumps({'version':VERSION,'status':'INTERRUPTED','stage':stage,'utc':now()},indent=2)+'\n'); log('FILE07a INTERRUPTED. Checkpoint preserved; re-run to resume.'); return 130
    except Exception as e:
        atomic_text(failure,json.dumps({'version':VERSION,'status':'FAILED','stage':stage,'utc':now(),'error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n'); log(f'FILE07a FAILED\nStage: {stage}\nError: {e}\nCrash record: {failure}'); return 1
    finally:
        try: lock.unlink(missing_ok=True)
        except Exception: pass

if __name__=='__main__':
    signal.signal(signal.SIGINT,sig); signal.signal(signal.SIGTERM,sig); raise SystemExit(main())
