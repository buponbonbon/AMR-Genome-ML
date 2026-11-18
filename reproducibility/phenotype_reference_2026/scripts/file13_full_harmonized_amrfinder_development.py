#!/usr/bin/env python3
"""
FILE13 — full-development AMRFinder harmonization + model freeze.

Purpose: remove the mixed NCBI-precomputed/local-AMRFinder feature-generation
source in the 4,227-genome development cohort before a genuinely new E3 is
scored. FILE13 never reads E2 labels, predictions or metrics.

Primary redevelopment for E3 (pre-specified):
  - AMRFinderPlus 4.2.7, DB 2026-08-07.1, nucleotide mode,
    -O Klebsiella_pneumoniae, 1 thread/genome by default.
  - Keep Type==AMR and Subtype in {AMR, POINT, POINT_DISRUPT}.
  - Determinant keys are case-insensitive; per-genome duplicates collapse.
  - Rebuild the feature universe from development calls only; no phenotype
    filtering and no E2-guided feature selection.
  - Reuse FILE07 folds exactly.
  - LogisticRegression(C=1, penalty='l2', solver='liblinear',
    class_weight=None, max_iter=3000, random_state=20260920), threshold 0.5.
  - Freeze model/schema/hashes before FILE14/E3.

Checkpoint/resume:
  - one atomic raw TSV + .done.json per genome;
  - successful genomes are skipped on rerun;
  - protocol lock prevents mixing configurations;
  - failures are durable and rerunnable;
  - live progress and ETA.

Outputs:
  checkpoints/file13_harmonized_amrfinder/
  data/features/known_amr_harmonized/file13/
  data/evaluation/file13/
  models/file13/
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, os, re, shutil, subprocess, time, traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, roc_auc_score)

VERSION='1.0.0'; PATCH_LEVEL='resume-compat-20260923'; SEED=20260920
N=4227; R=1601; S=2626; LOCKED_NF=668
AMR_DEFAULT=Path('/home/khang/micromamba/envs/amrfinder-4.2.7/bin/amrfinder')
DB_DEFAULT=Path('/home/khang/amrfinder_db/2026-08-07.1')
AMR_VERSION='4.2.7'; DB_VERSION='2026-08-07.1'
SUBTYPES={'AMR','POINT','POINT_DISRUPT'}
PRIMARY='harmonized_rebuilt_primary'
SENS='harmonized_original668_projection'
REF='locked_original_668_reference'
SCHEMES={'random_stratified':'random_fold','genomic_cluster_aware':'genomic_cluster_fold','mlst_aware':'mlst_fold'}
BOOT_METRICS=['roc_auc','average_precision','balanced_accuracy','mcc']

# Narrow, documented legacy-local bridge exception established by independent audit.
# This does NOT change the scientific protocol/model; it only changes the Stage-5
# integrity gate from "100% exact historical reproduction or abort" to a verified
# documented mismatch path when every expected audit condition is satisfied.
LEGACY_BRIDGE_EXPECTED_IDS={
    '573.15486','573.15478','573.15452','573.15459',
    '573.15445','573.15377','573.15368','573.13949'
}
LEGACY_BRIDGE_EXPECTED_LOCAL_N=379
LEGACY_BRIDGE_EXPECTED_CHANGED_CELLS=132
LEGACY_BRIDGE_EXPECTED_ZERO_TO_ONE=132
LEGACY_BRIDGE_EXPECTED_ONE_TO_ZERO=0


def ts(): return datetime.now().strftime('%H:%M:%S')
def utc(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def log(s): print(f'[{ts()}] {s}', flush=True)
def shatext(s): return hashlib.sha256(s.encode()).hexdigest()
def shafile(p, block=1<<20):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(block), b''): h.update(b)
    return h.hexdigest()
def shafasta_content(p, block=1<<20):
    p=Path(p); h=hashlib.sha256()
    op=gzip.open if p.name.lower().endswith('.gz') else open
    with op(p,'rb') as f:
        for b in iter(lambda:f.read(block), b''): h.update(b)
    return h.hexdigest()
def atomic_text(p,s):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.tmp.{os.getpid()}'); q.write_text(s,encoding='utf-8'); os.replace(q,p)
def atomic_json(p,o): atomic_text(p,json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def atomic_csv(p,df):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.tmp.{os.getpid()}'); df.to_csv(q,index=False,encoding='utf-8-sig'); os.replace(q,p)
def normid(x): return '' if pd.isna(x) else str(x).strip()
def safe(g):
    a=re.sub(r'[^A-Za-z0-9._-]+','_',g).strip('._') or 'genome'; return a+'__'+hashlib.sha1(g.encode()).hexdigest()[:10]
def reqfile(p,label):
    if not Path(p).is_file(): raise FileNotFoundError(f'Missing {label}: {p}')
def reqdir(p,label):
    if not Path(p).is_dir(): raise FileNotFoundError(f'Missing {label}: {p}')


def load_dev(root):
    cp=root/'data/processed/file07_common_cohort.csv'; fp=root/'data/splits/file07_outer_folds.csv'
    reqfile(cp,'FILE07 cohort'); reqfile(fp,'FILE07 folds')
    c=pd.read_csv(cp,dtype={'Genome ID':str}); f=pd.read_csv(fp,dtype={'Genome ID':str})
    needc={'Genome ID','Phenotype','y','MLST'}; needf=needc|{'Genomic Cluster','random_fold','genomic_cluster_fold','mlst_fold'}
    if not needc<=set(c): raise RuntimeError(f'Cohort missing {sorted(needc-set(c))}')
    if not needf<=set(f): raise RuntimeError(f'Folds missing {sorted(needf-set(f))}')
    c['Genome ID']=c['Genome ID'].map(normid); f['Genome ID']=f['Genome ID'].map(normid)
    if c['Genome ID'].duplicated().any() or f['Genome ID'].duplicated().any(): raise RuntimeError('Duplicate FILE07 Genome IDs')
    y=pd.to_numeric(c.y).astype(int)
    if len(c)!=N or y.sum()!=R or (y==0).sum()!=S: raise RuntimeError(f'Unexpected development cohort n={len(c)} R={y.sum()} S={(y==0).sum()}')
    z=c[['Genome ID','y']].merge(f[['Genome ID','y']],on='Genome ID',suffixes=('_c','_f'),validate='one_to_one')
    if len(z)!=N or not (pd.to_numeric(z.y_c).to_numpy()==pd.to_numeric(z.y_f).to_numpy()).all(): raise RuntimeError('FILE07 cohort/fold mismatch')
    f=c[['Genome ID']].merge(f,on='Genome ID',validate='one_to_one')
    return c,f


def feature_col(df):
    for c in ['feature','Feature','feature_name','Feature Name','determinant','canonical_label','Element symbol']:
        if c in df: return c
    best=None
    for c in df:
        s=df[c].dropna().astype(str)
        if len(s)<50: continue
        score=s.str.contains(r'bla|omp|gyr|par|aac|aph|aad|qnr|sul|tet|cat|fos|erm|ram|mgr|^mcr',case=False,regex=True).mean()
        cand=(score,s.nunique(),c)
        if best is None or cand>best: best=cand
    if best is None: raise RuntimeError('Cannot identify feature column in known-AMR metadata')
    return best[2]


def load_locked(root,ids):
    mp=root/'data/features/known_amr/known_amr_feature_metadata_final.csv'; xp=root/'data/features/known_amr/X_known_amr_final.csv'
    reqfile(mp,'known-AMR metadata'); reqfile(xp,'known-AMR matrix')
    m=pd.read_csv(mp); fc=feature_col(m); feats=list(dict.fromkeys(str(x).strip() for x in m[fc].dropna()))
    if len(feats)!=LOCKED_NF: raise RuntimeError(f'Locked schema detected {len(feats)} features, expected {LOCKED_NF}; feature column={fc}')
    if len({x.casefold() for x in feats})!=len(feats): raise RuntimeError('Case-insensitive duplicate locked features')
    h=pd.read_csv(xp,nrows=0); idc='Genome ID' if 'Genome ID' in h else h.columns[0]
    x=pd.read_csv(xp,dtype={idc:str}); x[idc]=x[idc].map(normid)
    if x[idc].duplicated().any(): raise RuntimeError('Duplicate locked matrix IDs; string read guard failed')
    cmap={str(c).strip().casefold():c for c in x if c!=idc}
    miss=[f for f in feats if f.casefold() not in cmap]
    if miss: raise RuntimeError(f'Locked matrix missing metadata features: {miss[:20]}')
    cols=[cmap[f.casefold()] for f in feats]; w=x[[idc]+cols].copy(); w.columns=['Genome ID']+feats
    w=pd.DataFrame({'Genome ID':ids}).merge(w,on='Genome ID',how='left',validate='one_to_one')
    if w[feats].isna().any().any(): raise RuntimeError('Locked matrix missing one or more 4,227 development rows')
    for f in feats: w[f]=pd.to_numeric(w[f],errors='raise').astype(np.uint8)
    return feats,w


def isfasta(p): return any(p.name.lower().endswith(e) for e in ['.fa','.fna','.fasta','.fas','.ffn','.fa.gz','.fna.gz','.fasta.gz','.fas.gz','.ffn.gz'])
def external(p): return any(x in {q.lower() for q in p.parts} for x in ['external_validation','file11c','file11d','file11e','file11f','file14'])
def stripfa(n):
    if n.lower().endswith('.gz'): n=n[:-3]
    for e in ['.fasta','.fna','.fa','.fas','.ffn']:
        if n.lower().endswith(e): n=n[:-len(e)]; break
    return n[:-8] if n.endswith('_genomic') else n

def idcol(df):
    for c in ['Genome ID','genome_id','GenomeID','genome','ID']:
        if c in df: return c
    raise RuntimeError(f'No Genome ID column in {list(df)}')
def manifest_map(p,idset):
    h=pd.read_csv(p,nrows=0); ic='Genome ID' if 'Genome ID' in h else None
    if ic is None:
        try: ic=idcol(h)
        except: return {}
    d=pd.read_csv(p,dtype={ic:str}); lowers={str(c).lower().strip():c for c in d}
    pc=None
    for k in ['fasta','fasta_path','path','file','filepath','sequence_path','local_path','genome_path']:
        if k in lowers: pc=lowers[k]; break
    if pc is None:
        for c in d:
            if c==ic: continue
            s=d[c].dropna().astype(str)
            if len(s) and s.str.contains(r'\.(?:fa|fna|fasta|fas|ffn)(?:\.gz)?$',case=False,regex=True).mean()>.5: pc=c; break
    if pc is None: return {}
    out={}
    for _,r in d.iterrows():
        gid=normid(r[ic]); raw=str(r[pc]).strip()
        if gid not in idset or not raw: continue
        q=Path(raw).expanduser(); q=q if q.is_absolute() else (Path(p).parent/q).resolve()
        if q.is_file() and isfasta(q): out[gid]=q
    return out

def scanroot(root,idset):
    idx=defaultdict(list)
    for p in Path(root).rglob('*'):
        if p.is_file() and isfasta(p) and not external(p): idx[stripfa(p.name)].append(p)
    out={}
    for gid in idset:
        hits=[]
        for v in [gid,gid.replace('.','_'),gid.replace('.','-')]: hits+=idx.get(v,[])
        hits=sorted(set(hits),key=str)
        if len(hits)==1: out[gid]=hits[0]
        elif len(hits)>1:
            sizes={p.stat().st_size for p in hits}
            if len(sizes)==1: out[gid]=sorted(hits,key=lambda p:(len(str(p)),str(p)))[0]
            else: raise RuntimeError(f'Ambiguous FASTAs for {gid}: {[str(x) for x in hits[:5]]}')
    return out

def discover_fastas(root,ids,manifest,fastaroot):
    idset=set(ids); out={}; src={}
    if manifest:
        out=manifest_map(manifest.resolve(),idset); src={'mode':'explicit_manifest','path':str(manifest.resolve())}
    elif fastaroot:
        out=scanroot(fastaroot.resolve(),idset); src={'mode':'explicit_root','path':str(fastaroot.resolve())}
    else:
        cands=[]
        for base in [root/'data',root/'manifests']:
            if base.is_dir():
                for p in base.rglob('*.csv'):
                    if not external(p) and any(k in p.name.lower() for k in ['manifest','genome','fasta']): cands.append(p)
        best={}; bp=None
        for p in sorted(set(cands),key=lambda x:(0 if 'manifest' in x.name.lower() else 1,x.stat().st_size,str(x))):
            try: m=manifest_map(p,idset)
            except: continue
            if len(m)>len(best): best,bp=m,p
            if len(best)==len(idset): break
        out=best; src={'mode':'auto_manifest','path':str(bp) if bp else None}
        if len(out)<len(idset):
            roots=[root/'data/genomes',root/'data/raw/genomes',root/'data/processed/genomes',root/'data/fasta',root/'genomes',root]
            for rr in roots:
                if not rr.is_dir(): continue
                m=scanroot(rr,idset)
                for g,p in m.items():
                    if g in out and out[g]!=p: raise RuntimeError(f'FASTA discovery conflict {g}: {out[g]} vs {p}')
                    out[g]=p
                if len(out)==len(idset): src={'mode':'auto_manifest_plus_scan','manifest':src.get('path'),'scan_root':str(rr)}; break
    miss=[g for g in ids if g not in out]
    if miss: raise RuntimeError(f'FASTA discovery {len(out)}/{len(ids)}; missing {len(miss)} e.g. {miss[:25]}. Supply --fasta-manifest or --fasta-root.')
    return out,src



def reuse_locked_fasta_manifest_if_valid(path,ids):
    """Load the existing FILE13 FASTA manifest on resume, validating row/order/path/stat."""
    p=Path(path)
    if not p.is_file(): return None
    try:
        d=pd.read_csv(p,dtype={'Genome ID':str})
        need={'Genome ID','fasta_path','bytes','mtime_ns'}
        if not need<=set(d) or len(d)!=len(ids): return None
        d['Genome ID']=d['Genome ID'].map(normid)
        if d['Genome ID'].tolist()!=list(ids): return None
        out={}
        for _,r in d.iterrows():
            gid=normid(r['Genome ID']); q=Path(str(r['fasta_path'])).expanduser()
            if not q.is_file() or not isfasta(q): return None
            if int(r['bytes'])!=q.stat().st_size or int(r['mtime_ns'])!=q.stat().st_mtime_ns: return None
            out[gid]=q
        return out
    except Exception:
        return None

def amrver(exe):
    for arg in ['--version','-V']:
        try:
            r=subprocess.run([str(exe),arg],capture_output=True,text=True,timeout=30); t=(r.stdout+'\n'+r.stderr).strip()
            if r.returncode==0 and t: return t
        except: pass
    return ''
def valid_tsv(p):
    p=Path(p)
    if not p.is_file() or p.stat().st_size==0: return False,'missing/empty'
    try: head=p.open(encoding='utf-8',errors='replace').readline().rstrip('\r\n').split('\t')
    except Exception as e: return False,str(e)
    need={'Type','Subtype','Element symbol'}
    return (need<=set(head), 'ok' if need<=set(head) else f'missing {sorted(need-set(head))}')

def make_protocol(amr,db,version,workers,thr,ids_sha,fm_sha,fold_sha,schema_sha,source):
    return {'file':'FILE13','script_version':VERSION,'scientific_status':'DEVELOPMENT_REDEVELOPMENT_BEFORE_E3',
      'guardrails':{'E2_labels_read':False,'E2_predictions_read':False,'E2_metrics_read':False,'E2_used_for_selection':False,'E3_must_not_be_scored_until_freeze_complete':True,'no_threshold_tuning':True,'no_E2_guided_feature_selection':True,'all_pre_specified_models_reported':True},
      'development':{'n':N,'R':R,'S':S,'cohort_ids_sha256':ids_sha,'fasta_manifest_sha256':fm_sha,'folds_file_sha256':fold_sha},
      'fasta_discovery':source,
      'amrfinder':{'executable':str(amr),'version_text':version,'database':str(db),'organism':'Klebsiella_pneumoniae','mode':'nucleotide','threads_per_genome':thr,'parallel_genome_workers':workers},
      'parser':{'Type':'AMR','Subtype_in':sorted(SUBTYPES),'symbol_column':'Element symbol','normalization':'strip+casefold','per_genome_deduplication':True,'canonical_label':'most-common exact spelling; lexical tie-break','phenotype_used_for_feature_universe':False,'manual_emrD_special_case':False,'minimum_prevalence_filter':None},
      'locked_original_schema':{'feature_count':LOCKED_NF,'schema_sha256':schema_sha},
      'models':{'primary_for_E3':PRIMARY,'all_reported':[REF,SENS,PRIMARY],'estimator':'LogisticRegression','params':{'penalty':'l2','C':1.0,'solver':'liblinear','class_weight':None,'max_iter':3000,'random_state':SEED},'decision_threshold':0.5},
      'internal_validation':{'reuse_existing_FILE07_folds':True,'folds_recomputed':False,'schemes':SCHEMES,'feature_selection':None,'bootstrap_group':'Genomic Cluster'}}
def _protocol_core_for_resume(obj):
    """
    Scientific/checkpoint compatibility view.

    Excludes execution-only / discovery-provenance fields that do not alter an
    AMRFinder result: parallel worker count and the route used to rediscover the
    already-locked FASTA mapping. The actual locked FASTA manifest SHA remains
    part of the comparison, as do AMRFinder version/database, parser, folds,
    schema, model specification, and threads-per-genome.
    """
    import copy
    x=copy.deepcopy(obj)
    x.pop('protocol_sha256',None); x.pop('locked_utc',None)
    x.pop('fasta_discovery',None)
    if isinstance(x.get('amrfinder'),dict):
        x['amrfinder'].pop('parallel_genome_workers',None)
    return x


def lock_protocol(cp,obj,runtime_workers=None):
    p=cp/'file13_protocol_lock.json'
    h=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if p.is_file():
        old=json.loads(p.read_text())
        oh=old.get('protocol_sha256')
        if not oh:
            raise RuntimeError(f'Existing protocol lock lacks protocol_sha256: {p}')
        old_core=_protocol_core_for_resume(old)
        cur_core=_protocol_core_for_resume(obj)
        if old_core!=cur_core:
            # Give a compact field-level hint without silently accepting a real
            # scientific/input protocol change.
            diffs=[]
            for k in sorted(set(old_core)|set(cur_core)):
                if old_core.get(k)!=cur_core.get(k): diffs.append(k)
            raise RuntimeError(
                'Protocol mismatch with checkpoint in science/input-critical fields. '
                f'Existing={oh} current_recomputed={h}; differing top-level fields={diffs}. '
                'Do not reuse this checkpoint for an intentional scientific/input change.'
            )
        # Preserve the ORIGINAL SHA carried by all per-genome .done.json files.
        # Runtime worker count may differ on resume; it is execution-only and
        # does not alter the AMRFinder command for any individual genome.
        atomic_json(cp/'file13_resume_runtime.json',{
            'status':'RESUME_COMPATIBLE_WITH_EXISTING_PROTOCOL_LOCK',
            'time_utc':utc(),
            'patch_level':PATCH_LEVEL,
            'existing_protocol_sha256':oh,
            'runtime_parallel_genome_workers':runtime_workers,
            'locked_parallel_genome_workers':old.get('amrfinder',{}).get('parallel_genome_workers'),
            'note':'Worker count and FASTA rediscovery route are execution/provenance details; science/input-critical protocol fields and locked FASTA-manifest hash matched.'
        })
        return oh,p
    q=dict(obj); q['protocol_sha256']=h; q['locked_utc']=utc(); atomic_json(p,q)
    return h,p


def prep_fa(fa,tmp):
    if not fa.name.lower().endswith('.gz'): return fa,None
    tmp.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(fa,'rb') as a, tmp.open('wb') as b: shutil.copyfileobj(a,b,1<<20)
    return tmp,tmp

def run_one(gid,fa,raw,done,amr,db,threads,psha,tmpdir,timeout,retries,force):
    if not force and done.is_file() and raw.is_file():
        try:
            d=json.loads(done.read_text()); ok,_=valid_tsv(raw)
            if ok and d.get('status')=='PASS' and d.get('protocol_sha256')==psha: return {'Genome ID':gid,'status':'SKIP_CHECKPOINT','attempts':0,'elapsed_seconds':0,'raw_path':str(raw),'message':''}
        except: pass
    t0=time.time(); err=''
    for a in range(1,retries+2):
        work=tmpdir/safe(gid); work.mkdir(parents=True,exist_ok=True); tmpout=raw.with_name(raw.name+f'.tmp.{os.getpid()}.{a}')
        cleanup=None
        try:
            nuc,cleanup=prep_fa(fa,work/(safe(gid)+'.fna'))
            cmd=[str(amr),'-n',str(nuc),'-O','Klebsiella_pneumoniae','--database',str(db),'--threads',str(threads)]
            st=time.time()
            with tmpout.open('w',encoding='utf-8') as oh: r=subprocess.run(cmd,stdout=oh,stderr=subprocess.PIPE,text=True,timeout=timeout)
            if r.returncode!=0: raise RuntimeError(f'returncode={r.returncode}; stderr={r.stderr[-4000:]}')
            ok,why=valid_tsv(tmpout)
            if not ok: raise RuntimeError(f'invalid TSV: {why}')
            raw.parent.mkdir(parents=True,exist_ok=True); os.replace(tmpout,raw); sha=shafile(raw)
            atomic_json(done,{'Genome ID':gid,'status':'PASS','completed_utc':utc(),'protocol_sha256':psha,'fasta_path':str(fa),'raw_path':str(raw),'output_sha256':sha,'attempt':a,'elapsed_seconds':time.time()-st,'command':cmd,'stderr_tail':(r.stderr or '')[-4000:]})
            return {'Genome ID':gid,'status':'PASS','attempts':a,'elapsed_seconds':time.time()-t0,'raw_path':str(raw),'message':''}
        except Exception as e:
            err=f'{type(e).__name__}: {e}'
            try: tmpout.unlink(missing_ok=True)
            except: pass
            if a<=retries: time.sleep(min(10,2*a))
        finally:
            if cleanup:
                try: cleanup.unlink(missing_ok=True)
                except: pass
    return {'Genome ID':gid,'status':'FAIL','attempts':retries+1,'elapsed_seconds':time.time()-t0,'raw_path':str(raw),'message':err}

def annotate(ids,fmap,rawdir,donedir,tmpdir,cp,amr,db,workers,threads,psha,timeout,retries,force):
    rows=[]; start=time.time(); n=len(ids); fresh=0
    def ck():
        if rows: atomic_csv(cp/'file13_annotation_status.csv',pd.DataFrame(rows))
        atomic_json(cp/'file13_annotation_state.json',{'updated_utc':utc(),'protocol_sha256':psha,'completed':len(rows),'total':n,'status_counts':Counter(r['status'] for r in rows),'elapsed_seconds':time.time()-start})
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(run_one,g,fmap[g],rawdir/(safe(g)+'.tsv'),donedir/(safe(g)+'.done.json'),amr,db,threads,psha,tmpdir,timeout,retries,force):g for g in ids}
        try:
            for f in as_completed(fut):
                g=fut[f]
                try: r=f.result()
                except Exception as e: r={'Genome ID':g,'status':'FAIL','attempts':0,'elapsed_seconds':0,'raw_path':'','message':f'worker:{e}'}
                rows.append(r); fresh+=int(r['status']=='PASS')
                if r['status']=='FAIL': log(f"[FAIL] {g} | {r['message']}")
                k=len(rows)
                if k==1 or k%25==0 or k==n:
                    elapsed=time.time()-start; rate=fresh/max(elapsed,1e-9); eta=(n-k)/rate if rate else math.nan; c=Counter(x['status'] for x in rows)
                    log(f"[PROGRESS] AMRFinder {k}/{n} ({100*k/n:5.1f}%) | PASS={c['PASS']} SKIP={c['SKIP_CHECKPOINT']} FAIL={c['FAIL']} | elapsed={elapsed/3600:.2f}h | ETA={(eta/3600 if math.isfinite(eta) else float('nan')):.2f}h")
                    ck()
        except KeyboardInterrupt:
            ck(); log('[INTERRUPTED] checkpoints preserved; rerun same command'); raise
    ck(); d=pd.DataFrame(rows); bad=d[d.status=='FAIL']
    if len(bad): atomic_csv(cp/'file13_annotation_failures.csv',bad); raise RuntimeError(f'{len(bad)} genomes failed; rerun same command after fixing cause')
    return d


def parse_tsv(p):
    d=pd.read_csv(p,sep='\t',dtype=str,keep_default_na=False); need={'Type','Subtype','Element symbol'}
    if not need<=set(d): raise RuntimeError(f'{p} missing {sorted(need-set(d))}')
    q=d[d.Type.str.strip().eq('AMR') & d.Subtype.str.strip().isin(SUBTYPES)]; out=[]; seen=set()
    for x in q['Element symbol']:
        s=str(x).strip(); k=s.casefold()
        if s and k not in seen: seen.add(k); out.append(s)
    return out

def build_calls(ids,rawdir):
    calls={}; spell=defaultdict(Counter); st=time.time()
    for i,g in enumerate(ids,1):
        p=rawdir/(safe(g)+'.tsv'); ok,w=valid_tsv(p)
        if not ok: raise RuntimeError(f'invalid raw output {g}: {w}')
        ks=set()
        for s in parse_tsv(p): ks.add(s.casefold()); spell[s.casefold()][s]+=1
        calls[g]=ks
        if i%250==0 or i==len(ids): log(f'[PROGRESS] parse {i}/{len(ids)} | determinant keys={len(spell)} | elapsed={time.time()-st:.1f}s')
    canon={k:sorted([s for s,n in c.items() if n==max(c.values())])[0] for k,c in spell.items()}
    return calls,canon

def matrices(ids,calls,canon,locked):
    keys=sorted(canon); labels=[canon[k] for k in keys]; ix={k:j for j,k in enumerate(keys)}; a=np.zeros((len(ids),len(keys)),np.uint8)
    for i,g in enumerate(ids):
        for k in calls[g]:
            if k in ix: a[i,ix[k]]=1
    rb=pd.DataFrame(a,columns=labels); rb.insert(0,'Genome ID',ids)
    counts=a.sum(0).astype(int); meta=pd.DataFrame({'normalized_key':keys,'feature':labels,'development_count':counts,'development_prevalence':counts/len(ids)})
    lkeys=[f.casefold() for f in locked]; lix={k:j for j,k in enumerate(lkeys)}; b=np.zeros((len(ids),len(locked)),np.uint8)
    for i,g in enumerate(ids):
        for k in calls[g]:
            if k in lix: b[i,lix[k]]=1
    pr=pd.DataFrame(b,columns=locked); pr.insert(0,'Genome ID',ids); meta['member_of_original_locked_668']=meta.normalized_key.isin(set(lkeys))
    return rb,meta,pr

def local_ids(root):
    p=root/'data/features/known_amr/amrfinder_local_completion_manifest.csv'
    if not p.is_file(): return set()
    h=pd.read_csv(p,nrows=0); ic='Genome ID' if 'Genome ID' in h else h.columns[0]; d=pd.read_csv(p,dtype={ic:str}); return {normid(x) for x in d[ic] if normid(x)}
def bridge(locked,proj,feats,locals_):
    a=locked.set_index('Genome ID')[feats]; b=proj.set_index('Genome ID')[feats]
    aa=a.to_numpy(np.uint8); bb=b.to_numpy(np.uint8); dif=aa!=bb
    rows=[{'feature':f,
           'locked_prevalence':aa[:,j].mean(),
           'harmonized_projection_prevalence':bb[:,j].mean(),
           'prevalence_delta_projection_minus_locked':bb[:,j].mean()-aa[:,j].mean(),
           'changed_cells':int(dif[:,j].sum())} for j,f in enumerate(feats)]
    ids=[g for g in a.index if g in locals_]; idx=[a.index.get_loc(g) for g in ids]
    ld=dif[idx] if idx else None
    lold=aa[idx] if idx else None; lnew=bb[idx] if idx else None
    bad_ids=[]
    z01=z10=None
    if ld is not None:
        bad_ids=[ids[i] for i,v in enumerate(ld.sum(1)) if int(v)>0]
        z01=int(((lold==0)&(lnew==1)).sum())
        z10=int(((lold==1)&(lnew==0)).sum())
    summ={
      'all_development':{
        'n':len(a),
        'changed_genomes':int((dif.sum(1)>0).sum()),
        'changed_cells':int(dif.sum()),
        'total_cells':int(dif.size),
        'changed_cell_fraction':float(dif.mean())
      },
      'original_local_branch_intersection':{
        'n':len(ids),
        'changed_genomes':int((ld.sum(1)>0).sum()) if ld is not None else None,
        'changed_cells':int(ld.sum()) if ld is not None else None,
        'total_cells':int(ld.size) if ld is not None else None,
        'changed_cell_fraction':float(ld.mean()) if ld is not None else None,
        'zero_to_one':z01,
        'one_to_zero':z10,
        'mismatched_genome_ids':bad_ids,
        'exact_reproduction':bool(ld.sum()==0) if ld is not None else None
      }
    }
    return pd.DataFrame(rows).sort_values(['changed_cells','feature'],ascending=[False,True]),summ


def stage3_complete(ids,rawdir,donedir,psha,verify_output_sha=True):
    """Return (ok, details). Never mutates checkpoints."""
    bad=[]; counts=Counter()
    for g in ids:
        rp=rawdir/(safe(g)+'.tsv'); dp=donedir/(safe(g)+'.done.json')
        if not rp.is_file() or not dp.is_file():
            bad.append({'Genome ID':g,'reason':'missing raw or done'}); continue
        ok,why=valid_tsv(rp)
        if not ok:
            bad.append({'Genome ID':g,'reason':f'invalid raw: {why}'}); continue
        try: d=json.loads(dp.read_text())
        except Exception as e:
            bad.append({'Genome ID':g,'reason':f'bad done json: {e}'}); continue
        if d.get('status')!='PASS':
            bad.append({'Genome ID':g,'reason':f"done status={d.get('status')}"}); continue
        if d.get('protocol_sha256')!=psha:
            bad.append({'Genome ID':g,'reason':'protocol mismatch'}); continue
        if verify_output_sha:
            want=d.get('output_sha256')
            if not want or shafile(rp)!=want:
                bad.append({'Genome ID':g,'reason':'raw output SHA mismatch'}); continue
        counts['valid']+=1
    return len(bad)==0, {'valid':counts['valid'],'total':len(ids),'bad':bad[:25],'bad_count':len(bad)}


def load_stage4_if_complete(ids,fd,feats):
    """Load already-built Stage-4 matrices only if they are structurally coherent."""
    rbp=fd/'X_known_amr_harmonized_rebuilt.csv'
    prp=fd/'X_known_amr_harmonized_original668_projection.csv'
    metap=fd/'file13_harmonized_feature_metadata.csv'
    if not (rbp.is_file() and prp.is_file() and metap.is_file()):
        return None
    try:
        rebuilt=pd.read_csv(rbp,dtype={'Genome ID':str})
        proj=pd.read_csv(prp,dtype={'Genome ID':str})
        meta=pd.read_csv(metap)
        for d in (rebuilt,proj):
            d['Genome ID']=d['Genome ID'].map(normid)
        if len(rebuilt)!=len(ids) or len(proj)!=len(ids):
            return None
        if rebuilt['Genome ID'].tolist()!=ids or proj['Genome ID'].tolist()!=ids:
            return None
        if list(proj.columns[1:])!=list(feats):
            return None
        if 'feature' not in meta or 'normalized_key' not in meta:
            return None
        pfs=meta.feature.astype(str).tolist()
        if list(rebuilt.columns[1:])!=pfs or len(pfs)!=len(set(x.casefold() for x in pfs)):
            return None
        if len(meta)!=len(pfs):
            return None
        return rebuilt,meta,proj,rbp,prp,metap
    except Exception:
        return None


def verify_documented_legacy_bridge(root,drive_root,fmap,rawdir,donedir,psha,bs):
    """
    Strict verification for the already-investigated 8-genome legacy-local mismatch.
    It passes only if ALL expected counts/directions/IDs match AND archived raw TSVs
    are byte-identical local-vs-Drive AND FILE13 input FASTAs are sequence-content
    identical to the archived Drive FASTAs AND all 8 done.json files share psha.
    """
    lb=bs['original_local_branch_intersection']
    checks={
      'local_n_exact': lb.get('n')==LEGACY_BRIDGE_EXPECTED_LOCAL_N,
      'changed_genomes_exact': lb.get('changed_genomes')==len(LEGACY_BRIDGE_EXPECTED_IDS),
      'changed_cells_exact': lb.get('changed_cells')==LEGACY_BRIDGE_EXPECTED_CHANGED_CELLS,
      'zero_to_one_exact': lb.get('zero_to_one')==LEGACY_BRIDGE_EXPECTED_ZERO_TO_ONE,
      'one_to_zero_exact': lb.get('one_to_zero')==LEGACY_BRIDGE_EXPECTED_ONE_TO_ZERO,
      'mismatch_ids_exact': set(lb.get('mismatched_genome_ids',[]))==LEGACY_BRIDGE_EXPECTED_IDS,
    }
    evidence={}
    if drive_root is None:
        checks['drive_root_supplied']=False
        return False, {'status':'FAIL_NO_DRIVE_ROOT','checks':checks,'evidence':evidence}
    drive=Path(drive_root).expanduser().resolve()
    checks['drive_root_supplied']=drive.is_dir()
    if not drive.is_dir():
        return False, {'status':'FAIL_DRIVE_ROOT_NOT_FOUND','checks':checks,'evidence':evidence,'drive_root':str(drive)}

    all_raw=True; all_fasta=True; all_done=True
    for gid in sorted(LEGACY_BRIDGE_EXPECTED_IDS):
        old_local=root/'data/features/known_amr/raw_amrfinder'/f'{gid}.amrfinder.tsv'
        old_drive=drive/'data/features/known_amr/raw_amrfinder'/f'{gid}.amrfinder.tsv'
        drive_fa=drive/'data/genomes/fasta_gz'/f'{gid}.fna.gz'
        file13_fa=Path(fmap[gid])
        dp=donedir/(safe(gid)+'.done.json')
        rec={'old_local':str(old_local),'old_drive':str(old_drive),
             'file13_fasta':str(file13_fa),'drive_fasta':str(drive_fa),
             'done_json':str(dp)}
        try:
            rec['old_local_sha256']=shafile(old_local)
            rec['old_drive_sha256']=shafile(old_drive)
            rec['archived_raw_byte_identical']=rec['old_local_sha256']==rec['old_drive_sha256']
        except Exception as e:
            rec['archived_raw_byte_identical']=False; rec['raw_error']=str(e)
        all_raw &= bool(rec['archived_raw_byte_identical'])
        try:
            rec['file13_fasta_content_sha256']=shafasta_content(file13_fa)
            rec['drive_fasta_content_sha256']=shafasta_content(drive_fa)
            rec['fasta_content_identical']=rec['file13_fasta_content_sha256']==rec['drive_fasta_content_sha256']
        except Exception as e:
            rec['fasta_content_identical']=False; rec['fasta_error']=str(e)
        all_fasta &= bool(rec['fasta_content_identical'])
        try:
            d=json.loads(dp.read_text())
            rec['done_protocol_sha256']=d.get('protocol_sha256')
            rec['done_protocol_matches']=d.get('status')=='PASS' and d.get('protocol_sha256')==psha
        except Exception as e:
            rec['done_protocol_matches']=False; rec['done_error']=str(e)
        all_done &= bool(rec['done_protocol_matches'])
        evidence[gid]=rec
    checks['all_8_archived_raw_byte_identical']=all_raw
    checks['all_8_fasta_content_identical']=all_fasta
    checks['all_8_done_protocol_matches']=all_done
    ok=all(checks.values())
    return ok, {
      'status':'PASS_WITH_DOCUMENTED_LEGACY_LOCAL_MISMATCH' if ok else 'FAIL_LEGACY_BRIDGE_VERIFICATION',
      'checks':checks,
      'evidence':evidence,
      'expected':{
        'local_n':LEGACY_BRIDGE_EXPECTED_LOCAL_N,
        'mismatched_genome_ids':sorted(LEGACY_BRIDGE_EXPECTED_IDS),
        'changed_cells':LEGACY_BRIDGE_EXPECTED_CHANGED_CELLS,
        'zero_to_one':LEGACY_BRIDGE_EXPECTED_ZERO_TO_ONE,
        'one_to_zero':LEGACY_BRIDGE_EXPECTED_ONE_TO_ZERO
      }
    }


def model(): return LogisticRegression(penalty='l2',C=1.0,solver='liblinear',class_weight=None,max_iter=3000,random_state=SEED)
def metrics(y,p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,pred),'mcc':matthews_corrcoef(y,pred),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'f1':f1_score(y,pred,zero_division=0),'accuracy':accuracy_score(y,pred),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn)}
def cvpred(x,y,folds,ids,rep,scheme):
    fa=folds.astype(str).to_numpy(); us=list(dict.fromkeys(fa));
    if len(us)!=5: raise RuntimeError(f'{scheme}: expected 5 folds, got {us}')
    p=np.full(len(y),np.nan); fr=[]
    for u in us:
        te=fa==u; tr=~te; m=model(); m.fit(x[tr],y[tr]); pp=m.predict_proba(x[te])[:,1]; p[te]=pp; fr.append({'representation':rep,'scheme':scheme,'fold':u,'n_train':int(tr.sum()),'n_test':int(te.sum()),**metrics(y[te],pp)})
    if np.isnan(p).any(): raise RuntimeError('Incomplete OOF predictions')
    return pd.DataFrame({'Genome ID':ids,'representation':rep,'scheme':scheme,'fold':fa,'y':y,'probability_R':p,'predicted_R':(p>=.5).astype(int)}),pd.DataFrame(fr),metrics(y,p)
def boot(y,p,groups,n,seed,label):
    g=groups.fillna('UNRESOLVED').astype(str).to_numpy(); ug=np.array(pd.unique(g),dtype=object); mem={u:np.flatnonzero(g==u) for u in ug}; rng=np.random.default_rng(seed); rows=[]
    for r in range(1,n+1):
        sm=rng.choice(ug,size=len(ug),replace=True); ix=np.concatenate([mem[u] for u in sm]); yy=y[ix]
        if len(np.unique(yy))<2: continue
        mm=metrics(yy,p[ix]); rows.append({'replicate':r,**{k:mm[k] for k in BOOT_METRICS}})
        if r==1 or r%250==0 or r==n: log(f'[PROGRESS] bootstrap {label} {r}/{n} | valid={len(rows)}')
    return pd.DataFrame(rows)
def bootsum(point,b,rep,scheme):
    return pd.DataFrame([{'representation':rep,'scheme':scheme,'metric':k,'point_estimate':point[k],'bootstrap_mean':b[k].mean(),'ci95_low':b[k].quantile(.025),'ci95_high':b[k].quantile(.975),'valid_replicates':len(b)} for k in BOOT_METRICS])
def freeze(xdf,y,features,name,mdir,psha):
    m=model(); m.fit(xdf[features].to_numpy(float),y); bundle={'model':m,'feature_names':features,'threshold':.5,'model_name':name,'protocol_sha256':psha,'script_version':VERSION,'trained_n':len(y),'trained_R':int((y==1).sum()),'trained_S':int((y==0).sum()),'created_utc':utc()}
    fn='file13_harmonized_primary_model.joblib' if name==PRIMARY else 'file13_harmonized_original668_sensitivity_model.joblib'; p=mdir/fn; joblib.dump(bundle,p); c=p.with_name(p.stem+'_coefficients.csv'); pd.DataFrame({'feature':features,'coefficient':m.coef_.ravel(),'abs_coefficient':abs(m.coef_.ravel())}).sort_values('abs_coefficient',ascending=False).to_csv(c,index=False)
    return {'model_name':name,'model_path':str(p),'model_sha256':shafile(p),'coefficient_path':str(c),'coefficient_sha256':shafile(c),'intercept':float(m.intercept_[0]),'n_features':len(features)}


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'a.tsv'; p.write_text('Type\tSubtype\tElement symbol\nAMR\tAMR\tblaKPC-2\nAMR\tPOINT\tgyrA_S83I\nAMR\tSTRESS\temrD\nAMR\tAMR\tblaKPC-2\n')
        assert valid_tsv(p)[0]; assert parse_tsv(p)==['blaKPC-2','gyrA_S83I']
    y=np.array([0,0,1,1]); p=np.array([.1,.4,.6,.9]); assert abs(metrics(y,p)['roc_auc']-1)<1e-12
    print('FILE13 self-test: PASS'); return 0


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,default=Path('/mnt/c/Users/ASUS/Desktop/AMR-Genome-ML')); ap.add_argument('--fasta-manifest',type=Path); ap.add_argument('--fasta-root',type=Path); ap.add_argument('--amrfinder',type=Path,default=AMR_DEFAULT); ap.add_argument('--amrfinder-database',type=Path,default=DB_DEFAULT); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--amrfinder-threads',type=int,default=1); ap.add_argument('--timeout-seconds',type=int,default=1800); ap.add_argument('--retries',type=int,default=2); ap.add_argument('--bootstrap-replicates',type=int,default=2000); ap.add_argument('--checkpoint-dir',type=Path); ap.add_argument('--force-rerun',action='store_true'); ap.add_argument('--allow-tool-mismatch',action='store_true'); ap.add_argument('--allow-bridge-mismatch',action='store_true'); ap.add_argument('--legacy-drive-root',type=Path,help='Mounted historical AMR-Genome-ML Drive root used only to verify the documented Stage-5 legacy-local mismatch.'); ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
    if a.self_test: return selftest()
    root=a.project_root.resolve(); cp=(a.checkpoint_dir.resolve() if a.checkpoint_dir else root/'checkpoints/file13_harmonized_amrfinder'); fd=root/'data/features/known_amr_harmonized/file13'; ed=root/'data/evaluation/file13'; md=root/'models/file13'; raw=fd/'raw_amrfinder'; done=cp/'per_genome_done'; tmp=cp/'tmp'
    for d in [cp,fd,ed,md,raw,done,tmp]: d.mkdir(parents=True,exist_ok=True)
    failjson=cp/'file13_last_failure.json'
    try:
        log('='*110); log('FILE13 — FULL DEVELOPMENT AMRFINDER HARMONIZATION + MODEL FREEZE'); log('='*110); log(f'Version       : {VERSION}'); log(f'Project root  : {root}'); log(f'Workers       : {a.workers}'); log(f'Bootstrap     : {a.bootstrap_replicates}'); log('E2 access     : PROHIBITED BY DESIGN'); log('E3 scoring    : PROHIBITED UNTIL FREEZE_COMPLETE')
        log('Stage 1/9 — load development cohort/folds and locked 668 schema')
        cohort,folds=load_dev(root); ids=cohort['Genome ID'].tolist(); y=pd.to_numeric(cohort.y).astype(int).to_numpy(); feats,locked=load_locked(root,ids)
        lschema=pd.DataFrame({'feature_index':range(len(feats)),'feature':feats,'normalized_key':[x.casefold() for x in feats]}); lsp=fd/'file13_locked_original_668_schema.csv'; lschema.to_csv(lsp,index=False); lsha=shafile(lsp); log(f'[PROGRESS] development n={len(ids)} R={y.sum()} S={(y==0).sum()} | locked features={len(feats)}')
        log('Stage 2/9 — discover/reuse FASTAs and validate existing protocol lock BEFORE annotation')
        fmp=fd/'file13_development_fasta_manifest.csv'
        existing_lock=cp/'file13_protocol_lock.json'
        fmap=None; source=None
        # On resume, prefer the already-written FILE13 manifest. This avoids a
        # mutable auto-discovery route (including accidentally rediscovering the
        # FILE13 manifest itself) from changing protocol provenance/hash.
        if existing_lock.is_file() and not a.fasta_manifest and not a.fasta_root:
            fmap=reuse_locked_fasta_manifest_if_valid(fmp,ids)
            if fmap is not None:
                try:
                    source=json.loads(existing_lock.read_text()).get('fasta_discovery',{'mode':'resume_locked_manifest'})
                except Exception:
                    source={'mode':'resume_locked_manifest'}
                log(f'[PASS] Stage 2 FASTA mapping reused from locked FILE13 manifest — {len(fmap)}/{len(ids)} paths/stat records verified')
        if fmap is None:
            fmap,source=discover_fastas(root,ids,a.fasta_manifest,a.fasta_root)
            fm=pd.DataFrame([{'Genome ID':g,'fasta_path':str(fmap[g].resolve()),'bytes':fmap[g].stat().st_size,'mtime_ns':fmap[g].stat().st_mtime_ns,'compressed_gzip':fmap[g].name.lower().endswith('.gz')} for g in ids])
            fm.to_csv(fmp,index=False)
        amr=a.amrfinder.expanduser().resolve(); db=a.amrfinder_database.expanduser().resolve(); reqfile(amr,'AMRFinder'); reqdir(db,'AMRFinder DB'); ver=amrver(amr)
        if not a.allow_tool_mismatch:
            if AMR_VERSION not in ver: raise RuntimeError(f'Expected AMRFinder {AMR_VERSION}; observed {ver}')
            if db.name!=DB_VERSION: raise RuntimeError(f'Expected DB {DB_VERSION}; observed {db.name}')
        prot=make_protocol(amr,db,ver,a.workers,a.amrfinder_threads,shatext('\n'.join(ids)+'\n'),shafile(fmp),shafile(root/'data/splits/file07_outer_folds.csv'),lsha,source)
        psha,plock=lock_protocol(cp,prot,runtime_workers=a.workers)
        log(f'[PROGRESS] FASTAs {len(fmap)}/{len(ids)} | source={source}')
        log(f'[PROGRESS] protocol LOCKED/RESUME-COMPATIBLE | sha256={psha}')
        log('Stage 3/9 — all-development AMRFinder annotation (checkpoint/resume)')
        s3ok,s3detail=stage3_complete(ids,raw,done,psha,verify_output_sha=True) if not a.force_rerun else (False,{'forced':True})
        if s3ok:
            log(f"[PASS] Stage 3 already complete — {s3detail['valid']}/{s3detail['total']} raw outputs + done.json + SHA + protocol verified; no AMRFinder rerun")
        else:
            if not a.force_rerun and s3detail.get('bad_count'):
                log(f"[PROGRESS] Stage 3 checkpoint incomplete/invalid for {s3detail['bad_count']} genomes; resume only those needing work")
            annotate(ids,fmap,raw,done,tmp,cp,amr,db,a.workers,a.amrfinder_threads,psha,a.timeout_seconds,a.retries,a.force_rerun)
            s3ok2,s3detail2=stage3_complete(ids,raw,done,psha,verify_output_sha=True)
            if not s3ok2: raise RuntimeError(f"Stage 3 post-annotation integrity check failed: {s3detail2}")
            log('[PASS] Stage 3 complete — all 4,227 AMRFinder outputs + checkpoints verified')

        log('Stage 4/9 — parse calls and rebuild label-independent harmonized matrices')
        cached4=None if a.force_rerun else load_stage4_if_complete(ids,fd,feats)
        if cached4 is not None:
            rebuilt,meta,proj,rbp,prp,metap=cached4
            meta['member_of_original_locked_668']=meta.normalized_key.isin({x.casefold() for x in feats})
            log(f"[PASS] Stage 4 already complete — reuse cached harmonized matrices | harmonized features={len(meta)} | projection={len(feats)}")
        else:
            calls,canon=build_calls(ids,raw)
            rebuilt,meta,proj=matrices(ids,calls,canon,feats)
            meta['member_of_original_locked_668']=meta.normalized_key.isin({x.casefold() for x in feats})
            rbp=fd/'X_known_amr_harmonized_rebuilt.csv'; prp=fd/'X_known_amr_harmonized_original668_projection.csv'; metap=fd/'file13_harmonized_feature_metadata.csv'
            rebuilt.to_csv(rbp,index=False); proj.to_csv(prp,index=False); meta.to_csv(metap,index=False)
            log(f"[PASS] Stage 4 complete — harmonized features={len(meta)} | new_vs_locked={(~meta.member_of_original_locked_668).sum()} | locked_not_observed={len({x.casefold() for x in feats}-set(meta.normalized_key))}")

        log('Stage 5/9 — harmonization bridge audit')
        fa,bs=bridge(locked,proj,feats,local_ids(root))
        fa.to_csv(fd/'file13_harmonization_feature_change_audit.csv',index=False)
        lb=bs['original_local_branch_intersection']
        log(f"[PROGRESS] original local branch n={lb['n']} | changed_genomes={lb['changed_genomes']} | changed_cells={lb['changed_cells']} | 0->1={lb.get('zero_to_one')} | 1->0={lb.get('one_to_zero')}")
        legacy_verification=None
        if lb['n'] and lb['exact_reproduction'] is False:
            legacy_ok,legacy_verification=verify_documented_legacy_bridge(root,a.legacy_drive_root,fmap,raw,done,psha,bs)
            atomic_json(fd/'file13_legacy_bridge_verification.json',legacy_verification)
            if legacy_ok:
                bs['bridge_status']='PASS_WITH_DOCUMENTED_LEGACY_LOCAL_MISMATCH'
                bs['legacy_verification_path']=str(fd/'file13_legacy_bridge_verification.json')
                log('[PASS_WITH_DOCUMENTED_LEGACY_LOCAL_MISMATCH] Stage 5 verified: 371/379 exact; 8 documented genomes; 132 cells all 0->1; archived raw + FASTA identity + protocol checks PASS')
            elif a.allow_bridge_mismatch:
                bs['bridge_status']='OVERRIDDEN_UNVERIFIED_BRIDGE_MISMATCH'
                bs['legacy_verification_path']=str(fd/'file13_legacy_bridge_verification.json')
                log('[WARNING] Stage 5 bridge mismatch overridden WITHOUT full documented verification')
            else:
                atomic_json(fd/'file13_harmonization_bridge_summary.json',bs)
                raise RuntimeError('Stage 5 legacy bridge mismatch did not pass strict verification. Supply --legacy-drive-root pointing to the mounted historical AMR-Genome-ML Drive copy; do not use --allow-bridge-mismatch unless intentionally accepting an unverified override.')
        else:
            bs['bridge_status']='PASS_EXACT_LOCAL_REPRODUCTION'
            log('[PASS] Stage 5 exact historical local-branch reproduction')
        atomic_json(fd/'file13_harmonization_bridge_summary.json',bs)
        log('Stage 6/9 — internal CV using existing FILE07 folds; report all pre-specified representations')
        reps={REF:(locked,feats),SENS:(proj,feats),PRIMARY:(rebuilt,meta.feature.astype(str).tolist())}; preds=[]; foldms=[]; points=[]; bsum=[]; groups=folds['Genomic Cluster'].astype(str)
        for ri,(rn,(xdf,fs)) in enumerate(reps.items(),1):
            X=xdf[fs].to_numpy(float)
            for si,(sn,fc) in enumerate(SCHEMES.items(),1):
                log(f'[PROGRESS] CV {rn}/{sn} | features={len(fs)}'); pdx,fmtr,pt=cvpred(X,y,folds[fc],ids,rn,sn); preds.append(pdx); foldms.append(fmtr); points.append({'representation':rn,'scheme':sn,**pt}); bb=boot(y,pdx.probability_R.to_numpy(float),groups,a.bootstrap_replicates,SEED+ri*1000+si*100,f'{rn}/{sn}'); bb.to_csv(ed/f'file13_bootstrap_{rn}_{sn}.csv.gz',index=False,compression='gzip'); bsum.append(bootsum(pt,bb,rn,sn)); log(f"[PROGRESS] {rn}/{sn} | AUROC={pt['roc_auc']:.4f} AP={pt['average_precision']:.4f} BA={pt['balanced_accuracy']:.4f} MCC={pt['mcc']:.4f}")
        predall=pd.concat(preds,ignore_index=True); foldall=pd.concat(foldms,ignore_index=True); pointall=pd.DataFrame(points); bsall=pd.concat(bsum,ignore_index=True); predall.to_csv(ed/'file13_internal_oof_predictions.csv.gz',index=False,compression='gzip'); foldall.to_csv(ed/'file13_internal_fold_metrics.csv',index=False); pointall.to_csv(ed/'file13_internal_point_metrics.csv',index=False); bsall.to_csv(ed/'file13_internal_cluster_bootstrap_summary.csv',index=False)
        log('Stage 7/9 — freeze harmonized primary + original-668 sensitivity models')
        pfs=meta.feature.astype(str).tolist(); pf=freeze(rebuilt,y,pfs,PRIMARY,md,psha); sf=freeze(proj,y,feats,SENS,md,psha); psp=md/'file13_harmonized_primary_feature_schema.csv'; pd.DataFrame({'feature_index':range(len(pfs)),'feature':pfs,'normalized_key':[x.casefold() for x in pfs],'development_count':meta.development_count,'development_prevalence':meta.development_prevalence}).to_csv(psp,index=False); ssp=md/'file13_harmonized_original668_sensitivity_feature_schema.csv'; lschema.to_csv(ssp,index=False); log(f"[PROGRESS] PRIMARY frozen | features={len(pfs)} | model_sha={pf['model_sha256']}")
        log('Stage 8/9 — write freeze manifest and E3 gate')
        prim=pointall[pointall.representation==PRIMARY].to_dict('records'); sens=pointall[pointall.representation==SENS].to_dict('records'); ref=pointall[pointall.representation==REF].to_dict('records')
        primary_schema_sha=shafile(psp); sensitivity_schema_sha=shafile(ssp)
        man={'file':'FILE13','script_version':VERSION,'status':'PASS_HARMONIZED_DEVELOPMENT_FREEZE','completed_utc':utc(),'protocol_sha256':psha,'guardrails':prot['guardrails'],'primary_for_E3':PRIMARY,'primary_model':pf,'primary_feature_schema':{'path':str(psp),'sha256':primary_schema_sha,'feature_count':len(pfs)},'sensitivity_model':sf,'sensitivity_feature_schema':{'path':str(ssp),'sha256':sensitivity_schema_sha,'feature_count':len(feats)},'harmonized_feature_matrix':{'path':str(rbp),'sha256':shafile(rbp)},'locked_projection_matrix':{'path':str(prp),'sha256':shafile(prp)},'fasta_manifest':{'path':str(fmp),'sha256':shafile(fmp)},'bridge_audit':bs,'internal_point_metrics':{'primary':prim,'sensitivity':sens,'reference':ref},'E3_gate':{'allowed_after_this_manifest':True,'FILE14_must_verify_model_hash':pf['model_sha256'],'FILE14_must_verify_schema_hash':primary_schema_sha,'FILE14_must_verify_protocol_hash':psha,'E3_must_not_be_used_to_reselect_model':True,'all_pre_specified_E3_models_must_be_reported':True}}
        mp=md/'file13_freeze_manifest.json'; atomic_json(mp,man); msha=shafile(mp); flag=cp/'FREEZE_COMPLETE.flag'; atomic_json(flag,{'status':'PASS_HARMONIZED_DEVELOPMENT_FREEZE','created_utc':utc(),'freeze_manifest':str(mp),'freeze_manifest_sha256':msha,'protocol_sha256':psha,'primary_model_sha256':pf['model_sha256'],'primary_schema_sha256':primary_schema_sha})
        log('Stage 9/9 — final integrity recheck and summary')
        if shafile(Path(pf['model_path']))!=pf['model_sha256'] or shafile(psp)!=primary_schema_sha or shafile(mp)!=msha: raise RuntimeError('Freeze integrity recheck failed')
        summ={'file':'FILE13','script_version':VERSION,'status':'PASS_HARMONIZED_DEVELOPMENT_FREEZE','completed_utc':utc(),'development_n':len(ids),'development_R':int(y.sum()),'development_S':int((y==0).sum()),'amrfinder_version_text':ver,'amrfinder_database':str(db),'protocol_sha256':psha,'harmonized_rebuilt_feature_count':len(pfs),'locked_projection_feature_count':len(feats),'bridge_audit':bs,'primary_model_sha256':pf['model_sha256'],'primary_schema_sha256':shafile(psp),'freeze_manifest_sha256':msha,'freeze_flag':str(flag),'internal_primary_metrics':prim,'guardrails':prot['guardrails'],'next_step':'FILE14 may construct/score a new E3 only after verifying FILE13 model/schema/protocol hashes.'}; final=cp/'file13_final_summary.json'; atomic_json(final,summ); failjson.unlink(missing_ok=True)
        log('='*110); log('FILE13 STATUS : PASS_HARMONIZED_DEVELOPMENT_FREEZE'); log(f'Development     : n={len(ids)} R={y.sum()} S={(y==0).sum()}'); log(f'Harmonized feat : {len(pfs)}'); log(f"Local bridge    : n={lb['n']} changed_genomes={lb['changed_genomes']} changed_cells={lb['changed_cells']}"); log(f'Protocol SHA    : {psha}'); log(f"Primary model   : {pf['model_sha256']}"); log(f'Primary schema  : {shafile(psp)}'); log(f'Freeze manifest : {msha}'); log(f'E3 gate         : {flag}'); log(f'Final summary   : {final}'); log('='*110); return 0
    except KeyboardInterrupt:
        atomic_json(failjson,{'file':'FILE13','version':VERSION,'status':'INTERRUPTED','time_utc':utc(),'traceback':traceback.format_exc()}); log('[INTERRUPTED] checkpoints preserved; rerun same command'); return 130
    except Exception as e:
        atomic_json(failjson,{'file':'FILE13','version':VERSION,'status':'FAILED','time_utc':utc(),'error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}); log('='*110); log('FILE13 STATUS : FAILED'); log(f'Error        : {type(e).__name__}: {e}'); log(f'Failure JSON : {failjson}'); log('Per-genome checkpoints are preserved; fix cause and rerun.'); log('='*110); return 1

if __name__=='__main__': raise SystemExit(main())
