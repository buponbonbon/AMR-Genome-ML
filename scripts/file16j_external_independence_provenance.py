#!/usr/bin/env python3
"""FILE16J: frozen E2/E3/development provenance & sequence-overlap research audit.

Research-only, read-only inputs. NO classifier changes, phenotype relabeling,
threshold tuning, or retrospective exclusion. Uses existing 4,227/525/306
manifests and the exact 831 annotation provenance rows.

No absence-of-overlap claim if FASTAs are missing or only identifiers were checked.
Optional Mash MinHash screening yields approximate candidates, not definitive ANI.
Optional NCBI queries never infer a BioProject from the free-text Source field.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, re, shutil, subprocess, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS={
 'development':'data/processed/file07_common_cohort.csv',
 'dev_fasta':'data/features/known_amr_harmonized/file13/file13_development_fasta_manifest.csv',
 'e2':'data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv',
 'e2_fasta':'data/external_validation/ncbi_pathogen_detection/file11c/file11c_blind_predictions.csv',
 'e3':'data/external_validation/e3/file14_final/file14_E3_FINAL_FROZEN.csv',
 'bvbrc_meta':'data/external_validation/e3/source_bvbrc/bvbrc_kp_meropenem_genome_metadata.csv',
 'evidence':'results/file16b_annotation_v1/file16b_family_annotation_evidence.csv',
 'e3_manifest':'data/external_validation/e3/file14_final/file14_E3_FINAL_fasta_manifest.csv',
 'outdir':'results/file16j_independence_v1',
}
PROJECT_RE=re.compile(r'^(?:PRJNA|PRJEB|PRJDB|PRJDA)\d+$',re.I)
BASE='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'


def now():return datetime.now(timezone.utc).isoformat()
def h256_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rows,cols):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def jsonout(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def good(v):
 v=str(v or '').strip();return '' if v.upper() in {'NA','N/A','NAN','NONE','NULL','UNKNOWN','-','.'} else v

def resolve(root,s):
 s=good(s)
 if not s:return None
 if re.match(r'^[A-Za-z]:[\\/]',s):
  s='/mnt/'+s[0].lower()+'/'+s[3:].replace('\\','/')
 if s.startswith('file://'):s=urllib.parse.unquote(s[7:])
 p=Path(s).expanduser()
 return p if p.is_absolute() else root/p

def uniq(rows,field,label):
 ids=[good(r.get(field)) for r in rows]
 if not all(ids) or len(ids)!=len(set(ids)):raise ValueError(f'{label}: blank/duplicate {field}, examples={Counter(ids).most_common(4)}')
 return dict(zip(ids,rows))
def require(rows, fields, label):
 if not rows:raise ValueError(f'{label}: empty CSV')
 absent=set(fields)-set(rows[0]);
 if absent:raise ValueError(f'{label}: missing {absent}')
def safe_equal_counts(name, rows, n, test):
 if not test and len(rows)!=n:raise ValueError(f'{name}: expected {n}, got {len(rows)}; STOP')

def input_cohorts(root, args):
 p={k:resolve(root,getattr(args,k)) for k in ('development','dev_fasta','e2','e2_fasta','e3','evidence')}
 p['e3_manifest']=resolve(root,args.e3_manifest)
 p['bvbrc_meta']=resolve(root,args.bvbrc_meta)
 for k in ('development','dev_fasta','e2','e2_fasta','e3','evidence'):
  if not p[k] or not p[k].is_file():raise FileNotFoundError(f'{k} missing: {p[k]}')
 dev,df,e2,e2f,e3,evidence=(read_csv(p[k]) for k in ('development','dev_fasta','e2','e2_fasta','e3','evidence'))
 require(dev,['Genome ID'],'development')
 require(df,['Genome ID','fasta_path'],'development FASTA')
 require(e2,['antibiotic','asm_acc','biosample_acc','bioproject_acc'],'E2 frozen')
 require(e2f,['asm_acc','fasta_path'],'E2 FASTA')
 require(e3,['Genome ID','BioSample','FASTA Path','fasta_content_sha256'],'E3 frozen')
 require(evidence,['cohort','genome_id','raw_tsv','raw_tsv_sha256'],'FILE16B evidence')
 e2=[r for r in e2 if good(r['antibiotic']).lower()=='meropenem']
 safe_equal_counts('dev',dev,4227,args.test_mode)
 safe_equal_counts('dev FASTA',df,4227,args.test_mode)
 safe_equal_counts('E2 meropenem',e2,525,args.test_mode)
 safe_equal_counts('E2 FASTA',e2f,525,args.test_mode)
 safe_equal_counts('E3',e3,306,args.test_mode)
 safe_equal_counts('annotation evidence',evidence,831,args.test_mode)
 devs=uniq(dev,'Genome ID','dev'); devf=uniq(df,'Genome ID','dev FASTA')
 e2s=uniq(e2,'asm_acc','E2'); e2fs=uniq(e2f,'asm_acc','E2 FASTA')
 e3s=uniq(e3,'Genome ID','E3'); evid=uniq(evidence,'genome_id','evidence')
 for x,y,label in ((devs,devf,'dev-FASTA'),(e2s,e2fs,'E2-FASTA')):
  if set(x)!=set(y):raise ValueError(f'{label} IDs mismatch: missing {sorted(set(x)-set(y))[:8]} extra {sorted(set(y)-set(x))[:8]}')
 if set(evid)!=(set(e2s)|set(e3s)) or set(e2s)&set(e3s):
  raise ValueError('FILE16B evidence does not precisely match E2/E3 frozen ID sets')
 for gid,r in evid.items():
  if r['cohort']!=('E2' if gid in e2s else 'E3'):raise ValueError(f'FILE16B wrong cohort for {gid}')
 out=[]
 for id,r in devs.items():
  f=devf[id]
  out.append(dict(cohort='Development',genome_id=id,assembly_accession='',biosample='',bioproject='',source='',
       erd_group='',fasta_path=good(f['fasta_path']),fasta_expected_sha256='', fasta_provenance='FILE13 dev manifest'))
 for id,r in e2s.items():
  f=e2fs[id]
  out.append(dict(cohort='E2',genome_id=id,assembly_accession=id,biosample=good(r['biosample_acc']),
       bioproject=good(r['bioproject_acc']),source='NCBI Pathogen Detection',erd_group=good(r.get('erd_group')),
       fasta_path=good(f['fasta_path']),fasta_expected_sha256=good(f.get('fasta_sha256')),
       fasta_provenance='FILE11C blind frozen predictions'))
 for id,r in e3s.items():
  out.append(dict(cohort='E3',genome_id=id,assembly_accession=good(r.get('Assembly Accession')),
       biosample=good(r['BioSample']),bioproject='',source=good(r.get('Source')),
       erd_group=good(r.get('NCBI SNP/ERD Group')),fasta_path=good(r['FASTA Path']),
       fasta_expected_sha256=good(r['fasta_content_sha256']),fasta_provenance='FILE14 E3 frozen'))
 return p,out,evidence

def local_e3_bioproject(root,records,optional_path,extra_csv=None):
 """Exact BioSample accession or unique BV-BRC genome_id mapping only."""
 sources=[]
 if optional_path and optional_path.is_file():sources.append(('BV-BRC genome metadata',optional_path))
 if extra_csv:
  p=resolve(root,extra_csv)
  if not p.is_file():raise FileNotFoundError(p)
  sources.append(('User-provided verified crosswalk',p))
 ix=defaultdict(set); prov=defaultdict(set)
 for name,path in sources:
  with path.open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):
    pid=good(r.get('bioproject_accession') or r.get('bioproject_acc') or r.get('BioProject'))
    if not PROJECT_RE.fullmatch(pid):continue
    keys=[('biosample',good(r.get('biosample_accession') or r.get('biosample_acc') or r.get('BioSample'))),
          ('genome_id',good(r.get('genome_id') or r.get('Genome ID')))]
    for kind,v in keys:
     if v:
      ix[(kind,v)].add(pid);prov[(kind,v)].add(name)
 result={}
 for r in records:
  if r['cohort']!='E3':continue
  matched=set();origins=set()
  for k,v in (('biosample',r['biosample']),('genome_id',r['genome_id'])):
   if v:
    matched|=ix.get((k,v),set());origins|=prov.get((k,v),set())
  if len(matched)==1:result[r['genome_id']]=(next(iter(matched)),'LOCAL_EXACT_KEY', ';'.join(sorted(origins)))
  elif len(matched)>1:result[r['genome_id']]=('','CONFLICT_MULTIPLE_BIOPROJECTS',';'.join(sorted(matched)))
  else:result[r['genome_id']]=('','UNRESOLVED_NO_LOCAL_CROSSWALK','')
 return result

# NCBI Eutils only with explicit --online, user e-mail, and resumable cache.
def request_eutils(endpoint, params, email):
 params=dict(params,tool='FILE16J_AMR_Research',email=email)
 url=BASE+endpoint+'?'+urllib.parse.urlencode(params)
 for retry in range(4):
  try:
   req=urllib.request.Request(url,headers={'User-Agent':f'FILE16J/1.0 ({email})'})
   with urllib.request.urlopen(req,timeout=25) as response:return response.read()
  except Exception:
   if retry==3:raise
   time.sleep(1.0*(retry+1))
  finally:time.sleep(.42)
def find_bioproject_from_biosample_xml(raw):
 e=ET.fromstring(raw)
 found=set()
 for el in e.iter():
  context=' '.join(str(v) for v in el.attrib.values()).lower()
  if 'bioproject' not in context and el.tag.lower() not in {'bioproject'}:continue
  for text in [el.text,el.attrib.get('accession'),el.attrib.get('id'),el.attrib.get('target')]:
   if not text:continue
   for a in re.findall(r'\b(?:PRJNA|PRJEB|PRJDB|PRJDA)\d+\b',str(text),flags=re.I):found.add(a.upper())
 return found

def ncbi_online_biosample(bio,email):
 raw=request_eutils('esearch.fcgi',dict(db='biosample',term=f'{bio}[Accession]',retmode='json'),email)
 data=json.loads(raw);uids=data.get('esearchresult',{}).get('idlist',[])
 if len(uids)!=1:return [],'NCBI_NOT_UNIQUE_OR_NOT_FOUND'
 raw=request_eutils('efetch.fcgi',dict(db='biosample',id=uids[0],retmode='xml'),email)
 found=find_bioproject_from_biosample_xml(raw)
 if not found:
  raw=request_eutils('elink.fcgi',dict(dbfrom='biosample',db='bioproject',id=uids[0],retmode='xml'),email)
  doc=ET.fromstring(raw)
  linked={x.text for x in doc.findall('.//LinkSetDb/Link/Id') if x.text}
  # Many-to-many links must be marked ambiguous, never pick arbitrarily.
  if linked:
   raw=request_eutils('esummary.fcgi',dict(db='bioproject',id=','.join(sorted(linked)),retmode='xml'),email)
   doc=ET.fromstring(raw)
   for el in doc.iter():
    if el.tag.lower() not in ('accession','projectacc','project_acc'):continue
    a=good(el.text)
    if PROJECT_RE.fullmatch(a):found.add(a.upper())
 return sorted(found),('NCBI_EXACT' if len(found)==1 else 'NCBI_AMBIGUOUS' if len(found)>1 else 'NCBI_NO_BIOPROJECT')

def enrich_e3_online(root, records, local, args):
 pending=sorted({r['biosample'] for r in records if r['cohort']=='E3' and r['biosample'] and local[r['genome_id']][1]=='UNRESOLVED_NO_LOCAL_CROSSWALK'})
 cache_dir=resolve(root,args.metadata_cache);cache_dir.mkdir(parents=True,exist_ok=True)
 if args.online and not args.email:
  raise ValueError('--online requires --email according to NCBI E-utilities guidance')
 count=0; errors=0
 for n,bio in enumerate(pending,1):
  cp=cache_dir/(bio+'.json')
  if cp.is_file():
   res=json.loads(cp.read_text(encoding='utf-8'))
  elif args.online and (args.max_online==0 or count<args.max_online):
   try:
    projects,status=ncbi_online_biosample(bio,args.email)
    res=dict(biosample=bio,bioprojects=projects,status=status,utc=now())
   except Exception as e:
    res=dict(biosample=bio,bioprojects=[],status='NCBI_API_FAILED',error=str(e),utc=now())
    errors+=1
   jsonout(cp,res);count+=1
   if count%25==0:print(f'[FILE16J] NCBI fetched {count}/{len(pending)} unresolved BioSamples',flush=True)
  else:continue
  if len(res.get('bioprojects',[]))==1 and PROJECT_RE.fullmatch(res['bioprojects'][0]):
   for r in records:
    if r['cohort']=='E3' and r['biosample']==bio and local[r['genome_id']][1]=='UNRESOLVED_NO_LOCAL_CROSSWALK':
     local[r['genome_id']]=(res['bioprojects'][0],'NCBI_BIOSAMPLE',bio)
  elif res.get('status','') in ('NCBI_AMBIGUOUS','NCBI_API_FAILED'):
   for r in records:
    if r['cohort']=='E3' and r['biosample']==bio and local[r['genome_id']][1]=='UNRESOLVED_NO_LOCAL_CROSSWALK':
     local[r['genome_id']]=('',res['status'],bio)
 return {'unresolved_prior':len(pending),'online_requested':args.online,'new_requests':count,'api_errors':errors,'cache_records':len(list(cache_dir.glob('*.json'))),'cache_dir':str(cache_dir)}


def canonical_fasta(path):
 """Exact unordered multiset of case-normalized contig sequences (not near ANI)."""
 op=gzip.open if str(path).lower().endswith('.gz') else open
 hashes=[]; curr=None; n=0; nchar=0; ambiguous=0
 with op(path,'rt',encoding='ascii',errors='strict') as f:
  for line in f:
   line=line.strip()
   if not line:continue
   if line[0]=='>':
    if curr is not None:
     if n==0:raise ValueError('empty contig')
     hashes.append((n,curr.hexdigest()))
    curr=hashlib.sha256();n=0
   else:
    if curr is None:raise ValueError('FASTA sequence before first header')
    seq=line.replace(' ','').upper().encode('ascii')
    if not re.fullmatch(rb'[A-Z*.-]+',seq):raise ValueError('invalid sequence characters')
    curr.update(seq);n+=len(seq);nchar+=len(seq)
    ambiguous+=sum(seq.count(bytes([c])) for c in b'NRYKMSWBDHV')
  if curr is not None:
   if n==0:raise ValueError('empty last contig')
   hashes.append((n,curr.hexdigest()))
 if not hashes:raise ValueError('no contigs')
 h=hashlib.sha256()
 for n,digest in sorted(hashes):h.update(f'{n}:{digest}\n'.encode())
 return h.hexdigest(),len(hashes),nchar,ambiguous

def e3_frozen_manifest_reconciliation(inputs, records, test_mode):
    """Consistency only; manifests and frozen hashes can use distinct hash conventions."""
    p=inputs.get('e3_manifest')
    if not p or not p.is_file():
        return [],{'status':'MISSING_OPTIONAL_FILE14_FASTA_MANIFEST','matched':0,'missing':306 if not test_mode else 0}
    rows=read_csv(p)
    require(rows,['Genome ID','fasta_path','fasta_content_sha256'],'E3 FINAL FASTA manifest')
    idx=uniq(rows,'Genome ID','E3 FINAL FASTA manifest')
    e3=[r for r in records if r['cohort']=='E3']
    report=[]
    for r in e3:
        f=idx.get(r['genome_id'])
        report.append(dict(genome_id=r['genome_id'],in_final_manifest=bool(f),
            frozen_fasta_path=r['fasta_path'],manifest_fasta_path=good(f.get('fasta_path')) if f else '',
            paths_exact_same=bool(f and good(f.get('fasta_path'))==r['fasta_path']),
            frozen_fasta_content_sha256=r['fasta_expected_sha256'],
            manifest_fasta_content_sha256=good(f.get('fasta_content_sha256')) if f else '',
            expected_fasta_hash_same=bool(f and good(f.get('fasta_content_sha256')).lower()==r['fasta_expected_sha256'].lower()),
            manifest_status=good(f.get('status')) if f else ''))
    missing=sum(not r['in_final_manifest'] for r in report)
    diffhash=sum(r['in_final_manifest'] and not r['expected_fasta_hash_same'] for r in report)
    samehash=len(e3)-missing-diffhash
    state='E3_FROZEN_MANIFEST_CONSISTENT' if missing==0 and diffhash==0 and len(idx)==len(e3) else 'INCOMPLETE_OR_CONFLICTING'
    return report,{'status':state,'matched_hashes':samehash,'missing':missing,'different_expected_hashes':diffhash,
       'manifest_rows':len(idx),'frozen_rows':len(e3)}


def seq_inventory(root,records,run):
 ret=[]
 for i,r in enumerate(records,1):
  p=resolve(root,r['fasta_path']); ok=bool(p and p.is_file())
  v=dict(cohort=r['cohort'],genome_id=r['genome_id'],fasta_path=str(p or ''),file_exists=ok,
     sequence_checked=False,sequence_status='NOT_REQUESTED',canonical_contig_multiset_sha256='',
     raw_bytes_sha256='',fasta_expected_sha256=r['fasta_expected_sha256'],
     expected_equals_raw='NOT_ASSESSED',contig_count='',assembly_bp='',ambiguity_count='')
  if not ok:v['sequence_status']='MISSING_FASTA'
  elif run:
   try:
    d,k,l,a=canonical_fasta(p)
    v.update(sequence_checked=True,sequence_status='CHECKED',canonical_contig_multiset_sha256=d,
             contig_count=k,assembly_bp=l,ambiguity_count=a)
    # Expensive duplicate-byte hash is avoided when no provenance reference.
    if r['fasta_expected_sha256']:
     raw=h256_file(p);v['raw_bytes_sha256']=raw
     v['expected_equals_raw']=str(raw.lower()==r['fasta_expected_sha256'].lower())
   except Exception as e:
    v['sequence_status']='BAD_OR_UNREADABLE_FASTA:'+str(e)[:120]
  ret.append(v)
  if run and i%500==0:print(f'[FILE16J] FASTA fingerprints {i}/{len(records)}',flush=True)
 return ret

def exact_overlap(seq):
 idx=defaultdict(list)
 for r in seq:
  if r['sequence_status']=='CHECKED':idx[r['canonical_contig_multiset_sha256']].append(r)
 pairs=[]
 for digest,group in idx.items():
  if len(group)<2:continue
  for a_i,a in enumerate(group):
   for b in group[a_i+1:]:
    if a['cohort']==b['cohort']:continue
    pairs.append(dict(cohort_a=a['cohort'],genome_a=a['genome_id'],cohort_b=b['cohort'],genome_b=b['genome_id'],
       canonical_contig_multiset_sha256=digest,scope='EXACT_IDENTICAL_CASE_NORMALIZED_CONTIG_MULTISET'))
 return pairs

def mash_screen(out, seq, distance, threads, sketch_size):
 exe=shutil.which('mash')
 if not exe:return [],{'status':'NOT_RUN_MASH_NOT_INSTALLED'}
 groups=defaultdict(list);paths=defaultdict(dict)
 for r in seq:
  if r['sequence_status']=='CHECKED':
   groups[r['cohort']].append(r['fasta_path']);paths[r['cohort']][r['fasta_path']]=r['genome_id']
 if any(not groups.get(k) for k in ('Development','E2','E3')):
  return [],{'status':'NOT_RUN_MISSING_COHORT_FASTA'}
 for k in groups:
  # A path reused across groups is a serious potential leak; still let exact_overlap expose it.
  if len(groups[k])!=len(set(groups[k])):return [],{'status':'NOT_RUN_REPEATED_FASTA_PATH_WITHIN_COHORT'}
 for cohort in ('Development','E2','E3'):
  lst=out/f'file16j_mash_{cohort.lower()}_inputs.txt'
  lst.write_text('\n'.join(groups[cohort])+'\n',encoding='utf-8')
  cmd=[exe,'sketch','-p',str(threads),'-s',str(sketch_size),'-k','21','-l','-o',str(out/f'file16j_{cohort.lower()}'),str(lst)]
  print('[FILE16J] sketching',cohort,len(groups[cohort]),flush=True)
  subprocess.run(cmd,check=True,capture_output=True,text=True)
 rows=[];comparisons={}
 for a,b in (('Development','E2'),('Development','E3'),('E2','E3')):
  cmd=[exe,'dist','-p',str(threads),'-d',str(distance),str(out/f'file16j_{a.lower()}.msh'),str(out/f'file16j_{b.lower()}.msh')]
  print('[FILE16J] Mash',a,'vs',b,flush=True)
  proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  n=0;unknown=0
  for line in proc.stdout:
   fields=line.rstrip('\n').split('\t')
   if len(fields)!=5:continue
   ref,query,d,p,shared=fields
   try:d=float(d)
   except ValueError:continue
   # Mash prints source filenames; do not guess IDs if filenames differ from expected paths.
   if ref not in paths[a] or query not in paths[b]:
    unknown+=1;continue
   rows.append(dict(cohort_a=a,genome_a=paths[a][ref],cohort_b=b,genome_b=paths[b][query],
          mash_distance=d,mash_pvalue=p,shared_hashes=shared,scope='APPROXIMATE_MINHASH_CANDIDATE_NOT_ANI'))
   n+=1
  stderr=proc.stderr.read();rc=proc.wait()
  if rc:raise RuntimeError(f'mash dist {a}-{b} failed: {stderr[-800:]}')
  comparisons[f'{a}_vs_{b}']=dict(n_candidates=n,n_unmapped_output_lines=unknown)
  if unknown:raise ValueError(f'Mash produced {unknown} unmapped hits: cannot claim completeness')
 return rows,{'status':'CANDIDATE_SCREEN_COMPLETE_NOT_INDEPENDENCE_PROOF','distance_max':distance,
             'k':21,'sketch_size':sketch_size,'threads':threads,'pairs':comparisons}

def annotations_protocol(root,evidence,verify):
 first={};rows=[];status=Counter()
 for i,r in enumerate(evidence,1):
  p=resolve(root,r['raw_tsv']);row=dict(cohort=r['cohort'],genome_id=r['genome_id'],
             raw_tsv=str(p or ''),exists=bool(p and p.is_file()),hash_checked=False,hash_match='',
             tsv_header_sha256='',n_fields=0,status='')
  if not row['exists']:row['status']='MISSING_TSV'
  else:
   try:
    with p.open('r',encoding='utf-8-sig',errors='replace') as f:
     header=f.readline().rstrip('\r\n').split('\t')
    if 'Element symbol' not in header:row['status']='UNEXPECTED_HEADER'
    else:
     row.update(n_fields=len(header),tsv_header_sha256=hashlib.sha256('\t'.join(header).encode()).hexdigest(),status='HEADER_CHECKED')
     first.setdefault(r['cohort'],[]).append(row['tsv_header_sha256'])
     if verify:
      hashval=h256_file(p);row['hash_checked']=True
      row['hash_match']=str(hashval==r['raw_tsv_sha256'])
      if not (hashval==r['raw_tsv_sha256']):row['status']='TSV_HASH_MISMATCH'
   except Exception as e:row['status']='TSV_READ_ERROR:'+str(e)[:90]
  status[row['status']]+=1;rows.append(row)
  if verify and i%100==0:print(f'[FILE16J] annotation TSV verified {i}/{len(evidence)}',flush=True)
 cols=defaultdict(set)
 for r in rows:
  if r['tsv_header_sha256']:cols[r['cohort']].add(r['tsv_header_sha256'])
 same_header=bool(cols['E2'] and cols['E3'] and cols['E2']==cols['E3'])
 return rows,dict(check_status=dict(status),headers_by_cohort={k:sorted(v) for k,v in cols.items()},
                   identical_column_header_sets=same_header,software_version_verified=False,
                   database_version_verified=False,parameter_parity_verified=False,
                   note='Identical TSV columns and SHA-256 do NOT prove same AMRFinderPlus executable, database, flags, or assembly preprocessing.')

def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 ap.add_argument('--project-root',default='.')
 for k,v in DEFAULTS.items():ap.add_argument('--'+k.replace('_','-'),default=v)
 ap.add_argument('--extra-bioproject-crosswalk',help='Optional verified CSV: BioSample, bioproject_accession or genome_id, bioproject_accession')
 ap.add_argument('--online',action='store_true',help='Explicit opt-in for NCBI E-utilities; requires --email')
 ap.add_argument('--email',help='Valid E-utilities user email; sent only to NCBI in --online mode')
 ap.add_argument('--metadata-cache',default='results/file16j_ncbi_metadata_cache',help='Persistent metadata cache reused across new run directories')
 ap.add_argument('--max-online',type=int,default=0,help='Limit NEW NCBI requests by BioSample; 0 means unlimited')
 ap.add_argument('--sequence',action='store_true',help='Read all existing FASTA files; fingerprints can take several minutes')
 ap.add_argument('--mash',action='store_true',help='Mash near-duplicate screening; implies --sequence; needs mash installed')
 ap.add_argument('--mash-distance',type=float,default=0.02,help='Upper Mash distance for reporting candidate near matches')
 ap.add_argument('--mash-sketch-size',type=int,default=1000)
 ap.add_argument('--threads',type=int,default=4)
 ap.add_argument('--verify-tsv',action='store_true',help='Rehash 831 annotation TSVs')
 ap.add_argument('--test-mode',action='store_true',help='ONLY FOR SYNTHETIC TEST: disables frozen row-count expectations')
 args=ap.parse_args()
 if args.mash:args.sequence=True
 if not 0<=args.mash_distance<=1 or args.threads<1 or args.max_online<0:ap.error('invalid mash distance/threads/max-online')
 root=Path(args.project_root).expanduser().resolve()
 out=resolve(root,args.outdir)
 if out.exists():raise FileExistsError(f'Refusing to overwrite: {out}; use new --outdir suffix')
 if args.online and not args.email:ap.error('NCBI --online requires --email')
 inputs,records,evidence=input_cohorts(root,args)
 out.mkdir(parents=True,exist_ok=False)
 try:
  print(f'[FILE16J] validated manifest IDs dev/E2/E3: {Counter(r["cohort"] for r in records)}',flush=True)
  manifest_rows,manifest_report=e3_frozen_manifest_reconciliation(inputs,records,args.test_mode)
  csvout(out/'file16j_E3_manifest_reconciliation.csv',manifest_rows,
    ['genome_id','in_final_manifest','frozen_fasta_path','manifest_fasta_path','paths_exact_same',
     'frozen_fasta_content_sha256','manifest_fasta_content_sha256','expected_fasta_hash_same','manifest_status'])
  local=local_e3_bioproject(root,records,inputs['bvbrc_meta'],args.extra_bioproject_crosswalk)
  online_stats=enrich_e3_online(root,records,local,args)
  crosswalk=[]
  for r in records:
   if r['cohort']=='E3':
    bp,status,src=local[r['genome_id']]
    crosswalk.append(dict(genome_id=r['genome_id'],biosample=r['biosample'],assembly_accession=r['assembly_accession'],
       source_free_text=r['source'],bioproject_acc=bp,lookup_status=status,evidence_source=src,
       erd_group=r['erd_group'],resolved=bool(bp)))
  csvout(out/'file16j_e3_bioproject_crosswalk.csv',crosswalk,list(crosswalk[0]))
  seq=seq_inventory(root,records,args.sequence)
  csvout(out/'file16j_fasta_integrity_manifest.csv',seq,list(seq[0]))
  exact=exact_overlap(seq)
  csvout(out/'file16j_exact_sequence_overlaps.csv',exact,
         ['cohort_a','genome_a','cohort_b','genome_b','canonical_contig_multiset_sha256','scope'])
  mashrows=[];mashreport={'status':'NOT_REQUESTED'}
  if args.mash:
   mashrows,mashreport=mash_screen(out,seq,args.mash_distance,args.threads,args.mash_sketch_size)
  csvout(out/'file16j_near_sequence_candidates.csv',mashrows,
        ['cohort_a','genome_a','cohort_b','genome_b','mash_distance','mash_pvalue','shared_hashes','scope'])
  protocol_rows,protocol=annotations_protocol(root,evidence,args.verify_tsv)
  csvout(out/'file16j_amrfinder_protocol_audit.csv',protocol_rows,list(protocol_rows[0]))
  c=Counter(r['cohort'] for r in seq if r['sequence_status']=='CHECKED')
  total=Counter(r['cohort'] for r in records)
  resolved=sum(bool(r['bioproject_acc']) for r in crosswalk)
  complete=all(c[k]==total[k] for k in ('Development','E2','E3'))
  exact_status='EXACT_FINGERPRINT_COMPLETE' if complete else 'INCOMPLETE_OR_NOT_RUN'
  summary=dict(status='FILE16J_RESEARCH_PROVENANCE_AUDIT_NOT_CLINICAL',utc=now(),test_mode=args.test_mode,
    cohort_n=dict(total),E3=dict(n=len(crosswalk),bioproject_resolved=resolved,
      bioproject_unresolved=len(crosswalk)-resolved,by_lookup_status=dict(Counter(r['lookup_status'] for r in crosswalk)),
      n_erd=sum(bool(r['erd_group']) for r in crosswalk),online=online_stats,
      fasta_manifest_reconciliation=manifest_report),
    sequences=dict(status=exact_status,checked_by_cohort=dict(c),missing_by_cohort=dict(Counter(r['cohort'] for r in seq if r['sequence_status']=='MISSING_FASTA')),
        unreadable_by_cohort=dict(Counter(r['cohort'] for r in seq if r['sequence_status'].startswith('BAD_'))),
        n_exact_cross_cohort_pairs=len(exact),
        n_approx_near_pairs=len(mashrows),mash=mashreport),
    AMRFinderPlus=protocol,
    input_SHA256={k:(h256_file(v) if v and v.is_file() else None) for k,v in inputs.items()},
    guardrails=['Preserves E2/E3 frozen predictions, labels, and full-cohort metrics.',
     'BioProject from exact accession mappings only; unresolved remain unresolved.',
     'Canonical FASTA fingerprint checks exact per-contig sequences regardless of header/order, not near-identity.',
     'Mash screening is approximate and requires follow-up alignment/ANI on hits; absence of hits cannot prove independent lineages.',
     'FASTA hash_expected may follow a different hashing convention; raw-hash mismatch is not a label error.',
     'TSV schema parity does NOT establish AMRFinderPlus version, database or flags parity.',
     'No AST/MIC provenance confirmation; source-level result is exploratory.',
     'Different 668/702-feature models, cannot attribute E2/E3 differences solely to source.'])
  jsonout(out/'file16j_summary.json',summary)
  report=f'''# FILE16J – evidence boundaries\n\nRun: {summary['utc']}\n\nE3 BioProject exact-key mappings: **{resolved}/{len(crosswalk)}**. Unknown records remain unknown.\n\nE3 frozen-to-FASTA manifest reconciliation: **{manifest_report['status']}**.\n\nSequence checking: **{exact_status}**; dev/E2/E3 FASTAs actually read: {dict(c)}. Exact inter-cohort hits: **{len(exact)}**.\n\nMash stage: **{mashreport['status']}**; candidates (not verified ANI): {len(mashrows)}.\n\nAMRFinder TSV header equality: **{protocol['identical_column_header_sets']}**; version/database/parameter parity: **UNVERIFIED**.\n\n## Publication constraints\n\n- Do NOT change the main frozen E2/E3 results or silently drop genomes.\n- Do NOT claim independent genomic cohorts unless ALL sequences were checked with adequate relatedness criteria.\n- Do NOT claim E3 BioProject-source transferability if BioProject coverage is incomplete.\n- Do NOT conflate TSV field compatibility with annotation protocol identity.\n- Do NOT infer AST/MIC validity from BioProject metadata or sequence similarity.\n- Paired ML vs gene-rule results remain exploratory, with different E2/E3 ML models.\n'''
  (out/'file16j_manuscript_evidence_gate.md').write_text(report,encoding='utf-8')
  jsonout(out/'file16j_output_manifest_sha256.json',{'outputs':{p.name:h256_file(p) for p in sorted(out.iterdir()) if p.is_file()},'frozen_inputs_modified':False})
  print('[FILE16J] COMPLETE',out,flush=True)
  print(json.dumps({'E3_bioproject':f'{resolved}/{len(crosswalk)}','sequence_status':exact_status,'exact_pairs':len(exact),
       'mash':mashreport['status'],'annotation_header_parity':protocol['identical_column_header_sets']},indent=2),flush=True)
 except Exception:
  print('[FILE16J] INCOMPLETE: partial outputs/cache retained for diagnosis; do not report as final',file=sys.stderr)
  raise

if __name__=='__main__':main()
