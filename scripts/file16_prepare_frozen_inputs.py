#!/usr/bin/env python3
"""FILE16 preparation for verified candidate E2/E3 locked predictions.

READ-ONLY relative to FILE01–FILE15 and all source data.  Creates NEW, normalized
file16 inputs and a provenance JSON without retraining, tuning, or relabeling.
The discovery CSV identifies filenames/headers only; this script reads and validates
actual rows on the researcher's computer before emitting downstream inputs.

Run (from project root):
  python scripts/file16_prepare_frozen_inputs.py --project-root . \
      --outdir results/file16_inputs_v1
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
from datetime import datetime, timezone

FILES = {
 'E2': 'data/external_validation/ncbi_pathogen_detection/file11c/file11c_blind_predictions.csv',
 'E3': 'data/evaluation/file14/file14_E3_primary_predictions_labeled.csv',
 'DEV': 'data/processed/file07_common_cohort.csv',
 'E2_MODEL': 'checkpoints/external_validation_file11c/file11c_final_model.joblib',
 'E3_MODEL': 'models/file13/file13_harmonized_primary_model.joblib',
 'E3_COHORT_LOCK': 'data/external_validation/e3/file14_final/file14_E3_FINAL_FROZEN.csv',
 'FILE15_CHECKS': 'results/file15_reporting/file15_integrity_checks.csv',
 'FILE15_MANIFEST': 'results/file15_reporting/file15_artifact_manifest.csv',
}
REQUIRED={'E2','E3','DEV','E2_MODEL','E3_MODEL','E3_COHORT_LOCK','FILE15_CHECKS'}
FIELDS={
 'E2':{'genome_id':'asm_acc','y_true':'label_binary','prob_resistance':'probability_R',
       'assembly_accession':'asm_acc','biosample':'biosample_acc','erd_group':'erd_group',
       'mlst_group':'external_MLST'},
 'E3':{'genome_id':'Genome ID','y_true':'y','prob_resistance':'probability_R',
       'assembly_accession':'Assembly Accession','biosample':'BioSample',
       'erd_group':'NCBI SNP/ERD Group','mlst_group':'MLST'},
 'DEV':{'genome_id':'Genome ID'},
}
NULLS={'','na','n/a','nan','none','null','unknown','unresolved','-','not available','missing'}

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def csvrows(p):
 with p.open('r',encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f)
  if not reader.fieldnames: raise ValueError(f'No CSV header: {p}')
  if len(set(reader.fieldnames))!=len(reader.fieldnames):
   raise ValueError(f'Duplicate CSV header fields: {p}')
  return list(reader),list(reader.fieldnames)

def val(x):
 if x is None: return ''
 v=str(x).strip()
 return '' if v.casefold() in NULLS else v

def normalize(source, rows, cols):
 mapping=FIELDS[source]
 mandatory=['genome_id','y_true','prob_resistance'] if source!='DEV' else ['genome_id']
 for field in mandatory:
  if mapping[field] not in cols:raise ValueError(f'{source}: missing {mapping[field]!r}, columns={cols}')
 for field in ('assembly_accession','biosample','erd_group','mlst_group'):
  if source!='DEV' and mapping[field] not in cols:
   raise ValueError(f'{source}: missing metadata {mapping[field]} needed for group/overlap audit')
 out=[]; ids=set(); countR=0
 for i,r in enumerate(rows,start=2):
  ident=val(r[mapping['genome_id']])
  if not ident: raise ValueError(f'{source}: empty ID line {i}')
  if ident in ids:raise ValueError(f'{source}: duplicate ID {ident} line {i}')
  ids.add(ident)
  rr={'genome_id':ident}
  if source!='DEV':
   lab=val(r[mapping['y_true']]); sc=val(r[mapping['prob_resistance']])
   if lab not in ('0','1'): raise ValueError(f'{source}: label {lab!r} line {i}, expected 0/1')
   try: prob=float(sc)
   except ValueError: raise ValueError(f'{source}: probability {sc!r} line {i}')
   if not math.isfinite(prob) or not 0<=prob<=1:
    raise ValueError(f'{source}: probability out of range at line {i}')
   rr['y_true']=lab; rr['prob_resistance']=format(prob,'.17g')
   countR+=int(lab)
   for field in ('assembly_accession','biosample','erd_group','mlst_group'):
    rr[field]=val(r[mapping[field]])
  out.append(rr)
 expectedN={'E2':525,'E3':306,'DEV':4227}[source]
 if len(out)!=expectedN:raise ValueError(f'{source}: count={len(out)}, expected={expectedN}')
 if source!='DEV':
  expectedR={'E2':163,'E3':122}[source]
  if countR!=expectedR:raise ValueError(f'{source}: resistant={countR}, expected={expectedR}')
 return out

def intersections(rowsA, rowsB, fields):
 ret={}
 for field in fields:
  a={val(x.get(field)).casefold() for x in rowsA if val(x.get(field))}
  b={val(x.get(field)).casefold() for x in rowsB if val(x.get(field))}
  both=sorted(a&b)
  ret[field]={'n_a':len(a),'n_b':len(b),'overlap_n':len(both),'examples':both[:5]}
 return ret

def writecsv(p, rows, fields):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def run(args):
 root=Path(args.project_root).expanduser().resolve()
 if not root.is_dir(): raise FileNotFoundError(root)
 out=Path(args.outdir).expanduser()
 if not out.is_absolute():out=(root/out).resolve()
 if out.exists(): raise ValueError(f'Output already exists (never overwrite): {out}')
 if root not in out.parents:raise ValueError('Output must be within project root for provenance safety')
 resolved={k:(root/p) for k,p in FILES.items()}
 for k in sorted(REQUIRED):
  if not resolved[k].is_file():raise FileNotFoundError(f'{k} missing: {resolved[k]}')
 e2src,h2=csvrows(resolved['E2']);e3src,h3=csvrows(resolved['E3']);devsrc,hd=csvrows(resolved['DEV'])
 e2=normalize('E2',e2src,h2);e3=normalize('E3',e3src,h3);dev=normalize('DEV',devsrc,hd)
 lock_rows,lock_headers=csvrows(resolved['E3_COHORT_LOCK'])
 if 'Genome ID' not in lock_headers:raise ValueError('E3 lock manifest lacks Genome ID')
 locked={val(r['Genome ID']) for r in lock_rows}
 if locked!={r['genome_id'] for r in e3}:
  raise ValueError(f'E3 IDs inconsistent with locked cohort: expected {len(locked)}, predictions {len(e3)}')
 overlap_e2_e3=intersections(e2,e3,['genome_id','assembly_accession','biosample'])
 overlap_dev_e3=intersections(dev,e3,['genome_id'])
 overlap_dev_e2=intersections(dev,e2,['genome_id'])
 overlapping=[(pair,k,d['overlap_n']) for pair,rs in [('E2/E3',overlap_e2_e3),('DEV/E3',overlap_dev_e3),('DEV/E2',overlap_dev_e2)] for k,d in rs.items() if d['overlap_n']]
 if overlapping:
  raise ValueError(f'Identifier overlap detected; STOP and investigate: {overlapping}')
 source_hashes={k:{'relative_path':FILES[k],'bytes':p.stat().st_size,'sha256':digest(p)} for k,p in resolved.items() if p.is_file()}
 report={
  'created_utc':datetime.now(timezone.utc).isoformat(),
  'purpose':'Normalize locked E2/E3 predictions for exploratory FILE16; no model fitting',
  'root':str(root),'outdir':str(out),'cohorts':{'E2':{'n':len(e2),'resistant':163},'E3':{'n':len(e3),'resistant':122},'DEV':{'n':len(dev)}},
  'model_sha_identical':source_hashes['E2_MODEL']['sha256']==source_hashes['E3_MODEL']['sha256'],
  'overlap_e2_e3':overlap_e2_e3,'overlap_dev_e3':overlap_dev_e3,'overlap_dev_e2':overlap_dev_e2,
  'source_hashes':source_hashes,
  'cautions':['Identifier overlap checks do NOT substitute for genotype/sequence similarity checks.',
   'E2 and E3 primary models use distinct feature schemas; comparing AUROCs does not isolate population effects.',
   'E3 genomic cluster metadata mostly unresolved; bootstrap CIs may be optimistic.',
   'No mechanism_group or E3 broad_rule inferred: deriving these requires audited genomic annotation.'],
 }
 out.mkdir(parents=True,exist_ok=False)
 cols=['genome_id','y_true','prob_resistance','assembly_accession','biosample','erd_group','mlst_group']
 writecsv(out/'file16_e2_predictions_standardized.csv',e2,cols)
 writecsv(out/'file16_e3_predictions_standardized.csv',e3,cols)
 writecsv(out/'file16_development_manifest_standardized.csv',dev,['genome_id'])
 (out/'file16_input_prep_provenance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PREPARED_NOT_YET_AUDITED','outdir':str(out),'e2':len(e2),'e3':len(e3),
                   'model_sha_identical':report['model_sha_identical'],
                   'E3_erd_missing':sum(1 for r in e3 if not val(r['erd_group']))},ensure_ascii=False,indent=2))
 print('NEXT: file16_external_reliability_audit.py audit --e2 ... --e3 ... --dev-manifest ...')

def main():
 ap=argparse.ArgumentParser(description='FILE16 frozen input preparation, with full provenance and no source modification')
 ap.add_argument('--project-root',default='.')
 ap.add_argument('--outdir',default='results/file16_inputs_v1')
 args=ap.parse_args()
 try:run(args)
 except Exception as e:
  import sys
  print('FILE16 PREP STOPPED (no source files modified):',e,file=sys.stderr)
  raise SystemExit(2)
if __name__=='__main__':main()
