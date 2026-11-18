#!/usr/bin/env python3
"""Replay published RAST calls under BMD and QuickMIC references on fixed cases."""
import argparse,csv,hashlib,itertools,json,zipfile
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,matthews_corrcoef
from analyze_native_ccre_challenge import weighted

NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
METHODS=['RAST4h','RAST6h','RAST8h','RAST16_20h']
REFS=['BMD_cat','QM_cat']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def extract(p):
    t=ET.fromstring(zipfile.ZipFile(p).read('word/document.xml'));tab=t.findall('.//w:tbl',NS)[1]
    headers=['isolate','BMD_mic','BMD_cat','QM_mic','QM_cat','RAST4h','RAST6h','RAST8h','RAST16_20h','time_QuickMIC']
    rows=[];group=None
    for tr in tab.findall('w:tr',NS)[1:]:
        vals=[''.join(x.text or '' for x in tc.findall('.//w:t',NS)).strip() for tc in tr.findall('w:tc',NS)]
        if len(vals)==1:group=vals[0];continue
        if not vals or not vals[0].startswith('QRC'):continue
        if len(vals)!=len(headers) or group is None:raise ValueError('Unexpected original table layout')
        drug,species=group.split(' - ',1);rows.append(dict(drug=drug.strip(),species=species.strip(),**dict(zip(headers,vals))))
    d=pd.DataFrame(rows)
    if d.duplicated(['isolate','drug','species']).any() or len(d)!=289:raise ValueError('Source membership mismatch')
    return d

def point(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel() if len(y) else [0,0,0,0];R=int(tp+fn);S=int(tn+fp)
    return dict(n=len(y),R=R,S=S,TP=int(tp),FN=int(fn),TN=int(tn),FP=int(fp),
        sensitivity=float(tp/R) if R else None,specificity=float(tn/S) if S else None,
        balanced_accuracy=float(balanced_accuracy_score(y,p)) if R and S else None,
        MCC=float(matthews_corrcoef(y,p)) if R and S else None)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root;amend=root/'CLINICAL_REPLAY_AMENDMENT.json';spec=json.loads(amend.read_text())
    assert sha(amend)==amend.with_suffix('.json.sha256').read_text().split()[0]
    source=root/'supplementary/PMC13435937/spectrum.03956-25-s0002.docx';assert sha(source)==spec['source_docx_sha256']
    d=extract(source);a.out.mkdir(parents=True,exist_ok=False);d.to_csv(a.out/'published_isolate_calls.csv',index=False)
    results=[];metrics=[];excluded=[]
    for ordinal,((species,drug),group) in enumerate(d.groupby(['species','drug'])):
        for pi,(m1,m2) in enumerate(itertools.combinations(METHODS,2)):
            cols=REFS+[m1,m2];good=group[cols].isin(['S','R']).all(axis=1);case=group.loc[good].copy()
            for row in group.loc[~good].to_dict('records'):
                excluded.append(dict(species=species,drug=drug,method1=m1,method2=m2,isolate=row['isolate'],excluded_fields=';'.join(k+'='+row[k] for k in cols if row[k] not in {'S','R'})))
            cells={}
            for ref,method in itertools.product(REFS,[m1,m2]):
                y=case[ref].eq('R').to_numpy(int);pred=case[method].eq('R').to_numpy(int)
                cell=point(y,pred);cells[(ref,method)]=cell;metrics.append(dict(species=species,drug=drug,method_pair=m1+'|'+m2,reference=ref,index_method=method,**cell))
            eligible=all(cells[(ref,m1)]['R'] and cells[(ref,m1)]['S'] for ref in REFS)
            r=dict(species=species,drug=drug,method1=m1,method2=m2,primary_pair=[m1,m2]==spec['primary_fixed_methods'],source_n=len(group),complete_n=len(case),excluded_n=int((~good).sum()),both_references_two_class=bool(eligible),
                same_cases_for_both_references=True,reference_RS_disagreements=int(case[REFS[0]].ne(case[REFS[1]]).sum()),
                original_BMD_category_counts=dict(Counter(group.BMD_cat)),original_QM_category_counts=dict(Counter(group.QM_cat)),
                BMD_method1_minus_method2_BA=None,QM_method1_minus_method2_BA=None,reference_change_in_method_contrast_BA=None,point_order_reverses=False)
            if eligible:
                b=cells[('BMD_cat',m1)]['balanced_accuracy']-cells[('BMD_cat',m2)]['balanced_accuracy'];q=cells[('QM_cat',m1)]['balanced_accuracy']-cells[('QM_cat',m2)]['balanced_accuracy']
                r.update(BMD_method1_minus_method2_BA=b,QM_method1_minus_method2_BA=q,reference_change_in_method_contrast_BA=q-b,point_order_reverses=b*q < -1e-12)
                rng=np.random.default_rng(20261112+10000+ordinal*10+pi);w=rng.multinomial(len(case),np.full(len(case),1/len(case)),size=10000)
                ys={ref:case[ref].eq('R').to_numpy(int) for ref in REFS};preds={m:case[m].eq('R').to_numpy(int) for m in [m1,m2]}
                arrays={(ref,m):weighted(ys[ref],preds[m],w)[:,6] for ref,m in itertools.product(REFS,[m1,m2])}
                db=arrays[('BMD_cat',m1)]-arrays[('BMD_cat',m2)];dq=arrays[('QM_cat',m1)]-arrays[('QM_cat',m2)]
                r['paired_composition_percentiles']={}
                for name,v in [('BMD_method_contrast',db),('QM_method_contrast',dq),('reference_by_method_interaction',dq-db)]:
                    finite=v[np.isfinite(v)];r['paired_composition_percentiles'][name]=dict(valid_draws=len(finite),percentile_2_5=float(np.quantile(finite,.025)),percentile_97_5=float(np.quantile(finite,.975)))
            results.append(r)
    pd.DataFrame(metrics).to_csv(a.out/'all_cell_metrics.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in results]).to_csv(a.out/'all_method_reference_contrasts.csv',index=False)
    pd.DataFrame(excluded).to_csv(a.out/'all_pair_specific_exclusions.csv',index=False)
    report=dict(status='COMPLETE_PUBLISHED_PHENOTYPIC_METHOD_REFERENCE_REPLAY',source_doi=spec['source_doi'],source_sha256=sha(source),amendment_sha256=sha(amend),source_rows=len(d),isolate_count=d.isolate.nunique(),strata_count=d.groupby(['species','drug']).ngroups,
        primary_pair=[r for r in results if r['primary_pair']],all_pairs=results,
        total_eligible_pair_strata=sum(r['both_references_two_class'] for r in results),point_order_reversal_pair_strata=sum(r['point_order_reverses'] for r in results),
        limits=spec['interpretation']+['Index tests and substitute reference were developed/evaluated in the source study. This replay is post-hoc reuse, not a new clinical trial.','One measured clinical study supplies all strata; drugs/organisms/timepoints are not independent source studies.','No original genomic model scores were generated for this secondary phenotypic-method analysis.'])
    save(a.out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'all_pairs','primary_pair','limits'}},indent=2))
    for r in report['primary_pair']:
        if r['both_references_two_class']:print(json.dumps(r))

if __name__=='__main__':main()
