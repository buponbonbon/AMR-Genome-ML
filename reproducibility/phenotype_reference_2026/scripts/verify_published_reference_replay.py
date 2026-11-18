#!/usr/bin/env python3
"""Re-read the clinical table with python-docx and verify every replay cell."""
import argparse,hashlib,itertools,json
from pathlib import Path
from docx import Document
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,matthews_corrcoef

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;out=root/'clinical_method_replay'
    f=root/'supplementary/PMC13435937/spectrum.03956-25-s0002.docx';rows=[];group=None
    names=['isolate','BMD_mic','BMD_cat','QM_mic','QM_cat','RAST4h','RAST6h','RAST8h','RAST16_20h','time_QuickMIC']
    for row in Document(f).tables[1].rows[1:]:
        values=[c.text.strip() for c in row.cells]
        if ' - ' in values[0] and not values[0].startswith('QRC'):group=values[0];continue
        if not values[0].startswith('QRC'):continue
        drug,species=group.split(' - ',1);rows.append(dict(drug=drug.strip(),species=species.strip(),**dict(zip(names,values))))
    raw=pd.DataFrame(rows);saved=pd.read_csv(out/'published_isolate_calls.csv',keep_default_na=False)
    assert raw.to_dict('records')==saved.to_dict('records')
    records=pd.read_csv(out/'all_cell_metrics.csv',keep_default_na=False);report=json.loads((out/'report.json').read_text());checks=0
    for r in records.to_dict('records'):
        pair=r['method_pair'].split('|');g=raw[raw.species.eq(r['species'])&raw.drug.eq(r['drug'])];g=g[g[['BMD_cat','QM_cat']+pair].isin(['S','R']).all(axis=1)]
        y=g[r['reference']].eq('R').to_numpy(int);q=g[r['index_method']].eq('R').to_numpy(int)
        cm=confusion_matrix(y,q,labels=[0,1]).ravel() if len(y) else [0,0,0,0];tn,fp,fn,tp=cm
        for k,v in dict(n=len(y),R=int(y.sum()),S=int((1-y).sum()),TP=tp,FN=fn,TN=tn,FP=fp).items():assert int(r[k])==v
        if y.any() and (1-y).any():
            assert abs(float(r['balanced_accuracy'])-balanced_accuracy_score(y,q))<1e-12
            assert abs(float(r['MCC'])-matthews_corrcoef(y,q))<1e-12
        else:assert r['balanced_accuracy']=='' and r['MCC']==''
        checks+=1
    reversals=0;eligible=0
    for r in report['all_pairs']:
        subset=records[records.species.eq(r['species'])&records.drug.eq(r['drug'])&records.method_pair.eq(r['method1']+'|'+r['method2'])]
        if not r['both_references_two_class']:continue
        eligible+=1;deltas=[]
        for ref in ['BMD_cat','QM_cat']:
            s=subset[subset.reference.eq(ref)].set_index('index_method');deltas.append(float(s.loc[r['method1'],'balanced_accuracy'])-float(s.loc[r['method2'],'balanced_accuracy']))
        assert abs(deltas[0]-r['BMD_method1_minus_method2_BA'])<1e-12
        assert abs(deltas[1]-r['QM_method1_minus_method2_BA'])<1e-12
        reversal=deltas[0]*deltas[1] < -1e-12;assert reversal==r['point_order_reverses'];reversals+=reversal
    assert eligible==report['total_eligible_pair_strata'] and reversals==report['point_order_reversal_pair_strata']
    result=dict(status='PASS_INDEPENDENT_DOCX_LIBRARY_AND_SKLEARN_RECONSTRUCTION',source_rows=len(raw),cell_checks=checks,eligible_pair_strata=eligible,point_order_reversals=reversals,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
