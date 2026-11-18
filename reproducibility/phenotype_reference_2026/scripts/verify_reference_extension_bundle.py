#!/usr/bin/env python3
"""Compare standalone reanalyses with archived reports, matrices and tables."""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    source=root/'results/independent_reference_extension_20261112';checks=0;maxdelta=0.
    def equal(x,y):
        nonlocal checks,maxdelta
        if isinstance(x,dict):
            assert isinstance(y,dict) and x.keys()==y.keys()
            for k in x:equal(x[k],y[k])
        elif isinstance(x,list):
            assert isinstance(y,list) and len(x)==len(y)
            for xx,yy in zip(x,y):equal(xx,yy)
        elif isinstance(x,(float,int)) and not isinstance(x,bool):
            assert isinstance(y,(float,int)) and not isinstance(y,bool)
            delta=abs(float(x)-float(y));maxdelta=max(maxdelta,delta);assert delta<=1e-12;checks+=1
        else:assert x==y;checks+=1
    pairs=[(source/'EuSCAPE_analysis',root/'reanalysis_euscape'),
           (source/'exact_ranking_robustness',root/'reanalysis_exact'),
           (source/'clinical_method_replay',root/'reanalysis_clinical')]
    arrays=0;tables=0
    for old,new in pairs:
        equal(json.loads((old/'report.json').read_text()),json.loads((new/'report.json').read_text()))
        for f in old.glob('*.npz'):
            x=np.load(f,allow_pickle=False);y=np.load(new/f.name,allow_pickle=False);assert x.files==y.files
            for k in x.files:
                assert x[k].shape==y[k].shape and np.allclose(x[k],y[k],rtol=0,atol=1e-12,equal_nan=True);arrays+=1
        for f in old.glob('*.csv'):
            x=pd.read_csv(f,keep_default_na=False);y=pd.read_csv(new/f.name,keep_default_na=False)
            assert x.shape==y.shape and x.columns.tolist()==y.columns.tolist()
            for c in x:
                try:
                    xx=x[c].to_numpy(float);yy=y[c].to_numpy(float)
                except (ValueError,TypeError):assert x[c].tolist()==y[c].tolist()
                else:assert np.allclose(xx,yy,rtol=0,atol=1e-12,equal_nan=True)
            tables+=1
    native=json.loads((source/'EuSCAPE_analysis/independent_verification.json').read_text())
    clinical=json.loads((source/'clinical_method_replay/independent_verification.json').read_text())
    assert native['status'].startswith('PASS_') and native['cases']==170
    assert clinical['status'].startswith('PASS_') and clinical['cell_checks']==288
    result=dict(status='PASS_STANDALONE_THREE_ANALYSES_REPORTS_ARRAYS_TABLES_AND_INDEPENDENT_NATIVE_CLINICAL_CHECKS',
        semantic_report_values_checked=checks,maximum_report_numeric_difference=maxdelta,
        bootstrap_arrays_matched=arrays,CSV_tables_matched=tables,absolute_numeric_tolerance=1e-12,
        native_cases=170,clinical_cells=288,python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
        source_report_sha256=hashlib.sha256((source/'EuSCAPE_analysis/report.json').read_bytes()).hexdigest(),
        raw_DNA_and_retained_metadata_omitted='Not rechecked from the light bundle; original full-input PASS records retained.')
    (root/'STANDALONE_REANALYSIS_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
