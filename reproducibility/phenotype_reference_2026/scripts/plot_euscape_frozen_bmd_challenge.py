#!/usr/bin/env python3
"""Draw the completed frozen-source discrimination and paired error counts."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.analysis/'report.json').read_text());assert r['status'].startswith('COMPLETE_')
    d=pd.read_csv(a.analysis/'paired_case_audit.csv');y=d.analysis_reference_R.to_numpy(int);a.out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    x,z,_=roc_curve(y,d.frozen_probability_R);ax[0].plot(x,z,color='#2364AA',lw=2,label=f"Frozen ML (AUROC {r['methods']['frozen_ML']['AUROC']:.3f})")
    labels={'native_broad_annotation_rule':'Broad annotation rule','native_KPC_rule':'KPC rule','author_carbapenemase_presence':'Source CP presence'}
    for m,marker,color in [('native_broad_annotation_rule','s','#B54A2A'),('native_KPC_rule','^','#666666'),('author_carbapenemase_presence','D','#8A6C9F')]:
        v=r['methods'][m];ax[0].scatter(1-v['specificity'],v['sensitivity'],marker=marker,s=50,color=color,label=labels[m],zorder=4)
    ax[0].plot([0,1],[0,1],':',color='gray');ax[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='A  Frozen EuSCAPE challenge');ax[0].legend(frameon=False,fontsize=9,loc='lower right')
    methods=['frozen_ML','native_broad_annotation_rule','native_KPC_rule','author_carbapenemase_presence'];xx=np.arange(4)
    for field,offset,label,color in [('FN',-.18,'False-susceptible','#B54A2A'),('FP',.18,'False-resistant','#2364AA')]:
        bars=ax[1].bar(xx+offset,[r['methods'][m][field] for m in methods],.36,label=label,color=color);ax[1].bar_label(bars,padding=3)
    ax[1].set(xticks=xx,xticklabels=['Frozen\nML','Broad\nrule','KPC\nrule','Source CP\npresence'],ylabel='Reference-relative error count',title=f"B  Same {r['scored']} cases and reference")
    ax[1].legend(frameon=False,fontsize=9);ax[1].margins(y=.2)
    fig.savefig(a.out/'euscape_frozen_BMD_challenge.pdf',bbox_inches='tight');fig.savefig(a.out/'euscape_frozen_BMD_challenge.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (a.out/'CAPTION.md').write_text('Frozen EuSCAPE challenge against a declared EUCAST14 non-meningitis scenario derived from published central BMD MICs, not author-reported S/I/R. The same model,668features,threshold0.5 and annotation rules were retained. Source CP presence uses separate author annotation. Binary comparator operating points and continuous-score ROC retain different output resolution. Error counts are reference-relative and do not establish clinical utility, adjudicate historical labels or replicate a temporal record-change event.\n')

if __name__=='__main__':main()
