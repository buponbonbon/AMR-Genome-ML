#!/usr/bin/env python3
"""Plot completed controlled cells and paired source-composition contrasts."""
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
    r=json.loads((a.analysis/'report.json').read_text())['cohorts'];e=r['E2'];c=r['CCRE']
    if e['status']!='COMPLETE_CONTROLLED_DATABASE_REFERENCE_ANALYSIS' or c['status']!=e['status']:raise ValueError('Analysis incomplete')
    a.out.mkdir(parents=True,exist_ok=True);d=pd.read_csv(a.analysis/'E2/paired_case_audit.csv')
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    ax=axes[0,0]
    for ref,col,color in [('Historical','historical_label','#B54A2A'),('October partial-source','scenario_label','#2364AA')]:
        y=d[col].eq('R').to_numpy(int)
        for db,ls in [('Jan','--'),('Aug','-')]:
            fpr,tpr,_=roc_curve(y,d['model_probability_'+db]);ax.plot(fpr,tpr,color=color,ls=ls,lw=1.8,label=ref+' / '+db+' DB')
    ax.plot([0,1],[0,1],ls=':',color='gray');ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='A  E2: one model, two references, two databases');ax.legend(frameon=False,fontsize=8,loc='lower right')
    ax=axes[0,1];conditions=[('Jan','historical'),('Aug','historical'),('Jan','October_partial'),('Aug','October_partial')];xx=np.arange(4)
    for method,offset,label,color in [('ML',-.17,'Frozen ML','#2364AA'),('broad_annotation_rule',.17,'Broad annotation rule','#B54A2A')]:
        values=[next(v['balanced_accuracy'] for v in e['cell_metrics'] if v['database']==db and v['reference']==ref and v['method']==method) for db,ref in conditions]
        bars=ax.bar(xx+offset,values,.34,label=label,color=color);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    ax.set(xticks=xx,xticklabels=['Historical\nJan DB','Historical\nAug DB','October\nJan DB','October\nAug DB'],ylim=(0,1.1),ylabel='Balanced accuracy',title='B  E2: paired method comparisons');ax.legend(frameon=False,fontsize=9)
    ax=axes[1,0];xx=np.arange(2)
    for method,offset,label,color in [('ML',-.17,'Frozen ML','#2364AA'),('broad_annotation_rule',.17,'Broad annotation rule','#B54A2A')]:
        values=[next(v['balanced_accuracy'] for v in c['cell_metrics'] if v['database']==db and v['method']==method) for db in ['Jan','Aug']]
        bars=ax.bar(xx+offset,values,.34,label=label,color=color);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set(xticks=xx,xticklabels=['January database','August database'],ylim=(0,1.1),ylabel='Balanced accuracy',title='C  CCRE: original PHAS reference held fixed');ax.legend(frameon=False,fontsize=9)
    ax=axes[1,1];contrasts=[]
    for key,label in [('DB_Aug_minus_Jan|historical|ML','E2: DB effect, historical reference'),('DB_Aug_minus_Jan|October_partial|ML','E2: DB effect, October reference'),('Reference_Oct_minus_historical|Jan|ML','E2: reference effect, January DB'),('Reference_Oct_minus_historical|Aug|ML','E2: reference effect, August DB')]:
        v=e['source_composition_sensitivity']['BioProject']['paired_contrast_percentiles'][key]
        contrasts.append((label,v['point']['balanced_accuracy'],v['metrics']['balanced_accuracy'],'#2364AA' if key.startswith('Reference') else '#666666'))
    for scheme,label in [('PDS','CCRE: DB effect, PDS resampling'),('country','CCRE: DB effect, country resampling')]:
        v=c['source_composition_sensitivity'][scheme]['paired_contrast_percentiles']['DB_Aug_minus_Jan|PHAS_original|ML'];contrasts.append((label,v['point']['balanced_accuracy'],v['metrics']['balanced_accuracy'],'#2B8A6E'))
    for i,(label,value,bounds,color) in enumerate(contrasts):
        lo,hi=bounds['percentile_2_5'],bounds['percentile_97_5'];ax.plot([lo,hi],[i,i],color=color,lw=2);ax.scatter([value],[i],color=color,s=30,zorder=3)
    ax.axvline(0,color='gray',ls=':');ax.set(yticks=np.arange(len(contrasts)),yticklabels=[x[0] for x in contrasts],xlabel='Change in ML balanced accuracy',title='D  Paired source-composition sensitivity');ax.invert_yaxis();ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Controlled phenotype-reference and annotation-database comparisons',fontsize=15)
    fig.savefig(a.out/'database_reference_factorial.pdf',bbox_inches='tight');fig.savefig(a.out/'database_reference_factorial.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (a.out/'CAPTION.md').write_text('Controlled comparisons with the same DNA, model weights, schema, threshold and rule code. A–B: E2 reference×database cells; October replaces only the previously documented focal labels. C: CCRE paired database conditions against unchanged original PHAS categories. D: Point contrasts and 2.5–97.5 percentile ranges under the specified source resampling. These ranges describe selected source composition, not clinical population confidence guarantees. DB contrasts are August minus January; reference contrasts are partial October minus historical. Binary rule operating points and continuous-score discrimination retain different output resolution. These results neither adjudicate AST correctness nor establish a second independent phenotype-version event.\n')


if __name__=='__main__':main()
