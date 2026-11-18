#!/usr/bin/env python3
"""Standalone scientific figure from completed, source-relative analyses."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=json.loads((a.analysis/'report.json').read_text());d=pd.read_csv(a.analysis/'available_case_audit.csv')
    a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    ax=axes[0];x=np.array([0,1]);ax.plot(x,[.639604, .916582],'o-',label='Frozen ML',color='#2364AA',lw=2)
    ax.plot(x,[.688506,.903678],'s-',label='Frozen annotation rule',color='#B54A2A',lw=2)
    ax.set(xticks=x,xticklabels=['Historical','October partial-source'],ylim=(.55,1),ylabel='Balanced accuracy',title='A  E2: fixed methods, dated references')
    ax.legend(frameon=False,fontsize=8,loc='lower right');ax.text(.02,.98,'525 cases; unchanged predictions\nPoint estimates; source-relative labels',transform=ax.transAxes,ha='left',va='top',fontsize=8)
    ax=axes[1];y=d['S/I/R (PHAS interpretation)_Meropenem'].eq('R').to_numpy(int)
    fpr,tpr,_=roc_curve(y,d.frozen_probability_R);m=r['methods'];ax.plot(fpr,tpr,color='#2364AA',label=f"Frozen ML (AUROC {m['frozen_ML']['AUROC']:.3f})",lw=2)
    for key,label,color,marker in [('frozen_broad_annotation_rule','Native annotation rule','#B54A2A','s'),('frozen_KPC_rule','Native KPC rule','#9467BD','^'),('author_Kleborate_carbapenemase_presence','Author Kleborate presence','#2B8A6E','D')]:
        z=m[key];ax.scatter(z['FP']/z['S'],z['TP']/z['R'],color=color,marker=marker,s=45,label=label,zorder=4)
    ax.plot([0,1],[0,1],ls=':',color='gray');ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='False-resistant rate (FP / source S)',ylabel='Resistance sensitivity (TP / source R)',title=f"B  CCRE: {r['scored']}/{r['fixed_target']} fixed cases")
    ax.legend(frameon=False,fontsize=7,loc='lower right')
    ax=axes[2];keys=list(m);labels=['Frozen ML','Native broad\nannotation rule','Native KPC\nrule','Author Kleborate\npresence'];xx=np.arange(len(keys))
    fn=np.array([m[k]['FN'] for k in keys]);fp=np.array([m[k]['FP'] for k in keys]);bars1=ax.bar(xx-.18,fn,.36,label='False susceptible (FN)',color='#D9822B');bars2=ax.bar(xx+.18,fp,.36,label='False resistant (FP)',color='#4E79A7')
    ax.bar_label(bars1,padding=2,fontsize=8);ax.bar_label(bars2,padding=2,fontsize=8);ax.set(xticks=xx,xticklabels=labels,ylabel='Number of source-reference errors',title='C  CCRE: paired error trade-offs');ax.tick_params(axis='x',labelsize=8);ax.set_ylim(0,max(max(fn),max(fp))*1.3+1);ax.legend(frameon=False,fontsize=7,loc='upper right')
    fig.suptitle('Reference version and method comparison with frozen predictions',fontsize=14)
    fig.savefig(a.out/'reference_native_challenge.pdf',bbox_inches='tight');fig.savefig(a.out/'reference_native_challenge.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (a.out/'CAPTION.md').write_text(f'''Reference version and method comparison with frozen predictions. A: E2 point balanced accuracies under the historical and partial-source October references, with predictions and methods held fixed; ranges and source-composition limits are reported separately. B: Native CCRE frozen-model ROC with binary comparator operating points, using the same {r['scored']} available cases. C: Paired false-susceptible and false-resistant counts against original PHAS categories. The broad native annotation rule includes qualifying porin disruptions; the author Kleborate control has separate assembly/annotation provenance. Mixed source AST does not establish reference-laboratory errors, patient utility or a second phenotype-version event. No threshold was optimized.\n''')


if __name__=='__main__':main()
