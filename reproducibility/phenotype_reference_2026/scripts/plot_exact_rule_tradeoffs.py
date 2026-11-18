#!/usr/bin/env python3
"""Plot exact fixed-method reference-relative ranking and error counts."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.metrics)
    colors={'frozen_ML':'#0072B2','frozen_rule':'#D55E00'}
    names={'frozen_ML':'Frozen ML','frozen_rule':'Fixed rule'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(11.6,4.4),gridspec_kw={'width_ratios':[1,1.1]})
    for method in ['frozen_ML','frozen_rule']:
        subset=data[data.method.eq(method)].set_index('reference').loc[['historical','october']]
        y=subset.balanced_accuracy.to_numpy()
        ax[0].plot([0,1],y,marker='o',linewidth=2,color=colors[method],label=names[method])
        for x,value in enumerate(y):
            offset=8 if method=='frozen_rule' and x==0 or method=='frozen_ML' and x==1 else -17
            ax[0].annotate(f'{value:.4f}',(x,value),xytext=(0,offset),textcoords='offset points',ha='center',color=colors[method])
    ax[0].set_xticks([0,1],['Historical\n163 R / 362 S','October\n266 R / 259 S'])
    ax[0].set_xlim(-.25,1.25);ax[0].set_ylim(.57,1.0);ax[0].set_ylabel('Balanced accuracy')
    ax[0].set_title('A  Ranking changes with recorded reference',loc='left',fontsize=11)
    ax[0].legend(loc='lower right',frameon=False)
    width=.32
    for j,method in enumerate(['frozen_ML','frozen_rule']):
        r=data[(data.method==method)&(data.reference=='october')].iloc[0]
        values=[r.fn/(r.tp+r.fn),r.fp/(r.tn+r.fp)]
        counts=[(int(r.fn),int(r.tp+r.fn)),(int(r.fp),int(r.tn+r.fp))]
        positions=[(j-.5)*width,1+(j-.5)*width]
        ax[1].bar(positions,values,width=width,color=colors[method],label=names[method])
        for x,v,(n,d) in zip(positions,values,counts):
            ax[1].text(x,v+.007,f'{n}/{d}\n{100*v:.1f}%',ha='center',va='bottom',fontsize=9)
    ax[1].set_xticks([0,1],['FN / reference R','FP / reference S'])
    ax[1].yaxis.set_major_formatter(PercentFormatter(1));ax[1].set_ylim(0,.24)
    ax[1].set_ylabel('Reference-relative error rate');ax[1].set_title('B  October error tradeoff',loc='left',fontsize=11)
    for a in ax:a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
    fig.text(.5,.015,'Same 525 samples and fixed decisions; recorded reference labels are not independently adjudicated clinical truth.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1))
    for extension in ['png','pdf']:
        fig.savefig(args.out/('reference_choice_and_error_tradeoff.'+extension),dpi=250,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
