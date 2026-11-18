#!/usr/bin/env python3
"""Plot point contrasts and predeclared source-resampling percentile ranges."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ranges',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    x=pd.read_csv(args.ranges);x=x[x.reference.eq('october')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.5),sharey=True)
    for ax,(metric,title,scale) in zip(axes,[('balanced_accuracy','Balanced accuracy difference (pp)',100),('mcc','MCC difference',1),('auroc','AUROC difference (pp)',100)]):
        for y,scheme,color in [(1,'all_projects','#0072B2'),(0,'focal_retained','#D55E00')]:
            r=x[(x.scheme==scheme)&(x.metric==metric)].iloc[0]
            lo,point,hi=scale*r.q025,scale*r.point_ML_minus_rule,scale*r.q975
            ax.hlines(y,lo,hi,color=color,linewidth=2);ax.plot([lo,hi],[y,y],'|',color=color,markersize=10)
            ax.plot(point,y,'o',color=color)
            fmt='.2f' if scale==100 else '.3f'
            ax.text((lo+hi)/2,y+.12,f'{point:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]',ha='center',fontsize=9,color=color)
        ax.axvline(0,color='#666666',linestyle='--',linewidth=1);ax.grid(axis='x',alpha=.15);ax.set_xlabel(title);ax.set_ylim(-.4,1.5)
    axes[0].set_yticks([1,0],['Resample all\n32 BioProjects','Retain focal source;\nresample background'])
    fig.suptitle('October reference: ML minus rule under paired project resampling',fontsize=13)
    fig.text(.5,.025,'10,000 draws per scheme. Bars: 2.5th–97.5th percentiles of source-composition sensitivity; not population confidence guarantees.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.94))
    for extension in ['png','pdf']:fig.savefig(args.out/('paired_project_bootstrap_sensitivity.'+extension),dpi=250,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
