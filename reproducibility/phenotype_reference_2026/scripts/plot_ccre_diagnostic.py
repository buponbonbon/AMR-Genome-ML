#!/usr/bin/env python3
"""Plot the eligibility audit and explicitly exploratory error trade-offs."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/AMR_matplotlib_cache')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/AMR_fontcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.root/'legacy_feature_diagnostic/results/report.json').read_text());b=json.loads((a.root/'catalog_comparator_bounds/report.json').read_text())['bounds'];m=r['point_metrics']['locked_ML_older_DB_diagnostic'];k=r['point_metrics']['KPC_family_presence_older_DB']
    fig,(ax,err)=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1,1.2]});fig.patch.set_facecolor('white')
    counts=[2973,2834,2450,2228,1259,907,355];labels=['Published species-complex samples','K. pneumoniae','Meropenem MIC available','Explicit original S/R + MIC','Recorded PDS cluster','No shared PDS with mapped fitting','Distinct retained PDS groups']
    ax.barh(np.arange(len(counts)),counts,color=['#bdc8d5']*4+['#7198b8','#2b608c','#184564']);ax.set_yticks(np.arange(len(counts)),labels,fontsize=9);ax.invert_yaxis();ax.set_xlabel('Samples (last row: PDS groups)');ax.set_title('Source and identity screening',fontsize=11)
    for i,n in enumerate(counts):ax.text(n+28,i,f'{n:,}',va='center',fontsize=9)
    ax.set_xlim(0,3450);ax.spines[['top','right']].set_visible(False)
    y=np.arange(3);fn=[m['FN'],k['FN'],b['FN_range'][0]];fp=[m['FP'],k['FP'],b['FP_range'][0]]
    err.barh(y-.18,fn,height=.32,color='#b74d42',label='False susceptible (FN)');err.barh(y+.18,fp,height=.32,color='#3d7c95',label='False resistant (FP)')
    err.errorbar([(b['FP_range'][0]+b['FP_range'][1])/2],[2.18],xerr=[(b['FP_range'][1]-b['FP_range'][0])/2],fmt='none',ecolor='#173d50',capsize=5,lw=3)
    for i,(n,f) in enumerate(zip(fn,fp)):
        err.text(n+4,i-.18,str(n),va='center',fontsize=9);err.text((b['FP_range'][1] if i==2 else f)+4,i+.18,f'{b["FP_range"][0]}–{b["FP_range"][1]}' if i==2 else str(f),va='center',fontsize=9)
    err.set_yticks(y,['Frozen ML\n(older feature DB)','KPC-family presence','Catalog-derived broad rule\n(annotation bounds)'],fontsize=9);err.invert_yaxis();err.set_xlim(0,365);err.set_xlabel('Errors relative to author PHAS S/R (n=907)');err.set_title('Exploratory error trade-offs',fontsize=11);err.spines[['top','right']].set_visible(False);err.legend(loc='upper right',fontsize=8)
    fig.suptitle('CCRE diagnostic: January-2026 features, frozen model and original source labels',fontsize=12,y=.98)
    fig.text(.5,.025,'Not the locked August-2026 pipeline validation. PDS available for 2,944/4,218 fitting BioSamples.\nMixed AST methods; 50 comparator annotations unresolved. Error bars are identification bounds, not confidence intervals.',ha='center',fontsize=8,color='#404040')
    fig.tight_layout(rect=[0,.09,1,.94]);a.out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(a.out/('CCRE_exploratory_diagnostic.'+ext),dpi=180,bbox_inches='tight')

if __name__=='__main__':main()
