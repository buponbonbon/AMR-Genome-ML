#!/usr/bin/env python3
"""Recreate the missing historical sensitivity figure from archived table data."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['three-drug','scores','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();d=pd.read_csv(a.three_drug);s=pd.read_csv(a.scores)
    if hashlib.sha256(a.scores.read_bytes()).hexdigest()!='f2983db5a1d4c2300fdba90dde888262f1bb96a9e357856746c942d48409ad5a':raise ValueError('Frozen E2 sensitivity input differs')
    if len(d)!=525 or d.biosample_acc.nunique()!=175:raise ValueError('Wrong three-drug cohort')
    d['transition']=d.historical_label+'→'+d.october_and_live_label
    wide=d.pivot(index='biosample_acc',columns='antibiotic',values='transition')[['meropenem','imipenem','ertapenem']]
    profiles=wide.agg(' | '.join,axis=1).value_counts()
    focal=s[s.focal.astype(str).str.lower().eq('true')].copy()
    if set(focal.biosample)!=set(wide.index) or len(profiles)!=6:raise ValueError('Focal identities/profile count differ')
    labels=s.historical_label.map({'R':1,'S':0}).to_numpy();scenario=s.scenario_label.map({'R':1,'S':0}).to_numpy()
    if abs(roc_auc_score(labels,s.probability)-.6673473884)>1e-9 or abs(roc_auc_score(scenario,s.probability)-.9689087584)>1e-9:raise ValueError('Archived ROC points not reproduced')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(8,11),gridspec_kw={'height_ratios':[1.1,1,1.4]},layout='constrained')
    ax=axes[0];yy=np.arange(len(profiles));ax.barh(yy,profiles.values,color='#4E79A7');ax.set_yticks(yy,profiles.index);ax.invert_yaxis();ax.set_xlabel('BioSamples');ax.set_title('a  Joint transitions: meropenem | imipenem | ertapenem',fontsize=11);ax.bar_label(ax.containers[0],padding=3);ax.set_xlim(0,profiles.max()*1.15)
    ax=axes[1];rng=np.random.default_rng(20261111)
    for i,(old,new) in enumerate([('S','R'),('R','S')]):
        v=focal.loc[focal.historical_label.eq(old)&focal.scenario_label.eq(new),'probability'].to_numpy()
        ax.scatter(i+rng.uniform(-.1,.1,len(v)),v,s=12,alpha=.65,color=['#B54A2A','#2364AA'][i]);ax.hlines(np.median(v),i-.2,i+.2,color='black',lw=2)
    ax.axhline(.5,color='gray',ls='--');ax.set(xticks=[0,1],xticklabels=['Historical S → October R','Historical R → October S'],ylabel='Frozen probability of R',ylim=(-.03,1.03),title='b  Same frozen scores in 175 focal genomes')
    ax=axes[2]
    for y,label,color in [(labels,'Historical reference','#B54A2A'),(scenario,'Partial-source October scenario','#2364AA')]:
        fpr,tpr,_=roc_curve(y,s.probability);auc=roc_auc_score(y,s.probability);ax.plot(fpr,tpr,label=f'{label} (AUROC {auc:.3f})',color=color,lw=2)
    ax.plot([0,1],[0,1],ls=':',color='gray');ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='c  All 525 predictions held fixed');ax.legend(frameon=False,fontsize=9,loc='lower right')
    a.out.mkdir(parents=True,exist_ok=True);fig.savefig(a.out/'FigS4.pdf');fig.savefig(a.out/'FigS4.png',dpi=150);plt.close(fig)
    profiles.rename_axis('joint_transition').reset_index(name='BioSamples').to_csv(a.out/'FigS4_joint_transition_counts.csv',index=False)
    (a.out/'FigS4_source_verification.json').write_text(json.dumps({'status':'REGENERATED_FROM_ARCHIVED_DATA_WITHOUT_MODEL_OR_REFERENCE_CHANGE','three_drug_sha256':hashlib.sha256(a.three_drug.read_bytes()).hexdigest(),'E2_scores_sha256':hashlib.sha256(a.scores.read_bytes()).hexdigest(),'profiles':len(profiles),'focal':len(focal),'E2':len(s)},indent=2)+'\n')


if __name__=='__main__':main()
