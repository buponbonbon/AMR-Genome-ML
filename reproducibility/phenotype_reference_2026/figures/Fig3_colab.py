# FIG3 — one Colab cell, no CSV upload, no manuscript edits.
# Genuine ROC coordinates from archived per-isolate scores; no simulated curves.
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DOWNLOAD = False
OUT = Path('/content') if Path('/content').is_dir() else Path.cwd()

# ROC entries are observed [false-positive count, true-positive count].
# Dividing by the actual S/R totals reconstructs the empirical ROC exactly.
ROC_COUNTS = {'CCRE': [[0, 0], [0, 1], [0, 38], [1, 38], [1, 80], [1, 83], [1, 86], [2, 86], [2, 91], [3, 91], [3, 103],
          [3, 105], [3, 106], [4, 106], [4, 128], [5, 128], [5, 135], [6, 135], [6, 142], [8, 142], [8, 144],
          [9, 144], [11, 144], [11, 149], [14, 149], [14, 150], [21, 150], [21, 152], [24, 152], [24, 153],
          [30, 153], [30, 154], [31, 154], [31, 155], [32, 155], [32, 156], [33, 156], [33, 157], [38, 157],
          [38, 158], [42, 158], [42, 159], [45, 159], [47, 159], [47, 160], [63, 160], [63, 161], [70, 161],
          [71, 162], [71, 163], [90, 163], [93, 163], [94, 163], [98, 163], [118, 163], [118, 164],
          [127, 164], [129, 164], [136, 164], [138, 164], [143, 164], [148, 164], [151, 164], [155, 164],
          [157, 164], [167, 164], [169, 164], [190, 164]],
 'EuSCAPE': [[0, 0], [0, 1], [0, 6], [0, 11], [0, 12], [1, 12], [1, 21], [2, 21], [2, 28], [3, 28], [3, 30],
             [6, 30], [6, 33], [11, 33], [11, 35], [16, 35], [16, 36], [17, 36], [20, 36], [23, 36], [23, 38],
             [41, 38], [41, 39], [55, 39], [59, 39], [64, 39], [64, 40], [100, 40], [102, 40], [118, 40],
             [120, 40], [130, 40]]}
# Each confusion vector is [TP, FN, TN, FP], threshold fixed at 0.5.
COUNTS = {'CCRE': {'Model': [149, 15, 176, 14],
          'Broad rule': [162, 2, 122, 68],
          'KPC rule': [60, 104, 187, 3],
          'Source CP': [160, 4, 133, 57]},
 'EuSCAPE': {'Model': [35, 5, 114, 16],
             'Broad rule': [40, 0, 84, 46],
             'KPC rule': [19, 21, 127, 3],
             'Source CP': [39, 1, 93, 37]}}
EXPECTED_AUC = {'CCRE': 0.968084082156611, 'EuSCAPE': 0.9546153846153845}
# cohort, MIC stratum, n, BA(model)-BA(rule), lower, upper, valid draws
MIC = [['CCRE', 'All', 354, 0.10247111681643128, 0.04100528786157166, 0.14672450072214746, 10000],
 ['CCRE', 'Robust', 283, 0.08228370844517863, 0.03524284441404001, 0.11815241479563734, 10000],
 ['EuSCAPE', 'All', 170, 0.05288461538461542, -0.015127137713771437, 0.12891830294530154, 10000],
 ['EuSCAPE', 'Robust', 137, 0.005108556832694733, -0.08031000797448168, 0.07971014492753614, 9999]]

METHODS = ['Model','Broad rule','KPC rule','Source CP']
BLUE,ORANGE = '#0072B2','#D55E00'
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':13.5,
    'axes.labelsize':14,'axes.titlesize':14,
    'xtick.labelsize':13.5,'ytick.labelsize':13.5,'legend.fontsize':13.5,
    'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white'
})
fig,axes = plt.subplots(2,2,figsize=(9.2,8.4))
fig.subplots_adjust(left=.11,right=.985,bottom=.14,top=.87,wspace=.32,hspace=.52)
a,b,c,d = axes.ravel()

def panel(ax,letter):
    ax.text(-.18,1.055,letter,transform=ax.transAxes,
            fontsize=17,fontweight='bold',ha='left',va='bottom')
    ax.set_axisbelow(True)

binary_styles = {
    'Broad rule':dict(marker='s',color=ORANGE,fill=ORANGE),
    'KPC rule':dict(marker='^',color='#777777',fill='white'),
    'Source CP':dict(marker='D',color='#555555',fill='#555555'),
}
trap = np.trapezoid if hasattr(np,'trapezoid') else np.trapz
roc_backing=[]

# a/b: real model ROC, plus operating points for binary annotation rules.
for ax,letter,cohort in [(a,'a','CCRE'),(b,'b','EuSCAPE')]:
    panel(ax,letter)
    tp,fn,tn,fp = COUNTS[cohort]['Model']
    R,S = tp+fn,tn+fp
    xy=np.asarray(ROC_COUNTS[cohort],dtype=float)
    fpr,tpr=xy[:,0]/S,xy[:,1]/R
    assert np.all(np.diff(fpr)>=0) and np.all(np.diff(tpr)>=0)
    auc=float(trap(tpr,fpr))
    assert np.isclose(auc,EXPECTED_AUC[cohort],rtol=0,atol=1e-12)
    assert [int(v) for v in xy[0]]==[0,0]
    assert [int(v) for v in xy[-1]]==[S,R]
    ax.plot([0,1],[0,1],color='#BBBBBB',ls='--',lw=1,zorder=0)
    # Linear empirical threshold segments preserve the AUC treatment of ties.
    ax.plot(fpr,tpr,color=BLUE,lw=2.2,zorder=2)
    ax.plot(fp/S,tp/R,'o',color=BLUE,ms=7,zorder=4)
    for method,style in binary_styles.items():
        mtp,mfn,mtn,mfp=COUNTS[cohort][method]
        assert mtp+mfn==R and mtn+mfp==S
        ax.plot(mfp/S,mtp/R,marker=style['marker'],ls='none',ms=7.5,
                mec=style['color'],mfc=style['fill'],mew=1.3,zorder=4)
    ax.set(xlim=(0,1.03),ylim=(0,1.03),
           xlabel='False-positive rate',ylabel='True-positive rate')
    ax.set_xticks(np.arange(0,1.01,.2))
    ax.set_yticks(np.arange(0,1.01,.2))
    ax.set_title(f'{cohort} (n={R+S})',loc='left',pad=11)
    ax.text(.96,.055,f'AUROC {auc:.3f}',transform=ax.transAxes,
            ha='right',va='bottom',color=BLUE,fontsize=13.5)
    ax.grid(ls=':',lw=.55,alpha=.22)
    for xx,yy in zip(xy[:,0].astype(int),xy[:,1].astype(int)):
        roc_backing.append([cohort,xx,yy,S,R,xx/S,yy/R])

# c: paired error counts within the same cases/reference, not population rates.
panel(c,'c')
x=np.arange(4)
for method,offset,color in [('Model',-.18,BLUE),('Broad rule',.18,ORANGE)]:
    values=[COUNTS['CCRE'][method][1],COUNTS['CCRE'][method][3],
            COUNTS['EuSCAPE'][method][1],COUNTS['EuSCAPE'][method][3]]
    c.bar(x+offset,values,width=.32,color=color,alpha=.9,zorder=3)
    for xi,value in zip(x+offset,values):
        c.text(xi,value+1.4,str(value),ha='center',va='bottom',fontsize=13.5)
c.axvline(1.5,color='#B0B0B0',ls=':',lw=1)
c.set(xticks=x,xticklabels=['FN','FP','FN','FP'],
      ylim=(0,80),ylabel='Count',xlim=(-.5,3.5))
c.set_yticks([0,20,40,60,80])
c.grid(axis='y',ls=':',lw=.55,alpha=.22)
for xx,name in [(.5,'CCRE'),(2.5,'EuSCAPE')]:
    c.text(xx,-.22,name,transform=c.get_xaxis_transform(),ha='center',va='top',fontsize=13.5)

# d: paired country-composition percentiles, same saved draws for All/Robust.
# Robust = the declared MIC-boundary stress subset, not newly corrected labels.
panel(d,'d')
ticklabels=[]
for xi,row in enumerate(MIC):
    cohort,stratum,n,value,low,high,valid=row
    assert low<=high and valid in (9999,10000)
    d.vlines(xi,low,high,color='#444444',lw=1.6,zorder=2)
    d.hlines([low,high],xi-.055,xi+.055,color='#444444',lw=1.2,zorder=2)
    d.plot(xi,value,'o',color='#222222',ms=7,zorder=3)
    d.text(xi,high+.012,f'{value:+.3f}',ha='center',va='bottom',fontsize=13.5)
    ticklabels.append(f'{stratum}\nn={n}')
d.axhline(0,color='#777777',ls='--',lw=1)
d.axvline(1.5,color='#B0B0B0',ls=':',lw=1)
d.set(xticks=np.arange(4),xticklabels=ticklabels,xlim=(-.45,3.45),
      ylim=(-.12,.20),ylabel='BA: model − rule')
d.set_yticks([-.10,-.05,0,.05,.10,.15,.20])
d.grid(axis='y',ls=':',lw=.55,alpha=.22)
for xx,name in [(.5,'CCRE'),(2.5,'EuSCAPE')]:
    d.text(xx,-.31,name,transform=d.get_xaxis_transform(),ha='center',va='top',fontsize=13.5)

handles=[
    Line2D([0],[0],color=BLUE,lw=2.2,marker='o',ms=6,label='Model'),
    Line2D([0],[0],color=ORANGE,ls='none',marker='s',ms=7,label='Broad rule'),
    Line2D([0],[0],color='#777777',ls='none',marker='^',mfc='white',ms=7,label='KPC rule'),
    Line2D([0],[0],color='#555555',ls='none',marker='D',ms=7,label='Source CP'),
]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.995),
           ncol=4,frameon=False,columnspacing=1.2,handletextpad=.4,handlelength=1.5)

fig.savefig(OUT/'Fig3.pdf',bbox_inches='tight',pad_inches=.08)
fig.savefig(OUT/'Fig3.png',dpi=400,bbox_inches='tight',pad_inches=.08)
with (OUT/'Fig3_ROC_coordinates.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['cohort','FP','TP','S','R','FPR','TPR']);writer.writerows(roc_backing)
with (OUT/'Fig3_MIC_points.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['cohort','stratum','n','BA_difference','lower','upper','valid_draws']);writer.writerows(MIC)
plt.show()
print('Saved Fig3.pdf and Fig3.png in',OUT)
print('a/b: empirical ROC from archived model scores; model dot uses threshold0.5.')
print('Other markers are binary operating points, not continuous ROC curves.')
print('CCRE: original PHAS categories. EuSCAPE: declared EUCAST14 non-meningitis scenario from published central BMD MICs.')
print('c: FN=false-susceptible and FP=false-resistant, relative to each stated reference; counts are not cross-cohort clinical rates.')
print('d: All vs MIC-boundary-robust subset; hypothetical one-dilution stress, not measured assay error.')
print('Bars in d: 2.5th–97.5th paired country-composition percentiles, seed20261114; not patient-population CIs.')
print('Source CP is an author-reported genomic control. These cohorts do not adjudicate historical E2 labels.')
if DOWNLOAD:
    from google.colab import files
    files.download(str(OUT/'Fig3.pdf'))
    files.download(str(OUT/'Fig3.png'))
