# FIG2 — two-panel paired-contrast preview; one Google Colab cell.
# No CSV upload, no local-file collection, no manuscript edits.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DOWNLOAD = False
OUT = Path('/content') if Path('/content').is_dir() else Path.cwd()

# Frozen observed contrasts and paired source-resampling percentiles.
# Columns: panel, condition, resampling, estimate, lower, upper.
# Values in b are RAW AUROC differences; only their display is x1000.
DATA = [
    ['a', 'Sep.', 'Whole', -0.048901806595939346, -0.10802425110389301, 0.030300003570670295],
    ['b', 'E2–Sep', 'Whole', 0.00016947429075009346, -0.0002934499010250307, 0.0019208436474461878],
    ['a', 'Oct. partial', 'Whole', 0.012903881324933919, -0.08519851316940519, 0.08136081972603476],
    ['b', 'E2–Oct', 'Whole', 0.00034836125061687984, 0.0, 0.0019045261070400913],
    ['a', 'Sep.', 'Focal', -0.048901806595939346, -0.10879868319737339, -0.002234512736582787],
    ['b', 'E2–Sep', 'Focal', 0.00016947429075009346, -0.0003263460489112718, 0.0013898507716059907],
    ['a', 'Oct. partial', 'Focal', 0.012903881324933919, -0.024396958008361537, 0.07733897454202038],
    ['b', 'E2–Oct', 'Focal', 0.00034836125061687984, 0.0, 0.0013956015778169152],
    ['b', 'CCRE', 'Country', 0.00019255455712452463, 0.0, 0.0005346478547870125],
]
df = pd.DataFrame(DATA,columns=['panel','condition','scheme','value','low','high'])
assert len(df)==9 and np.isfinite(df[['value','low','high']].to_numpy()).all()
assert df.low.le(df.high).all()
assert not df.duplicated(['panel','condition','scheme']).any()

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':13.5,
    'axes.labelsize':14,'xtick.labelsize':13.5,'ytick.labelsize':13.5,
    'legend.fontsize':13.5,'axes.linewidth':.8,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'
})
styles = {
    'Whole':dict(color='#0072B2',marker='o',offset=-.065,filled=True),
    'Focal':dict(color='#D55E00',marker='s',offset=.065,filled=False),
    'Country':dict(color='#555555',marker='D',offset=0,filled=True),
}
fig,(a,b) = plt.subplots(1,2,figsize=(9.2,4.8),
                         gridspec_kw={'width_ratios':[1,1.35]})
fig.subplots_adjust(left=.11,right=.985,bottom=.22,top=.80,wspace=.48)

def setup(ax,letter):
    ax.text(-.18,1.045,letter,transform=ax.transAxes,fontsize=17,
            fontweight='bold',ha='left',va='bottom')
    ax.axhline(0,color='#555555',ls='--',lw=1,zorder=1)
    ax.grid(axis='y',ls=':',lw=.65,alpha=.28,zorder=0)
    ax.set_axisbelow(True)

def draw(ax,panel,conditions,multiplier):
    observed=[]
    for x,condition in enumerate(conditions):
        sub=df[(df.panel==panel)&(df.condition==condition)]
        # Both E2 schemes use the same observed data/point estimate.
        assert sub.value.nunique()==1
        value=float(sub.value.iloc[0])*multiplier
        observed.append(value)
        for r in sub.itertuples(index=False):
            style=styles[r.scheme];xi=x+style['offset']
            lo,hi=r.low*multiplier,r.high*multiplier
            ax.vlines(xi,lo,hi,color=style['color'],lw=1.65,zorder=2)
            ax.hlines([lo,hi],xi-.028,xi+.028,color=style['color'],lw=1.25,zorder=2)
            ax.plot(xi,value,marker=style['marker'],ms=7,ls='none',
                    mec=style['color'],mew=1.4,
                    mfc=style['color'] if style['filled'] else 'white',zorder=3)
        # Label the observed contrast once, rather than repeating per scheme.
        label=f'{value:+.3f}' if panel=='a' else f'{value:+.2f}'
        ax.text(x+(.36 if panel=='a' else .42),value,label,
                ha='center',va='center',fontsize=13.5,color='#222222')
    ax.set_xticks(range(len(conditions)),conditions)
    return observed

# a — model-minus-rule BA changes sign across phenotype references.
# August annotation database is held fixed. No marginal-CI subtraction.
setup(a,'a')
draw(a,'a',['Sep.','Oct. partial'],1)
a.set_xlim(-.35,1.75)
a.set_ylim(-.13,.115)
a.set_yticks([-.10,-.05,0,.05,.10])
a.set_ylabel('BA: model − rule')
a.set_xlabel('Phenotype reference',labelpad=8)

# b — actual continuous-score changes from database substitution.
# The scale is explicitly 10^-3 AUROC; it is NOT the BA scale in a.
setup(b,'b')
draw(b,'b',['E2–Sep','E2–Oct','CCRE'],1000)
b.set_xlim(-.38,2.78)
b.set_ylim(-.55,2.20)
b.set_yticks([-.5,0,.5,1,1.5,2])
b.set_ylabel(r'Model $\Delta$AUROC ($\times10^{-3}$)')
b.set_xlabel('Database replay: Aug − Jan',labelpad=8)
b.text(.98,.045,r'$\Delta$BA = 0',transform=b.transAxes,
       ha='right',va='bottom',fontsize=13.5,color='#555555')

handles=[Line2D([0],[0],color=s['color'],marker=s['marker'],ls='none',
                ms=7,mew=1.4,mfc=s['color'] if s['filled'] else 'white',label=name)
         for name,s in styles.items()]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.99),
           ncol=3,frameon=False,columnspacing=1.7,handletextpad=.4)

fig.savefig(OUT/'Fig2.pdf',bbox_inches='tight',pad_inches=.08)
fig.savefig(OUT/'Fig2.png',dpi=400,bbox_inches='tight',pad_inches=.08)
# Optional reproducibility output; this file is NOT an input or upload request.
df.assign(metric=df.panel.map({'a':'BA_model_minus_rule','b':'AUROC_Aug_minus_Jan'}),
          display_multiplier=df.panel.map({'a':1,'b':1000})).to_csv(
              OUT/'Figure2_paired_contrasts.csv',index=False)
plt.show()
print('Saved Fig2.pdf and Fig2.png in',OUT)
print('Whole = whole BioProjects; Focal = focal project retained once; Country = CCRE countries.')
print('Bars: 2.5th–97.5th percentiles from 10,000 PAIRED source resamples; not patient-population CIs.')
print('Panel a: same 525 E2 genomes; frozen model/rule; source reference changes.')
print('Panel b: same-DNA database replay, 2026-08-07.1 minus 2026-01-21.1; E2 n=525, CCRE n=354.')
print('Both model and rule have unchanged BA in b; AUROC changes are very small.')
print('Panels a and b use DIFFERENT metrics; do not compare their vertical scales.')
if DOWNLOAD:
    from google.colab import files
    files.download(str(OUT/'Fig2.pdf'))
    files.download(str(OUT/'Fig2.png'))
