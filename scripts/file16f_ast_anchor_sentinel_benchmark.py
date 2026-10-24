#!/usr/bin/env python3
"""FILE16F: label-budgeted AST-anchor source reliability benchmark.

Research-only post-hoc feasibility. Original predictions, thresholds, labels unchanged.
No tuning on evaluation holdouts. Not clinical software.
"""
from __future__ import annotations
import argparse, collections, csv, datetime as dt, hashlib, json, math, random, statistics
from pathlib import Path

DEFAULT_FOCUS='PRJNA1322038'
FROZEN_COLUMNS=('genome_id','source','y','p','erd','mechanism','high_conf_error')
METHODS=('anchor_error_rate','anchor_brier','anchor_high_conf_error','anchor_genotype_phenotype_discordance','dual_discordance_fixed_half','label_free_file16e')

def load_rows(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def save_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def parse_bool(v):
    if str(v).lower() in ('true','1'): return True
    if str(v).lower() in ('false','0'): return False
    raise ValueError(f'Invalid boolean {v!r}')

def read_input(path,summary_path):
    data=load_rows(path); assert len(data)==525, f'Expected frozen E2 525; got {len(data)}'
    keys=[];out=[]
    for a in data:
        g=a['asm_acc'].strip();src=a['bioproject_acc'].strip(); erd=a['erd_group'].strip()
        assert g and src
        y=int(a['y_true']);p=float(a['prob_resistance'])
        assert y in (0,1) and 0<=p<=1
        keys.append(g)
        pred=int(a['pred_resistant']);assert pred==int(p>=0.5),f'Frozen pred disagreement {g}'
        wrong=parse_bool(a['wrong']); assert wrong==bool(pred!=y),f'Wrong mismatch {g}'
        hc=parse_bool(a['high_conf_wrong_recalc']); assert hc==(wrong and (p>=0.9 or p<=0.1)),f'High conf mismatch {g}'
        raw_mech=a['mechanism_group'].strip()
        assert raw_mech and raw_mech not in ('NA','nan'),f'Missing mechanism {g}'
        if erd in ('','NA','nan','Unknown','Unresolved'):
            erd='UNRESOLVED_'+g
        present = raw_mech != 'No_detected_major_family'
        out.append(dict(genome_id=g,source=src,y=y,p=p,erd=erd,mechanism=raw_mech,
                        high_conf_error=int(hc),error=int(wrong),geno_discord=int(present!=bool(y)),
                        brier=(p-y)**2))
    assert len(keys)==len(set(keys)),'Duplicate genome ID'
    # Check published frozen aggregate; avoid accidental swap, label changes, or model changes.
    n_wrong=sum(d['error'] for d in out)
    assert n_wrong==195 and sum(d['y'] for d in out)==163 and sum(d['high_conf_error'] for d in out)==128
    if summary_path and summary_path.is_file():
        old=json.loads(summary_path.read_text())
        assert sha(path)==old['input_files']['predictions']['sha256'] if path.name=='file16_per_genome_prediction_error_audit.csv' else True
    return out

def score(anchors,label_free):
    n=len(anchors)
    return {
       'anchor_error_rate':sum(r['error'] for r in anchors)/n,
       'anchor_brier':sum(r['brier'] for r in anchors)/n,
       'anchor_high_conf_error':sum(r['high_conf_error'] for r in anchors)/n,
       'anchor_genotype_phenotype_discordance':sum(r['geno_discord'] for r in anchors)/n,
       'dual_discordance_fixed_half':0.5*sum(r['brier'] for r in anchors)/n+0.5*sum(r['geno_discord'] for r in anchors)/n,
       'label_free_file16e':label_free,
    }

def eligible_source(source_rows,k,mode,required_holdout):
    if len(source_rows)<max(10,k+required_holdout): return False
    if mode=='erd_disjoint':
        if len(set(r['erd'] for r in source_rows))<k: return False
    return True

def split_source(rs,k,mode,rng,min_holdout):
    if mode=='iid':
        chosen=set(rng.sample(range(len(rs)),k))
        train=[r for i,r in enumerate(rs) if i in chosen]
        test=[r for i,r in enumerate(rs) if i not in chosen]
    else:
        grouped=collections.defaultdict(list)
        for r in rs:grouped[r['erd']].append(r)
        ks=rng.sample(list(sorted(grouped)),k)
        train=[rng.choice(grouped[gp]) for gp in ks]
        forbidden=set(ks)
        test=[r for r in rs if r['erd'] not in forbidden]
    if len(test)<min_holdout:return None
    return train,test

def run(args):
    inputs=Path(args.input)
    scoresfile=Path(args.label_free_scores)
    dst=Path(args.outdir)
    if dst.exists():raise SystemExit(f'Output already exists; use NEW --outdir: {dst}')
    data=read_input(inputs,None)
    rank_source={r['bioproject_acc']:float(r['label_free_score']) for r in load_rows(scoresfile)}
    bysource=collections.defaultdict(list)
    for r in data:
        assert r['source'] in rank_source,f"Source absent from label-free file {r['source']}"
        bysource[r['source']].append(r)
    assert len(bysource)==32 and len(bysource[DEFAULT_FOCUS])==175
    dst.mkdir(parents=True)
    trialrows=[];source_rows=[];configs=[]
    for mode in ('iid','erd_disjoint'):
      for budget in args.budgets:
        candidates={s:by_source for s,by_source in sorted(bysource.items()) if eligible_source(by_source,budget,mode,args.min_holdout)}
        if len(candidates)<3: continue
        rng=random.Random(args.seed+budget*1007+(0 if mode=='iid' else 1_000_003))
        usable_trials=0
        for rep in range(args.trials):
            splits={}
            for s,rr in candidates.items():
                sample=split_source(rr,budget,mode,rng,args.min_holdout)
                if sample is not None:splits[s]=sample
            if len(splits)<3 or args.focus not in splits: continue
            usable_trials+=1
            source_stats={}
            for s,(anchors,holdout) in splits.items():
                sc=score(anchors,rank_source[s])
                held_err=sum(r['error'] for r in holdout)
                source_stats[s]=dict(sc=sc,holdout_n=len(holdout),holdout_error=held_err,
                      holdout_rate=held_err/len(holdout),anchor_wrong=sum(r['error'] for r in anchors),
                      holdout_fn=sum(r['error'] and r['y']==1 for r in holdout),
                      holdout_resistant=sum(r['y']==1 for r in holdout))
                if args.save_per_trial_sources:
                    source_rows.append({'mode':mode,'budget':budget,'trial':rep,'source':s,
                      'anchor_n':len(anchors),'anchor_wrong':source_stats[s]['anchor_wrong'],
                      'holdout_n':len(holdout),'holdout_wrong':held_err,
                      'holdout_error_rate':source_stats[s]['holdout_rate'],**sc})
            totalerrors=sum(st['holdout_error'] for st in source_stats.values())
            for meth in METHODS:
                ordered=sorted(source_stats,key=lambda s:(-source_stats[s]['sc'][meth],s))
                flagged=ordered[0];st=source_stats[flagged]
                focus_st=source_stats[args.focus]
                trialrows.append(dict(mode=mode,budget=budget,trial=rep,n_sources=len(source_stats),
                    method=meth,flagged_source=flagged,focus_flagged=int(flagged==args.focus),
                    focus_rank=ordered.index(args.focus)+1,flagged_score=st['sc'][meth],
                    flagged_holdout_n=st['holdout_n'],flagged_holdout_wrong=st['holdout_error'],
                    flagged_holdout_error_rate=st['holdout_rate'],
                    flagged_high_error=int(st['holdout_rate']>=args.high_error_definition),
                    flagged_holdout_error_capture=st['holdout_error']/totalerrors if totalerrors else float('nan'),
                    focus_holdout_n=focus_st['holdout_n'],
                    focus_holdout_error_rate=focus_st['holdout_rate'],
                    eligible_total_holdout_errors=totalerrors,
                    flagged_holdout_fn=st['holdout_fn'],
                    flagged_holdout_resistant=st['holdout_resistant']))
        configs.append({'mode':mode,'budget':budget,'sources_eligible_pre_split':len(candidates),
              'eligible_sources':sorted(candidates),'valid_trials':usable_trials})
    if not trialrows:raise ValueError('No configurations could be evaluated')
    result_summary=[]
    groups=collections.defaultdict(list)
    for r in trialrows:groups[(r['mode'],r['budget'],r['method'])].append(r)
    for (mode,budget,method),rows in sorted(groups.items()):
        result_summary.append(dict(mode=mode,budget=budget,method=method,trials=len(rows),
           median_n_sources=statistics.median(r['n_sources'] for r in rows),
           focus_detect_rate=sum(r['focus_flagged'] for r in rows)/len(rows),
           focus_rank_mean=statistics.mean(r['focus_rank'] for r in rows),
           mean_flagged_holdout_error_rate=statistics.mean(r['flagged_holdout_error_rate'] for r in rows),
           mean_flagged_holdout_error_capture=statistics.mean(r['flagged_holdout_error_capture'] for r in rows if math.isfinite(r['flagged_holdout_error_capture'])),
           heldout_high_error_flag_rate=sum(r['flagged_high_error'] for r in rows)/len(rows),
           mean_flagged_n=statistics.mean(r['flagged_holdout_n'] for r in rows)))
    save_csv(dst/'file16f_method_comparison.csv',result_summary,list(result_summary[0]))
    save_csv(dst/'file16f_trials.csv',trialrows,list(trialrows[0]))
    if source_rows:
        save_csv(dst/'file16f_source_trial_details.csv',source_rows,list(source_rows[0]))
    summary={
      'status':'POST_HOC_FEASIBILITY_NOT_INDEPENDENT_VALIDATION_NOT_CLINICAL',
      'generated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'input':str(inputs.resolve()), 'input_sha256':sha(inputs),
      'label_free_source_scores_sha256':sha(scoresfile),
      'e2_genomes':525,'e2_sources':32,'focus_source_posthoc_known':args.focus,
      'frozen_classifier_modified':False,'training_or_recalibration':False,
      'threshold_optimized':False,'source_ranking_policy':'select the highest-scored source for each method, per trial',
      'budget':args.budgets,'trials_requested_per_budget_mode':args.trials,
      'seed':args.seed,'min_holdout':args.min_holdout,
      'high_error_definition':args.high_error_definition,
      'methods':{
        'anchor_error_rate':'Mean 0/1 frozen prediction errors on anchor AST labels (baseline)',
        'anchor_brier':'Mean squared frozen score minus anchor resistant phenotype (baseline)',
        'anchor_high_conf_error':'Mean anchor high confidence (score <=.1 or >=.9) incorrect classifications (baseline)',
        'anchor_genotype_phenotype_discordance':'Mean discordance between restricted 5-family panel detected (binary) and phenotype; NOT a causal mechanism',
        'dual_discordance_fixed_half':'0.5 * anchor Brier + 0.5 * panel-phenotype discordance, fixed exploratory candidate; NOT calibrated, NOT novel by itself',
        'label_free_file16e':'FILE16E source anomaly score, benchmark with zero AST labels (different label budget)'
      },
      'configurations':configs,
      'limitations':[
        'Focus source was discovered after observing E2 outcomes; this is NOT independent confirmation or prospective validation.',
        'Other BioProjects from the same E2 cohort are NOT an independent external test of method design.',
        'No method was optimized on withheld genomes, but methods and budget experiment were conceived post hoc.',
        'Random holdouts from same source can have genomic or ascertainment dependence; ERD-disjoint sensitivity partly addresses this; unresolved genomic independence remains.',
        'Some sources have all resistant or all susceptible; area-under-curve cannot be estimated there.',
        'Absolute ranking score is not a calibrated probability of clinical failure; no patient-level recommendation or AST substitute.',
        'The restricted 5-family panel does not establish causality; negative does not mean absence of resistance mechanisms.',
        'Different budgets use different eligible source sets; compare methods WITHIN a budget/mode only.',
        'Repeated Monte Carlo trials are dependent and must not be used as independent confidence interval replicates.',
        'If anchor-based method does not outperform the error-rate or Brier baseline, there is NO demonstrated algorithmic novelty.'
      ]
    }
    (dst/'file16f_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[FILE16F] COMPLETE exploratory benchmark: {dst.resolve()}')
    for r in result_summary:
        if r['method'] in ('anchor_error_rate','dual_discordance_fixed_half','label_free_file16e'):
            print(f"  {r['mode']:12s} k={r['budget']:2d} {r['method']:32s}: focus_top1={r['focus_detect_rate']:.3f}, flagged_holdout_error={r['mean_flagged_holdout_error_rate']:.3f}, sources≈{r['median_n_sources']}")
    return summary

def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--input',default='results/file16d_source_audit_v1/file16d_manual_review_priority.csv')
  p.add_argument('--label-free-scores',default='results/file16e_label_free_v1/file16e_source_scores_LABEL_FREE.csv')
  p.add_argument('--outdir',default='results/file16f_anchor_benchmark_v1')
  p.add_argument('--budgets',nargs='+',type=int,default=[5,10,20])
  p.add_argument('--trials',type=int,default=300)
  p.add_argument('--seed',type=int,default=42)
  p.add_argument('--min-holdout',type=int,default=5)
  p.add_argument('--focus',default=DEFAULT_FOCUS)
  p.add_argument('--high-error-definition',type=float,default=0.3)
  p.add_argument('--save-per-trial-sources',action='store_true')
  args=p.parse_args()
  if any(k<1 for k in args.budgets) or args.trials<1 or args.min_holdout<1: p.error('budget/trials/min-holdout must be positive')
  run(args)
if __name__=='__main__':main()
