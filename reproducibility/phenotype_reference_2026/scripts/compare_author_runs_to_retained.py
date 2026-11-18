#!/usr/bin/env python3
"""Exact run-to-BioSample comparison of a pinned author label matrix and retained AST."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from compare_ncbi_retained_ast import parse_ast


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--old",type=Path,required=True)
    p.add_argument("--new",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    matrix=pd.read_csv(args.matrix,sep='\t',usecols=['sample_id','antibiotic','label'],dtype=str,keep_default_na=False)
    runs={s for s in matrix.sample_id if re.fullmatch(r'(?:SRR|ERR|DRR)\d+',s)}
    observations={}
    for side,path in [('old',args.old),('new',args.new)]:
        hits=defaultdict(list)
        with path.open() as h:
            for row in csv.DictReader(h,delimiter='\t',quoting=csv.QUOTE_NONE):
                matched=runs & set(re.findall(r'(?:SRR|ERR|DRR)\d+',row['Run']))
                for run in matched:
                    hits[run].append({'target_acc':row['target_acc'],'biosample':row['biosample_acc'],'project':row['bioproject_acc'],
                                      'scientific_name':row['scientific_name'],'ast':parse_ast(row['AST_phenotypes'])})
        observations[side]=hits
    rows=[]
    for record in matrix.drop_duplicates(['sample_id','antibiotic','label']).to_dict('records'):
        run,drug=record['sample_id'],record['antibiotic'].strip().casefold()
        if run not in runs:continue
        a,b=observations['old'].get(run,[]),observations['new'].get(run,[])
        samples={r['biosample'] for r in a+b}
        result={**record,'old_BioSamples':'|'.join(sorted({r['biosample'] for r in a})),
                'new_BioSamples':'|'.join(sorted({r['biosample'] for r in b})),
                'old_projects':'|'.join(sorted({r['project'] for r in a})),
                'new_projects':'|'.join(sorted({r['project'] for r in b})),
                'old_source_species':'|'.join(sorted({r['scientific_name'] for r in a})),
                'new_source_species':'|'.join(sorted({r['scientific_name'] for r in b}))}
        if len(samples)>1:status='AMBIGUOUS_RUN_TO_MULTIPLE_BIOSAMPLES'
        elif not a or not b:status='RUN_NOT_PAIRED_IN_RETAINED_REFERENCES'
        else:status='PAIRED_RUN_BIOSAMPLE'
        for side,records in [('old',a),('new',b)]:
            values=set().union(*(r['ast'].get(drug,frozenset()) for r in records)) if records else set()
            binary={0 if x=='S' else 1 for x in values if x in {'S','I','R'}}
            result[side+'_native_category_set']='|'.join(sorted(values))
            result[side+'_binary_label']=str(next(iter(binary))) if len(binary)==1 and values<={'S','I','R'} else ''
        result['linkage_status']=status
        result['known_focal_project']=any('PRJNA1322038' in r['project'] for r in a+b)
        result['author_matches_old']=record['label']==result['old_binary_label'] if result['old_binary_label'] else 'NOT_COMPARABLE'
        result['author_matches_new']=record['label']==result['new_binary_label'] if result['new_binary_label'] else 'NOT_COMPARABLE'
        rows.append(result)
    with (args.out/'author_run_reference_comparisons.csv').open('x',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paired=[r for r in rows if r['linkage_status']=='PAIRED_RUN_BIOSAMPLE']
    comparable=[r for r in paired if r['old_binary_label'] and r['new_binary_label']]
    report={'status':'EXPLORATORY_EXACT_RUN_LABEL_REFERENCE_COMPARISON_NOT_PAPER_CORRECTION',
            'author_matrix_unique_sample_ids':matrix.sample_id.nunique(),'author_unique_SRA_runs':len(runs),
            'old_matched_runs':len(observations['old']),'new_matched_runs':len(observations['new']),
            'author_run_drug_rows':len(rows),'linkage_status_counts':dict(Counter(r['linkage_status'] for r in rows)),
            'paired_SIR_binary_comparable_run_drug_rows':len(comparable),
            'old_new_binary_label_differences':sum(r['old_binary_label']!=r['new_binary_label'] for r in comparable),
            'author_label_differs_from_old':sum(r['author_matches_old'] is False for r in comparable),
            'author_label_differs_from_new':sum(r['author_matches_new'] is False for r in comparable),
            'known_focal_project_run_drug_rows':sum(r['known_focal_project'] for r in rows),
            'known_focal_project_unique_runs':len({r['sample_id'] for r in rows if r['known_focal_project']}),
            'input_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.matrix,args.old,args.new]},
            'limitations':['Exact run/BioSample links do not establish that all samples belong to distinct patients or studies.',
                           'Unmatched runs remain unresolved; absence of a link is not proof of independence.',
                           'I/R binary mapping follows the author convention only for this comparison, not the main audit.',
                           'Agreement with a particular reference state does not prove authors used that snapshot or establish clinical truth.',
                           'No author models were deserialized and no predictions were generated.']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
