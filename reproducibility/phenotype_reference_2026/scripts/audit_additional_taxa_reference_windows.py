#!/usr/bin/env python3
"""Screen the four pinned additional taxa, with an independent parser check."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
from audit_november_retained_phenotypes import load,compare
from verify_november_phenotype_audit import load as independent_load

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    protocol_path=root/'new_taxa_temporal_protocol.json';protocol=json.loads(protocol_path.read_text())
    assert sha(protocol_path)==protocol_path.with_suffix('.json.sha256').read_text().split()[0]
    data=root/'new_taxa_metadata';manifest=json.loads((data/'manifest.json').read_text());assert manifest['all_complete']
    receipts={r['build']:r for r in manifest['responses']};reports=[];allrows=[];target_changes=[]
    for t in protocol['targets']:
        paths=[data/(t[phase+'_build']+'.metadata.tsv') for phase in ['old','new']]
        old,new=[load(p,t['species'],receipts[t[phase+'_build']]['sha256']) for p,phase in zip(paths,['old','new'])]
        excluded=old[2]|new[2];rows,summary=compare(old[0],new[0],excluded,t['species'],'BioSample_drug_aggregated')
        second,second_summary=compare(old[1],new[1],excluded,t['species'],'target_BioSample_drug')
        # A separate chunked pandas loader uses a regex token parser and flat keys.
        io,of=independent_load(paths[0],t['species']);inew,nf=independent_load(paths[1],t['species']);checked=Counter()
        for key in io.keys()&inew.keys():
            if key[0] in of|nf:continue
            aa=io[key]&{'S','I','R'};bb=inew[key]&{'S','I','R'}
            state='NON_SIR_OR_MISSING_INVOLVED' if not aa or not bb else 'SAME_SIR_SET' if aa==bb else 'UNOPPOSED_R_S_REVERSAL_CANDIDATE' if (aa,bb) in [({'S'},{'R'}),({'R'},{'S'})] else 'OTHER_SIR_SET_CHANGE'
            checked[state]+=1
        assert dict(checked)==summary['counts']
        changed=[r for r in rows if r['comparison'] not in {'SAME_SIR_SET','NON_SIR_OR_MISSING_INVOLVED'}]
        reports.append(dict(**t,old_inventory=old[3],new_inventory=new[3],excluded_focal_BioSamples=len(excluded),aggregated=summary,secondary_target=second_summary,
            paired_SIR_keys=sum(n for k,n in summary['counts'].items() if k!='NON_SIR_OR_MISSING_INVOLVED'),
            primary_drug_counts={drug:dict(Counter(r['comparison'] for r in rows if r['antibiotic']==drug)) for drug in protocol['primary_drugs']},independent_counts_match=True))
        allrows.extend(rows);target_changes.extend(r for r in second if r['comparison'] not in {'SAME_SIR_SET','NON_SIR_OR_MISSING_INVOLVED'})
        print(json.dumps(reports[-1]),flush=True)
    fields=['species','biosample','target_acc','antibiotic','old_categories','new_categories','comparison','raw_categories_changed','key_type']
    for name,rows in [('new_taxa_all_paired_keys.csv',allrows),('new_taxa_SIR_candidates.csv',[r for r in allrows if r['comparison'] not in {'SAME_SIR_SET','NON_SIR_OR_MISSING_INVOLVED'}]),('new_taxa_target_candidates.csv',target_changes)]:
        with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    r=dict(status='COMPLETE_ADDITIONAL_TAXA_PAIRED_SCREEN_INDEPENDENTLY_RECONSTRUCTED',protocol_sha256=sha(protocol_path),targets=reports,
        total_paired_keys=len(allrows),paired_SIR_keys=sum(x['paired_SIR_keys'] for x in reports),unopposed_RS_candidates=sum(r['comparison']=='UNOPPOSED_R_S_REVERSAL_CANDIDATE' for r in allrows),other_SIR_candidates=sum(r['comparison']=='OTHER_SIR_SET_CHANGE' for r in allrows),
        limits=['A source record change is not independently adjudicated AST error.','No changes are inferred for unpaired records.','These new taxa are not scored by the K.pneumoniae classifier.','Candidate edits require project/source and accession history followup before counting independent events.','Source-composition and temporal-window selection prevent population incidence estimates.'])
    (root/'new_taxa_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='targets'},indent=2))

if __name__=='__main__':main()
