#!/usr/bin/env python3
"""Bounded, pinned release comparison; annotation changes are not AST errors."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from compare_ncbi_retained_ast import parse_ast, comparison, selected


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while b := f.read(2**20): h.update(b)
    return h.hexdigest()


def save(p, obj):
    t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj, indent=2)+'\n'); t.replace(p)


def fetch(target, root):
    p = root/(target['new_build']+'.metadata.tsv'); receipt = p.with_suffix('.receipt.json')
    if p.exists() and receipt.exists():
        d = json.loads(receipt.read_text())
        if digest(p) == d['sha256'] and d['url'] == target['new_url']: return d
        raise ValueError('Existing acquisition hash mismatch')
    tmp = p.with_suffix('.part'); h = hashlib.sha256(); size = 0
    with urllib.request.urlopen(target['new_url'], timeout=60) as r, tmp.open('wb') as f:
        status = r.status; headers = dict(r.headers)
        if status != 200: raise ValueError('Incomplete HTTP response')
        while b := r.read(2**20): f.write(b); h.update(b); size += len(b)
    if headers.get('Content-Length') and int(headers['Content-Length']) != size:
        raise ValueError('Incomplete metadata download')
    tmp.replace(p)
    d = {'url': target['new_url'], 'http_status': status, 'bytes': size,
         'sha256': h.hexdigest(), 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
         'last_modified': headers.get('Last-Modified'), 'file': p.name}
    save(receipt, d); print(json.dumps({'download': p.name, 'bytes': size}), flush=True)
    return d


def load(p, species, expected):
    if digest(p) != expected: raise ValueError('Pinned metadata hash mismatch')
    bios, targets = defaultdict(dict), {}; focal = set(); counts = Counter()
    with p.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            counts['all_rows'] += 1
            if None in row or any(v is None for v in row.values()): raise ValueError('Bad row width')
            bio = row['biosample_acc']
            # Exclusion is release-wide, including AST-missing rows and taxon changes.
            if 'PRJNA1322038' in re.findall(r'PRJ(?:NA|EB|DB)\d+', row['bioproject_acc']): focal.add(bio)
            if not selected(row['scientific_name'], species): continue
            counts['source_species_rows'] += 1
            ast = parse_ast(row['AST_phenotypes'])
            if not ast or not re.fullmatch(r'SAM[NED][A-Z]?\d+', bio): continue
            counts['source_species_AST_rows_with_BioSample'] += 1
            for drug, cats in ast.items():
                bios[bio].setdefault(drug, set()).update(cats)
            key = row['target_acc'], bio
            if key in targets and targets[key] != ast: raise ValueError('Conflicting target duplicate')
            targets[key] = ast
    return bios, targets, focal, dict(counts)


def compare(old, new, excluded, species, keytype):
    rows = []; summary = Counter(); compared_bios = set()
    for key in sorted(set(old) & set(new)):
        bio = key if isinstance(key, str) else key[1]
        if bio in excluded: continue
        for drug in sorted(set(old[key]) & set(new[key])):
            a, b = old[key][drug], new[key][drug]; status = comparison(a, b)
            summary[status] += 1; compared_bios.add(bio)
            rows.append({'species': species, 'biosample': bio, 'target_acc': '' if isinstance(key, str) else key[0],
                         'antibiotic': drug, 'old_categories': '|'.join(sorted(a)),
                         'new_categories': '|'.join(sorted(b)), 'comparison': status,
                         'raw_categories_changed': a != b, 'key_type': keytype})
    return rows, {'counts': dict(summary), 'paired_keys': len(rows), 'BioSamples': len(compared_bios)}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); root = a.root; protocol = json.loads((root/'PROTOCOL.json').read_text())
    expected = (root/'PROTOCOL.sha256').read_text().split()[0]
    if digest(root/'PROTOCOL.json') != expected: raise ValueError('Protocol changed')
    with ThreadPoolExecutor(max_workers=4) as pool: receipts = list(pool.map(lambda t: fetch(t, root), protocol['targets']))
    allrows = []; reports = []
    # Sequential loading bounds peak memory on the large E. coli release.
    for t, receipt in zip(protocol['targets'], receipts):
        old = load(Path(t['old_file']), t['species'], t['old_sha256'])
        new = load(root/receipt['file'], t['species'], receipt['sha256'])
        excluded = old[2] | new[2]
        r, point = compare(old[0], new[0], excluded, t['species'], 'BioSample_drug_aggregated')
        secondary, secondary_point = compare(old[1], new[1], excluded, t['species'], 'target_BioSample_drug')
        primary = [x for x in r if x['antibiotic'] in protocol['primary_drugs']]
        report = dict(species=t['species'], old_build=t['old_build'], new_build=t['new_build'],
                      old_inventory=old[3], new_inventory=new[3], excluded_focal_BioSamples=len(excluded),
                      aggregated=point, secondary_target=secondary_point,
                      primary_meropenem_counts=dict(Counter(x['comparison'] for x in primary)))
        allrows.extend(r); reports.append(report); print(json.dumps(report), flush=True)
        changed = [x for x in secondary if x['comparison'] not in {'SAME_SIR_SET', 'NON_SIR_OR_MISSING_INVOLVED'}]
        if changed:
            with (root/(t['new_build']+'_secondary_target_changes.csv')).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(changed[0])); w.writeheader(); w.writerows(changed)
        del old, new
    fields = ['species','biosample','target_acc','antibiotic','old_categories','new_categories','comparison','raw_categories_changed','key_type']
    for name, rows in [('all_aggregated_comparisons.csv', allrows), ('SIR_change_candidates.csv', [r for r in allrows if r['comparison'] not in {'SAME_SIR_SET','NON_SIR_OR_MISSING_INVOLVED'}])]:
        with (root/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    save(root/'report.json', {'status':'COMPLETE_BOUNDED_RELEASE_ANNOTATION_AUDIT', 'protocol_sha256':expected,
         'targets':reports, 'total_primary_aggregated_keys':len(allrows),
         'unopposed_RS_candidates':sum(r['comparison']=='UNOPPOSED_R_S_REVERSAL_CANDIDATE' for r in allrows),
         'other_SIR_change_candidates':sum(r['comparison']=='OTHER_SIR_SET_CHANGE' for r in allrows),
         'limitations':['Metadata release annotations, not independently repeated AST.', 'No observed reversal does not prove absence in unpaired records or other windows.', 'Species labels are source annotations; one project is not one laboratory.', 'No historical MIC series is supplied by these metadata.']})


if __name__ == '__main__': main()
