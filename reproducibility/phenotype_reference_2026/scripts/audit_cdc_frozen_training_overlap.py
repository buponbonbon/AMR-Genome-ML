#!/usr/bin/env python3
"""Audit CDC candidate identity against frozen fitting IDs using retained public metadata.

Never equates absence of an exact ID match with clonal or clinical independence.
No fitting, scoring, or phenotype recoding is performed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assembly_key(accession):
    """Only for an additional conservative identity screen; retain raw accession."""
    match = re.fullmatch(r'GC[AF]_(\d+)\.\d+', accession or '')
    return match.group(1) if match else None


def audit(candidates, folds, cases, queried_records):
    if not candidates.biosample_acc.is_unique or not folds['Genome ID'].is_unique:
        raise ValueError('Nonunique candidate BioSamples or fitting IDs')
    fit = folds.set_index('Genome ID')
    selected = candidates.set_index('biosample_acc')
    samples = set(selected.index)
    assembly_samples = defaultdict(set)
    for sample, row in selected.iterrows():
        assembly_samples[assembly_key(row.asm_acc)].add(sample)
    if None in assembly_samples or any(len(v) != 1 for v in assembly_samples.values()):
        raise ValueError('Missing or ambiguous candidate assembly identity')
    observations = defaultdict(list)
    for case in cases['cases']:
        if case['biosample_acc'] in samples and case['antibiotic'] == 'meropenem':
            observations[case['biosample_acc']].extend(case['CDC_observations'])
    by_sample = defaultdict(dict)
    linkage = []
    for source, records in queried_records.items():
        for record in records:
            genome_id = record['genome_id']
            bio = record.get('biosample_accession', '')
            key = assembly_key(record.get('assembly_accession', ''))
            via_assembly = assembly_samples.get(key, set())
            possible = ({bio} if bio in samples else set()) | via_assembly
            if not possible:
                raise ValueError(f'Public response does not match candidate query: {genome_id}')
            if len(possible) != 1 or (bio and bio not in possible):
                raise ValueError(f'Conflicting BioSample/assembly linkage: {genome_id}')
            sample = next(iter(possible))
            prior = by_sample[sample].get(genome_id)
            if prior is not None and prior != record:
                raise ValueError(f'Inconsistent duplicate provider record: {genome_id}')
            by_sample[sample][genome_id] = record
            linkage.append({
                'query': source, 'candidate_biosample': sample,
                'provider_biosample': bio, 'provider_genome_id': genome_id,
                'provider_assembly': record.get('assembly_accession', ''),
                'provider_genome_name': record.get('genome_name', ''),
                'provider_taxon_id': record.get('taxon_id', ''),
                'same_assembly_number_ignoring_GCA_GCF_and_version': key in assembly_samples,
                'exact_fitting_id_overlap': genome_id in fit.index,
                'training_label_if_overlap': fit.loc[genome_id, 'Phenotype'] if genome_id in fit.index else '',
            })
    rows = []
    for sample, selected_row in selected.iterrows():
        obs = observations[sample]
        if not obs:
            raise ValueError(f'No primary CDC meropenem observation: {sample}')
        labels = sorted({o['category_raw'] for o in obs})
        if len(labels) != 1:
            raise ValueError(f'Conflicting CDC categories require adjudication: {sample}')
        label = labels[0]
        ids = sorted(by_sample[sample])
        overlaps = sorted(set(ids) & set(fit.index))
        species_ok = selected_row.scientific_name == 'Klebsiella pneumoniae'
        if not ids:
            status = 'IDENTITY_UNRESOLVED'
        elif not species_ok:
            status = 'EXCLUDE_SPECIES_MISMATCH'
        elif overlaps:
            status = 'EXCLUDE_KNOWN_FITTING_SAMPLE'
        elif label not in {'R', 'S'}:
            status = 'EXCLUDE_FROM_BINARY_RS_KEEP_ORIGINAL_CATEGORY'
        else:
            status = 'NO_EXACT_FITTING_ID_MATCH_CLONAL_GATE_PENDING'
        rows.append({
            'biosample_acc': sample,
            'CDC_ARBankNums': ';'.join(sorted({o['ARBankNum'] for o in obs})),
            'CDC_organisms': ';'.join(sorted({o['CDC_organism'] for o in obs})),
            'NCBI_scientific_name': selected_row.scientific_name,
            'NCBI_assembly': selected_row.asm_acc,
            'CDC_meropenem_MIC_raw': ';'.join(sorted({o['MIC_raw'] for o in obs})),
            'CDC_meropenem_category_raw': label,
            'CDC_BMD_stated_all_observations': all(o['BMD_stated'] for o in obs),
            'CDC_CLSI_editions_stated': ';'.join(sorted({s for o in obs for s in o['CLSI_editions_stated']})),
            'provider_genome_ids': ';'.join(ids),
            'exact_fitting_ids': ';'.join(overlaps),
            'fitting_labels_for_overlapping_ids': ';'.join(f'{g}:{fit.loc[g,"Phenotype"]}' for g in overlaps),
            'known_fitting_overlap': bool(overlaps),
            'species_matches_frozen_model': species_ok,
            'screening_status': status,
            'clonal_independence_established': False,
        })
    table = pd.DataFrame(rows)
    kp = table[table.species_matches_frozen_model]
    binary = kp[kp.CDC_meropenem_category_raw.isin(['R', 'S'])]
    remaining = binary[~binary.known_fitting_overlap]
    report = {
        'status': 'CDC_PILOT_FAILS_INDEPENDENT_TWO_CLASS_VALIDATION_GATE',
        'candidate_biosamples': len(table),
        'matching_species_biosamples': len(kp),
        'known_fitting_overlap_biosamples_all_candidates': int(table.known_fitting_overlap.sum()),
        'distinct_overlapping_fitting_genome_ids': len({g for s in table.exact_fitting_ids for g in s.split(';') if g}),
        'matching_species_original_label_counts': dict(Counter(kp.CDC_meropenem_category_raw)),
        'matching_species_binary_RS': len(binary),
        'binary_RS_known_fitting_overlap_biosamples': int(binary.known_fitting_overlap.sum()),
        'binary_RS_no_exact_fitting_match_biosamples': len(remaining),
        'remaining_binary_label_counts': dict(Counter(remaining.CDC_meropenem_category_raw)),
        'screening_status_counts': dict(Counter(table.screening_status)),
        'all_candidates_have_provider_identity_records': bool(table.provider_genome_ids.ne('').all()),
        'no_training_or_tuning': True,
        'I_or_ND_recoded': False,
        'limitations': [
            'The CDC pilot is a selected public reference collection, not a representative clinical cohort.',
            'Exact fitting-ID overlap proves sample reuse; no exact match does not establish genome-distance or clonal independence.',
            'GCA/GCF number and version normalization is only an additional conservative identity screen, not a sequence-identity claim.',
            'No clinical performance estimate is produced from this single-class residual sample set.',
            'The overlap concerns this proposed CDC challenge, not evidence that the original E2 validation overlaps training.',
        ],
    }
    return table, pd.DataFrame(linkage), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = {}
    input_hashes = {}
    for source in ['by_biosample', 'by_assembly']:
        path = args.root/'cdc_identity_queries'/f'{source}.json'
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        input_hashes[str(path)] = digest(path)
        if receipt['status'] != 200 or digest(path) != receipt['sha256']:
            raise ValueError(f'Public metadata receipt/hash failed: {source}')
        records[source] = json.loads(path.read_text())
        total = int(receipt['content_range'].rsplit('/', 1)[1])
        if len(records[source]) != total or len(records[source]) != receipt['record_count'] or total >= 1000:
            raise ValueError(f'Incomplete paginated response: {source}')
    candidate_path = args.root/'cdc_candidate_retained_genome_links.csv'
    fold_path = args.root/'received/project/data/splits/file07_outer_folds.csv'
    for path in [candidate_path, fold_path, args.cases]:
        input_hashes[str(path)] = digest(path)
    candidates = pd.read_csv(candidate_path, dtype=str, keep_default_na=False)
    folds = pd.read_csv(fold_path, dtype=str, keep_default_na=False)
    table, links, report = audit(candidates, folds, json.loads(args.cases.read_text()), records)
    report['input_sha256'] = input_hashes
    args.out.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.out/'CDC_frozen_model_identity_gate.csv', index=False)
    links.to_csv(args.out/'CDC_provider_identity_evidence.csv', index=False)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
