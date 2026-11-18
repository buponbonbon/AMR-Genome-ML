#!/usr/bin/env python3
"""Screen a published MIC cohort against the unchanged frozen fitting identities.

This is a metadata/eligibility audit, not prediction or clinical validation.
Original MICs, S/I/R, guideline cells and identifiers are retained verbatim.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

FOLDS_SHA = '3183e2b458fc089633fde779204e3957c7519f8b20651b4207dee9f7e10a21c4'
WORKBOOK_SHA = 'a486d288842447fed0d50d97ed9e78471b8a2c671fc6015f95bee230d5c2443a'
E2_SHA = '6b56e1eeb60421261d6933bf89897d0a10faae4ba7cea8399787b76008db9ef1'
LABEL = 'S/I/R (PHAS interpretation)_Meropenem'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_retained(path):
    receipt = json.loads(path.with_name(path.name + '.receipt.json').read_text())
    if receipt.get('http_status') != 200 or sha(path) != receipt['sha256']:
        raise ValueError(f'Acquisition receipt failed: {path}')
    return receipt


def st_number(value):
    match = re.fullmatch(r'(?:ST)?([1-9][0-9]*)', str(value).strip())
    return match.group(1) if match else ''


def training_identities(root, folds):
    records = []
    receipts = {}
    for path in sorted((root/'identity').glob('training_identity_[0-9][0-9].json')):
        receipt = read_retained(path)
        rows = json.loads(path.read_text())
        queried = set(receipt['queried_genome_ids'])
        if len(rows) != len(queried) or {r['genome_id'] for r in rows} != queried:
            raise ValueError('Missing, duplicate or unexpected fitting genome identity')
        if int(receipt['content_range'].split('/')[-1]) != len(rows):
            raise ValueError('Incomplete provider response')
        records.extend(rows)
        receipts[str(path.relative_to(root))] = sha(path)
    training = pd.DataFrame(records).fillna('')
    if not training.genome_id.is_unique or set(training.genome_id) != set(folds['Genome ID']):
        raise ValueError('Provider identity set differs from frozen File07 cohort')
    supplemental_path = root/'unlinked_training/training_unlinked_full_records.json'
    receipt = read_retained(supplemental_path)
    extra = json.loads(supplemental_path.read_text())
    missing_ids = set(training.loc[training.biosample_accession.eq(''), 'genome_id'])
    if {r['genome_id'] for r in extra} != missing_ids or len(extra) != len(missing_ids):
        raise ValueError('Supplemental records do not cover exactly the missing fitting BioSamples')
    secondary_path = root/'secondary_resolution/training_secondary_full_records.json'
    read_retained(secondary_path)
    secondary = json.loads(secondary_path.read_text())
    secondary_ids = set(training.loc[training.biosample_accession.str.fullmatch(r'ERS[0-9]+'), 'genome_id'])
    if {r['genome_id'] for r in secondary} != secondary_ids or len(secondary) != len(secondary_ids):
        raise ValueError('Supplemental secondary sample-ID coverage mismatch')
    extra.extend(secondary)
    bridge_path = root/'training_bridge/training_missing_biosample_ENA.tsv'
    read_retained(bridge_path)
    bridge = pd.read_csv(bridge_path, sep='\t', dtype=str, keep_default_na=False)
    secondary_bridge_path = root/'secondary_SRA/training_secondary_SRA_ENA.tsv'
    read_retained(secondary_bridge_path)
    secondary_bridge = pd.read_csv(secondary_bridge_path, sep='\t', dtype=str, keep_default_na=False)
    bridge = pd.concat([bridge, secondary_bridge], ignore_index=True)
    if not bridge.run_accession.is_unique:
        raise ValueError('Ambiguous ENA training-run bridge')
    bridge = bridge.set_index('run_accession')
    linked = training.set_index('genome_id')
    linked['provider_sample_accession_original'] = linked.biosample_accession
    linked['identity_link_source'] = 'BV_BRC_BIOSAMPLE'
    linked['training_run_bridge'] = ''
    for row in extra:
        runs = re.findall(r'(?:SRR|ERR|DRR)\d+', row.get('sra_accession', ''))
        if not runs or not set(runs) <= set(bridge.index):
            raise ValueError(f'Unresolved training SRA identifier: {row["genome_id"]}')
        samples = {bridge.loc[run, 'sample_accession'] for run in runs}
        if len(samples) != 1 or not next(iter(samples)):
            raise ValueError('Ambiguous or empty training BioSample bridge')
        original_id = linked.loc[row['genome_id'], 'provider_sample_accession_original']
        if original_id and {bridge.loc[run, 'secondary_sample_accession'] for run in runs} != {original_id}:
            raise ValueError('The provider secondary sample does not match the ENA run mapping')
        linked.loc[row['genome_id'], 'biosample_accession'] = next(iter(samples))
        linked.loc[row['genome_id'], 'identity_link_source'] = (
            'BV_BRC_SECONDARY_SAMPLE_AND_SRA_TO_ENA_BIOSAMPLE' if original_id else 'BV_BRC_SRA_TO_ENA_BIOSAMPLE')
        linked.loc[row['genome_id'], 'training_run_bridge'] = ';'.join(runs)
    if linked.biosample_accession.eq('').any():
        raise ValueError('An original fitting ID is still unlinked')
    receipts[str(supplemental_path.relative_to(root))] = sha(supplemental_path)
    receipts[str(bridge_path.relative_to(root))] = sha(bridge_path)
    receipts[str(secondary_path.relative_to(root))] = sha(secondary_path)
    receipts[str(secondary_bridge_path.relative_to(root))] = sha(secondary_bridge_path)
    return linked.reset_index(), receipts


def make_cluster_query(training, cohort):
    """Prepare a read-only query; never execute it or alter any remote dataset."""
    role_samples = []
    for role, values in [('fitting', training.biosample_accession),
                         ('CCRE', cohort.sample_accession)]:
        samples = sorted(set(values))
        if any(not re.fullmatch(r'(?:SAM[NED][A-Z]?)[0-9]+', s) for s in samples):
            raise ValueError('Unexpected BioSample syntax in SQL input')
        array = ',\n'.join("    '" + sample + "'" for sample in samples)
        role_samples.append("SELECT '"+role+"' AS role, biosample_acc FROM UNNEST([\n"+array+"\n  ]) AS biosample_acc")
    return """-- Read-only BigQuery export for the remaining ERD/SNP-cluster gate.
-- Prepared from retained BioSample identities; NOT executed in this audit.
-- Export all rows, including null ERD. Preserve the query/job ID and UTC time.
WITH requested AS (
""" + '\nUNION ALL\n'.join(role_samples) + """
)
SELECT r.role, r.biosample_acc, i.target_acc, i.asm_acc, i.erd_group
FROM requested AS r
LEFT JOIN `ncbi-pathogen-detect.pdbrowser.isolates` AS i
  ON i.biosample_acc = r.biosample_acc
ORDER BY r.role, r.biosample_acc, i.target_acc, i.asm_acc, i.erd_group;
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--E2-predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    workbook = args.root/'appendices/CCRE_appendix2.xlsx'
    if sha(args.folds) != FOLDS_SHA or sha(workbook) != WORKBOOK_SHA or sha(args.E2_predictions) != E2_SHA:
        raise ValueError('Frozen or primary input hash differs from retained source')
    read_retained(workbook)
    folds = pd.read_csv(args.folds, dtype=str, keep_default_na=False)
    e2 = pd.read_csv(args.E2_predictions, dtype=str, keep_default_na=False)
    training, input_hashes = training_identities(args.root, folds)
    raw = pd.read_excel(workbook, dtype=str, keep_default_na=False)
    if not raw['Run accession'].is_unique or not raw['Sample name'].is_unique:
        raise ValueError('Repeated sample or run in primary workbook')
    primary = raw[raw.species.eq('Klebsiella pneumoniae')].copy()
    ena_frames = []
    for project in ['PRJEB39943','PRJEB51224']:
        path = args.root/'identity'/f'{project}_ENA_runs.tsv'
        read_retained(path)
        input_hashes[str(path.relative_to(args.root))] = sha(path)
        ena_frames.append(pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False))
    ena = pd.concat(ena_frames, ignore_index=True)
    if not ena.run_accession.is_unique:
        raise ValueError('Repeated ENA runs across CCRE projects')
    merged = primary.merge(ena, left_on='Run accession', right_on='run_accession',
                           how='left', validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all() or not merged.sample_accession.is_unique:
        raise ValueError('Missing or repeated candidate BioSample identity')
    merged = merged.drop(columns='_merge')
    bio_to_training = defaultdict(list)
    for row in training.itertuples():
        bio_to_training[row.biosample_accession].append(row.genome_id)
    merged['overlapping_fitting_genome_ids'] = merged.sample_accession.map(
        lambda bio: ';'.join(sorted(bio_to_training[bio])))
    merged['exact_biosample_overlap_fitting'] = merged.overlapping_fitting_genome_ids.ne('')
    merged['exact_biosample_overlap_E2'] = merged.sample_accession.isin(set(e2.biosample_acc))
    merged['species_annotation_agrees_ENA'] = merged.scientific_name.eq('Klebsiella pneumoniae')
    merged['MIC_meropenem_present'] = ~merged.MIC_Meropenem.str.strip().isin(['','Missing','Not tested','NULL','NA'])
    merged['explicit_original_binary_RS'] = merged[LABEL].isin(['R','S'])
    merged['primary_MIC_RS_and_no_detected_exact_overlap'] = (
        merged.MIC_meropenem_present & merged.explicit_original_binary_RS &
        ~merged.exact_biosample_overlap_fitting & ~merged.exact_biosample_overlap_E2 &
        merged.species_annotation_agrees_ENA)
    train_st = {st_number(x) for x in folds.MLST} - {''}
    merged['reported_Kleborate_ST_normalized'] = merged.ST.map(st_number)
    merged['ST_seen_in_fitting_File07'] = merged.reported_Kleborate_ST_normalized.map(
        lambda st: 'UNKNOWN' if not st else ('YES' if st in train_st else 'NO'))
    merged['ERD_SNP_firewall_status'] = 'PENDING_NOT_ESTABLISHED_BY_BIOSAMPLE_OR_MLST'
    eligible = merged[merged.primary_MIC_RS_and_no_detected_exact_overlap]
    input_hashes.update({'primary_workbook':sha(workbook), 'File07':sha(args.folds),
                         'original_E2_predictions':sha(args.E2_predictions)})
    report = {
        'status':'PRIMARY_MIC_COHORT_IDENTIFIED_EXACT_ID_SCREEN_PASSED_CLUSTER_GATE_PENDING',
        'primary_source_DOI':'10.1016/j.lanmic.2025.101320',
        'primary_workbook_sha256':WORKBOOK_SHA,
        'primary_source_all_species_samples':len(raw),
        'primary_K_pneumoniae_samples':len(primary),
        'Kp_original_meropenem_category_counts':primary[LABEL].value_counts().to_dict(),
        'Kp_nonmissing_meropenem_MIC':int(merged.MIC_meropenem_present.sum()),
        'Kp_MIC_original_category_counts':merged.loc[merged.MIC_meropenem_present,LABEL].value_counts().to_dict(),
        'frozen_fitting_genome_IDs':len(training),
        'frozen_fitting_IDs_linked_to_BioSamples':int(training.biosample_accession.ne('').sum()),
        'frozen_fitting_unique_BioSamples':training.biosample_accession.nunique(),
        'fitting_IDs_resolved_via_SRA_ENA_bridge':int(training.identity_link_source.eq('BV_BRC_SRA_TO_ENA_BIOSAMPLE').sum()),
        'fitting_IDs_with_secondary_ENA_sample_alias_resolved':int(training.identity_link_source.eq('BV_BRC_SECONDARY_SAMPLE_AND_SRA_TO_ENA_BIOSAMPLE').sum()),
        'candidate_Kp_runs_linked_to_unique_BioSamples':len(merged),
        'candidate_Kp_exact_BioSample_overlap_fitting':int(merged.exact_biosample_overlap_fitting.sum()),
        'candidate_Kp_exact_BioSample_overlap_E2':int(merged.exact_biosample_overlap_E2.sum()),
        'candidate_ENA_primary_species_annotation_agreement':int(merged.species_annotation_agrees_ENA.sum()),
        'eligible_primary_MIC_original_RS_no_detected_exact_overlap':len(eligible),
        'eligible_original_RS_counts':eligible[LABEL].value_counts().to_dict(),
        'eligible_reporting_hospitals':eligible['Hospital code (CCRE survey)'].nunique(),
        'eligible_reporting_countries':eligible['Country of origin'].nunique(),
        'eligible_ST_seen_in_fitting_counts':eligible.ST_seen_in_fitting_File07.value_counts().to_dict(),
        'eligible_unseen_ST_original_RS_counts':eligible.loc[eligible.ST_seen_in_fitting_File07.eq('NO'),LABEL].value_counts().to_dict(),
        'frozen_model_predictions_computed':False,
        'new_model_training_or_threshold_tuning':False,
        'I_missing_or_NS_recoded_to_binary_R_S':False,
        'genomic_cluster_or_ERD_independence_established':False,
        'MIC_breakpoint_reclassification_performed':False,
        'key_limits':[
            'No exact BioSample overlap was found after linking all fitting IDs; this does not establish genomic-cluster, patient or hospital independence.',
            'This is an enriched surveillance collection with carbapenem-R/I and carbapenem-S sampling, not an unselected prevalence cohort.',
            'The study used heterogeneous AST methods including disc diffusion, BMD, gradient tests and automated systems; the workbook does not give a method or breakpoint edition for each MIC.',
            'The meropenem reference here is the original centrally validated PHAS S/I/R column; the broad carbapenem collection group and genotype are not labels.',
            'The original Interpretation guideline used column is retained, including CLSI entries; do not assume these are all original EUCAST or BMD measurements.',
            'Seen/unseen MLST is descriptive and is not a substitute for the original ERD/SNP-cluster firewall.',
            'No model accuracy, advantage, clinical utility or new independent label-reversal event is established by this screening.',
        ],
        'input_sha256':input_hashes,
    }
    sql = make_cluster_query(training, merged)
    args.out.mkdir(parents=True, exist_ok=False)
    training.to_csv(args.out/'frozen_training_identity_complete.csv',index=False)
    merged.to_csv(args.out/'CCRE_Kp_identity_and_MIC_screen.csv',index=False)
    eligible.to_csv(args.out/'CCRE_candidate_primary_MIC_RS.csv',index=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'CCRE_and_fitting_ERD_query.sql').write_text(sql)
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
