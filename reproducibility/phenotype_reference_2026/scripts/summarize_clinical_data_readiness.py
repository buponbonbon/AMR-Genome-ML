#!/usr/bin/env python3
"""Inventory public clinical cohort records and exact missing prediction artifacts."""
import argparse
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def sample_rows(path,study):
    out=[]
    for s in ET.parse(path).getroot().findall('BioSample'):
        attrs={a.attrib.get('attribute_name',''):(a.text or '') for a in s.findall('.//Attribute')}
        isolate=attrs.get('isolate',attrs.get('strain',''))
        mutant=bool(re.search(r'm(?:17\.11|24\.4|72\.11|73\.9)$',isolate)) if study=='Meekes_2025' else False
        out.append({'study':study,'biosample_acc':s.attrib['accession'],'biosample_numeric_id':s.attrib['id'],
                    'isolate':isolate,'source_type':'engineered_meropenem_exposed_mutant' if mutant else 'clinical_collection',
                    'country_raw':attrs.get('geo_loc_name',''),'host_raw':attrs.get('host',''),
                    'structured_AST_tables':len(s.findall('.//Table')),
                    'MIC_related_attribute_names':'|'.join(sorted(k for k in attrs if any(w in k.lower() for w in ['mic','merop','imip','suscept','resist'])))})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    folder=args.base/'clinical_public_access'
    rows=sample_rows(folder/'gopel_all_retained_biosamples.xml','Gopel_2025')+sample_rows(folder/'gopel_additional_biosamples.xml','Gopel_2025')+sample_rows(folder/'meekes_biosamples.xml','Meekes_2025')
    if len({r['biosample_acc'] for r in rows})!=len(rows):raise ValueError('Repeated BioSample records')
    g=[r for r in rows if r['study']=='Gopel_2025'];m=[r for r in rows if r['study']=='Meekes_2025']
    assert len(g)==105 and len(m)==19 and sum(r['source_type']=='engineered_meropenem_exposed_mutant' for r in m)==4
    with (args.out/'candidate_clinical_cohort_identifiers.csv').open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    model=json.loads((args.base/'model_artifact_inventory/current_cloud_readiness.json').read_text())
    report={'status':'SOURCE_IDENTIFIER_FEASIBILITY_COMPLETE_NEW_EXTERNAL_PREDICTION_NOT_RUN',
            'Gopel':{'project':'PRJNA1043438','project_linked_BioSamples':len(g),'in_retained_NCBI_build':102,'structured_AST_tables':sum(r['structured_AST_tables'] for r in g),'MIC_related_attributes_found':sum(bool(r['MIC_related_attribute_names']) for r in g),'paper_assay':'Broth microdilution, EUCAST v13; selected carbapenem-resistant collection'},
            'Meekes':{'project':'PRJNA1241873','public_BioSamples':len(m),'clinical_genome_isolates':15,'engineered_mutants_excluded_from_clinical_count':4,'structured_AST_tables':sum(r['structured_AST_tables'] for r in m),'MIC_related_attributes_found':sum(bool(r['MIC_related_attribute_names']) for r in m),'paper_assay':'BMD; paper cites EUCAST 2024; 74 phenotyped but only 15 clinical genomes plus four mutants'},
            'missing_frozen_model_inputs':model['missing'],'model_expected_sha256':model['expected_frozen_model_sha256'],
            'blocking_conditions':['Original primary isolate-level MIC matrix not obtained through the public endpoints tried; BioSample metadata do not supply it.','Six fixed model/schema/training/reproduction inputs are absent from the cloud; only the old summary and E2 probabilities are available.','Training/genome/lineage overlap must be checked before calling any new cohort external.'],
            'limitations':['A genome accession or genotype is not a measured MIC reference.','Do not count engineered mutants as independent clinical isolates or 74 phenotyped isolates as 74 genome-linked validation cases.','Selected resistant or mechanism-enriched collections are challenge cohorts, not representative population validation.','Different BioProjects are not alone proof of independent laboratories or nonoverlapping genomes.']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
