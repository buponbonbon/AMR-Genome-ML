"""Guardrails for joins, frozen references, metric ties, MIC censoring, and collection."""
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import provenance_audit as audit
import collect_provenance_bundle as collector


def csv_file(path, rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.old=[{"target_acc":f"t{i}","biosample":f"b{i}","drug":"meropenem",
                   "label":lab,"project":"FOCAL" if i<2 else "OTHER"}
                  for i,lab in enumerate(["S","R","R","S"])]
        self.current=deepcopy(self.old)
        self.current[0]["label"]="R";self.current[1]["label"]="S"
        self.preds=[{"target_acc":r["target_acc"],"biosample":r["biosample"],"label":r["label"],"probability":p}
                    for r,p in zip(self.old,[".9",".1",".8",".2"])]
        self.cfg={"_base":str(self.root),"focal_project":"FOCAL","target_drug":"meropenem",
                  "expected":{"prediction_n":4,"focal_prediction_n":2,"historical_R":2,"scenario_R":2}}
        for name,rows in [("historical",self.old),("current",self.current),("predictions",self.preds)]:
            self.set_table(name,rows)

    def set_table(self,name,rows):
        p=self.root/(name+".csv")
        csv_file(p,rows)
        self.cfg[name]={"path":p.name,"columns":{k:k for k in rows[0]}}

    def test_reference_scenario_is_same_score_different_label(self):
        before={p.name:audit.digest(p) for p in self.root.glob("*.csv")}
        report,*_=audit.reconcile(self.cfg)
        self.assertEqual(report["historical_E2"]["roc_auc"],.25)
        self.assertEqual(report["partial_source_substitution_POSTHOC"]["roc_auc"],1)
        self.assertFalse(report["clinical_adjudication"])
        self.assertEqual(before,{p.name:audit.digest(p) for p in self.root.glob("*.csv")})

    def test_conflicting_duplicate_is_not_arbitrarily_deduplicated(self):
        conflicting={**self.current[0],"label":"S"}
        self.set_table("current",self.current+[conflicting])
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_identical_projected_duplicates_counted(self):
        self.set_table("current",self.current+[self.current[0]])
        report,*_=audit.reconcile(self.cfg)
        self.assertEqual(report["current_duplicate_projected_rows"],1)

    def test_duplicate_frozen_predictions_block_even_if_identical(self):
        self.set_table("predictions",self.preds+[self.preds[0]])
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_missing_focal_match_blocks_scenario(self):
        self.set_table("current",self.current[1:])
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_current_unmatched_nonfocal_does_not_replace_historical_anchor(self):
        self.set_table("current",self.current[:2])
        report,*_=audit.reconcile(self.cfg)
        self.assertEqual(report["historical_only"],2)
        self.assertEqual(report["partial_source_substitution_POSTHOC"]["roc_auc"],1)

    def test_frozen_label_mismatch_blocks(self):
        self.preds[0]["label"]="R";self.set_table("predictions",self.preds)
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_intermediate_is_not_susceptible(self):
        self.current[0]["label"]="I";self.set_table("current",self.current)
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_accession_join_not_row_order(self):
        self.set_table("current",list(reversed(self.current)))
        report,*_=audit.reconcile(self.cfg)
        self.assertEqual(report["partial_source_substitution_POSTHOC"]["roc_auc"],1)
        self.current[0]["biosample"]="wrong";self.set_table("current",self.current)
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_missing_project_not_treated_as_other(self):
        self.old[0]["project"]="";self.set_table("historical",self.old)
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_expected_cohort_count_guard(self):
        self.cfg["expected"]["prediction_n"]=525
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_no_overwriting_output_or_input(self):
        out=self.root/"new";audit.new_output(out)
        with self.assertRaises(FileExistsError):audit.new_output(out)
        with self.assertRaises(audit.AuditError):audit.new_output(self.root,[self.root/"historical.csv"])

    def test_metric_ties_and_single_class(self):
        m=audit.metrics([0,1,0,1],[.5,.5,.5,.5])
        self.assertEqual(m["roc_auc"],.5);self.assertEqual(m["average_precision"],.5)
        self.assertIsNone(audit.metrics([1,1],[.2,.8])["roc_auc"])
        for p in [float("nan"),float("inf"),1.1,-.1]:
            with self.assertRaises(audit.AuditError):audit.metrics([0,1],[p,.5])

    def test_metrics_match_sklearn_if_available(self):
        try:
            from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,matthews_corrcoef
        except ImportError:
            self.skipTest("Optional sklearn cross-check unavailable; no installation required")
        import random
        rng=random.Random(19)
        for _ in range(30):
            y=[0,1]+[rng.randrange(2) for _ in range(48)]
            p=[rng.choice([0,.1,.5,.8,1]) for _ in y]
            m=audit.metrics(y,p)
            self.assertAlmostEqual(m["roc_auc"],roc_auc_score(y,p))
            self.assertAlmostEqual(m["average_precision"],average_precision_score(y,p))
            self.assertAlmostEqual(m["brier"],brier_score_loss(y,p))
            self.assertAlmostEqual(m["mcc"],matthews_corrcoef(y,[v>=.5 for v in p]))

    def test_mic_boundary_and_censoring(self):
        # Artificial breakpoints for testing only, NOT clinical EUCAST/CLSI values.
        row={"mic_value":"2","mic_sign":"=","unit":"u","method":"MIC","standard":"TEST",
             "version":"v1","organism":"test organism","drug":"test drug","context":"test context"}
        bp={**row,"s_max":"2","r_above":"8","source_url":"https://example.invalid/TEST_ONLY",
            "source_sha256":"a"*64,"convention":"S_LE_R_GT"}
        cases=[("=","2","S"),("==","2","S"),("<=","2","S"),(">","2","I|R"),(">=","2","S|I|R"),
               ("=","8","I"),(">","8","R"),(">=","8","I|R"),("<","8","S|I"),
               ("=","9","R"),("<","2","S"),("<=","8","S|I")]
        for sign,value,want in cases:
            with self.subTest(sign=sign,value=value):
                self.assertEqual(audit.mic_interpret({**row,"mic_sign":sign,"mic_value":value},[bp])["possible_categories"],want)
        self.assertEqual(audit.mic_interpret(row,[])["status"],"NO_UNIQUE_DOCUMENTED_BREAKPOINT")
        self.assertEqual(audit.mic_interpret({**row,"version":""},[bp])["status"],"INCOMPLETE_METADATA")
        self.assertEqual(audit.mic_interpret({**row,"mic_value":"NaN"},[bp])["status"],"INVALID_MIC")
        self.assertEqual(audit.mic_interpret(row,[{**bp,"convention":"R_GE"}])["status"],"UNSUPPORTED_BREAKPOINT_CONVENTION")

    def test_xml_schema_only_then_explicit_extraction(self):
        p=self.root/"sample.xml"
        p.write_text('<BioSampleSet><BioSample accession="B1"><AST drug="meropenem" phenotype="R" value="4"/></BioSample></BioSampleSet>')
        cfg={"_base":str(self.root),"xml":{"path":p.name}}
        report,rows=audit.xml_extract(cfg)
        self.assertEqual(report["status"],"SCHEMA_ONLY_NOT_RECONCILED");self.assertEqual(rows,[])
        cfg["xml"].update({"record_path":"AST","columns":{"drug":"@drug","label":"@phenotype","mic_value":"@value"}})
        _,rows=audit.xml_extract(cfg)
        self.assertEqual(rows[0]["biosample"],"B1");self.assertEqual(rows[0]["mic_value"],"4")
        p.write_text('<!DOCTYPE x [<!ENTITY x "bad">]><BioSampleSet/>')
        with self.assertRaises(audit.AuditError):audit.xml_extract(cfg)

    def test_static_bibliography_is_not_compilation(self):
        (self.root/"main.tex").write_text(r'\cite{A,B} S1--S5 % \cite{IGNORED}')
        (self.root/"supp.tex").write_text(r'\citep[see]{A,C}')
        (self.root/"refs.bib").write_text('@article{A,title={A}}\n@article{A,title={duplicate}}\n@book{B,title={B}}')
        cfg={"_base":str(self.root),"main_tex":{"path":"main.tex"},"supplementary_tex":{"path":"supp.tex"},"bibliography":{"path":"refs.bib"}}
        report=audit.manuscript(cfg)
        self.assertEqual(report["missing_keys"],["C"]);self.assertEqual(report["duplicate_keys"],["A"])
        self.assertEqual(report["compilation"],"NOT_RUN")

    def test_cli_writes_failed_report_not_success_on_hash_mismatch(self):
        config=self.root/"cfg.json"
        self.cfg["historical"]["sha256"]="0"*64
        config.write_text(json.dumps(self.cfg))
        out=self.root/"cli"
        self.assertEqual(audit.main(["reconcile","--config",str(config),"--out",str(out)]),1)
        self.assertFalse((out/"report.json").exists())

    def test_collector_only_selected_files_and_keeps_sources(self):
        (self.root/"file19_temporal_summary.json").write_text('{}')
        (self.root/"secret.env").write_text('not for collection')
        (self.root/"genome.fna").write_text('>x\nACGT')
        out=self.root/"bundle.zip"
        before=audit.digest(self.root/"file19_temporal_summary.json")
        m=collector.make_bundle([("project",self.root)],out)
        with zipfile.ZipFile(out) as z:
            self.assertIn("MANIFEST.json",z.namelist())
            self.assertNotIn("project/secret.env",z.namelist())
            self.assertNotIn("project/genome.fna",z.namelist())
            for r in m["files"]:
                self.assertEqual(hashlib.sha256(z.read(r["archive_path"])).hexdigest(),r["sha256"])
        self.assertEqual(before,audit.digest(self.root/"file19_temporal_summary.json"))
        with self.assertRaises(FileExistsError):collector.make_bundle([("project",self.root)],out)

    def test_cli_success_with_unavailable_unrelated_sections(self):
        self.cfg["xml"]={"path":"not_here.xml"}
        self.cfg["bibliography"]={"path":"not_here.bib"}
        config=self.root/"cfg.json";config.write_text(json.dumps(self.cfg))
        out=self.root/"success"
        self.assertEqual(audit.main(["reconcile","--config",str(config),"--out",str(out)]),0)
        report=json.loads((out/"report.json").read_text())
        self.assertFalse(report["clinical_adjudication"])
        self.assertNotIn("xml",report["inputs"])
        self.assertTrue((out/"E2_POSTHOC_reference_scenario.csv").exists())

    def test_xml_missing_focal_current_records_is_review_needed(self):
        self.set_table("biosample_table",[{"biosample":"b2","drug":"meropenem","label":"R"}])
        report,*_=audit.reconcile(self.cfg)
        self.assertTrue(report["xml_review_required"])
        self.assertEqual(report["focal_current_vs_biosample"]["missing"],2)

    def test_membership_conflict_blocks(self):
        self.set_table("membership",[{"biosample":"b0","project":"WRONG"}])
        with self.assertRaises(audit.AuditError):audit.reconcile(self.cfg)

    def test_search_directories_do_not_collect_other_manuscripts(self):
        other=self.root/"elsewhere";other.mkdir()
        (other/"main.tex").write_text("another project")
        (other/"file19_temporal_175_three_drug_crosswalk.csv").write_text("header\n")
        project=self.root/"project";project.mkdir()
        selected,_=collector.collect([("project",project)],search_dirs=[other])
        self.assertEqual(len(selected),1)
        self.assertEqual(next(iter(selected)).name,"file19_temporal_175_three_drug_crosswalk.csv")

    def test_mic_completeness_does_not_count_duplicates_twice(self):
        rows=[{"biosample":"B","drug":"meropenem","mic_value":"2"}]*2
        self.set_table("mic",rows)
        report,_=audit.mic_audit(self.cfg)
        self.assertEqual(report["rows"],1)
        self.assertEqual(report["raw_rows"],2)
        self.assertEqual(report["nonempty_fields"]["mic_value"],1)

    def test_collector_finds_nested_renamed_stage_artifacts(self):
        folder=self.root/"results"/"FILE19_temporal_run";folder.mkdir(parents=True)
        (folder/"summary.json").write_text('{}')
        (folder/"crosswalk.csv").write_text('x\n')
        (self.root/"file19_bigquery_current_175_3drugs.csv").write_text('x\n')
        out=self.root/"v2.zip"
        m=collector.make_bundle([("project",self.root)],out)
        collected={r["archive_path"] for r in m["files"]}
        self.assertIn("project/results/FILE19_temporal_run/summary.json",collected)
        self.assertIn("project/file19_bigquery_current_175_3drugs.csv",collected)
        self.assertIn("current_source",m["missing_required"])
        self.assertTrue(m["research_file_index"])


if __name__=="__main__":
    unittest.main()
