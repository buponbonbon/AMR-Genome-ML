"""Scientific boundary and evidence tests; all generated records are software controls."""
import copy
import unittest

from reference_evidence_auditor import audit_case, parse_mic, intervals_overlap


def case(a="2", b="4", ac="S", bc="I"):
    return {"biosample_acc": "CONTROL_NOT_A_REAL_BIOSAMPLE", "antibiotic": "meropenem", "as_of_date": "2026-11-10",
            "NCBI_observations": [{"MIC_raw": a, "units_raw": "mg/L", "category_raw": ac}],
            "CDC_observations": [{"MIC_raw": b, "units_raw": "µg/mL", "category_raw": bc}], "MIC_history": []}


def history(c, old="2", new="4"):
    return {"biosample_acc": c["biosample_acc"], "antibiotic": c["antibiotic"], "old_value": old, "new_value": new,
            "date_raw": "08/11/22", "source_url": "https://example.invalid/software-control", "artifact_sha256": "a"*64}


class EvidenceTests(unittest.TestCase):
    def test_censoring_endpoint_membership(self):
        for a,b,overlaps in [("<=1","1",True),("<1","1",False),(">8","8",False),(">=8","8",True),("<=0.5","8",False)]:
            with self.subTest(a=a,b=b):
                self.assertEqual(intervals_overlap(parse_mic(a),parse_mic(b)),overlaps)

    def test_equivalent_numeric_format_and_units(self):
        r=audit_case(case("==2.0","2","S","S"))
        self.assertEqual(r["evidence_state"],"SAME_REPORTED_REFERENCE")

    def test_MIC_difference_hidden_by_same_SIR(self):
        r=audit_case(case("1","2","S","S"))
        self.assertFalse(r["category_difference"])
        self.assertTrue(r["MIC_disjointness"])

    def test_I_is_neither_S_nor_R(self):
        self.assertTrue(audit_case(case("2","4","S","I"))["category_difference"])
        self.assertTrue(audit_case(case("4","8","I","R"))["category_difference"])

    def test_ND_is_not_a_susceptibility_category(self):
        r=audit_case(case(ac="ND",bc="R"))
        self.assertIsNone(r["category_difference"])
        self.assertFalse(r["mechanism_resolved"])

    def test_unsupported_unit_abstains(self):
        c=case();c["CDC_observations"][0]["units_raw"]="mg/mL"
        self.assertEqual(audit_case(c)["evidence_state"],"UNSUPPORTED_OR_MISSING_MIC_OR_UNIT")

    def test_missing_pair_abstains(self):
        c=case();c["NCBI_observations"]=[]
        self.assertEqual(audit_case(c)["evidence_state"],"INSUFFICIENT_PAIRED_REFERENCE")

    def test_conflicting_assays_not_cherry_picked(self):
        c=case();c["NCBI_observations"].append({"MIC_raw":"8","units_raw":"mg/L","category_raw":"R"})
        c["MIC_history"]=[history(c)]
        self.assertEqual(audit_case(c)["evidence_state"],"AMBIGUOUS_MULTIPLE_REFERENCE_STATES")

    def test_concordant_duplicate_rows_do_not_multiply_cases(self):
        c=case();c["NCBI_observations"]*=2
        self.assertEqual(audit_case(c)["NCBI_distinct_states"],1)

    def test_documented_update_requires_both_endpoint_values(self):
        c=case();c["MIC_history"]=[history(c)]
        self.assertTrue(audit_case(c)["exact_history_link"])
        c["MIC_history"]=[history(c,old="1")]
        self.assertTrue(audit_case(c)["native_history_only"])
        self.assertFalse(audit_case(c)["exact_history_link"])

    def test_history_for_another_drug_or_sample_not_linked(self):
        for field,value in [("antibiotic","imipenem"),("biosample_acc","DIFFERENT_CONTROL")]:
            c=case();h=history(c);h[field]=value;c["MIC_history"]=[h]
            self.assertFalse(audit_case(c)["exact_history_link"])

    def test_future_or_untraceable_history_not_used(self):
        for field,value in [("date_raw","12/01/27"),("artifact_sha256",""),("source_url","")]:
            c=case();h=history(c);h[field]=value;c["MIC_history"]=[h]
            self.assertFalse(audit_case(c)["exact_history_link"])

    def test_category_change_with_identical_MIC_not_assumed_breakpoint_change(self):
        r=audit_case(case("2","2","S","I"))
        self.assertEqual(r["evidence_state"],"CATEGORY_ONLY_REFERENCE_DIFFERENCE")
        self.assertFalse(r["mechanism_resolved"])

    def test_reordering_observations_does_not_change_decision(self):
        c=case();c["NCBI_observations"].append({"MIC_raw":"8","units_raw":"mg/L","category_raw":"R"})
        d=copy.deepcopy(c);d["NCBI_observations"].reverse()
        self.assertEqual(audit_case(c),audit_case(d))

    def test_no_history_is_not_proof_of_no_update(self):
        r=audit_case(case())
        self.assertFalse(r["exact_history_link"])
        self.assertFalse(r["mechanism_resolved"])
        self.assertEqual(r["clinical_correctness"],"NOT_ADJUDICATED")


if __name__ == "__main__":
    unittest.main()
