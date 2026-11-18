"""Compare recomputed main and MIC results with the archived observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-out", required=True, type=Path)
    parser.add_argument("--mic-out", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    comparisons = []
    tables = [
        (root / "review_reproduced/recomputed_main_metrics.csv",
         args.main_out / "recomputed_main_metrics.csv"),
        (root / "review_reproduced/recomputed_MIC_contrasts.csv",
         args.main_out / "recomputed_MIC_contrasts.csv"),
    ]
    mic = root / "results/rsos_readiness_extension_20261113/MIC_boundary_analysis"
    tables.extend((mic / name, args.mic_out / name)
                  for name in ["per_case_MIC_boundary_audit.csv", "stratum_metrics.csv"])
    for expected, actual in tables:
        original = pd.read_csv(expected, keep_default_na=False)
        regenerated = pd.read_csv(actual, keep_default_na=False)
        assert_frame_equal(original, regenerated, check_exact=False, rtol=1e-12, atol=1e-12)
        comparisons.append({"table": expected.name, "rows": len(original), "status": "PASS"})
    archive_name = "paired_country_bootstrap_metrics.npz"
    with np.load(mic / archive_name, allow_pickle=False) as expected, \
            np.load(args.mic_out / archive_name, allow_pickle=False) as actual:
        if set(expected.files) != set(actual.files):
            raise AssertionError("Bootstrap array names differ")
        for name in expected.files:
            np.testing.assert_allclose(expected[name], actual[name], rtol=1e-12,
                                       atol=1e-12, equal_nan=True)
        arrays = len(expected.files)
    report = json.loads((args.main_out / "verification.json").read_text())
    if report.get("status") != "PASS" or len(report.get("frozen_score_checks", [])) != 5:
        raise AssertionError("The five frozen model score checks did not pass")
    print(json.dumps({"status": "PASS", "table_comparisons": comparisons,
                      "bootstrap_arrays_compared": arrays, "frozen_score_checks": 5}, indent=2))


if __name__ == "__main__":
    main()
