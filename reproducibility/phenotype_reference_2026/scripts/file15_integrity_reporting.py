#!/usr/bin/env python3
"""
FILE15 — FINAL INTEGRITY + MANUSCRIPT REPORTING
================================================

Purpose
-------
FILE15 is reporting/integrity only. It MUST NOT fit, tune, recalibrate, reselect,
or otherwise modify the frozen FILE13 model or the locked E3 predictions.

It verifies the completed FILE14 blind external validation, reconstructs a
publication-ready metrics table, inventories final artifacts with SHA-256
hashes, records the final bootstrap grouping metadata, and freezes a final
analysis/reporting lock.

Expected upstream state
-----------------------
- FILE14 status: PASS_E3_BLIND_EXTERNAL_VALIDATION
- Final E3 cohort frozen before scoring
- Frozen FILE13 model/schema/protocol
- Blind scores frozen before metrics
- No E3 tuning/reselection/recalibration
- Decision threshold unchanged
- Final cluster bootstrap uses resolved NCBI SNP/ERD groups as clusters and
  treats unresolved group IDs as singleton resampling units

Outputs
-------
results/file15_reporting/
  file15_integrity_checks.csv
  file15_artifact_manifest.csv
  file15_primary_metrics_manuscript.csv
  file15_sensitivity_metrics_manuscript.csv   (when available)
  FILE15_MANUSCRIPT_REPORT.md

checkpoints/file15_integrity_reporting/
  file15_final_summary.json
  file15_final_lock.json
  FILE15_COMPLETE.flag

results/FILE15_FINAL_RUN.log

FILE15 never writes into FILE13/FILE14 scientific output files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "1.0.0"
DEFAULT_ROOT = Path("/mnt/c/Users/ASUS/Desktop/AMR-Genome-ML")
FINAL_FILE14_STATUS = "PASS_E3_BLIND_EXTERNAL_VALIDATION"
FINAL_FILE15_STATUS = "PASS_FILE15_INTEGRITY_REPORTING"
PRIMARY_REP = "harmonized_rebuilt_primary"
SENS_REP = "harmonized_original668_projection"
CORE_METRICS = ["roc_auc", "average_precision", "balanced_accuracy", "mcc", "brier"]
FLOAT_TOL = 1e-12

LOG_FILE: Path | None = None


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def shafile(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def shatext(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_sha(obj: Any) -> str:
    return shatext(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        x = json.load(f)
    if not isinstance(x, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return x


def resolve_recorded_path(root: Path, value: Any) -> Path:
    if value is None or str(value).strip() == "":
        raise RuntimeError("Recorded path is empty")
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def isclose(a: Any, b: Any, tol: float = FLOAT_TOL) -> bool:
    try:
        aa = float(a)
        bb = float(b)
    except Exception:
        return a == b
    if math.isnan(aa) and math.isnan(bb):
        return True
    return abs(aa - bb) <= tol


def phenotype_to_y(x: Any) -> int:
    s = str(x).strip().lower()
    if s in {"r", "resistant", "1", "true"}:
        return 1
    if s in {"s", "susceptible", "0", "false"}:
        return 0
    raise ValueError(f"Unrecognized phenotype: {x!r}")


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: str, severity: str = "ERROR") -> None:
        self.rows.append({
            "check": name,
            "status": "PASS" if passed else ("WARN" if severity == "WARN" else "FAIL"),
            "severity": severity,
            "detail": detail,
        })

    def require(self, name: str, passed: bool, detail: str) -> None:
        self.add(name, passed, detail, "ERROR")

    def warn(self, name: str, passed: bool, detail: str) -> None:
        self.add(name, passed, detail, "WARN")

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == "FAIL"]

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == "WARN"]

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "status", "severity", "detail"])


def build_artifact_manifest(root: Path, entries: list[tuple[str, Path, bool]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role, p, required in entries:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "role": role,
            "required": bool(required),
            "exists": p.is_file(),
            "path": rel(root, p),
            "bytes": int(p.stat().st_size) if p.is_file() else None,
            "sha256": shafile(p) if p.is_file() else "",
        })
    return pd.DataFrame(rows)


def metric_label(k: str) -> str:
    return {
        "roc_auc": "AUROC",
        "average_precision": "Average precision",
        "balanced_accuracy": "Balanced accuracy",
        "mcc": "MCC",
        "brier": "Brier score",
    }.get(k, k)


def fmt4(x: Any) -> str:
    return f"{float(x):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="FILE15 final integrity + manuscript reporting only")
    ap.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        assert phenotype_to_y("Resistant") == 1
        assert phenotype_to_y("Susceptible") == 0
        assert isclose(1.0, 1.0 + 5e-13)
        print("FILE15 self-test: PASS")
        return 0

    root = a.project_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Project root not found: {root}")

    cp14 = root / "checkpoints/file14_final_e3_validation"
    cp15 = root / "checkpoints/file15_integrity_reporting"
    out = root / "results/file15_reporting"
    cp15.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    global LOG_FILE
    LOG_FILE = root / "results/FILE15_FINAL_RUN.log"
    LOG_FILE.write_text("", encoding="utf-8")

    failure_json = cp15 / "file15_failure.json"
    lock_path = cp15 / "file15_final_lock.json"
    summary15_path = cp15 / "file15_final_summary.json"
    complete_path = cp15 / "FILE15_COMPLETE.flag"

    try:
        log("=" * 112)
        log("FILE15 — FINAL INTEGRITY + MANUSCRIPT REPORTING")
        log("=" * 112)
        log(f"Version      : {VERSION}")
        log(f"Project root : {root}")
        log("Mode         : REPORTING/INTEGRITY ONLY — NO FITTING/TUNING/RECALIBRATION/RESELECTION")

        checks = Checks()

        flag14_path = cp14 / "E3_VALIDATION_COMPLETE.flag"
        summary14_path = cp14 / "file14_final_summary.json"
        checks.require("FILE14 completion flag exists", flag14_path.is_file(), str(flag14_path))
        checks.require("FILE14 final summary exists", summary14_path.is_file(), str(summary14_path))
        if checks.failures:
            raise RuntimeError("Required FILE14 final artifacts are missing")

        flag14 = load_json(flag14_path)
        summary14 = load_json(summary14_path)

        script15_path = Path(__file__).resolve()
        script14_path = root / "scripts/file14_independent_E3_blind_validation.py"
        script15_sha = shafile(script15_path)
        script14_sha = shafile(script14_path) if script14_path.is_file() else ""

        source_anchor_obj = {
            "file14_completion_flag_sha256": shafile(flag14_path),
            "file14_final_summary_sha256": shafile(summary14_path),
            "file14_script_sha256": script14_sha,
            "file15_script_sha256": script15_sha,
        }
        source_anchor_sha = canonical_sha(source_anchor_obj)

        # If FILE15 is already frozen, never silently refresh it.
        if lock_path.is_file():
            old_lock = load_json(lock_path)
            if old_lock.get("source_anchor_sha256") != source_anchor_sha:
                raise RuntimeError(
                    "Existing FILE15 lock was created from different upstream/script bytes. "
                    "Archive checkpoints/file15_integrity_reporting and results/file15_reporting "
                    "before intentionally rebuilding FILE15."
                )
            locked_outputs = old_lock.get("locked_outputs", {})
            bad: list[str] = []
            for pstr, expected in locked_outputs.items():
                p = root / pstr
                if not p.is_file() or shafile(p) != expected:
                    bad.append(pstr)
            if bad:
                raise RuntimeError("Existing FILE15 lock output mismatch: " + ", ".join(bad))
            log(f"[PASS] Existing FILE15 final lock reproduced | source_anchor={source_anchor_sha}")
            if complete_path.is_file():
                log(f"FILE15 STATUS : {FINAL_FILE15_STATUS} (REUSED)")
                return 0
            raise RuntimeError("FILE15 final lock exists but completion flag is missing")

        log("Stage 1/6 — verify FILE14 completion and frozen guardrails")
        checks.require(
            "FILE14 status is final PASS",
            summary14.get("status") == FINAL_FILE14_STATUS and flag14.get("status") == FINAL_FILE14_STATUS,
            f"summary={summary14.get('status')} flag={flag14.get('status')}",
        )
        checks.require(
            "FILE14 summary SHA matches completion flag",
            flag14.get("file14_final_summary_sha256") == shafile(summary14_path),
            f"flag={flag14.get('file14_final_summary_sha256')} actual={shafile(summary14_path)}",
        )
        checks.require(
            "E3 cohort SHA consistent",
            summary14.get("E3", {}).get("cohort_sha256") == flag14.get("e3_cohort_sha256"),
            f"summary={summary14.get('E3', {}).get('cohort_sha256')} flag={flag14.get('e3_cohort_sha256')}",
        )
        checks.require(
            "FILE13 primary model SHA consistent",
            summary14.get("FILE13", {}).get("primary_model_sha256") == flag14.get("file13_primary_model_sha256"),
            f"model={summary14.get('FILE13', {}).get('primary_model_sha256')}",
        )

        guard = summary14.get("guardrails", {})
        expected_guard = {
            "historical_development_qc_rewritten": False,
            "E3_used_for_tuning": False,
            "E3_used_for_reselection": False,
            "E3_used_for_recalibration": False,
            "decision_threshold_changed": False,
            "blind_scores_frozen_before_metrics": True,
        }
        for k, v in expected_guard.items():
            checks.require(f"Guardrail {k}", guard.get(k) is v, f"expected={v} actual={guard.get(k)}")

        score_lock = summary14.get("blind_score_lock", {})
        checks.require(
            "Blind score lock status",
            score_lock.get("status") == "BLIND_SCORES_FROZEN_BEFORE_METRICS",
            str(score_lock.get("status")),
        )
        for k in ["no_tuning", "no_recalibration", "no_model_reselection"]:
            checks.require(f"Blind score guard {k}", score_lock.get(k) is True, f"actual={score_lock.get(k)}")
        threshold = float(score_lock.get("threshold"))
        checks.require("Frozen decision threshold is 0.5", isclose(threshold, 0.5), f"threshold={threshold}")

        log("Stage 2/6 — verify frozen E3 cohort and blind predictions")
        final_manifest = resolve_recorded_path(root, summary14["E3"]["final_manifest"])
        checks.require("Final E3 manifest exists", final_manifest.is_file(), str(final_manifest))
        if not final_manifest.is_file():
            raise RuntimeError("Final E3 manifest missing")
        checks.require(
            "Final E3 manifest SHA matches FILE14",
            shafile(final_manifest) == summary14["E3"]["final_manifest_sha256"],
            f"expected={summary14['E3']['final_manifest_sha256']} actual={shafile(final_manifest)}",
        )

        e3 = pd.read_csv(final_manifest, dtype={"Genome ID": str}, keep_default_na=False, encoding="utf-8-sig")
        required_cols = {"Genome ID", "Phenotype", "NCBI SNP/ERD Group"}
        checks.require("E3 required columns present", required_cols <= set(e3.columns), f"columns={list(e3.columns)}")
        checks.require("E3 Genome ID unique", e3["Genome ID"].nunique() == len(e3), f"n={len(e3)} unique={e3['Genome ID'].nunique()}")
        y = e3["Phenotype"].map(phenotype_to_y).to_numpy(int)
        n, nr, ns = len(e3), int(y.sum()), int((y == 0).sum())
        checks.require("E3 n matches FILE14", n == int(summary14["E3"]["n"]), f"n={n}")
        checks.require("E3 R matches FILE14", nr == int(summary14["E3"]["R"]), f"R={nr}")
        checks.require("E3 S matches FILE14", ns == int(summary14["E3"]["S"]), f"S={ns}")

        raw_group = e3["NCBI SNP/ERD Group"].astype(str).str.strip()
        unresolved = raw_group.eq("") | raw_group.str.upper().isin(["UNRESOLVED", "NA", "N/A", "NAN", "NONE"])
        resolved = raw_group[~unresolved]
        resolved_groups = int(resolved.nunique())
        unresolved_n = int(unresolved.sum())
        effective_bootstrap_units = resolved_groups + unresolved_n
        resolved_max_cluster_size = int(resolved.value_counts().max()) if len(resolved) else 0
        checks.require(
            "Bootstrap grouping has >1 effective unit",
            effective_bootstrap_units > 1,
            f"effective_units={effective_bootstrap_units}",
        )
        checks.warn(
            "Unresolved lineage/group metadata documented",
            unresolved_n == 0,
            f"unresolved={unresolved_n}/{n}; FILE14 final rule treats unresolved IDs as singleton resampling units",
        )

        blind_primary = resolve_recorded_path(root, score_lock["blind_primary_scores"])
        checks.require("Blind primary score file exists", blind_primary.is_file(), str(blind_primary))
        if not blind_primary.is_file():
            raise RuntimeError("Blind primary score file missing")
        primary_score_sha = shafile(blind_primary)
        checks.require(
            "Blind primary SHA matches final score lock",
            primary_score_sha == score_lock.get("blind_primary_scores_sha256"),
            f"expected={score_lock.get('blind_primary_scores_sha256')} actual={primary_score_sha}",
        )
        checks.require(
            "Blind primary SHA matches FILE14 completion flag",
            primary_score_sha == flag14.get("blind_primary_scores_sha256"),
            f"flag={flag14.get('blind_primary_scores_sha256')} actual={primary_score_sha}",
        )

        blind = pd.read_csv(blind_primary, dtype={"Genome ID": str}, keep_default_na=False, encoding="utf-8-sig")
        checks.require("Blind score row count matches E3", len(blind) == n, f"scores={len(blind)} E3={n}")
        checks.require(
            "Blind score Genome ID/order matches E3",
            blind["Genome ID"].tolist() == e3["Genome ID"].tolist(),
            "exact ordered comparison",
        )
        p = pd.to_numeric(blind["probability_R"], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(blind["predicted_R"], errors="coerce").to_numpy(float)
        checks.require("Blind probabilities finite", bool(np.isfinite(p).all()), "all probability_R finite")
        checks.require("Blind probabilities in [0,1]", bool(((p >= 0) & (p <= 1)).all()), f"min={np.nanmin(p):.6g} max={np.nanmax(p):.6g}")
        checks.require(
            "predicted_R consistent with frozen threshold",
            bool(np.array_equal(pred.astype(int), (p >= threshold).astype(int))),
            f"threshold={threshold}",
        )

        log("Stage 3/6 — verify point metrics and corrected bootstrap summary")
        ed = root / "data/evaluation/file14"
        point_path = ed / "file14_E3_point_metrics.csv"
        boot_summary_path = ed / "file14_E3_cluster_bootstrap_summary.csv"
        checks.require("FILE14 point metrics exists", point_path.is_file(), str(point_path))
        checks.require("FILE14 bootstrap summary exists", boot_summary_path.is_file(), str(boot_summary_path))
        if not point_path.is_file() or not boot_summary_path.is_file():
            raise RuntimeError("Required FILE14 metric outputs missing")

        points = pd.read_csv(point_path, encoding="utf-8-sig")
        boots = pd.read_csv(boot_summary_path, encoding="utf-8-sig")
        prow = points[points["representation"].astype(str) == PRIMARY_REP]
        checks.require("Exactly one primary point-metric row", len(prow) == 1, f"rows={len(prow)}")
        if len(prow) != 1:
            raise RuntimeError("Primary point-metric row missing/duplicated")
        prow = prow.iloc[0]

        summary_point = summary14.get("primary_metrics", {})
        for k in CORE_METRICS:
            checks.require(
                f"Primary point metric {k} matches FILE14 summary",
                isclose(prow[k], summary_point[k]),
                f"csv={prow[k]} summary={summary_point[k]}",
            )

        bprimary = boots[boots["representation"].astype(str) == PRIMARY_REP].copy()
        checks.require("Primary bootstrap has five core metrics", set(bprimary["metric"]) == set(CORE_METRICS), f"metrics={list(bprimary['metric'])}")
        checks.require("Primary bootstrap valid replicates all 2000", bool((bprimary["valid_replicates"].astype(int) == 2000).all()), f"replicates={sorted(bprimary['valid_replicates'].astype(int).unique())}")

        summary_boot = {str(x["metric"]): x for x in summary14.get("primary_bootstrap_summary", [])}
        for _, r in bprimary.iterrows():
            k = str(r["metric"])
            checks.require(
                f"Bootstrap point estimate {k} matches point metrics",
                isclose(r["point_estimate"], prow[k]),
                f"bootstrap={r['point_estimate']} point={prow[k]}",
            )
            if k in summary_boot:
                for fld in ["point_estimate", "bootstrap_mean", "ci95_low", "ci95_high"]:
                    checks.require(
                        f"Bootstrap {k} {fld} matches FILE14 summary",
                        isclose(r[fld], summary_boot[k][fld]),
                        f"csv={r[fld]} summary={summary_boot[k][fld]}",
                    )
                checks.require(
                    f"Bootstrap {k} replicate count matches FILE14 summary",
                    int(r["valid_replicates"]) == int(summary_boot[k]["valid_replicates"]),
                    f"csv={r['valid_replicates']} summary={summary_boot[k]['valid_replicates']}",
                )
            lo, hi, pt, bm = map(float, [r["ci95_low"], r["ci95_high"], r["point_estimate"], r["bootstrap_mean"]])
            checks.require(f"Bootstrap {k} CI ordered", lo <= hi, f"low={lo} high={hi}")
            checks.warn(f"Bootstrap {k} point inside percentile CI", lo <= pt <= hi, f"point={pt} CI=[{lo},{hi}]")
            checks.warn(f"Bootstrap {k} mean close to point", abs(bm - pt) <= 0.05, f"point={pt} mean={bm}")

        # Manuscript-ready primary table.
        primary_table = bprimary[["metric", "point_estimate", "bootstrap_mean", "ci95_low", "ci95_high", "valid_replicates"]].copy()
        primary_table.insert(1, "metric_label", primary_table["metric"].map(metric_label))
        primary_table["estimate_95CI"] = primary_table.apply(
            lambda r: f"{float(r['point_estimate']):.3f} ({float(r['ci95_low']):.3f}–{float(r['ci95_high']):.3f})", axis=1
        )
        primary_table = primary_table.sort_values("metric", key=lambda s: s.map({k: i for i, k in enumerate(CORE_METRICS)}))

        sens_table = pd.DataFrame()
        bsens = boots[boots["representation"].astype(str) == SENS_REP].copy()
        if len(bsens):
            sens_table = bsens[["metric", "point_estimate", "bootstrap_mean", "ci95_low", "ci95_high", "valid_replicates"]].copy()
            sens_table.insert(1, "metric_label", sens_table["metric"].map(metric_label))
            sens_table["estimate_95CI"] = sens_table.apply(
                lambda r: f"{float(r['point_estimate']):.3f} ({float(r['ci95_low']):.3f}–{float(r['ci95_high']):.3f})", axis=1
            )
            sens_table = sens_table.sort_values("metric", key=lambda s: s.map({k: i for i, k in enumerate(CORE_METRICS)}))
            checks.require("Sensitivity bootstrap valid replicates all 2000", bool((sens_table["valid_replicates"].astype(int) == 2000).all()), f"rows={len(sens_table)}")

        log("Stage 4/6 — verify supporting audit artifacts and build provenance manifest")
        technical = summary14.get("technical_qc", {})
        envelope = resolve_recorded_path(root, technical["envelope"]) if technical.get("envelope") else root / "__missing__"
        technical_audit = root / "data/external_validation/e3/file14_final/file14_E3_technical_qc_audit.csv"
        preassembly = resolve_recorded_path(root, summary14["preassembly"]["manifest"])
        diagnostics_path = ed / "file14_E3_diagnostics_summary.json"
        subgroup_path = ed / "file14_E3_mlst_subgroup_metrics.csv"
        errors_path = ed / "file14_E3_error_diagnostics.csv"
        shift_path = ed / "file14_E3_feature_prevalence_shift.csv"
        boot_primary_raw = ed / "file14_E3_primary_cluster_bootstrap.csv.gz"
        boot_sens_raw = ed / "file14_E3_sensitivity_cluster_bootstrap.csv.gz"

        if envelope.is_file():
            checks.require("Technical-QC envelope SHA matches FILE14", shafile(envelope) == technical.get("envelope_sha256"), f"actual={shafile(envelope)}")
        else:
            checks.require("Technical-QC envelope exists", False, str(envelope))
        checks.require("Technical-QC audit exists", technical_audit.is_file(), str(technical_audit))
        if technical_audit.is_file():
            checks.require("Technical-QC audit SHA matches FILE14", shafile(technical_audit) == technical.get("audit_sha256"), f"actual={shafile(technical_audit)}")
        checks.require("Preassembly manifest exists", preassembly.is_file(), str(preassembly))
        if preassembly.is_file():
            checks.require("Preassembly manifest SHA matches FILE14", shafile(preassembly) == summary14["preassembly"]["manifest_sha256"], f"actual={shafile(preassembly)}")

        diagnostics = load_json(diagnostics_path) if diagnostics_path.is_file() else summary14.get("diagnostics", {})
        checks.require(
            "Diagnostics counts match FILE14 summary",
            int(diagnostics.get("errors", -1)) == int(summary14.get("diagnostics", {}).get("errors", -2))
            and int(diagnostics.get("high_confidence_wrong", -1)) == int(summary14.get("diagnostics", {}).get("high_confidence_wrong", -2)),
            f"diagnostics={diagnostics}",
        )

        artifact_entries: list[tuple[str, Path, bool]] = [
            ("FILE14 completion flag", flag14_path, True),
            ("FILE14 final summary", summary14_path, True),
            ("FILE14 final E3 manifest", final_manifest, True),
            ("FILE14 blind primary scores", blind_primary, True),
            ("FILE14 point metrics", point_path, True),
            ("FILE14 corrected cluster-bootstrap summary", boot_summary_path, True),
            ("FILE14 primary raw bootstrap", boot_primary_raw, True),
            ("FILE14 sensitivity raw bootstrap", boot_sens_raw, False),
            ("FILE14 technical-QC envelope", envelope, True),
            ("FILE14 technical-QC audit", technical_audit, True),
            ("FILE14 preassembly manifest", preassembly, True),
            ("FILE14 diagnostics summary", diagnostics_path, True),
            ("FILE14 error diagnostics", errors_path, False),
            ("FILE14 MLST subgroup metrics", subgroup_path, False),
            ("FILE14 feature prevalence shift", shift_path, False),
            ("FILE14 production script", script14_path, True),
            ("FILE15 reporting script", script15_path, True),
        ]
        if score_lock.get("blind_sensitivity_scores"):
            sens_score_path = resolve_recorded_path(root, score_lock["blind_sensitivity_scores"])
            artifact_entries.append(("FILE14 blind sensitivity scores", sens_score_path, False))
            if sens_score_path.is_file() and score_lock.get("blind_sensitivity_scores_sha256"):
                checks.require(
                    "Blind sensitivity SHA matches final score lock",
                    shafile(sens_score_path) == score_lock["blind_sensitivity_scores_sha256"],
                    f"actual={shafile(sens_score_path)}",
                )

        old_boot_dir = ed / "pre_bootstrap_cluster_fix"
        if old_boot_dir.is_dir():
            for pth in sorted(old_boot_dir.iterdir()):
                if pth.is_file():
                    artifact_entries.append(("Archived pre-bootstrap-fix audit", pth, False))

        artifact_manifest = build_artifact_manifest(root, artifact_entries)
        required_missing = artifact_manifest[(artifact_manifest["required"] == True) & (artifact_manifest["exists"] == False)]
        checks.require("All required provenance artifacts exist", len(required_missing) == 0, f"missing={required_missing['path'].tolist()}")

        checks_path = out / "file15_integrity_checks.csv"
        manifest_path = out / "file15_artifact_manifest.csv"
        primary_table_path = out / "file15_primary_metrics_manuscript.csv"
        sensitivity_table_path = out / "file15_sensitivity_metrics_manuscript.csv"
        report_path = out / "FILE15_MANUSCRIPT_REPORT.md"

        # Write checks before failing so any issue is auditable.
        atomic_csv(checks_path, checks.dataframe())
        if checks.failures:
            raise RuntimeError(f"FILE15 integrity checks failed: {len(checks.failures)}")

        atomic_csv(manifest_path, artifact_manifest)
        atomic_csv(primary_table_path, primary_table)
        if len(sens_table):
            atomic_csv(sensitivity_table_path, sens_table)

        log("Stage 5/6 — generate manuscript-ready reporting package")
        pm = {str(r["metric"]): r for _, r in bprimary.iterrows()}
        prevalence = nr / n if n else float("nan")
        errs = int(summary14.get("diagnostics", {}).get("errors", 0))
        hcw = int(summary14.get("diagnostics", {}).get("high_confidence_wrong", 0))

        md: list[str] = []
        md.append("# FILE15 — Final integrity and manuscript reporting")
        md.append("")
        md.append(f"**Status:** {FINAL_FILE15_STATUS}")
        md.append("")
        md.append("## Locked analysis state")
        md.append("")
        md.append(
            f"The final lineage-disjoint E3 validation cohort contained **{n} genomes** "
            f"(**{nr} resistant, {ns} susceptible; resistant prevalence {prevalence:.1%}**). "
            "The E3 cohort was frozen before model scoring. The frozen FILE13 model/schema/protocol "
            "were carried into FILE14 without E3-based model tuning, model reselection, recalibration, "
            "or threshold modification. Blind prediction scores were frozen before confirmatory metrics."
        )
        md.append("")
        md.append("## Primary external-validation performance")
        md.append("")
        md.append("| Metric | Point estimate | 95% bootstrap CI | Bootstrap mean | Replicates |")
        md.append("|---|---:|---:|---:|---:|")
        for k in CORE_METRICS:
            r = pm[k]
            md.append(
                f"| {metric_label(k)} | {float(r['point_estimate']):.3f} | "
                f"{float(r['ci95_low']):.3f}–{float(r['ci95_high']):.3f} | "
                f"{float(r['bootstrap_mean']):.3f} | {int(r['valid_replicates'])} |"
            )
        md.append("")
        md.append(
            f"Primary E3 performance was AUROC **{float(pm['roc_auc']['point_estimate']):.3f}** "
            f"(95% CI {float(pm['roc_auc']['ci95_low']):.3f}–{float(pm['roc_auc']['ci95_high']):.3f}), "
            f"average precision **{float(pm['average_precision']['point_estimate']):.3f}** "
            f"({float(pm['average_precision']['ci95_low']):.3f}–{float(pm['average_precision']['ci95_high']):.3f}), "
            f"balanced accuracy **{float(pm['balanced_accuracy']['point_estimate']):.3f}**, "
            f"MCC **{float(pm['mcc']['point_estimate']):.3f}**, and Brier score **{float(pm['brier']['point_estimate']):.3f}**."
        )
        md.append("")
        md.append("## Bootstrap uncertainty specification")
        md.append("")
        md.append(
            f"Confidence intervals were generated from **2,000 cluster-bootstrap replicates**. "
            f"Resolved NCBI SNP/ERD groups were retained as cluster resampling units; genomes lacking "
            f"a resolved group identifier were treated as singleton resampling units. In the final E3 "
            f"cohort, {unresolved_n}/{n} genomes lacked a resolved group identifier, yielding "
            f"{effective_bootstrap_units} effective bootstrap units; the largest resolved cluster "
            f"contained {resolved_max_cluster_size} genome(s)."
        )
        md.append("")
        if len(sens_table):
            sm = {str(r["metric"]): r for _, r in bsens.iterrows()}
            md.append("## Frozen 668-feature sensitivity representation")
            md.append("")
            md.append(
                f"The frozen 668-feature sensitivity projection produced AUROC "
                f"**{float(sm['roc_auc']['point_estimate']):.3f}** and average precision "
                f"**{float(sm['average_precision']['point_estimate']):.3f}**, closely matching the "
                "primary harmonized representation. This comparison is descriptive sensitivity analysis; "
                "it was not used for E3-based model selection."
            )
            md.append("")
        md.append("## Error and transportability diagnostics")
        md.append("")
        md.append(
            f"FILE14 recorded **{errs}** prediction errors, including **{hcw}** high-confidence wrong "
            "predictions. These diagnostics were descriptive only and were not used to modify, reselect, "
            "or recalibrate the frozen model."
        )
        md.append("")
        md.append("## Reporting limitations to retain in the manuscript")
        md.append("")
        md.append(
            f"A substantial fraction of E3 genomes ({unresolved_n}/{n}) lacked resolved NCBI SNP/ERD "
            "group identifiers. Treating these observations as singleton bootstrap units avoids falsely "
            "collapsing all missing group labels into one pseudo-cluster, but unknown relatedness among "
            "unresolved genomes cannot be fully represented by the available metadata."
        )
        md.append("")
        md.append("## Analysis freeze")
        md.append("")
        md.append(
            "FILE15 is the terminal analysis/integrity stage. No further model fitting, threshold tuning, "
            "feature reselection, recalibration, or E3-based optimization should be performed for the "
            "reported study unless a separately documented audit identifies a genuine implementation error. "
            "Subsequent work is manuscript/figure/supplement/repository preparation only."
        )
        md.append("")
        md.append("## Provenance anchors")
        md.append("")
        md.append(f"- E3 cohort SHA-256: `{summary14['E3']['cohort_sha256']}`")
        md.append(f"- FILE13 primary model SHA-256: `{summary14['FILE13']['primary_model_sha256']}`")
        md.append(f"- FILE13 primary schema SHA-256: `{summary14['FILE13']['primary_schema_sha256']}`")
        md.append(f"- FILE14 annotation protocol SHA-256: `{summary14.get('annotation_protocol_sha256', '')}`")
        md.append(f"- FILE14 blind primary score SHA-256: `{primary_score_sha}`")
        md.append(f"- FILE14 final summary SHA-256: `{shafile(summary14_path)}`")
        md.append(f"- FILE14 production script SHA-256: `{script14_sha}`")
        md.append(f"- FILE15 script SHA-256: `{script15_sha}`")
        md.append("")
        atomic_text(report_path, "\n".join(md) + "\n")

        log("Stage 6/6 — freeze FILE15 summary, lock, and completion flag")
        final15 = {
            "file": "FILE15",
            "version": VERSION,
            "status": FINAL_FILE15_STATUS,
            "created_utc": utc(),
            "mode": "integrity_and_reporting_only",
            "analysis_freeze": True,
            "pipeline_complete": True,
            "source_anchor_sha256": source_anchor_sha,
            "source_anchor": source_anchor_obj,
            "E3": {
                "n": n,
                "R": nr,
                "S": ns,
                "prevalence_R": prevalence,
                "cohort_sha256": summary14["E3"]["cohort_sha256"],
            },
            "FILE13": summary14["FILE13"],
            "FILE14": {
                "status": summary14["status"],
                "final_summary_sha256": shafile(summary14_path),
                "annotation_protocol_sha256": summary14.get("annotation_protocol_sha256"),
                "fasta_content_manifest_sha256": summary14.get("fasta_content_manifest_sha256"),
                "blind_primary_scores_sha256": primary_score_sha,
                "script_version": summary14.get("script_version"),
                "script_sha256": script14_sha,
            },
            "bootstrap": {
                "method": "cluster bootstrap; resolved NCBI SNP/ERD groups clustered; unresolved group IDs treated as singleton units",
                "replicates": 2000,
                "resolved_group_count": resolved_groups,
                "unresolved_singleton_count": unresolved_n,
                "effective_bootstrap_units": effective_bootstrap_units,
                "max_resolved_cluster_size": resolved_max_cluster_size,
            },
            "primary_metrics": primary_table.to_dict("records"),
            "sensitivity_metrics": sens_table.to_dict("records") if len(sens_table) else [],
            "diagnostics": summary14.get("diagnostics", {}),
            "guardrails": summary14.get("guardrails", {}),
            "integrity": {
                "checks_total": len(checks.rows),
                "checks_failed": len(checks.failures),
                "warnings": len(checks.warnings),
            },
            "outputs": {
                "integrity_checks": rel(root, checks_path),
                "artifact_manifest": rel(root, manifest_path),
                "primary_metrics_manuscript": rel(root, primary_table_path),
                "sensitivity_metrics_manuscript": rel(root, sensitivity_table_path) if len(sens_table) else None,
                "manuscript_report": rel(root, report_path),
            },
            "next_step": "Manuscript/figures/supplement/repository preparation only; no further E3-driven model modification.",
        }
        atomic_json(summary15_path, final15)

        locked_outputs = {
            rel(root, checks_path): shafile(checks_path),
            rel(root, manifest_path): shafile(manifest_path),
            rel(root, primary_table_path): shafile(primary_table_path),
            rel(root, report_path): shafile(report_path),
            rel(root, summary15_path): shafile(summary15_path),
        }
        if len(sens_table):
            locked_outputs[rel(root, sensitivity_table_path)] = shafile(sensitivity_table_path)

        final_lock = {
            "file": "FILE15",
            "version": VERSION,
            "status": "FILE15_FINAL_LOCKED",
            "locked_utc": utc(),
            "source_anchor_sha256": source_anchor_sha,
            "source_anchor": source_anchor_obj,
            "locked_outputs": locked_outputs,
            "analysis_freeze": True,
        }
        atomic_json(lock_path, final_lock)

        complete = {
            "file": "FILE15",
            "version": VERSION,
            "status": FINAL_FILE15_STATUS,
            "created_utc": utc(),
            "analysis_freeze": True,
            "pipeline_complete": True,
            "file15_final_summary": str(summary15_path),
            "file15_final_summary_sha256": shafile(summary15_path),
            "file15_final_lock": str(lock_path),
            "file15_final_lock_sha256": shafile(lock_path),
            "source_anchor_sha256": source_anchor_sha,
            "e3_cohort_sha256": summary14["E3"]["cohort_sha256"],
            "blind_primary_scores_sha256": primary_score_sha,
        }
        atomic_json(complete_path, complete)
        failure_json.unlink(missing_ok=True)

        log("=" * 112)
        log(f"FILE15 STATUS : {FINAL_FILE15_STATUS}")
        log(f"E3            : n={n} R={nr} S={ns}")
        log(f"Primary AUROC : {float(pm['roc_auc']['point_estimate']):.6f} [{float(pm['roc_auc']['ci95_low']):.6f}, {float(pm['roc_auc']['ci95_high']):.6f}]")
        log(f"Primary AP    : {float(pm['average_precision']['point_estimate']):.6f} [{float(pm['average_precision']['ci95_low']):.6f}, {float(pm['average_precision']['ci95_high']):.6f}]")
        log(f"Checks        : {len(checks.rows)} total | {len(checks.failures)} failed | {len(checks.warnings)} warnings")
        log(f"Report        : {report_path}")
        log(f"Final summary : {summary15_path}")
        log(f"Completion    : {complete_path}")
        log("ANALYSIS FREEZE: ENABLED — computational validation pipeline complete")
        log("=" * 112)
        return 0

    except Exception as e:
        obj = {
            "file": "FILE15",
            "version": VERSION,
            "status": "FAILED",
            "time_utc": utc(),
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        atomic_json(failure_json, obj)
        log("=" * 112)
        log("FILE15 STATUS : FAILED")
        log(f"Error         : {type(e).__name__}: {e}")
        log(f"Failure JSON  : {failure_json}")
        log("FILE13/FILE14 scientific artifacts were not modified.")
        log("=" * 112)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
