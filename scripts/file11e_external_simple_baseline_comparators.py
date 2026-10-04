#!/usr/bin/env python3
"""
FILE11E — POST-HOC SIMPLE GENOTYPE BASELINE COMPARATORS
=======================================================

Purpose
-------
Compare the frozen FILE11C meropenem ML model against transparent genotype
rules on the same clonal-firewalled E2 cohort.

IMPORTANT: FILE11E is SECONDARY / POST-HOC.
The comparator analysis was designed after the primary FILE11C external result
was known. It therefore MUST NOT replace the locked FILE11C primary result and
MUST NOT be described as a pre-specified confirmatory comparison.

Scientific guardrails
---------------------
- Requires FILE11C PASS_BLIND_EXTERNAL_VALIDATION.
- Requires FILE11D PASS_POSTHOC_DIAGNOSTIC.
- Verifies frozen E2/model/prediction hashes from FILE11C.
- Reproduces the frozen ML metrics exactly.
- Never changes the ML model, 668-feature schema, threshold, or predictions.
- Never tunes a genotype rule using E2 labels.
- Reports ALL fixed comparator rules; does not select a winner.
- Uses paired NCBI ERD/SNP-cluster bootstrap contrasts.
- Rule calls are generated from the already-frozen FILE11C AMRFinder outputs
  BEFORE phenotype labels are used for scoring.

Fixed comparator rules
----------------------
A) development_prevalence_null
   Continuous score = development resistant prevalence (1601/4227).
   Hard call uses the same 0.5 threshold, therefore all susceptible.

B) kpc_family_rule
   Resistant iff an AMRFinder AMR determinant has Element symbol blaKPC*
   (case-insensitive).

C) amrfinder_carbapenemase_rule
   Resistant iff an AMRFinder AMR determinant is annotated as a carbapenemase
   by AMRFinder text fields (Class/Subclass/Element name containing
   "carbapenem") OR belongs to a fixed canonical carbapenemase family fallback
   (KPC, NDM, VIM, IMP, SME, IMI, NMC, SPM, GIM, SIM, DIM, AIM, TMB, FRI, BIC).

The semantic AMRFinder rule intentionally avoids treating all OXA or all GES
enzymes as carbapenemases. OXA-48-like and carbapenemase GES alleles are
captured when AMRFinder annotates them as carbapenem-related.

Typical run
-----------
python scripts/file11e_external_simple_baseline_comparators.py \
  --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML \
  --bootstrap-replicates 2000

Outputs
-------
data/external_validation/ncbi_pathogen_detection/file11e/
  file11e_rule_calls.csv
  file11e_rule_determinant_manifest.csv
  file11e_method_metrics.csv
  file11e_cluster_bootstrap_ci.csv
  file11e_paired_cluster_bootstrap_contrasts.csv
  file11e_lineage_method_metrics.csv
  file11e_error_overlap.csv
  file11e_discordant_cases.csv
  file11e_analysis_design.json

checkpoints/external_validation_file11e/
  file11e_final_summary.json
  file11e_last_failure.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

VERSION = "1.0.0"
RANDOM_SEED = 20260922

EXPECTED_N = 525
EXPECTED_R = 163
EXPECTED_S = 362
EXPECTED_DEV_N = 4227
EXPECTED_DEV_R = 1601
EXPECTED_DEV_S = 2626
EXPECTED_ERD_GROUPS = 289

ALLOWED_SUBTYPES = {"AMR", "POINT", "POINT_DISRUPT"}

# Fixed family fallback. Do not alter based on E2 results.
CANONICAL_CARBAPENEMASE_FAMILY_RE = re.compile(
    r"^(?:bla)?(?:"
    r"kpc|ndm|vim|imp|sme|imi|nmc|spm|gim|sim|dim|aim|tmb|fri|bic"
    r")(?:[-_].*|$)",
    flags=re.IGNORECASE,
)

KPC_RE = re.compile(r"^(?:bla)?kpc(?:[-_].*|$)", flags=re.IGNORECASE)

CARBAPENEM_SEMANTIC_RE = re.compile(
    r"carbapenem(?:ase|[- ]?hydroly\w*|[- ]?resistan\w*)?|carbapenem",
    flags=re.IGNORECASE,
)

STOP = False


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.csv", dir=path.parent)
    os.close(fd)
    p = Path(tmp)
    try:
        df.to_csv(p, index=False)
        with p.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(p, path)
    except Exception:
        p.unlink(missing_ok=True)
        raise


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            pid = int(old.get("pid", -1))
        except Exception:
            pid = -1
        if pid > 0 and Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"Active FILE11E lock exists: {path} (PID {pid})")
        path.unlink(missing_ok=True)
    atomic_json(path, {"pid": os.getpid(), "started_utc": utc_now(), "version": VERSION})


def signal_handler(_signum, _frame) -> None:
    global STOP
    STOP = True


def norm_symbol(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


# =============================================================================
# Metrics
# =============================================================================

def metric_bundle(
    y: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
    *,
    probability_score: bool,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=np.uint8)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    out = {
        "n": int(len(y)),
        "R": int(y.sum()),
        "S": int((1 - y).sum()),
        "roc_auc": (
            float(roc_auc_score(y, score))
            if len(np.unique(y)) == 2
            else float("nan")
        ),
        "average_precision": (
            float(average_precision_score(y, score))
            if int(y.sum()) > 0
            else float("nan")
        ),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, pred))
            if len(np.unique(y)) == 2
            else float("nan")
        ),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if probability_score:
        clipped = np.clip(score, 1e-15, 1 - 1e-15)
        out["brier"] = float(brier_score_loss(y, score))
        out["log_loss"] = float(log_loss(y, clipped, labels=[0, 1]))
    else:
        out["brier"] = float("nan")
        out["log_loss"] = float("nan")

    return out


CLASSIFICATION_METRICS = (
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
)

PROBABILITY_METRICS = ("brier", "log_loss")


# =============================================================================
# AMRFinder parsing and fixed rule calls
# =============================================================================

def retained_amr_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"Element symbol", "Type", "Subtype"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"Unexpected AMRFinder schema in {path}; missing={sorted(required-set(df.columns))}"
        )

    keep = (
        df["Type"].astype(str).str.upper().eq("AMR")
        & df["Subtype"].astype(str).str.upper().isin(ALLOWED_SUBTYPES)
    )
    return df.loc[keep].copy()


def semantic_text(row: pd.Series) -> str:
    candidate_cols = [
        "Element name",
        "Name",
        "Class",
        "Subclass",
        "Element symbol",
    ]
    vals = []
    for c in candidate_cols:
        if c in row.index and pd.notna(row[c]):
            vals.append(str(row[c]).strip())
    return " | ".join(vals)


def call_fixed_rules(amr_path: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    df = retained_amr_rows(amr_path)

    kpc_hits = []
    carba_hits = []
    det_rows = []

    for _, row in df.iterrows():
        symbol = norm_symbol(row["Element symbol"])
        text = semantic_text(row)

        kpc = bool(KPC_RE.match(symbol))
        semantic_carba = bool(CARBAPENEM_SEMANTIC_RE.search(text))
        fallback_carba = bool(CANONICAL_CARBAPENEMASE_FAMILY_RE.match(symbol))
        carba = semantic_carba or fallback_carba

        if kpc:
            kpc_hits.append(symbol)
        if carba:
            carba_hits.append(symbol)

        if kpc or carba:
            det_rows.append(
                {
                    "element_symbol": symbol,
                    "kpc_family_match": kpc,
                    "carbapenem_semantic_match": semantic_carba,
                    "canonical_family_fallback_match": fallback_carba,
                    "amrfinder_carbapenemase_match": carba,
                    "annotation_text": text,
                }
            )

    calls = {
        "kpc_family_rule": int(bool(kpc_hits)),
        "amrfinder_carbapenemase_rule": int(bool(carba_hits)),
    }
    return calls, det_rows


def build_rule_calls(
    pred: pd.DataFrame,
    amr_dir: Path,
    development_prevalence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    call_rows = []
    det_rows_all = []

    # Rule generation phase: no phenotype labels are used.
    for i, acc in enumerate(pred["asm_acc"].astype(str), start=1):
        path = amr_dir / f"{acc}.amrfinder.tsv"
        require_file(path, f"FILE11C AMRFinder TSV for {acc}")

        calls, det_rows = call_fixed_rules(path)

        row = {
            "asm_acc": acc,
            "development_prevalence_null_score": float(development_prevalence),
            "development_prevalence_null_call": int(development_prevalence >= 0.5),
            **calls,
        }
        call_rows.append(row)

        for d in det_rows:
            det_rows_all.append({"asm_acc": acc, **d})

        if i == 1 or i % 50 == 0 or i == len(pred):
            log(f"[PROGRESS] genotype-rule calls {i}/{len(pred)}")

    calls_df = pd.DataFrame(call_rows)
    det_df = pd.DataFrame(det_rows_all)

    if len(calls_df) != EXPECTED_N or calls_df["asm_acc"].nunique() != EXPECTED_N:
        raise RuntimeError("Rule-call row count/uniqueness mismatch.")

    return calls_df, det_df


# =============================================================================
# Method matrix
# =============================================================================

def method_specs(
    pred: pd.DataFrame,
    calls: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    c = calls.set_index("asm_acc").loc[pred["asm_acc"].astype(str)]

    return {
        "frozen_ml": {
            "score": pred["probability_R"].to_numpy(dtype=float),
            "pred": pred["predicted_R"].to_numpy(dtype=np.uint8),
            "probability_score": True,
            "description": "Frozen FILE11C known-AMR logistic regression, threshold 0.5",
        },
        "development_prevalence_null": {
            "score": c["development_prevalence_null_score"].to_numpy(dtype=float),
            "pred": c["development_prevalence_null_call"].to_numpy(dtype=np.uint8),
            "probability_score": True,
            "description": (
                "Constant probability equal to development R prevalence; "
                "hard call at 0.5"
            ),
        },
        "kpc_family_rule": {
            "score": c["kpc_family_rule"].to_numpy(dtype=float),
            "pred": c["kpc_family_rule"].to_numpy(dtype=np.uint8),
            "probability_score": False,
            "description": "AMRFinder blaKPC-family presence/absence rule",
        },
        "amrfinder_carbapenemase_rule": {
            "score": c["amrfinder_carbapenemase_rule"].to_numpy(dtype=float),
            "pred": c["amrfinder_carbapenemase_rule"].to_numpy(dtype=np.uint8),
            "probability_score": False,
            "description": (
                "Any AMRFinder carbapenemase semantic annotation or fixed "
                "canonical-family fallback"
            ),
        },
    }


# =============================================================================
# Cluster bootstrap
# =============================================================================

def cluster_bootstrap_all(
    pred_df: pd.DataFrame,
    methods: dict[str, dict[str, Any]],
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_all = pred_df["label_binary"].to_numpy(dtype=np.uint8)
    erd = pred_df["erd_group"].astype(str).to_numpy()
    unique = np.unique(erd)
    by_group = {g: np.flatnonzero(erd == g) for g in unique}

    rng = np.random.default_rng(seed)

    metrics_by_method = {
        method: {m: [] for m in CLASSIFICATION_METRICS + PROBABILITY_METRICS}
        for method in methods
    }

    comparator_names = [m for m in methods if m != "frozen_ml"]
    contrasts = {
        comp: {m: [] for m in CLASSIFICATION_METRICS + PROBABILITY_METRICS}
        for comp in comparator_names
    }

    valid = 0
    for b in range(n_boot):
        if STOP:
            raise KeyboardInterrupt

        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled_groups])
        y = y_all[idx]
        if len(np.unique(y)) < 2:
            continue

        boot_metrics = {}
        for name, spec in methods.items():
            mb = metric_bundle(
                y,
                np.asarray(spec["score"])[idx],
                np.asarray(spec["pred"])[idx],
                probability_score=bool(spec["probability_score"]),
            )
            boot_metrics[name] = mb
            for metric in CLASSIFICATION_METRICS + PROBABILITY_METRICS:
                v = mb[metric]
                if np.isfinite(v):
                    metrics_by_method[name][metric].append(v)

        ml = boot_metrics["frozen_ml"]
        for comp in comparator_names:
            cm = boot_metrics[comp]
            for metric in CLASSIFICATION_METRICS:
                if np.isfinite(ml[metric]) and np.isfinite(cm[metric]):
                    contrasts[comp][metric].append(ml[metric] - cm[metric])

            if methods[comp]["probability_score"]:
                for metric in PROBABILITY_METRICS:
                    if np.isfinite(ml[metric]) and np.isfinite(cm[metric]):
                        contrasts[comp][metric].append(ml[metric] - cm[metric])

        valid += 1
        if b == 0 or (b + 1) % 250 == 0 or b + 1 == n_boot:
            log(f"[PROGRESS] paired ERD-cluster bootstrap {b+1}/{n_boot} | valid={valid}")

    ci_rows = []
    for method, metric_map in metrics_by_method.items():
        for metric, vals in metric_map.items():
            arr = np.asarray(vals, dtype=float)
            if len(arr) == 0:
                continue
            ci_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_boot_requested": n_boot,
                    "n_boot_valid": len(arr),
                    "bootstrap_mean": float(arr.mean()),
                    "ci95_low": float(np.quantile(arr, 0.025)),
                    "ci95_high": float(np.quantile(arr, 0.975)),
                }
            )

    contrast_rows = []
    for comp, metric_map in contrasts.items():
        for metric, vals in metric_map.items():
            arr = np.asarray(vals, dtype=float)
            if len(arr) == 0:
                continue
            contrast_rows.append(
                {
                    "contrast": f"frozen_ml_minus_{comp}",
                    "comparator": comp,
                    "metric": metric,
                    "delta_definition": (
                        "ML minus comparator; positive favors ML for discrimination/"
                        "classification metrics; for brier/log_loss positive means ML has "
                        "higher loss (worse)"
                    ),
                    "n_boot_requested": n_boot,
                    "n_boot_valid": len(arr),
                    "bootstrap_delta_mean": float(arr.mean()),
                    "ci95_low": float(np.quantile(arr, 0.025)),
                    "ci95_high": float(np.quantile(arr, 0.975)),
                    "fraction_delta_gt_0": float(np.mean(arr > 0)),
                    "fraction_delta_lt_0": float(np.mean(arr < 0)),
                }
            )

    return pd.DataFrame(ci_rows), pd.DataFrame(contrast_rows)


# =============================================================================
# Descriptive comparisons
# =============================================================================

def point_metrics(
    pred: pd.DataFrame,
    methods: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    y = pred["label_binary"].to_numpy(dtype=np.uint8)
    rows = []
    for name, spec in methods.items():
        m = metric_bundle(
            y,
            spec["score"],
            spec["pred"],
            probability_score=bool(spec["probability_score"]),
        )
        rows.append(
            {
                "method": name,
                "description": spec["description"],
                "score_type": (
                    "probability" if spec["probability_score"] else "binary_rule"
                ),
                **m,
            }
        )
    return pd.DataFrame(rows)


def lineage_metrics(
    pred: pd.DataFrame,
    methods: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for lineage, g in pred.groupby("lineage_novelty", dropna=False):
        idx = g.index.to_numpy(dtype=int)
        y = pred.loc[idx, "label_binary"].to_numpy(dtype=np.uint8)
        for name, spec in methods.items():
            m = metric_bundle(
                y,
                np.asarray(spec["score"])[idx],
                np.asarray(spec["pred"])[idx],
                probability_score=bool(spec["probability_score"]),
            )
            rows.append(
                {
                    "lineage_novelty": lineage,
                    "method": name,
                    "description": spec["description"],
                    **m,
                }
            )
    return pd.DataFrame(rows)


def error_overlap(
    pred: pd.DataFrame,
    methods: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pred["label_binary"].to_numpy(dtype=np.uint8)
    ml_correct = np.asarray(methods["frozen_ml"]["pred"]) == y

    rows = []
    discordant_rows = []

    for name, spec in methods.items():
        if name == "frozen_ml":
            continue
        comp_pred = np.asarray(spec["pred"])
        comp_correct = comp_pred == y

        both_correct = int(np.sum(ml_correct & comp_correct))
        ml_only = int(np.sum(ml_correct & ~comp_correct))
        comp_only = int(np.sum(~ml_correct & comp_correct))
        both_wrong = int(np.sum(~ml_correct & ~comp_correct))

        rows.append(
            {
                "comparator": name,
                "both_correct": both_correct,
                "frozen_ml_only_correct": ml_only,
                "comparator_only_correct": comp_only,
                "both_wrong": both_wrong,
                "n": len(y),
            }
        )

        mask = ml_correct != comp_correct
        for i in np.flatnonzero(mask):
            discordant_rows.append(
                {
                    "asm_acc": pred.iloc[i]["asm_acc"],
                    "target_acc": pred.iloc[i]["target_acc"],
                    "erd_group": pred.iloc[i]["erd_group"],
                    "external_MLST": pred.iloc[i]["external_MLST"],
                    "lineage_novelty": pred.iloc[i]["lineage_novelty"],
                    "label_binary": int(y[i]),
                    "ml_probability_R": float(methods["frozen_ml"]["score"][i]),
                    "ml_call": int(methods["frozen_ml"]["pred"][i]),
                    "comparator": name,
                    "comparator_call": int(comp_pred[i]),
                    "which_was_correct": (
                        "frozen_ml" if ml_correct[i] else name
                    ),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(discordant_rows)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="FILE11E post-hoc simple genotype comparator analysis."
    )
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = ap.parse_args()

    if args.bootstrap_replicates < 1000:
        raise SystemExit("--bootstrap-replicates must be >=1000")

    root = args.project_root.resolve()
    out_dir = root / "data/external_validation/ncbi_pathogen_detection/file11e"
    ckpt = root / "checkpoints/external_validation_file11e"
    amr_dir = root / "checkpoints/external_validation_file11c/amrfinder"

    inputs = {
        "file11c_summary": root
        / "checkpoints/external_validation_file11c/file11c_final_summary.json",
        "file11d_summary": root
        / "checkpoints/external_validation_file11d/file11d_final_summary.json",
        "predictions": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_blind_predictions.csv",
        "e2": root
        / "data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv",
        "model": root
        / "checkpoints/external_validation_file11c/file11c_final_model.joblib",
    }

    outputs = {
        "rule_calls": out_dir / "file11e_rule_calls.csv",
        "rule_determinants": out_dir / "file11e_rule_determinant_manifest.csv",
        "metrics": out_dir / "file11e_method_metrics.csv",
        "bootstrap": out_dir / "file11e_cluster_bootstrap_ci.csv",
        "contrasts": out_dir / "file11e_paired_cluster_bootstrap_contrasts.csv",
        "lineage": out_dir / "file11e_lineage_method_metrics.csv",
        "overlap": out_dir / "file11e_error_overlap.csv",
        "discordant": out_dir / "file11e_discordant_cases.csv",
        "design": out_dir / "file11e_analysis_design.json",
    }

    lock_path = ckpt / "file11e.lock"
    failure_path = ckpt / "file11e_last_failure.json"
    summary_path = ckpt / "file11e_final_summary.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    acquire_lock(lock_path)

    stage = "startup"
    started = time.time()

    try:
        log("=" * 108)
        log("FILE11E — POST-HOC SIMPLE GENOTYPE BASELINE COMPARATORS")
        log("=" * 108)
        log(f"Version       : {VERSION}")
        log(f"Project root  : {root}")
        log(f"Bootstrap     : {args.bootstrap_replicates}")
        log("Mode          : SECONDARY POST-HOC — NO MODEL OR RULE TUNING")

        for label, path in inputs.items():
            require_file(path, label)
        if not amr_dir.is_dir():
            raise FileNotFoundError(f"Missing FILE11C AMRFinder directory: {amr_dir}")

        # ---------------------------------------------------------------------
        stage = "stage1_integrity"
        log("Stage 1/7 — verify FILE11C/FILE11D freeze and reproduce primary ML result")

        csum = json.loads(inputs["file11c_summary"].read_text(encoding="utf-8"))
        dsum = json.loads(inputs["file11d_summary"].read_text(encoding="utf-8"))

        if csum.get("status") != "PASS_BLIND_EXTERNAL_VALIDATION":
            raise RuntimeError("FILE11C is not PASS_BLIND_EXTERNAL_VALIDATION.")
        if dsum.get("status") != "PASS_POSTHOC_DIAGNOSTIC":
            raise RuntimeError("FILE11D is not PASS_POSTHOC_DIAGNOSTIC.")

        hash_checks = {
            "E2": (inputs["e2"], csum["frozen_e2_sha256"]),
            "model": (inputs["model"], csum["frozen_model_sha256"]),
            "predictions": (
                inputs["predictions"],
                csum["frozen_prediction_sha256"],
            ),
        }
        for name, (path, expected) in hash_checks.items():
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(
                    f"Frozen {name} SHA256 mismatch: expected={expected}, observed={observed}"
                )

        pred = pd.read_csv(inputs["predictions"], dtype={"asm_acc": str})
        if len(pred) != EXPECTED_N or pred["asm_acc"].nunique() != EXPECTED_N:
            raise RuntimeError("Prediction cohort size/uniqueness mismatch.")
        if int(pred["label_binary"].sum()) != EXPECTED_R:
            raise RuntimeError("External R count mismatch.")
        if int((1 - pred["label_binary"]).sum()) != EXPECTED_S:
            raise RuntimeError("External S count mismatch.")
        if pred["erd_group"].nunique() != EXPECTED_ERD_GROUPS:
            raise RuntimeError("Unexpected ERD/SNP-cluster count.")

        y = pred["label_binary"].to_numpy(dtype=np.uint8)
        p_ml = pred["probability_R"].to_numpy(dtype=float)
        pr_ml = pred["predicted_R"].to_numpy(dtype=np.uint8)

        ml_check = metric_bundle(y, p_ml, pr_ml, probability_score=True)
        for metric, expected in csum["metrics"].items():
            if metric not in ml_check:
                continue
            if abs(float(ml_check[metric]) - float(expected)) > 1e-12:
                raise RuntimeError(f"Frozen ML metric reproduction failed: {metric}")

        log(
            f"[PROGRESS] freeze PASS | n={len(pred)} | "
            f"AUROC={ml_check['roc_auc']:.4f} | "
            f"prediction_sha256={csum['frozen_prediction_sha256']}"
        )

        dev = csum["development"]
        if (
            int(dev["n"]) != EXPECTED_DEV_N
            or int(dev["R"]) != EXPECTED_DEV_R
            or int(dev["S"]) != EXPECTED_DEV_S
        ):
            raise RuntimeError("Unexpected development cohort counts.")
        dev_prev = EXPECTED_DEV_R / EXPECTED_DEV_N

        design = {
            "script_version": VERSION,
            "created_utc": utc_now(),
            "analysis_class": "secondary post-hoc comparator analysis",
            "important_limitation": (
                "Comparator analysis was designed after FILE11C primary E2 results "
                "were known; it is exploratory/reviewer-defense, not confirmatory."
            ),
            "primary_external_result_source": "FILE11C",
            "primary_external_result_modified": False,
            "ML_threshold_changed": False,
            "ML_retrained": False,
            "E2_used_to_tune_rules": False,
            "all_fixed_rules_reported": True,
            "bootstrap_unit": "NCBI ERD/SNP cluster",
            "bootstrap_replicates": args.bootstrap_replicates,
            "rules": {
                "development_prevalence_null": {
                    "score": f"{EXPECTED_DEV_R}/{EXPECTED_DEV_N}",
                    "hard_threshold": 0.5,
                },
                "kpc_family_rule": {
                    "definition": "AMRFinder retained AMR hit Element symbol blaKPC*"
                },
                "amrfinder_carbapenemase_rule": {
                    "definition": (
                        "AMRFinder annotation text contains carbapenem-related "
                        "terminology OR fixed canonical-family fallback"
                    ),
                    "canonical_family_fallback": (
                        "KPC, NDM, VIM, IMP, SME, IMI, NMC, SPM, GIM, SIM, "
                        "DIM, AIM, TMB, FRI, BIC"
                    ),
                    "note": (
                        "All OXA and all GES enzymes are NOT automatically classified "
                        "as carbapenemases; AMRFinder semantic annotation is used."
                    ),
                },
            },
        }
        atomic_json(outputs["design"], design)

        # ---------------------------------------------------------------------
        stage = "stage2_rule_generation"
        log("Stage 2/7 — generate fixed genotype-rule calls from frozen AMRFinder TSVs")
        calls, det_manifest = build_rule_calls(pred, amr_dir, dev_prev)
        atomic_csv(outputs["rule_calls"], calls)
        atomic_csv(outputs["rule_determinants"], det_manifest)

        log(
            f"[PROGRESS] calls frozen | KPC+={int(calls['kpc_family_rule'].sum())} | "
            f"AMRFinder-carbapenemase+={int(calls['amrfinder_carbapenemase_rule'].sum())}"
        )

        methods = method_specs(pred, calls)

        # ---------------------------------------------------------------------
        stage = "stage3_point_metrics"
        log("Stage 3/7 — score all methods on frozen E2 labels")
        metrics = point_metrics(pred, methods)
        atomic_csv(outputs["metrics"], metrics)
        for r in metrics.itertuples(index=False):
            log(
                f"[PROGRESS] {r.method}: "
                f"BA={r.balanced_accuracy:.4f} | MCC={r.mcc:.4f} | "
                f"Sens={r.sensitivity:.4f} | Spec={r.specificity:.4f}"
            )

        # ---------------------------------------------------------------------
        stage = "stage4_bootstrap"
        log("Stage 4/7 — paired ERD/SNP-cluster bootstrap uncertainty and contrasts")
        boot, contrasts = cluster_bootstrap_all(
            pred,
            methods,
            args.bootstrap_replicates,
            RANDOM_SEED,
        )
        atomic_csv(outputs["bootstrap"], boot)
        atomic_csv(outputs["contrasts"], contrasts)
        log("[PROGRESS] paired cluster bootstrap PASS")

        # ---------------------------------------------------------------------
        stage = "stage5_lineage"
        log("Stage 5/7 — descriptive method performance by seen/unseen ST")
        lin = lineage_metrics(pred, methods)
        atomic_csv(outputs["lineage"], lin)
        log("[PROGRESS] lineage method table PASS")

        # ---------------------------------------------------------------------
        stage = "stage6_error_overlap"
        log("Stage 6/7 — paired error overlap and discordant-case audit")
        overlap, discordant = error_overlap(pred, methods)
        atomic_csv(outputs["overlap"], overlap)
        atomic_csv(outputs["discordant"], discordant)
        log(f"[PROGRESS] discordant case rows={len(discordant)}")

        # ---------------------------------------------------------------------
        stage = "stage7_freeze_summary"
        log("Stage 7/7 — freeze comparator summary")

        metric_index = metrics.set_index("method")
        summary = {
            "script_version": VERSION,
            "status": "PASS_POSTHOC_COMPARATOR",
            "completed_utc": utc_now(),
            "elapsed_seconds": time.time() - started,
            "analysis_class": "secondary post-hoc comparator analysis",
            "confirmatory_claim_allowed": False,
            "frozen_primary_unchanged": True,
            "frozen_prediction_sha256": csum["frozen_prediction_sha256"],
            "development_prevalence": dev_prev,
            "method_metrics": {
                method: {
                    "n": int(metric_index.loc[method, "n"]),
                    "R": int(metric_index.loc[method, "R"]),
                    "S": int(metric_index.loc[method, "S"]),
                    "roc_auc": float(metric_index.loc[method, "roc_auc"]),
                    "average_precision": float(
                        metric_index.loc[method, "average_precision"]
                    ),
                    "balanced_accuracy": float(
                        metric_index.loc[method, "balanced_accuracy"]
                    ),
                    "sensitivity": float(metric_index.loc[method, "sensitivity"]),
                    "specificity": float(metric_index.loc[method, "specificity"]),
                    "f1": float(metric_index.loc[method, "f1"]),
                    "mcc": float(metric_index.loc[method, "mcc"]),
                    "accuracy": float(metric_index.loc[method, "accuracy"]),
                    "brier": (
                        None
                        if pd.isna(metric_index.loc[method, "brier"])
                        else float(metric_index.loc[method, "brier"])
                    ),
                    "log_loss": (
                        None
                        if pd.isna(metric_index.loc[method, "log_loss"])
                        else float(metric_index.loc[method, "log_loss"])
                    ),
                }
                for method in metrics["method"]
            },
            "rule_positive_counts": {
                "kpc_family_rule": int(calls["kpc_family_rule"].sum()),
                "amrfinder_carbapenemase_rule": int(
                    calls["amrfinder_carbapenemase_rule"].sum()
                ),
            },
            "guardrails": {
                "ML_retrained": False,
                "ML_threshold_changed": False,
                "ML_predictions_changed": False,
                "rules_tuned_on_E2_labels": False,
                "all_rules_reported": True,
                "paired_cluster_bootstrap": True,
                "posthoc_secondary_analysis": True,
            },
            "outputs": {
                k: str(v.relative_to(root))
                for k, v in outputs.items()
            },
        }
        atomic_json(summary_path, summary)

        failure_path.unlink(missing_ok=True)

        log("=" * 108)
        log("FILE11E STATUS : PASS_POSTHOC_COMPARATOR")
        log(
            f"Frozen ML                  : BA={metric_index.loc['frozen_ml','balanced_accuracy']:.4f} "
            f"| MCC={metric_index.loc['frozen_ml','mcc']:.4f}"
        )
        log(
            f"KPC-family rule            : BA={metric_index.loc['kpc_family_rule','balanced_accuracy']:.4f} "
            f"| MCC={metric_index.loc['kpc_family_rule','mcc']:.4f}"
        )
        log(
            f"AMRFinder carbapenemase    : BA={metric_index.loc['amrfinder_carbapenemase_rule','balanced_accuracy']:.4f} "
            f"| MCC={metric_index.loc['amrfinder_carbapenemase_rule','mcc']:.4f}"
        )
        log(
            f"Development prevalence null: Brier="
            f"{metric_index.loc['development_prevalence_null','brier']:.4f}"
        )
        log(f"Final summary              : {summary_path}")
        log("=" * 108)
        return 0

    except KeyboardInterrupt:
        atomic_json(
            failure_path,
            {
                "script_version": VERSION,
                "status": "INTERRUPTED",
                "stage": stage,
                "utc": utc_now(),
                "message": "FILE11C/FILE11D frozen artifacts remain unchanged.",
            },
        )
        log("FILE11E INTERRUPTED. Frozen primary artifacts remain unchanged.")
        return 130

    except Exception as exc:
        atomic_json(
            failure_path,
            {
                "script_version": VERSION,
                "status": "FAILED",
                "stage": stage,
                "utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        log("")
        log("FILE11E FAILED")
        log(f"Stage : {stage}")
        log(f"Error : {exc}")
        log(f"Crash record: {failure_path}")
        return 1

    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    raise SystemExit(main())
