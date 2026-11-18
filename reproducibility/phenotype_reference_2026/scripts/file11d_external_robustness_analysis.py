#!/usr/bin/env python3
"""
FILE11D — POST-HOC EXTERNAL robustness analysis
===================================================================

Purpose
-------
Explain the performance drop observed in FILE11C without changing, tuning,
retraining, recalibrating, or re-scoring the frozen primary external model.

This script is intentionally POST-HOC and DIAGNOSTIC ONLY.

Scientific guardrails
---------------------
- Requires FILE11C status PASS_BLIND_EXTERNAL_VALIDATION.
- Verifies the frozen E2, model, and prediction SHA256 hashes recorded by FILE11C.
- Reproduces the primary external metrics exactly before any diagnostic analysis.
- Never changes the frozen 0.5 threshold.
- Never performs feature selection or hyperparameter selection using E2.
- Never writes a replacement prediction file.
- Never reports a "better" post-hoc model as the primary external result.
- Calibration intercept/slope are diagnostic estimates only; no recalibration is applied.
- All subgroup/error analyses are labeled exploratory and should not replace the
  locked FILE11C primary result.

Analyses
--------
1. Integrity and exact metric reproduction.
2. Internal-vs-external performance gap for the exact known_amr/logreg_l2 model.
3. Calibration / overconfidence diagnostics.
4. Seen-ST vs unseen-ST cluster-bootstrap contrasts.
5. Per-ST descriptive performance (all STs; small groups flagged).
6. ERD/SNP-cluster error concentration and leave-one-cluster-out sensitivity.
7. Coefficient-weighted frozen-feature prevalence shift.
8. Known-feature associations with FP/FN/error status.
9. Per-genome out-of-schema determinant reconstruction from FILE11C AMRFinder TSVs.
10. Out-of-schema determinant associations with errors.
11. Error-case profiles with high-impact frozen features and unseen determinants.

Typical run
-----------
python scripts/file11d_external_reviewer_defense.py \
  --project-root /mnt/c/Users/ASUS/Desktop/AMR-Genome-ML \
  --bootstrap-replicates 2000

Outputs
-------
data/external_validation/ncbi_pathogen_detection/file11d/
  file11d_primary_metric_reproduction.csv
  file11d_internal_external_gap.csv
  file11d_calibration_bins.csv
  file11d_calibration_summary.csv
  file11d_confidence_error_profile.csv
  file11d_lineage_metrics.csv
  file11d_lineage_cluster_bootstrap.csv
  file11d_st_metrics.csv
  file11d_erd_cluster_error_profile.csv
  file11d_erd_loco_sensitivity.csv
  file11d_coefficient_weighted_domain_shift.csv
  file11d_domain_shift_summary.csv
  file11d_known_feature_error_associations.csv
  file11d_out_of_schema_per_genome.csv
  file11d_out_of_schema_error_associations.csv
  file11d_error_case_profiles.csv
  file11d_analysis_design.json

checkpoints/external_validation_file11d/
  file11d_final_summary.json
  file11d_last_failure.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import fisher_exact
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
EXPECTED_FEATURES = 668
EXPECTED_BRIDGE_N = 382

ALLOWED_AMR_SUBTYPES = {"AMR", "POINT", "POINT_DISRUPT"}

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
            raise RuntimeError(f"Active FILE11D lock exists: {path} (PID {pid})")
        path.unlink(missing_ok=True)
    atomic_json(path, {"pid": os.getpid(), "started_utc": utc_now(), "version": VERSION})


def signal_handler(_signum, _frame):
    global STOP
    STOP = True


def norm_key(x: Any) -> str:
    return str(x).strip().lower()


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if not np.any(ok):
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = q
    out[np.flatnonzero(ok)] = restored
    return out


# =============================================================================
# Metrics
# =============================================================================

def metric_bundle(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=np.uint8)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y)),
        "R": int(y.sum()),
        "S": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": (
            float(average_precision_score(y, p)) if int(y.sum()) > 0 else float("nan")
        ),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")
        ),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": (
            float(matthews_corrcoef(y, pred))
            if len(np.unique(y)) == 2 and len(np.unique(pred)) == 2
            else float("nan")
        ),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def error_type(y: pd.Series, pred: pd.Series) -> pd.Series:
    out = np.full(len(y), "UNKNOWN", dtype=object)
    yy = y.to_numpy(dtype=int)
    pp = pred.to_numpy(dtype=int)
    out[(yy == 1) & (pp == 1)] = "TP"
    out[(yy == 0) & (pp == 0)] = "TN"
    out[(yy == 0) & (pp == 1)] = "FP"
    out[(yy == 1) & (pp == 0)] = "FN"
    return pd.Series(out, index=y.index)


# =============================================================================
# Calibration
# =============================================================================

def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """
    Diagnostic calibration model:
        logit(P(Y=1)) = intercept + slope * logit(p_frozen)

    This does NOT recalibrate predictions.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    def nll(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * logit_p
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    res = minimize(
        nll,
        x0=np.asarray([0.0, 1.0], dtype=float),
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-9},
    )
    if not res.success:
        # Retry from prevalence-only intercept.
        prev = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        start = np.asarray([math.log(prev / (1 - prev)), 0.5], dtype=float)
        res = minimize(nll, x0=start, method="BFGS", options={"maxiter": 2000})
    if not np.all(np.isfinite(res.x)):
        raise RuntimeError("Calibration intercept/slope optimization failed.")
    return float(res.x[0]), float(res.x[1])


def calibration_bins(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    work = pred_df[["label_binary", "probability_R"]].copy()

    rows = []
    eces = {}

    # Equal-width, 10 bins.
    edges = np.linspace(0.0, 1.0, 11)
    work["bin"] = pd.cut(
        work["probability_R"],
        bins=edges,
        include_lowest=True,
        right=True,
    )
    for b, g in work.groupby("bin", observed=False):
        if len(g) == 0:
            continue
        rows.append(
            {
                "binning": "equal_width_10",
                "bin": str(b),
                "n": len(g),
                "mean_probability_R": float(g["probability_R"].mean()),
                "observed_R_rate": float(g["label_binary"].mean()),
                "calibration_gap_observed_minus_predicted": float(
                    g["label_binary"].mean() - g["probability_R"].mean()
                ),
            }
        )
    tmp = pd.DataFrame([r for r in rows if r["binning"] == "equal_width_10"])
    eces["ece_equal_width_10"] = float(
        np.sum(
            tmp["n"].to_numpy() / len(work)
            * np.abs(tmp["calibration_gap_observed_minus_predicted"].to_numpy())
        )
    )

    # Equal-frequency, nominally 10 bins.
    work["qbin"] = pd.qcut(
        work["probability_R"],
        q=10,
        duplicates="drop",
    )
    qrows = []
    for b, g in work.groupby("qbin", observed=False):
        if len(g) == 0:
            continue
        qrows.append(
            {
                "binning": "equal_frequency_10",
                "bin": str(b),
                "n": len(g),
                "mean_probability_R": float(g["probability_R"].mean()),
                "observed_R_rate": float(g["label_binary"].mean()),
                "calibration_gap_observed_minus_predicted": float(
                    g["label_binary"].mean() - g["probability_R"].mean()
                ),
            }
        )
    rows.extend(qrows)
    tmp = pd.DataFrame(qrows)
    eces["ece_equal_frequency_10"] = float(
        np.sum(
            tmp["n"].to_numpy() / len(work)
            * np.abs(tmp["calibration_gap_observed_minus_predicted"].to_numpy())
        )
    )

    return pd.DataFrame(rows), eces


# =============================================================================
# Cluster bootstrap
# =============================================================================

def lineage_cluster_bootstrap(
    pred_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    metrics = ["roc_auc", "average_precision", "balanced_accuracy", "mcc", "brier"]

    groups = pred_df["erd_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    group_indices = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)

    point = {}
    for lineage in ("seen_ST", "unseen_ST"):
        g = pred_df[pred_df["lineage_novelty"].eq(lineage)]
        point[lineage] = metric_bundle(
            g["label_binary"].to_numpy(dtype=np.uint8),
            g["probability_R"].to_numpy(dtype=float),
            g["predicted_R"].to_numpy(dtype=np.uint8),
        )

    draws = {m: [] for m in metrics}
    valid = 0

    for b in range(n_boot):
        if STOP:
            raise KeyboardInterrupt

        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled_groups])
        sample = pred_df.iloc[idx]

        a = sample[sample["lineage_novelty"].eq("seen_ST")]
        c = sample[sample["lineage_novelty"].eq("unseen_ST")]

        if (
            len(a) < 2
            or len(c) < 2
            or a["label_binary"].nunique() < 2
            or c["label_binary"].nunique() < 2
        ):
            continue

        ma = metric_bundle(
            a["label_binary"].to_numpy(dtype=np.uint8),
            a["probability_R"].to_numpy(dtype=float),
            a["predicted_R"].to_numpy(dtype=np.uint8),
        )
        mc = metric_bundle(
            c["label_binary"].to_numpy(dtype=np.uint8),
            c["probability_R"].to_numpy(dtype=float),
            c["predicted_R"].to_numpy(dtype=np.uint8),
        )

        for m in metrics:
            if np.isfinite(ma[m]) and np.isfinite(mc[m]):
                draws[m].append(ma[m] - mc[m])

        valid += 1
        if b == 0 or (b + 1) % 250 == 0 or b + 1 == n_boot:
            log(f"[PROGRESS] lineage cluster bootstrap {b+1}/{n_boot} | valid={valid}")

    rows = []
    for m in metrics:
        arr = np.asarray(draws[m], dtype=float)
        if len(arr) == 0:
            continue
        rows.append(
            {
                "contrast": "seen_ST_minus_unseen_ST",
                "metric": m,
                "point_seen_ST": point["seen_ST"][m],
                "point_unseen_ST": point["unseen_ST"][m],
                "point_delta": point["seen_ST"][m] - point["unseen_ST"][m],
                "n_boot_requested": n_boot,
                "n_boot_valid": len(arr),
                "bootstrap_delta_mean": float(np.mean(arr)),
                "ci95_low": float(np.quantile(arr, 0.025)),
                "ci95_high": float(np.quantile(arr, 0.975)),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# AMRFinder parser / out-of-schema reconstruction
# =============================================================================

def parse_amrfinder_keys(path: Path) -> set[str]:
    df = pd.read_csv(path, sep="\t")
    required = {"Element symbol", "Type", "Subtype"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Unexpected AMRFinder schema: {path}")
    keep = (
        df["Type"].astype(str).str.upper().eq("AMR")
        & df["Subtype"].astype(str).str.upper().isin(ALLOWED_AMR_SUBTYPES)
    )
    out = set()
    for x in df.loc[keep, "Element symbol"]:
        if pd.isna(x):
            continue
        s = norm_key(x)
        if s:
            out.add(s)
    out.discard("emrd")
    return out


def reconstruct_unknown_per_genome(
    pred_df: pd.DataFrame,
    amr_dir: Path,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    schema = {norm_key(x) for x in feature_names}
    unknown_sets: dict[str, set[str]] = {}
    rows = []

    for i, row in enumerate(pred_df.itertuples(index=False), start=1):
        acc = str(row.asm_acc)
        path = amr_dir / f"{acc}.amrfinder.tsv"
        require_file(path, f"FILE11C AMRFinder output for {acc}")
        keys = parse_amrfinder_keys(path)
        unknown = keys - schema
        unknown_sets[acc] = unknown
        rows.append(
            {
                "asm_acc": acc,
                "amrfinder_determinants_total": len(keys),
                "frozen_schema_determinants_present": len(keys & schema),
                "out_of_schema_count": len(unknown),
                "out_of_schema_present": int(len(unknown) > 0),
                "out_of_schema_determinants": ";".join(sorted(unknown)),
            }
        )
        if i == 1 or i % 50 == 0 or i == len(pred_df):
            log(f"[PROGRESS] unknown determinant reconstruction {i}/{len(pred_df)}")

    return pd.DataFrame(rows), unknown_sets


# =============================================================================
# Fisher associations
# =============================================================================

def fisher_row(
    present: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[int, int, int, int, float, float]:
    pa = int(np.sum(present & mask_a))
    aa = int(np.sum((~present) & mask_a))
    pb = int(np.sum(present & mask_b))
    ab = int(np.sum((~present) & mask_b))
    try:
        odds, pval = fisher_exact([[pa, aa], [pb, ab]], alternative="two-sided")
        odds = float(odds)
        pval = float(pval)
    except Exception:
        odds, pval = float("nan"), float("nan")
    return pa, aa, pb, ab, odds, pval


def build_feature_error_associations(
    pred_df: pd.DataFrame,
    X_df: pd.DataFrame,
    coefficients: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    y = pred_df["label_binary"].to_numpy(dtype=int)
    pr = pred_df["predicted_R"].to_numpy(dtype=int)
    err = y != pr

    mask_fp = (y == 0) & (pr == 1)
    mask_tn = (y == 0) & (pr == 0)
    mask_fn = (y == 1) & (pr == 0)
    mask_tp = (y == 1) & (pr == 1)
    mask_err = err
    mask_correct = ~err

    coef_map = coefficients.set_index("feature")["coefficient"].to_dict()

    rows = []
    X = X_df[feature_names].to_numpy(dtype=np.uint8)

    for j, feature in enumerate(feature_names):
        present = X[:, j].astype(bool)

        e = fisher_row(present, mask_err, mask_correct)
        s = fisher_row(present, mask_fp, mask_tn)
        r = fisher_row(present, mask_fn, mask_tp)

        rows.append(
            {
                "feature": feature,
                "coefficient": float(coef_map.get(feature, np.nan)),
                "abs_coefficient": abs(float(coef_map.get(feature, np.nan))),
                "external_present": int(present.sum()),
                "prevalence_all": float(present.mean()),
                "prevalence_TP": safe_div(int(np.sum(present & mask_tp)), int(mask_tp.sum())),
                "prevalence_TN": safe_div(int(np.sum(present & mask_tn)), int(mask_tn.sum())),
                "prevalence_FP": safe_div(int(np.sum(present & mask_fp)), int(mask_fp.sum())),
                "prevalence_FN": safe_div(int(np.sum(present & mask_fn)), int(mask_fn.sum())),
                "error_vs_correct_odds_ratio": e[4],
                "error_vs_correct_p": e[5],
                "FP_vs_TN_odds_ratio": s[4],
                "FP_vs_TN_p": s[5],
                "FN_vs_TP_odds_ratio": r[4],
                "FN_vs_TP_p": r[5],
            }
        )

    out = pd.DataFrame(rows)
    out["error_vs_correct_q_BH"] = bh_fdr(out["error_vs_correct_p"].to_numpy())
    out["FP_vs_TN_q_BH"] = bh_fdr(out["FP_vs_TN_p"].to_numpy())
    out["FN_vs_TP_q_BH"] = bh_fdr(out["FN_vs_TP_p"].to_numpy())
    return out.sort_values(
        ["error_vs_correct_q_BH", "error_vs_correct_p", "abs_coefficient"],
        ascending=[True, True, False],
        na_position="last",
    )


def build_unknown_error_associations(
    pred_df: pd.DataFrame,
    unknown_sets: dict[str, set[str]],
) -> pd.DataFrame:
    all_unknown = sorted(set().union(*unknown_sets.values())) if unknown_sets else []

    y = pred_df["label_binary"].to_numpy(dtype=int)
    pr = pred_df["predicted_R"].to_numpy(dtype=int)
    asm = pred_df["asm_acc"].astype(str).tolist()

    mask_fp = (y == 0) & (pr == 1)
    mask_tn = (y == 0) & (pr == 0)
    mask_fn = (y == 1) & (pr == 0)
    mask_tp = (y == 1) & (pr == 1)
    mask_err = y != pr
    mask_correct = ~mask_err

    rows = []
    for det in all_unknown:
        present = np.asarray([det in unknown_sets[a] for a in asm], dtype=bool)

        e = fisher_row(present, mask_err, mask_correct)
        s = fisher_row(present, mask_fp, mask_tn)
        r = fisher_row(present, mask_fn, mask_tp)

        rows.append(
            {
                "determinant_key": det,
                "external_genomes_present": int(present.sum()),
                "prevalence_all": float(present.mean()),
                "prevalence_TP": safe_div(int(np.sum(present & mask_tp)), int(mask_tp.sum())),
                "prevalence_TN": safe_div(int(np.sum(present & mask_tn)), int(mask_tn.sum())),
                "prevalence_FP": safe_div(int(np.sum(present & mask_fp)), int(mask_fp.sum())),
                "prevalence_FN": safe_div(int(np.sum(present & mask_fn)), int(mask_fn.sum())),
                "error_vs_correct_odds_ratio": e[4],
                "error_vs_correct_p": e[5],
                "FP_vs_TN_odds_ratio": s[4],
                "FP_vs_TN_p": s[5],
                "FN_vs_TP_odds_ratio": r[4],
                "FN_vs_TP_p": r[5],
                "stable_interpretation_flag": int(present.sum()) >= 5,
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out["error_vs_correct_q_BH"] = bh_fdr(out["error_vs_correct_p"].to_numpy())
        out["FP_vs_TN_q_BH"] = bh_fdr(out["FP_vs_TN_p"].to_numpy())
        out["FN_vs_TP_q_BH"] = bh_fdr(out["FN_vs_TP_p"].to_numpy())
        out = out.sort_values(
            ["stable_interpretation_flag", "error_vs_correct_q_BH", "external_genomes_present"],
            ascending=[False, True, False],
            na_position="last",
        )
    return out


# =============================================================================
# Group summaries
# =============================================================================

def st_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st, g in pred_df.groupby("external_MLST", dropna=False):
        y = g["label_binary"].to_numpy(dtype=np.uint8)
        p = g["probability_R"].to_numpy(dtype=float)
        pr = g["predicted_R"].to_numpy(dtype=np.uint8)
        m = metric_bundle(y, p, pr)

        novelty_values = sorted(g["lineage_novelty"].dropna().astype(str).unique())
        m.update(
            {
                "external_MLST": st if pd.notna(st) else "UNRESOLVED",
                "lineage_novelty": (
                    novelty_values[0] if len(novelty_values) == 1 else ";".join(novelty_values)
                ),
                "errors": int(np.sum(y != pr)),
                "error_rate": float(np.mean(y != pr)),
                "stable_interpretation_flag": bool(
                    len(g) >= 10 and int(y.sum()) >= 3 and int((1 - y).sum()) >= 3
                ),
            }
        )
        rows.append(m)
    return pd.DataFrame(rows).sort_values(["n", "external_MLST"], ascending=[False, True])


def lineage_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lineage, g in pred_df.groupby("lineage_novelty", dropna=False):
        m = metric_bundle(
            g["label_binary"].to_numpy(dtype=np.uint8),
            g["probability_R"].to_numpy(dtype=float),
            g["predicted_R"].to_numpy(dtype=np.uint8),
        )
        m["lineage_novelty"] = lineage
        rows.append(m)
    return pd.DataFrame(rows)


def cluster_error_profile(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, g in pred_df.groupby("erd_group", dropna=False):
        y = g["label_binary"].to_numpy(dtype=np.uint8)
        pr = g["predicted_R"].to_numpy(dtype=np.uint8)
        et = error_type(g["label_binary"], g["predicted_R"])
        rows.append(
            {
                "erd_group": cluster,
                "n": len(g),
                "R": int(y.sum()),
                "S": int((1 - y).sum()),
                "errors": int(np.sum(y != pr)),
                "error_rate": float(np.mean(y != pr)),
                "TP": int((et == "TP").sum()),
                "TN": int((et == "TN").sum()),
                "FP": int((et == "FP").sum()),
                "FN": int((et == "FN").sum()),
                "mean_probability_R": float(g["probability_R"].mean()),
                "median_probability_R": float(g["probability_R"].median()),
                "unique_ST_count": int(g["external_MLST"].nunique(dropna=True)),
                "STs": ";".join(sorted(g["external_MLST"].dropna().astype(str).unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["errors", "n", "error_rate"], ascending=[False, False, False]
    )


def cluster_loco(pred_df: pd.DataFrame, cluster_profile: pd.DataFrame) -> pd.DataFrame:
    base = metric_bundle(
        pred_df["label_binary"].to_numpy(dtype=np.uint8),
        pred_df["probability_R"].to_numpy(dtype=float),
        pred_df["predicted_R"].to_numpy(dtype=np.uint8),
    )
    rows = []
    # Report all clusters with n >= 2; no post-hoc selection.
    for cluster in cluster_profile.loc[cluster_profile["n"] >= 2, "erd_group"]:
        g = pred_df[pred_df["erd_group"].astype(str).ne(str(cluster))]
        if g["label_binary"].nunique() < 2:
            continue
        m = metric_bundle(
            g["label_binary"].to_numpy(dtype=np.uint8),
            g["probability_R"].to_numpy(dtype=float),
            g["predicted_R"].to_numpy(dtype=np.uint8),
        )
        original = cluster_profile[
            cluster_profile["erd_group"].astype(str).eq(str(cluster))
        ].iloc[0]
        rows.append(
            {
                "excluded_erd_group": cluster,
                "excluded_n": int(original["n"]),
                "excluded_errors": int(original["errors"]),
                "n_remaining": int(len(g)),
                "roc_auc_after_exclusion": m["roc_auc"],
                "delta_roc_auc_vs_full": m["roc_auc"] - base["roc_auc"],
                "average_precision_after_exclusion": m["average_precision"],
                "delta_average_precision_vs_full": (
                    m["average_precision"] - base["average_precision"]
                ),
                "balanced_accuracy_after_exclusion": m["balanced_accuracy"],
                "delta_balanced_accuracy_vs_full": (
                    m["balanced_accuracy"] - base["balanced_accuracy"]
                ),
                "mcc_after_exclusion": m["mcc"],
                "delta_mcc_vs_full": m["mcc"] - base["mcc"],
                "brier_after_exclusion": m["brier"],
                "delta_brier_vs_full": m["brier"] - base["brier"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "delta_roc_auc_vs_full", key=lambda s: np.abs(s), ascending=False
    )


# =============================================================================
# Internal-external comparison
# =============================================================================

def internal_external_gap(
    perf09: pd.DataFrame,
    external_metrics: dict[str, float],
) -> pd.DataFrame:
    sel = perf09[
        perf09["representation"].astype(str).eq("known_amr")
        & perf09["model"].astype(str).eq("logreg_l2")
    ].copy()
    if len(sel) != 3:
        raise RuntimeError(
            f"Expected 3 File09 known_amr/logreg_l2 scheme rows, found {len(sel)}"
        )

    metric_map = {
        "roc_auc": "roc_auc_mean",
        "average_precision": "average_precision_mean",
        "balanced_accuracy": "balanced_accuracy_mean",
        "sensitivity": "sensitivity_mean",
        "specificity": "specificity_mean",
        "mcc": "mcc_mean",
        "brier": "brier_mean",
        "log_loss": "log_loss_mean",
    }

    rows = []
    for r in sel.itertuples(index=False):
        for metric, col in metric_map.items():
            internal = float(getattr(r, col))
            external = float(external_metrics[metric])
            rows.append(
                {
                    "internal_validation_scheme": r.scheme,
                    "representation": "known_amr",
                    "model": "logreg_l2",
                    "metric": metric,
                    "internal_mean": internal,
                    "external_E2": external,
                    "external_minus_internal": external - internal,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Error-case profiles
# =============================================================================

def error_case_profiles(
    pred_df: pd.DataFrame,
    X_df: pd.DataFrame,
    feature_names: list[str],
    coefficients: pd.DataFrame,
    unknown_per_genome: pd.DataFrame,
) -> pd.DataFrame:
    coef = coefficients.set_index("feature")["coefficient"].reindex(feature_names)
    coef_arr = coef.to_numpy(dtype=float)

    xidx = X_df.set_index("asm_acc")
    unknown_idx = unknown_per_genome.set_index("asm_acc")

    work = pred_df.copy()
    work["error_type"] = error_type(work["label_binary"], work["predicted_R"])
    work = work[work["error_type"].isin(["FP", "FN"])].copy()

    rows = []
    for row in work.itertuples(index=False):
        acc = str(row.asm_acc)
        vec = xidx.loc[acc, feature_names].to_numpy(dtype=np.uint8)
        present_idx = np.flatnonzero(vec == 1)

        pos = [
            (feature_names[j], coef_arr[j])
            for j in present_idx
            if np.isfinite(coef_arr[j]) and coef_arr[j] > 0
        ]
        neg = [
            (feature_names[j], coef_arr[j])
            for j in present_idx
            if np.isfinite(coef_arr[j]) and coef_arr[j] < 0
        ]
        pos.sort(key=lambda x: x[1], reverse=True)
        neg.sort(key=lambda x: x[1])

        et = "FP" if int(row.label_binary) == 0 else "FN"
        wrong_conf = (
            float(row.probability_R)
            if et == "FP"
            else float(1.0 - row.probability_R)
        )

        rows.append(
            {
                "asm_acc": acc,
                "target_acc": row.target_acc,
                "erd_group": row.erd_group,
                "external_MLST": row.external_MLST,
                "lineage_novelty": row.lineage_novelty,
                "error_type": et,
                "probability_R": float(row.probability_R),
                "wrong_class_confidence": wrong_conf,
                "high_confidence_error_ge_0.90": bool(wrong_conf >= 0.90),
                "top_positive_present_features": ";".join(
                    f"{f}:{c:.6g}" for f, c in pos[:7]
                ),
                "top_negative_present_features": ";".join(
                    f"{f}:{c:.6g}" for f, c in neg[:7]
                ),
                "out_of_schema_count": int(unknown_idx.loc[acc, "out_of_schema_count"]),
                "out_of_schema_determinants": unknown_idx.loc[
                    acc, "out_of_schema_determinants"
                ],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["wrong_class_confidence", "error_type"], ascending=[False, True]
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="FILE11D post-hoc reviewer-defense diagnostics for frozen FILE11C."
    )
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = ap.parse_args()

    if args.bootstrap_replicates < 1000:
        raise SystemExit("--bootstrap-replicates must be >= 1000")

    root = args.project_root.resolve()

    out_dir = root / "data/external_validation/ncbi_pathogen_detection/file11d"
    ckpt = root / "checkpoints/external_validation_file11d"
    lock_path = ckpt / "file11d.lock"
    failure_path = ckpt / "file11d_last_failure.json"
    final_summary_path = ckpt / "file11d_final_summary.json"

    inputs = {
        "file11c_summary": root
        / "checkpoints/external_validation_file11c/file11c_final_summary.json",
        "file11c_model": root
        / "checkpoints/external_validation_file11c/file11c_final_model.joblib",
        "file11c_predictions": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_blind_predictions.csv",
        "file11c_X_external": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_external_known_amr_668.csv",
        "file11c_coefficients": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_model_coefficients.csv",
        "file11c_shift": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_feature_prevalence_shift.csv",
        "file11c_unknown_summary": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_external_out_of_schema_determinants.csv",
        "file11c_bridge": root
        / "data/external_validation/ncbi_pathogen_detection/file11c/file11c_bridge_audit.csv",
        "e2": root
        / "data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv",
        "file09_performance": root / "data/evaluation/file09_performance_summary.csv",
    }
    amr_dir = root / "checkpoints/external_validation_file11c/amrfinder"

    outputs = {
        "primary_reproduction": out_dir / "file11d_primary_metric_reproduction.csv",
        "gap": out_dir / "file11d_internal_external_gap.csv",
        "calibration_bins": out_dir / "file11d_calibration_bins.csv",
        "calibration_summary": out_dir / "file11d_calibration_summary.csv",
        "confidence_errors": out_dir / "file11d_confidence_error_profile.csv",
        "lineage_metrics": out_dir / "file11d_lineage_metrics.csv",
        "lineage_bootstrap": out_dir / "file11d_lineage_cluster_bootstrap.csv",
        "st_metrics": out_dir / "file11d_st_metrics.csv",
        "cluster_profile": out_dir / "file11d_erd_cluster_error_profile.csv",
        "cluster_loco": out_dir / "file11d_erd_loco_sensitivity.csv",
        "weighted_shift": out_dir / "file11d_coefficient_weighted_domain_shift.csv",
        "shift_summary": out_dir / "file11d_domain_shift_summary.csv",
        "known_associations": out_dir / "file11d_known_feature_error_associations.csv",
        "unknown_per_genome": out_dir / "file11d_out_of_schema_per_genome.csv",
        "unknown_associations": out_dir / "file11d_out_of_schema_error_associations.csv",
        "error_profiles": out_dir / "file11d_error_case_profiles.csv",
        "design": out_dir / "file11d_analysis_design.json",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    acquire_lock(lock_path)

    stage = "startup"
    t0 = time.time()

    try:
        log("=" * 108)
        log("FILE11D — POST-HOC EXTERNAL VALIDATION REVIEWER-DEFENSE DIAGNOSTICS")
        log("=" * 108)
        log(f"Version       : {VERSION}")
        log(f"Project root  : {root}")
        log(f"Bootstrap     : {args.bootstrap_replicates}")
        log("Mode          : POST-HOC DIAGNOSTIC ONLY — NO MODEL CHANGES")

        for label, path in inputs.items():
            require_file(path, label)
        if not amr_dir.is_dir():
            raise FileNotFoundError(f"Missing FILE11C AMRFinder checkpoint directory: {amr_dir}")

        # ---------------------------------------------------------------------
        stage = "stage1_integrity"
        log("Stage 1/9 — verify FILE11C freeze, hashes, and exact primary metric reproduction")

        summary = json.loads(inputs["file11c_summary"].read_text(encoding="utf-8"))
        if summary.get("status") != "PASS_BLIND_EXTERNAL_VALIDATION":
            raise RuntimeError("FILE11C did not finish PASS_BLIND_EXTERNAL_VALIDATION.")
        if summary.get("bridge_audit", {}).get("status") != "PASS_EXACT":
            raise RuntimeError("FILE11C bridge audit is not PASS_EXACT.")
        if int(summary["bridge_audit"]["exact_row_matches"]) != EXPECTED_BRIDGE_N:
            raise RuntimeError("Unexpected FILE11C bridge audit row count.")
        if int(summary["bridge_audit"]["mismatched_cells"]) != 0:
            raise RuntimeError("FILE11C bridge audit has mismatched cells.")

        expected_hashes = {
            "frozen_e2_sha256": (inputs["e2"], summary["frozen_e2_sha256"]),
            "frozen_model_sha256": (
                inputs["file11c_model"],
                summary["frozen_model_sha256"],
            ),
            "frozen_prediction_sha256": (
                inputs["file11c_predictions"],
                summary["frozen_prediction_sha256"],
            ),
        }
        hash_rows = []
        for name, (path, expected) in expected_hashes.items():
            observed = sha256_file(path)
            hash_rows.append(
                {
                    "artifact": name,
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "match": observed == expected,
                }
            )
            if observed != expected:
                raise RuntimeError(
                    f"Frozen artifact hash mismatch for {name}: "
                    f"expected={expected}, observed={observed}"
                )

        pred = pd.read_csv(inputs["file11c_predictions"], dtype={"asm_acc": str})
        if len(pred) != EXPECTED_N or pred["asm_acc"].nunique() != EXPECTED_N:
            raise RuntimeError("FILE11C prediction row count/assembly uniqueness mismatch.")
        if int(pred["label_binary"].sum()) != EXPECTED_R:
            raise RuntimeError("External resistant count mismatch.")
        if int((1 - pred["label_binary"]).sum()) != EXPECTED_S:
            raise RuntimeError("External susceptible count mismatch.")

        y = pred["label_binary"].to_numpy(dtype=np.uint8)
        p = pred["probability_R"].to_numpy(dtype=float)
        frozen_pred = pred["predicted_R"].to_numpy(dtype=np.uint8)

        # Locked decision threshold must reproduce the stored class call.
        threshold_pred = (p >= 0.5).astype(np.uint8)
        if not np.array_equal(threshold_pred, frozen_pred):
            raise RuntimeError("Stored FILE11C class calls do not equal probability >= 0.5.")

        recomputed = metric_bundle(y, p, frozen_pred)
        stored = summary["metrics"]

        metric_rows = []
        for metric in (
            "roc_auc",
            "average_precision",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "mcc",
            "brier",
            "log_loss",
        ):
            obs = float(recomputed[metric])
            exp = float(stored[metric])
            delta = obs - exp
            metric_rows.append(
                {
                    "metric": metric,
                    "file11c_stored": exp,
                    "file11d_recomputed": obs,
                    "absolute_difference": abs(delta),
                    "exact_within_1e-12": abs(delta) <= 1e-12,
                }
            )
            if abs(delta) > 1e-12:
                raise RuntimeError(
                    f"Primary metric reproduction failed for {metric}: "
                    f"stored={exp}, recomputed={obs}"
                )
        atomic_csv(outputs["primary_reproduction"], pd.DataFrame(metric_rows))
        log(
            f"[PROGRESS] integrity PASS | n={len(pred)} | "
            f"AUROC={recomputed['roc_auc']:.4f} | "
            f"prediction_sha256={summary['frozen_prediction_sha256']}"
        )

        # Frozen model schema.
        artifact = joblib.load(inputs["file11c_model"])
        feature_names = list(artifact["features"])
        if len(feature_names) != EXPECTED_FEATURES:
            raise RuntimeError("Frozen model does not contain exactly 668 features.")

        Xext = pd.read_csv(inputs["file11c_X_external"], dtype={"asm_acc": str})
        if Xext.shape != (EXPECTED_N, EXPECTED_FEATURES + 1):
            raise RuntimeError(f"External frozen matrix shape mismatch: {Xext.shape}")
        if set(Xext["asm_acc"]) != set(pred["asm_acc"]):
            raise RuntimeError("External matrix and prediction assembly sets differ.")
        if [c for c in Xext.columns if c != "asm_acc"] != feature_names:
            raise RuntimeError("External frozen matrix feature order differs from frozen model.")

        # Align all per-genome data to the prediction row order.
        Xext = (
            Xext.set_index("asm_acc")
            .loc[pred["asm_acc"].astype(str), ["asm_acc"] if False else feature_names]
            .reset_index()
        )

        # Analysis design is frozen before the diagnostic results are written.
        design = {
            "script_version": VERSION,
            "created_utc": utc_now(),
            "analysis_class": "post-hoc exploratory reviewer-defense diagnostics",
            "primary_external_result_source": "FILE11C",
            "primary_external_result_is_modified": False,
            "model_retraining": False,
            "feature_selection": False,
            "hyperparameter_selection": False,
            "threshold_tuning": False,
            "threshold": 0.5,
            "recalibration_applied": False,
            "calibration_intercept_slope_use": "diagnostic only",
            "subgroup_results_use": "descriptive/exploratory only",
            "multiple_testing": "Benjamini-Hochberg FDR within each association family",
            "lineage_contrast_bootstrap_unit": "NCBI ERD/SNP cluster",
            "lineage_bootstrap_replicates": args.bootstrap_replicates,
            "small_ST_flag_rule": "stable flag requires n>=10, R>=3, S>=3",
        }
        atomic_json(outputs["design"], design)

        # ---------------------------------------------------------------------
        stage = "stage2_internal_external_gap"
        log("Stage 2/9 — quantify internal-to-external performance gap")
        perf09 = pd.read_csv(inputs["file09_performance"])
        gap = internal_external_gap(perf09, recomputed)
        atomic_csv(outputs["gap"], gap)
        log("[PROGRESS] internal/external gap table PASS")

        # ---------------------------------------------------------------------
        stage = "stage3_calibration"
        log("Stage 3/9 — calibration and high-confidence error diagnostics")
        cal_bins, eces = calibration_bins(pred)
        atomic_csv(outputs["calibration_bins"], cal_bins)

        cal_intercept, cal_slope = calibration_intercept_slope(y, p)
        observed_prev = float(y.mean())
        mean_pred = float(p.mean())
        oe = safe_div(observed_prev, mean_pred)

        cal_summary = pd.DataFrame(
            [
                {
                    "n": len(pred),
                    "observed_R_prevalence": observed_prev,
                    "mean_predicted_probability_R": mean_pred,
                    "observed_to_expected_ratio": oe,
                    "calibration_intercept": cal_intercept,
                    "calibration_slope": cal_slope,
                    "ideal_intercept": 0.0,
                    "ideal_slope": 1.0,
                    "ece_equal_width_10": eces["ece_equal_width_10"],
                    "ece_equal_frequency_10": eces["ece_equal_frequency_10"],
                    "brier": recomputed["brier"],
                    "log_loss": recomputed["log_loss"],
                }
            ]
        )
        atomic_csv(outputs["calibration_summary"], cal_summary)

        et = error_type(pred["label_binary"], pred["predicted_R"])
        conf_rows = []
        for t in (0.80, 0.90, 0.95, 0.99):
            fp_hi = int(((et == "FP") & (pred["probability_R"] >= t)).sum())
            fn_hi = int(((et == "FN") & (pred["probability_R"] <= (1 - t))).sum())
            n_confident = int(
                ((pred["probability_R"] >= t) | (pred["probability_R"] <= 1 - t)).sum()
            )
            conf_rows.append(
                {
                    "wrong_class_confidence_threshold": t,
                    "all_predictions_at_or_above_confidence": n_confident,
                    "high_confidence_FP": fp_hi,
                    "high_confidence_FN": fn_hi,
                    "high_confidence_errors_total": fp_hi + fn_hi,
                    "fraction_of_all_errors": safe_div(
                        fp_hi + fn_hi, int((y != frozen_pred).sum())
                    ),
                }
            )
        confidence_df = pd.DataFrame(conf_rows)
        atomic_csv(outputs["confidence_errors"], confidence_df)

        log(
            f"[PROGRESS] calibration | slope={cal_slope:.3f} | "
            f"intercept={cal_intercept:.3f} | "
            f"mean_p={mean_pred:.3f} vs observed={observed_prev:.3f}"
        )

        # ---------------------------------------------------------------------
        stage = "stage4_lineage"
        log("Stage 4/9 — lineage/ST robustness and ERD-cluster bootstrap contrast")
        lin = lineage_metrics(pred)
        atomic_csv(outputs["lineage_metrics"], lin)

        st = st_metrics(pred)
        atomic_csv(outputs["st_metrics"], st)

        lin_boot = lineage_cluster_bootstrap(
            pred,
            args.bootstrap_replicates,
            RANDOM_SEED,
        )
        atomic_csv(outputs["lineage_bootstrap"], lin_boot)
        log("[PROGRESS] lineage diagnostics PASS")

        # ---------------------------------------------------------------------
        stage = "stage5_cluster_concentration"
        log("Stage 5/9 — ERD/SNP-cluster error concentration and LOCO sensitivity")
        cluster_profile = cluster_error_profile(pred)
        atomic_csv(outputs["cluster_profile"], cluster_profile)
        loco = cluster_loco(pred, cluster_profile)
        atomic_csv(outputs["cluster_loco"], loco)
        log(
            f"[PROGRESS] ERD clusters={len(cluster_profile)} | "
            f"multi-isolate clusters={int((cluster_profile['n'] >= 2).sum())}"
        )

        # ---------------------------------------------------------------------
        stage = "stage6_domain_shift"
        log("Stage 6/9 — coefficient-weighted frozen-feature domain shift")
        shift = pd.read_csv(inputs["file11c_shift"])
        coefs = pd.read_csv(inputs["file11c_coefficients"])

        required_shift = {
            "feature",
            "internal_prevalence",
            "external_prevalence",
            "absolute_prevalence_shift",
        }
        if not required_shift.issubset(shift.columns):
            raise RuntimeError("Unexpected FILE11C prevalence-shift schema.")
        if not {"feature", "coefficient"}.issubset(coefs.columns):
            raise RuntimeError("Unexpected FILE11C coefficient schema.")

        weighted = shift.merge(
            coefs[["feature", "coefficient"]],
            on="feature",
            how="left",
            validate="one_to_one",
        )
        if weighted["coefficient"].isna().any():
            raise RuntimeError("Coefficient missing for at least one frozen feature.")

        weighted["external_minus_internal_prevalence"] = (
            weighted["external_prevalence"] - weighted["internal_prevalence"]
        )
        weighted["abs_coefficient"] = weighted["coefficient"].abs()
        weighted["coefficient_weighted_abs_shift"] = (
            weighted["abs_coefficient"] * weighted["absolute_prevalence_shift"]
        )
        weighted["marginal_signed_logit_shift"] = (
            weighted["coefficient"]
            * weighted["external_minus_internal_prevalence"]
        )
        weighted = weighted.sort_values(
            "coefficient_weighted_abs_shift", ascending=False
        )
        atomic_csv(outputs["weighted_shift"], weighted)

        shift_summary = pd.DataFrame(
            [
                {
                    "feature_count": len(weighted),
                    "features_abs_shift_gt_0.05": int(
                        (weighted["absolute_prevalence_shift"] > 0.05).sum()
                    ),
                    "features_abs_shift_gt_0.10": int(
                        (weighted["absolute_prevalence_shift"] > 0.10).sum()
                    ),
                    "features_abs_shift_gt_0.15": int(
                        (weighted["absolute_prevalence_shift"] > 0.15).sum()
                    ),
                    "max_absolute_prevalence_shift": float(
                        weighted["absolute_prevalence_shift"].max()
                    ),
                    "mean_absolute_prevalence_shift": float(
                        weighted["absolute_prevalence_shift"].mean()
                    ),
                    "sum_coefficient_weighted_abs_shift": float(
                        weighted["coefficient_weighted_abs_shift"].sum()
                    ),
                    "sum_marginal_signed_logit_shift": float(
                        weighted["marginal_signed_logit_shift"].sum()
                    ),
                    "interpretation": (
                        "Descriptive marginal shift only; ignores feature correlation "
                        "and is not a causal decomposition."
                    ),
                }
            ]
        )
        atomic_csv(outputs["shift_summary"], shift_summary)
        log(
            f"[PROGRESS] domain shift | >10pp="
            f"{int((weighted['absolute_prevalence_shift'] > 0.10).sum())} features"
        )

        # ---------------------------------------------------------------------
        stage = "stage7_known_feature_errors"
        log("Stage 7/9 — frozen known-feature associations with FP/FN/error status")
        known_assoc = build_feature_error_associations(
            pred, Xext, coefs, feature_names
        )
        atomic_csv(outputs["known_associations"], known_assoc)
        log("[PROGRESS] known-feature error associations PASS")

        # ---------------------------------------------------------------------
        stage = "stage8_unknown_features"
        log("Stage 8/9 — reconstruct and test out-of-schema determinants per genome")
        unknown_per_genome, unknown_sets = reconstruct_unknown_per_genome(
            pred, amr_dir, feature_names
        )
        atomic_csv(outputs["unknown_per_genome"], unknown_per_genome)

        # Cross-check aggregate FILE11C unknown determinant table exactly.
        aggregate = (
            pd.Series(
                {
                    k: sum(k in s for s in unknown_sets.values())
                    for k in sorted(set().union(*unknown_sets.values()))
                },
                name="external_genomes_present",
            )
            .rename_axis("determinant_key")
            .reset_index()
        )
        file11c_unknown = pd.read_csv(inputs["file11c_unknown_summary"]).copy()
        aggregate_cmp = aggregate.sort_values("determinant_key").reset_index(drop=True)
        stored_cmp = (
            file11c_unknown[["determinant_key", "external_genomes_present"]]
            .sort_values("determinant_key")
            .reset_index(drop=True)
        )
        if not aggregate_cmp.equals(stored_cmp):
            raise RuntimeError(
                "Per-genome out-of-schema reconstruction does not exactly reproduce "
                "FILE11C aggregate unknown-determinant counts."
            )

        unknown_assoc = build_unknown_error_associations(pred, unknown_sets)
        atomic_csv(outputs["unknown_associations"], unknown_assoc)
        log(
            f"[PROGRESS] out-of-schema reproduction PASS | "
            f"unique determinants={len(aggregate)}"
        )

        # ---------------------------------------------------------------------
        stage = "stage9_error_profiles"
        log("Stage 9/9 — build error-case profiles and freeze diagnostic summary")
        error_profiles = error_case_profiles(
            pred,
            Xext,
            feature_names,
            coefs,
            unknown_per_genome,
        )
        atomic_csv(outputs["error_profiles"], error_profiles)

        # Error concentration summaries.
        total_errors = int((y != frozen_pred).sum())
        cp = cluster_profile.sort_values("errors", ascending=False)
        sp = (
            st[["external_MLST", "errors"]]
            .sort_values("errors", ascending=False)
            .reset_index(drop=True)
        )

        cluster_concentration = {}
        for k in (1, 3, 5, 10, 20):
            nerr = int(cp.head(k)["errors"].sum())
            cluster_concentration[f"top_{k}_clusters_errors"] = nerr
            cluster_concentration[f"top_{k}_clusters_error_fraction"] = safe_div(
                nerr, total_errors
            )

        st_concentration = {}
        for k in (1, 3, 5, 10):
            nerr = int(sp.head(k)["errors"].sum())
            st_concentration[f"top_{k}_STs_errors"] = nerr
            st_concentration[f"top_{k}_STs_error_fraction"] = safe_div(
                nerr, total_errors
            )

        line_dict = {
            str(r["lineage_novelty"]): {
                k: (
                    int(r[k])
                    if k in {"n", "R", "S", "tn", "fp", "fn", "tp"}
                    else safe_float(r[k])
                )
                for k in lin.columns
                if k != "lineage_novelty"
            }
            for _, r in lin.iterrows()
        }

        high_conf_90 = confidence_df[
            confidence_df["wrong_class_confidence_threshold"].eq(0.90)
        ].iloc[0]

        final_summary = {
            "script_version": VERSION,
            "status": "PASS_POSTHOC_DIAGNOSTIC",
            "completed_utc": utc_now(),
            "elapsed_seconds": time.time() - t0,
            "analysis_class": "post-hoc exploratory reviewer-defense diagnostics",
            "primary_external_result_modified": False,
            "frozen_artifact_hashes_verified": True,
            "primary_metric_reproduction_exact": True,
            "external_primary": recomputed,
            "calibration": {
                "observed_R_prevalence": observed_prev,
                "mean_predicted_probability_R": mean_pred,
                "observed_to_expected_ratio": oe,
                "calibration_intercept": cal_intercept,
                "calibration_slope": cal_slope,
                "ece_equal_width_10": eces["ece_equal_width_10"],
                "ece_equal_frequency_10": eces["ece_equal_frequency_10"],
                "high_confidence_errors_ge_0_90": int(
                    high_conf_90["high_confidence_errors_total"]
                ),
                "high_confidence_FP_ge_0_90": int(
                    high_conf_90["high_confidence_FP"]
                ),
                "high_confidence_FN_ge_0_90": int(
                    high_conf_90["high_confidence_FN"]
                ),
            },
            "lineage_metrics": line_dict,
            "domain_shift": {
                "features_abs_shift_gt_0_05": int(
                    (weighted["absolute_prevalence_shift"] > 0.05).sum()
                ),
                "features_abs_shift_gt_0_10": int(
                    (weighted["absolute_prevalence_shift"] > 0.10).sum()
                ),
                "features_abs_shift_gt_0_15": int(
                    (weighted["absolute_prevalence_shift"] > 0.15).sum()
                ),
                "out_of_schema_unique_determinants": int(len(aggregate)),
                "genomes_with_out_of_schema_determinant": int(
                    unknown_per_genome["out_of_schema_present"].sum()
                ),
                "median_out_of_schema_count_per_genome": float(
                    unknown_per_genome["out_of_schema_count"].median()
                ),
                "max_out_of_schema_count_per_genome": int(
                    unknown_per_genome["out_of_schema_count"].max()
                ),
            },
            "error_concentration": {
                "total_errors": total_errors,
                **cluster_concentration,
                **st_concentration,
            },
            "multiple_testing": {
                "method": "Benjamini-Hochberg",
                "known_feature_tests": int(len(known_assoc)),
                "unknown_determinant_tests": int(len(unknown_assoc)),
            },
            "guardrails": {
                "model_retrained": False,
                "threshold_changed": False,
                "external_predictions_changed": False,
                "recalibration_applied": False,
                "subgroup_model_selection": False,
                "diagnostics_are_posthoc": True,
            },
            "outputs": {k: str(v.relative_to(root)) for k, v in outputs.items()},
        }
        atomic_json(final_summary_path, final_summary)

        if failure_path.exists():
            failure_path.unlink()

        log("=" * 108)
        log("FILE11D STATUS : PASS_POSTHOC_DIAGNOSTIC")
        log(f"Primary AUROC retained     : {recomputed['roc_auc']:.4f}")
        log(f"Calibration slope          : {cal_slope:.4f}")
        log(f"Calibration intercept      : {cal_intercept:.4f}")
        log(f"Mean predicted R           : {mean_pred:.4f}")
        log(f"Observed R prevalence      : {observed_prev:.4f}")
        log(
            f">10 percentage-point shifts: "
            f"{int((weighted['absolute_prevalence_shift'] > 0.10).sum())}"
        )
        log(f"Out-of-schema determinants : {len(aggregate)}")
        log(f"Total errors               : {total_errors}/{len(pred)}")
        log(f"Final summary              : {final_summary_path}")
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
                "message": "FILE11C frozen artifacts are unchanged.",
            },
        )
        log("FILE11D INTERRUPTED. FILE11C frozen artifacts remain unchanged.")
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
        log("FILE11D FAILED")
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
