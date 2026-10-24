#!/usr/bin/env python3
"""DL01 -- fixed, development-only 668-feature MLP feasibility study.

Reuses FILE07's 5 outer folds for random_stratified,
genomic_cluster_aware and mlst_aware; compares with FILE08's exact
known_amr/logreg_l2 OOF rows. Does not open E2/E3 and does not alter old files.

Method: locked 668 -> 64 -> 16 -> 1 MLP, dropout=.25, AdamW,
BCEWithLogitsLoss, early stopping by *train-only* inner holdout; refit
on the full outer train for the selected number of epochs before scoring
outer test. 5 fixed seeds by default. No architecture/threshold search.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

EXPECTED_N = 4227
EXPECTED_P = 668
SCHEMES = {
    "random_stratified": "random_fold",
    "genomic_cluster_aware": "genomic_cluster_fold",
    "mlst_aware": "mlst_fold",
}
SEEDS_DEFAULT = [42, 123, 999, 2026, 2027]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fail_if_output_exists(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {path}. Choose a new --outdir.")


def check_fold_df(df: pd.DataFrame, n_expected: int = EXPECTED_N) -> None:
    required = {"Genome ID", "y", "Genomic Cluster", "MLST", *SCHEMES.values()}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"FILE07 missing columns: {sorted(missing)}")
    if len(df) != n_expected or df["Genome ID"].isna().any() or df["Genome ID"].duplicated().any():
        raise RuntimeError(f"Invalid FILE07: n={len(df)}, expected={n_expected}, IDs not unique/nonempty")
    y = pd.to_numeric(df.y, errors="raise").to_numpy()
    if not set(np.unique(y)) <= {0, 1}:
        raise RuntimeError("Nonbinary FILE07 target")
    for scheme, col in SCHEMES.items():
        fold = pd.to_numeric(df[col], errors="raise")
        if set(fold.to_numpy()) != {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{scheme}: outer folds must be exactly 0..4")
        for k in range(5):
            it = np.flatnonzero(fold.to_numpy() == k)
            tr = np.flatnonzero(fold.to_numpy() != k)
            if not len(it) or not len(tr) or len(np.unique(y[it])) != 2 or len(np.unique(y[tr])) != 2:
                raise RuntimeError(f"Invalid class distribution for {scheme} fold={k}")


def _group_values(df: pd.DataFrame, col: str) -> np.ndarray:
    """Unknown lineage labels must not be incorrectly treated as one giant group."""
    missing_tokens = {"", "na", "nan", "n/a", "none", "null", "unknown", "unresolved", "-"}
    out = []
    for raw, genome in zip(df[col].tolist(), df["Genome ID"].tolist()):
        val = "" if pd.isna(raw) else str(raw).strip()
        if val.lower() in missing_tokens or val == "0":
            out.append("unknown_singleton:" + genome)
        else:
            out.append("group:" + val)
    return np.asarray(out, dtype=str)


def make_inner_split(y_train: np.ndarray, groups_train: np.ndarray | None, inner_seed: int):
    if groups_train is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=inner_seed)
        a, b = next(splitter.split(np.zeros(len(y_train)), y_train))
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=inner_seed)
        candidates = []
        for a0, b0 in splitter.split(np.zeros(len(y_train)), y_train, groups_train):
            if len(np.unique(y_train[a0])) == 2 and len(np.unique(y_train[b0])) == 2:
                candidates.append((a0, b0))
        if not candidates:
            raise RuntimeError("No valid group-disjoint inner validation fold. Do not fall back to iid.")
        a, b = min(candidates, key=lambda z: abs(len(z[1]) / len(y_train) - .20))
        if set(groups_train[a]) & set(groups_train[b]):
            raise RuntimeError("Inner validation group overlap")
    if not len(a) or not len(b):
        raise RuntimeError("Empty inner partition")
    return a.astype(np.int64), b.astype(np.int64)


def metrics(y, prob, threshold=0.5):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_R": int(y.sum()),
        "auroc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else None,
        "ap": float(average_precision_score(y, prob)) if y.sum() else None,
        "brier": float(brier_score_loss(y, prob)),
        "sensitivity_at_0p5": float(tp/(tp+fn)) if tp+fn else None,
        "specificity_at_0p5": float(tn/(tn+fp)) if tn+fp else None,
        "balanced_accuracy_at_0p5": float(balanced_accuracy_score(y, pred)),
        "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
    }


def create_model(torch, nn):
    return nn.Sequential(
        nn.Linear(EXPECTED_P, 64), nn.ReLU(), nn.Dropout(.25),
        nn.Linear(64, 16), nn.ReLU(), nn.Dropout(.25),
        nn.Linear(16, 1)
    )


def set_seed(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(torch, net, opt, loss_fn, data_x, data_y, batch_size, rng):
    net.train()
    perm = rng.permutation(len(data_y))
    acc = 0.0
    for start in range(0, len(perm), batch_size):
        batch = perm[start:start+batch_size]
        xb = data_x[batch]
        yb = data_y[batch]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net(xb).squeeze(-1), yb)
        loss.backward()
        opt.step()
        acc += float(loss.item()) * len(batch)
    return acc / len(data_y)


def fit_and_predict(torch, nn, xtr, ytr, xte, groups_train, seed, args):
    """One inner validation for early stopping, then full outer-train refit."""
    set_seed(torch, seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda, but torch.cuda.is_available() is false")
    a, b = make_inner_split(ytr, groups_train, args.inner_seed)
    if int(ytr[a].sum()) == 0 or int(ytr[b].sum()) == 0:
        raise RuntimeError("Inner split lacks positive class")
    xa = torch.from_numpy(xtr[a]).to(device)
    ya = torch.from_numpy(ytr[a].astype(np.float32)).to(device)
    xb = torch.from_numpy(xtr[b]).to(device)
    yb = torch.from_numpy(ytr[b].astype(np.float32)).to(device)
    net = create_model(torch, nn).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf"); best_epoch = 0; stale = 0
    rng = np.random.default_rng(seed + 9700)
    for epoch in range(1, args.max_epochs + 1):
        _run_epoch(torch, net, opt, loss_fn, xa, ya, args.batch_size, rng)
        net.eval()
        with torch.no_grad():
            loss_val = float(loss_fn(net(xb).squeeze(-1), yb).item())
        if loss_val < best_loss - args.min_delta:
            best_loss = loss_val; best_epoch = epoch; stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[DL01] INNER seed={seed} "
                f"epoch={epoch}/{args.max_epochs} "
                f"best_epoch={best_epoch} val_loss={loss_val:.5f}",
                flush=True
            )
        if stale >= args.patience:
            break
    if best_epoch < 1:
        raise RuntimeError("No finite validation-loss improvement in MLP")
    # Fresh full outer-training model, seeded identically and run for best_epoch.
    set_seed(torch, seed)
    net2 = create_model(torch, nn).to(device)
    opt2 = torch.optim.AdamW(net2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xx = torch.from_numpy(xtr).to(device)
    yy = torch.from_numpy(ytr.astype(np.float32)).to(device)
    rng = np.random.default_rng(seed + 9700)
    for refit_epoch in range(1,best_epoch+1):
        _run_epoch(torch, net2, opt2, loss_fn, xx, yy, args.batch_size, rng)
        if refit_epoch % 10 == 0 or refit_epoch == best_epoch:
            print(
                f"[DL01] REFIT seed={seed} "
                f"epoch={refit_epoch}/{best_epoch}",
                flush=True
            )
    net2.eval()
    with torch.no_grad():
        probs = torch.sigmoid(net2(torch.from_numpy(xte).to(device)).squeeze(-1)).cpu().numpy()
    if not np.isfinite(probs).all():
        raise RuntimeError("Nonfinite output probabilities")
    return np.asarray(probs, dtype=float), best_epoch, float(best_loss), net2


def read_data(root: Path, args):
    fold_path = root / "data/splits/file07_outer_folds.csv"
    known_path = root / "data/features/known_amr/X_known_amr_final.csv"
    lr_path = root / "data/modeling/file08_oof_predictions.csv"
    paths = [fold_path, known_path, lr_path]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Required original FILE07/08 artifact missing: {p}")
    fold = pd.read_csv(fold_path, dtype={"Genome ID":str})
    check_fold_df(fold)
    known = pd.read_csv(known_path, dtype={"Genome ID":str})
    if "Genome ID" not in known or len(known.columns) != EXPECTED_P+1 or known["Genome ID"].duplicated().any():
        raise RuntimeError("Known-AMR matrix must have Genome ID and exactly 668 unique features")
    feat_cols = [x for x in known.columns if x != "Genome ID"]
    if "emrD" in feat_cols:
        raise RuntimeError("Known-AMR matrix unexpectedly contains emrD")
    if known["Genome ID"].isna().any():
        raise RuntimeError("Missing Genome ID in AMR matrix")

    fold["Genome ID"] = fold["Genome ID"].str.strip()
    known["Genome ID"] = known["Genome ID"].str.strip()

    if (fold["Genome ID"].eq("").any()
            or known["Genome ID"].eq("").any()
            or fold["Genome ID"].duplicated().any()
            or known["Genome ID"].duplicated().any()):
        raise RuntimeError("Empty/duplicate IDs after normalization")

    fold_ids = set(fold["Genome ID"])
    known_ids = set(known["Genome ID"])
    missing = sorted(fold_ids - known_ids)
    extra = sorted(known_ids - fold_ids)

    print(
        f"[DL01] ID AUDIT: FILE07={len(fold_ids)} "
        f"AMR={len(known_ids)} missing={len(missing)} "
        f"extra={len(extra)}", flush=True
    )
    if missing:
        print("[DL01] Missing examples:", missing[:20], flush=True)
        print("[DL01] Extra examples:", extra[:20], flush=True)
        raise RuntimeError(
            "Development genome IDs missing from AMR features. "
            "Do not align by row position."
        )
    if extra:
        print(
            f"[DL01] Ignoring {len(extra)} extra AMR rows "
            "not in FILE07", flush=True
        )
    x = known.set_index("Genome ID").loc[fold["Genome ID"], feat_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all() or not np.isin(x, [0,1]).all():
        raise RuntimeError("Expected strictly binary 668-feature AMR matrix")
    y = pd.to_numeric(fold["y"],errors="raise").to_numpy(dtype=np.int64)
    lr = pd.read_csv(lr_path, dtype={"Genome ID":str})
    lr["Genome ID"] = lr["Genome ID"].str.strip()
    cols = {"Genome ID", "y", "scheme", "fold", "representation", "model", "score", "pred"}
    if not cols.issubset(lr.columns):
        raise RuntimeError(f"FILE08 predictions missing {sorted(cols-set(lr.columns))}")
    lr = lr.loc[(lr["model"]=="logreg_l2") & (lr["representation"]=="known_amr")].copy()
    if len(lr) != EXPECTED_N*3:
        raise RuntimeError(f"Expected FILE08 LR OOF 12681 rows; got {len(lr)}")
    if not set(lr["scheme"]) == set(SCHEMES):
        raise RuntimeError("Unexpected FILE08 schemes")
    fold_index = fold.set_index("Genome ID")
    lr["y"] = pd.to_numeric(lr["y"],errors="raise").astype(int)
    lr["fold"] = pd.to_numeric(lr["fold"],errors="raise").astype(int)
    lr["score"] = pd.to_numeric(lr["score"],errors="raise").astype(float)
    for scheme, col in SCHEMES.items():
        sub = lr.loc[lr.scheme==scheme]
        if sub["Genome ID"].duplicated().any() or set(sub["Genome ID"]) != set(fold["Genome ID"]):
            raise RuntimeError(f"Bad FILE08 LR OOF coverage for {scheme}")
        if not np.array_equal(sub.y.to_numpy(), fold_index.loc[sub["Genome ID"], "y"].astype(int).to_numpy()):
            raise RuntimeError(f"FILE08 LR targets differ from FILE07 for {scheme}")
        if not np.array_equal(sub.fold.to_numpy(), fold_index.loc[sub["Genome ID"], col].astype(int).to_numpy()):
            raise RuntimeError(f"FILE08 LR fold IDs differ from FILE07 for {scheme}")
    if not np.isfinite(lr.score.to_numpy()).all() or not lr.score.between(0,1).all():
        raise RuntimeError("FILE08 logistic OOF must contain valid probabilities")
    return fold, x, y, lr, dict(zip(["folds", "known_amr", "file08_oof"], map(sha, paths)))


def run(args):
    root = Path(args.project_root).expanduser().resolve()
    outdir = (root / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir).resolve()
    if len(set(args.seeds)) != len(args.seeds):
        raise RuntimeError("Duplicate seed IDs")
    if args.device not in ["cpu", "cuda"]:
        raise RuntimeError("Only cpu and cuda devices are supported")
    fold, x, y, lr, hashes = read_data(root, args)
    import torch
    from torch import nn
    torch.set_num_threads(max(1,args.threads))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.resume:
        if not outdir.is_dir():
            raise RuntimeError("Resume requires an existing output directory")
    else:
        fail_if_output_exists(outdir)
        outdir.mkdir(parents=True)
    # Immutable input provenance written before training starts.
    params = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SMOKE_TEST_NOT_RESEARCH_EVIDENCE" if args.smoke else "DEVELOPMENT_ONLY_EXPLORATORY_MLP_FEASIBILITY_NOT_EXTERNAL_CONFIRMATION",
        "folds_from": "FILE07 original fixed 5-fold schemes",
        "baseline_from": "FILE08 original known_amr/logreg_l2 frozen OOF",
        "x_shape": list(x.shape), "n_resistant":int(y.sum()), "n_susceptible":int(len(y)-y.sum()),
        "model": "668-64-16-1/ReLU/Dropout0.25/AdamW/BCEWithLogits",
        "hyperparameters_fixed": {"lr":args.lr,"weight_decay":args.weight_decay,"patience":args.patience,
            "max_epochs":args.max_epochs,"batch_size":args.batch_size,"inner_seed":args.inner_seed,
            "min_delta":args.min_delta},
        "seeds":args.seeds, "input_sha256": hashes,
        "rules": ["Do not use E2/E3 to select MLP or tune hyperparameters", 
            "Inner early-stopping split uses only each outer TRAIN fold; group-aware for cluster/MLST schemes", 
            "No augmentation, class reweighting, external cohorts, architecture search or threshold optimization",
            "No independent external novelty claim from this development experiment"]
    }
    params["device"] = args.device
    params["threads"] = args.threads
    params["save_fold_models"] = bool(args.save_fold_models)
    params["torch_version"] = torch.__version__
    digest_input = {k:v for k,v in params.items() if k!="created_utc"}
    digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True).encode()
    ).hexdigest()
    params["protocol_digest"] = digest

    protocol_path = outdir/"dl01_protocol_and_provenance.json"
    if args.resume:
        if not protocol_path.is_file():
            raise RuntimeError("Missing resume protocol")
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous.get("protocol_digest") != digest:
            raise RuntimeError("Protocol/input hash mismatch: refusing unsafe resume")
    else:
        protocol_path.write_text(
            json.dumps(params,indent=2),encoding="utf-8"
        )

    ckpt_dir = outdir/"checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    baseline = lr.set_index(["scheme","Genome ID"])
    predictions = []
    fold_rows = []
    elapsed = time.time()
    active_schemes = [("random_stratified", "random_fold")] if args.smoke else list(SCHEMES.items())
    active_seeds = args.seeds[:1] if args.smoke else args.seeds
    total_runs = len(active_schemes)*(1 if args.smoke else 5)*len(active_seeds)
    restored = 0

    def show_progress(scheme, fold_id, seed, mode):
        done = len(fold_rows)
        wall = time.time()-elapsed
        fresh = done-restored
        eta = (
            f"{wall/fresh*(total_runs-done)/60:.1f} min"
            if fresh > 0 else "calculating"
        )
        print(
            f"[DL01] PROGRESS {done}/{total_runs} "
            f"({100*done/total_runs:.1f}%) "
            f"scheme={scheme} fold={fold_id} seed={seed} "
            f"mode={mode} elapsed={wall/60:.1f} min ETA={eta}",
            flush=True
        )
    for scheme, col in active_schemes:
        fs = pd.to_numeric(fold[col], errors="raise").astype(int).to_numpy()
        group_all = (_group_values(fold, "Genomic Cluster") if scheme == "genomic_cluster_aware" else
                    _group_values(fold, "MLST") if scheme == "mlst_aware" else None)
        for outer_fold in ([0] if args.smoke else range(5)):
            train_idx = np.flatnonzero(fs != outer_fold)
            test_idx = np.flatnonzero(fs == outer_fold)
            if group_all is not None and set(group_all[train_idx]) & set(group_all[test_idx]):
                raise RuntimeError(f"FILE07 outer train/test group overlap: {scheme},fold={outer_fold}")
            test_ids = fold.iloc[test_idx]["Genome ID"].tolist()
            ref = baseline.loc[[(scheme,g) for g in test_ids]]
            yte = y[test_idx]
            p_ref = ref["score"].to_numpy(dtype=float)
            lr_m = metrics(yte,p_ref)
            groups_train = group_all[train_idx] if group_all is not None else None
            for seed in active_seeds:
                ckpt = ckpt_dir/f"{scheme}_fold{outer_fold}_seed{seed}.json"

                if args.resume and ckpt.is_file():
                    saved = json.loads(ckpt.read_text(encoding="utf-8"))
                    if saved.get("protocol_digest") != digest:
                        raise RuntimeError(f"Checkpoint hash mismatch: {ckpt}")
                    if (saved.get("scheme") != scheme
                            or saved.get("fold") != outer_fold
                            or saved.get("seed") != seed):
                        raise RuntimeError(f"Checkpoint identity mismatch: {ckpt}")
                    cached_preds = saved["predictions"]
                    if (len(cached_preds) != len(test_ids)
                            or {r["Genome ID"] for r in cached_preds} != set(test_ids)):
                        raise RuntimeError(f"Checkpoint genome coverage mismatch: {ckpt}")
                    fold_rows.append(saved["fold_row"])
                    predictions.extend(cached_preds)
                    restored += 1
                    show_progress(scheme, outer_fold, seed, "RESTORED")
                    continue

                print(
                    f"[DL01] START {scheme} fold={outer_fold} seed={seed}",
                    flush=True
                )
                p_mlp, epochs, inner_loss, net = fit_and_predict(
                    torch, nn, x[train_idx], y[train_idx], x[test_idx], groups_train, seed, args)
                mlp_m = metrics(yte,p_mlp)
                fold_rows.append({"scheme":scheme,"fold":outer_fold,"seed":seed,
                    "n_test":len(test_idx),"selected_epochs":epochs,"inner_bce":inner_loss,
                    **{"mlp_"+k:v for k,v in mlp_m.items()},
                    **{"lr_"+k:v for k,v in lr_m.items()},
                    **{"delta_mlp_minus_lr_"+k:(mlp_m[k]-lr_m[k] if mlp_m[k] is not None and lr_m[k] is not None else None)
                       for k in ["auroc","ap","brier","sensitivity_at_0p5","specificity_at_0p5","balanced_accuracy_at_0p5"]}})
                for i, gid in enumerate(test_ids):
                    predictions.append({"Genome ID":gid,"y":int(yte[i]),"scheme":scheme,
                        "fold":outer_fold,"seed":seed,"representation":"known_amr",
                        "model":"mlp_64_16_fixed","score":float(p_mlp[i]),
                        "pred":int(p_mlp[i]>=0.5), "lr_file08_score":float(p_ref[i]),
                        "lr_file08_pred":int(p_ref[i]>=0.5)})
                if args.save_fold_models:
                    torch.save({"state_dict":net.cpu().state_dict(),
                        "model_spec":params["model"], "input_columns": pd.read_csv(
                            root/"data/features/known_amr/X_known_amr_final.csv",nrows=0).columns[1:].tolist(),
                        "scheme":scheme,"outer_fold":outer_fold,"seed":seed,
                        "epochs":epochs,"training_data_sha256":hashes["known_amr"]},
                        outdir/f"dl01_{scheme}_fold{outer_fold}_seed{seed}.pt")
                payload = {
                    "protocol_digest": digest,
                    "scheme": scheme,
                    "fold": outer_fold,
                    "seed": seed,
                    "fold_row": fold_rows[-1],
                    "predictions": predictions[-len(test_ids):],
                }
                temporary = ckpt.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(payload, allow_nan=False), encoding="utf-8"
                )
                os.replace(temporary, ckpt)
                print(
                    f"[DL01] DONE epochs={epochs} "
                    f"AUC_MLP={mlp_m['auroc']:.4f} "
                    f"AUC_LR={lr_m['auroc']:.4f}",
                    flush=True
                )
                show_progress(scheme, outer_fold, seed, "SAVED")
    fold_metrics = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(predictions)
    # Audit each scheme/seed exactly OOF all 4227 and only one prediction per ID
    for scheme, _ in active_schemes:
        for seed in active_seeds:
            g = pred_df.loc[(pred_df.scheme == scheme) & (pred_df.seed == seed)]
            expected_n = int((pd.to_numeric(fold["random_fold"])==0).sum()) if args.smoke else EXPECTED_N
            if len(g)!=expected_n or g["Genome ID"].nunique()!=expected_n:
                raise RuntimeError(f"OOF integrity failed for {scheme}/{seed}")
    seed_rows=[]
    for (scheme,seed),g in pred_df.groupby(["scheme","seed"],sort=False):
        # Fold-macro metrics, aligned to FILE09 convention.
        f = fold_metrics.loc[(fold_metrics.scheme == scheme)&(fold_metrics.seed == seed)]
        row={"scheme":scheme,"seed":int(seed)}
        for metric in ["auroc","ap","brier","sensitivity_at_0p5","specificity_at_0p5","balanced_accuracy_at_0p5"]:
            row[f"mlp_macro_{metric}"]=float(f["mlp_"+metric].mean())
            row[f"lr_macro_{metric}"]=float(f["lr_"+metric].mean())
            row[f"delta_macro_{metric}"]=float(f["mlp_"+metric].mean()-f["lr_"+metric].mean())
        seed_rows.append(row)
    seed_metrics=pd.DataFrame(seed_rows)
    # Five random seeds on the same 5 outer folds are NOT 25 independent replicates.
    summary={**params, "runtime_s":time.time()-elapsed, "torch_version":torch.__version__,
        "device":args.device,"n_fit_runs":len(fold_rows),
        "macro_by_scheme":{scheme:{metric:{"seed_median":float(g[metric].median()),
            "seed_min":float(g[metric].min()),"seed_max":float(g[metric].max())}
            for metric in ["delta_macro_auroc","delta_macro_ap","delta_macro_brier",
                           "delta_macro_sensitivity_at_0p5","delta_macro_specificity_at_0p5",
                           "delta_macro_balanced_accuracy_at_0p5"]}
            for scheme,g in seed_metrics.groupby("scheme")},
        "limitations":["No E2/E3 external inference done", "No new genomic representation",
            "Outer folds reused from FILE07, model choice exploratory", 
            "Seed variation is not independent cohort uncertainty", 
            "Decision to promote MLP cannot be based on posthoc E2/E3 adaptation"]}
    fold_metrics.to_csv(outdir/"dl01_fold_metrics.csv",index=False)
    seed_metrics.to_csv(outdir/"dl01_seed_macro_metrics.csv",index=False)
    pred_df.to_csv(outdir/"dl01_oof_predictions.csv",index=False)
    (outdir/"dl01_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(f"[DL01] COMPLETE: {outdir}",flush=True)
    print(json.dumps({"n_models":len(fold_rows),"n_oof_predictions":len(pred_df),
        "macro_by_scheme":summary["macro_by_scheme"]},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",default=".")
    p.add_argument("--outdir",default="results/dl01_mlp_feasibility_v1")
    p.add_argument("--seeds",type=int,nargs="+",default=SEEDS_DEFAULT)
    p.add_argument("--inner-seed",type=int,default=20260920)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--weight-decay",type=float,default=0.01)
    p.add_argument("--max-epochs",type=int,default=100)
    p.add_argument("--patience",type=int,default=12)
    p.add_argument("--min-delta",type=float,default=0.0001)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--save-fold-models",action="store_true")
    p.add_argument("--resume",action="store_true",
                   help="Resume from completed scheme/fold/seed checkpoints")
    p.add_argument("--smoke",action="store_true",help="Only random-stratified outer fold 0, first seed. Not research evidence.")
    args=p.parse_args()
    try: run(args)
    except Exception as e:
        print(f"[DL01] ERROR: {type(e).__name__}: {e}",file=sys.stderr,flush=True)
        raise

if __name__=="__main__":main()
