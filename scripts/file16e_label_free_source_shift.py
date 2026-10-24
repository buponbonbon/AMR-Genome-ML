#!/usr/bin/env python3
"""FILE16E: label-free source-shift feasibility audit for frozen meropenem predictions.

Scope: retrospective E2 study only. Unlabelled source screening is kept separate
from phenotype-based post-hoc evaluation. No fitting, tuning, exclusion or calibration.
Not a clinical prediction tool. Python 3.10+, requires numpy pandas scipy sklearn.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# Fixed for this RUN only. Formula and cutoffs were selected POST-HOC after
# the E2 anomaly was already known; this is not prospective preregistration.
FAMILIES = (
    'No_detected_major_family', 'KPC_family', 'NDM_family',
    'OXA48_like_family', 'Multiple_major_families', 'VIM_family',
    'IMP_family',
)
GENE_COLS = (
    'has_CTX_M', 'has_candidate_AmpC_gene_family', 'has_SHV_family',
    'has_TEM_family', 'has_other_candidate_carbapenemase_family',
    'has_porin_related_amrfinder_symbol',
)
P_BINS = np.array([0, .1, .3, .5, .7, .9, 1.0000001])
WEIGHTS = {'model_score_js': .4, 'major_family_js': .4, 'gene_marginal_gap': .2}
FINGERPRINT_INPUTS = ['bioproject_acc', 'genome_id', 'prob_resistance', 'mechanism_group', *GENE_COLS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(buf)
    return digest.hexdigest()


def check_unique(df: pd.DataFrame, col: str, name: str) -> None:
    if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
        raise ValueError(f'{name}: null/blank {col}')
    if df[col].duplicated().any():
        raise ValueError(f'{name}: nonunique {col}; first duplicate = {df.loc[df[col].duplicated(), col].iloc[0]}')


def as_bool(series: pd.Series, name: str) -> pd.Series:
    mapping = {'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError(f'Invalid boolean {name}: {series[parsed.isna()].unique()[:5]}')
    return parsed.astype(int)


def hist_p(x: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=P_BINS)
    return counts / max(1, counts.sum())


def hist_family(x: pd.Series) -> np.ndarray:
    missing = set(x.dropna().unique()) - set(FAMILIES)
    if missing:
        raise ValueError(f'Unexpected mechanism groups: {missing}')
    counts = x.value_counts().reindex(FAMILIES, fill_value=0).to_numpy(dtype=float)
    return counts / max(1, counts.sum())


def jsd(a: np.ndarray, b: np.ndarray) -> float:
    # scipy returns sqrt(JSD). Square to return the conventional Jensen-Shannon divergence in [0,1].
    return float(jensenshannon(a, b, base=2.0) ** 2)


def score_one(target: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Reads ONLY pre-phenotype columns: no y_true, error, AST, MIC or phenotype."""
    model_score_js = jsd(hist_p(target.prob_resistance.to_numpy()),
                        hist_p(reference.prob_resistance.to_numpy()))
    major_family_js = jsd(hist_family(target.mechanism_group),
                          hist_family(reference.mechanism_group))
    gene_gap = float(np.abs(target[list(GENE_COLS)].mean().to_numpy() -
                             reference[list(GENE_COLS)].mean().to_numpy()).mean())
    composite = (WEIGHTS['model_score_js'] * model_score_js +
                 WEIGHTS['major_family_js'] * major_family_js +
                 WEIGHTS['gene_marginal_gap'] * gene_gap)
    p = target.prob_resistance.to_numpy()
    entropy = -(np.clip(p, 1e-15, 1-1e-15) * np.log2(np.clip(p, 1e-15, 1-1e-15)) +
                (1-np.clip(p, 1e-15, 1-1e-15)) * np.log2(1-np.clip(p, 1e-15, 1-1e-15)))
    family_present = target.mechanism_group.ne('No_detected_major_family').to_numpy()
    model_r = p >= .5
    return {
        'n': len(target), 'reference_n': len(reference),
        'label_free_score': float(composite),
        'model_score_js': model_score_js,
        'major_family_js': major_family_js,
        'gene_marginal_gap': gene_gap,
        'baseline_model_uncertainty': float(np.mean(entropy)),
        'baseline_panel_model_disagreement': float(np.mean(model_r != family_present)),
        'baseline_high_confidence_fraction': float(np.mean((p <= .1) | (p >= .9))),
        'mean_predicted_resistance_score': float(np.mean(p)),
    }


def get_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.project_root.resolve()
    files = {
        'frozen_e2': args.frozen or root / 'data/external_validation/ncbi_pathogen_detection/final/file11_external_strict_E2_FROZEN.csv',
        'predictions': args.predictions or root / 'results/file16_audit_mechanisms_v1/file16_per_genome_prediction_error_audit.csv',
        'genes': args.genes or root / 'results/file16c_discordance_v1/file16c_per_genome_gene_audit.csv',
    }
    result = {k: Path(v).expanduser().resolve() for k, v in files.items()}
    for k, p in result.items():
        if not p.is_file():
            raise FileNotFoundError(f'{k} not found: {p}')
    return result


def load_clean(paths: dict[str, Path]) -> pd.DataFrame:
    frozen_all = pd.read_csv(paths['frozen_e2'], low_memory=False)
    needed_frozen = {'antibiotic', 'asm_acc', 'bioproject_acc', 'label_binary', 'biosample_acc'}
    if missing := needed_frozen - set(frozen_all.columns):
        raise ValueError(f'frozen_e2 missing: {missing}')
    frozen = frozen_all[frozen_all.antibiotic.astype(str).str.lower().str.strip().eq('meropenem')].copy()
    if len(frozen) != 525:
        raise ValueError(f'Frozen meropenem E2 requires 525 rows, got {len(frozen)}')
    check_unique(frozen, 'asm_acc', 'frozen meropenem')
    if frozen.bioproject_acc.isna().any():
        raise ValueError('Missing BioProject in frozen E2')
    predictions_all = pd.read_csv(paths['predictions'], low_memory=False)
    preds = predictions_all[predictions_all.cohort.eq('E2')].copy()
    genes_all = pd.read_csv(paths['genes'], low_memory=False)
    genes = genes_all[genes_all.cohort.eq('E2')].copy()
    for nm, df in [('predictions', preds), ('genes', genes)]:
        check_unique(df, 'genome_id', nm)
        if len(df) != 525: raise ValueError(f'{nm}: expected 525 E2 genomes, got {len(df)}')
    pred_cols = ['genome_id', 'y_true', 'prob_resistance', 'mechanism_group']
    gene_cols = ['genome_id', *GENE_COLS]
    for col in pred_cols:
        if col not in preds: raise ValueError(f'prediction missing {col}')
    for col in gene_cols:
        if col not in genes: raise ValueError(f'gene missing {col}')
    d = frozen[['asm_acc', 'biosample_acc', 'bioproject_acc', 'label_binary']].merge(
        preds[pred_cols], left_on='asm_acc', right_on='genome_id', how='left', validate='one_to_one', indicator=True)
    if not d['_merge'].eq('both').all(): raise ValueError('E2 predictions do not match all frozen genome accessions')
    d = d.drop(columns='_merge').merge(genes[gene_cols], on='genome_id', how='left', validate='one_to_one', indicator=True)
    if not d['_merge'].eq('both').all(): raise ValueError('E2 gene audit does not match all frozen genome accessions')
    d = d.drop(columns='_merge')
    d['y_true'] = pd.to_numeric(d['y_true'], errors='raise').astype(int)
    d['label_binary'] = pd.to_numeric(d['label_binary'], errors='raise').astype(int)
    if set(d.y_true.unique()) != {0, 1} or (d.y_true != d.label_binary).any():
        raise ValueError('Frozen AST labels and FILE16 predictions disagree. Stop.')
    if int(d.y_true.sum()) != 163: raise ValueError('Expected resistant=163; check frozen cohort')
    d['prob_resistance'] = pd.to_numeric(d.prob_resistance, errors='raise')
    if d.prob_resistance.isna().any() or ((d.prob_resistance < 0) | (d.prob_resistance > 1)).any():
        raise ValueError('Non-numeric/missing/out-of-range probability')
    for c in GENE_COLS:
        d[c] = as_bool(d[c], c)
    if d.mechanism_group.isna().any(): raise ValueError('Missing mechanism groups')
    return d


def evaluate(scores: pd.DataFrame, observations: pd.DataFrame, top_fraction: float = .1) -> tuple[pd.DataFrame, dict]:
    # Label access is restricted to this procedure. No loop back into score construction.
    obs = observations.copy()
    obs['wrong'] = (obs.prob_resistance.ge(.5).astype(int) != obs.y_true).astype(int)
    obs['high_conf_wrong'] = obs.wrong.eq(1) & (obs.prob_resistance.le(.1) | obs.prob_resistance.ge(.9))
    outcome = []
    for group, d in obs.groupby('bioproject_acc', sort=True):
        r = d[d.y_true.eq(1)]
        auc = float(roc_auc_score(d.y_true, d.prob_resistance)) if d.y_true.nunique() == 2 else None
        outcome.append({
            'bioproject_acc': group, 'observed_n': len(d),
            'observed_error_rate': float(d.wrong.mean()),
            'observed_n_wrong': int(d.wrong.sum()),
            'observed_high_conf_error_rate': float(d.high_conf_wrong.mean()),
            'observed_fn_rate_among_resistant': float(((r.prob_resistance < .5).mean())) if len(r) else None,
            'observed_auc_if_both_labels': auc,
            'resistant_n': len(r),
        })
    merged = scores.merge(pd.DataFrame(outcome), on='bioproject_acc', how='left', validate='one_to_one')
    eligible = merged.loc[merged.is_eligible_for_ranking].copy()
    k = max(1, int(math.ceil(top_fraction * len(eligible))))
    top = eligible.sort_values(['label_free_score', 'bioproject_acc'], ascending=[False, True]).head(k)
    target = 'PRJNA1322038'
    best = scores.sort_values(['label_free_score', 'bioproject_acc'], ascending=[False, True]).reset_index(drop=True)
    loc = best.index[best.bioproject_acc.eq(target)]
    target_rank = (int(loc[0]) + 1) if len(loc) else None
    baseline_rank = {}
    for col, ascending in [('baseline_model_uncertainty', False),
                            ('baseline_panel_model_disagreement', False),
                            ('baseline_high_confidence_fraction', False)]:
        rnk = eligible.sort_values([col, 'bioproject_acc'], ascending=[ascending, True]).reset_index(drop=True)
        idx = rnk.index[rnk.bioproject_acc.eq(target)]
        baseline_rank[col] = (int(idx[0]) + 1) if len(idx) else None
    p = spearmanr(eligible.label_free_score, eligible.observed_error_rate) if len(eligible) >= 3 else None
    summary = {
        'n_all_sources': int(len(scores)),
        'n_eligible_sources': int(len(eligible)),
        'n_genomes_in_eligible_sources': int(eligible.n.sum()),
        'top_fraction_fixed_posthoc_in_script': float(top_fraction),
        'top_k': int(k),
        'source_critical_bioproject': target,
        'source_critical_label_free_rank_all_sources': target_rank,
        'source_critical_label_free_rank_eligible_sources': (int(np.where(eligible.sort_values(['label_free_score', 'bioproject_acc'], ascending=[False, True]).bioproject_acc.to_numpy() == target)[0][0]) + 1) if target in set(eligible.bioproject_acc) else None,
        'source_critical_flagged_top_k': bool(target in set(top.bioproject_acc)),
        'flagged_bioprojects': top.bioproject_acc.tolist(),
        'source_critical_baseline_ranks_eligible': baseline_rank,
        'source_level_spearman_score_vs_error': float(p.statistic) if p is not None and np.isfinite(p.statistic) else None,
        'source_level_spearman_pvalue_exploratory_only': float(p.pvalue) if p is not None and np.isfinite(p.pvalue) else None,
        'warning': 'No external reference: reference distribution pools OTHER E2 projects without labels. This is transductive retrospective screening, NOT validated prospective detection.',
    }
    return merged.sort_values('label_free_score', ascending=False), summary


def main() -> None:
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--project-root', type=Path, default=Path('.'))
    pa.add_argument('--frozen', type=Path, default=None)
    pa.add_argument('--predictions', type=Path, default=None)
    pa.add_argument('--genes', type=Path, default=None)
    pa.add_argument('--outdir', type=Path, default=Path('results/file16e_label_free_v1'))
    pa.add_argument('--min-source-n', type=int, default=10)
    args = pa.parse_args()
    if args.min_source_n < 2: raise ValueError('--min-source-n must be >= 2')
    out = args.outdir.resolve()
    if out.exists(): raise FileExistsError(f'Output directory already exists: {out}; choose a fresh _v2 directory')
    paths = get_input_paths(args)
    d = load_clean(paths)

    # Strict information boundary: the scores below are computed from an explicitly
    # projected, label-free dataframe. 'y_true' is not passed to score_one().
    features = d[FINGERPRINT_INPUTS].copy()
    rows = []
    for group, source in features.groupby('bioproject_acc', sort=True):
        rest = features.loc[features.bioproject_acc.ne(group)]
        result = score_one(source, rest)
        result['bioproject_acc'] = str(group)
        result['is_eligible_for_ranking'] = len(source) >= args.min_source_n
        rows.append(result)
    unsupervised = pd.DataFrame(rows)
    evaluated, summary = evaluate(unsupervised, d)
    # Safety: prevent cherry-picking by rerunning with a different n threshold.
    summary['min_source_n'] = args.min_source_n
    summary['weights_fixed_not_trained'] = WEIGHTS
    summary['probability_bins_fixed'] = P_BINS.tolist()
    summary['dropped_gene_columns'] = []
    summary['phenotype_used_in_screening_score'] = False
    summary['phenotype_used_only_for_afterward_evaluation'] = True
    summary['training_or_recalibration'] = False
    summary['threshold_tuned'] = False
    summary['project_selected_for_investigation_post_hoc'] = True
    summary['scope'] = 'E2_ONLY_POST_HOC_LABEL_FREE_SOURCE_SHIFT_FEASIBILITY'
    summary['no_claim_of_prospective_or_independent_validation'] = True
    summary['E2_primary_auroc_unmodified'] = 0.6673473884011795
    summary['input_sha256'] = {k: sha256(path) for k, path in paths.items()}
    summary['run_utc'] = datetime.now(timezone.utc).isoformat()
    # The output labelled rows are for retrospective verification only.
    out.mkdir(parents=True, exist_ok=False)
    unsupervised.sort_values('label_free_score', ascending=False).to_csv(
        out/'file16e_source_scores_LABEL_FREE.csv', index=False)
    evaluated.to_csv(out/'file16e_source_scores_WITH_POSTHOC_OUTCOMES.csv', index=False)
    with (out/'file16e_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    spec = {
        'status': 'RESEARCH_ONLY_NOT_CLINICAL',
        'unit': 'BioProject cohort, not individual patient',
        'inputs_to_label_free_gate': FINGERPRINT_INPUTS,
        'forbidden_at_prediction_time': ['y_true', 'phenotype', 'AST', 'MIC', 'error_type', 'high_conf_wrong'],
        'score_definition': '0.4*JS(model score histogram)+0.4*JS(panel family histogram)+0.2*mean(abs(gene prevalence differences))',
        'reference_policy': 'E2 other BioProjects only; NOT valid for deployment without a frozen development reference',
        'abstention_policy': 'NONE; does not issue clinical recommendation or reject individuals',
        'known_limitations': [
            'Scores depend on source size and project composition.',
            'A genotype-phenotype label inversion can be invisible without AST labels.',
            'Predictors and carbapenemase panel correlate by design.',
            'E3 cannot be used as independent source-level confirmation without BioProject metadata and matched model reference.',
            'Scores are not calibrated risk probabilities.',
        ],
    }
    with (out/'file16e_research_gate_contract.json').open('w', encoding='utf-8') as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print('[FILE16E] COMPLETE (post-hoc feasibility; NOT a validated reliability alarm)')
    print(json.dumps({k: summary[k] for k in [
        'n_all_sources', 'n_eligible_sources', 'top_k', 'flagged_bioprojects',
        'source_critical_label_free_rank_eligible_sources', 'source_critical_flagged_top_k',
        'source_level_spearman_score_vs_error', 'scope']}, indent=2))
    print(f'[FILE16E] outputs: {out}')

if __name__ == '__main__':
    main()
