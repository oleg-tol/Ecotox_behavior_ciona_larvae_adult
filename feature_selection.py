#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature pre-filtering and PCA for the larval and adult behavioural space.

The same pipeline applies to both stages; only the input feature matrix,
metadata and category definitions differ.

Pipeline
--------
1. impute_missing_by_group : per-group median imputation (no exclusions)
2. log_transform           : reduce right skew in positive metrics
3. cv_filter               : drop metrics with insufficient between-animal variation
4. correlation_filter      : category-aware, two-pass redundancy removal
5. run_pca                 : fit scaler + PCA on reference animals, project the rest
6. n_components_kaiser     : number of retained PCs (eigenvalue > 1)
7. select_retained         : top-loading metrics for display, with family quota

Steps 3-5 use reference animals only (larvae: single-compound concentration
series incl. controls; adults: late post-exposure). Projected animals
(larvae: combination experiment; adults: early post-exposure) are mapped
into the same space without refitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Default thresholds ────────────────────────────────────────────────────────
CV_MIN         = 0.05   # minimum coefficient of variation to keep a metric
CORR_MAX       = 0.75   # within-category Spearman |r| limit
CORR_MAX_CROSS = 0.92   # across-category Spearman |r| limit
KAISER_MIN     = 1.0    # eigenvalue threshold for retained PCs
N_PCS_MIN      = 2
N_PCS_MAX      = 7
N_RETAINED     = 10     # metrics shown in loading / bubble / box plots


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  PER-GROUP MEDIAN IMPUTATION  (no animals or metrics excluded)
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_by_group(df: pd.DataFrame,
                            metrics: list[str],
                            group_cols: list[str]) -> pd.DataFrame:
    """
    Fill missing values with the median of the same group (larvae: Batch +
    Drug_type; adults: Condition), falling back to the overall median.
    Used for the multivariate analysis only; univariate tests use raw values.
    """
    df = df.copy()
    for m in metrics:
        if m not in df.columns:
            continue
        df[m] = df.groupby(group_cols)[m].transform(lambda s: s.fillna(s.median()))
        df[m] = df[m].fillna(df[m].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  LOG TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

def log_transform(df: pd.DataFrame,
                  metrics: list[str],
                  log_metrics: set[str]) -> pd.DataFrame:
    """
    Apply log1p to right-skewed metrics. Metrics with a minimum <= 0 are
    shifted to be positive first. Only metrics listed in log_metrics are affected.
    """
    df = df.copy()
    for m in metrics:
        if m not in log_metrics or m not in df.columns:
            continue
        mn = df[m].min(skipna=True)
        if pd.notna(mn) and mn <= 0:
            df[m] = df[m] - mn + 1e-6
        df[m] = np.log1p(df[m])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  CV FILTER
# ─────────────────────────────────────────────────────────────────────────────

def cv_filter(df: pd.DataFrame,
              metrics: list[str],
              cv_min: float = CV_MIN) -> tuple[list[str], pd.DataFrame]:
    """
    Drop metrics with coefficient of variation (std / |mean|) below cv_min.
    """
    kept  = []
    audit = []
    for m in metrics:
        v  = df[m].dropna()
        mn = v.mean(); sd = v.std(ddof=1)
        cv = sd / abs(mn) if abs(mn) > 1e-9 else 0.0
        if cv >= cv_min:
            kept.append(m)
            audit.append({'metric': m, 'cv': round(cv, 4), 'decision': 'kept'})
        else:
            audit.append({'metric': m, 'cv': round(cv, 4),
                          'decision': f'dropped cv={cv:.3f} < {cv_min}'})
    return kept, pd.DataFrame(audit)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  CORRELATION FILTER  (category-aware, two-pass)
# ─────────────────────────────────────────────────────────────────────────────

def correlation_filter(df: pd.DataFrame,
                       metrics: list[str],
                       category: dict[str, str],
                       seeds: dict[str, list[str]] | None = None,
                       min_per_category: dict[str, int] | None = None,
                       corr_max: float = CORR_MAX,
                       corr_max_cross: float = CORR_MAX_CROSS,
                       enforce_quota: bool = True
                       ) -> tuple[list[str], pd.DataFrame]:
    """
    Greedy redundancy removal based on Spearman correlation.

    Pass 1 (within category): metrics are processed seeds first, then in
    descending order of variance; a metric is dropped if |r| >= corr_max with
    an already-kept metric of the same category.
    Pass 2 (across categories): the first min_per_category metrics of each
    category go first, the rest follow by variance; a metric is dropped if
    |r| >= corr_max_cross with any kept metric, unless its category is still
    below its quota.

    Returns (kept_metrics, audit_df).
    """
    seeds = seeds or {}
    min_per_category = min_per_category or {}
    corr = df[metrics].corr(method='spearman').abs()
    var  = df[metrics].var(ddof=1).sort_values(ascending=False)
    cat  = {m: category.get(m, 'other') for m in metrics}
    cats = list(min_per_category) + [c for c in dict.fromkeys(cat.values())
                                     if c not in min_per_category]
    audit = []

    # Pass 1 -- within category
    reps = {c: [] for c in cats}
    for c in cats:
        members = [m for m in var.index if cat[m] == c]
        order   = [s for s in seeds.get(c, []) if s in members]
        order  += [m for m in members if m not in order]
        for m in order:
            if reps[c]:
                r = corr.loc[m, reps[c]].max()
                if r >= corr_max:
                    p = corr.loc[m, reps[c]].idxmax()
                    audit.append({'metric': m, 'max_r': round(r, 3), 'kept_instead': p,
                                  'decision': f'dropped within-{c} |r|={r:.2f} with {p}'})
                    continue
            reps[c].append(m)

    # Pass 2 -- across categories
    seeded, leftover = [], []
    for c in cats:
        n = min_per_category.get(c, 1)
        seeded += reps[c][:n]; leftover += reps[c][n:]
    leftover.sort(key=lambda m: var[m], reverse=True)

    kept = []
    for m in seeded + leftover:
        if not kept:
            kept.append(m)
            continue
        r = corr.loc[m, kept].max()
        if r < corr_max_cross:
            kept.append(m)
            continue
        p = corr.loc[m, kept].idxmax()
        n_cat = sum(cat[k] == cat[m] for k in kept)
        if enforce_quota and n_cat < min_per_category.get(cat[m], 1):
            kept.append(m)
            audit.append({'metric': m, 'max_r': round(r, 3), 'kept_instead': p,
                          'decision': f'kept by {cat[m]} quota |r|={r:.2f} with {p}'})
        else:
            audit.append({'metric': m, 'max_r': round(r, 3), 'kept_instead': p,
                          'decision': f'dropped cross-category |r|={r:.2f} with {p}'})

    return kept, pd.DataFrame(audit)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  STANDARDISE + PCA  (fit on reference, project all)
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(df: pd.DataFrame,
            metrics: list[str],
            ref_mask: pd.Series
            ) -> tuple[np.ndarray, PCA, StandardScaler]:
    """
    Fit StandardScaler + PCA on reference animals only, then transform all
    animals with the same scaler and loadings (projected animals never refit).

    Returns (Z, pca_model, scaler) where Z is (n_animals, n_components).
    """
    X_ref  = df.loc[ref_mask, metrics].values
    scaler = StandardScaler().fit(X_ref)
    n_comp = min(len(metrics), X_ref.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42).fit(scaler.transform(X_ref))
    Z      = pca.transform(scaler.transform(df[metrics].values))
    return Z, pca, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  NUMBER OF COMPONENTS  (Kaiser)
# ─────────────────────────────────────────────────────────────────────────────

def n_components_kaiser(pca: PCA,
                        eig_min: float = KAISER_MIN,
                        n_min: int = N_PCS_MIN,
                        n_max: int = N_PCS_MAX) -> int:
    """Number of PCs with eigenvalue > eig_min, clipped to [n_min, n_max]."""
    return int(np.clip((pca.explained_variance_ > eig_min).sum(), n_min, n_max))


def pca_summary(pca: PCA) -> pd.DataFrame:
    """
    Variance explained table for all fitted components.
    Columns: PC, eigenvalue, variance_explained, cumulative_variance.
    """
    var = pca.explained_variance_ratio_
    return pd.DataFrame({
        'PC':                  [f'PC{i+1}' for i in range(len(var))],
        'eigenvalue':          pca.explained_variance_,
        'variance_explained':  var,
        'cumulative_variance': np.cumsum(var),
    })


def loadings_table(pca: PCA, metrics: list[str], n_pcs: int) -> pd.DataFrame:
    """
    Feature loadings for the retained PCs, indexed by metric.
    Extra column max_abs = largest |loading| across retained PCs.
    """
    L = pd.DataFrame(pca.components_[:n_pcs].T, index=metrics,
                     columns=[f'PC{i+1}' for i in range(n_pcs)])
    L['max_abs'] = L.abs().max(axis=1)
    return L.sort_values('max_abs', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  RETAINED METRICS  (display set)
# ─────────────────────────────────────────────────────────────────────────────

def select_retained(loadings: pd.DataFrame,
                    n: int = N_RETAINED,
                    family: dict[str, str] | None = None,
                    required: tuple[str, ...] = (),
                    max_per_family: int | None = None) -> list[str]:
    """
    Top-n metrics by max |loading| across retained PCs.

    family / required : guarantee >= 1 metric per required family, replacing
                        the weakest current pick (larvae: quiescent / dwelling /
                        roaming state families; adults: oral / atrial / mantle /
                        activity categories).
    max_per_family    : cap per family, freed slots backfilled by loading
                        from other families (larvae: 1; adults: None).
    """
    family = family or {}
    mag  = loadings['max_abs']
    top  = mag.sort_values(ascending=False).index.tolist()
    kept = top[:n]
    added = []

    for f in required:
        if any(family.get(m) == f for m in kept):
            continue
        cand = [m for m in top if family.get(m) == f]
        if not cand:
            print(f"  [WARN] no '{f}' metric available -- quota not met")
            continue
        weakest = min((m for m in kept if m not in added), key=mag.get)
        kept.remove(weakest); kept.append(cand[0]); added.append(cand[0])

    if max_per_family:
        for f in {family.get(m) for m in kept} - {None}:
            members = sorted((m for m in kept if family.get(m) == f), key=lambda m: -mag[m])
            for m in members[max_per_family:]:
                kept.remove(m)
                back = next((c for c in top if c not in kept and family.get(c) != f), None)
                if back is not None:
                    kept.append(back)

    return sorted(kept, key=lambda m: -mag[m])


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline(df: pd.DataFrame,
                  metrics: list[str],
                  log_metrics: set[str],
                  group_cols: list[str],
                  category: dict[str, str],
                  ref_mask: pd.Series | None = None,
                  seeds: dict[str, list[str]] | None = None,
                  min_per_category: dict[str, int] | None = None,
                  family: dict[str, str] | None = None,
                  required_families: tuple[str, ...] = (),
                  max_per_family: int | None = None,
                  cv_min: float = CV_MIN,
                  corr_max: float = CORR_MAX,
                  corr_max_cross: float = CORR_MAX_CROSS,
                  n_retained: int = N_RETAINED,
                  verbose: bool = True) -> dict:
    """
    Run the full feature-selection and PCA pipeline in one call.

    Parameters
    ----------
    df                : wide DataFrame, one row per animal (reference + projected)
    metrics           : candidate metric column names
    log_metrics       : subset of metrics to log-transform
    group_cols        : columns defining imputation groups
    category          : metric -> category (correlation filter)
    ref_mask          : True for reference animals (PCA fit); default all
    seeds             : category -> priority metrics for the within-category pass
    min_per_category  : category -> minimum metrics kept in the cross-category pass
    family, required_families, max_per_family : see select_retained

    Larvae : group_cols=['Batch', 'Drug_type'],
             min_per_category={'kinematic': 3, 'arena': 2, 'behavioral': 2},
             family = state name for state metrics, required_families=
             ('quiescent', 'dwelling', 'roaming'), max_per_family=1
    Adults : group_cols=['Condition'],
             min_per_category={'oral': 2, 'atrial': 2, 'mantle': 2, 'activity': 1},
             family=category, required_families=('oral', 'atrial', 'mantle', 'activity')

    Returns a dict with keys:
        df_clean  : all animals (no exclusions), imputed + log-transformed, PC scores appended
        metrics   : metrics that survived all filters (PCA input)
        n_pcs     : number of retained PCs (Kaiser)
        retained  : display metrics (top loadings, family quota)
        pca, scaler, loadings, summary
        audit     : dict of audit DataFrames (cv, correlation)
    """
    def _log(msg):
        if verbose: print(msg)

    if ref_mask is None:
        ref_mask = pd.Series(True, index=df.index)

    # 1-2. Impute + log, separately for reference and projected animals
    def _prep(sub):
        sub = impute_missing_by_group(sub, metrics, group_cols)
        return log_transform(sub, metrics, log_metrics)
    parts = [_prep(df[ref_mask])]
    if (~ref_mask).any():
        parts.append(_prep(df[~ref_mask]))
    df  = pd.concat(parts).loc[df.index]
    ref = df[ref_mask]
    _log(f"  Imputation: {len(df)} animals retained (no exclusions), "
         f"{int(ref_mask.sum())} reference / {int((~ref_mask).sum())} projected")

    # 3. CV filter (reference only)
    kept, cv_audit = cv_filter(ref, metrics, cv_min)
    _log(f"  CV filter (>={cv_min}): {len(kept)}/{len(metrics)} metrics kept")

    # 4. Correlation filter (reference only)
    kept, corr_audit = correlation_filter(ref, kept, category, seeds, min_per_category,
                                          corr_max, corr_max_cross)
    _log(f"  Correlation filter (<{corr_max} within, <{corr_max_cross} across): "
         f"{len(kept)} metrics kept")
    if verbose and len(corr_audit):
        for _, r in corr_audit.iterrows():
            print(f"    {r['metric']:30s}  {r['decision']}")

    # 5-6. PCA on reference, project all, Kaiser
    Z, pca, scaler = run_pca(df, kept, ref_mask)
    n_pcs = n_components_kaiser(pca)
    var   = pca.explained_variance_ratio_
    _log(f"  PCA: {n_pcs} PCs (Kaiser), {var[:n_pcs].sum()*100:.1f}% variance; "
         f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%")

    df = df.copy()
    for i in range(n_pcs):
        df[f'PC{i+1}'] = Z[:, i]

    # 7. Retained display metrics
    loadings = loadings_table(pca, kept, n_pcs)
    retained = select_retained(loadings, n_retained, family,
                               required_families, max_per_family)
    _log(f"  Retained {len(retained)} metrics: {retained}")

    return {
        'df_clean': df,
        'metrics':  kept,
        'n_pcs':    n_pcs,
        'retained': retained,
        'pca':      pca,
        'scaler':   scaler,
        'loadings': loadings,
        'summary':  pca_summary(pca),
        'audit':    {'cv': cv_audit, 'correlation': corr_audit},
    }
