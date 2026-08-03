#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature pre-filtering and PCA for the larval and adult behavioural space.

The same pipeline applies to both stages; only the input feature matrix
and metadata differ.

Pipeline
--------
1. log_transform         : reduce right skew in strictly positive metrics
2. impute_missing_by_group : per-group median imputation 
3. cv_filter             : drop metrics with insufficient between-animal variation
4. correlation_filter    : drop redundant metrics (keep higher-variance of each pair)
5. standardise           : z-score all surviving metrics
6. run_pca               : scikit-learn PCA on all animals (controls included)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Default thresholds ────────────────────────────────────────────────────────
CV_MIN          = 0.10   # minimum coefficient of variation to keep a metric
CORR_MAX        = 0.70   # maximum Spearman |r| before one metric is dropped
N_PCA_COMPONENTS= 10     # maximum number of PCs to retain


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  LOG TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

def log_transform(df: pd.DataFrame,
                  metrics: list[str],
                  log_metrics: set[str]) -> pd.DataFrame:
    """
    Apply log1p to right-skewed metrics (speed, acceleration, displacement,
    duration).  Negative values are shifted to be positive before transforming.
    Only metrics listed in log_metrics are affected.
    """
    df = df.copy()
    for m in metrics:
        if m not in log_metrics or m not in df.columns:
            continue
        mn = df[m].min(skipna=True)
        if pd.notna(mn) and mn <= 0:
            df[m] = df[m] - mn + 1e-6
        df[m] = np.log1p(df[m].clip(lower=0))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  PER-GROUP MEDIAN IMPUTATION  (no animals or metrics excluded)
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_by_group(df: pd.DataFrame,
                             metrics: list[str],
                             group_col: str) -> pd.DataFrame:
    df = df.copy()
    for m in metrics:
        if m not in df.columns:
            continue
        df[m] = df.groupby(group_col)[m].transform(lambda s: s.fillna(s.median()))
        df[m] = df[m].fillna(df[m].median())
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
# STEP 4  CORRELATION FILTER
# ─────────────────────────────────────────────────────────────────────────────

def correlation_filter(df: pd.DataFrame,
                        metrics: list[str],
                        corr_max: float = CORR_MAX
                        ) -> tuple[list[str], pd.DataFrame]:
    """
    Greedy redundancy removal based on Spearman correlation.

    Metrics are processed in descending order of variance.  A metric is
    kept if its absolute Spearman correlation with every already-kept metric
    is below corr_max; otherwise it is dropped.  This preserves the metric
    with the greatest discriminating power from each correlated group.

    Returns (kept_metrics, dropped_df) where dropped_df logs which metric
    was dropped and which kept metric caused the removal.
    """
    corr_mat  = df[metrics].corr(method='spearman').abs()
    variances = df[metrics].var(ddof=1).sort_values(ascending=False)
    kept      = []
    dropped   = []

    for m in variances.index:
        if not kept:
            kept.append(m)
            continue
        max_r   = corr_mat.loc[m, kept].max()
        if max_r < corr_max:
            kept.append(m)
        else:
            culprit = corr_mat.loc[m, kept].idxmax()
            dropped.append({'metric': m,
                            'max_r':  round(max_r, 3),
                            'kept_instead': culprit,
                            'decision': f'dropped |r|={max_r:.2f} with {culprit}'})

    return kept, pd.DataFrame(dropped)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  STANDARDISE
# ─────────────────────────────────────────────────────────────────────────────

def standardise(df: pd.DataFrame,
                metrics: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """
    Z-score all surviving metrics to zero mean and unit variance.
    Returns (X_scaled, fitted_scaler) where X_scaled is an (n_animals, n_metrics)
    array ready for PCA.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[metrics].values)
    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  PCA
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(X: np.ndarray,
            n_components: int = N_PCA_COMPONENTS
            ) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA on all animals (including controls).

    Returns (Z, pca_model) where Z is (n_animals, n_components).
    """
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    Z      = pca.fit_transform(X)
    return Z, pca


def pca_summary(pca: PCA) -> pd.DataFrame:
    """
    Variance explained table for all fitted components.
    Columns: PC, variance_explained, cumulative_variance.
    """
    var = pca.explained_variance_ratio_
    return pd.DataFrame({
        'PC':                  [f'PC{i+1}' for i in range(len(var))],
        'variance_explained':  var,
        'cumulative_variance': np.cumsum(var),
    })


def loadings_table(pca: PCA, metrics: list[str]) -> pd.DataFrame:
    """
    Feature loadings for the first four PCs, sorted by magnitude on PC1-2.
    Columns: metric, PC1_loading, PC2_loading, PC3_loading, PC4_loading,
             magnitude_PC12, magnitude_PC34.
    """
    nc  = pca.components_.shape[0]
    def _comp(i):
        return pca.components_[i] if i < nc else np.full(len(metrics), np.nan)

    return pd.DataFrame({
        'metric':         metrics,
        'PC1_loading':    _comp(0),
        'PC2_loading':    _comp(1),
        'PC3_loading':    _comp(2),
        'PC4_loading':    _comp(3),
        'magnitude_PC12': np.sqrt(_comp(0)**2 + _comp(1)**2),
        'magnitude_PC34': np.sqrt(_comp(2)**2 + _comp(3)**2),
    }).sort_values('magnitude_PC12', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline(df: pd.DataFrame,
                  metrics: list[str],
                  log_metrics: set[str],
                  group_col: str,
                  cv_min:    float = CV_MIN,
                  corr_max:  float = CORR_MAX,
                  n_pca_components: int = N_PCA_COMPONENTS,
                  verbose: bool = True
                  ) -> dict:
    """
    Run the full feature-selection and PCA pipeline in one call.

    Parameters
    ----------
    df              : wide DataFrame, one row per animal
    metrics         : candidate metric column names
    log_metrics     : subset of metrics to log-transform
    group_col       : column identifying treatment groups (used for
                      per-group median imputation)
    verbose         : print progress and audit information

    Returns a dict with keys:
        df_clean    : full animal DataFrame (no exclusions) with PC scores appended
        metrics     : list of metrics that survived all filters
        pca         : fitted sklearn PCA object
        scaler      : fitted StandardScaler
        audit       : dict of audit DataFrames (cv, correlation)
    """
    def _log(msg):
        if verbose: print(msg)

    n0 = len(df)

    # 1. Log transform
    df = log_transform(df, metrics, log_metrics)

    # 2. Per-group median imputation -- no animals or metrics dropped
    df = impute_missing_by_group(df, metrics, group_col)
    _log(f"  Imputation: {n0} animals retained (no exclusions), "
         f"{len(metrics)} candidate metrics imputed per-group where needed")

    # 3. CV filter
    metrics, cv_audit = cv_filter(df, metrics, cv_min)
    _log(f"  CV filter (>{cv_min}): {len(metrics)} metrics kept")

    # 4. Correlation filter
    metrics, corr_audit = correlation_filter(df, metrics, corr_max)
    _log(f"  Correlation filter (<{corr_max}): {len(metrics)} metrics kept")
    if verbose and len(corr_audit):
        for _, r in corr_audit.iterrows():
            print(f"    {r['metric']:30s}  {r['decision']}")

    # 5. Standardise
    X, scaler = standardise(df, metrics)

    # 6. PCA
    Z, pca = run_pca(X, n_pca_components)
    var    = pca.explained_variance_ratio_
    _log(f"  PCA: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  "
         f"PC1+2={var[:2].sum()*100:.1f}%")

    # Attach PC scores to dataframe
    df = df.copy()
    for i in range(Z.shape[1]):
        df[f'PC{i+1}'] = Z[:, i]

    return {
        'df_clean': df,
        'metrics':  metrics,
        'pca':      pca,
        'scaler':   scaler,
        'audit':    {'cv': cv_audit, 'correlation': corr_audit},
    }
