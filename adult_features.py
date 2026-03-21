#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morphometric feature extraction for adult Ciona intestinalis.

Pipeline summary
----------------
1. compute_morphometrics     : raw polygon/siphon/mantle areas and widths
2. median_normalise          : centre each individual's time series on 1.0
3. seasonal_decompose_trend  : extract slow contraction trend, suppress noise
4. cleanup_trend             : remove edge artefacts, fill gaps, drop sparse rows
5. zscore_series             : standardise for event detection
6. detect_contractions       : peak detection on z-scored trends
7. classify_contraction      : full vs half based on relative peak height
8. detect_quiescence         : intervals between contraction events
9. unify_states              : majority vote across three signal channels
10. compute_event_features   : rate, IPI, duration per individual
11. compute_state_fractions  : time budget in each unified state
12. compute_amp_speed        : amplitude and contraction/relaxation speed (µm)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose


# ── Physical calibration ──────────────────────────────────────────────────────
TILE_SIZE_UM = 245.0   # 1 normalised unit = 245 µm


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  RAW MORPHOMETRICS
# ─────────────────────────────────────────────────────────────────────────────

def shoelace_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Polygon area via the shoelace formula (fixed landmark order).
    Returns NaN if any coordinate is missing.
    """
    if np.any(np.isnan(xs)) or np.any(np.isnan(ys)):
        return np.nan
    xs2 = np.roll(xs, -1)
    ys2 = np.roll(ys, -1)
    return 0.5 * abs(float(np.sum(xs * ys2 - ys * xs2)))


def polygon_area_shapely(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Full-body polygon area via Shapely (variable landmark count per frame).
    Landmarks are sorted by polar angle before polygon construction to
    avoid self-intersection when DLC landmark order is inconsistent.
    Returns NaN for fewer than 3 valid points.
    """
    from shapely.geometry import Polygon
    valid = np.isfinite(xs) & np.isfinite(ys)
    xv, yv = xs[valid], ys[valid]
    if len(xv) < 3:
        return np.nan
    cx, cy = xv.mean(), yv.mean()
    order  = np.argsort(np.arctan2(yv - cy, xv - cx))
    xv, yv = xv[order], yv[order]
    try:
        poly = Polygon(zip(xv, yv))
        return float(poly.area) if poly.is_valid else np.nan
    except Exception:
        return np.nan


def euclidean_distance_series(x1: np.ndarray, y1: np.ndarray,
                               x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """
    Frame-by-frame distance between two landmarks.
    Used for siphon opening width (dorsal–ventral landmark pair).
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  MEDIAN NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def median_normalise(series: np.ndarray,
                     lo_clip: float,
                     hi_clip: float,
                     smooth_window: int = 5) -> np.ndarray:
    """
    Per-individual median normalisation.

    Divides the time series by the individual's temporal median so that
    values are centred on 1.0, removing between-individual size variation.
    Values outside [lo_clip, hi_clip] are clipped; values at or below
    lo_clip are set to NaN as likely artefacts.  A rolling mean smooths
    residual high-frequency noise.
    """
    s   = pd.to_numeric(pd.Series(series), errors='coerce')
    med = s.median()
    if pd.notna(med) and med != 0:
        s = s / med
    s = s.clip(lower=lo_clip, upper=hi_clip)
    s[s <= lo_clip] = np.nan
    s = s.rolling(window=smooth_window, center=True, min_periods=1).mean()
    return s.to_numpy(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  SEASONAL DECOMPOSITION — TREND EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_trend(series: np.ndarray,
                  period: int = 50) -> np.ndarray | None:
    """
    Extract the slow contraction-relaxation trend via additive seasonal
    decomposition (statsmodels).

    The trend is inverted (* -1) so contraction events appear as positive
    deflections, consistent with the physical expectation that body area
    decreases during contraction.

    Returns None if the series has fewer than 2 * period valid frames,
    as the decomposition is unreliable for short recordings.
    """
    s     = pd.Series(series).dropna()
    if len(s) < 2 * period:
        return None
    try:
        result = seasonal_decompose(s, model='additive', period=period)
        trend  = result.trend * -1
        return trend.reindex(np.arange(len(series))).to_numpy(dtype=float)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  TREND CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def recover_leading_edge(trend: np.ndarray, n_points: int = 25) -> np.ndarray:
    """
    Back-extrapolate the first n_points NaN frames introduced by seasonal
    decomposition using the trend of the subsequent valid frames.
    """
    out   = trend.copy()
    src   = np.arange(n_points, len(out))
    valid = ~np.isnan(out[n_points:])
    if valid.sum() < 2:
        return out
    f = interp1d(src[valid], out[n_points:][valid], kind='linear',
                 fill_value='extrapolate')
    out[:n_points] = f(np.arange(n_points))
    return out


def fill_internal_gaps(trend: np.ndarray, max_gap: int = 200) -> np.ndarray:
    """
    Linear interpolation over internal NaN gaps up to max_gap frames long.
    Leading and trailing NaN regions are left unchanged.
    """
    s = pd.Series(trend.copy())
    last_valid = s.last_valid_index()
    if last_valid is None:
        return trend
    s.iloc[:last_valid + 1] = (s.iloc[:last_valid + 1]
                                .interpolate(method='linear', limit=max_gap))
    return s.to_numpy(dtype=float)


def mark_boundary_frames(trend: np.ndarray) -> np.ndarray:
    """
    Set the first and last valid frame to -1 as a boundary artefact marker.
    Event detection ignores frames with value -1, preventing edge peaks
    introduced by seasonal decomposition from being counted as contractions.
    """
    out   = trend.copy()
    valid = np.where(np.isfinite(out))[0]
    if valid.size > 0:
        out[valid[0]]  = -1
        out[valid[-1]] = -1
    return out


def is_sparse(trend: np.ndarray, min_valid_fraction: float = 0.20) -> bool:
    """Return True if fewer than min_valid_fraction of frames are valid."""
    return float(np.isfinite(trend).mean()) < min_valid_fraction


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Z-SCORE NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def zscore_series(trend: np.ndarray) -> np.ndarray:
    """
    Standardise an individual's cleaned trend to zero mean and unit variance.
    NaN frames are excluded from mean/std computation and preserved as NaN.
    """
    valid = np.isfinite(trend)
    out   = trend.copy()
    mn    = trend[valid].mean()
    sd    = trend[valid].std(ddof=1)
    if sd > 0:
        out[valid] = (trend[valid] - mn) / sd
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6–7  CONTRACTION DETECTION AND CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def smooth_for_detection(signal: np.ndarray,
                          window: int = 20,
                          order:  int = 1) -> np.ndarray:
    """
    Savitzky-Golay smoothing prior to peak detection.
    Applied to the z-scored trend with edge-padding to avoid boundary effects.
    """
    pad = window
    ext = np.pad(signal, pad, mode='edge')
    sm  = savgol_filter(ext, window, order, mode='nearest')
    return sm[pad:-pad]


def detect_contractions(signal: np.ndarray,
                         prominence: float = 0.7,
                         min_distance: int  = 5,
                         min_width: int     = 5
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    """
    Detect contraction peaks in a single z-scored signal channel.

    A contraction is a local maximum in the smoothed trend that exceeds
    a minimum prominence, is separated from other peaks by at least
    min_distance frames, and has a minimum width of min_width frames.

    Returns
    -------
    peaks      : array of peak frame indices
    left_ips   : left interpolated width positions (onset)
    right_ips  : right interpolated width positions (offset)
    prominences: peak prominence values
    binary     : binary event mask (1 = within a contraction, 0 = not)
    """
    # Interpolate NaN, offset to positive, then smooth
    s = pd.Series(signal).interpolate(method='linear',
                                       limit_direction='both').to_numpy()
    mn = np.nanmin(s)
    s  = s - mn if mn < 0 else s + 1
    sm = smooth_for_detection(s)

    T = len(sm)
    peaks, props = find_peaks(sm,
                               prominence=prominence,
                               distance=min_distance,
                               width=min_width)
    proms  = props.get('prominences', np.array([]))
    lefts  = np.floor(props.get('left_ips',  np.array([]))).astype(int)
    rights = np.ceil( props.get('right_ips', np.array([]))).astype(int)

    binary = np.zeros(T, dtype=float)
    for l, r in zip(lefts, rights):
        binary[max(int(l), 0) : min(int(r), T-1) + 1] = 1

    return peaks, lefts, rights, proms, binary


def classify_contraction(peak_value: float,
                          max_peak_value: float,
                          threshold: float = 0.6) -> str:
    """
    Classify a single contraction as 'full' or 'half'.
    A peak is full if its height exceeds threshold * max_peak_value for
    that individual and metric; otherwise half.  This separates strong
    whole-body contractions from partial or siphon-only events.
    """
    return 'full' if peak_value >= threshold * max_peak_value else 'half'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  QUIESCENCE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_quiescence(binary: np.ndarray,
                       smoothed: np.ndarray,
                       threshold: float,
                       tolerance: float = 0.6) -> list[tuple[int, int]]:
    """
    Identify quiescent intervals — frames not in a contraction where the
    signal is below the individual's median (threshold) plus tolerance.

    Returns a list of (start_frame, end_frame) tuples.
    """
    events, start = [], None
    for i, val in enumerate(binary):
        in_q = (val == 0) and (smoothed[i] < threshold + tolerance)
        if in_q and start is None:
            start = i
        elif (not in_q) and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(binary) - 1))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9  UNIFIED STATE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def unify_states(mantle_binary: np.ndarray,
                 oral_binary: np.ndarray,
                 atrial_binary: np.ndarray,
                 mantle_cat: list[str],
                 oral_cat: list[str],
                 atrial_cat: list[str]) -> list[str]:
    """
    Majority-vote unified state across three signal channels per frame.

    Full Contraction : all three channels show a full-contraction event.
    Half Contraction : at least one channel shows any contraction.
    Quiescence       : no channel shows a contraction.

    mantle/oral/atrial_binary : binary event mask arrays (length n_frames)
    mantle/oral/atrial_cat    : per-frame contraction category strings
                                ('full', 'half', or '' for non-event frames)
    """
    n       = len(mantle_binary)
    unified = []

    for t in range(n):
        m_in = mantle_binary[t] == 1
        o_in = oral_binary[t]   == 1
        a_in = atrial_binary[t] == 1

        if (m_in and o_in and a_in and
                mantle_cat[t] == 'full' and
                oral_cat[t]   == 'full' and
                atrial_cat[t] == 'full'):
            unified.append('Full Contraction')
        elif m_in or o_in or a_in:
            unified.append('Half Contraction')
        else:
            unified.append('Quiescence')

    return unified


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10  EVENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_event_features(peak_frames: np.ndarray,
                            durations: np.ndarray,
                            total_minutes: float) -> dict:
    """
    Summary features derived from detected contraction events.

    peak_frames   : array of peak frame indices (sorted)
    durations     : array of event durations (frames)
    total_minutes : total recording duration in minutes

    Returns
    -------
    event_count          : total number of detected events
    rate_per_min         : events per minute
    avg_ipi_min          : mean inter-peak interval in minutes
    ipi_cv               : coefficient of variation of IPI (regularity index)
    avg_duration_frames  : mean event duration in frames
    """
    n      = len(peak_frames)
    rate   = n / total_minutes if total_minutes > 0 else np.nan
    ipis   = np.diff(np.sort(peak_frames)) if n > 1 else np.array([])
    ipi_m  = float(np.mean(ipis))  if ipis.size > 0 else np.nan
    ipi_cv = (float(np.std(ipis, ddof=1) / np.mean(ipis))
              if ipis.size > 1 and np.mean(ipis) > 0 else np.nan)
    return {
        'event_count':         n,
        'rate_per_min':        rate,
        'avg_ipi_min':         ipi_m,
        'ipi_cv':              ipi_cv,
        'avg_duration_frames': float(np.mean(durations)) if len(durations) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11  STATE FRACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_state_fractions(unified_states: list[str]) -> dict:
    """
    Time budget across unified behavioural states.

    unified_states : per-frame state labels from unify_states()

    Returns fractions for Full Contraction, Half Contraction, Quiescence,
    and the overall duty cycle (Full + Half combined).
    """
    total = len(unified_states)
    if total == 0:
        return {}
    n_full  = unified_states.count('Full Contraction')
    n_half  = unified_states.count('Half Contraction')
    n_quiet = unified_states.count('Quiescence')
    return {
        'frac_full_contraction':  n_full  / total,
        'frac_half_contraction':  n_half  / total,
        'frac_quiescence':        n_quiet / total,
        'duty_cycle':             (n_full + n_half) / total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12  AMPLITUDE AND SPEED (physical units)
# ─────────────────────────────────────────────────────────────────────────────

def compute_amp_speed(raw_series: np.ndarray,
                      start_idx: int,
                      peak_idx: int,
                      end_idx: int,
                      dt_s: float,
                      tile_um: float = TILE_SIZE_UM) -> dict:
    """
    Contraction amplitude, contraction speed, and relaxation speed in µm.

    Computed from the median-normalised (non-z-scored) time series so that
    physical units are preserved.  One normalised unit = tile_um micrometres.

    amplitude          : peak-to-trough change over the event window (µm)
    speed_contract     : rate of change from onset to peak (µm s-1)
    speed_relax        : rate of change from peak to offset (µm s-1)
    """
    seg   = raw_series[start_idx : end_idx + 1] * tile_um
    p_rel = max(0, min(peak_idx - start_idx, len(seg) - 1))

    if len(seg) < 2 or np.all(np.isnan(seg)):
        return {'amplitude_um': np.nan,
                'speed_contract_um_s': np.nan,
                'speed_relax_um_s': np.nan}

    amplitude    = float(np.nanmax(seg) - np.nanmin(seg))
    t_contract   = p_rel * dt_s
    speed_c      = (float((seg[p_rel] - seg[0]) / t_contract)
                    if t_contract > 0 else np.nan)
    t_relax      = (len(seg) - 1 - p_rel) * dt_s
    speed_r      = (float((seg[p_rel] - seg[-1]) / t_relax)
                    if t_relax > 0 else np.nan)

    return {'amplitude_um':       amplitude,
            'speed_contract_um_s': speed_c,
            'speed_relax_um_s':    speed_r}

