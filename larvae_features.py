#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioural feature extraction for Ciona intestinalis larvae.

Features are grouped into three categories:
  - Kinematic  : how the animal moves (speed, acceleration, body shape)
  - Arena      : how the animal uses the available space
  - State      : time budget across quiescent / dwelling / roaming states

All functions operate on per-experiment arrays and return 1-D arrays
of length equal to the number of (resampled) frames for that experiment.
Wide DataFrames are assembled downstream in the pipeline script.
"""

import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ─────────────────────────────────────────────────────────────────────────────
# KINEMATIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_speed(com_x: np.ndarray,
                  com_y: np.ndarray,
                  dt: float,
                  px_to_mm: float) -> np.ndarray:
    """
    Instantaneous swimming speed (mm s-1).
    Frame-to-frame Euclidean displacement of the centre of mass,
    converted to mm and divided by the inter-frame interval.
    First frame is NaN (no prior position).
    """
    dx  = np.diff(com_x)
    dy  = np.diff(com_y)
    spd = (np.sqrt(dx**2 + dy**2) * px_to_mm) / (dt * 1000)
    return np.insert(spd, 0, np.nan)


def compute_acceleration(speed: np.ndarray, dt: float) -> np.ndarray:
    """
    Absolute linear acceleration (mm s-2).
    Rate of change of speed between consecutive frames.
    First frame is NaN.
    """
    return np.insert(np.abs(np.diff(speed)) / dt, 0, np.nan)


def compute_heading(com_x: np.ndarray,
                    com_y: np.ndarray) -> np.ndarray:
    """
    Frame-to-frame heading angle (radians).
    Angle of the velocity vector relative to the positive x-axis.
    Returns an array of length n_frames - 1.
    """
    h = np.full(com_x.size - 1, np.nan)
    v = (np.isfinite(com_x[:-1]) & np.isfinite(com_x[1:]) &
         np.isfinite(com_y[:-1]) & np.isfinite(com_y[1:]))
    h[v] = np.arctan2((com_y[1:] - com_y[:-1])[v],
                      (com_x[1:] - com_x[:-1])[v])
    return h


def compute_omega(heading: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Angular velocity (deg s-1).
    Signed and absolute change in heading per unit time, wrapped to [-180, 180].
    Returns (signed_omega, abs_omega), each of length equal to heading.
    """
    om = np.full_like(heading, np.nan)
    for i in range(1, len(heading)):
        if np.isfinite(heading[i]) and np.isfinite(heading[i-1]):
            dth   = heading[i] - heading[i-1]
            dth   = (dth + math.pi) % (2 * math.pi) - math.pi
            om[i] = (dth * 180.0 / math.pi) / dt
    return om, np.abs(om)


def compute_angular_acceleration(omega_signed: np.ndarray,
                                  dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Angular acceleration (deg s-2).
    Rate of change of angular velocity between consecutive frames.
    Returns (signed_angacc, abs_angacc).
    """
    a = np.full_like(omega_signed, np.nan)
    for i in range(1, len(omega_signed)):
        if np.isfinite(omega_signed[i]) and np.isfinite(omega_signed[i-1]):
            a[i] = (omega_signed[i] - omega_signed[i-1]) / dt
    return a, np.abs(a)


def compute_curvature(skel_x: np.ndarray,
                      skel_y: np.ndarray,
                      savgol_window: int = 15,
                      savgol_order:  int = 2) -> np.ndarray:
    """
    Mean absolute curvature of the body midline (a.u.).

    Curvature is computed frame by frame using the Savitzky-Golay derivative
    method applied only to the valid (non-NaN) keypoints in that frame.
    This prevents tracking dropouts from contaminating neighbouring frames
    with zeroed values.  The filter is applied per-experiment so it never
    crosses animal boundaries.

    skel_x, skel_y : (n_frames, n_keypoints)
    Returns : 1-D array of length n_frames
    """
    n_frames, n_bps = skel_x.shape
    curvatures = np.full((n_frames, n_bps - 2), np.nan)
    eps = 1e-6

    for t in range(n_frames):
        vm = np.isfinite(skel_x[t]) & np.isfinite(skel_y[t])
        if vm.sum() <= savgol_window:
            continue
        vx  = skel_x[t, vm]
        vy  = skel_y[t, vm]
        dx  = savgol_filter(vx, savgol_window, savgol_order, deriv=1, mode='nearest')
        dy  = savgol_filter(vy, savgol_window, savgol_order, deriv=1, mode='nearest')
        ddx = savgol_filter(vx, savgol_window, savgol_order, deriv=2, mode='nearest')
        ddy = savgol_filter(vy, savgol_window, savgol_order, deriv=2, mode='nearest')
        k   = (dx[:-2]*ddy[2:] - dy[:-2]*ddx[2:]) / (dx[:-2]**2 + dy[:-2]**2 + eps)**1.5
        vi  = np.where(vm)[0]
        curvatures[t, vi[:-2]] = k

    return np.nanmean(np.abs(curvatures), axis=1)


def compute_quirkiness(skel_x: np.ndarray,
                       skel_y: np.ndarray,
                       head_range: int = 10,
                       tail_range: int = 10) -> np.ndarray:
    """
    Total angular deviation of the body axis from a straight head-to-tail line (a.u.).
    Quantifies body sinuosity independent of travel direction.

    The head centroid is the mean of the first head_range keypoints;
    the tail centroid is the mean of the last tail_range keypoints.
    Deviation at each inter-keypoint segment is wrapped to [-pi, pi].

    skel_x, skel_y : (n_frames, n_keypoints)
    Returns : 1-D array of length n_frames
    """
    n_frames = skel_x.shape[0]
    quirkiness = np.full(n_frames, np.nan)

    for t in range(n_frames):
        vm = np.isfinite(skel_x[t]) & np.isfinite(skel_y[t])
        if vm.sum() <= max(head_range, tail_range):
            continue
        vx = skel_x[t, vm]
        vy = skel_y[t, vm]
        hc = np.array([vx[:head_range].mean(),  vy[:head_range].mean()])
        tc = np.array([vx[-tail_range:].mean(), vy[-tail_range:].mean()])
        ax_angle = np.arctan2(tc[1] - hc[1], tc[0] - hc[0])
        dev = np.arctan2(np.diff(vy), np.diff(vx)) - ax_angle
        dev = (dev + np.pi) % (2 * np.pi) - np.pi
        quirkiness[t] = np.sum(np.abs(dev))

    return quirkiness


# ─────────────────────────────────────────────────────────────────────────────
# ARENA FEATURES  (sliding-window, 3-second default)
# ─────────────────────────────────────────────────────────────────────────────

def compute_tortuosity(com_x: np.ndarray,
                       com_y: np.ndarray,
                       window_frames: int,
                       min_disp_px: float = 5.0) -> np.ndarray:
    """
    Path tortuosity over a sliding window (dimensionless).
    Ratio of total path length to straight-line displacement within each window.
    Windows where net displacement < min_disp_px are set to NaN to avoid
    division noise when the animal is nearly stationary.
    Value of 1 = perfectly straight; higher = more convoluted path.
    """
    n   = com_x.size
    out = np.full(n, np.nan)
    for i in range(n - window_frames + 1):
        bx = com_x[i:i+window_frames]
        by = com_y[i:i+window_frames]
        if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(by))):
            continue
        path = float(np.sum(np.hypot(np.diff(bx), np.diff(by))))
        disp = float(math.hypot(bx[-1]-bx[0], by[-1]-by[0]))
        if disp >= min_disp_px and path > 0:
            out[i] = path / disp
    return out


def compute_path_entropy(com_x: np.ndarray,
                         com_y: np.ndarray,
                         window_frames: int) -> np.ndarray:
    """
    Trajectory complexity via SVD entropy (bits), sliding window.
    The 2-D trajectory within each window is normalised to zero mean and
    unit variance, decomposed by SVD, and the entropy of the singular-value
    distribution is returned.  High entropy = irregular multi-directional
    movement; low entropy = linear or circular trajectory.
    """
    n   = com_x.size
    out = np.full(n, np.nan)
    for i in range(n - window_frames + 1):
        bx = com_x[i:i+window_frames]
        by = com_y[i:i+window_frames]
        if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(by))):
            continue
        sx = np.std(bx); sy = np.std(by)
        if sx == 0 or sy == 0:
            continue
        Z = np.vstack([(bx - bx.mean())/sx, (by - by.mean())/sy])
        try:
            s = np.linalg.svd(Z, full_matrices=False, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        s_sum = s.sum()
        if s_sum <= 0 or not np.isfinite(s_sum):
            continue
        p      = s / s_sum
        out[i] = float(-np.sum(p * np.log2(p + 1e-10)))
    return out


def compute_net_displacement(com_x: np.ndarray,
                             com_y: np.ndarray,
                             window_frames: int) -> np.ndarray:
    """
    Net (straight-line) displacement within a sliding window (pixels).
    Separates translocation from in-place oscillation.
    """
    n   = com_x.size
    out = np.full(n, np.nan)
    for i in range(n - window_frames + 1):
        bx = com_x[i:i+window_frames]
        by = com_y[i:i+window_frames]
        if np.all(np.isfinite(bx)) and np.all(np.isfinite(by)):
            out[i] = math.hypot(bx[-1]-bx[0], by[-1]-by[0])
    return out


def compute_arena_metrics(com_x: np.ndarray,
                          com_y: np.ndarray,
                          window_frames: int) -> dict[str, np.ndarray]:
    """
    Spatial exploration metrics within a sliding window.

    r_gyr      : radius of gyration — RMS distance from trajectory centroid.
                 Large = wide-ranging exploration.
    ringness   : CV of distance from centroid. High = ring-like alternation
                 between central and peripheral space.
    anisotropy : ratio of largest to smallest eigenvalue of the position
                 covariance matrix.  High = strongly directional movement.
    winding    : cumulative angular rotations of the velocity vector.
                 High = circular or spiral swimming.

    Returns a dict of arrays, each of length n_frames.
    """
    n = com_x.size
    out = {k: np.full(n, np.nan)
           for k in ('r_gyr', 'ringness', 'anisotropy', 'winding')}

    for i in range(n - window_frames + 1):
        xw = com_x[i:i+window_frames]
        yw = com_y[i:i+window_frames]
        m  = np.isfinite(xw) & np.isfinite(yw)
        if m.sum() < 4:
            continue
        xv, yv   = xw[m], yw[m]
        dx_, dy_ = xv - xv.mean(), yv - yv.mean()
        r        = np.hypot(dx_, dy_)

        out['r_gyr'][i]    = float(np.sqrt(np.mean(r**2)))
        rm, rs             = r.mean(), r.std(ddof=0)
        out['ringness'][i] = float(rs / rm) if rm > 0 else np.nan

        try:
            w, _ = np.linalg.eig(np.cov(np.vstack([dx_, dy_])))
            w    = np.sort(np.real(w))[::-1]
            if w.size >= 2 and w[1] > 0:
                out['anisotropy'][i] = float(w[0] / w[1])
        except Exception:
            pass

        x1_, x2_ = xw[:-1], xw[1:]
        y1_, y2_ = yw[:-1], yw[1:]
        ok = np.isfinite(x1_) & np.isfinite(x2_) & np.isfinite(y1_) & np.isfinite(y2_)
        if ok.sum() > 0:
            theta = np.full_like(x1_, np.nan)
            theta[ok] = np.arctan2((y2_-y1_)[ok], (x2_-x1_)[ok])
            out['winding'][i] = float(
                abs(_sum_unwrapped_angles(theta)) / (2 * np.pi))

    return out


def compute_wall_fraction(com_x: np.ndarray,
                          com_y: np.ndarray,
                          arena_cx: float,
                          arena_cy: float,
                          arena_r: float,
                          wall_band: tuple[float, float] = (0.0, 0.35),
                          window_frames: int = 1) -> np.ndarray:
    """
    Fraction of time spent in the outer wall_band of the arena, per frame.
    wall_band = (inner_edge, outer_edge) as proportions of the arena radius.
    Thigmotaxis (wall-hugging) elevates this metric.
    """
    edge_norm = (arena_r - np.hypot(com_x - arena_cx, com_y - arena_cy)
                 ) / max(arena_r, 1e-9)
    lo, hi    = wall_band
    in_band   = (edge_norm >= lo) & (edge_norm <= hi) & np.isfinite(edge_norm)
    return in_band.astype(float)


def _sum_unwrapped_angles(theta: np.ndarray) -> float:
    """Sum of unwrapped angle differences across contiguous finite segments."""
    m   = np.isfinite(theta).astype(int)
    p   = np.concatenate([[0], m, [0]])
    d   = np.diff(p)
    tot = 0.0
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        seg = theta[s:e]
        if seg.size >= 2:
            tot += np.unwrap(seg)[-1] - np.unwrap(seg)[0]
    return float(tot)


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOURAL STATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_roaming_threshold(speed_df: pd.DataFrame,
                               control_keywords: list[str],
                               quiescence_threshold: float,
                               min_valid_fraction: float = 0.2) -> float:
    """
    Roaming threshold for one experimental batch (mm s-1).

    Pooled median of active frames (speed >= quiescence_threshold) across all
    control animals in this batch.  Anchors the threshold to the spontaneous
    activity level of unexposed animals in the same recording session,
    removing inter-batch speed variation as a confound.

    Falls back to the global median of all frames if no active frames are found.
    """
    time_cols = [c for c in speed_df.columns if c.startswith('Time_')]
    is_ctrl   = speed_df['Treatment'].str.lower().apply(
        lambda x: any(kw in x for kw in control_keywords))
    ctrl_df   = speed_df[is_ctrl] if is_ctrl.any() else speed_df

    speeds = ctrl_df[time_cols].apply(pd.to_numeric, errors='coerce')
    keep   = speeds.notna().mean(axis=1) >= min_valid_fraction
    flat   = speeds[keep].to_numpy().ravel()
    flat   = flat[np.isfinite(flat)]

    if flat.size == 0:
        return np.nan

    active = flat[flat >= quiescence_threshold]
    return float(np.median(active)) if active.size > 0 else float(np.median(flat))


def classify_states(speed: np.ndarray,
                    quiescence_thr: float,
                    roaming_thr: float) -> np.ndarray:
    """
    Assign each frame to quiescent (0), dwelling (1), or roaming (2).

    NaN speed frames remain NaN.  Isolated single-frame state flips flanked
    by the same state on both sides are smoothed to remove jitter without
    altering bout boundaries.
    """
    state    = np.full_like(speed, np.nan)
    valid    = np.isfinite(speed)
    roam_thr = roaming_thr if np.isfinite(roaming_thr) else np.inf

    state[valid & (speed <  quiescence_thr)]                          = 0
    state[valid & (speed >= roam_thr)]                                = 2
    state[valid & (speed >= quiescence_thr) & (speed < roam_thr)]    = 1

    for i in range(1, len(state) - 1):
        if (np.isfinite(state[i-1]) and np.isfinite(state[i]) and
                np.isfinite(state[i+1])):
            if state[i] != state[i-1] and state[i-1] == state[i+1]:
                state[i] = state[i-1]

    return state


def detect_bouts(state: np.ndarray,
                 dt: float,
                 min_bout_frames: int = 5,
                 max_gap_frames:  int = 2) -> list[dict]:
    """
    Identify contiguous bouts from a state array.

    Small NaN gaps up to max_gap_frames within the same state are bridged.
    Only bouts of at least min_bout_frames are returned.

    Returns a list of dicts with keys:
        start, end, duration_frames, duration_seconds, state, state_name
    """
    state  = np.asarray(state, float)
    valid  = np.isfinite(state)
    state_f = state.copy()

    for i in range(1, len(state) - max_gap_frames):
        if not valid[i]:
            for j in range(i+1, min(i + max_gap_frames + 1, len(state))):
                if valid[j]:
                    if state[i-1] == state[j]:
                        state_f[i:j] = state[i-1]
                        valid[i:j]   = True
                    break

    si     = np.full_like(state_f, -1, int)
    vm     = np.isfinite(state_f)
    si[vm] = np.clip(state_f[vm].astype(int), 0, 2)

    bounds = [0] + [i for i in range(1, len(si)) if si[i] != si[i-1]] + [len(si)]
    names  = {0: 'quiescent', 1: 'dwelling', 2: 'roaming'}
    bouts  = []

    for s, e in zip(bounds[:-1], bounds[1:]):
        dur = e - s
        bs  = si[s]
        if dur >= min_bout_frames and 0 <= bs <= 2:
            bouts.append({'start':            int(s),
                          'end':              int(e),
                          'duration_frames':  int(dur),
                          'duration_seconds': float(dur * dt),
                          'state':            bs,
                          'state_name':       names[bs]})
    return bouts


def state_summary(state: np.ndarray,
                  bouts: list[dict],
                  dt: float) -> dict:
    """
    Compute per-animal state statistics from a classified state array
    and its detected bouts.

    Returns a flat dict of:
      pct_quiescent / dwelling / roaming / active,
      n_bouts and mean/median/max bout duration for each state,
      n_transitions, transition_rate_per_min.
    """
    n_valid = int(np.sum(np.isfinite(state)))
    if n_valid == 0:
        return {}

    nq = int(np.sum(state == 0))
    nd = int(np.sum(state == 1))
    nr = int(np.sum(state == 2))
    tt = n_valid * dt

    def _bout_stats(bl):
        if not bl:
            return {'n': 0, 'mean': np.nan, 'median': np.nan, 'max': np.nan}
        d = [b['duration_seconds'] for b in bl]
        return {'n': len(bl), 'mean': float(np.mean(d)),
                'median': float(np.median(d)), 'max': float(np.max(d))}

    qs = _bout_stats([b for b in bouts if b['state'] == 0])
    ds = _bout_stats([b for b in bouts if b['state'] == 1])
    rs = _bout_stats([b for b in bouts if b['state'] == 2])
    nt = max(len(bouts) - 1, 0)

    return {
        'pct_quiescent':              100 * nq / n_valid,
        'pct_dwelling':               100 * nd / n_valid,
        'pct_roaming':                100 * nr / n_valid,
        'pct_active':                 100 * (nd + nr) / n_valid,
        'quiescent_n_bouts':          qs['n'],
        'quiescent_mean_duration_s':  qs['mean'],
        'quiescent_median_duration_s':qs['median'],
        'quiescent_max_duration_s':   qs['max'],
        'dwelling_n_bouts':           ds['n'],
        'dwelling_mean_duration_s':   ds['mean'],
        'dwelling_median_duration_s': ds['median'],
        'dwelling_max_duration_s':    ds['max'],
        'roaming_n_bouts':            rs['n'],
        'roaming_mean_duration_s':    rs['mean'],
        'roaming_median_duration_s':  rs['median'],
        'roaming_max_duration_s':     rs['max'],
        'n_transitions':              nt,
        'transition_rate_per_min':    nt / (tt / 60) if tt > 0 else 0.0,
    }
