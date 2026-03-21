#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared data loading utilities for larval and adult Ciona experiments.
Parses DeepLabCut output CSVs, assembles coordinate DataFrames,
and applies initial quality filters.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Quality filter parameters ─────────────────────────────────────────────────

DLC_LIKELIHOOD_THRESHOLD = 0.99   # keypoints below this are set to NaN
IQR_OUTLIER_FACTOR       = 2.0    # IQR multiplier for spatial outlier removal
MAX_JUMP_PX              = 100    # max allowed inter-frame displacement in pixels


# ── DLC file parsing ──────────────────────────────────────────────────────────

def read_dlc_csv(filepath: Path) -> pd.DataFrame:
    """
    Parse a DeepLabCut output CSV (single- or multi-animal project).

    Returns a long DataFrame:
        ExperimentID | Individuals | Bodyparts | Coordinates | t0 … tN
    """
    experiment_id = filepath.stem
    raw = None
    for n_header_rows in (4, 3):
        try:
            df = pd.read_csv(filepath, header=list(range(n_header_rows)),
                             index_col=0)
            if isinstance(df.columns, pd.MultiIndex):
                raw = df
                break
        except Exception:
            continue

    if raw is None:
        raise ValueError(f"{filepath.name}: could not parse MultiIndex headers.")

    nlevels = raw.columns.nlevels
    if nlevels >= 4:
        individuals = [c[1] for c in raw.columns]
        bodyparts   = [c[2] for c in raw.columns]
        coords      = [c[3] for c in raw.columns]
    else:
        individuals = ['animal'] * raw.shape[1]
        bodyparts   = [c[1] for c in raw.columns]
        coords      = [c[2] for c in raw.columns]

    meta = pd.DataFrame({
        'ExperimentID': experiment_id,
        'Individuals':  individuals,
        'Bodyparts':    bodyparts,
        'Coordinates':  coords,
    })
    keep = meta['Coordinates'].str.lower() != 'likelihood'
    meta = meta.loc[keep].reset_index(drop=True)
    data = raw.iloc[:, keep.values].to_numpy()

    frame_cols = [f't{i}' for i in range(data.shape[0])]
    return pd.concat([meta, pd.DataFrame(data.T, columns=frame_cols)], axis=1)


# ── Coordinate quality filters ────────────────────────────────────────────────

def apply_likelihood_mask(coords: np.ndarray,
                          likelihood: np.ndarray,
                          threshold: float = DLC_LIKELIHOOD_THRESHOLD
                          ) -> np.ndarray:
    
    out = coords.copy().astype(float)
    out[likelihood < threshold] = np.nan
    return out


def iqr_outlier_removal(coords: np.ndarray,
                        factor: float = IQR_OUTLIER_FACTOR) -> np.ndarray:
    q25 = np.nanpercentile(coords, 25, axis=0)
    q75 = np.nanpercentile(coords, 75, axis=0)
    iqr = q75 - q25
    lo  = q25 - factor * iqr
    hi  = q75 + factor * iqr
    return np.where((coords < lo) | (coords > hi), np.nan, coords)


def filter_large_jumps(coords_x: np.ndarray,
                       coords_y: np.ndarray,
                       max_px: float = MAX_JUMP_PX
                       ) -> tuple[np.ndarray, np.ndarray]:
    
    dist = np.sqrt(np.diff(coords_x, axis=0)**2 + np.diff(coords_y, axis=0)**2)
    mask = np.insert(dist <= max_px, 0, True, axis=0)
    return (np.where(mask, coords_x, np.nan),
            np.where(mask, coords_y, np.nan))


def calculate_com(coords_x: np.ndarray,
                  coords_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass per frame as the mean over all valid bodyparts."""
    return np.nanmean(coords_x, axis=1), np.nanmean(coords_y, axis=1)


# ── Folder-based data loader ──────────────────────────────────────────────────

def load_experiment_folder(base_path: Path,
                            fps_source: int,
                            resample_rate: int,
                            keep_last_min: float,
                            file_pattern: str = '*_filtered.csv'
                            ) -> tuple[list, list, list, list, pd.DataFrame, pd.DataFrame]:
    
    bpx_list, bpy_list, comx_list, comy_list = [], [], [], []
    exp_rows, load_log = [], []
    file_count = 0

    fps_eff        = fps_source / resample_rate
    frames_to_keep = int(keep_last_min * 60 * fps_eff)

    for dirpath, _, filenames in sorted_walk(base_path):
        depth = Path(dirpath).resolve().relative_to(
            Path(base_path).resolve()).parts
        if len(depth) > 1:
            continue

        for filename in sorted(filenames):
            if not filename.endswith('_filtered.csv'):
                continue

            file_path = Path(dirpath) / filename
            file_count += 1

            try:
                data = pd.read_csv(file_path, low_memory=False)
            except Exception as e:
                print(f"  [skip] {filename}: {e}")
                continue

            cx = data.iloc[2:, 1::3].astype(float).values
            cy = data.iloc[2:, 2::3].astype(float).values
            lk = data.iloc[2:, 3::3].astype(float).values

            cx = apply_likelihood_mask(cx, lk)
            cy = apply_likelihood_mask(cy, lk)
            cx = iqr_outlier_removal(cx)
            cy = iqr_outlier_removal(cy)
            cx, cy = filter_large_jumps(cx, cy)
            com_x, com_y = calculate_com(cx, cy)

            drug_type, treatment, exp_id = parse_folder_metadata(
                file_path, base_path)

            # Resample then keep last N minutes
            def _resample_keep(arr):
                arr = arr[::resample_rate]
                return arr[-frames_to_keep:] if len(arr) >= frames_to_keep else arr

            cx_r  = _resample_keep(cx)
            cy_r  = _resample_keep(cy)
            cx_r_com = _resample_keep(com_x)
            cy_r_com = _resample_keep(com_y)

            array_index = len(bpx_list)
            bpx_list.append(pd.DataFrame(
                cx_r, columns=[f"BP_{i+1}" for i in range(cx_r.shape[1])]))
            bpy_list.append(pd.DataFrame(
                cy_r, columns=[f"BP_{i+1}" for i in range(cy_r.shape[1])]))
            comx_list.append(cx_r_com)
            comy_list.append(cy_r_com)

            for _ in range(len(cx_r_com)):
                exp_rows.append({'Experiment_ID': exp_id,
                                 'Drug_type':     drug_type,
                                 'Treatment':     treatment,
                                 'Filepath':      str(file_path)})

            load_log.append({'array_index':   array_index,
                             'file_count':    file_count,
                             'filename':      filename,
                             'Experiment_ID': exp_id,
                             'Drug_type':     drug_type,
                             'Treatment':     treatment,
                             'n_raw_frames':  len(com_x),
                             'n_resampled':   len(cx_r_com)})

    return (bpx_list, bpy_list, comx_list, comy_list,
            pd.DataFrame(exp_rows), pd.DataFrame(load_log))


# ── Metadata helpers ──────────────────────────────────────────────────────────

def parse_folder_metadata(file_path: Path,
                           data_root: Path) -> tuple[str, str, str]:
    """
    Extract drug_type, treatment, and experiment_id from the file path.
    Assumes structure: data_root / <drug_type> / <experiment>_filtered.csv
    """
    rel   = file_path.resolve().relative_to(data_root.resolve())
    parts = rel.parts
    if len(parts) == 2:
        return parts[0], parts[0], file_path.stem
    return "Unknown", "Unknown", file_path.stem


def make_comp_id(df: pd.DataFrame) -> pd.Series:
    
    return df["Drug_type"] + "__" + df["Treatment"] + "__" + df["Experiment_ID"]


def build_exp_meta(load_log_df: pd.DataFrame,
                   res_comx: list) -> pd.DataFrame:
    
    rows, cumulative = [], 0
    for _, log_row in load_log_df.iterrows():
        i = int(log_row['array_index'])
        n = len(res_comx[i])
        rows.append({
            'array_index':   i,
            'Experiment_ID': log_row['Experiment_ID'],
            'Drug_type':     log_row['Drug_type'],
            'Treatment':     log_row['Treatment'],
            'start_frame':   cumulative,
            'end_frame':     cumulative + n,
            'n_frames':      n,
        })
        cumulative += n
    return pd.DataFrame(rows)


# ── os.walk with sorted directory traversal ───────────────────────────────────

import os

def sorted_walk(top):
    
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames.sort()
        yield dirpath, dirnames, sorted(filenames)