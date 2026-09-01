#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Supress Warnings
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.round_ = np.round

import xarray as xr
from herbie import Herbie
import metpy.calc as mpcalc
import metpy.constants as const
from metpy.calc import saturation_vapor_pressure, relative_humidity_from_mixing_ratio, mixing_ratio_from_relative_humidity
from metpy.units import units
from utils import interpolate_var,get_dx_dy,get_mm_files,compute_thw,haversine,extract_within_radius,rotate_grid
import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from mpi4py import MPI
from sklearn.cluster import DBSCAN
import pickle
import os, time, zlib
import xml.etree.ElementTree as ET
import requests
import rasterio
from rasterio.warp import transform as rio_transform
from pathlib import Path
from functools import lru_cache
from netCDF4 import num2date,date2num

DEFAULT_PRESSURE_LEVELS = np.array([1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0])

def grib_key(date_init, fxx=0, model='rap'):
    # Herbie fxx=0 for analysis in your code
    return f"{model}_{date_init:%Y%m%d%H}_f{int(fxx):02d}"

def grib_owner(key, size):
    # stable, fast, deterministic
    return zlib.adler32(key.encode("utf-8")) % max(size, 1)

def ready_path_for(key):
    # Sentinels must live on a shared (parallel) filesystem visible to all MPI ranks.
    # On Schooner /tmp is node-local, so we default to $SCRATCH/grib_ready which IS
    # shared.  Override with GRIB_READY_DIR env var if needed.
    _scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'mpiuser')}")
    default_dir = os.path.join(_scratch, "grib_ready")
    d = Path(os.environ.get("GRIB_READY_DIR", default_dir))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.done"


def failed_path_for(key):
    """Shared sentinel for GRIB times known to be unavailable or failed.

    This prevents non-owner ranks from waiting several minutes for a file that
    rank 0 or the owner already proved unavailable.
    """
    _scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'mpiuser')}")
    default_dir = os.path.join(_scratch, "grib_ready")
    d = Path(os.environ.get("GRIB_READY_DIR", default_dir))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.failed"


def model_for_date(date_init):
    return "rap" if date_init.year > 2012 else "ruc"


def herbie_pressure_levels(H, max_levels=8):
    if getattr(H, 'grib', None) is None:
        return DEFAULT_PRESSURE_LEVELS.copy()
    try:
        df = H.index_as_dataframe
        if callable(df):
            df = df()
    except Exception:
        return DEFAULT_PRESSURE_LEVELS.copy()
    levels = []
    for val in np.unique(df.level.values):
        if isinstance(val, str) and ('mb' in val) and ('-' not in val):
            try:
                levels.append(float(val.split()[0]))
            except Exception:
                continue
    if not levels:
        return DEFAULT_PRESSURE_LEVELS.copy()
    return np.sort(np.asarray(levels, dtype=float))[::-1][:max_levels]


def parse_args():
    p = argparse.ArgumentParser(description='Build RAP/RUC base-state pickle products with optional diagnostics.')
    p.add_argument('--dir-transects', default='../data/', help='Directory containing transect_times.csv.')
    p.add_argument('--tracking-dir', default='../data/storm_tracks/', help='Storm track directory.')
    p.add_argument('--download-pth', default='../data/RAP/', help='Directory for RAP/RUC downloads.')
    p.add_argument('--mesonet-dir', default=None,
                   help='Optional mesonet NetCDF archive used to derive analysis-window mean elevation Z directly from the same observations used in the mesonet workflow.')
    p.add_argument('--mesonet-pickle', default=None,
                   help='Optional final mobile mesonet pickle used to derive analysis-time mean elevation Z directly from the reconciled observations.')
    p.add_argument('--time-to-space', type=int, default=10, choices=[5, 10], help='Time-to-space conversion in minutes.')
    p.add_argument('--modes', nargs='+', default=['T', 'N'], choices=['T', 'N'], help='Modes to process.')
    p.add_argument('--years', type=int, nargs='*', default=None,
                   help='Optional case Tstart years to process. Use this to exclude 1990s years when RUC files are unavailable.')
    p.add_argument('--dx', type=float, default=50.0, help='Half-width of RAP subset in km east/west.')
    p.add_argument('--dy', type=float, default=50.0, help='Half-width of RAP subset in km north/south.')
    p.add_argument('--normalized-storm-motion', type=float, default=90.0, help='Target storm-motion angle in degrees.')
    p.add_argument('--dt-tracking', type=float, default=1/60, help='Storm-track interpolation interval in minutes.')
    p.add_argument('--output-prefix', default=None, help='Optional prefix for output pickle names.')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip analysis times already present in an existing base-state pickle.')
    p.add_argument('--resume-from', default=None,
                   help='Existing merged base-state pickle to use for skip/merge behavior. Defaults to {output_prefix}.pkl when present.')
    p.add_argument('--debug-dir', default=None, help='Optional directory for per-case debug diagnostics.')
    p.add_argument('--share-base-by-model-hour', action='store_true',
                   help='Reuse the final base-state value within the same case and rounded model hour. This is a method choice, not just an I/O optimization.')
    p.add_argument('--use-inflow-mask', action='store_true', help='Restrict base-state sampling to a storm-relative inflow sector.')
    p.add_argument('--inflow-center-angle', type=float, default=0.0, help='Sector center angle in storm-relative coordinates (deg, 0 = along storm motion).')
    p.add_argument('--inflow-half-angle', type=float, default=60.0, help='Half-width of inflow sector in degrees.')
    p.add_argument('--inflow-inner-radius-km', type=float, default=20.0, help='Inner exclusion radius for inflow mask in km.')
    p.add_argument('--inflow-outer-radius-km', type=float, default=np.nan, help='Optional outer radius for inflow mask in km. Default uses the full RAP subset.')
    p.add_argument('--grib-dir', default=None,
                   help='Pre-staged directory of GRIB files. If a matching file exists here it is '
                        'used directly and no network download is attempted. Useful on HPC nodes '
                        'without outbound internet access (e.g. Schooner compute nodes).')
    p.add_argument('--elev-dir', default='../data/elevation',
                   help='Directory containing USGS 3DEP/TNM GeoTIFF elevation tiles. Used to derive DEM regional Z when --elevation-cache is not supplied.')
    p.add_argument('--elevation-cache', default=None,
                   help='Optional analysis_elevation_cache_tsc*.pkl file. If supplied, Z is read from this cache before attempting DEM reads.')
    p.add_argument('--dem-min-bbox-deg', type=float, default=0.002,
                   help='Minimum lon/lat span for DEM sampling bbox. Tiny mobile-mesonet footprints are expanded to this size before reading GeoTIFFs.')
    p.add_argument('--skip-pre-download', action='store_true',
                   help='Do not have rank 0 pre-download all GRIB files. Workers will use local files/download-on-demand.')
    p.add_argument('--pre-download-only', action='store_true',
                   help='Only pre-download/validate GRIB files, then exit before base-state computation.')
    p.add_argument('--download-timeout', type=int, default=180,
                   help='HTTP timeout in seconds for direct NCEI downloads.')
    return p.parse_args()


def maybe_load_pickle(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def infer_resume_paths(output_prefix, resume_from):
    final_path = resume_from or f'{output_prefix}.pkl'
    if final_path.endswith('.pkl'):
        prefix = final_path[:-4]
    else:
        prefix = final_path
        final_path = f'{final_path}.pkl'
    return final_path, prefix


def has_real_base_value(value):
    return isinstance(value, dict) and len(value) > 0


def analysis_offsets_for(time_to_space_conversion):
    if time_to_space_conversion == 5:
        return np.arange(-20, 21, 5)
    if time_to_space_conversion == 10:
        return np.array([-15, -5, 5, 15])
    raise ValueError('Unsupported time_to_space_conversion')


def within_k_sigma(arr, base_mask, k=2.0):
    """Return mask where arr is within k sigma of values selected by base_mask.

    This replaces the old import from make_mesonet_data, which is no longer
    part of the current workflow. If the subset is too small or has zero/invalid
    spread, keep all values rather than rejecting an entire field.
    """
    arr = np.asarray(arr, dtype=float)
    base_mask = np.asarray(base_mask, dtype=bool)
    subset = arr[base_mask & np.isfinite(arr)]
    if subset.size < 8:
        return np.ones(arr.shape, dtype=bool)
    mu = np.nanmean(subset)
    sig = np.nanstd(subset, ddof=1)
    if not np.isfinite(sig) or sig <= 0:
        return np.ones(arr.shape, dtype=bool)
    return np.isfinite(arr) & (np.abs(arr - mu) <= float(k) * sig)


def stat_triplet(arr):
    arr = np.asarray(arr, dtype=float)
    good = np.isfinite(arr)
    if not np.any(good):
        return np.nan, np.nan, np.nan
    vals = arr[good]
    return float(np.nanmin(vals)), float(np.nanmedian(vals)), float(np.nanmax(vals))


def km_between(lon1, lat1, lon2, lat2):
    return float(haversine(lon1, lat1, lon2, lat2) / 1000.0)


def make_output_prefix(time_to_space_conversion, user_prefix=None):
    if user_prefix:
        return user_prefix
    return f'base_state_data_tsc_{time_to_space_conversion}_min'


def _pickle_path_from_prefix(output_prefix, suffix):
    """Return a pickle path made from an output prefix and suffix.

    output_prefix may include directories, e.g.
    ../data/base_states_RAP_RUC/base_state_data_tsc5.
    Parent directories are created before writing.
    """
    path = Path(f"{output_prefix}{suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _debug_pickle_path(debug_dir, output_prefix, mode, rank):
    """Return a per-rank debug pickle path.

    Use only the basename/stem of output_prefix inside debug_dir so a prefix
    containing directories does not create paths like
    debug/base_states_RAP_RUC/base_state_data...
    """
    stem = Path(str(output_prefix)).name
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}_{mode}_debug_rank{rank}.pkl"


def _debug_csv_path(debug_dir, output_prefix, mode):
    stem = Path(str(output_prefix)).name
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}_{mode}_debug.csv"


@lru_cache(maxsize=512)
def _mesonet_time_elev_arrays(nc_path: str):
    with xr.open_dataset(nc_path) as ds:
        if "time" not in ds or "elev" not in ds:
            return np.array([], dtype=object), np.array([], dtype=float)
        tm = num2date(ds["time"].values, 'seconds since 1970-01-01', only_use_cftime_datetimes=False)
        tm = [datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in tm]
        elev = np.asarray(ds["elev"].values, dtype=float)
    n = min(len(tm), elev.size)
    return np.array(tm[:n], dtype=object), elev[:n]


def derive_mesonet_window_z(dir_mesonet: Path, analysis_dt: datetime, half_window: timedelta):
    mmfs = get_mm_files(str(dir_mesonet), analysis_dt)
    elev_samples = []
    files_used = 0
    for fi in mmfs:
        tm, elev = _mesonet_time_elev_arrays(str(fi))
        if tm.size == 0 or elev.size == 0:
            continue
        keep = (tm >= analysis_dt - half_window) & (tm <= analysis_dt + half_window)
        if not np.any(keep):
            continue
        vals = np.asarray(elev[keep], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        elev_samples.append(vals)
        files_used += 1
    if not elev_samples:
        return np.nan, 0, 0
    merged = np.concatenate(elev_samples)
    return float(np.nanmean(merged)), files_used, int(merged.size)


def build_mesonet_z_lookup(dir_mesonet: Path, transects_csv: Path, mode: str, time_to_space_conversion: int):
    t0, _, tr = get_t0s(str(transects_csv), mode)
    half_window = timedelta(seconds=(time_to_space_conversion * 60) / 2)
    offsets = analysis_offsets_for(time_to_space_conversion)
    z_lookup = {}
    z_meta = {}
    for case_idx in range(len(t0)):
        for offset in offsets:
            analysis_dt = t0[case_idx] + timedelta(minutes=int(offset))
            analysis_key = analysis_dt.strftime('%Y%m%d%H%M')
            z_val, files_used, npts = derive_mesonet_window_z(dir_mesonet, analysis_dt, half_window)
            z_lookup[analysis_key] = z_val
            z_meta[analysis_key] = {'files_used': files_used, 'n_elev_samples': npts}
    return z_lookup, z_meta


def build_mesonet_pickle_z_lookup(mesonet_pickle: Path, mode: str):
    with open(mesonet_pickle, 'rb') as f:
        mm = pickle.load(f)
    mode_dict = mm.get(mode, {}) if isinstance(mm, dict) else {}
    z_lookup = {}
    z_meta = {}
    for analysis_key, obs in mode_dict.items():
        try:
            elev = np.asarray(obs.get('elev', []), dtype=float)
        except Exception:
            elev = np.array([], dtype=float)
        finite = elev[np.isfinite(elev)]
        z_lookup[analysis_key] = float(np.nanmean(finite)) if finite.size else np.nan
        z_meta[analysis_key] = {'n_elev_samples': int(finite.size), 'source': 'mesonet_pickle'}
    return z_lookup, z_meta


def wmean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights)
    if not np.any(good):
        return np.nan
    vs = values[good]
    ws = weights[good]
    denom = np.nansum(ws)
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.nansum(vs * ws) / denom)


def angle_diff_deg(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


def build_inflow_mask(dx_km, dy_km, u_storm, v_storm, target_angle=90.0,
                      center_angle_deg=0.0, half_angle_deg=60.0,
                      inner_radius_km=20.0, outer_radius_km=np.nan):
    rx, ry = rotate_grid(dx_km, dy_km, u_storm, v_storm, target_angle=target_angle, pivot_x=0, pivot_y=0)
    radius = np.hypot(rx, ry)
    azimuth = np.degrees(np.arctan2(ry, rx))
    sector = np.abs(angle_diff_deg(azimuth, center_angle_deg)) <= half_angle_deg
    mask = np.isfinite(radius) & sector & (radius >= inner_radius_km)
    if np.isfinite(outer_radius_km):
        mask &= radius <= outer_radius_km
    return mask, rx, ry, azimuth, radius
    
def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

def subset_data_km(clat, clon, lat2d, lon2d, data2d, dlat_km, dlon_km):
    """Subset a curvilinear grid within +/- dlat_km north/south and +/- dlon_km east/west."""
    lat2d = np.asarray(lat2d); lon2d = np.asarray(lon2d); data2d = np.asarray(data2d)

    # normalize longitudes to the convention of clon
    def wrap_like(lon, ref):
        if np.nanmax(lon) > 180:
            lon = lon % 360.0
            ref = ref % 360.0
        else:
            lon = ((lon + 180) % 360) - 180
            ref = ((ref + 180) % 360) - 180
        return lon, ref
    lon2d, clon = wrap_like(lon2d, clon)

    # simple local equirectangular approximation (fine over a few hundred km)
    deg2km = 111.32
    coslat = np.cos(np.deg2rad(clat))
    dx_km = ( ((lon2d - clon + 180) % 360) - 180 ) * coslat * deg2km
    dy_km = (lat2d - clat) * deg2km

    m = (np.abs(dx_km) <= dlon_km) & (np.abs(dy_km) <= dlat_km)
    if not np.any(m):
        raise ValueError("No grid points fell inside the requested km box.")

    i_idx, j_idx = np.where(m)
    i0, i1 = i_idx.min(), i_idx.max()
    j0, j1 = j_idx.min(), j_idx.max()
    return (lat2d[i0:i1+1, j0:j1+1],
            lon2d[i0:i1+1, j0:j1+1],
            data2d[i0:i1+1, j0:j1+1])

def state_at_height(p_lev_hPa, z3d_m, T3d_C, RH3d_pct, U3d, V3d, Z_m,
                    fill_value=np.nan, extrapolate="nearest"):
    """
    Sample a consistent thermodynamic-kinematic state at height Z.

    Inputs
    ------
    p_lev_hPa : (nz,)          pressure levels (hPa), matching first axis of z3d/T3d/etc.
    z3d_m     : (nz,ny,nx)     geopotential height [m]
    T3d_C     : (nz,ny,nx)     temperature [°C]
    RH3d_pct  : (nz,ny,nx)     relative humidity [%]
    U3d, V3d  : (nz,ny,nx)     wind components (m s^-1) (or consistent units)
    Z_m       : scalar or (ny,nx) target height [m]

    Returns
    -------
    pZ_hPa, TZ_C, qZ_kgkg, UZ, VZ, RHZ_pct
        All on (ny,nx), evaluated at the same height Z.
    """
    z3d   = np.asarray(z3d_m, dtype=float)
    T3d   = np.asarray(T3d_C, dtype=float)
    RH3d  = np.asarray(RH3d_pct, dtype=float)
    U3d   = np.asarray(U3d, dtype=float)
    V3d   = np.asarray(V3d, dtype=float)

    nz, ny, nx = z3d.shape
    Z = np.broadcast_to(np.asarray(Z_m, dtype=float), (ny, nx))

    # Ensure z increases with level index
    p1d = np.asarray(p_lev_hPa, dtype=float)
    if np.nanmean(np.diff(z3d[:, 0, 0])) < 0:
        z3d = z3d[::-1, ...]
        T3d = T3d[::-1, ...]
        RH3d = RH3d[::-1, ...]
        U3d = U3d[::-1, ...]
        V3d = V3d[::-1, ...]
        p1d = p1d[::-1]

    # Build 3D pressure from levels for moisture conversions
    p3d = (p1d[:, None, None] * np.ones_like(z3d))  # hPa

    # Convert RH->mixing ratio w at each native level (vectorized)
    # (MetPy expects units)
    w3d = mixing_ratio_from_relative_humidity(
        (p3d * units.hectopascal),
        (T3d * units.degC),
        (RH3d * units.percent)
    ).m  # kg/kg

    # Find bracketing indices for Z
    z0 = z3d[:-1]; z1 = z3d[1:]
    cross = (z0 <= Z) & (z1 >= Z)
    k_cross = cross.argmax(axis=0)
    has_cross = cross.any(axis=0)
    below = Z < np.nanmin(z3d, axis=0)
    above = Z > np.nanmax(z3d, axis=0)

    if extrapolate == "nearest":
        k = np.where(has_cross, k_cross, np.where(below, 0, np.where(above, nz-2, k_cross)))
    else:
        k = np.where(has_cross, k_cross, -1)

    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # Initialize outputs
    TZ = np.full((ny, nx), fill_value, float)
    wZ = np.full((ny, nx), fill_value, float)
    UZ = np.full((ny, nx), fill_value, float)
    VZ = np.full((ny, nx), fill_value, float)

    valid = (k >= 0)
    if np.any(valid):
        kv = k[valid]; jv = jj[valid]; iv = ii[valid]; Zv = Z[valid]

        def lerp(f3d):
            fk  = f3d[kv,   jv, iv]
            fk1 = f3d[kv+1, jv, iv]
            zk  = z3d[kv,   jv, iv]
            zk1 = z3d[kv+1, jv, iv]
            w = np.where(np.abs(zk1 - zk) > 0, (Zv - zk) / (zk1 - zk), 0.0)
            return fk + w * (fk1 - fk)

        TZ[valid] = lerp(T3d)   # °C
        wZ[valid] = lerp(w3d)   # kg/kg
        UZ[valid] = lerp(U3d)
        VZ[valid] = lerp(V3d)

    # Pressure at the same Z
    pZ = pressure_at_height(p1d, z3d, Z, fill_value=fill_value, extrapolate=extrapolate)  # hPa

    # Recover RH at Z from (pZ, TZ, wZ)
    RHZ = np.full((ny, nx), fill_value, float)
    ok = np.isfinite(pZ) & np.isfinite(TZ) & np.isfinite(wZ)
    if np.any(ok):
        RHZ[ok] = relative_humidity_from_mixing_ratio(
            (pZ[ok] * units.hectopascal),
            (TZ[ok] * units.degC),
            (wZ[ok] * units.dimensionless)
        ).to('percent').m

    return pZ, TZ, UZ, VZ, RHZ

def pressure_at_height(p_lev_hPa, z3d_m, Z_m, fill_value=np.nan, extrapolate="nearest"):
    """
    Interpolate pressure to a target geometric height Z.

    Parameters
    ----------
    p_lev_hPa : (nz,)     1-D pressure coordinate (hPa), matching z3d's vertical order
    z3d_m     : (nz,ny,nx) geopotential height (m) at each (level,y,x)
    Z_m       : scalar or (ny,nx) target height(s) in meters (broadcastable to (ny,nx))
    fill_value: value to use where interpolation is impossible
    extrapolate: {'nearest','nan'}
        - 'nearest': clamp to the nearest two layers (i.e., extrapolate linearly in z on ln p)
        - 'nan'    : return fill_value where Z is outside [min(z), max(z)] in a column

    Returns
    -------
    pZ_hPa : (ny,nx) pressure (hPa) at height Z
    """
    z3d = np.asarray(z3d_m, dtype=float)
    p1d = np.asarray(p_lev_hPa, dtype=float)
    nz, ny, nx = z3d.shape

    # broadcast Z to (ny,nx)
    Z = np.broadcast_to(np.asarray(Z_m, dtype=float), (ny, nx))

    # assume k increases upward; if not, flip both consistently
    if np.nanmean(np.diff(z3d[:,0,0])) < 0:  # heights decreasing with k
        z3d = z3d[::-1, ...]
        p1d = p1d[::-1]

    # Find bracketing indices k such that z[k] <= Z <= z[k+1]
    z0 = z3d[:-1]     # (nz-1,ny,nx)
    z1 = z3d[1:]
    cross = (z0 <= Z) & (z1 >= Z)            # (nz-1,ny,nx)

    # Where there is a crossing, take the first k along vertical
    k_cross = cross.argmax(axis=0)           # (ny,nx) (0 if no True)
    has_cross = cross.any(axis=0)            # (ny,nx)

    # Edge cases per column
    below = Z < np.nanmin(z3d, axis=0)       # Z below lowest level
    above = Z > np.nanmax(z3d, axis=0)       # Z above highest level

    if extrapolate == "nearest":
        k = np.where(has_cross, k_cross,
             np.where(below, 0, np.where(above, nz-2, k_cross)))
    else:  # 'nan'
        k = np.where(has_cross, k_cross, -1)  # -1 will mark invalid

    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # Initialize output
    pZ = np.full((ny, nx), fill_value, dtype=float)

    # valid mask to compute
    valid = (k >= 0)
    if not np.any(valid):
        return pZ

    kv = k[valid]
    jv = jj[valid]
    iv = ii[valid]
    Zv = Z[valid]

    zk  = z3d[kv,   jv, iv]
    zk1 = z3d[kv+1, jv, iv]
    denom = (zk1 - zk)
    w = np.where(np.abs(denom) > 0, (Zv - zk) / denom, 0.0)  # 0..1

    # log-linear interpolation in pressure between levels k and k+1
    ln_pk  = np.log(p1d[kv])
    ln_pk1 = np.log(p1d[kv+1])
    ln_pZ  = (1.0 - w) * ln_pk + w * ln_pk1
    pZ[valid] = np.exp(ln_pZ)

    return pZ  # hPa

def download_read_file(date, download_pth, levels, grid=130, grib_dir=None, timeout=180):
    Path(download_pth).mkdir(parents=True, exist_ok=True)

    if date.year > 2012:  # use RAP
        filename = f"rap_{grid}_{date:%Y%m%d_%H%M}_000.grb2"
        # Primary URL: NCEI historical archive (covers ~pre-2022).
        urls = [
            f"https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/"
            f"{date:%Y%m}/{date:%Y%m%d}/" + filename,
        ]
        if date.year >= 2022:
            # Recent RAP data is in a separate NCEI bucket not yet mirrored to historical/.
            urls.append(
                f"https://www.ncei.noaa.gov/data/rapid-refresh/access/rap-130-13km/analysis/"
                f"{date:%Y%m}/{date:%Y%m%d}/" + filename
            )
    else:  # use RUC
        filename = f"ruc2anl_{grid}_{date:%Y%m%d_%H%M}_000.grb2"
        urls = [
            f"https://www.ncei.noaa.gov/thredds/fileServer/model-ruc{grid}anl/"
            f"{date:%Y%m}/{date:%Y%m%d}/" + filename,
        ]

    # If a pre-staged directory is given, prefer those files and skip downloads.
    if grib_dir:
        staged = Path(grib_dir) / filename
        if staged.exists() and staged.stat().st_size > 0:
            target = staged
            urls = []  # nothing to download
        else:
            target = Path(download_pth) / filename
    else:
        target = Path(download_pth) / filename

    def download_atomic(src_url, dst_path):
        tmp_path = dst_path.with_suffix(dst_path.suffix + '.part')
        if tmp_path.exists():
            tmp_path.unlink()
        urlretrieve(src_url, str(tmp_path))
        os.replace(tmp_path, dst_path)

    def ensure_local_file():
        if target.exists() and target.stat().st_size > 0:
            return True
        for url in urls:
            try:
                download_atomic(url, target)
                return True
            except Exception:
                print(f"[WARN] download failed: {url}")
        return False

    if not ensure_local_file():
        return

    #Open and read the file
    def open_level(path, level, shortName=None):
        indexless = {"indexpath": ""}
        fkeys = {"typeOfLevel": "isobaricInhPa", "level": int(level)}
        if shortName is not None:
            fkeys["shortName"] = shortName
        return xr.open_dataset(path, engine="cfgrib",
                               backend_kwargs=dict(filter_by_keys=fkeys, **indexless))

    def read_iso_dataset(path):
        ds_t  = xr.concat([open_level(path, L, "t")  for L in levels], dim="isobaricInhPa")
        ds_u  = xr.concat([open_level(path, L, "u")  for L in levels], dim="isobaricInhPa")
        ds_v  = xr.concat([open_level(path, L, "v")  for L in levels], dim="isobaricInhPa")
        ds_gh = xr.concat([open_level(path, L, "gh") for L in levels], dim="isobaricInhPa")
        ds_r  = xr.concat([open_level(path, L, "r")  for L in levels], dim="isobaricInhPa")
        return xr.merge([ds_t, ds_u, ds_v, ds_gh, ds_r], compat="override")

    try:
        ds_iso = read_iso_dataset(str(target))
    except Exception:
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass
        if not ensure_local_file():
            return
        ds_iso = read_iso_dataset(str(target))

    return ds_iso, str(target)

def download_rap_from_source(atime, download_pth, clat, clon, levels, grid=130, dx=50, dy=50, grib_dir=None, timeout=180):
    out = download_read_file(hour_rounder(atime), download_pth, levels, grid=grid, grib_dir=grib_dir, timeout=timeout)
    if out is None:
        return
    ds_iso, filename = out
    if ds_iso is not None:
        
        for al in range(len(levels)):
            ds = ds_iso.sel(isobaricInhPa=levels[al])  # dims: time, isobaricInhPa, y, x
        
            latitudes, longitudes = ds.latitude.values, ds.longitude.values - 360
        
            #Get variables we want
            lat_slice, lon_slice, at   = subset_data_km(clat, clon, latitudes, longitudes, ds.t.values,   dy, dx)
            lat_slice, lon_slice, au   = subset_data_km(clat, clon, latitudes, longitudes, ds.u.values,   dy, dx)
            lat_slice, lon_slice, av   = subset_data_km(clat, clon, latitudes, longitudes, ds.v.values,   dy, dx)
            lat_slice, lon_slice, agh  = subset_data_km(clat, clon, latitudes, longitudes, ds.gh.values,  dy, dx)
            lat_slice, lon_slice, ar   = subset_data_km(clat, clon, latitudes, longitudes, ds.r.values,   dy, dx)
        
            if al == 0:
                t,u = np.zeros((len(levels),lat_slice.shape[0],lat_slice.shape[1]))*np.nan,np.zeros((len(levels),lat_slice.shape[0],lat_slice.shape[1]))*np.nan
                v,gh = np.zeros((len(levels),lat_slice.shape[0],lat_slice.shape[1]))*np.nan,np.zeros((len(levels),lat_slice.shape[0],lat_slice.shape[1]))*np.nan
                r = np.zeros((len(levels),lat_slice.shape[0],lat_slice.shape[1]))*np.nan
        
            t[al,:,:] = at - 273.15 #C
            u[al,:,:] = au
            v[al,:,:] = av
            gh[al,:,:] = agh
            r[al,:,:] = ar
        
        return t,u,v,gh,r,levels,lat_slice,lon_slice,filename
    else:
        return 

# -----------------------------------------------------------------------------
# Current-workflow transect, storm-track, and DEM-derived regional Z helpers
# -----------------------------------------------------------------------------
_ELEV_WARNED_TILES_LOCAL = set()


def resolve_transects_path(path_like):
    p = Path(path_like)
    if p.is_file():
        return p
    for name in ("transect_times.xlsx", "transect_times.xls", "transect_times.csv"):
        cand = p / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find transect_times file from {path_like}")


def _read_transects_table(path_like):
    path = resolve_transects_path(path_like)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, sheet_name="transect_times")
        except Exception:
            return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)


def get_t0s(file_path, mode):
    """Current-workflow get_t0s replacement supporting xlsx/csv."""
    tr = _read_transects_table(file_path)
    tr.columns = [str(c).strip() for c in tr.columns]
    if mode == 'T':
        tr = tr.loc[tr.Mode.astype(str) == 'T'].copy()
        amode = 'T'
    elif mode == 'N':
        tr = tr.loc[tr.Mode.astype(str) == 'N0'].copy()
        amode = 'N'
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    tr['Tstart'] = tr['Tstart'].astype(str).str.strip()
    t0 = np.array([datetime.strptime(val, '%Y%m%d-%H%M') for val in tr['Tstart'].values], dtype=object)
    tr = tr.reset_index(drop=True)
    return t0, amode, tr



def _parse_transect_time_value(value):
    """Parse common transect time formats defensively."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none"}:
        return None
    for fmt in ("%Y%m%d-%H%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        ts = pd.to_datetime(s)
        if pd.isna(ts):
            return None
        return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    except Exception:
        return None


def _analysis_window_for_atime(atime, time_to_space_conversion):
    half = timedelta(seconds=int((float(time_to_space_conversion) * 60.0) / 2.0))
    return atime - half, atime + half


def _windows_overlap(a0, a1, b0, b1):
    return (a0 <= b1) and (a1 >= b0)


def build_analysis_jobs(t0, tr, amode, analysis_times, time_to_space_conversion):
    """Build analysis jobs and skip windows with no observation overlap.

    This keeps the RAP/RUC workflow consistent with the sounding and mesonet
    synthesis workflows: the nominal offsets are still centered on Tstart, but
    an analysis time is processed only when its finite time-to-space window
    overlaps Observation Start/Observation End.
    """
    jobs = []
    skipped_no_obs_overlap = 0
    has_obs = {"Observation Start", "Observation End"}.issubset(set(tr.columns))
    for case_idx in range(len(t0)):
        row = tr.iloc[case_idx]
        case_id = str(row["CaseID"]).strip()
        obs0 = _parse_transect_time_value(row["Observation Start"]) if has_obs else None
        obs1 = _parse_transect_time_value(row["Observation End"]) if has_obs else None
        if obs0 is not None and obs1 is not None and obs1 < obs0:
            raise ValueError(f"Observation End precedes Observation Start for CaseID={row.get('CaseID')}: {obs0} to {obs1}")
        for off in analysis_times:
            atime = t0[case_idx] + timedelta(minutes=int(off))
            w0, w1 = _analysis_window_for_atime(atime, time_to_space_conversion)
            if obs0 is not None and obs1 is not None and not _windows_overlap(w0, w1, obs0, obs1):
                skipped_no_obs_overlap += 1
                continue
            jobs.append({
                "atime": atime,
                "atime_key": atime.strftime("%Y%m%d%H%M"),
                "case_id": case_id,
                "case_label": str(row.get("CaseID", case_id)),
                "offset_min": int(off),
                "obs_start": obs0,
                "obs_end": obs1,
                "window_start": w0,
                "window_end": w1,
            })
    return jobs, skipped_no_obs_overlap


def _track_subdir_for_mode(mode):
    return 'tornadic' if str(mode) in {'T', 'L', 'S', 'W', 'Sk'} else 'nontornadic'


def _find_case_track_file(case_id, tracking_dir, mode):
    case_id = str(case_id).strip()
    subdir = Path(tracking_dir) / _track_subdir_for_mode(mode)
    matches = sorted(
        p for p in subdir.glob(f"{case_id}_*.csv")
        if p.is_file() and not p.name.startswith((".", "._"))
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple storm-track files match CaseID={case_id}: {[str(p) for p in matches]}")
    return None


def _read_storm_track_columns(track_file):
    strk = pd.read_csv(track_file)
    if strk.empty:
        return None
    colmap = {str(c).strip().lower(): c for c in strk.columns}
    time_col = next((colmap[c] for c in ('time', 'datetime', 'time_utc', 'date_time') if c in colmap), None)
    lat_col = next((colmap[c] for c in ('lat', 'latitude', 'mesocyclone_latitude', 'track lat') if c in colmap), None)
    lon_col = next((colmap[c] for c in ('lon', 'longitude', 'mesocyclone_longitude', 'track lon') if c in colmap), None)
    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"{track_file} missing recognizable time/lat/lon columns: {strk.columns.tolist()}")

    mt = pd.to_datetime(strk[time_col].values, errors='coerce')
    good_time = ~pd.isna(mt)
    if good_time.sum() < 2:
        return None

    lat = pd.to_numeric(strk[lat_col], errors='coerce').to_numpy(float)[good_time]
    lon = pd.to_numeric(strk[lon_col], errors='coerce').to_numpy(float)[good_time]
    mt = mt[good_time]
    good_pos = np.isfinite(lat) & np.isfinite(lon)
    if good_pos.sum() < 2:
        return None

    mt = mt[good_pos]
    lat = lat[good_pos]
    lon = lon[good_pos]
    mtimes = np.array([datetime(v.year, v.month, v.day, v.hour, v.minute, v.second) for v in mt], dtype=object)
    return mtimes, lat, lon


def parse_storm_tracks(case_id, tracking_dir, mode, atime, dt_tracking=1/60, time_to_space_conversion=10):
    """Return storm-track samples and storm motion for one analysis window.

    Uses the literal CaseID from transect_times.xlsx and supports both the new
    time/lat/lon schema and the older datetime/latitude/longitude schema.
    """
    track_file = _find_case_track_file(case_id, tracking_dir, mode)
    if track_file is None:
        return None

    try:
        out = _read_storm_track_columns(track_file)
    except Exception:
        return None
    if out is None:
        return None

    mtimes, lat, lon = out
    tnum = date2num(mtimes, 'seconds since 1970-01-01')
    order = np.argsort(tnum)
    tnum = np.asarray(tnum, dtype=float)[order]
    lat = np.asarray(lat, dtype=float)[order]
    lon = np.asarray(lon, dtype=float)[order]
    tnum, keep = np.unique(tnum, return_index=True)
    lat = lat[keep]
    lon = lon[keep]
    if tnum.size < 2:
        return None

    dt_seconds = float(dt_tracking) * 60.0
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 1.0
    time_grid = np.arange(tnum[0], tnum[-1] + dt_seconds, dt_seconds)
    if time_grid.size < 2:
        return None

    lat_i = np.interp(time_grid, tnum, lat)
    lon_i = np.interp(time_grid, tnum, lon)
    mt_i = num2date(time_grid, 'seconds since 1970-01-01')
    mt_i = np.array([datetime(v.year, v.month, v.day, v.hour, v.minute, v.second) for v in mt_i], dtype=object)
    half = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    imeso = np.where((mt_i >= atime - half) & (mt_i <= atime + half))[0]
    if imeso.size < 2:
        return None

    mt_i, lat_i, lon_i = mt_i[imeso], lat_i[imeso], lon_i[imeso]
    isort = np.argsort(mt_i)
    mt_i, lat_i, lon_i = mt_i[isort], lat_i[isort], lon_i[isort]
    dt_storm = (mt_i[-1] - mt_i[0]).total_seconds()
    if dt_storm <= 0:
        return None
    dx_storm, dy_storm = get_dx_dy(lon_i[0], lat_i[0], lon_i[-1], lat_i[-1])
    return mt_i, lat_i, lon_i, dx_storm / dt_storm, dy_storm / dt_storm


def _elev_tile_id_from_floor(lat_floor, lon_floor):
    lat_id = int(lat_floor) + 1 if lat_floor >= 0 else int(lat_floor)
    lon_id = int(-np.floor(float(lon_floor)))
    return f"{'n' if lat_id >= 0 else 's'}{abs(lat_id):02d}{'w' if lon_id >= 0 else 'e'}{abs(lon_id):03d}"


def _tile_ids_for_bbox(lon_min, lat_min, lon_max, lat_max):
    vals = [lon_min, lat_min, lon_max, lat_max]
    if not np.all(np.isfinite(vals)):
        return []
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    # Expand tiny bboxes so rasterio gets nonzero pixels.
    min_span = getattr(_tile_ids_for_bbox, 'min_bbox_deg', 0.002)
    if (lon_max - lon_min) < min_span:
        mid = 0.5 * (lon_min + lon_max)
        lon_min, lon_max = mid - min_span / 2.0, mid + min_span / 2.0
    if (lat_max - lat_min) < min_span:
        mid = 0.5 * (lat_min + lat_max)
        lat_min, lat_max = mid - min_span / 2.0, mid + min_span / 2.0
    lats = range(int(np.floor(lat_min)), int(np.floor(lat_max)) + 1)
    lons = range(int(np.floor(lon_min)), int(np.floor(lon_max)) + 1)
    return sorted(set(_elev_tile_id_from_floor(la, lo) for la in lats for lo in lons))


@lru_cache(maxsize=None)
def _find_local_elev_tile(elev_dir, tile_id):
    matches = sorted(glob(os.path.join(str(elev_dir), f"*{tile_id}*.tif")) + glob(os.path.join(str(elev_dir), f"*{tile_id}*.tiff")))
    return matches[0] if matches else None


def _download_elev_tile(elev_dir, tile_id, timeout=60):
    elev_dir = Path(elev_dir)
    prefix = f"StagedProducts/Elevation/13/TIFF/historical/{tile_id}/"
    try:
        resp = requests.get("https://prd-tnm.s3.amazonaws.com", params={"prefix": prefix}, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception:
        return None
    keys = []
    for elem in root.iter():
        if elem.tag.endswith("Key") and elem.text and elem.text.endswith(".tif"):
            keys.append(elem.text)
    if not keys:
        if tile_id not in _ELEV_WARNED_TILES_LOCAL:
            print(f"[WARN] elevation tile {tile_id} not found at TNM S3", flush=True)
            _ELEV_WARNED_TILES_LOCAL.add(tile_id)
        return None
    key = sorted(keys)[-1]
    out_path = elev_dir / os.path.basename(key)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    try:
        elev_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with requests.get(f"https://prd-tnm.s3.amazonaws.com/{key}", stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.replace(out_path)
        _find_local_elev_tile.cache_clear()
        print(f"[INFO] downloaded elevation tile https://prd-tnm.s3.amazonaws.com/{key}", flush=True)
        return out_path
    except Exception:
        tmp.unlink(missing_ok=True)
        return None


def _resolve_elev_tile_path(elev_dir, tile_id, allow_download=True):
    local_path = _find_local_elev_tile(str(elev_dir), tile_id)
    if local_path is None and allow_download:
        got = _download_elev_tile(Path(elev_dir), tile_id)
        if got is not None:
            local_path = str(got)
    return local_path


def _expanded_bbox(lon_min, lat_min, lon_max, lat_max, min_bbox_deg=0.002):
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if (lon_max - lon_min) < min_bbox_deg:
        mid = 0.5 * (lon_min + lon_max)
        lon_min, lon_max = mid - min_bbox_deg / 2.0, mid + min_bbox_deg / 2.0
    if (lat_max - lat_min) < min_bbox_deg:
        mid = 0.5 * (lat_min + lat_max)
        lat_min, lat_max = mid - min_bbox_deg / 2.0, mid + min_bbox_deg / 2.0
    return lon_min, lat_min, lon_max, lat_max


def _dem_mean_in_bbox(elev_dir, lon_min, lat_min, lon_max, lat_max, min_bbox_deg=0.002):
    lon_min, lat_min, lon_max, lat_max = _expanded_bbox(lon_min, lat_min, lon_max, lat_max, min_bbox_deg)
    _tile_ids_for_bbox.min_bbox_deg = min_bbox_deg
    tile_ids = _tile_ids_for_bbox(lon_min, lat_min, lon_max, lat_max)
    vals = []
    used_files = []
    n_pixels = 0
    for tile_id in tile_ids:
        local_path = _resolve_elev_tile_path(elev_dir, tile_id, allow_download=True)
        if local_path is None:
            continue
        try:
            with rasterio.open(local_path) as src:
                xs = [lon_min, lon_max]
                ys = [lat_min, lat_max]
                if src.crs is not None and not src.crs.is_geographic:
                    xs, ys = rio_transform('EPSG:4326', src.crs, xs, ys)
                left, right = min(xs), max(xs)
                bottom, top = min(ys), max(ys)
                try:
                    window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
                    window = window.round_offsets().round_lengths()
                    full = rasterio.windows.Window(0, 0, src.width, src.height)
                    window = window.intersection(full)
                except Exception:
                    continue
                if window.width <= 0 or window.height <= 0:
                    continue
                arr = src.read(1, window=window, masked=True)
                data = arr.filled(np.nan).astype(float) if np.ma.isMaskedArray(arr) else np.asarray(arr, dtype=float)
                if src.nodata is not None:
                    data[np.isclose(data, src.nodata)] = np.nan
                data[(data < -500.0) | (data > 5000.0)] = np.nan
                finite = data[np.isfinite(data)]
                if finite.size:
                    vals.append(finite)
                    used_files.append(str(local_path))
                    n_pixels += int(finite.size)
        except Exception as exc:
            print(f"[WARN] failed reading DEM tile {local_path}: {exc!r}", flush=True)
    return float(np.nanmean(np.concatenate(vals))) if vals else np.nan, {'dem_files': used_files, 'n_dem_pixels': n_pixels, 'tile_ids': tile_ids}


def mesonet_bbox_for_time(dir_mesonet, atime, time_to_space_conversion=5):
    try:
        mmfs = get_mm_files(str(dir_mesonet), atime)
    except Exception:
        return None, []
    half = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    t0 = date2num(atime - half, 'seconds since 1970-01-01')
    t1 = date2num(atime + half, 'seconds since 1970-01-01')
    lats, lons = [], []
    for fi in mmfs:
        try:
            with xr.open_dataset(fi, decode_times=False) as ds:
                if 'time' not in ds or 'lat' not in ds or 'lon' not in ds:
                    continue
                tm = ds['time'].values
                mask = (tm >= t0) & (tm <= t1)
                if not np.any(mask):
                    continue
                lat = np.asarray(ds['lat'].values[mask], dtype=float)
                lon = np.asarray(ds['lon'].values[mask], dtype=float)
                lon = np.where(lon > 0, -lon, lon)
                good = np.isfinite(lat) & np.isfinite(lon) & (lon < 0)
                if np.any(good):
                    lats.append(lat[good])
                    lons.append(lon[good])
        except Exception:
            continue
    if not lats:
        return None, mmfs
    la = np.concatenate(lats)
    lo = np.concatenate(lons)
    return (float(np.nanmin(lo)), float(np.nanmin(la)), float(np.nanmax(lo)), float(np.nanmax(la))), mmfs


def _z_lookup_from_elevation_cache(cache_path, mode):
    if not cache_path or not Path(cache_path).exists():
        return {}, {}
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    by_key = cache.get('by_key', {}) if isinstance(cache, dict) else {}
    z_lookup = {}
    z_meta = {}
    for key, rec in by_key.items():
        if not isinstance(rec, dict):
            continue
        rec_mode = rec.get('mode')
        if rec_mode is not None and str(rec_mode) != str(mode):
            continue
        z_lookup[str(key)[:12]] = float(rec.get('z_ref_m', np.nan))
        z_meta[str(key)[:12]] = dict(rec)
        z_meta[str(key)[:12]]['source'] = 'analysis_elevation_cache'
    return z_lookup, z_meta


def build_regional_dem_z_lookup(dir_mesonet, elev_dir, transects_file, mode, time_to_space_conversion, min_bbox_deg=0.002):
    t0, _, tr = get_t0s(str(transects_file), mode)
    offsets = analysis_offsets_for(time_to_space_conversion)
    z_lookup = {}
    z_meta = {}
    for case_idx in range(len(t0)):
        for offset in offsets:
            analysis_dt = t0[case_idx] + timedelta(minutes=int(offset))
            key = analysis_dt.strftime('%Y%m%d%H%M')
            bbox, mmfs = mesonet_bbox_for_time(Path(dir_mesonet), analysis_dt, time_to_space_conversion=time_to_space_conversion)
            if bbox is None:
                z_lookup[key] = np.nan
                z_meta[key] = {'source': 'dem_mesonet_bbox', 'status': 'no_mesonet_bbox', 'n_mm_files': len(mmfs)}
                continue
            lon_min, lat_min, lon_max, lat_max = bbox
            z, meta = _dem_mean_in_bbox(Path(elev_dir), lon_min, lat_min, lon_max, lat_max, min_bbox_deg=min_bbox_deg)
            z_lookup[key] = z
            z_meta[key] = {'source': 'dem_mesonet_bbox', 'bbox': bbox, 'n_mm_files': len(mmfs), **meta}
    return z_lookup, z_meta



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == '__main__':
    args = parse_args()
    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
    if not args.elevation_cache and not args.mesonet_dir and not args.mesonet_pickle:
        raise ValueError('Provide --elevation-cache, --mesonet-dir, or --mesonet-pickle so base-state Z comes from the current DEM/mesonet workflow.')
    if args.mesonet_dir and (not Path(args.mesonet_dir).exists()):
        raise FileNotFoundError(f'--mesonet-dir does not exist: {args.mesonet_dir}')
    if args.mesonet_pickle and (not Path(args.mesonet_pickle).exists()):
        raise FileNotFoundError(f'--mesonet-pickle does not exist: {args.mesonet_pickle}')
    if rank == 0 and (args.mesonet_dir or args.mesonet_pickle) and args.share_base_by_model_hour:
        print('[WARN] --share-base-by-model-hour can reuse one base state across 5-minute offsets even when mesonet-derived Z differs by analysis time.')

    # === configuration ===
    Rd = float(np.array(const.dry_air_gas_constant))
    g = float(np.array(const.earth_gravity))
    dir_transects = args.dir_transects
    transects_file = resolve_transects_path(dir_transects)
    tracking_dir = args.tracking_dir
    download_pth = args.download_pth
    time_to_space_conversion = args.time_to_space
    dt_tracking = args.dt_tracking
    normalized_storm_motion = args.normalized_storm_motion
    dx, dy = args.dx, args.dy
    analysis_times = analysis_offsets_for(time_to_space_conversion)
    modes = args.modes
    output_prefix = make_output_prefix(time_to_space_conversion, args.output_prefix)
    resume_final_path, resume_prefix = infer_resume_paths(output_prefix, args.resume_from)
    existing_base = maybe_load_pickle(resume_final_path) if (args.skip_existing or args.resume_from) else {}
    existing_keys = {key for key, value in existing_base.items() if has_real_base_value(value)}

    # One pass per mode; each rank handles a subset of atimes
    for mode in modes:
        base = {}
        debug_rows = []
        shared_base_cache = {}
        shared_debug_cache = {}

        # === get t0, amode, tr ===
        t0, amode, tr = get_t0s(str(transects_file), mode)
        if args.years:
            wanted_years = set(int(y) for y in args.years)
            keep_year = np.array([dt.year in wanted_years for dt in t0], dtype=bool)
            t0 = t0[keep_year]
            tr = tr.loc[keep_year].reset_index(drop=True)
            if rank == 0:
                print(f"[INFO] mode={mode} year filter {sorted(wanted_years)} -> cases={len(t0)}", flush=True)
        if len(t0) == 0:
            if rank == 0:
                print(f"[INFO] mode={mode}: no cases after filtering; skipping mode", flush=True)
            continue

        # Build analysis jobs, honoring Observation Start/End.
        jobs, skipped_no_obs_overlap = build_analysis_jobs(t0, tr, amode, analysis_times, time_to_space_conversion)
        atimes = np.array([job['atime'] for job in jobs], dtype=object)
        caseIDs = np.array([job['case_id'] for job in jobs], dtype=object)
        atime_keys = np.array([job['atime_key'] for job in jobs], dtype=object)

        # === MPI partition of dt indices, with upfront skip-existing filtering ===
        n_jobs = len(jobs)
        my_idx_all = list(range(rank, n_jobs, size))
        my_idx = [dt for dt in my_idx_all if not (args.skip_existing and atime_keys[dt] in existing_keys)]
        n_skipped = len(my_idx_all) - len(my_idx)
        if rank == 0:
            print(
                f"[INFO] mode={mode}, retained tasks={n_jobs}, skipped_no_obs_overlap={skipped_no_obs_overlap}, ranks={size}",
                flush=True,
            )
        print(f"[INFO] rank={rank} mode={mode} assigned={len(my_idx_all)} active={len(my_idx)} skipped_existing={n_skipped}", flush=True)

        # Storm information only for analysis times that still need work.
        storm_data = {
            dt: parse_storm_tracks(
                caseIDs[dt],
                tracking_dir,
                amode,
                atimes[dt],
                dt_tracking=dt_tracking,
                time_to_space_conversion=time_to_space_conversion,
            )
            for dt in my_idx
        }

        mesonet_z_lookup = {}
        mesonet_z_meta = {}
        transects_csv = transects_file
        if rank == 0:
            if args.elevation_cache:
                mesonet_z_lookup, mesonet_z_meta = _z_lookup_from_elevation_cache(args.elevation_cache, mode)
                z_source_label = 'analysis_elevation_cache'
            elif args.mesonet_pickle:
                mesonet_z_lookup, mesonet_z_meta = build_mesonet_pickle_z_lookup(
                    Path(args.mesonet_pickle),
                    mode,
                )
                z_source_label = 'mesonet_pickle'
            else:
                mesonet_z_lookup, mesonet_z_meta = build_regional_dem_z_lookup(
                    Path(args.mesonet_dir),
                    Path(args.elev_dir),
                    transects_csv,
                    mode,
                    time_to_space_conversion,
                    min_bbox_deg=args.dem_min_bbox_deg,
                )
                z_source_label = 'dem_mesonet_bbox'
            n_finite = int(np.sum(np.isfinite(np.array(list(mesonet_z_lookup.values()), dtype=float))))
            print(f"[INFO] mode={mode} {z_source_label}-derived Z finite={n_finite}/{len(mesonet_z_lookup)}")
        mesonet_z_lookup = comm.bcast(mesonet_z_lookup, root=0)
        mesonet_z_meta = comm.bcast(mesonet_z_meta, root=0)

        # === Rank-0 pre-download phase ===
        # This is optional because some HPC compute nodes have poor/no outbound
        # internet. It is useful on login/data-transfer nodes or when the data
        # filesystem is shared. Missing GRIBs get a .failed sentinel so ranks do
        # not wait for unavailable files.
        if rank == 0 and not args.skip_pre_download:
            all_active_idx = [
                dt for dt in range(n_jobs)
                if (not (args.skip_existing and atime_keys[dt] in existing_keys))
                and np.isfinite(float(mesonet_z_lookup.get(atime_keys[dt], np.nan)))
            ]
            unique_date_inits = sorted(set(hour_rounder(atimes[dt]) for dt in all_active_idx))
            print(f"[PRE-DL] mode={mode}: pre-downloading/checking {len(unique_date_inits)} unique GRIB times", flush=True)
            for icount, date_init_pre in enumerate(unique_date_inits, start=1):
                model_pre = model_for_date(date_init_pre)
                key_pre = grib_key(date_init_pre, fxx=0, model=model_pre)
                ready_file_pre = ready_path_for(key_pre)
                failed_file_pre = failed_path_for(key_pre)
                if ready_file_pre.exists() or failed_file_pre.exists():
                    continue
                print(f"[PRE-DL] {icount}/{len(unique_date_inits)} {key_pre} init={date_init_pre:%Y-%m-%d %H:%M}", flush=True)

                ok_pre = False
                # Herbie is useful for RAP-era data. For RUC-era data (<=2012),
                # skip Herbie and go straight to the NCEI RUC archive.
                if (date_init_pre.year > 2012) and (not args.grib_dir):
                    try:
                        H_pre = Herbie(date_init_pre.strftime('%Y-%m-%d %H:%M'), model='rap', fxx=0)
                        if H_pre.grib is None:
                            raise FileNotFoundError(f'no RAP GRIB for {date_init_pre}')
                        _ = H_pre.download(save_dir=download_pth, verbose=False)
                        ok_pre = True
                        print(f"[PRE-DL] herbie OK: {key_pre}", flush=True)
                    except Exception as exc:
                        print(f"[PRE-DL] herbie miss/fail: {key_pre}: {type(exc).__name__}: {exc}", flush=True)

                if not ok_pre:
                    try:
                        result = download_read_file(
                            date_init_pre, download_pth,
                            DEFAULT_PRESSURE_LEVELS.copy(),
                            grid=130, grib_dir=args.grib_dir, timeout=args.download_timeout,
                        )
                        ok_pre = result is not None
                        if ok_pre:
                            print(f"[PRE-DL] ncei/local OK: {key_pre}", flush=True)
                    except Exception as exc:
                        print(f"[PRE-DL] ncei/local exception: {key_pre}: {type(exc).__name__}: {exc}", flush=True)
                        ok_pre = False

                if ok_pre:
                    ready_file_pre.touch()
                    failed_file_pre.unlink(missing_ok=True)
                else:
                    failed_file_pre.touch()
                    print(f"[PRE-DL] FAILED: {key_pre}", flush=True)

        comm.Barrier()
        if args.pre_download_only:
            if rank == 0:
                print(f"[INFO] --pre-download-only set; exiting after GRIB pre-download/check for mode={mode}", flush=True)
            continue

        # === core loop: unchanged logic, but iterate over my_idx only ===
        for dt in tqdm(my_idx,total=len(my_idx)):
            atime = atimes[dt]
            atime_key = atime.strftime('%Y%m%d%H%M')
            if storm_data[dt] is None:
                base[atime_key] = np.nan
                debug_rows.append({
                    'mode': mode, 'time': atime_key, 'case_id': str(caseIDs[dt]),
                    'source_used': 'no_storm_track', 'n_valid_points': 0, 'n_clusters': 0,
                    'flag_no_storm_track': True,
                })
                continue
            meso_time, meso_lat, meso_lon, u_storm, v_storm = storm_data[dt]
            z_mesonet = float(mesonet_z_lookup.get(atime_key, np.nan))
            Z = z_mesonet
            z_source = mesonet_z_meta.get(atime_key, {}).get('source', 'unknown')
            clat, clon = meso_lat[int(meso_lat.size/2)], meso_lon[int(meso_lon.size/2)]
            date_init = hour_rounder(atime)
            shared_key = (str(caseIDs[dt]).strip(), date_init.strftime('%Y%m%d%H%M'))
            levels = DEFAULT_PRESSURE_LEVELS.copy()

            # Reuse already-computed base states within the same case/model hour even if
            # this specific 5-minute offset has a missing Z value.
            if args.share_base_by_model_hour and shared_key in shared_base_cache:
                cached_base = dict(shared_base_cache[shared_key])
                cached_base['time'] = atime
                base[atime_key] = cached_base
                cached_debug = dict(shared_debug_cache[shared_key])
                cached_debug['time'] = atime_key
                cached_debug['Z'] = float(Z) if np.isfinite(Z) else np.nan
                cached_debug['Z_legacy'] = np.nan
                cached_debug['Z_mesonet'] = z_mesonet
                cached_debug['Z_source'] = z_source
                cached_debug['clat'] = float(clat)
                cached_debug['clon'] = float(clon)
                cached_debug['flag_shared_model_hour'] = True
                debug_rows.append(cached_debug)
                continue

            #Remove nan transects (noted when Zs=nan)
            if not np.isnan(Z):
                source_used = 'none'
            
                # ----- your existing RAP/backup download + subset + interpolation code as-is -----
                key = grib_key(date_init, fxx=0, model=model_for_date(date_init))
                owner = grib_owner(key, size)
                ready_file = ready_path_for(key)
                failed_file = failed_path_for(key)

                # Only the owner downloads when no pre-download sentinel exists.
                # For RUC-era dates, skip Herbie and use the direct NCEI path.
                if rank == owner and (not ready_file.exists()) and (not failed_file.exists()) and (not args.skip_pre_download):
                    herbie_ok = False
                    if date_init.year > 2012 and not args.grib_dir:
                        try:
                            H = Herbie(date_init.strftime('%Y-%m-%d %H:%M'), model='rap', fxx=0)
                            if H.grib is None:
                                raise FileNotFoundError(f'Herbie missing RAP analysis for {date_init:%Y-%m-%d %H:%M}')
                            _ = H.download(save_dir=download_pth, verbose=False)
                            herbie_ok = True
                        except Exception:
                            herbie_ok = False
                    if not herbie_ok:
                        try:
                            _ = download_rap_from_source(
                                atime, download_pth, clat, clon, levels=levels, grid=130,
                                dx=dx, dy=dy, grib_dir=args.grib_dir, timeout=args.download_timeout,
                            )
                            herbie_ok = _ is not None
                        except Exception:
                            herbie_ok = False
                    if herbie_ok:
                        ready_file.touch()
                        failed_file.unlink(missing_ok=True)
                    else:
                        failed_file.touch()
                else:
                    herbie_ok = None

                # Non-owners wait briefly for owner unless a failed sentinel appears.
                if rank != owner and (not ready_file.exists()) and (not failed_file.exists()):
                    t0_wait = time.time()
                    while (not ready_file.exists()) and (not failed_file.exists()):
                        if time.time() - t0_wait > max(30, args.download_timeout):
                            break
                        time.sleep(0.25)

                if failed_file.exists() and not ready_file.exists():
                    base[atime_key] = np.nan
                    debug_rows.append({
                        'mode': mode, 'time': atime_key, 'case_id': str(caseIDs[dt]),
                        'clat': float(clat), 'clon': float(clon), 'Z': float(Z),
                        'Z_legacy': np.nan, 'Z_mesonet': z_mesonet, 'Z_source': z_source,
                        'rap_init': date_init.strftime('%Y%m%d%H%M'), 'source_used': 'missing_sentinel',
                        'n_valid_points': 0, 'n_clusters': 0, 'flag_missing_grib': True,
                    })
                    continue
                
                # Now attempt to OPEN (read) using your existing logic.
                # Try Herbie first (preferred), else fallback to your local source.
                try:
                    if date_init.year <= 2012:
                        raise FileNotFoundError(f'RUC-era date uses direct NCEI path, not Herbie: {date_init:%Y-%m-%d %H:%M}')
                    H = Herbie(date_init.strftime('%Y-%m-%d %H:%M'), model='rap', fxx=0)
                    if H.grib is None:
                        raise FileNotFoundError(f'Herbie missing RAP analysis for {date_init:%Y-%m-%d %H:%M}')
                    levels = herbie_pressure_levels(H)
                    level_str = []
                    for val in levels:
                        if val == 1013.2:
                            level_str.append(':{} mb'.format(val))
                        else:
                            level_str.append(':{} mb'.format(int(val)))
                
                    for al in range(len(levels)):
                        ds = H.xarray(level_str[al])   # safe now: file should be fully written
                        latitudes, longitudes = ds.latitude.values, ds.longitude.values - 360
                
                        lat_slice, lon_slice, at  = subset_data_km(clat, clon, latitudes, longitudes, ds.t.values,  dy, dx)
                        lat_slice, lon_slice, au  = subset_data_km(clat, clon, latitudes, longitudes, ds.u.values,  dy, dx)
                        lat_slice, lon_slice, av  = subset_data_km(clat, clon, latitudes, longitudes, ds.v.values,  dy, dx)
                        lat_slice, lon_slice, agh = subset_data_km(clat, clon, latitudes, longitudes, ds.gh.values, dy, dx)
                        lat_slice, lon_slice, ar  = subset_data_km(clat, clon, latitudes, longitudes, ds.r.values,  dy, dx)
                
                        if al == 0:
                            t  = np.zeros((len(levels), lat_slice.shape[0], lat_slice.shape[1])) * np.nan
                            u  = np.zeros_like(t); v = np.zeros_like(t)
                            gh = np.zeros_like(t); r = np.zeros_like(t)
                
                        t[al,:,:]  = at - 273.15
                        u[al,:,:]  = au
                        v[al,:,:]  = av
                        gh[al,:,:] = agh
                        r[al,:,:]  = ar
                
                    source_used = 'herbie'
                    no_file = False
                
                except Exception:
                    # Either Herbie not available or still not ready (e.g., owner used fallback)
                    try:
                        t, u, v, gh, r, levels, lat_slice, lon_slice, filename = download_rap_from_source(
                            atime,
                            download_pth,
                            clat,
                            clon,
                            levels=levels,
                            grid=130,
                            dx=dx,
                            dy=dy,
                            grib_dir=args.grib_dir,
                            timeout=args.download_timeout,
                        )
                        source_used = 'fallback'
                        no_file = False
                    except Exception:
                        no_file = True

    
                if no_file:
                    base[atime_key] = np.nan
                    debug_rows.append({
                        'mode': mode,
                        'time': atime_key,
                        'case_id': str(caseIDs[dt]),
                        'clat': float(clat),
                        'clon': float(clon),
                        'Z': float(Z),
                        'Z_legacy': np.nan,
                        'Z_mesonet': z_mesonet,
                        'Z_source': z_source,
                        'rap_init': date_init.strftime('%Y%m%d%H%M'),
                        'source_used': 'missing',
                        'n_valid_points': 0,
                        'n_clusters': 0,
                        'flag_missing_grib': True,
                    })
                    continue
    
                # ----- unchanged: state_at_height, filtering, derived variables, clustering, weights -----
                pres, temp, u, v, rh = state_at_height(levels, gh, t, r, u, v, Z, fill_value=np.nan, extrapolate="nearest")
                temp, rh, u, v, hrrr_lat, hrrr_lon, pres = temp.ravel(), rh.ravel(), u.ravel(), v.ravel(), lat_slice.ravel(), lon_slice.ravel(), pres.ravel()
    
                valid0 = (
                    (~np.isnan(hrrr_lat)) & (~np.isnan(hrrr_lon)) & (hrrr_lon < 0) &
                    (~np.isnan(temp)) & (~np.isnan(rh)) & (~np.isnan(pres)) &
                    (~np.isnan(u)) & (~np.isnan(v)) &
                    (temp > -100) & (temp < 120) &
                    (rh >= 0) & (rh <= 110) &
                    (pres > 100) & (pres < 1050) &
                    (u > -100) & (u < 120) &
                    (v > -100) & (v < 120)
                )
                m_temp = within_k_sigma(temp, valid0, k=2.0)
                m_rh   = within_k_sigma(rh,   valid0, k=2.0)
                m_pres = within_k_sigma(pres, valid0, k=2.0)
                m_u    = within_k_sigma(u,    valid0, k=2.0)
                m_v    = within_k_sigma(v,    valid0, k=2.0)
                valid = valid0 & m_temp & m_rh & m_pres & m_u & m_v
    
                hrrr_lat, hrrr_lon = hrrr_lat[valid], hrrr_lon[valid]
                temp, pres, rh = temp[valid], pres[valid], rh[valid]
                u, v = u[valid], v[valid]
                if pres.size == 0:
                    base[atime_key] = np.nan
                    debug_rows.append({
                        'mode': mode,
                        'time': atime_key,
                        'case_id': str(caseIDs[dt]),
                        'clat': float(clat),
                        'clon': float(clon),
                        'Z': float(Z),
                        'Z_legacy': np.nan,
                        'Z_mesonet': z_mesonet,
                        'Z_source': z_source,
                        'rap_init': date_init.strftime('%Y%m%d%H%M'),
                        'source_used': source_used,
                        'n_valid_points': 0,
                        'n_clusters': 0,
                        'flag_missing_grib': False,
                        'flag_no_valid_points': True,
                    })
                    continue
    
                qv  = mpcalc.mixing_ratio_from_relative_humidity(np.array(pres)*units.hPa, np.array(temp)*units.degC, np.array(rh)*units.percent).m
                tv  = mpcalc.virtual_temperature(np.array(temp)*units.degC, qv*units.dimensionless).m
                dew = mpcalc.dewpoint_from_relative_humidity(np.array(temp)*units.degC, np.array(rh)*units.percent).m
                qvs = mpcalc.saturation_mixing_ratio(np.array(pres)*units.hPa, np.array(temp)*units.degC).m
                th  = mpcalc.potential_temperature(np.array(pres)*units.hPa, np.array(temp)*units.degC).m
                thv = mpcalc.virtual_potential_temperature(np.array(pres)*units.hPa, np.array(temp)*units.degC, qv*units.dimensionless).m
                the = mpcalc.equivalent_potential_temperature(np.array(pres)*units.hPa, np.array(temp)*units.degC, np.array(dew)*units.degC).m
                thw = np.array([compute_thw(temp[i], pres[i], rh[i]) for i in range(pres.size)])
    
                dist = np.array([get_dx_dy(clon, clat, hrrr_lon[i], hrrr_lat[i]) for i in range(hrrr_lat.size)])
                hrrr_dx = np.array([val[0] for val in dist])/1000.0
                hrrr_dy = np.array([val[1] for val in dist])/1000.0
                n_valid_points_pre_mask = int(pres.size)
                inflow_empty = False
                inflow_kept = int(pres.size)
                inflow_radius_min = np.nan
                inflow_radius_med = np.nan
                inflow_radius_max = np.nan

                if args.use_inflow_mask:
                    inflow_keep, inflow_rx, inflow_ry, inflow_azimuth, inflow_radius = build_inflow_mask(
                        hrrr_dx, hrrr_dy,
                        u_storm, v_storm,
                        target_angle=normalized_storm_motion,
                        center_angle_deg=args.inflow_center_angle,
                        half_angle_deg=args.inflow_half_angle,
                        inner_radius_km=args.inflow_inner_radius_km,
                        outer_radius_km=args.inflow_outer_radius_km,
                    )
                    inflow_kept = int(np.sum(inflow_keep))
                    inflow_radius_min, inflow_radius_med, inflow_radius_max = stat_triplet(inflow_radius[inflow_keep] if np.any(inflow_keep) else np.array([np.nan]))
                    if inflow_kept == 0:
                        inflow_empty = True
                    else:
                        hrrr_lat, hrrr_lon = hrrr_lat[inflow_keep], hrrr_lon[inflow_keep]
                        temp, pres, rh = temp[inflow_keep], pres[inflow_keep], rh[inflow_keep]
                        u, v = u[inflow_keep], v[inflow_keep]
                        qv, tv, dew, qvs, th, thv, the, thw = (
                            qv[inflow_keep], tv[inflow_keep], dew[inflow_keep], qvs[inflow_keep],
                            th[inflow_keep], thv[inflow_keep], the[inflow_keep], thw[inflow_keep]
                        )
                        hrrr_dx, hrrr_dy = hrrr_dx[inflow_keep], hrrr_dy[inflow_keep]
                if pres.size == 0 or inflow_empty:
                    base[atime_key] = np.nan
                    debug_rows.append({
                        'mode': mode,
                        'time': atime_key,
                        'case_id': str(caseIDs[dt]),
                        'clat': float(clat),
                        'clon': float(clon),
                        'Z': float(Z),
                        'Z_legacy': np.nan,
                        'Z_mesonet': z_mesonet,
                        'Z_source': z_source,
                        'rap_init': date_init.strftime('%Y%m%d%H%M'),
                        'source_used': source_used,
                        'n_valid_points': int(pres.size),
                        'n_valid_points_pre_mask': n_valid_points_pre_mask,
                        'n_valid_points_post_mask': inflow_kept,
                        'n_clusters': 0,
                        'flag_missing_grib': False,
                        'flag_no_valid_points': bool(pres.size == 0),
                        'flag_inflow_mask_applied': bool(args.use_inflow_mask),
                        'flag_inflow_mask_empty': inflow_empty,
                    })
                    continue
    
                all_data = {key: [] for key in ['lat','lon','pres','pres_corr','temp','rh','u','v','qv','dew','th','thv','the','thw']}
                all_data['lat'] = hrrr_lat
                all_data['lon'] = hrrr_lon
                all_data['pres'] = pres
                all_data['pres_corr'] = pres
                all_data['temp'] = temp
                all_data['rh'] = rh
                all_data['u'] = u
                all_data['v'] = v
                all_data['qv'] = qv
                all_data['dew'] = dew
                all_data['th'] = th
                all_data['thv'] = thv
                all_data['the'] = the
                all_data['thw'] = thw
    
                X = np.vstack([hrrr_dx, hrrr_dy]).T
                labels = DBSCAN(eps=30.0, min_samples=1).fit(X).labels_
                cluster_ids = np.unique(labels)
                nC = cluster_ids.size
    
                super_lat  = np.empty(nC)
                super_lon  = np.empty(nC)
                super_dx   = np.empty(nC)
                super_dy   = np.empty(nC)
                super_vars = {k: np.empty(nC) for k in ['pres_corr','pres','temp','rh','u','v','qv','dew','th','thv','the','thw','lat','lon']}
    
                for j, lab in enumerate(cluster_ids):
                    idx = np.where(labels == lab)[0]
                    super_lat[j] = np.nanmean(all_data['lat'][idx])
                    super_lon[j] = np.nanmean(all_data['lon'][idx])
                    super_dx[j], super_dy[j] = np.nanmean(X[idx, :], axis=0)
                    for k in super_vars:
                        super_vars[k][j] = np.nanmean(all_data[k][idx])
    
                r_km = np.hypot(super_dx, super_dy)
                R = float(dx)
                N = len(super_dx)
                Dn = np.sqrt(np.pi * R * R / N)
                k0 = (2.0 * Dn / np.pi) ** 2
                w  = np.exp(-(r_km**2) / (2.0 * k0))
    
                base_pres_corr = wmean(super_vars['pres_corr'], w)
                base_pres = wmean(super_vars['pres'], w)
                base_temp = wmean(super_vars['temp'], w)
                base_rh   = wmean(super_vars['rh'],   w)
                base_u    = wmean(super_vars['u'],    w)
                base_v    = wmean(super_vars['v'],    w)
                base_qv   = wmean(super_vars['qv'],   w)
                base_dew  = wmean(super_vars['dew'],  w)
                base_th   = wmean(super_vars['th'],   w)
                base_thv  = wmean(super_vars['thv'],  w)
                base_the  = wmean(super_vars['the'],  w)
                base_thw  = wmean(super_vars['thw'],  w)
    
                base_dict = {'pres': base_pres, 'pres_corr': base_pres_corr, 'temp': base_temp, 'rh': base_rh,
                             'u': base_u, 'v': base_v, 'qv': base_qv, 'dew': base_dew, 'th': base_th,
                             'thv': base_thv, 'the': base_the, 'thw': base_thw}
    
                base_ru, base_rv = rotate_grid(base_dict['u'], base_dict['v'], u_storm, v_storm,
                                               pivot_x=0, pivot_y=0, target_angle=normalized_storm_motion)
                base_return = {
                    'time': atime, 'pres': base_dict['pres'], 'pres_corr': base_dict['pres_corr'],
                    'temp': base_dict['temp'], 'rh': base_dict['rh'], 'u': base_dict['u'], 'v': base_dict['v'],
                    'ru': base_ru, 'rv': base_rv, 'qv': base_dict['qv'], 'dew': base_dict['dew'],
                    'th': base_dict['th'], 'thv': base_dict['thv'], 'the': base_dict['the'], 'thw': base_dict['thw']
                }
    
                base[atime_key] = base_return
                pres_min, pres_med, pres_max = stat_triplet(pres)
                temp_min, temp_med, temp_max = stat_triplet(temp)
                rh_min, rh_med, rh_max = stat_triplet(rh)
                qv_min, qv_med, qv_max = stat_triplet(qv)
                the_min, the_med, the_max = stat_triplet(the)
                thv_min, thv_med, thv_max = stat_triplet(thv)
                w_min, w_med, w_max = stat_triplet(w)
                rkm_min, rkm_med, rkm_max = stat_triplet(r_km)
                center_km = km_between(float(meso_lon[0]), float(meso_lat[0]), float(meso_lon[-1]), float(meso_lat[-1])) if len(meso_lon) > 1 else 0.0
                debug_rows.append({
                    'mode': mode,
                    'time': atime_key,
                    'case_id': str(caseIDs[dt]),
                    'clat': float(clat),
                    'clon': float(clon),
                    'Z': float(Z),
                    'Z_legacy': np.nan,
                    'Z_mesonet': z_mesonet,
                    'Z_source': z_source,
                    'rap_init': date_init.strftime('%Y%m%d%H%M'),
                    'source_used': source_used,
                    'n_valid_points': int(pres.size),
                    'n_valid_points_pre_mask': n_valid_points_pre_mask,
                    'n_valid_points_post_mask': inflow_kept,
                    'n_clusters': int(nC),
                    'sample_pres_min': pres_min,
                    'sample_pres_med': pres_med,
                    'sample_pres_max': pres_max,
                    'sample_temp_min': temp_min,
                    'sample_temp_med': temp_med,
                    'sample_temp_max': temp_max,
                    'sample_rh_min': rh_min,
                    'sample_rh_med': rh_med,
                    'sample_rh_max': rh_max,
                    'sample_qv_min': qv_min,
                    'sample_qv_med': qv_med,
                    'sample_qv_max': qv_max,
                    'sample_the_min': the_min,
                    'sample_the_med': the_med,
                    'sample_the_max': the_max,
                    'sample_thv_min': thv_min,
                    'sample_thv_med': thv_med,
                    'sample_thv_max': thv_max,
                    'cluster_r_km_min': rkm_min,
                    'cluster_r_km_med': rkm_med,
                    'cluster_r_km_max': rkm_max,
                    'weight_min': w_min,
                    'weight_med': w_med,
                    'weight_max': w_max,
                    'flag_inflow_mask_applied': bool(args.use_inflow_mask),
                    'flag_inflow_mask_empty': inflow_empty,
                    'flag_shared_model_hour': False,
                    'inflow_radius_min': inflow_radius_min,
                    'inflow_radius_med': inflow_radius_med,
                    'inflow_radius_max': inflow_radius_max,
                    'inflow_center_angle': float(args.inflow_center_angle),
                    'inflow_half_angle': float(args.inflow_half_angle),
                    'inflow_inner_radius_km': float(args.inflow_inner_radius_km),
                    'inflow_outer_radius_km': float(args.inflow_outer_radius_km) if np.isfinite(args.inflow_outer_radius_km) else np.nan,
                    'base_pres_corr': float(base_dict['pres_corr']),
                    'base_temp': float(base_dict['temp']),
                    'base_rh': float(base_dict['rh']),
                    'base_qv': float(base_dict['qv']),
                    'base_the': float(base_dict['the']),
                    'base_thv': float(base_dict['thv']),
                    'flag_base_pres_outside_sample': bool((base_dict['pres_corr'] < pres_min) or (base_dict['pres_corr'] > pres_max)),
                    'storm_path_span_km': center_km,
                })
                if args.share_base_by_model_hour:
                    shared_base_cache[shared_key] = dict(base_return)
                    shared_debug_cache[shared_key] = dict(debug_rows[-1])
            else:
                base[atime_key] = np.nan
                debug_rows.append({
                    'mode': mode,
                    'time': atime_key,
                    'case_id': str(caseIDs[dt]),
                    'clat': float(clat),
                    'clon': float(clon),
                    'Z': np.nan,
                    'rap_init': np.nan,
                    'source_used': 'nan_Z',
                    'n_valid_points': 0,
                    'n_clusters': 0,
                    'flag_missing_grib': False,
                    'flag_shared_model_hour': False,
                })

        # === write per-rank, per-mode partial result ===
        rank_out = _pickle_path_from_prefix(output_prefix, f'_{mode}_rank{rank}.pkl')
        with open(rank_out, 'wb') as f:
            pickle.dump(base, f)
        if args.debug_dir:
            debug_rank_out = _debug_pickle_path(args.debug_dir, output_prefix, mode, rank)
            with open(debug_rank_out, 'wb') as f:
                pickle.dump(debug_rows, f)

    # === synchronize ranks before merging ===
    comm.Barrier()

    # === rank 0 merges rank pickles per mode, then merges modes ===
    if rank == 0:
        merged_mode_files = []
        for mode in modes:
            existing_mode = maybe_load_pickle(f'{resume_prefix}_{mode}.pkl') if (args.skip_existing or args.resume_from) else {}
            merged = dict(existing_mode)
            merged_debug_rows = []
            # gather all rank files for this mode
            for r in range(size):
                fn = _pickle_path_from_prefix(output_prefix, f'_{mode}_rank{r}.pkl')
                if os.path.exists(fn):
                    with open(fn, 'rb') as f:
                        try:
                            merged.update(pickle.load(f))
                        except Exception:
                            pass
                if args.debug_dir:
                    dfn = _debug_pickle_path(args.debug_dir, output_prefix, mode, r)
                    if dfn.exists():
                        with open(dfn, 'rb') as f:
                            try:
                                merged_debug_rows.extend(pickle.load(f))
                            except Exception:
                                pass
            mode_out = _pickle_path_from_prefix(output_prefix, f'_{mode}.pkl')
            with open(mode_out, 'wb') as f:
                pickle.dump(merged, f)
            merged_mode_files.append(str(mode_out))
            if args.debug_dir and merged_debug_rows:
                debug_df = pd.DataFrame(merged_debug_rows).sort_values(['time'])
                base_cols = ['base_pres_corr', 'base_temp', 'base_qv', 'base_the', 'base_thv']
                for col in base_cols + ['clat', 'clon', 'Z']:
                    if col not in debug_df:
                        debug_df[col] = np.nan
                debug_df['prev_clat'] = debug_df['clat'].shift(1)
                debug_df['prev_clon'] = debug_df['clon'].shift(1)
                debug_df['prev_Z'] = debug_df['Z'].shift(1)
                for col in base_cols:
                    debug_df[f'prev_{col}'] = debug_df[col].shift(1)
                debug_df['delta_center_km'] = [
                    km_between(lon1, lat1, lon2, lat2) if np.isfinite(lat1) and np.isfinite(lon1) and np.isfinite(lat2) and np.isfinite(lon2) else np.nan
                    for lon1, lat1, lon2, lat2 in zip(debug_df['prev_clon'], debug_df['prev_clat'], debug_df['clon'], debug_df['clat'])
                ]
                debug_df['delta_Z'] = debug_df['Z'] - debug_df['prev_Z']
                debug_df['flag_repeated_state'] = (
                    np.isfinite(debug_df['delta_center_km']) &
                    (debug_df['delta_center_km'] > 5.0) &
                    (np.abs(debug_df['delta_Z']) > 5.0) &
                    np.isclose(debug_df['base_pres_corr'], debug_df['prev_base_pres_corr'], atol=0.01, rtol=0) &
                    np.isclose(debug_df['base_temp'], debug_df['prev_base_temp'], atol=0.01, rtol=0) &
                    np.isclose(debug_df['base_qv'], debug_df['prev_base_qv'], atol=1e-5, rtol=0) &
                    np.isclose(debug_df['base_the'], debug_df['prev_base_the'], atol=0.01, rtol=0) &
                    np.isclose(debug_df['base_thv'], debug_df['prev_base_thv'], atol=0.01, rtol=0)
                )
                debug_csv = _debug_csv_path(args.debug_dir, output_prefix, mode)
                debug_df.to_csv(debug_csv, index=False)
        
        # final combined (T+N) just like your original logic
        all_base = dict(existing_base)
        for file in merged_mode_files:
            with open(file, 'rb') as f:
                all_base.update(pickle.load(f))
        
        final_out = _pickle_path_from_prefix(output_prefix, '.pkl')
        with open(final_out, 'wb') as f:
            pickle.dump(all_base, f)

        print(f"[INFO] Wrote merged per-mode pickles: {merged_mode_files}")
        print(f"[INFO] Wrote final: {final_out}")
