#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import pickle
from glob import glob
from functools import lru_cache
from pathlib import Path
from typing import Optional
import pandas as pd
from utils import interpolate_var,get_dx_dy,get_mm_files,compute_thw,haversine,extract_within_radius,rotate_grid,get_elev_file_vectorized,get_elevation,compute_precipitable_water,derive_ave_elev,moving_stats
from mesonet_reader import readMM, get_file_date_mm, detect_mm_format

# Keep this local so utils.py does not need to house cached file discovery.
from functools import lru_cache as _lru_cache_local

@_lru_cache_local(maxsize=4096)
def get_mm_files_cached(outdir_mm, meso_time):
    """Cached wrapper around utils.get_mm_files for processed NetCDF mesonet files."""
    return get_mm_files(outdir_mm, meso_time)

from pyart.core.transforms import geographic_to_cartesian_aeqd as geo2cart_aeqd
import sharppy.sharptab.profile as profile
import sharppy.sharptab.params as params
from datetime import datetime, timedelta, timezone

import pyart
from pyart.io.nexrad_common import NEXRAD_LOCATIONS
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator as _Interp  # optional, guarded below if you want
from netCDF4 import date2num, num2date
import rasterio
from rasterio.warp import transform as rio_transform
import requests
from tqdm import tqdm

import metpy.calc as mpcalc
from metpy.units import units
import metpy.constants as const

try:
    from mpi4py import MPI
except Exception:
    class _SerialComm:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def Get_size():
            return 1

        @staticmethod
        def Barrier():
            return None

    class _SerialMPI:
        COMM_WORLD = _SerialComm()

    MPI = _SerialMPI()

try:
    import nexradaws
except Exception:
    nexradaws = None

try:
    import numpy.core.numeric as _np_numeric
    sys.modules.setdefault('numpy._core.numeric', _np_numeric)
except Exception:
    pass

_ELEV_WARNED_TILES = set()
_RADAR_WARNED_KEYS = set()

# Wind QC constants. Raw HailCam/LIFT-style files can carry wind speed in mph,
# but all processed NetCDF output is stored in m s-1. Keep the physical upper
# bound tied to the user-facing 150 mph limit.
MPH_TO_MPS = 0.44704
MAX_WSPD_MPH = 150.0
MAX_WSPD_MPS = MAX_WSPD_MPH * MPH_TO_MPS
WIND_SPIKE_WINDOW_SEC = 10.0
WIND_SPIKE_ABS_JUMP_MPS = 25.0 * MPH_TO_MPS
WIND_SPIKE_MAD_SIGMA = 6.0
WIND_SPIKE_MIN_NEIGHBORS = 4


class _RadarScanRef:
    __slots__ = ("key", "scan_time")

    def __init__(self, key, scan_time):
        self.key = key
        self.scan_time = scan_time

def within_k_sigma(arr, base_mask, k=2.5):
    arr = np.asarray(arr, dtype=float)
    subset = arr[base_mask]
    if subset.size < 8:  # guard against tiny samples
        return np.ones_like(arr, dtype=bool)
    m = np.nanmean(subset)
    s = np.nanstd(subset, ddof=1)
    if not np.isfinite(s) or s == 0:
        return np.ones_like(arr, dtype=bool)
    return np.abs(arr - m) <= k * s

def normalize_rh_to_percent(rh):
    rh = np.asarray(rh, dtype=float)
    finite = rh[np.isfinite(rh)]
    finite = finite[np.abs(finite) < 1e20]
    if finite.size == 0:
        return rh
    if np.nanmax(finite) <= 1.5:
        return rh * 100.0
    return rh

def normalize_temp_to_celsius(temp):
    temp = np.asarray(temp, dtype=float)
    finite = temp[np.isfinite(temp)]
    finite = finite[np.abs(finite) < 1e20]
    if finite.size == 0:
        return temp
    if np.nanmedian(finite) > 150.0:
        return temp - 273.15
    return temp

def _to_utc_datetime(val):
    if isinstance(val, np.datetime64):
        val = pd.Timestamp(val).to_pydatetime()
    elif isinstance(val, pd.Timestamp):
        val = val.to_pydatetime()
    if val.tzinfo is None:
        return val.replace(tzinfo=timezone.utc)
    return val.astimezone(timezone.utc)

def _great_circle_distance_m(lon1, lat1, lon2, lat2):
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371000.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def _beam_height_m(range_m, elev_deg, radar_alt_m=0.0, ke=4.0/3.0):
    theta = np.deg2rad(float(elev_deg))
    reff = ke * 6371000.0
    center = np.sqrt(range_m**2 + reff**2 + 2.0 * range_m * reff * np.sin(theta)) - reff
    return center + float(radar_alt_m)

def _radar_site_alt_m(radar_id):
    site = NEXRAD_LOCATIONS.get(radar_id)
    if site is None:
        return np.nan
    elev = float(site.get('elev', np.nan))
    # Py-ART NEXRAD site elevations are stored in feet.
    return elev * 0.3048 if np.isfinite(elev) else np.nan

def _parse_candidate_radars(candidate_radars):
    if candidate_radars is None:
        return []
    if isinstance(candidate_radars, str):
        return [val.strip() for val in candidate_radars.split(',') if val.strip()]
    return [str(val).strip() for val in candidate_radars if str(val).strip()]


def _extract_radar_id_from_path(path_value):
    base = os.path.basename(str(path_value))
    if len(base) >= 4 and base[:4].upper().startswith('K'):
        return base[:4].upper()
    parts = base.replace('-', '_').split('_')
    for part in parts:
        part_u = part.upper()
        if len(part_u) == 4 and part_u.startswith('K'):
            return part_u
    return None

@lru_cache(maxsize=1)
def _get_nexradaws_interface():
    if nexradaws is None:
        raise ImportError('nexradaws is required for AWS-backed radar selection.')
    return nexradaws.NexradAwsInterface()


def _bucket_list_radar_day(radar_id, day_key, timeout=60):
    prefix = f"{day_key[:4]}/{day_key[4:6]}/{day_key[6:8]}/{radar_id}/"
    urls = [
        f"https://unidata-nexrad-level2.s3.amazonaws.com?prefix={prefix}",
        f"https://noaa-nexrad-level2.s3.amazonaws.com?prefix={prefix}",
    ]
    last_exc = None
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            scans = []
            for node in root.findall("s3:Contents", ns):
                key_node = node.find("s3:Key", ns)
                if key_node is None or not key_node.text:
                    continue
                key = key_node.text.strip()
                base = os.path.basename(key)
                try:
                    scan_time = _to_utc_datetime(_radar_file_datetime(base))
                except Exception:
                    scan_time = None
                if scan_time is None:
                    continue
                scans.append(_RadarScanRef(key, scan_time))
            return tuple(sorted(scans, key=lambda s: s.scan_time))
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return tuple()


def _download_radar_volume(scan_key, radar_file_dir, timeout=180):
    if radar_file_dir is None:
        return None
    radar_root = Path(radar_file_dir)
    out_path = radar_root / scan_key
    if out_path.exists():
        return str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://unidata-nexrad-level2.s3.amazonaws.com/{scan_key}",
        f"https://noaa-nexrad-level2.s3.amazonaws.com/{scan_key}",
    ]
    for url in urls:
        try:
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            print(f"[INFO] downloaded radar volume {url}", flush=True)
            return str(out_path)
        except Exception:
            continue

    if scan_key not in _RADAR_WARNED_KEYS:
        print(f"[WARN] failed to download radar volume for {scan_key}", flush=True)
        _RADAR_WARNED_KEYS.add(scan_key)
    return None

@lru_cache(maxsize=None)
def _get_avail_scans_day(radar_id, day_key):
    day = datetime.strptime(day_key, '%Y%m%d')
    try:
        conn = _get_nexradaws_interface()
        return tuple(conn.get_avail_scans(day.year, day.month, day.day, radar_id))
    except Exception:
        return _bucket_list_radar_day(radar_id, day_key)

def _get_scans_in_range(radar_id, start_dt, end_dt):
    scans = []
    start_day = datetime(start_dt.year, start_dt.month, start_dt.day, tzinfo=timezone.utc)
    end_day = datetime(end_dt.year, end_dt.month, end_dt.day, tzinfo=timezone.utc)
    day = start_day
    while day <= end_day:
        scans.extend(_get_avail_scans_day(radar_id, day.strftime('%Y%m%d')))
        day += timedelta(days=1)
    scans = [scan for scan in scans if (scan.scan_time >= start_dt) and (scan.scan_time <= end_dt)]
    scans.sort(key=lambda x: x.scan_time)
    return scans

def _read_radar_volume(scan_key, radar_file_dir):
    local_path = _resolve_local_radar_volume(scan_key, radar_file_dir) if radar_file_dir else None
    if local_path and os.path.exists(local_path):
        return pyart.io.read_nexrad_archive(local_path, delay_field_loading=True)
    local_path = _download_radar_volume(scan_key, radar_file_dir)
    if local_path and os.path.exists(local_path):
        return pyart.io.read_nexrad_archive(local_path, delay_field_loading=True)
    last_err = None
    for _ in range(2):
        try:
            return pyart.io.read_nexrad_archive(
                f's3://unidata-nexrad-level2/{scan_key}',
                delay_field_loading=True,
                storage_options={'anon': True},
            )
        except Exception as exc:
            last_err = exc
    raise last_err


@lru_cache(maxsize=None)
def _resolve_local_radar_volume(scan_key, radar_file_dir):
    if radar_file_dir is None:
        return None
    radar_root = str(radar_file_dir)
    direct = os.path.join(radar_root, scan_key)
    if os.path.exists(direct):
        return direct

    base = os.path.basename(scan_key)
    direct_base = os.path.join(radar_root, base)
    if os.path.exists(direct_base):
        return direct_base

    matches = glob(os.path.join(radar_root, '**', base), recursive=True)
    if matches:
        return matches[0]

    return None


def _radar_file_datetime(fname):
    base = os.path.basename(str(fname))
    parts = base.replace('-', '_').split('_')
    for p in range(len(parts) - 1):
        d, t = parts[p], parts[p + 1]
        if len(d) == 8 and len(t) == 6 and d.isdigit() and t.isdigit():
            return datetime.strptime(d + t, "%Y%m%d%H%M%S")
    try:
        d = base.split('_')[0][-8:]
        t = base.split('_')[1][:6]
        return datetime.strptime(d + t, "%Y%m%d%H%M%S")
    except Exception:
        return None


@lru_cache(maxsize=None)
def _discover_local_radar_files(radar_file_dir, mode, case_index, candidate_radars_key, day_key):
    radar_root = str(radar_file_dir)
    direct = sorted(glob(os.path.join(radar_root, f"{mode}_{case_index}", "*")))
    if direct:
        return tuple(direct)

    candidate_radars = set(_parse_candidate_radars(candidate_radars_key))
    all_files = [p for p in glob(os.path.join(radar_root, "**", "*"), recursive=True) if os.path.isfile(p)]
    if candidate_radars:
        all_files = [p for p in all_files if _extract_radar_id_from_path(p) in candidate_radars]
    if day_key:
        filtered = []
        for p in all_files:
            fdt = _radar_file_datetime(p)
            if fdt is not None and fdt.strftime('%Y%m%d') == day_key:
                filtered.append(p)
        all_files = filtered
    return tuple(sorted(all_files))


def _cache_candidate_radar_volumes(candidate_radars, mtime, radar_file_dir, time_tolerance_sec):
    radars = _parse_candidate_radars(candidate_radars)
    if not radars or radar_file_dir is None or len(mtime) == 0:
        return
    try:
        start_dt = min(_to_utc_datetime(val) for val in np.asarray(mtime)) - timedelta(seconds=int(time_tolerance_sec))
        end_dt = max(_to_utc_datetime(val) for val in np.asarray(mtime)) + timedelta(seconds=int(time_tolerance_sec))
        downloaded = 0
        attempted = 0
        for radar_id in radars:
            scans = _get_scans_in_range(radar_id, start_dt, end_dt)
            print(f"[INFO] radar cache search {radar_id} scans={len(scans)} window={start_dt.isoformat()}..{end_dt.isoformat()}", flush=True)
            for scan in scans:
                attempted += 1
                if _download_radar_volume(scan.key, radar_file_dir) is not None:
                    downloaded += 1
        print(f"[INFO] radar cache attempted={attempted} downloaded_or_present={downloaded}", flush=True)
    except Exception as exc:
        print(f"[WARN] radar cache search/download failed for {radars} in {start_dt.isoformat()}..{end_dt.isoformat()}: {exc!r}", flush=True)
        return


def _get_sweep_datetimes(radar):
    vol0 = datetime.strptime(radar.time["units"].split()[2], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    sweep_times = np.full(radar.nsweeps, None, dtype=object)
    sweep_means = np.full(radar.nsweeps, np.nan, dtype=float)
    for si in range(radar.nsweeps):
        istart = radar.sweep_start_ray_index["data"][si]
        iend = radar.sweep_end_ray_index["data"][si] + 1
        elev = np.asarray(radar.elevation["data"][istart:iend], dtype=float)
        tsec = np.asarray(radar.time["data"][istart:iend], dtype=float)
        sweep_means[si] = np.nanmean(elev)
        sweep_times[si] = vol0 + timedelta(seconds=float(np.nanmedian(tsec)))
    return sweep_times, sweep_means


def _choose_best_sweep_indices(radar, obs_times, elev_angle, elev_tol=0.3):
    obs_times = np.array([_to_utc_datetime(val) for val in np.asarray(obs_times)], dtype=object)
    sweep_times, sweep_means = _get_sweep_datetimes(radar)
    eligible = np.where(np.abs(sweep_means - float(elev_angle)) <= float(elev_tol))[0]
    if eligible.size == 0:
        eligible = np.array([int(np.argmin(np.abs(sweep_means - float(elev_angle))))], dtype=int)
    best = np.full(obs_times.size, -1, dtype=int)
    for i, t_obs in enumerate(obs_times):
        dt_sec = np.array([abs((sweep_times[si] - t_obs).total_seconds()) for si in eligible], dtype=float)
        best[i] = int(eligible[int(np.argmin(dt_sec))])
    return best

def _select_best_radar_scans(mtime, mlat, mlon, melev, candidate_radars, elev_angle, time_tolerance_sec):
    obs_time = np.array([_to_utc_datetime(val) for val in np.asarray(mtime)], dtype=object)
    obs_lat = np.asarray(mlat, dtype=float)
    obs_lon = np.asarray(mlon, dtype=float)
    obs_elev = np.asarray(melev, dtype=float)
    radars = _parse_candidate_radars(candidate_radars)
    if not radars or obs_time.size == 0:
        return {}

    scan_window_start = min(obs_time) - timedelta(seconds=int(time_tolerance_sec))
    scan_window_end = max(obs_time) + timedelta(seconds=int(time_tolerance_sec))

    radar_meta = {}
    for radar_id in radars:
        site = NEXRAD_LOCATIONS.get(radar_id)
        if site is None:
            continue
        scans = _get_scans_in_range(radar_id, scan_window_start, scan_window_end)
        if not scans:
            continue
        scan_times = np.array([scan.scan_time for scan in scans], dtype=object)
        radar_meta[radar_id] = {
            'lat': float(site['lat']),
            'lon': float(site['lon']),
            'alt_m': _radar_site_alt_m(radar_id),
            'scans': scans,
            'scan_times': scan_times,
        }

    groups = {}
    for i, (t_obs, lat_obs, lon_obs, elev_obs) in enumerate(zip(obs_time, obs_lat, obs_lon, obs_elev)):
        best = None
        for radar_id, meta in radar_meta.items():
            scan_times = meta['scan_times']
            idx = np.searchsorted(scan_times, t_obs)
            cand_idxs = []
            if idx < scan_times.size:
                cand_idxs.append(idx)
            if idx > 0:
                cand_idxs.append(idx - 1)
            if not cand_idxs:
                continue
            dt_sec = np.array([abs((scan_times[j] - t_obs).total_seconds()) for j in cand_idxs], dtype=float)
            jbest = cand_idxs[int(np.argmin(dt_sec))]
            best_dt = float(np.min(dt_sec))
            if best_dt > float(time_tolerance_sec):
                continue

            range_m = float(_great_circle_distance_m(meta['lon'], meta['lat'], lon_obs, lat_obs))
            beam_agl = float(_beam_height_m(range_m, elev_angle, radar_alt_m=meta['alt_m']) - elev_obs)
            cand = {
                'radar': radar_id,
                'scan': meta['scans'][jbest],
                'beam_agl': beam_agl,
                'dt_sec': best_dt,
            }
            if (best is None) or (cand['beam_agl'] < best['beam_agl'] - 1e-6) or (abs(cand['beam_agl'] - best['beam_agl']) <= 1e-6 and cand['dt_sec'] < best['dt_sec']):
                best = cand

        if best is not None:
            key = (best['radar'], best['scan'].key)
            groups.setdefault(key, []).append(i)
    return groups

def _storm_track_mode_dir(mode):
    """Return the expected storm-track subdirectory for a transect mode."""
    return "tornadic" if str(mode) in {"T", "L", "S", "W", "Sk"} else "nontornadic"


def _normalize_case_id(caseid, mode=None):
    """Normalize a storm-track case identifier.

    New storm-track files are named with the literal CaseID from
    transect_times.xlsx, e.g. ``T9_20190527_2254_autosave_edited.csv``.
    Older versions of this script passed zero-based numeric indices.  Keep a
    small compatibility layer here, but prefer literal CaseID strings everywhere
    else.
    """
    value = str(caseid).strip()
    if not value:
        raise ValueError("empty storm-track case identifier")
    if re.match(r"^[A-Za-z]+\d+$", value):
        return value.upper()
    if re.match(r"^\d+$", value):
        if mode is None:
            raise ValueError(f"numeric storm-track id {caseid!r} requires mode")
        prefix = "T" if str(mode) in {"T", "L", "S", "W", "Sk"} else "N"
        return f"{prefix}{int(value)}"
    return value.upper()


def _track_column(track_df, candidates, track_path):
    lower_lookup = {str(col).strip().lower(): col for col in track_df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_lookup:
            return lower_lookup[key]
    raise KeyError(
        f"Storm-track file {track_path} is missing one of columns {candidates}; "
        f"available columns={list(track_df.columns)}"
    )


def _parse_track_times(values):
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"storm-track file contains {bad} unparseable timestamp(s)")
    return np.array([
        val.to_pydatetime().astimezone(timezone.utc).replace(tzinfo=None)
        for val in ts
    ], dtype=object)


def _find_storm_track_file(caseid, tracking_dir, mode):
    case_id = _normalize_case_id(caseid, mode)
    tracking_path = Path(tracking_dir)
    expected_dir = tracking_path / _storm_track_mode_dir(mode)

    patterns = []
    if expected_dir.exists():
        patterns.append(expected_dir / f"{case_id}_*.csv")
    patterns.extend([
        tracking_path / "tornadic" / f"{case_id}_*.csv",
        tracking_path / "nontornadic" / f"{case_id}_*.csv",
        tracking_path / f"{case_id}_*.csv",
    ])

    matches = []
    seen = set()
    for pattern in patterns:
        for match in sorted(glob(str(pattern))):
            resolved = str(Path(match).resolve())
            if resolved not in seen:
                seen.add(resolved)
                matches.append(Path(match))

    if not matches:
        searched = ", ".join(str(p) for p in patterns)
        raise FileNotFoundError(
            f"No storm-track file found for CaseID={case_id!r}, mode={mode!r}. "
            f"Searched: {searched}"
        )
    if len(matches) > 1:
        msg = "\n  ".join(str(m) for m in matches)
        raise FileExistsError(
            f"Multiple storm-track files found for CaseID={case_id!r}. "
            f"Refusing to guess. Matches:\n  {msg}"
        )
    return matches[0]


@lru_cache(maxsize=None)
def _load_storm_track(caseid, tracking_dir, mode):
    track_path = _find_storm_track_file(caseid, tracking_dir, mode)
    strk = pd.read_csv(track_path)

    time_col = _track_column(strk, ["time", "datetime", "date_time"], track_path)
    lat_col = _track_column(strk, ["lat", "latitude"], track_path)
    lon_col = _track_column(strk, ["lon", "longitude"], track_path)

    mtimes = _parse_track_times(strk[time_col].values)
    srlat = strk[lat_col].values.astype(float)
    srlon = strk[lon_col].values.astype(float)

    isort = np.argsort(mtimes)
    return mtimes[isort], srlat[isort], srlon[isort]


@lru_cache(maxsize=None)
def _load_interp_storm_track(caseid, tracking_dir, mode, dt_tracking):
    mtimes, srlat, srlon = _load_storm_track(caseid, tracking_dir, mode)
    time_grid = np.arange(
        date2num(mtimes[0], 'seconds since 1970-01-01'),
        date2num(mtimes[-1], 'seconds since 1970-01-01'),
        dt_tracking * 60,
    )
    srlat_i = interpolate_var(srlat, date2num(mtimes, 'seconds since 1970-01-01'), time_grid)
    srlon_i = interpolate_var(srlon, date2num(mtimes, 'seconds since 1970-01-01'), time_grid)
    mtimes_i = num2date(time_grid, 'seconds since 1970-01-01')
    mtimes_i = np.array(
        [datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in mtimes_i],
        dtype=object,
    )
    return mtimes_i, srlat_i, srlon_i


def _read_transects_table(path_value) -> pd.DataFrame:
    """Read transect timing table from CSV or Excel.

    Supports:
      - .csv
      - .xlsx / .xls, using sheet 'transect_times' when present,
        otherwise the first sheet.
    """
    path = Path(path_value)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path).copy()

    if suffix in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet = "transect_times" if "transect_times" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet).copy()

    raise ValueError(f"Unsupported transect table format: {path}. Expected .csv, .xlsx, or .xls.")

def get_t0s(file_path,mode,long_track_length=None,wide_thresh=None):
    tr = _read_transects_table(file_path)
    if mode == 'T':
        tr = tr.loc[tr.Mode=='T']
        amode = 'T'
    elif mode == 'N':
        tr = tr.loc[tr.Mode=='N0']
        amode = 'N'
    elif mode == 'W':
        tr = tr.loc[(tr.Mode=='T')&(tr['Width [m]']>=wide_thresh)]
        amode = 'T'
    elif mode == 'Sk':
        tr = tr.loc[(tr.Mode=='T')&(tr['Width [m]']<wide_thresh)]
        amode = 'T'
    elif mode == 'L':
        tr = tr.loc[(tr.Mode=='T')&(tr['Length [km]']>=long_track_length)]
        amode = 'T'
    elif mode == 'S':
        tr = tr.loc[(tr.Mode=='T')&(tr['Length [km]']<long_track_length)]
        amode = 'T'
        
    t0 = np.array([datetime.strptime(val,'%Y%m%d-%H%M') for val in tr['Tstart'].values])
    return t0, amode, tr

def parse_storm_tracks(x,tracking_dir,mode,atime,dt_tracking=1/60,time_to_space_conversion=10):
    mtimes, srlat, srlon = _load_interp_storm_track(x, tracking_dir, mode, float(dt_tracking))

    #Index the time to get the actual time bounds
    imeso = np.where((mtimes>=atime-timedelta(seconds=int((time_to_space_conversion*60)/2)))&(mtimes<=atime+timedelta(seconds=int((time_to_space_conversion*60)/2))))[0]
    mtimes,srlat,srlon = mtimes[imeso],srlat[imeso],srlon[imeso]
    
    isort = np.argsort(mtimes)
    mtimes,srlat,srlon = mtimes[isort],srlat[isort],srlon[isort]
    
    #Compute the storm motion during this period
    dx_storm,dy_storm = get_dx_dy(srlon[0],srlat[0],srlon[-1],srlat[-1])
    dt_storm = (mtimes[-1]-mtimes[0]).total_seconds()
    u_storm,v_storm = dx_storm/dt_storm, dy_storm/dt_storm

    return mtimes,srlat,srlon,u_storm,v_storm

def compute_distances(meso_times, meso_lats, meso_lons, point_times, point_lats, point_lons, max_time_diff=60):
    """
    Compute distances from each point to the mesocyclone location at the closest time step.

    Parameters:
    - meso_times (array-like): Timestamps of mesocyclone locations.
    - meso_lats (array-like): Latitudes of mesocyclone locations.
    - meso_lons (array-like): Longitudes of mesocyclone locations.
    - point_times (array-like): Timestamps of points.
    - point_lats (array-like): Latitudes of points.
    - point_lons (array-like): Longitudes of points.
    - max_time_diff (int): Maximum allowed time difference in seconds.

    Returns:
    - distances (array): Distances (km) between mesocyclone and points.
    - matched_times (array): Matched mesocyclone times for each point.
    """

    distances = []
    matched_times = []

    # Convert times to pandas datetime for easy matching
    meso_times = pd.to_datetime(meso_times)
    point_times = pd.to_datetime(point_times)

    for p_time, p_lat, p_lon in zip(point_times, point_lats, point_lons):
        # Find the mesocyclone time closest to the point time
        time_diffs = np.abs((meso_times - p_time).total_seconds())
        closest_idx = np.argmin(time_diffs)
        
        # Check if the time difference is within the allowed window
        if time_diffs[closest_idx] <= max_time_diff:
            matched_times.append(meso_times[closest_idx])
            
            # Compute distance
            meso_loc = (meso_lats[closest_idx], meso_lons[closest_idx])
            point_loc = (p_lat, p_lon)
            distance_km = get_dx_dy(meso_lons[closest_idx],meso_lats[closest_idx],p_lon,p_lat)
            distances.append(distance_km)
        else:
            # If no close time match, store NaN
            matched_times.append(np.nan)
            distances.append([np.nan,np.nan])

    return np.array(distances)

def get_sticknet_elevation(atime,sticknet_dir,meso_time, meso_lat, meso_lon, max_distance_range=50):
    stick_files = sorted(glob(sticknet_dir+'/*{}*.nc*'.format(atime.strftime('%Y%m%d'))))
    all_data = {key: [] for key in ['lat', 'lon', 'elev', 'dx','dy']}
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    
    for fi in stick_files:
        with xr.open_dataset(fi) as m:
            tm = m.time.values
            mask = (tm >= date2num(atime - time_window, 'seconds since 1970-01-01')) & (tm <= date2num(atime + time_window,'seconds since 1970-01-01'))
            
            elev,lat,lon,tm = m.alt.sel(time=mask).values,m.lat.sel(time=mask).values,m.lon.sel(time=mask).values,num2date(m.time.sel(time=mask).values,'seconds since 1970-01-01',only_use_cftime_datetimes=False)
            lon = np.array([-val if val>0 else val for val in lon])
            valid = (~np.isnan(elev)) & (~np.isnan(lat)) & (~np.isnan(lon)) & (lon<0)
            
            if np.any(valid):
                dist = compute_distances(meso_time, meso_lat, meso_lon, tm[valid], lat[valid], lon[valid])
                all_data['dx'].append(dist[:,0]/1000)
                all_data['dy'].append(dist[:,1]/1000)
            
                all_data['lat'].append(lat[valid])
                all_data['lon'].append(lon[valid])
                all_data['elev'].append(elev[valid])
    
    #Flatten all arrays
    for ky in list(all_data.keys()):
        all_data[ky] = np.array([x for l in all_data[ky] for x in l])
    
    #Get the elevation values from the sticknets
    dist = np.sqrt((all_data['dx']**2)+(all_data['dy']**2))
    istick = np.where(dist<=max_distance_range)[0]
    stick_elev = all_data['elev'][istick]
    
    return stick_elev

    
def get_reflectivity_along_path(
    x, mtime, mlat, mlon, radar_file_dir, mode,
    candidate_radars=None,
    obs_elev=None,
    elev_angle=0.5,
    time_tolerance_sec=360,         # max |Δt| between obs and sweep (sec)
    search_radius_m=500.0,          # neighborhood radius (meters)
    neighborhood_shape="circle",    # "circle" or "square"
    min_gates=1,                    # require at least this many gates; else fallback/NaN
    fallback_to_nearest=True        # if no gates in neighborhood, use nearest gate
):
    """
    Median dBZ around each point, using the nearest-in-time 0.5° sweep from the nearest
    radar files on disk (already downloaded).

    Returns
    -------
    mesonet_ref : (N,) array of float
        Median reflectivity (dBZ) per observation (NaN where unavailable).
    """

    mtime = np.asarray(mtime)
    mlat  = np.asarray(mlat, dtype=float)
    mlon  = np.asarray(mlon, dtype=float)
    obs_elev = None if obs_elev is None else np.asarray(obs_elev, dtype=float)
    case_key = min(mtime).strftime('%Y%m%d%H%M') if len(mtime) else 'unknown'
    chosen_groups = {}
    chosen_group_count = 0
    chosen_read_success = 0
    if candidate_radars is not None and obs_elev is not None:
        try:
            chosen_groups = _select_best_radar_scans(mtime, mlat, mlon, obs_elev, candidate_radars, elev_angle, time_tolerance_sec)
            chosen_group_count = len(chosen_groups)
        except Exception:
            chosen_groups = {}

    if chosen_groups:
        mesonet_ref = np.full(mtime.size, np.nan, dtype=float)
        for (_radar_id, scan_key), idxs in chosen_groups.items():
            idxs = np.asarray(idxs, dtype=int)
            try:
                radar = _read_radar_volume(scan_key, radar_file_dir)
                chosen_read_success += 1
            except Exception:
                continue

            sweep_for_obs = _choose_best_sweep_indices(radar, mtime[idxs], elev_angle)
            unique_sweeps = np.unique(sweep_for_obs[sweep_for_obs >= 0])

            rlon = float(radar.longitude["data"][0])
            rlat = float(radar.latitude["data"][0])
            px_all, py_all = geo2cart_aeqd(mlon[idxs], mlat[idxs], rlon, rlat)

            for si in unique_sweeps:
                local = np.where(sweep_for_obs == si)[0]
                if local.size == 0:
                    continue

                ref = None
                for fname in ("reflectivity", "DBZ", "REF", "CZ"):
                    if fname in radar.fields:
                        try:
                            ref = radar.get_field(sweep=int(si), field_name=fname)
                            break
                        except Exception:
                            pass
                if ref is None:
                    continue

                gx, gy, _gz = radar.get_gate_x_y_z(sweep=int(si))
                gx_flat = gx.ravel().astype(float)
                gy_flat = gy.ravel().astype(float)

                if np.ma.isMaskedArray(ref):
                    ref_flat = ref.filled(np.nan).ravel().astype(float)
                else:
                    ref_flat = np.asanyarray(ref).ravel().astype(float)
                ref_flat[(ref_flat < -32.0) | (ref_flat > 100.0)] = np.nan

                tree = cKDTree(np.c_[gx_flat, gy_flat])
                px = px_all[local]
                py = py_all[local]

                if neighborhood_shape.lower() == "circle":
                    neigh = tree.query_ball_point(np.c_[px, py], r=search_radius_m)
                    for jlocal, nbr in enumerate(neigh):
                        out_idx = idxs[local[jlocal]]
                        if len(nbr) >= min_gates:
                            vals = ref_flat[nbr]
                            if np.isfinite(vals).any():
                                mesonet_ref[out_idx] = np.nanmedian(vals)
                                continue
                        if fallback_to_nearest:
                            _d, j = tree.query([px[jlocal], py[jlocal]], k=1)
                            mesonet_ref[out_idx] = ref_flat[j] if np.isfinite(ref_flat[j]) else np.nan
                elif neighborhood_shape.lower() == "square":
                    r = float(search_radius_m)
                    neigh = tree.query_ball_point(np.c_[px, py], r=r*np.sqrt(2))
                    for jlocal, nbr in enumerate(neigh):
                        out_idx = idxs[local[jlocal]]
                        if not nbr:
                            if fallback_to_nearest:
                                _d, j = tree.query([px[jlocal], py[jlocal]], k=1)
                                mesonet_ref[out_idx] = ref_flat[j] if np.isfinite(ref_flat[j]) else np.nan
                            continue
                        dx = gx_flat[nbr] - px[jlocal]
                        dy = gy_flat[nbr] - py[jlocal]
                        keep = np.where((np.abs(dx) <= r) & (np.abs(dy) <= r))[0]
                        if keep.size >= min_gates:
                            vals = ref_flat[np.array(nbr)[keep]]
                            if np.isfinite(vals).any():
                                mesonet_ref[out_idx] = np.nanmedian(vals)
                            elif fallback_to_nearest:
                                _d, j = tree.query([px[jlocal], py[jlocal]], k=1)
                                mesonet_ref[out_idx] = ref_flat[j] if np.isfinite(ref_flat[j]) else np.nan
                        elif fallback_to_nearest:
                            _d, j = tree.query([px[jlocal], py[jlocal]], k=1)
                            mesonet_ref[out_idx] = ref_flat[j] if np.isfinite(ref_flat[j]) else np.nan
                else:
                    del radar
                    raise ValueError("neighborhood_shape must be 'circle' or 'square'.")

            del radar
        if np.isfinite(mesonet_ref).any():
            return mesonet_ref
        # If candidate-radar selection succeeded but none of the chosen
        # volumes could actually be read from local cache/S3, fall back to
        # the case-local radar file search below instead of returning all NaN.

    # -- 1) Discover candidate radar files near the observation period
    day_key = datetime(min(mtime).year, min(mtime).month, min(mtime).day).strftime('%Y%m%d') if len(mtime) else None
    nexrad_files = list(_discover_local_radar_files(radar_file_dir, mode, x, candidate_radars if candidate_radars is not None else "", day_key))
    if not nexrad_files and candidate_radars is not None:
        _cache_candidate_radar_volumes(candidate_radars, mtime, radar_file_dir, time_tolerance_sec)
        nexrad_files = list(_discover_local_radar_files(radar_file_dir, mode, x, candidate_radars if candidate_radars is not None else "", day_key))
    if not nexrad_files:
        print(f"[WARN] no radar files found case={mode}_{x} key={min(mtime).strftime('%Y%m%d%H%M') if len(mtime) else 'unknown'} radars={candidate_radars}", flush=True)
        return np.full(mtime.size, np.nan, dtype=float)
    candidate_radar_ids = set(_parse_candidate_radars(candidate_radars))
    if candidate_radar_ids:
        filtered = [f for f in nexrad_files if _extract_radar_id_from_path(f) in candidate_radar_ids]
        if filtered:
            nexrad_files = filtered

    file_times = np.array([_radar_file_datetime(f) for f in nexrad_files], dtype=object)
    good_files = [f for f, ft in zip(nexrad_files, file_times) if ft is not None]
    if not good_files:
        return np.full(mtime.size, np.nan, dtype=float)

    # Pick a subset of files that bracket the obs times
    # (unique searchsorted indices gives a small representative set)
    ft = np.array([_radar_file_datetime(f) for f in good_files])
    ft_sort_idx = np.argsort(ft)
    good_files = list(np.array(good_files)[ft_sort_idx])
    ft = ft[ft_sort_idx]
    # map each obs to the previous file index
    idx_prev = np.clip(np.searchsorted(ft, mtime, side="right") - 1, 0, len(ft)-1)
    idx_next = np.clip(idx_prev + 1, 0, len(ft)-1)
    candidate_files = np.unique(np.r_[idx_prev, idx_next])
    files_for_obs = np.array(good_files)[candidate_files]

    # -- 2) For those files, find all ~0.5° sweeps and their times
    sweep_times = []     # list of datetimes
    sweep_nums = []      # list of ints
    sweep_files = []     # list of file paths

    for rf in files_for_obs:
        try:
            radar = pyart.io.read_nexrad_archive(rf)
        except Exception:
            continue

        # base time of volume
        vol0 = datetime.strptime(radar.time["units"].split()[2], "%Y-%m-%dT%H:%M:%SZ")

        # for each sweep, compute mean elevation and mean time
        for si in range(radar.nsweeps):
            # elevation per ray in this sweep
            istart = radar.sweep_start_ray_index["data"][si]
            iend   = radar.sweep_end_ray_index["data"][si] + 1
            elev = np.asarray(radar.elevation["data"][istart:iend])

            if np.isclose(np.nanmean(elev), elev_angle, atol=0.2):
                # mean time (sec offset) for rays in sweep
                tsec = np.asarray(radar.time["data"][istart:iend])
                tdt  = vol0 + timedelta(seconds=float(np.nanmedian(tsec)))
                sweep_times.append(tdt)
                sweep_nums.append(si)
                sweep_files.append(rf)

        # free memory early
        del radar

    local_sweep_count = len(sweep_times)
    if not sweep_times:
        print(f"[WARN] no low-elevation sweeps found case={mode}_{x} key={min(mtime).strftime('%Y%m%d%H%M') if len(mtime) else 'unknown'} files={len(good_files)} radars={candidate_radars}", flush=True)
        return np.full(mtime.size, np.nan, dtype=float)

    sweep_times = np.array(sweep_times)
    sweep_nums  = np.array(sweep_nums, dtype=int)
    sweep_files = np.array(sweep_files, dtype=object)

    # -- 3) Match each observation time to the local sweep that best preserves
    # the low-beam candidate-radar logic: minimize beam height AGL first, then
    # time offset.
    sort_idx = np.argsort(sweep_times)
    st_sorted = sweep_times[sort_idx]
    sf_sorted = sweep_files[sort_idx]
    sn_sorted = sweep_nums[sort_idx]
    matched_files = np.full(mtime.size, None, dtype=object)
    matched_sweeps = np.full(mtime.size, -1, dtype=int)
    if candidate_radar_ids and obs_elev is not None:
        for i, (t_obs, lat_obs, lon_obs, elev_obs) in enumerate(zip(mtime, mlat, mlon, obs_elev)):
            best = None
            for rf, stime, si in zip(sf_sorted, st_sorted, sn_sorted):
                dt_this = abs((stime - t_obs) / timedelta(seconds=1))
                if dt_this > time_tolerance_sec:
                    continue
                radar_id = _extract_radar_id_from_path(rf)
                if radar_id is None:
                    continue
                site = NEXRAD_LOCATIONS.get(radar_id)
                if site is None:
                    continue
                range_m = float(_great_circle_distance_m(float(site['lon']), float(site['lat']), lon_obs, lat_obs))
                beam_agl = float(_beam_height_m(range_m, elev_angle, radar_alt_m=_radar_site_alt_m(radar_id)) - elev_obs)
                cand = (beam_agl, dt_this, rf, si)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            if best is not None:
                matched_files[i] = best[2]
                matched_sweeps[i] = int(best[3])
    else:
        # find nearest index in sorted sweep times
        ii = np.searchsorted(st_sorted, mtime)
        ii = np.clip(ii, 1, len(st_sorted)-1)
        left  = ii - 1
        right = ii
        use_right = (np.abs(mtime - st_sorted[left]) > np.abs(st_sorted[right] - mtime))
        nearest = np.where(use_right, right, left)
        dt_sec = np.abs((st_sorted[nearest] - mtime) / timedelta(seconds=1))
        ok = dt_sec <= time_tolerance_sec
        matched_files[ok] = sf_sorted[nearest[ok]]
        matched_sweeps[ok] = sn_sorted[nearest[ok]]

    # -- 4) Group obs by (file, sweep) so we open each file once
    mesonet_ref = np.full(mtime.size, np.nan, dtype=float)
    # dictionary: (filepath, sweep) -> indices
    groups = {}
    for i in np.where(matched_sweeps >= 0)[0]:
        key = (matched_files[i], int(matched_sweeps[i]))
        groups.setdefault(key, []).append(i)

    # -- 5) Process each (file, sweep): build KDTree in meters and query neighborhoods
    for (rf, si), idxs in groups.items():
        if rf is None or si < 0:
            continue
        idxs = np.asarray(idxs, dtype=int)

        try:
            radar = pyart.io.read_nexrad_archive(rf)
        except Exception:
            continue

        # reflectivity field name: try common aliases
        ref = None
        for fname in ("reflectivity", "DBZ", "REF", "CZ"):
            if fname in radar.fields:
                try:
                    ref = radar.get_field(sweep=si, field_name=fname)
                    break
                except Exception:
                    pass
        if ref is None:
            del radar
            continue

        # gate geometry in meters (relative to radar)
        gx, gy, _gz = radar.get_gate_x_y_z(sweep=si)
        gx_flat = gx.ravel().astype(float)
        gy_flat = gy.ravel().astype(float)

        # reflectivity values aligned with gates
        if np.ma.isMaskedArray(ref):
            ref_flat = ref.filled(np.nan).ravel().astype(float)
        else:
            ref_flat = np.asanyarray(ref).ravel().astype(float)
        
        # keep a plausible dBZ range; tweak as you like
        ref_flat[(ref_flat < -32.0) | (ref_flat > 100.0)] = np.nan
        
        # build KDTree once per sweep
        tree = cKDTree(np.c_[gx_flat, gy_flat])

        # convert obs lat/lon to local meters relative to radar site
        rlon = float(radar.longitude["data"][0])
        rlat = float(radar.latitude["data"][0])
        # aeqd projection: returns (x, y) in meters
        px, py = geo2cart_aeqd(mlon[idxs], mlat[idxs], rlon, rlat)
        
        if neighborhood_shape.lower() == "circle":
            # list of neighbor indices per point within radius
            neigh = tree.query_ball_point(np.c_[px, py], r=search_radius_m)
            for k, nbr in enumerate(neigh):
                if len(nbr) >= min_gates:
                    vals = ref_flat[nbr]
                    if np.isfinite(vals).any():
                        mesonet_ref[idxs[k]] = np.nanmedian(vals)
                        continue
                if fallback_to_nearest:
                    d, j = tree.query([px[k], py[k]], k=1)
                    mesonet_ref[idxs[k]] = ref_flat[j] if np.isfinite(ref_flat[j]) else np.nan
                # else remains NaN

        elif neighborhood_shape.lower() == "square":
            # do a circle pre-filter, then keep gates with |dx|<=r & |dy|<=r (square of side 2r)
            r = float(search_radius_m)
            neigh = tree.query_ball_point(np.c_[px, py], r=r*np.sqrt(2))  # superset
            for k, nbr in enumerate(neigh):
                if not nbr:
                    if fallback_to_nearest:
                        d, j = tree.query([px[k], py[k]], k=1)
                        mesonet_ref[idxs[k]] = ref_flat[j] if np.isfinite(ref_flat[j]) else np.nan
                    continue
                dx = gx_flat[nbr] - px[k]
                dy = gy_flat[nbr] - py[k]
                keep = np.where((np.abs(dx) <= r) & (np.abs(dy) <= r))[0]
                if keep.size >= min_gates:
                    vals = ref_flat[np.array(nbr)[keep]]
                    mesonet_ref[idxs[k]] = np.nanmedian(vals) if np.isfinite(vals).any() else (ref_flat[np.array(nbr)[keep][0]] if fallback_to_nearest else np.nan)
                elif fallback_to_nearest:
                    d, j = tree.query([px[k], py[k]], k=1)
                    mesonet_ref[idxs[k]] = ref_flat[j] if np.isfinite(ref_flat[j]) else np.nan
                # else remains NaN
        else:
            raise ValueError("neighborhood_shape must be 'circle' or 'square'.")

        del radar  # free memory

    if not np.isfinite(mesonet_ref).any():
        print(
            f"[WARN] reflectivity all-NaN case={mode}_{x} key={min(mtime).strftime('%Y%m%d%H%M') if len(mtime) else 'unknown'} "
            f"radars={candidate_radars} chosen_groups={chosen_group_count} chosen_reads={chosen_read_success} "
            f"local_files={len(good_files)} local_sweeps={local_sweep_count}"
        , flush=True)
    return mesonet_ref

def keep_moving_points(lat, lon, time, min_move_m=30.0, min_speed_mps=0.5):
    lat = np.asarray(lat); lon = np.asarray(lon); t = np.asarray(time)

    # seconds between samples (works for numpy datetime64 or datetime)
    if np.issubdtype(t.dtype, np.datetime64):
        dt = (t[1:] - t[:-1]) / np.timedelta64(1, 's')
    else:
        dt = np.diff([pd.Timestamp(v).to_datetime64() for v in t]) / np.timedelta64(1, 's')

    # consecutive distances in meters (use your haversine)
    d = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speed = d / np.maximum(dt, 1e-6)

    move = (d >= min_move_m) | (speed >= min_speed_mps)
    keep = np.r_[True, move]  # keep first point, then only moving ones
    return keep


def _interp_unique(x_new, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 2:
        return np.full(np.asarray(x_new).shape, np.nan, dtype=float)
    order = np.argsort(x[good])
    xs = x[good][order]
    ys = y[good][order]
    unique = np.r_[True, np.diff(xs) > 0]
    xs = xs[unique]
    ys = ys[unique]
    if xs.size < 2:
        return np.full(np.asarray(x_new).shape, np.nan, dtype=float)
    return np.interp(np.asarray(x_new, dtype=float), xs, ys, left=np.nan, right=np.nan)


def accelerating_vehicle_wind_mask(lat, lon, time, dv_thresh_mps=0.4, dv_window_sec=6.0, use_vector=True):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    t = np.asarray(time, dtype=object)
    if lat.size < 3:
        return np.zeros(lat.size, dtype=bool)

    tq = np.array([pd.Timestamp(val).timestamp() for val in t], dtype=float)
    seg_t = []
    veh_u = []
    veh_v = []
    veh_spd = []
    for i in range(lat.size - 1):
        dt = tq[i + 1] - tq[i]
        if not np.isfinite(dt) or dt <= 0:
            continue
        dx, dy = get_dx_dy(lon[i], lat[i], lon[i + 1], lat[i + 1])
        seg_t.append((tq[i] + tq[i + 1]) / 2.0)
        veh_u.append(dx / dt)
        veh_v.append(dy / dt)
        veh_spd.append(np.hypot(dx, dy) / dt)
    if len(seg_t) < 2:
        return np.zeros(lat.size, dtype=bool)

    seg_t = np.asarray(seg_t, dtype=float)
    if use_vector:
        uq = _interp_unique(tq, seg_t, veh_u)
        vq = _interp_unique(tq, seg_t, veh_v)
        uq_prev = _interp_unique(tq - float(dv_window_sec), tq, uq)
        vq_prev = _interp_unique(tq - float(dv_window_sec), tq, vq)
        metric = np.sqrt((uq - uq_prev) ** 2 + (vq - vq_prev) ** 2)
    else:
        sq = _interp_unique(tq, seg_t, veh_spd)
        sq_prev = _interp_unique(tq - float(dv_window_sec), tq, sq)
        metric = np.abs(sq - sq_prev)
    return np.isfinite(metric) & (metric > float(dv_thresh_mps))

def interpolate_profile_on_pressure(prof, dp_hPa=0.2, pmin_hPa=None, pmax_hPa=None, return_profile=True):
    """
    Resample a SHARPpy Profile onto a uniform pressure grid.

    Parameters
    ----------
    prof : SHARPpy profile-like object with arrays:
        pres (hPa), tmpc (C), dwpc (C), hght (m), wspd (kt or m/s), wdir (deg)
        (wspd can be m/s or kt; we’ll try to detect)
    dp_hPa : float
        Pressure increment for target grid (e.g., 0.2 hPa)
    pmin_hPa, pmax_hPa : float or None
        Limits for new grid. Defaults to [min(p), max(p)] within data range.
    return_profile : bool
        If True, returns a new sharppy Profile; else returns a dict of arrays.

    Returns
    -------
    new_prof or dict with keys: pres, tmpc, dwpc, hght, wspd, wdir, u, v
    """
    # pull source arrays
    p = np.asarray(prof.pres, dtype=float)       # hPa
    T = np.asarray(prof.tmpc, dtype=float)       # C
    Td = np.asarray(getattr(prof, 'dwpc', np.full_like(T, np.nan)), dtype=float)
    z = np.asarray(getattr(prof, 'hght', np.full_like(T, np.nan)), dtype=float)

    # wind: try wspd in m/s; if looks like knots, convert
    wspd = np.asarray(getattr(prof, 'wspd', np.full_like(T, np.nan)), dtype=float)
    wdir = np.asarray(getattr(prof, 'wdir', np.full_like(T, np.nan)), dtype=float)
    # crude unit check: if median wspd > 120, assume knots
    if np.nanmedian(wspd) > 120:
        wspd = wspd * 0.514444  # kt -> m/s

    # build uniform pressure grid (descending, surface to top)
    p_valid = p[np.isfinite(p)]
    pmax = np.nanmax(p_valid) if pmax_hPa is None else pmax_hPa
    pmin = np.nanmin(p_valid) if pmin_hPa is None else pmin_hPa
    if pmax < pmin:
        pmax, pmin = pmin, pmax
    p_new = np.arange(pmax, pmin - 1e-6, -float(dp_hPa))
    # Interpolators want ascending x: use reversed arrays for x=p
    # We’ll pass x ascending via -p (or just reverse arrays and x_new).

    # interpolate scalars vs pressure
    # Use mixing ratio instead of RH for moisture interpolation
    w_gkg = _mixing_ratio_from_Td_p_gkg(Td, p)

    T_new   = _interp_1d(p[::-1], T[::-1], p_new)        # linear in T(C)
    w_new   = _interp_1d(p[::-1], w_gkg[::-1], p_new)     # linear in w(g/kg)
    Td_new  = _Td_from_mixing_ratio_p(w_new, p_new)       # back to dewpoint
    z_new   = _interp_1d(p[::-1], z[::-1], p_new)

    # winds: interpolate in u/v
    u, v = _to_uv(wspd, wdir)
    u_new = _interp_1d(p[::-1], u[::-1], p_new)
    v_new = _interp_1d(p[::-1], v[::-1], p_new)
    wspd_new, wdir_new = _to_wind(u_new, v_new)

    out = {
        'pres': p_new,
        'tmpc': T_new,
        'dwpc': Td_new,
        'hght': z_new,
        'u': u_new,
        'v': v_new,
        'wspd_ms': wspd_new,
        'wdir_deg': wdir_new,
    }

    if not return_profile:
        return out

    # Build a new SHARPpy Profile if available
    try:
        from sharppy.sharptab.profile import Profile
        # SHARPpy expects wind speed in knots
        wspd_kt = wspd_new / 0.514444
        new_prof = profile.create_profile(profile='default',pres=out['pres'], hght=out['hght']-out['hght'][0],
                           tmpc=out['tmpc'], dwpc=out['dwpc'],
                           wspd=wspd_kt, wdir=out['wdir_deg'],
                           strictQC=False, missing=np.nan)
        return new_prof
    except Exception:
        # Fall back to dict if SHARPpy not importable
        return out

try:
    from scipy.interpolate import PchipInterpolator as _Interp
    _HAVE_PCHIP = True
except Exception:
    _HAVE_PCHIP = False

# --- helpers ---
_EPS = 0.62197  # Rd/Rv

def _dewpoint_from_e_hPa(e_hPa):
    # Magnus over water, inverse
    ln_ratio = np.log(np.maximum(e_hPa, 1e-6) / 6.112)
    Td = (243.5 * ln_ratio) / (17.67 - ln_ratio)
    return Td

def _e_from_dewpoint_hPa(Td_C):
    # Magnus over water
    return 6.112 * np.exp((17.67 * Td_C) / (Td_C + 243.5))

def _mixing_ratio_from_Td_p_gkg(Td_C, p_hPa):
    e = _e_from_dewpoint_hPa(Td_C)  # hPa
    return 1000.0 * (_EPS * e / np.maximum(p_hPa - e, 1e-6))  # g/kg

def _Td_from_mixing_ratio_p(w_gkg, p_hPa):
    w = np.maximum(w_gkg, 0.0) / 1000.0
    e = (w * p_hPa) / (_EPS + w)  # hPa
    return _dewpoint_from_e_hPa(e)

def _to_uv(wspd_ms, wdir_deg):
    # meteorological to u/v (from-direction)
    rad = np.deg2rad(wdir_deg)
    u = -wspd_ms * np.sin(rad)
    v = -wspd_ms * np.cos(rad)
    return u, v

def _to_wind(u, v):
    wspd = np.hypot(u, v)
    # atan2 returns [-pi, pi]; convert to met direction (from which)
    wdir = (np.rad2deg(np.arctan2(-u, -v)) + 360.0) % 360.0
    return wspd, wdir

def _interp_1d(x, y, x_new):
    """Monotone-safe 1D interpolation; ignores NaNs; no extrapolation."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    # ensure strictly increasing x for interpolators
    order = np.argsort(x)
    x, y = x[order], y[order]
    # drop duplicate x
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    xmin, xmax = x.min(), x.max()
    xq = np.array(x_new)
    oob = (xq < xmin) | (xq > xmax)
    if _HAVE_PCHIP:
        f = _Interp(x, y, extrapolate=False)
        out = f(xq)
    else:
        out = np.interp(np.clip(xq, xmin, xmax), x, y)
        out[oob] = np.nan
    return out

def _prep_profile(prof, smooth_window=0, search_agl=(0., 6000.)):
    # Compute theta_e and RH on the sounding; return z_agl, theta_e[K], RH[0-1]
    p  = np.asarray(prof.pres)*units.hPa
    T  = np.asarray(prof.tmpc)*units.degC
    Td = np.asarray(prof.dwpc)*units.degC
    z  = np.asarray(prof.hght, float)
    good = np.isfinite(p.m) & np.isfinite(T.m) & np.isfinite(Td.m) & np.isfinite(z)
    p, T, Td, z = p[good], T[good], Td[good], z[good]
    z0 = np.nanmin(z)
    z_agl = z - z0

    sl = (z_agl >= search_agl[0]) & (z_agl <= search_agl[1])
    p, T, Td, z, z_agl = p[sl], T[sl], Td[sl], z[sl], z_agl[sl]

    theta_e = mpcalc.equivalent_potential_temperature(p, T, Td).m  # K
    RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('dimensionless').m

    if smooth_window and smooth_window >= 3 and smooth_window % 2 == 1 and smooth_window < z_agl.size:
        theta_e = savgol_filter(theta_e, smooth_window, 2)
        RH      = np.clip(savgol_filter(RH,      smooth_window, 2), 0.0, 1.0)

    return z, z_agl, theta_e, RH

def _find_origin_heights(z, z_agl, theta_e_profile, m_theta_e, delta_the=1.0, prefer='low', evap_flag=None):
    # For each mobile sample's theta_e, pick a crossing height (low vs high)
    m_theta_e = np.asarray(m_theta_e, float)
    n = m_theta_e.size
    out_agl = np.full(n, np.nan)
    if evap_flag is None:
        evap_flag = np.zeros(n, dtype=bool)

    for i, mthe in enumerate(m_theta_e):
        crossings = []

        # straddles
        for j in range(len(theta_e_profile)-1):
            th1, th2 = theta_e_profile[j], theta_e_profile[j+1]
            if (th1 - mthe) * (th2 - mthe) < 0:
                zc = z[j] + (mthe - th1) * (z[j+1] - z[j]) / (th2 - th1)
                crossings.append(zc)

        # near matches
        idx = np.where(np.abs(theta_e_profile - mthe) <= delta_the)[0]
        crossings.extend(list(z[idx]))

        if crossings:
            crossings = np.sort(np.unique(crossings))
            z_pick = crossings[-1] if (evap_flag[i] or prefer == 'high') else crossings[0]
            out_agl[i] = z_pick - z.min()
        else:
            jmin = int(np.nanargmin(np.abs(theta_e_profile - mthe)))
            out_agl[i] = z[jmin] - z.min()

    return out_agl  # meters AGL

def _weight_function(z_agl, mode='triangular', center=2000.0, halfwidth=1000.0, sigma=600.0):
    """
    Return W(z) on z_agl (meters). Options:
      - 'none'       : W=1
      - 'triangular' : peak=1 at center, linear to 0 at center±halfwidth
      - 'gaussian'   : exp(-0.5*((z-center)/sigma)^2)
    """
    z_agl = np.asarray(z_agl, float)
    if mode == 'none':
        return np.ones_like(z_agl)
    if mode == 'triangular':
        w = np.maximum(0.0, 1.0 - np.abs(z_agl - center)/halfwidth)
        return w
    if mode == 'gaussian':
        return np.exp(-0.5*((z_agl - center)/sigma)**2)
    raise ValueError("Unknown weight mode")

def _evaporation_exposure(z_agl, RH_profile, z_origin_agl,
                          weight_mode='triangular', center=2000.0, halfwidth=1000.0, sigma=600.0):
    """
    Compute E = ∫ (1 - RH(z)) * W(z) dz from surface (0) to z_origin_agl.
    Returns E (same units as z: meters). You can normalize if desired.
    """
    # Interpolants on a fine monotonic grid
    z_sorted = np.asarray(z_agl)
    RH_sorted = np.clip(np.asarray(RH_profile), 0.0, 1.0)
    f_RH = interp1d(z_sorted, RH_sorted, bounds_error=False, fill_value=(RH_sorted[0], RH_sorted[-1]))

    # same for weight
    W_full = _weight_function(z_sorted, weight_mode, center, halfwidth, sigma)
    f_W = interp1d(z_sorted, W_full, bounds_error=False, fill_value=(W_full[0], W_full[-1]))
    
    z0 = 0.0
    # evaluate on a 10-m grid up to each origin
    def integrate_to(z_top):
        if not np.isfinite(z_top) or z_top <= z0:
            return np.nan
        zz = np.linspace(z0, z_top, int(max(2, np.ceil((z_top - z0)/10.0))))
        integrand = (1.0 - f_RH(zz)) * f_W(zz)
        # trapezoidal integral in meters (unitless * m)
        return np.trapz(integrand, zz)

    return np.array([integrate_to(z_o) for z_o in np.asarray(z_origin_agl)])

def compute_the_origin(prof, mdat, 
                       search_agl=(0., 6000.),   # meters AGL to search
                       delta_the=1.0,           # K tolerance for “near match”
                       prefer='low',            # 'low' (inflow) or 'high' (outflow)
                       smooth_window=0,         # e.g., 7 levels; 0 = no smoothing
                       evap_flag=None,
                       weight_mode='triangular',  # 'none'|'triangular'|'gaussian'
                       weight_center_m=2000.0,
                       weight_halfwidth_m=1000.0,
                       weight_sigma_m=600.0):
    """
    Return origin height(s) where profile θe matches mobile θe.
    Picks the lowest crossing by default (inflow). If evap_flag[i] is True,
    the highest crossing is chosen for that sample (downdraft/outflow).
    """
    # profile θe (MetPy expects p [Pa or hPa], T [K or °C], Td [K or °C])
    theta_e = mpcalc.equivalent_potential_temperature(np.asarray(prof.pres)*units.hPa,
                  np.asarray(prof.tmpc)*units.degC,
                  np.asarray(prof.dwpc)*units.degC).m
    z = np.asarray(prof.hght).astype(float)

    # clean + AGL conversion
    good = np.isfinite(theta_e) & np.isfinite(z)
    z, theta_e = z[good], theta_e[good]
    z0 = np.nanmin(z)
    z_agl = z - z0

    # restrict to search layer
    sl = (z_agl >= search_agl[0]) & (z_agl <= search_agl[1])
    z, z_agl, theta_e = z[sl], z_agl[sl], theta_e[sl]
    if z.size < 2:
        return np.full_like(np.asarray(mdat['the'], float), np.nan)

    # optional light smoothing to avoid noisy multi-crossings
    if smooth_window and smooth_window >= 3 and smooth_window % 2 == 1 and smooth_window < z.size:
        theta_e = savgol_filter(theta_e, smooth_window, 2)

    m_the = np.asarray(mdat['the'], float)
    n = m_the.size
    out = np.full(n, np.nan)

    if evap_flag is None:
        evap_flag = np.zeros(n, dtype=bool)

    for i, mthe in enumerate(m_the):
        if not np.isnan(mthe):
            crossings = []
    
            # exact straddle crossings
            for j in range(len(theta_e) - 1):
                th1, th2 = theta_e[j], theta_e[j+1]
                if (th1 - mthe) * (th2 - mthe) < 0:
                    z_interp = z[j] + (mthe - th1) * (z[j+1] - z[j]) / (th2 - th1)
                    crossings.append(z_interp)
    
            # near matches within ±delta_the
            idx = np.where(np.abs(theta_e - mthe) <= delta_the)[0]
            crossings.extend(list(z[idx]))
    
            if crossings:
                crossings = np.sort(np.unique(crossings))
                z_pick = crossings[-1] if evap_flag[i] or (prefer == 'high') else crossings[0]
                out[i] = z_pick - z0
            else:
                # nearest θe fallback
                jmin = int(np.nanargmin(np.abs(theta_e - mthe)))
                out[i] = z[jmin] - z0
        else:
            out[i] = np.nan
    
    RH_prof = mpcalc.relative_humidity_from_dewpoint(np.array(prof.tmpc)*units.degC, np.array(prof.dwpc)*units.degC).to('dimensionless').m
    RH_prof = RH_prof[good][sl]
    E = _evaporation_exposure(z_agl, RH_prof, out, weight_mode=weight_mode, center=weight_center_m, halfwidth=weight_halfwidth_m,sigma=weight_sigma_m)
    
    return out,E

def compute_sounding_layer_params(prof,mes,shear_layermin,shear_layermax,u_storm,v_storm):
    shear_mag,shear_dir = [],[]
    tdd,lr,cape,cin,pwat = [],[],[],[],[]
    for ish in range(len(shear_layermin)):
        Z = (np.array(prof.hght)-np.array(prof.hght)[0])[:-1] #m
    
        #Get the height indices of the layer
        iZ0,iZ1 = np.argmin(np.abs(Z-shear_layermin[ish])),np.argmin(np.abs(Z-shear_layermax[ish]))
        if Z[iZ0:iZ1].size > 0:
            su,sv,stemp,sdew,spres = np.array(prof.u)*0.514444,np.array(prof.v)*0.514444,np.array(prof.tmpc),np.array(prof.dwpc),np.array(prof.pres) #m/s
    
            #Compute SR bulk shear vector (m/s)
            du,dv = su[iZ1]-su[iZ0],sv[iZ1]-sv[iZ0]
            shear_mag.append([np.sqrt((du**2)+(dv**2))]*mes['thv'].size)
            shear_dir.append([mpcalc.wind_direction(du*(units.meters/units.seconds),dv*(units.meters/units.seconds)).magnitude]*mes['thv'].size)
    
            #Compute mean dewpoint depressions in these layers
            tdd.append([np.nanmean(stemp[iZ0:iZ1]-sdew[iZ0:iZ1])]*mes['thv'].size)
            lr.append([-(stemp[iZ1]-stemp[iZ0])/(Z[iZ1]-Z[iZ0])]*mes['thv'].size)
    
            #Compute CAPE and CIN in the layers
            aprof = profile.create_profile(profile='default',pres=np.array(prof.pres)[iZ0:iZ1],
                                           hght=Z[iZ0:iZ1],tmpc=np.array(prof.tmpc)[iZ0:iZ1],
                                           dwpc=np.array(prof.dwpc)[iZ0:iZ1],wspd=np.array(prof.wspd)[iZ0:iZ1],
                                           wdir=np.array(prof.wdir)[iZ0:iZ1],missing=-9999.,strictQC=False)
            pcl = params.parcelx(aprof, flag=1)
            cape.append([pcl.bplus]*len(mes['time']))
            cin.append([pcl.bminus]*len(mes['time']))
    
            spwat = compute_precipitable_water(np.array(aprof.wvmr)/1000, np.array(aprof.pres)*100)
            pwat.append([spwat]*len(mes['time']))
    
            #Compute evaporation potential energy (My creation)
            T,Td,z = np.array(prof.tmpc)[iZ0:iZ1] + 273.15, np.array(prof.dwpc)[iZ0:iZ1]+273.15,Z[iZ0:iZ1]
            mask = (~np.isnan(T)) & (~np.isnan(Td)) & (~np.isnan(z))
            T, Td, z = T[mask], Td[mask], z[mask]
        else:
            org_height,z_ci_low,z_ci_high,z_class = np.zeros((mes['the'].size))*np.nan,np.zeros((mes['the'].size))*np.nan,np.zeros((mes['the'].size))*np.nan,np.zeros((mes['the'].size))*np.nan
            shear_mag,shear_dir = np.zeros((len(shear_layermin),mes['thv'].size))*np.nan,np.zeros((len(shear_layermin),mes['thv'].size))*np.nan
            tdd,lr = np.zeros((len(shear_layermin),mes['thv'].size))*np.nan,np.zeros((len(shear_layermin),mes['thv'].size))*np.nan
            cape,cin = np.zeros((len(shear_layermin),mes['thv'].size))*np.nan,np.zeros((len(shear_layermin),mes['thv'].size))*np.nan
            pwat = np.zeros((len(shear_layermin),mes['thv'].size))*np.nan
    
    shear_mag,shear_dir = np.array(shear_mag),np.array(shear_dir)
    tdd,lr = np.array(tdd),np.array(lr)
    cape,cin,pwat = np.array(cape),np.array(cin),np.array(pwat)
    return_dict = {'shear_mag':shear_mag,'shear_dir':shear_dir,'tdd':tdd,'lr':lr,'cape':cape,'cin':cin,'pwat':pwat}
    return return_dict

def get_mesonet_data(dir_mesonet, elev_file_path, atime, radar_file_dir, mode, caseid, candidate_radars=None, time_to_space_conversion=5, elev_angle=0.5, time_tolerance_sec=360,search_radius_m=100.0,neighborhood_shape="circle",min_gates=1,fallback_to_nearest=True,filter_accelerating_winds=False,vehicle_dv_thresh_mps=0.4,vehicle_dv_window_sec=6.0,vehicle_dv_use_vector=True):
    """
    Retrieves mobile mesonet data within a specified time period.

    Parameters:
    - dir_mesonet (str): Path to mesonet data.
    - atime (datetime): Analysis time.
    - time_to_space_conversion (int, optional): Time window in minutes (default: 10).

    Returns:
    - data (dict): Dictionary with mesonet data.
    """

    # Derive average elevation and fetch mesonet file paths
    mmfs = get_mm_files_cached(dir_mesonet, atime)
    
    # Initialize lists for all data
    all_data = {key: [] for key in ['time', 'lat', 'lon', 'pres', 'temp', 'dew', 'rh','pres_corr','ref','ql',
                                    'wspd', 'wdir', 'u', 'v','qv', 'th','thv_ql','tv_ql',
                                    'thv','the','thw','tv','elev', 'campaign', 'probe']}
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    z_ref_m = _analysis_reference_elevation(dir_mesonet, atime, time_to_space_conversion=time_to_space_conversion)
    if not np.isfinite(z_ref_m):
        print(f"[WARN] no finite reference elevation for analysis time {atime}; pres_corr will be NaN", flush=True)

    # Process each mesonet file
    for fi in mmfs:
        with xr.open_dataset(fi, decode_times=False) as m:
            winds_prefiltered = str(m.attrs.get("accelerating_winds_filtered", "false")).lower() == "true"
            tm = m.time.values
            mask = (tm >= date2num(atime - time_window, 'seconds since 1970-01-01')) & (tm <= date2num(atime + time_window,'seconds since 1970-01-01'))
            # Extract relevant data and filter
            try:
                temp,pres,rh,wspd,wdir,elev = m.temp_unbiased.sel(time=mask).values,m.pres_unbiased.sel(time=mask).values,m.rh_unbiased.sel(time=mask).values,m.wspd_unbiased.sel(time=mask).values,m.wdir_unbiased.sel(time=mask).values,m.elev.sel(time=mask).values
            except AttributeError: #Dead file
                continue
            lat,lon,tm = m.lat.sel(time=mask).values,m.lon.sel(time=mask).values,num2date(m.time.sel(time=mask).values,'seconds since 1970-01-01',only_use_cftime_datetimes=False)
            lon = np.array([-val if val>0 else val for val in lon])
            if temp.size == 0:
                continue

            #Unit check
            rh = normalize_rh_to_percent(rh)
            temp = normalize_temp_to_celsius(temp)
            
            # Preserve all rows in the analysis window. Apply QC per variable
            # instead of deleting the whole observation when one field is bad.
            # Processed archives are expected to already be in m/s; do not
            # unit-convert here, but do remove unsupported spikes if an older
            # archive slipped through.
            wspd, wdir = _qc_processed_wind_speed_series(wspd, wdir, tm, label=str(fi))

            pos_valid = (~np.isnan(lat)) & (~np.isnan(lon)) & (lon < 0)
            temp_valid0 = pos_valid & (~np.isnan(temp)) & (temp > -60) & (temp < 120)
            rh_valid0 = pos_valid & (~np.isnan(rh)) & (rh >= 0) & (rh <= 110)
            pres_valid0 = pos_valid & (~np.isnan(pres)) & (pres > 100) & (pres < 1050)
            elev_valid0 = pos_valid & np.isfinite(elev)
            wind_valid0 = pos_valid & (~np.isnan(wspd)) & (wspd >= 0) & (wspd <= MAX_WSPD_MPS) & (~np.isnan(wdir)) & (wdir >= 0) & (wdir <= 360)
            
            u, v = mpcalc.wind_components(wspd * (units.meters / units.seconds), wdir * units.degrees)
            u, v = u.m, v.m

            m_temp = within_k_sigma(temp, temp_valid0, k=2.5)
            m_rh   = within_k_sigma(rh,   rh_valid0, k=2.5)
            m_pres = within_k_sigma(pres, pres_valid0, k=2.5)
            wind_base = wind_valid0
            m_u    = within_k_sigma(u,    wind_base, k=2.5)
            m_v    = within_k_sigma(v,    wind_base, k=2.5)
            m_elev = within_k_sigma(elev, elev_valid0, k=2.5)

            temp = np.asarray(temp, dtype=float).copy()
            rh = np.asarray(rh, dtype=float).copy()
            pres = np.asarray(pres, dtype=float).copy()
            elev = np.asarray(elev, dtype=float).copy()
            wspd = np.asarray(wspd, dtype=float).copy()
            wdir = np.asarray(wdir, dtype=float).copy()
            u = np.asarray(u, dtype=float).copy()
            v = np.asarray(v, dtype=float).copy()

            temp[~(temp_valid0 & m_temp)] = np.nan
            rh[~(rh_valid0 & m_rh)] = np.nan
            pres_bad = ~(pres_valid0 & m_pres)
            pres[pres_bad] = np.nan
            elev[~(elev_valid0 & m_elev)] = np.nan
            pres_corr = np.full(pres.shape, np.nan, dtype=float)
            pres_corr_valid = (np.isfinite(z_ref_m) & np.isfinite(pres) & np.isfinite(temp) & np.isfinite(rh) & np.isfinite(elev))
            if np.any(pres_corr_valid):
                pres_corr[pres_corr_valid] = _reduce_pressure_to_z(
                    pres[pres_corr_valid].astype(np.float32),
                    temp[pres_corr_valid].astype(np.float32),
                    (rh[pres_corr_valid] / 100.0).astype(np.float32),
                    elev[pres_corr_valid].astype(np.float32),
                    float(z_ref_m),
                )
            wind_keep = wind_valid0 & m_u & m_v
            wspd[~wind_keep] = np.nan
            wdir[~wind_keep] = np.nan
            u[~wind_keep] = np.nan
            v[~wind_keep] = np.nan

            rho = np.full(pres.shape, np.nan, dtype=float)
            rho_valid = np.isfinite(pres) & np.isfinite(temp)
            rho[rho_valid] = (pres[rho_valid] * 100) / (Rd * (temp[rho_valid] + 273.15))
            qv = np.full(pres.shape, np.nan, dtype=float)
            tv = np.full(pres.shape, np.nan, dtype=float)
            thermo_valid = np.isfinite(pres) & np.isfinite(temp) & np.isfinite(rh)
            if np.any(thermo_valid):
                qv_valid = mpcalc.mixing_ratio_from_relative_humidity(
                    np.array(pres[thermo_valid]) * units.hPa,
                    np.array(temp[thermo_valid]) * units.degC,
                    np.array(rh[thermo_valid]) * units.percent
                ).m
                qv[thermo_valid] = qv_valid
                tv_valid = mpcalc.virtual_temperature(
                    np.array(temp[thermo_valid]) * units.degC,
                    np.array(qv_valid) * units.dimensionless
                ).m
                tv[thermo_valid] = tv_valid
            
            if tm[0].year == 2009:
                moving_keep = keep_moving_points(lat, lon, tm, min_move_m=30, min_speed_mps=0.5)
                stationary = ~moving_keep
                if np.any(stationary):
                    wspd[stationary] = np.nan
                    wdir[stationary] = np.nan
                    u[stationary] = np.nan
                    v[stationary] = np.nan

            if filter_accelerating_winds and (not winds_prefiltered) and tm.size > 0:
                bad_wind = accelerating_vehicle_wind_mask(
                    lat,
                    lon,
                    tm,
                    dv_thresh_mps=vehicle_dv_thresh_mps,
                    dv_window_sec=vehicle_dv_window_sec,
                    use_vector=vehicle_dv_use_vector,
                )
                if np.any(bad_wind):
                    wspd = np.asarray(wspd, dtype=float).copy()
                    wdir = np.asarray(wdir, dtype=float).copy()
                    u = np.asarray(u, dtype=float).copy()
                    v = np.asarray(v, dtype=float).copy()
                    wspd[bad_wind] = np.nan
                    wdir[bad_wind] = np.nan
                    u[bad_wind] = np.nan
                    v[bad_wind] = np.nan

            if candidate_radars is not None and tm.size > 0:
                _cache_candidate_radar_volumes(candidate_radars, tm, radar_file_dir, time_tolerance_sec)

            ref = np.full(tm.shape, np.nan, dtype=float)
            ref_idx = np.where(np.isfinite(lat) & np.isfinite(lon))[0]
            if ref_idx.size:
                ref_subset = get_reflectivity_along_path(
                    caseid, tm[ref_idx], lat[ref_idx], lon[ref_idx], radar_file_dir, mode,
                    candidate_radars=candidate_radars, obs_elev=elev[ref_idx],
                    elev_angle=elev_angle, time_tolerance_sec=time_tolerance_sec,
                    search_radius_m=search_radius_m, neighborhood_shape=neighborhood_shape,
                    min_gates=min_gates, fallback_to_nearest=fallback_to_nearest,
                )
                ref[ref_idx] = np.asarray(ref_subset, dtype=float)
            ref_for_ql = np.where(np.isfinite(ref), np.maximum(ref, 0.0), np.nan)

            #Derive the liquid water mixing ratio [Rutledge & Hobbs (1984) vs. Hane & Ray (1985)]
            ql = np.full(ref.shape, np.nan, dtype=float)
            #ql = (10**((ref_for_ql-42.2)/16.8))/1000
            alpha = 0.1*ref_for_ql
            ql = ((1/rho) * (((10**alpha)/17300)**(4/7)))/1000
            ql[np.isnan(ql)] = 0 #NaN reflectivity is zero
            
            dew = np.full(temp.shape, np.nan, dtype=float)
            qvs = np.full(temp.shape, np.nan, dtype=float)
            th = np.full(temp.shape, np.nan, dtype=float)
            if np.any(thermo_valid):
                dew[thermo_valid] = mpcalc.dewpoint_from_relative_humidity(
                    np.array(temp[thermo_valid]) * units.degC,
                    np.array(rh[thermo_valid]) * units.percent
                ).m
                qvs[thermo_valid] = mpcalc.saturation_mixing_ratio(
                    np.array(pres[thermo_valid]) * units.hPa,
                    np.array(temp[thermo_valid]) * units.degC
                ).m
                th[thermo_valid] = mpcalc.potential_temperature(
                    np.array(pres[thermo_valid]) * units.hPa,
                    np.array(temp[thermo_valid]) * units.degC
                ).m
            thv = th*(1+(0.61*qv))
            thv_ql = th*(1+(0.61*qv)-ql)
            tv_ql = ((temp+273.15)*(1+(0.61*qv)-ql)) - 273.15

            the = np.full(temp.shape, np.nan, dtype=float)
            if np.any(thermo_valid):
                the[thermo_valid] = mpcalc.equivalent_potential_temperature(
                    np.array(pres[thermo_valid]) * units.hPa,
                    np.array(temp[thermo_valid]) * units.degC,
                    np.array(dew[thermo_valid]) * units.degC
                ).m
            thw = np.full(temp.shape, np.nan, dtype=float)
            for i in np.where(thermo_valid)[0]:
                try:
                    thw[i] = compute_thw(temp[i], pres[i], rh[i])
                except Exception:
                    thw[i] = np.nan

            year = tm[0].year if len(tm) > 0 else None
            campaign = {
                2009: 'V2_09', 2010: 'V2_10', 2017: 'RiVorS', 2019: 'TORUS_19',
                2022: 'TORUS_22', 2023: 'TORUS_LiTE', 2024: 'LIFT'
            }.get(year, 'Unknown')

            all_data['time'].append(tm)
            all_data['lat'].append(lat)
            all_data['lon'].append(lon)
            all_data['ql'].append(ql)
            all_data['pres'].append(pres)
            all_data['pres_corr'].append(pres_corr)
            all_data['ref'].append(ref)
            all_data['temp'].append(temp)
            all_data['dew'].append(dew)
            all_data['rh'].append(rh)
            all_data['wspd'].append(wspd)
            all_data['wdir'].append(wdir)
            all_data['u'].append(u)
            all_data['v'].append(v)
            all_data['qv'].append(qv)
            all_data['th'].append(th)
            all_data['thv'].append(thv)
            all_data['tv_ql'].append(tv_ql)
            all_data['thv_ql'].append(thv_ql)
            all_data['the'].append(the)
            all_data['thw'].append(thw)
            all_data['tv'].append(tv)
            all_data['elev'].append(elev)
            all_data['campaign'].append([campaign] * len(temp))
            all_data['probe'].append([fi.split('/')[-1]] * len(temp))
            
    # Convert lists to arrays
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key]) if all_data[key] else np.array([])
    
    return all_data

def process_case(dir_mesonet, sticknet_dir, elev_file_path, sound, shear_layermin, shear_layermax, atime, u_storm, v_storm, radar_file_dir, mode, caseid, meso_time, meso_lat, meso_lon, candidate_radars=None, time_to_space_conversion=5, filter_accelerating_winds=False, vehicle_dv_thresh_mps=0.4, vehicle_dv_window_sec=6.0, vehicle_dv_use_vector=True):

    mes = get_mesonet_data(
        dir_mesonet,
        elev_file_path,
        atime,
        radar_file_dir,
        mode,
        caseid,
        candidate_radars=candidate_radars,
        time_to_space_conversion=time_to_space_conversion,
        filter_accelerating_winds=filter_accelerating_winds,
        vehicle_dv_thresh_mps=vehicle_dv_thresh_mps,
        vehicle_dv_window_sec=vehicle_dv_window_sec,
        vehicle_dv_use_vector=vehicle_dv_use_vector,
    )
    
    ######################################################
    # Get sounding-derived observations
    ######################################################

    # Interpolate the sounding if one exists. VORTEX-era cases often do not
    # have matching sounding profiles in soundings_RAP_use_this.pkl. Missing
    # soundings are not fatal; all sounding-derived quantities are filled with
    # NaN arrays so the mesonet observations/perturbations can still be used.
    sounding_key = atime.strftime('%Y%m%d%H%M%S')
    nobs = len(mes['time'])
    nlayers = len(shear_layermin)

    try:
        if sound is None or sounding_key not in sound:
            raise KeyError(sounding_key)

        prof = interpolate_profile_on_pressure(sound[sounding_key], dp_hPa=sounding_dpres, return_profile=True)
        qc = np.asarray(prof.pres, dtype=float)
        org_height, evap_idx = compute_the_origin(prof, mes)

        # Compute shear in various layers (storm-relative)
        shear_layers = compute_sounding_layer_params(prof, mes, shear_layermin, shear_layermax, u_storm, v_storm)

    except Exception as exc:
        print(
            f"[WARN] no usable sounding for {sounding_key}; "
            f"filling sounding-derived fields with NaN ({type(exc).__name__}: {exc})",
            flush=True,
        )

        prof = np.nan
        org_height = np.full(nobs, np.nan, dtype=float)
        evap_idx = np.full(nobs, np.nan, dtype=float)
        shear_layers = {
            'shear_mag': np.full((nlayers, nobs), np.nan, dtype=float),
            'shear_dir': np.full((nlayers, nobs), np.nan, dtype=float),
            'tdd': np.full((nlayers, nobs), np.nan, dtype=float),
            'lr': np.full((nlayers, nobs), np.nan, dtype=float),
            'cape': np.full((nlayers, nobs), np.nan, dtype=float),
            'cin': np.full((nlayers, nobs), np.nan, dtype=float),
            'pwat': np.full((nlayers, nobs), np.nan, dtype=float),
        }
    
    #Append the data
    mes['prof_sr'] = prof
    mes['org_height'] = org_height
    mes['evap_exposure'] = evap_idx
    mes['shear_mag_layers'] = shear_layers['shear_mag']
    mes['shear_dir_layers'] = shear_layers['shear_dir']
    mes['Tdd_layers'] = shear_layers['tdd']
    mes['lapse_rate_layers'] = shear_layers['lr']
    mes['cape_layers'] = shear_layers['cape']
    mes['cin_layers'] = shear_layers['cin']
    mes['pwat_layers']= shear_layers['pwat']
    
    del shear_layers
    
    #Compute distance from the mesocyclone
    try:
        dist = compute_distances(meso_time, meso_lat, meso_lon, mes['time'], mes['lat'], mes['lon'])
        mes['dx'] = dist[:,0]
        mes['dy'] = dist[:,1]
    except IndexError: #Meaning no mesonet data found in this time frame
        mes['dx'] = np.zeros((mes['time'].size))*np.nan
        mes['dy'] = np.zeros((mes['time'].size))*np.nan
        
    #Rotate the grid for the MM observations to a normalized storm motion
    mdat_rx,mdat_ry = rotate_grid(mes['dx'], mes['dy'], u_storm,v_storm, pivot_x=0,pivot_y=0,target_angle=normalized_storm_motion)
    mes['rdx'] = mdat_rx/1000 #km
    mes['rdy'] = mdat_ry/1000 #km
    
    #Rotate the storm-relative winds to match the normalized storm motion
    sru,srv = mes['u']-u_storm, mes['v']-v_storm
    sru,srv = rotate_grid(sru, srv, u_storm,v_storm, pivot_x=0,pivot_y=0,target_angle=normalized_storm_motion)
    mes['sru'] = sru
    mes['srv'] = srv
    
    #Rotate grid-relative winds
    ru,rv = rotate_grid(mes['u'], mes['v'], u_storm,v_storm, pivot_x=0,pivot_y=0,target_angle=normalized_storm_motion)
    mes['gru'] = ru
    mes['grv'] = rv
    
    return mes
    
#Define constants
Rd = float(np.array(const.dry_air_gas_constant))
g = float(np.array(const.earth_gravity))
normalized_storm_motion = 90  # deg
sounding_dpres = 0.2  # hPa
show_pbar = (MPI.COMM_WORLD.Get_rank() == 0)


def _resolve(root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def _to_output_name(raw_path: Path) -> str:
    name = raw_path.name
    if name.endswith(".dat.qcd"):
        return name[:-8] + ".nc"
    if name.endswith(".qcd"):
        return name[:-4] + ".nc"
    if name.endswith(".dat"):
        return name[:-4] + ".nc"
    if name.endswith(".nc"):
        return name
    return raw_path.stem + ".nc"


def _iter_raw_files(raw_dir: Path, pattern: str = "*", limit: Optional[int] = None, allowed_dates: Optional[set[str]] = None) -> list[Path]:
    """
    Discover raw mobile mesonet files using mesonet_reader.py.

    The old version only allowed .dat.qcd/.qcd/.dat/.nc, which silently skipped
    original VORTEX/SUBVORTEX files such as:
      - P1_17April95
      - C1_2June95_friona_p
      - p1_25May97.raw
      - p4_02June99.raw
    and newer raw LIFT files such as:
      - Probe1_4_4_2025_180028_obs.txt
      - HailCam_4_4_2025_124129_obs.txt
      - dltruck_20250404.txt

    This function now asks mesonet_reader to parse the date/format. Files with
    unknown/ignored formats are skipped. This keeps raw discovery aligned with
    the actual reader used by _build_rebuilt_dataset().
    """
    files = []
    raw_dir = Path(raw_dir)

    # Use recursive search only if the caller passed ** in pattern. This preserves
    # the old shallow behavior by default.
    candidates = sorted(raw_dir.glob(pattern))

    for p in candidates:
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.startswith("._") or "__MACOSX" in str(p):
            continue

        # Let mesonet_reader decide whether this is a supported mobile mesonet file.
        try:
            fdate = get_file_date_mm(str(p), strict=False)
        except TypeError:
            fdate = get_file_date_mm(str(p))
        except Exception:
            continue

        if fdate is None:
            continue

        try:
            fmt = detect_mm_format(str(p), fdate)
        except Exception:
            fmt = "unknown"

        if fmt in {"ignore", "unknown"}:
            continue

        # Exclude original 1994 comma-style raw files. These do not contain
        # defensible decoded lat/lon in the available raw format, so they cannot
        # support spatial analysis or USGS elevation sampling.
        if getattr(fdate, "year", None) == 1994:
            continue

        # Exclude pressure-corrected VORTEX 1995 variants. This workflow uses
        # raw station pressure and applies its own pressure correction to common
        # elevation later. Pressure-corrected input would double-correct.
        lower_name = p.name.lower()
        if fmt == "vortex_mm_13col" and (
            lower_name.endswith("_p")
            or "_dimmitt_p" in lower_name
            or "_friona_p" in lower_name
        ):
            continue

        if allowed_dates is not None:
            try:
                fdate_key = fdate.strftime("%Y%m%d")
            except Exception:
                continue
            if fdate_key not in allowed_dates:
                continue

        files.append(p)

    if limit is not None:
        files = files[:limit]
    return files


def _write_atomic_netcdf(ds: xr.Dataset, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ds.to_netcdf(tmp)
    tmp.replace(path)


def _load_reference_dataset(source_path: Optional[Path]) -> Optional[xr.Dataset]:
    if source_path is None or not source_path.exists():
        return None
    return xr.open_dataset(source_path)


def _elev_tile_id(lat: float, lon: float) -> Optional[str]:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None
    lat_prefix = "n" if lat >= 0 else "s"
    lon_prefix = "e" if lon >= 0 else "w"
    lat_deg = int(np.ceil(abs(lat)))
    lon_deg = int(np.ceil(abs(lon)))
    return f"{lat_prefix}{lat_deg:02d}{lon_prefix}{lon_deg:03d}"


@lru_cache(maxsize=None)
def _find_local_elev_tile(elev_dir: str, tile_id: str) -> Optional[str]:
    matches = sorted(glob(os.path.join(elev_dir, f"*{tile_id}*.tif")))
    return matches[0] if matches else None


def _resolve_elev_tile_path(elev_dir: Path, tile_id: str, allow_download: bool = True) -> Optional[str]:
    local_path = _find_local_elev_tile(str(elev_dir), tile_id)
    if local_path is None and allow_download:
        downloaded = _download_elev_tile(elev_dir, tile_id)
        if downloaded is not None:
            local_path = str(downloaded)
            _find_local_elev_tile.cache_clear()
    return local_path


def _is_valid_elev_tile(tile_path) -> bool:
    """Return True only when rasterio can open the tile and read basic metadata.

    TNM/S3 downloads occasionally leave an HTML/XML error response, a partial
    file, or a zero-byte file with a .tif suffix. Under MPI this is especially
    easy to trigger when multiple ranks request the same missing tile. Treat
    those as missing, remove them, and let the normal fallback path continue
    instead of killing the whole job at the next barrier.
    """
    try:
        path = Path(tile_path)
        if (not path.exists()) or path.stat().st_size <= 0:
            return False
        with rasterio.open(path) as src:
            _ = src.width, src.height, src.count, src.crs
        return True
    except Exception:
        return False


def _remove_bad_elev_tile(tile_path, reason: str = "invalid") -> None:
    try:
        path = Path(tile_path)
        if path.exists():
            path.unlink()
            print(f"[WARN] removed {reason} elevation tile {path}", flush=True)
    except Exception as exc:
        print(f"[WARN] could not remove bad elevation tile {tile_path}: {exc}", flush=True)


@lru_cache(maxsize=128)
def _open_elev_tile(tile_path: str):
    try:
        src = rasterio.open(tile_path)
    except Exception:
        _open_elev_tile.cache_clear()
        _remove_bad_elev_tile(tile_path, reason="unreadable")
        raise
    is_geographic = bool(src.crs is None or src.crs.is_geographic)
    return src, is_geographic, src.nodata


def _download_elev_tile(elev_dir: Path, tile_id: str, timeout: int = 60) -> Optional[Path]:
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
        if tile_id not in _ELEV_WARNED_TILES:
            print(f"[WARN] elevation tile {tile_id} not found at https://prd-tnm.s3.amazonaws.com/{prefix} ; falling back if needed")
            _ELEV_WARNED_TILES.add(tile_id)
        return None

    key = sorted(keys)[-1]
    out_path = elev_dir / os.path.basename(key)
    if out_path.exists():
        if _is_valid_elev_tile(out_path):
            return out_path
        _remove_bad_elev_tile(out_path, reason="corrupt/stale")

    try:
        with requests.get(f"https://prd-tnm.s3.amazonaws.com/{key}", stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            if not _is_valid_elev_tile(tmp):
                _remove_bad_elev_tile(tmp, reason="invalid downloaded")
                if tile_id not in _ELEV_WARNED_TILES:
                    print(f"[WARN] downloaded elevation tile {tile_id} was not a readable GeoTIFF; falling back if needed", flush=True)
                    _ELEV_WARNED_TILES.add(tile_id)
                return None
            tmp.replace(out_path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if tile_id not in _ELEV_WARNED_TILES:
            print(f"[WARN] failed to download elevation tile {tile_id} from https://prd-tnm.s3.amazonaws.com/{key} ; falling back if needed")
            _ELEV_WARNED_TILES.add(tile_id)
        return None
    print(f"[INFO] downloaded elevation tile https://prd-tnm.s3.amazonaws.com/{key}")
    return out_path


def _sample_elevation_from_dir(elev_dir: Path, lats: np.ndarray, lons: np.ndarray, fallback: Optional[np.ndarray] = None, allow_download: bool = True, fallback_label: str = "fallback values") -> np.ndarray:
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    elev = np.full(lats.shape, np.nan, dtype=np.float32)
    tile_ids = np.array([_elev_tile_id(lat, lon) for lat, lon in zip(lats, lons)], dtype=object)

    for tile_id in sorted(set(val for val in tile_ids if val is not None)):
        idx = np.where(tile_ids == tile_id)[0]
        local_path = _resolve_elev_tile_path(elev_dir, tile_id, allow_download=allow_download)
        if local_path is None:
            if tile_id not in _ELEV_WARNED_TILES:
                print(f"[WARN] no local/downloaded elevation tile for {tile_id}; using {fallback_label}")
                _ELEV_WARNED_TILES.add(tile_id)
            continue

        try:
            src, is_geographic, nodata = _open_elev_tile(local_path)
            xs = lons[idx]
            ys = lats[idx]
            if not is_geographic:
                xs, ys = rio_transform("EPSG:4326", src.crs, xs.tolist(), ys.tolist())
            samples = np.fromiter((val[0] for val in src.sample(list(zip(xs, ys)))), dtype=np.float32, count=idx.size)
            if nodata is not None:
                samples[np.isclose(samples, nodata)] = np.nan
            elev[idx] = samples
        except Exception as exc:
            if tile_id not in _ELEV_WARNED_TILES:
                print(f"[WARN] elevation tile {tile_id} could not be sampled from {local_path}: {exc}; using {fallback_label}", flush=True)
                _ELEV_WARNED_TILES.add(tile_id)
            continue

    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float32)
        use_fallback = ~np.isfinite(elev) & np.isfinite(fallback)
        elev[use_fallback] = fallback[use_fallback]
    return elev



def _clean_elevation_values(elev, *, label="elev", require_all_valid=True):
    """Validate USGS/TNM elevation values in meters.

    This project now requires every processed mobile-mesonet elevation value to
    come from USGS/TNM elevation data. No raw-platform altitude, source-archive
    elevation, or median-fill fallback is allowed.

    Parameters
    ----------
    elev : array-like
        Elevation values sampled from USGS/TNM GeoTIFF tiles.
    label : str
        Label used in error/warning messages.
    require_all_valid : bool
        If True, raise ValueError when any elevation value is missing or outside
        the physically plausible processing range.

    Returns
    -------
    ndarray
        float32 elevation array in meters.
    """
    elev = np.asarray(elev, dtype=np.float32).copy()

    bad = (~np.isfinite(elev)) | (elev < -500.0) | (elev > 5000.0)

    if np.any(bad):
        nbad = int(np.sum(bad))
        finite = elev[np.isfinite(elev)]
        if finite.size:
            msg = (
                f"{label}: {nbad} invalid/missing USGS elevation values; "
                f"finite range=({np.nanmin(finite):.3f}, {np.nanmax(finite):.3f})"
            )
        else:
            msg = f"{label}: all USGS elevation values are missing/invalid"

        if require_all_valid:
            raise ValueError(msg)

        print(f"[WARN] {msg}; setting invalid values to NaN", flush=True)
        elev[bad] = np.nan

    return elev.astype(np.float32)

def _reference_elevation(source_ds: Optional[xr.Dataset], size: int, alt: np.ndarray) -> np.ndarray:
    if source_ds is not None and "elev" in source_ds:
        elev = np.asarray(source_ds["elev"].values, dtype=np.float32)
        if elev.size == size:
            return elev
        if elev.size > 0:
            return np.full(size, np.nanmedian(elev), dtype=np.float32)
    return np.asarray(alt, dtype=np.float32)


def _apply_variable_metadata(ds: xr.Dataset) -> xr.Dataset:
    meta = {
        "temp": {"units": "deg C"},
        "temp_mean": {"units": "deg C"},
        "temp_std": {"units": "deg C"},
        "temp_var": {"units": "deg C^2"},
        "temp_unbiased": {"units": "deg C"},
        "pres": {"units": "hPa"},
        "pres_mean": {"units": "hPa"},
        "pres_std": {"units": "hPa"},
        "pres_var": {"units": "hPa^2"},
        "pres_unbiased": {"units": "hPa"},
        "rh": {"units": "fraction"},
        "rh_mean": {"units": "fraction"},
        "rh_std": {"units": "fraction"},
        "rh_var": {"units": "fraction^2"},
        "rh_unbiased": {"units": "fraction"},
        "wspd": {"units": "m/s"},
        "wspd_mean": {"units": "m/s"},
        "wspd_std": {"units": "m/s"},
        "wspd_var": {"units": "m^2/s^2"},
        "wspd_unbiased": {"units": "m/s"},
        "wdir": {"units": "deg"},
        "wdir_mean": {"units": "deg"},
        "wdir_std": {"units": "deg"},
        "wdir_var": {"units": "deg^2"},
        "wdir_unbiased": {"units": "deg"},
        "u": {"units": "m/s"},
        "u_mean": {"units": "m/s"},
        "u_std": {"units": "m/s"},
        "u_var": {"units": "m^2/s^2"},
        "u_unbiased": {"units": "m/s"},
        "v": {"units": "m/s"},
        "v_mean": {"units": "m/s"},
        "v_std": {"units": "m/s"},
        "v_var": {"units": "m^2/s^2"},
        "v_unbiased": {"units": "m/s"},
        "elev": {"units": "m", "long_name": "elevation above mean sea level"},
        "bias_magnitudes": {"units": "mixed", "long_name": "[temp_C, reduced_pres_hPa, rh_fraction, u_mps, v_mps] bias components"},
    }
    for var, attrs in meta.items():
        if var in ds:
            ds[var].attrs.update(attrs)
    return ds


def _validate_dataset_units(ds: xr.Dataset, label: str) -> None:
    checks = {
        "temp": (-80.0, 60.0),
        "temp_mean": (-80.0, 60.0),
        "temp_unbiased": (-80.0, 60.0),
        "pres": (400.0, 1100.0),
        "pres_mean": (400.0, 1100.0),
        "pres_unbiased": (400.0, 1100.0),
        "rh": (0.0, 1.05),
        "rh_mean": (0.0, 1.05),
        "rh_unbiased": (0.0, 1.05),
        "wspd": (0.0, MAX_WSPD_MPS),
        "wspd_mean": (0.0, MAX_WSPD_MPS),
        "wspd_unbiased": (0.0, MAX_WSPD_MPS),
        "wdir": (0.0, 360.0),
        "wdir_mean": (0.0, 360.0),
        "wdir_unbiased": (0.0, 360.0),
        "u": (-MAX_WSPD_MPS, MAX_WSPD_MPS),
        "u_mean": (-MAX_WSPD_MPS, MAX_WSPD_MPS),
        "u_unbiased": (-MAX_WSPD_MPS, MAX_WSPD_MPS),
        "v": (-MAX_WSPD_MPS, MAX_WSPD_MPS),
        "v_mean": (-MAX_WSPD_MPS, MAX_WSPD_MPS),
        "v_unbiased": (-MAX_WSPD_MPS, MAX_WSPD_MPS),
        "elev": (-500.0, 5000.0),
    }
    for var, (vmin, vmax) in checks.items():
        if var not in ds:
            continue
        arr = np.asarray(ds[var].values)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        if np.nanmin(finite) < vmin or np.nanmax(finite) > vmax:
            raise ValueError(f"{label}: variable {var} failed unit/range validation ({np.nanmin(finite):.3f}, {np.nanmax(finite):.3f})")



def _processed_file_date(path_value):
    """Return date for a processed NetCDF filename, or None if unparseable.

    Processed files preserve raw basenames but usually replace the source suffix
    with .nc. Older get_file_date_mm versions recognize raw *_obs.txt names but
    not processed *_obs.nc names, so handle those here.
    """
    p = Path(path_value)
    try:
        dt = get_file_date_mm(str(p))
        if dt is not None:
            return dt
    except Exception:
        pass

    base = p.name

    m = re.search(
        r"^(?:Probe1|Probe2|HailCam)_?(\d{1,2})_(\d{1,2})_(\d{4})_\d{1,6}_obs\.nc$",
        base,
        flags=re.IGNORECASE,
    )
    if m:
        month, day, year = m.groups()
        return datetime(int(year), int(month), int(day))

    m = re.search(r"^dltruck_(\d{8})\.nc$", base, flags=re.IGNORECASE)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")

    return None


def _analysis_times_for_tsc(time_to_space_minutes: int) -> np.ndarray:
    """Return analysis offsets in minutes for the trim/reduced-pressure stages."""
    if int(time_to_space_minutes) <= 0:
        raise ValueError("time_to_space_minutes must be positive")
    return np.arange(-20, 20 + 1, int(time_to_space_minutes), dtype=int)

@lru_cache(maxsize=None)
def _analysis_case_windows_cached(transects_csv_str: str, time_to_space_minutes: int) -> pd.DataFrame:
    tr = _read_transects_table(transects_csv_str)
    tr["Tstart_dt"] = [datetime.strptime(val, "%Y%m%d-%H%M") for val in tr["Tstart"].values]
    analysis_offsets = _analysis_times_for_tsc(time_to_space_minutes)
    half_window = timedelta(seconds=(time_to_space_minutes * 60) / 2)
    rows = []
    for _, row in tr.iterrows():
        for offset in analysis_offsets:
            analysis_dt = row["Tstart_dt"] + timedelta(minutes=int(offset))
            rows.append(
                {
                    "Tstart_dt": row["Tstart_dt"],
                    "analysis_dt": analysis_dt,
                    "win_start": analysis_dt - half_window,
                    "win_end": analysis_dt + half_window,
                }
            )
    return pd.DataFrame(rows)


def _analysis_windows_for_file(file_date: datetime, transects_csv: Path, time_to_space_minutes: int) -> list[tuple[datetime, datetime]]:
    tr = _analysis_case_windows_cached(str(transects_csv), time_to_space_minutes)
    windows = []
    target_date = datetime(file_date.year, file_date.month, file_date.day)
    for _, row in tr.iterrows():
        adt = row["analysis_dt"]
        if abs((datetime(adt.year, adt.month, adt.day) - target_date).total_seconds()) <= 86400:
            windows.append((row["win_start"], row["win_end"]))
    return windows


def _transect_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first matching column from a transect table.

    Matching is case-insensitive and ignores spaces, underscores, and hyphens so
    both ``Observation Start`` and ``observation_start`` work.
    """
    def norm(value):
        return re.sub(r"[\s_\-]+", "", str(value).strip().lower())

    by_norm = {norm(col): col for col in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in by_norm:
            return by_norm[key]
    raise KeyError(
        "Could not find any of these transect columns: "
        + ", ".join(candidates)
        + f". Available columns: {list(df.columns)}"
    )


def _parse_transect_datetime(value) -> datetime:
    """Parse a transect-table datetime into a naive UTC datetime.

    The legacy tables usually use YYYYMMDD-HHMM strings, but Excel-loaded
    tables can also contain pandas/Excel datetimes.
    """
    if pd.isna(value):
        raise ValueError("missing transect datetime")

    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime().replace(tzinfo=None)

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().replace(tzinfo=None)

    text = str(value).strip()
    for fmt in (
        "%Y%m%d-%H%M",
        "%Y%m%d-%H%M%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    ts = pd.to_datetime(text, errors="raise")
    return ts.to_pydatetime().replace(tzinfo=None)


@lru_cache(maxsize=None)
def _observation_case_windows_cached(transects_csv_str: str) -> pd.DataFrame:
    """Return observation start/end windows from the transect timing table."""
    tr = _read_transects_table(transects_csv_str)
    start_col = _transect_col(
        tr,
        [
            "Observation Start",
            "Obs Start",
            "ObsStart",
            "ObservationStart",
            "obs_start",
            "start_time",
        ],
    )
    end_col = _transect_col(
        tr,
        [
            "Observation End",
            "Obs End",
            "ObsEnd",
            "ObservationEnd",
            "obs_end",
            "end_time",
        ],
    )

    rows = []
    for i, row in tr.iterrows():
        try:
            start = _parse_transect_datetime(row[start_col])
            end = _parse_transect_datetime(row[end_col])
        except Exception as exc:
            print(f"[WARN] skipping transect row {i}: could not parse observation window: {exc}", flush=True)
            continue

        if end <= start:
            print(
                f"[WARN] skipping transect row {i}: Observation End <= Observation Start "
                f"({start} .. {end})",
                flush=True,
            )
            continue

        rows.append({"obs_start": start, "obs_end": end})

    return pd.DataFrame(rows)


def _observation_windows_for_file(file_date: datetime, transects_csv: Path) -> list[tuple[datetime, datetime]]:
    """Return observation start/end windows relevant to one processed file date.

    A raw platform file can cross UTC midnight, so include windows that overlap a
    loose date envelope around the file date rather than requiring the start time
    to fall exactly on that date.
    """
    tr = _observation_case_windows_cached(str(transects_csv))
    if tr.empty:
        return []

    target_start = datetime(file_date.year, file_date.month, file_date.day) - timedelta(days=1)
    target_end = datetime(file_date.year, file_date.month, file_date.day) + timedelta(days=2)
    windows = []
    for _, row in tr.iterrows():
        start = row["obs_start"]
        end = row["obs_end"]
        if end >= target_start and start <= target_end:
            windows.append((start, end))
    return windows


def _trim_dataset_to_observation_windows(ds: xr.Dataset, file_date: datetime, transects_csv: Path) -> Optional[xr.Dataset]:
    """Trim a processed platform NetCDF to Observation Start/End windows."""
    windows = _observation_windows_for_file(file_date, transects_csv)
    if not windows:
        return None

    tm = num2date(ds["time"].values, "seconds since 1970-01-01")
    tm = np.array([datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in tm], dtype=object)
    keep = np.zeros(tm.size, dtype=bool)
    for start, end in windows:
        keep |= (tm >= start) & (tm <= end)

    if not np.any(keep):
        return None
    return ds.isel(time=np.where(keep)[0])


def _trim_dataset_to_analysis_windows(ds: xr.Dataset, file_date: datetime, transects_csv: Path, time_to_space_minutes: int) -> Optional[xr.Dataset]:
    windows = _analysis_windows_for_file(file_date, transects_csv, time_to_space_minutes)
    if not windows:
        return None
    tm = num2date(ds["time"].values, "seconds since 1970-01-01")
    tm = np.array([datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in tm], dtype=object)
    keep = np.zeros(tm.size, dtype=bool)
    for start, end in windows:
        keep |= (tm >= start) & (tm <= end)
    if not np.any(keep):
        return None
    return ds.isel(time=np.where(keep)[0])


def _reduce_pressure_to_z(pres_hpa: np.ndarray, temp_c: np.ndarray, rh_frac: np.ndarray, elev_m: np.ndarray, z_ref_m: float) -> np.ndarray:
    rh_use = np.asarray(rh_frac, dtype=np.float32).copy()
    if np.nanmax(rh_use) > 1.5:
        rh_use = rh_use / 100.0
    rh_use[(rh_use < 0) | (rh_use > 1.5)] = np.nan
    qv = mpcalc.mixing_ratio_from_relative_humidity(
        np.asarray(pres_hpa, dtype=np.float32) * units.hPa,
        np.asarray(temp_c, dtype=np.float32) * units.degC,
        rh_use * units.dimensionless,
    ).magnitude
    tv = ((np.asarray(temp_c, dtype=np.float32) + 273.15) * (1 + (0.61 * qv))) - 273.15
    return (np.asarray(pres_hpa, dtype=np.float32) * np.exp(((np.asarray(elev_m, dtype=np.float32) - z_ref_m) / (Rd * (tv + 273.15))) * g)).astype(np.float32)


def _analysis_reference_elevation(dir_mesonet, atime, time_to_space_conversion=5):
    """Mean platform elevation for one analysis-time window."""
    mmfs = get_mm_files_cached(dir_mesonet, atime)
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    elev_samples = []
    for fi in mmfs:
        try:
            with xr.open_dataset(fi, decode_times=False) as m:
                if "time" not in m or "elev" not in m:
                    continue
                tm_num = m.time.values
                mask = (tm_num >= date2num(atime - time_window, "seconds since 1970-01-01")) & (tm_num <= date2num(atime + time_window, "seconds since 1970-01-01"))
                if not np.any(mask):
                    continue
                elev = np.asarray(m.elev.sel(time=mask).values, dtype=float)
                if "lat" in m and "lon" in m:
                    lat = np.asarray(m.lat.sel(time=mask).values, dtype=float)
                    lon = np.asarray(m.lon.sel(time=mask).values, dtype=float)
                    lon = np.where(lon > 0, -lon, lon)
                    keep = np.isfinite(elev) & np.isfinite(lat) & np.isfinite(lon) & (lon < 0)
                else:
                    keep = np.isfinite(elev)
                if np.any(keep):
                    elev_samples.append(elev[keep])
        except Exception:
            continue
    if not elev_samples:
        return np.nan
    vals = np.concatenate(elev_samples)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def _reference_series(source_ds: Optional[xr.Dataset], name: str, size: int) -> Optional[np.ndarray]:
    if source_ds is None or name not in source_ds:
        return None
    arr = np.asarray(source_ds[name].values)
    if arr.ndim != 1 or arr.size != size:
        return None
    return arr.astype(np.float32)


def _wind_components(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, v = mpcalc.wind_components(speed * (units.meter / units.second), direction_deg * units.degrees)
    return np.asarray(u.magnitude, dtype=np.float32), np.asarray(v.magnitude, dtype=np.float32)


def _wind_dir_speed(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(
        mpcalc.wind_direction(u * (units.meter / units.second), v * (units.meter / units.second)),
        dtype=np.float32,
    )
    speed = np.sqrt(u * u + v * v).astype(np.float32)
    return direction, speed


def _seconds_from_time_array(time_values) -> np.ndarray:
    """Return seconds since epoch for pandas/numpy/python datetime arrays."""
    try:
        ts = pd.to_datetime(time_values, errors="coerce")
        return ts.view("int64").astype(float) / 1.0e9
    except Exception:
        out = []
        for val in np.asarray(time_values, dtype=object):
            try:
                out.append(pd.Timestamp(val).timestamp())
            except Exception:
                out.append(np.nan)
        return np.asarray(out, dtype=float)


def _maybe_convert_raw_wspd_to_mps(wspd: np.ndarray, label: str = "") -> tuple[np.ndarray, bool]:
    """Convert likely mph wind-speed series to m s-1.

    Processed archives are always m s-1. Some raw HailCam-style text files can
    report mph. A max near 150 with otherwise plausible values should not be
    interpreted as 150 m s-1. This conversion is intentionally conservative and
    is applied only during raw-file rebuilds.
    """
    arr = np.asarray(wspd, dtype=np.float32).copy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, False

    fmax = float(np.nanmax(finite))
    fmed = float(np.nanmedian(finite))
    lname = str(label).lower()

    likely_mph_file = any(tok in lname for tok in ("hailcam", "probe", "lift"))
    # If values exceed the 150 mph equivalent in m/s but remain within a
    # plausible mph range, treat the raw series as mph. Also catch HailCam-like
    # files whose median strongly suggests mph rather than m/s.
    likely_mph_values = (MAX_WSPD_MPS < fmax <= MAX_WSPD_MPH + 5.0) or (likely_mph_file and 35.0 < fmed <= 80.0 and fmax <= MAX_WSPD_MPH + 5.0)

    if likely_mph_values:
        arr = arr * MPH_TO_MPS
        print(
            f"[INFO] converting wind speed from mph to m/s for {label}: "
            f"raw median={fmed:.2f}, raw max={fmax:.2f}",
            flush=True,
        )
        return arr.astype(np.float32), True

    return arr, False


def _unsupported_wspd_spike_mask(
    wspd: np.ndarray,
    time_values,
    *,
    max_wspd_mps: float = MAX_WSPD_MPS,
    window_sec: float = WIND_SPIKE_WINDOW_SEC,
    abs_jump_mps: float = WIND_SPIKE_ABS_JUMP_MPS,
    mad_sigma: float = WIND_SPIKE_MAD_SIGMA,
    min_neighbors: int = WIND_SPIKE_MIN_NEIGHBORS,
) -> np.ndarray:
    """Mask isolated wind-speed spikes unsupported by nearby observations.

    A point is considered unsupported only when it is a strong outlier relative
    to finite neighboring samples inside +/- window_sec. This preserves coherent
    high-wind periods while removing single-sample spikes. Values above the
    physical 150 mph cap are always masked.
    """
    v = np.asarray(wspd, dtype=float)
    bad = (~np.isfinite(v)) | (v < 0.0) | (v > float(max_wspd_mps))
    if v.size < 3:
        return bad

    t = _seconds_from_time_array(time_values)
    if not np.isfinite(t).any():
        # Fallback to sample index spacing when timestamps are unavailable.
        t = np.arange(v.size, dtype=float)
        window_sec = max(2.0, float(min_neighbors))

    finite = np.isfinite(v) & np.isfinite(t) & (v >= 0.0) & (v <= float(max_wspd_mps))
    idxs = np.where(finite)[0]
    if idxs.size < max(min_neighbors + 1, 5):
        return bad

    for i in idxs:
        neigh = finite & (np.abs(t - t[i]) <= float(window_sec))
        neigh[i] = False
        vals = v[neigh]
        vals = vals[np.isfinite(vals)]
        if vals.size < int(min_neighbors):
            continue
        med = float(np.nanmedian(vals))
        mad = float(1.4826 * np.nanmedian(np.abs(vals - med)))
        if not np.isfinite(mad):
            mad = 0.0
        tol = max(float(abs_jump_mps), float(mad_sigma) * mad)
        p05 = float(np.nanpercentile(vals, 5))
        p95 = float(np.nanpercentile(vals, 95))
        high_spike = (v[i] > p95 + tol) and (v[i] - med > tol)
        low_spike = (v[i] < p05 - tol) and (med - v[i] > tol)
        if high_spike or low_spike:
            bad[i] = True
    return bad


def _qc_raw_wind_speed_series(wspd, wdir, time_values, label: str = ""):
    """Normalize raw wind speed to m/s and mask unsupported wind spikes."""
    wspd_mps, converted = _maybe_convert_raw_wspd_to_mps(wspd, label=label)
    wdir_arr = np.asarray(wdir, dtype=np.float32).copy()
    bad = _unsupported_wspd_spike_mask(wspd_mps, time_values)
    bad |= ~np.isfinite(wdir_arr) | (wdir_arr < 0.0) | (wdir_arr > 360.0)
    nbad = int(np.count_nonzero(bad & np.isfinite(wspd_mps)))
    if nbad:
        print(f"[WARN] masking {nbad} unsupported/invalid wind observation(s) from {label}", flush=True)
    wspd_mps = np.asarray(wspd_mps, dtype=np.float32).copy()
    wspd_mps[bad] = np.nan
    wdir_arr[bad] = np.nan
    return wspd_mps, wdir_arr, {"converted_mph_to_mps": converted, "n_wind_masked": nbad}


def _qc_processed_wind_speed_series(wspd, wdir, time_values, label: str = ""):
    """Mask unsupported wind spikes in already-processed m/s archives."""
    wspd_arr = np.asarray(wspd, dtype=np.float32).copy()
    wdir_arr = np.asarray(wdir, dtype=np.float32).copy()
    bad = _unsupported_wspd_spike_mask(wspd_arr, time_values)
    bad |= ~np.isfinite(wdir_arr) | (wdir_arr < 0.0) | (wdir_arr > 360.0)
    if np.any(bad & np.isfinite(wspd_arr)):
        print(f"[WARN] masking {int(np.count_nonzero(bad & np.isfinite(wspd_arr)))} processed wind spike(s) from {label}", flush=True)
    wspd_arr[bad] = np.nan
    wdir_arr[bad] = np.nan
    return wspd_arr, wdir_arr


def _build_rebuilt_dataset(
    raw_path: Path,
    source_path: Optional[Path],
    window_seconds: int,
    elev_dir: Optional[Path] = None,
    filter_accelerating_winds: bool = False,
    vehicle_dv_thresh_mps: float = 0.4,
    vehicle_dv_window_sec: float = 6.0,
    vehicle_dv_use_vector: bool = True,
) -> xr.Dataset:
    # Guard against direct processing of pressure-corrected VORTEX variants.
    # This workflow requires raw pressure because pressure correction is applied
    # consistently after USGS elevation sampling.
    _raw_name_lower = raw_path.name.lower()
    if (
        _raw_name_lower.endswith("_p")
        or "_dimmitt_p" in _raw_name_lower
        or "_friona_p" in _raw_name_lower
    ):
        raise ValueError(f"pressure-corrected input excluded from raw-pressure workflow: {raw_path}")

    raw = readMM(str(raw_path))
    if raw is None:
        raise ValueError(f"readMM returned no data for {raw_path}")

    if isinstance(raw, dict) and raw.get("position_available", True) is False:
        raise ValueError(f"position_available False for {raw_path}")

    # Convert times defensively. Some raw logger files contain malformed GPS
    # date/time fields that can produce impossible datetimes, e.g.,
    # year 0004. Pandas cannot represent those as ns-resolution datetimes,
    # so filter them before pd.to_datetime().
    raw_time = np.asarray(raw["time"], dtype=object)

    valid_time = np.array([
        (
            t is not None
            and hasattr(t, "year")
            and 1900 <= int(t.year) <= 2100
        )
        for t in raw_time
    ], dtype=bool)

    if raw_time.size == 0 or not np.any(valid_time):
        raise ValueError(f"readMM returned no valid in-range datetimes for {raw_path}")

    if not np.all(valid_time):
        nbad = int((~valid_time).sum())
        print(
            f"[WARN] dropping {nbad} out-of-range/malformed timestamp rows from {raw_path.name}",
            flush=True,
        )

    # Apply the same valid-time mask to every 1-D raw array with matching length.
    raw = {
        k: (np.asarray(v)[valid_time] if np.asarray(v).shape[:1] == raw_time.shape[:1] else v)
        for k, v in raw.items()
    }

    time = pd.to_datetime(raw["time"], errors="coerce")
    good_pd_time = ~pd.isna(time)

    if time.size == 0 or not np.any(good_pd_time):
        raise ValueError(f"readMM returned no pandas-convertible datetimes for {raw_path}")

    if not np.all(good_pd_time):
        nbad = int((~good_pd_time).sum())
        print(
            f"[WARN] dropping {nbad} pandas-unconvertible timestamp rows from {raw_path.name}",
            flush=True,
        )
        raw = {
            k: (np.asarray(v)[good_pd_time] if np.asarray(v).shape[:1] == time.shape[:1] else v)
            for k, v in raw.items()
        }
        time = pd.to_datetime(raw["time"], errors="raise")

    if time.size == 0:
        raise ValueError(f"readMM returned zero usable rows for {raw_path}")

    # Drop rows whose parsed timestamp is wildly inconsistent with the file date.
    # This catches valid-looking but impossible GPS dates, e.g., a 1999 timestamp
    # embedded in a 2025 LIFT raw file, without trying to "correct" the value.
    try:
        file_date_for_time_qc = get_file_date_mm(str(raw_path), strict=False)
    except Exception:
        file_date_for_time_qc = None

    if file_date_for_time_qc is not None:
        lower_time_bound = pd.Timestamp(file_date_for_time_qc) - pd.Timedelta(days=1)
        upper_time_bound = pd.Timestamp(file_date_for_time_qc) + pd.Timedelta(days=2)
        near_file_date = (time >= lower_time_bound) & (time <= upper_time_bound)

        if not np.any(near_file_date):
            raise ValueError(
                f"all parsed timestamps are outside ±1/+2 day file-date window for {raw_path}"
            )

        if not np.all(near_file_date):
            nbad = int((~near_file_date).sum())
            print(
                f"[WARN] dropping {nbad} off-date timestamp rows from {raw_path.name} "
                f"outside {lower_time_bound}..{upper_time_bound}",
                flush=True,
            )
            raw = {
                k: (np.asarray(v)[near_file_date] if np.asarray(v).shape[:1] == time.shape[:1] else v)
                for k, v in raw.items()
            }
            time = pd.to_datetime(raw["time"], errors="raise")

    # Enforce chronological order before moving-window statistics and NetCDF write.
    # Some raw files contain valid blocks concatenated out of order. Sorting keeps
    # each observation with its matching variables and avoids false time-decrease
    # failures without inventing/correcting timestamps.
    if time.size > 1:
        order = np.argsort(time.to_numpy())
        if not np.array_equal(order, np.arange(time.size)):
            print(
                f"[WARN] sorting {raw_path.name} chronologically before processing",
                flush=True,
            )
            raw = {
                k: (np.asarray(v)[order] if np.asarray(v).shape[:1] == time.shape[:1] else v)
                for k, v in raw.items()
            }
            time = time[order]

    time_num = date2num(time.to_pydatetime(), "seconds since 1970-01-01").astype(np.float64)
    temp = np.asarray(raw["temp"], dtype=np.float32)
    pres = np.asarray(raw["pres"], dtype=np.float32)
    rh = np.asarray(raw["rh"], dtype=np.float32)

    # Defensive length checks. Some raw files may parse to no usable rows after
    # malformed timestamps or unsupported raw formats. Fail early with a useful
    # filename so _prepare_mesonet_archive can skip only that file.
    required_arrays = {
        "temp": temp,
        "pres": pres,
        "rh": rh,
        "wspd": np.asarray(raw["wspd"], dtype=np.float32),
        "wdir": np.asarray(raw["wdir"], dtype=np.float32),
        "lat": np.asarray(raw["lat"], dtype=np.float32),
        "lon": np.asarray(raw["lon"], dtype=np.float32),
        "alt": np.asarray(raw["alt"], dtype=np.float32),
    }
    for _name, _arr in required_arrays.items():
        if _arr.size == 0:
            raise ValueError(f"readMM returned zero-size {_name} array for {raw_path}")
        if _arr.size != time.size:
            raise ValueError(
                f"readMM returned mismatched {_name} length for {raw_path}: "
                f"{_arr.size} vs time {time.size}"
            )

    finite_rh = rh[np.isfinite(rh)]
    if finite_rh.size and np.nanmax(finite_rh) > 2:
        rh = rh / 100.0
    wspd = np.asarray(raw["wspd"], dtype=np.float32)
    wdir = np.asarray(raw["wdir"], dtype=np.float32)
    lat = np.asarray(raw["lat"], dtype=np.float32)
    lon = np.asarray(raw["lon"], dtype=np.float32)
    lon = np.where(lon > 0, -lon, lon).astype(np.float32)
    alt = np.asarray(raw["alt"], dtype=np.float32)

    # Drop rows without valid lat/lon before trying to sample USGS elevation.
    # These rows cannot be located physically, so keeping them would only create
    # invalid elevation samples or corrupted storm-relative calculations.
    pos_good = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & (lon >= -180.0)
        & (lon <= 180.0)
        & (lon != 0.0)
    )

    if not np.any(pos_good):
        raise ValueError(f"no finite position data for {raw_path}")

    if not np.all(pos_good):
        nbad = int((~pos_good).sum())
        print(f"[WARN] dropping {nbad} rows without valid lat/lon from {raw_path.name}", flush=True)

        time = time[pos_good]
        time_num = time_num[pos_good]
        temp = temp[pos_good]
        pres = pres[pos_good]
        rh = rh[pos_good]
        wspd = np.asarray(raw["wspd"], dtype=np.float32)[pos_good]
        wdir = np.asarray(raw["wdir"], dtype=np.float32)[pos_good]
        lat = lat[pos_good]
        lon = lon[pos_good]
        alt = alt[pos_good]

        # Keep raw dictionary lengths aligned if anything later touches it.
        for _k, _v in list(raw.items()):
            _arr = np.asarray(_v)
            if _arr.shape[:1] == pos_good.shape[:1]:
                raw[_k] = _arr[pos_good]

    if alt.size and np.isfinite(alt).any() and np.nanmedian(alt) < 1:
        alt = alt * 1000.0

    source_ds = _load_reference_dataset(source_path)

    # Elevation must come from USGS/TNM tiles only. Raw platform altitude and
    # old source-archive elevation are not allowed as fallbacks.
    source_elev = None

    if elev_dir is not None:
        elev = _sample_elevation_from_dir(
            elev_dir,
            lat,
            lon,
            fallback=None,
            allow_download=True,
            fallback_label="USGS elevation unavailable",
        )
    else:
        raise ValueError(
            f"{raw_path.name}: --elev-dir is required because all elevations must come from USGS/TNM"
        )

    # Enforce USGS-only elevation for all retained rows. If a row cannot be
    # assigned valid USGS elevation, drop that row. Do NOT fall back to raw
    # platform altitude or source-archive elevation.
    elev = np.asarray(elev, dtype=np.float32)
    elev_good = np.isfinite(elev) & (elev >= -500.0) & (elev <= 5000.0)

    if not np.any(elev_good):
        raise ValueError(f"no valid USGS elevation values for {raw_path}")

    if not np.all(elev_good):
        nbad = int((~elev_good).sum())
        frac = nbad / float(elev_good.size)
        print(
            f"[WARN] dropping {nbad} rows without valid USGS elevation from {raw_path.name} "
            f"({100.0 * frac:.3f}% of rows)",
            flush=True,
        )

        time = time[elev_good]
        time_num = time_num[elev_good]
        temp = temp[elev_good]
        pres = pres[elev_good]
        rh = rh[elev_good]
        wspd = wspd[elev_good]
        wdir = wdir[elev_good]
        lat = lat[elev_good]
        lon = lon[elev_good]
        alt = alt[elev_good]
        elev = elev[elev_good]

    elev = _clean_elevation_values(
        elev,
        label=f"{raw_path.name} USGS elevation",
        require_all_valid=True,
    )

    # Normalize raw wind speed to m/s and remove isolated, unsupported spikes
    # before u/v and moving-window statistics are derived.
    wspd, wdir, wind_qc_info = _qc_raw_wind_speed_series(
        wspd,
        wdir,
        np.asarray(time.to_pydatetime(), dtype=object),
        label=raw_path.name,
    )
    u, v = _wind_components(wspd, wdir)

    if filter_accelerating_winds and time.size > 0:
        bad_wind = accelerating_vehicle_wind_mask(
            lat,
            lon,
            np.asarray(time.to_pydatetime(), dtype=object),
            dv_thresh_mps=vehicle_dv_thresh_mps,
            dv_window_sec=vehicle_dv_window_sec,
            use_vector=vehicle_dv_use_vector,
        )
        if np.any(bad_wind):
            wspd = np.asarray(wspd, dtype=np.float32).copy()
            wdir = np.asarray(wdir, dtype=np.float32).copy()
            u = np.asarray(u, dtype=np.float32).copy()
            v = np.asarray(v, dtype=np.float32).copy()
            wspd[bad_wind] = np.nan
            wdir[bad_wind] = np.nan
            u[bad_wind] = np.nan
            v[bad_wind] = np.nan

    data = np.full((temp.size, 7), np.nan, dtype=np.float32)
    data[:, 0] = temp
    data[:, 1] = pres
    data[:, 2] = rh
    data[:, 3] = wspd
    data[:, 4] = wdir
    data[:, 5] = u
    data[:, 6] = v
    mean_data, std_data = moving_stats(np.asarray(time.to_pydatetime()), data, timedelta(seconds=window_seconds))
    mean_data = np.asarray(mean_data, dtype=np.float32)
    std_data = np.asarray(std_data, dtype=np.float32)
    var_data = (std_data ** 2).astype(np.float32)

    var_names = [
        "temp", "temp_mean", "temp_std", "temp_var",
        "pres", "pres_mean", "pres_std", "pres_var",
        "rh", "rh_mean", "rh_std", "rh_var",
        "wspd", "wspd_mean", "wspd_std", "wspd_var",
        "wdir", "wdir_mean", "wdir_std", "wdir_var",
        "u", "u_mean", "u_std", "u_var",
        "v", "v_mean", "v_std", "v_var",
        "elev",
    ]
    ncdata = [
        data[:, 0], mean_data[:, 0], std_data[:, 0], var_data[:, 0],
        data[:, 1], mean_data[:, 1], std_data[:, 1], var_data[:, 1],
        data[:, 2], mean_data[:, 2], std_data[:, 2], var_data[:, 2],
        data[:, 3], mean_data[:, 3], std_data[:, 3], var_data[:, 3],
        data[:, 4], mean_data[:, 4], std_data[:, 4], var_data[:, 4],
        data[:, 5], mean_data[:, 5], std_data[:, 5], var_data[:, 5],
        data[:, 6], mean_data[:, 6], std_data[:, 6], var_data[:, 6],
        elev,
    ]

    ds = xr.Dataset(coords={"time": ("time", time_num)})
    ds["lat"] = ("time", lat)
    ds["lon"] = ("time", lon)
    for name, values in zip(var_names, ncdata):
        ds[name] = ("time", np.asarray(values, dtype=np.float32))

    # Do not align the rebuilt archive to a source/reference NetCDF time axis.
    # Older source archives may already be trimmed to analysis windows, and using
    # them as the authoritative time axis would silently cut the rebuilt output
    # back to those windows.  The processed archive should preserve every valid
    # raw timestamp in each platform file; source archives are retained only as
    # optional historical inputs for workflows that still pass --source-dir.

    ds = ds.assign_coords(bias_mag=np.arange(5, dtype=np.int32))
    ds["bias_magnitudes"] = ("bias_mag", np.zeros(5, dtype=np.float32))

    n_time = ds.sizes["time"]
    # Do not overwrite USGS elevation from source/reference archives.
    # Source archives are used only for time-axis alignment.

    ds["elev"] = (
        "time",
        _clean_elevation_values(
            ds["elev"].values,
            label=f"{raw_path.name} final USGS elevation",
            require_all_valid=True,
        ),
    )

    # The source archive is only used for time-axis alignment
    # fallback. Unbiased fields are recomputed in this workflow, so do not
    # copy legacy unbiased values forward.
    ds["temp_unbiased"] = ("time", ds["temp_mean"].values.astype(np.float32))
    ds["pres_unbiased"] = ("time", ds["pres_mean"].values.astype(np.float32))
    ds["rh_unbiased"] = ("time", ds["rh_mean"].values.astype(np.float32))
    ds["u_unbiased"] = ("time", ds["u_mean"].values.astype(np.float32))
    ds["v_unbiased"] = ("time", ds["v_mean"].values.astype(np.float32))
    ds["wdir_unbiased"] = ("time", ds["wdir_mean"].values.astype(np.float32))
    ds["wspd_unbiased"] = ("time", ds["wspd_mean"].values.astype(np.float32))

    ds = _apply_variable_metadata(ds)
    _validate_dataset_units(ds, raw_path.name)
    ds.attrs["description"] = "Mobile mesonet observations rebuilt from raw data"
    ds.attrs["history"] = f"Created {datetime.utcnow():%Y-%m-%d %H:%M}"
    ds.attrs["accelerating_winds_filtered"] = "true" if filter_accelerating_winds else "false"
    ds.attrs["raw_wspd_converted_mph_to_mps"] = "true" if bool(wind_qc_info.get("converted_mph_to_mps", False)) else "false"
    ds.attrs["raw_wind_spike_or_invalid_count"] = int(wind_qc_info.get("n_wind_masked", 0))
    ds.attrs["max_wspd_mph"] = float(MAX_WSPD_MPH)
    ds.attrs["max_wspd_mps"] = float(MAX_WSPD_MPS)
    ds.attrs["vehicle_dv_thresh_mps"] = float(vehicle_dv_thresh_mps)
    ds.attrs["vehicle_dv_window_sec"] = float(vehicle_dv_window_sec)
    ds.attrs["vehicle_dv_use_vector"] = "true" if vehicle_dv_use_vector else "false"
    if source_ds is not None:
        source_ds.close()
    return ds


def load_storm_track_window(case_id, tracking_dir: Path, mode: str, start_time: datetime, end_time: datetime, dt_tracking: float = 1 / 60):
    mtimes, srlat, srlon = _load_storm_track(case_id, str(tracking_dir), mode)
    isort = np.argsort(mtimes)
    mtimes = mtimes[isort]
    srlat = srlat[isort]
    srlon = srlon[isort]

    time_grid = np.arange(
        date2num(mtimes[0], "seconds since 1970-01-01"),
        date2num(mtimes[-1], "seconds since 1970-01-01"),
        dt_tracking * 60,
    )
    srlat = interpolate_var(srlat, date2num(mtimes, "seconds since 1970-01-01"), time_grid)
    srlon = interpolate_var(srlon, date2num(mtimes, "seconds since 1970-01-01"), time_grid)
    mtimes = num2date(time_grid, "seconds since 1970-01-01")
    mtimes = np.array([datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in mtimes], dtype=object)
    iwin = np.where((mtimes >= start_time) & (mtimes <= end_time))[0]
    return mtimes[iwin], srlat[iwin], srlon[iwin]


def get_mesonet_data_for_bias_window(mmfs: np.ndarray, start_time: datetime, end_time: datetime):
    all_data = {key: [] for key in ["time", "lat", "lon", "pres", "temp", "rh", "u", "v", "elev"]}
    mm_file = []
    for fi in mmfs:
        with xr.open_dataset(fi) as m:
            tm = num2date(m.time.values, "seconds since 1970-01-01")
            tm = np.array([datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in tm])
            itm = np.where((tm >= start_time) & (tm <= end_time))[0]
            if itm.size == 0:
                continue

            temp = m.temp[itm].values
            pres = m.pres_mean[itm].values
            rh = m.rh_mean[itm].values
            wspd = m.wspd_mean[itm].values
            wdir = m.wdir_mean[itm].values
            elev = m.elev[itm].values
            lat = m.lat[itm].values
            lon = m.lon[itm].values
            tm = tm[itm]

            n = min(len(tm), len(temp), len(pres), len(rh), len(wspd), len(wdir), len(elev), len(lat), len(lon))
            tm = tm[:n]
            temp = temp[:n]
            pres = pres[:n]
            rh = rh[:n]
            wspd = wspd[:n]
            wdir = wdir[:n]
            elev = elev[:n]
            lat = lat[:n]
            lon = lon[:n]

            u, v = _wind_components(np.asarray(wspd, dtype=np.float32), np.asarray(wdir, dtype=np.float32))
            lon = np.where(lon > 0, -lon, lon)

            valid_mask = (
                (pres < 2000)
                & (rh >= 0)
                & (rh <= 1.2)
                & (wspd >= 0)
                & (wspd <= MAX_WSPD_MPS)
                & (wdir >= 0)
                & (wdir <= 360)
                & ~np.isnan(np.array([temp, pres, rh, wspd, wdir, u, v, elev, lat, lon])).any(axis=0)
            )
            if not np.any(valid_mask):
                continue

            all_data["time"].append(np.array(tm)[valid_mask])
            all_data["lat"].append(lat[valid_mask])
            all_data["lon"].append(lon[valid_mask])
            all_data["pres"].append(pres[valid_mask])
            all_data["temp"].append(temp[valid_mask])
            all_data["rh"].append(rh[valid_mask])
            all_data["u"].append(u[valid_mask])
            all_data["v"].append(v[valid_mask])
            all_data["elev"].append(elev[valid_mask])
            mm_file.append(str(fi))
    return all_data, mm_file


def _storm_distances_km(track_times, track_lats, track_lons, sample_time, sample_lats, sample_lons, max_time_diff=60):
    if len(track_times) == 0:
        return np.full(sample_lats.shape, np.nan, dtype=np.float32)
    time_diffs = np.abs(np.array([(tt - sample_time).total_seconds() for tt in track_times], dtype=np.float64))
    closest_idx = int(np.argmin(time_diffs))
    if time_diffs[closest_idx] > max_time_diff:
        return np.full(sample_lats.shape, np.nan, dtype=np.float32)
    dists = np.zeros(sample_lats.shape, dtype=np.float32)
    for i, (plat, plon) in enumerate(zip(sample_lats, sample_lons)):
        dx, dy = get_dx_dy(track_lons[closest_idx], track_lats[closest_idx], plon, plat)
        dists[i] = np.sqrt(dx * dx + dy * dy) / 1000.0
    return dists


def idw_interpolation(xy_known: np.ndarray, values: np.ndarray, xy_target: np.ndarray, power: float = 2):
    valid_mask = ~np.isnan(values)
    xy_known_valid = xy_known[valid_mask]
    values_valid = values[valid_mask]
    if len(values_valid) == 0:
        raise ValueError("No valid data points to perform interpolation.")

    distances = haversine(xy_known_valid[:, 1], xy_known_valid[:, 0], xy_target[:, 1], xy_target[:, 0]) * 1000
    distances = distances.reshape(1, -1)
    weights = 1.0 / np.power(distances, power)
    weights[distances == 0] = 1e-10
    weights /= weights.sum(axis=1, keepdims=True)
    return np.dot(weights, values_valid)


def _apply_bias_update(nc_path: Path, bias_vec: np.ndarray) -> None:
    with xr.open_dataset(nc_path) as ds:
        out = ds.load()

    current_bias = out["bias_magnitudes"].values.astype(np.float32)
    finite_bias = np.isfinite(bias_vec)
    current_bias[finite_bias] = bias_vec[finite_bias].astype(np.float32)
    out = out.assign_coords(bias_mag=np.arange(5, dtype=np.int32))
    out["bias_magnitudes"] = ("bias_mag", current_bias)

    temp_u = out["temp_unbiased"].values.astype(np.float32)
    pres_u = out["pres_unbiased"].values.astype(np.float32)
    rh_u = out["rh_unbiased"].values.astype(np.float32)
    u_u = out["u_unbiased"].values.astype(np.float32)
    v_u = out["v_unbiased"].values.astype(np.float32)

    if np.isfinite(current_bias[0]):
        temp_u = (out["temp_mean"].values - current_bias[0]).astype(np.float32)
    if np.isfinite(current_bias[1]):
        pres_u = (out["pres_mean"].values - current_bias[1]).astype(np.float32)
    if np.isfinite(current_bias[2]):
        rh_u = (out["rh_mean"].values - current_bias[2]).astype(np.float32)
    if np.isfinite(current_bias[3]):
        u_u = (out["u_mean"].values - current_bias[3]).astype(np.float32)
    if np.isfinite(current_bias[4]):
        v_u = (out["v_mean"].values - current_bias[4]).astype(np.float32)

    # RH is carried as fraction in this archive. Small negative or >1 excursions
    # can appear after bias adjustment and should be bounded physically.
    rh_u = np.clip(rh_u, 0.0, 1.0).astype(np.float32)

    nwdir, nwspd = _wind_dir_speed(u_u, v_u)
    wind_too_large = np.isfinite(nwspd) & (nwspd > MAX_WSPD_MPS)
    if np.any(wind_too_large):
        print(f"[WARN] masking {int(np.count_nonzero(wind_too_large))} rebias wind value(s) exceeding {MAX_WSPD_MPH:.0f} mph in {nc_path.name}", flush=True)
        u_u = np.asarray(u_u, dtype=np.float32).copy()
        v_u = np.asarray(v_u, dtype=np.float32).copy()
        nwdir = np.asarray(nwdir, dtype=np.float32).copy()
        nwspd = np.asarray(nwspd, dtype=np.float32).copy()
        u_u[wind_too_large] = np.nan
        v_u[wind_too_large] = np.nan
        nwdir[wind_too_large] = np.nan
        nwspd[wind_too_large] = np.nan
    out["temp_unbiased"] = ("time", temp_u)
    out["pres_unbiased"] = ("time", pres_u)
    out["rh_unbiased"] = ("time", rh_u)
    out["u_unbiased"] = ("time", u_u)
    out["v_unbiased"] = ("time", v_u)
    out["wdir_unbiased"] = ("time", nwdir.astype(np.float32))
    out["wspd_unbiased"] = ("time", nwspd.astype(np.float32))

    out = _apply_variable_metadata(out)
    _validate_dataset_units(out, nc_path.name)
    _write_atomic_netcdf(out, nc_path, overwrite=True)


def run_case_rebias(
    outdir: Path,
    transects_csv: Path,
    tracking_dir: Path,
    dist_thresh_km: float,
    dt_tracking_min: float,
    time_to_space_conversion: int,
    legacy_distance_bool: bool = False,
    rank: int = 0,
    size: int = 1,
    comm=None,
    allowed_dates: Optional[set[str]] = None,
) -> None:
    tr = _read_transects_table(transects_csv)
    obs_start = np.array([datetime.strptime(val, "%Y%m%d-%H%M") for val in tr["Observation Start"].values])
    modes = np.array(["N" if val == "N0" else "T" for val in tr["Mode"].values])
    case_ids = tr["CaseID"].values
    obs_dates = np.array([datetime(val.year, val.month, val.day) for val in obs_start], dtype=object)
    if allowed_dates:
        keep = np.array([dt.strftime('%Y%m%d') in allowed_dates for dt in obs_dates], dtype=bool)
        obs_start = obs_start[keep]
        modes = modes[keep]
        case_ids = case_ids[keep]
        obs_dates = obs_dates[keep]
    unique_dates = np.array(sorted(set(obs_dates)), dtype=object)
    my_dates = set(unique_dates[rank::size])
    file_candidates = sorted(outdir.glob("*.nc"))
    my_case_indices = [i for i, dt in enumerate(obs_dates) if dt in my_dates]

    for x in tqdm(my_case_indices, desc=f"rebias-rank{rank}", disable=not show_pbar):
        bias_start = obs_start[x] - timedelta(hours=8)
        bias_end = obs_start[x]
        if bias_end <= bias_start:
            continue
        mmfs = np.array([str(p) for p in file_candidates if abs((get_file_date_mm(str(p)) - datetime(obs_start[x].year, obs_start[x].month, obs_start[x].day)).total_seconds()) <= 86400])
        mes, mm_file = get_mesonet_data_for_bias_window(mmfs, bias_start, bias_end)
        if not mm_file or len(mm_file) <= 2:
            continue

        case_id = str(case_ids[x]).strip()
        track_times, track_lats, track_lons = load_storm_track_window(
            case_id,
            tracking_dir,
            modes[x],
            dt_tracking=dt_tracking_min,
            start_time=bias_start,
            end_time=bias_end,
        )

        start = np.nanmax(np.array([mes["time"][i][0] for i in range(len(mes["time"]))]))
        end = np.nanmin(np.array([mes["time"][i][-1] for i in range(len(mes["time"]))]))
        if end <= start:
            continue
        tgrid = np.array([start + timedelta(seconds=tx) for tx in range(int((end - start).total_seconds()) + 1)])
        istart = np.array([np.where(mes["time"][i] >= start)[0][0] for i in range(len(mes["time"]))])
        iend = np.array([np.where(mes["time"][i] >= end)[0][0] for i in range(len(mes["time"]))])
        if np.any(iend < istart):
            continue

        try:
            all_data = {key: [] for key in mes.keys()}
            for key in mes.keys():
                for i in range(len(mes["time"])):
                    if key != "time":
                        var = interpolate_var(
                            mes[key][i][istart[i]:iend[i] + 1],
                            date2num(mes["time"][i][istart[i]:iend[i] + 1], "seconds since 1970-01-01"),
                            date2num(tgrid, "seconds since 1970-01-01"),
                        )
                        all_data[key].append(var)
                    else:
                        all_data[key].append(tgrid)
        except ValueError:
            continue

        mes = all_data
        cutoff_idx = tgrid.size
        for dt in range(1, tgrid.size):
            lons_cut = np.array([mes["lon"][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)
            lats_cut = np.array([mes["lat"][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)
            valid_cut = np.where(np.isfinite(lats_cut) & np.isfinite(lons_cut))[0]
            if valid_cut.size < 3:
                continue
            storm_dists_cut = _storm_distances_km(track_times, track_lats, track_lons, tgrid[dt], lats_cut[valid_cut], lons_cut[valid_cut])
            if np.isfinite(storm_dists_cut).all() and np.nanmax(storm_dists_cut) <= 20.0:
                cutoff_idx = dt
                break

        target_vars = ["temp", "pres", "rh", "u", "v"]
        offset = np.full((tgrid.size, len(mes["time"]), len(target_vars)), np.nan, dtype=np.float32)

        for dt in range(1, cutoff_idx):
            lons = np.array([mes["lon"][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)
            lats = np.array([mes["lat"][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)
            elevs_dt = np.array([mes["elev"][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)
            pres_dt = np.array([mes["pres"][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)
            temps_dt = np.array([mes["temp"][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)
            rh_dt = np.array([mes["rh"][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)
            z_ref = np.nanmean(elevs_dt)

            rh_dt[(rh_dt > 100) | (rh_dt < 0)] = np.nan
            if np.nanmax(rh_dt) > 2:
                rh_dt = rh_dt / 100.0

            qv = mpcalc.mixing_ratio_from_relative_humidity(
                pres_dt * units.hPa,
                temps_dt * units.degC,
                rh_dt * units.dimensionless,
            ).magnitude
            tv = ((temps_dt + 273.15) * (1 + (0.61 * qv))) - 273.15
            pres_dt_corr = pres_dt * np.exp(((elevs_dt - z_ref) / (Rd * (tv + 273.15))) * g)

            valid_platforms = np.where(np.isfinite(lats) & np.isfinite(lons))[0]
            if valid_platforms.size < 3:
                continue

            for irem in valid_platforms:
                if legacy_distance_bool:
                    target_dist = np.array([np.sqrt(np.sum(np.square(get_dx_dy(lons[irem], lats[irem], lons[idx], lats[idx])))) / 1000.0 for idx in valid_platforms], dtype=np.float32)
                    candidate_keep = np.setdiff1d(valid_platforms, irem)
                    keep_cluster = bool(target_dist[np.isin(valid_platforms, candidate_keep)].all() <= dist_thresh_km)
                    ikeep = candidate_keep if keep_cluster else np.array([], dtype=int)
                else:
                    target_dist = np.array([np.sqrt(np.sum(np.square(get_dx_dy(lons[irem], lats[irem], lons[idx], lats[idx])))) / 1000.0 for idx in valid_platforms], dtype=np.float32)
                    ikeep = valid_platforms[(target_dist <= dist_thresh_km) & (valid_platforms != irem)]
                if ikeep.size < 2:
                    continue

                coords = np.zeros((ikeep.size, 2), dtype=np.float32)
                coords[:, 0] = lats[ikeep]
                coords[:, 1] = lons[ikeep]
                tar_coord = np.array([lats[irem], lons[irem]], dtype=np.float32).reshape(1, 2)

                for k, tvar in enumerate(target_vars):
                    if tvar != "pres":
                        measured_val = np.array([mes[tvar][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)[ikeep]
                        tar_val = np.array([mes[tvar][i][dt] for i in range(len(mes["time"]))], dtype=np.float32)[irem]
                    else:
                        measured_val = pres_dt_corr[ikeep]
                        tar_val = pres_dt_corr[irem]
                    interp_var = idw_interpolation(coords, measured_val, tar_coord, power=2)[0]
                    offset[dt, irem, k] = tar_val - interp_var

        bias_mag = np.nanmean(offset, axis=0)
        for ifile, fname in enumerate(mm_file):
            _apply_bias_update(Path(fname), np.asarray(bias_mag[ifile], dtype=np.float32))
    if comm is not None:
        comm.Barrier()


def recompute_reduced_pressure(outdir: Path, transects_csv: Path, time_to_space_minutes: int, rank: int = 0, size: int = 1, comm=None) -> None:
    """Deprecated no-op. Reduced pressure is computed per analysis time in get_mesonet_data()."""
    if rank == 0:
        print("[INFO] reduced-pressure archive recompute skipped; pres_corr is computed during synthesis", flush=True)
    if comm is not None:
        comm.Barrier()

def _prepare_mesonet_archive(
    raw_dir: Path,
    source_dir: Optional[Path],
    prepared_dir: Path,
    transects_csv: Path,
    tracking_dir: Path,
    elev_dir: Optional[Path],
    time_to_space_minutes: int,
    window_seconds: int,
    dist_thresh_km: float,
    dt_tracking_min: float,
    overwrite: bool,
    rank: int = 0,
    size: int = 1,
    comm=None,
    allowed_dates: Optional[set[str]] = None,
    filter_accelerating_winds: bool = False,
    vehicle_dv_thresh_mps: float = 0.4,
    vehicle_dv_window_sec: float = 6.0,
    vehicle_dv_use_vector: bool = True,
    skip_rebias: bool = False,
    trim_mode: str = "observation",
) -> None:
    raw_files = _iter_raw_files(raw_dir, allowed_dates=allowed_dates)
    if not raw_files:
        raise FileNotFoundError(f"No raw mobile mesonet files found in {raw_dir}")
    if rank == 0:
        prepared_dir.mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()
    for raw_path in tqdm(raw_files[rank::size], desc=f"rebuild mesonet-rank{rank}", disable=not show_pbar):
        out_name = _to_output_name(raw_path)
        source_path = (source_dir / out_name) if source_dir is not None else None
        out_path = prepared_dir / out_name
        try:
            rebuilt_ds = _build_rebuilt_dataset(
                raw_path,
                source_path if source_path is not None and source_path.exists() else None,
                window_seconds,
                elev_dir=elev_dir,
                filter_accelerating_winds=filter_accelerating_winds,
                vehicle_dv_thresh_mps=vehicle_dv_thresh_mps,
                vehicle_dv_window_sec=vehicle_dv_window_sec,
                vehicle_dv_use_vector=vehicle_dv_use_vector,
            )
        except ValueError as exc:
            msg = str(exc)
            skippable = (
                "zero usable rows" in msg
                or "zero-size" in msg
                or "readMM returned no data" in msg
                or "mismatched" in msg
                or "no valid in-range datetimes" in msg
                or "all parsed timestamps are outside" in msg
                or "no pandas-convertible datetimes" in msg
                or "position_available" in msg
                or "no finite position data" in msg
                or "no valid USGS elevation values" in msg
                or "pressure-corrected input excluded" in msg
            )
            if skippable:
                print(f"[WARN] skipping {raw_path.name}: {exc}", flush=True)
                continue
            raise
        _write_atomic_netcdf(rebuilt_ds, out_path, overwrite=overwrite)
        rebuilt_ds.close()
    if comm is not None:
        if rank == 0:
            print('[INFO] all ranks completed raw rebuild; entering rebias stage', flush=True)
        comm.Barrier()

    if not skip_rebias:
        run_case_rebias(
        outdir=prepared_dir,
        transects_csv=transects_csv,
        tracking_dir=tracking_dir,
        dist_thresh_km=dist_thresh_km,
        dt_tracking_min=dt_tracking_min,
        time_to_space_conversion=time_to_space_minutes,
        legacy_distance_bool=False,
        rank=rank,
        size=size,
        comm=comm,
        allowed_dates=allowed_dates,
        )
    elif rank == 0:
        print('[INFO] skipping rebias stage (--skip-rebias)', flush=True)

    if comm is not None:
        comm.Barrier()
        if rank == 0:
            print('[INFO] all ranks completed/skipped rebias; entering trim stage', flush=True)

    trim_mode = str(trim_mode or "observation").lower()
    if trim_mode in {"none", "skip", "full", "raw"}:
        if rank == 0:
            print('[INFO] preserving full raw timestamp span (--skip-trim/trim_mode=none)', flush=True)
        if comm is not None:
            comm.Barrier()
        return
    if trim_mode not in {"observation", "analysis"}:
        raise ValueError(f"unknown trim_mode={trim_mode!r}; expected observation, analysis, or none")

    if rank == 0:
        if trim_mode == "observation":
            print('[INFO] trimming processed NetCDF files to Observation Start/End windows', flush=True)
        else:
            print('[INFO] trimming processed NetCDF files to analysis windows around Tstart', flush=True)

    nc_paths = sorted(prepared_dir.glob("*.nc"))
    for nc_path in nc_paths[rank::size]:
        file_date = _processed_file_date(nc_path)
        if file_date is None:
            print(f"[WARN] trim stage removing unparseable processed file date: {nc_path.name}", flush=True)
            nc_path.unlink(missing_ok=True)
            continue
        with xr.open_dataset(nc_path) as ds:
            loaded = ds.load()
        if trim_mode == "observation":
            trimmed = _trim_dataset_to_observation_windows(loaded, file_date, transects_csv)
        else:
            trimmed = _trim_dataset_to_analysis_windows(loaded, file_date, transects_csv, time_to_space_minutes)
        loaded.close()
        if trimmed is None:
            nc_path.unlink(missing_ok=True)
            continue
        trimmed = _apply_variable_metadata(trimmed)
        _validate_dataset_units(trimmed, nc_path.name)
        _write_atomic_netcdf(trimmed, nc_path, overwrite=True)
        trimmed.close()
    if comm is not None:
        comm.Barrier()

    # Reduced pressure is intentionally computed in build_mobile_mesonet_synthesis.py
    # because the reference elevation is analysis-time dependent.



def _parse_args():
    p = argparse.ArgumentParser(
        description='Build mobile_mesonet_processed/ from raw mobile mesonet files. No pickle reconciliation is performed.'
    )
    p.add_argument('--workdir', default='.', help='Workspace containing transects, storm_tracks, elevation tiles, etc.')
    p.add_argument('--raw-dir', required=True, help='Raw mobile mesonet directory.')
    p.add_argument('--source-dir', default=None, help='Optional reference/source NetCDF archive used only for time-axis alignment and source-elevation fallback when rebuilding from raw.')
    p.add_argument('--prepared-dir', default='mobile_mesonet_processed', help='Output directory for processed/rebiased mobile mesonet NetCDF files.')
    p.add_argument('--transects', default='transect_times.csv', help='Transect timing table (.csv, .xlsx, or .xls).')
    p.add_argument('--tracking-dir', default='storm_tracks', help='Storm track directory.')
    p.add_argument('--elev-dir', default='elevation', help='Directory of USGS elevation GeoTIFF tiles.')
    p.add_argument('--time-to-space', type=int, default=5, choices=[5, 10], help='Time-to-space conversion in minutes.')
    p.add_argument('--window-seconds', type=int, default=10, help='Moving-statistics window length in seconds for raw rebuild.')
    p.add_argument('--dist-thresh-km', type=float, default=5.0, help='Platform proximity threshold in km for bias estimation.')
    p.add_argument('--dt-tracking-min', type=float, default=1/60, help='Storm-track interpolation interval in minutes for bias estimation.')
    p.add_argument('--filter-accelerating-winds', action='store_true',
                   help='Blank wind variables when GPS-derived vehicle velocity changes too rapidly, following Markowski-style wind QC.')
    p.add_argument('--vehicle-dv-thresh-mps', type=float, default=0.4,
                   help='Vehicle-velocity change threshold in m/s over the QC window for accelerating-wind filtering.')
    p.add_argument('--vehicle-dv-window-sec', type=float, default=6.0,
                   help='Time window in seconds for the accelerating-wind filter.')
    p.add_argument('--vehicle-dv-use-vector', action='store_true', default=True,
                   help='Use full vehicle velocity-vector change instead of only speed change for accelerating-wind filtering.')
    p.add_argument('--vehicle-dv-use-speed', dest='vehicle_dv_use_vector', action='store_false',
                   help='Use only vehicle speed change instead of full vector change for accelerating-wind filtering.')
    p.add_argument('--overwrite-prepared', action='store_true', help='Delete/rebuild the processed directory before writing.')
    p.add_argument('--skip-rebias', action='store_true', help='Skip inter-platform bias correction after rebuilding NetCDF files. Useful for smoke tests/debugging.')
    p.add_argument(
        '--trim-analysis-windows',
        action='store_true',
        help='Legacy option: trim processed NetCDF files to +/-20 min analysis windows around each transect Tstart instead of Observation Start/End.',
    )
    p.add_argument(
        '--skip-trim',
        action='store_true',
        help='Preserve all valid raw timestamps and do not trim processed NetCDF files. Default now trims to Observation Start/End.',
    )
    p.add_argument('--analysis-keys', nargs='*', default=None, help='Optional list of YYYYMMDDHHMM analysis keys to run as a subset smoke test.')
    return p.parse_args()


def _expanded_allowed_dates_for_analysis_keys(analysis_keys: set[str]) -> Optional[set[str]]:
    if not analysis_keys:
        return None
    out = set()
    for key in analysis_keys:
        try:
            base = datetime.strptime(key[:8], "%Y%m%d")
        except Exception:
            continue
        for delta_days in (-1, 0, 1):
            out.add((base + timedelta(days=delta_days)).strftime("%Y%m%d"))
    return out


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = _parse_args()
    workdir = Path(os.path.abspath(args.workdir))
    raw_dir = _resolve(workdir, args.raw_dir)
    source_dir = _resolve(workdir, args.source_dir) if args.source_dir is not None else None
    prepared_dir = _resolve(workdir, args.prepared_dir)
    transects_path = _resolve(workdir, args.transects)
    tracking_dir = _resolve(workdir, args.tracking_dir)
    elev_file_path = _resolve(workdir, args.elev_dir)
    analysis_keys = set(args.analysis_keys or [])
    allowed_dates = _expanded_allowed_dates_for_analysis_keys(analysis_keys)

    required_paths = [
        (raw_dir, 'raw mobile mesonet directory'),
        (transects_path, 'transect_times table'),
        (tracking_dir, 'storm_tracks directory'),
        (elev_file_path, 'elevation directory'),
    ]
    if source_dir is not None:
        required_paths.append((source_dir, 'reference/source mesonet NetCDF directory'))
    for path_value, label in required_paths:
        if not os.path.exists(path_value):
            raise FileNotFoundError(f'Missing {label}: {path_value}')

    os.chdir(str(workdir))

    if rank == 0 and prepared_dir.exists() and args.overwrite_prepared:
        shutil.rmtree(prepared_dir)
    comm.Barrier()

    _prepare_mesonet_archive(
        raw_dir=Path(raw_dir),
        source_dir=Path(source_dir) if source_dir is not None else None,
        prepared_dir=Path(prepared_dir),
        transects_csv=Path(transects_path),
        tracking_dir=Path(tracking_dir),
        elev_dir=Path(elev_file_path) if elev_file_path is not None else None,
        time_to_space_minutes=args.time_to_space,
        window_seconds=args.window_seconds,
        dist_thresh_km=args.dist_thresh_km,
        dt_tracking_min=args.dt_tracking_min,
        overwrite=args.overwrite_prepared,
        rank=rank,
        size=size,
        comm=comm,
        allowed_dates=allowed_dates,
        filter_accelerating_winds=args.filter_accelerating_winds,
        vehicle_dv_thresh_mps=args.vehicle_dv_thresh_mps,
        vehicle_dv_window_sec=args.vehicle_dv_window_sec,
        vehicle_dv_use_vector=args.vehicle_dv_use_vector,
        skip_rebias=args.skip_rebias,
        trim_mode=("none" if args.skip_trim else ("analysis" if args.trim_analysis_windows else "observation")),
    )
    comm.Barrier()

    if rank == 0:
        print(f'[INFO] Wrote processed mobile mesonet NetCDF archive: {prepared_dir}')


if __name__ == '__main__':
    main()
