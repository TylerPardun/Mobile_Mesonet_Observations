#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import pickle
import re
import random
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

def _is_wsr88d_radar_id(radar_id):
    """Return True for WSR-88D/NEXRAD radar IDs and False for TDWR sites.

    TDWR sites commonly use T-prefix IDs such as TOKC and TDAL. For consistency
    in the mesonet synthesis reflectivity sampling, keep only WSR-88D/NEXRAD
    sites from Py-ART's NEXRAD_LOCATIONS table.
    """
    rid = str(radar_id or "").strip().upper()
    if not rid:
        return False
    # Explicitly reject terminal Doppler weather radars.
    if rid.startswith("T"):
        return False
    return rid in NEXRAD_LOCATIONS


def _filter_wsr88d_radars(radars):
    """Deduplicate and keep only WSR-88D/NEXRAD radar IDs."""
    out = []
    dropped = []
    for rid in radars or []:
        r = str(rid).strip().upper()
        if not r:
            continue
        if _is_wsr88d_radar_id(r):
            if r not in out:
                out.append(r)
        else:
            if r not in dropped:
                dropped.append(r)
    if dropped:
        print(f"[INFO] dropping non-WSR-88D radar candidates: {','.join(dropped)}", flush=True)
    return out


def _parse_candidate_radars(candidate_radars):
    if candidate_radars is None:
        return []
    if isinstance(candidate_radars, str):
        vals = re.split(r"[,;\s]+", candidate_radars)
    else:
        vals = candidate_radars
    out = []
    for val in vals:
        text = str(val).strip().upper()
        # Only parse WSR-88D/NEXRAD-looking IDs. This deliberately excludes
        # T-prefix TDWR IDs like TOKC and TDAL.
        m = re.search(r"\b[A-Z][A-Z0-9]{3}\b", text)
        if not m:
            continue
        rid = m.group(0)
        if rid not in out:
            out.append(rid)
    return _filter_wsr88d_radars(out)


def _candidate_radars_cache_key(candidate_radars):
    """Return a hashable, normalized radar-candidate key for lru-cached functions."""
    return ",".join(_parse_candidate_radars(candidate_radars))


def _rank_nearest_nexrad_radars(lat, lon):
    """Return NEXRAD IDs sorted by distance from a lat/lon point."""
    try:
        lat0 = float(lat)
        lon0 = float(lon)
    except Exception:
        return []
    if not (np.isfinite(lat0) and np.isfinite(lon0)):
        return []

    ranked = []
    for rid, loc in NEXRAD_LOCATIONS.items():
        rid = str(rid).upper()
        if not _is_wsr88d_radar_id(rid):
            continue
        try:
            rlat = float(loc["lat"])
            rlon = float(loc["lon"])
        except Exception:
            continue
        dist = float(_great_circle_distance_m(rlon, rlat, lon0, lat0))
        ranked.append((dist, rid))
    ranked.sort(key=lambda item: item[0])
    return [rid for _dist, rid in ranked]


def _expand_candidate_radars(candidate_radars, meso_lat=None, meso_lon=None, max_candidate_radars=8):
    """Preserve spreadsheet radars, then append nearest WSR-88Ds.

    This lets unavailable radars such as KFDX be skipped without limiting the
    search to only the 4 radar IDs listed in transect_times.xlsx.
    """
    out = []
    for rid in _parse_candidate_radars(candidate_radars):
        if rid not in out:
            out.append(rid)

    out = _filter_wsr88d_radars(out)

    if meso_lat is not None and meso_lon is not None and max_candidate_radars is not None:
        lat = np.asarray(meso_lat, dtype=float)
        lon = np.asarray(meso_lon, dtype=float)
        good = np.isfinite(lat) & np.isfinite(lon)
        if np.any(good):
            mid = int(np.sum(good) // 2)
            lat0 = lat[good][mid]
            lon0 = lon[good][mid]
            for rid in _rank_nearest_nexrad_radars(lat0, lon0):
                if rid not in out:
                    out.append(rid)
                if len(out) >= int(max_candidate_radars):
                    break

    return _filter_wsr88d_radars(out)


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
                if "MDM" in base.upper():
                    continue
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

    scan_key = str(scan_key)
    base = os.path.basename(scan_key)
    # MDM files are metadata/messages, not Level-II radar volumes. Do not try to
    # read/download them as reflectivity data.
    if "MDM" in base.upper():
        return None

    radar_root = Path(radar_file_dir)
    out_path = radar_root / scan_key
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://unidata-nexrad-level2.s3.amazonaws.com/{scan_key}",
        f"https://noaa-nexrad-level2.s3.amazonaws.com/{scan_key}",
    ]
    for url in urls:
        tmp = None
        try:
            # Unique temp file avoids MPI ranks clobbering each other while
            # downloading the same volume.
            tmp = out_path.with_name(
                f".{out_path.name}.{os.getpid()}.{random.randint(0, 999999):06d}.tmp"
            )
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size <= 0:
                tmp.unlink(missing_ok=True)
                continue

            # Another rank may have published it while we were downloading.
            if out_path.exists() and out_path.stat().st_size > 0:
                tmp.unlink(missing_ok=True)
                return str(out_path)

            tmp.replace(out_path)
            print(f"[INFO] downloaded radar volume {url}", flush=True)
            return str(out_path)
        except Exception:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
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
    cleaned = []
    for scan in scans:
        try:
            key = getattr(scan, "key", str(scan))
            base = os.path.basename(str(key))
            if "MDM" in base.upper():
                continue
            rid = _extract_radar_id_from_path(base) or _extract_radar_id_from_path(key)
            if rid is not None and not _is_wsr88d_radar_id(rid):
                continue
            if (scan.scan_time >= start_dt) and (scan.scan_time <= end_dt):
                cleaned.append(scan)
        except Exception:
            continue
    scans = cleaned
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
    direct = sorted(
        p for p in glob(os.path.join(radar_root, f"{mode}_{case_index}", "*"))
        if os.path.isfile(p)
        and "MDM" not in os.path.basename(p).upper()
        and (_extract_radar_id_from_path(p) is None or _is_wsr88d_radar_id(_extract_radar_id_from_path(p)))
    )
    if direct:
        return tuple(direct)

    candidate_radars = set(_parse_candidate_radars(candidate_radars_key))
    all_files = [
        p for p in glob(os.path.join(radar_root, "**", "*"), recursive=True)
        if os.path.isfile(p) and "MDM" not in os.path.basename(p).upper()
    ]
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
    """Return per-sweep times/elevations, tolerating malformed sweeps.

    Some Level-II files contain sweeps whose ray time array is all masked/NaN.
    The old implementation tried to build a timedelta from NaN and killed the
    MPI rank.  Mark those sweeps unusable instead; downstream matching will
    ignore them and continue to the next readable sweep/radar.
    """
    vol0 = datetime.strptime(radar.time["units"].split()[2], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    sweep_times = np.full(radar.nsweeps, None, dtype=object)
    sweep_means = np.full(radar.nsweeps, np.nan, dtype=float)
    for si in range(radar.nsweeps):
        istart = radar.sweep_start_ray_index["data"][si]
        iend = radar.sweep_end_ray_index["data"][si] + 1
        elev = np.asarray(radar.elevation["data"][istart:iend], dtype=float)
        tsec = np.asarray(radar.time["data"][istart:iend], dtype=float)
        good_t = np.isfinite(tsec)
        good_e = np.isfinite(elev)
        if np.any(good_e):
            sweep_means[si] = float(np.nanmean(elev[good_e]))
        if np.any(good_t):
            sweep_times[si] = vol0 + timedelta(seconds=float(np.nanmedian(tsec[good_t])))
    return sweep_times, sweep_means


def _choose_best_sweep_indices(radar, obs_times, elev_angle, elev_tol=0.3):
    obs_times = np.array([_to_utc_datetime(val) for val in np.asarray(obs_times)], dtype=object)
    sweep_times, sweep_means = _get_sweep_datetimes(radar)
    usable = np.array(
        [si for si in range(len(sweep_times)) if sweep_times[si] is not None and np.isfinite(sweep_means[si])],
        dtype=int,
    )
    if usable.size == 0:
        return np.full(obs_times.size, -1, dtype=int)

    eligible = usable[np.abs(sweep_means[usable] - float(elev_angle)) <= float(elev_tol)]
    if eligible.size == 0:
        eligible = np.array([int(usable[int(np.argmin(np.abs(sweep_means[usable] - float(elev_angle))))])], dtype=int)

    best = np.full(obs_times.size, -1, dtype=int)
    for i, t_obs in enumerate(obs_times):
        dt_sec = np.array([abs((sweep_times[si] - t_obs).total_seconds()) for si in eligible], dtype=float)
        if dt_sec.size and np.isfinite(dt_sec).any():
            best[i] = int(eligible[int(np.nanargmin(dt_sec))])
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

@lru_cache(maxsize=None)
def _find_case_track_file(caseid, tracking_dir, mode):
    caseid = str(caseid).strip()
    subdir = 'tornadic' if mode in {'T', 'L', 'S', 'W', 'Sk'} else 'nontornadic'
    pattern = os.path.join(str(tracking_dir), subdir, f'{caseid}_*.csv')
    matches = sorted(
        p for p in glob(pattern)
        if os.path.isfile(p) and not os.path.basename(p).startswith(('.', '._'))
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple storm-track files match CaseID={caseid}: {matches}")
    return None


@lru_cache(maxsize=None)
def _load_storm_track(caseid, tracking_dir, mode):
    """Load one storm track by literal CaseID.

    Supports both current track columns (time, lat, lon) and legacy columns
    (datetime, latitude, longitude).
    """
    mmfs = _find_case_track_file(caseid, tracking_dir, mode)
    if mmfs is None:
        raise FileNotFoundError(f"No storm-track file found for CaseID={caseid} mode={mode} in {tracking_dir}")

    strk = pd.read_csv(mmfs)
    if strk.empty:
        raise ValueError(f"Storm-track file is empty: {mmfs}")

    colmap = {str(c).strip().lower(): c for c in strk.columns}
    time_col = next((colmap[c] for c in ('time', 'datetime', 'time_utc', 'date_time') if c in colmap), None)
    lat_col = next((colmap[c] for c in ('lat', 'latitude', 'mesocyclone_latitude', 'track lat') if c in colmap), None)
    lon_col = next((colmap[c] for c in ('lon', 'longitude', 'mesocyclone_longitude', 'track lon') if c in colmap), None)
    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"{mmfs} missing recognizable time/lat/lon columns: {strk.columns.tolist()}")

    mt = pd.to_datetime(strk[time_col].values, errors='coerce')
    good_time = ~pd.isna(mt)
    if good_time.sum() < 2:
        raise ValueError(f"Storm-track file has fewer than two valid times: {mmfs}")

    mt = mt[good_time]
    srlat = pd.to_numeric(strk[lat_col], errors='coerce').to_numpy(float)[good_time]
    srlon = pd.to_numeric(strk[lon_col], errors='coerce').to_numpy(float)[good_time]
    good_pos = np.isfinite(srlat) & np.isfinite(srlon)
    if good_pos.sum() < 2:
        raise ValueError(f"Storm-track file has fewer than two valid positions: {mmfs}")

    mt = mt[good_pos]
    srlat = srlat[good_pos]
    srlon = srlon[good_pos]
    mtimes = np.array([datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in mt], dtype=object)

    isort = np.argsort(mtimes)
    mtimes, srlat, srlon = mtimes[isort], srlat[isort], srlon[isort]
    tnum = np.asarray(date2num(mtimes, 'seconds since 1970-01-01'), dtype=float)
    _, unique_idx = np.unique(tnum, return_index=True)
    unique_idx = np.sort(unique_idx)
    return mtimes[unique_idx], srlat[unique_idx], srlon[unique_idx]


@lru_cache(maxsize=None)
def _load_interp_storm_track(caseid, tracking_dir, mode, dt_tracking):
    mtimes, srlat, srlon = _load_storm_track(caseid, tracking_dir, mode)
    time_grid = np.arange(
        date2num(mtimes[0], 'seconds since 1970-01-01'),
        date2num(mtimes[-1], 'seconds since 1970-01-01') + float(dt_tracking) * 60.0,
        float(dt_tracking) * 60.0,
    )
    srlat_i = np.interp(time_grid, date2num(mtimes, 'seconds since 1970-01-01'), srlat)
    srlon_i = np.interp(time_grid, date2num(mtimes, 'seconds since 1970-01-01'), srlon)
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

def parse_storm_tracks(x, tracking_dir, mode, atime, dt_tracking=1/60, time_to_space_conversion=10):
    """Return storm-track samples and storm motion in the analysis window.

    x is the literal CaseID from transect_times.xlsx, not a zero-based numeric
    index. Returns None when the requested window falls outside track coverage.
    """
    try:
        mtimes_i, srlat_i, srlon_i = _load_interp_storm_track(str(x).strip(), str(tracking_dir), mode, float(dt_tracking))
    except Exception:
        return None

    win_start, win_end = _analysis_window(atime, time_to_space_conversion)
    imeso = np.where((mtimes_i >= win_start) & (mtimes_i <= win_end))[0]
    if imeso.size < 2:
        return None

    mtimes_i = mtimes_i[imeso]
    srlat_i = srlat_i[imeso]
    srlon_i = srlon_i[imeso]
    isort = np.argsort(mtimes_i)
    mtimes_i, srlat_i, srlon_i = mtimes_i[isort], srlat_i[isort], srlon_i[isort]

    dt_storm = (mtimes_i[-1] - mtimes_i[0]).total_seconds()
    if dt_storm <= 0:
        return None
    dx_storm, dy_storm = get_dx_dy(srlon_i[0], srlat_i[0], srlon_i[-1], srlat_i[-1])
    u_storm, v_storm = dx_storm / dt_storm, dy_storm / dt_storm
    return mtimes_i, srlat_i, srlon_i, u_storm, v_storm


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
    nexrad_files = list(_discover_local_radar_files(radar_file_dir, mode, x, _candidate_radars_cache_key(candidate_radars), day_key))
    if not nexrad_files and candidate_radars is not None:
        _cache_candidate_radar_volumes(candidate_radars, mtime, radar_file_dir, time_tolerance_sec)
        nexrad_files = list(_discover_local_radar_files(radar_file_dir, mode, x, _candidate_radars_cache_key(candidate_radars), day_key))
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
            elev = np.asarray(radar.elevation["data"][istart:iend], dtype=float)
            tsec = np.asarray(radar.time["data"][istart:iend], dtype=float)
            good_e = np.isfinite(elev)
            good_t = np.isfinite(tsec)
            if (not np.any(good_e)) or (not np.any(good_t)):
                continue

            if np.isclose(np.nanmean(elev[good_e]), elev_angle, atol=0.2):
                # mean time (sec offset) for rays in sweep
                tdt  = vol0 + timedelta(seconds=float(np.nanmedian(tsec[good_t])))
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

def get_mesonet_data(dir_mesonet, elev_file_path, atime, radar_file_dir, mode, caseid, candidate_radars=None, time_to_space_conversion=5, elev_angle=0.5, time_tolerance_sec=360,search_radius_m=100.0,neighborhood_shape="circle",min_gates=1,fallback_to_nearest=True,filter_accelerating_winds=False,vehicle_dv_thresh_mps=0.4,vehicle_dv_window_sec=6.0,vehicle_dv_use_vector=True,skip_radar=False,elevation_cache=None):
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
                                    'thv','the','thw','tv','elev', 'elev_raw', 'campaign', 'probe',
                                    'z_ref_m', 'z_ref_method', 'z_ref_bbox',
                                    'z_ref_dem_files', 'z_ref_n_dem_pixels']}
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    z_ref_info = _analysis_reference_elevation(elev_file_path, dir_mesonet, atime, time_to_space_conversion=time_to_space_conversion, elevation_cache=elevation_cache, caseid=caseid, mode=mode)
    z_ref_m = float(z_ref_info.get('z_ref_m', np.nan))
    if not np.isfinite(z_ref_m):
        print(
            f"[WARN] no finite DEM reference elevation for analysis time {atime}; "
            f"bbox={z_ref_info.get('bbox_string', '')}; pres_corr will be NaN",
            flush=True,
        )

    # Process each mesonet file
    for fi in mmfs:
        with xr.open_dataset(fi, decode_times=False) as m:
            winds_prefiltered = str(m.attrs.get("accelerating_winds_filtered", "false")).lower() == "true"
            tm = m.time.values
            mask = (tm >= date2num(atime - time_window, 'seconds since 1970-01-01')) & (tm <= date2num(atime + time_window,'seconds since 1970-01-01'))
            # Extract relevant data and filter
            try:
                temp = m.temp_unbiased.sel(time=mask).values
                pres = m.pres_unbiased.sel(time=mask).values
                rh = m.rh_unbiased.sel(time=mask).values
                wspd = m.wspd_unbiased.sel(time=mask).values
                wdir = m.wdir_unbiased.sel(time=mask).values
                elev_raw = m.elev.sel(time=mask).values
                elev = (m.elev_mean if "elev_mean" in m else m.elev).sel(time=mask).values
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
            pos_valid = (~np.isnan(lat)) & (~np.isnan(lon)) & (lon < 0)
            temp_valid0 = pos_valid & (~np.isnan(temp)) & (temp > -60) & (temp < 120)
            rh_valid0 = pos_valid & (~np.isnan(rh)) & (rh >= 0) & (rh <= 110)
            pres_valid0 = pos_valid & (~np.isnan(pres)) & (pres > 100) & (pres < 1050)
            elev_valid0 = pos_valid & np.isfinite(elev)
            wind_valid0 = pos_valid & (~np.isnan(wspd)) & (wspd >= 0) & (wspd < 200) & (~np.isnan(wdir)) & (wdir >= 0) & (wdir <= 360)
            
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
            elev_raw = np.asarray(elev_raw, dtype=float).copy()
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
            pres_corr_valid = (
                np.isfinite(z_ref_m)
                & np.isfinite(pres)
                & np.isfinite(temp)
                & np.isfinite(rh)
                & np.isfinite(elev)
            )
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

            ref = np.full(tm.shape, np.nan, dtype=float)
            if not skip_radar:
                if candidate_radars is not None and tm.size > 0:
                    _cache_candidate_radar_volumes(candidate_radars, tm, radar_file_dir, time_tolerance_sec)

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

            # Derive the liquid water mixing ratio [Rutledge & Hobbs (1984) vs. Hane & Ray (1985)].
            # If radar is skipped or unavailable, NaN reflectivity implies ql=0.
            ql = np.full(ref.shape, np.nan, dtype=float)
            alpha = 0.1 * ref_for_ql
            ql = ((1 / rho) * (((10 ** alpha) / 17300) ** (4 / 7))) / 1000
            ql[np.isnan(ql)] = 0
            
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
                2022: 'TORUS_22', 2023: 'TORUS_LiTE', 2024: 'LIFT', 2025: 'LIFT'
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
            all_data['elev_raw'].append(elev_raw)
            all_data['campaign'].append([campaign] * len(temp))
            all_data['probe'].append([fi.split('/')[-1]] * len(temp))
            all_data['z_ref_m'].append(np.full(len(temp), z_ref_m, dtype=float))
            all_data['z_ref_method'].append(['dem_mesonet_bbox'] * len(temp))
            all_data['z_ref_bbox'].append([z_ref_info.get('bbox_string', '')] * len(temp))
            all_data['z_ref_dem_files'].append([z_ref_info.get('dem_files_string', '')] * len(temp))
            all_data['z_ref_n_dem_pixels'].append(np.full(len(temp), z_ref_info.get('n_dem_pixels', 0), dtype=int))
            
    # Convert lists to arrays
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key]) if all_data[key] else np.array([])
    
    return all_data

def process_case(dir_mesonet, sticknet_dir, elev_file_path, sound, shear_layermin, shear_layermax, atime, u_storm, v_storm, radar_file_dir, mode, caseid, meso_time, meso_lat, meso_lon, candidate_radars=None, time_to_space_conversion=5, filter_accelerating_winds=False, vehicle_dv_thresh_mps=0.4, vehicle_dv_window_sec=6.0, vehicle_dv_use_vector=True, skip_radar=False, elevation_cache=None, max_candidate_radars=8):

    candidate_radars = _expand_candidate_radars(
        candidate_radars,
        meso_lat=meso_lat,
        meso_lon=meso_lon,
        max_candidate_radars=max_candidate_radars,
    )

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
        skip_radar=skip_radar,
        elevation_cache=elevation_cache,
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
    matches = sorted(glob(os.path.join(elev_dir, f"*{tile_id}*.tif")) + glob(os.path.join(elev_dir, f"*{tile_id}*.tiff")))
    return matches[0] if matches else None


def _resolve_elev_tile_path(elev_dir: Path, tile_id: str, allow_download: bool = True) -> Optional[str]:
    local_path = _find_local_elev_tile(str(elev_dir), tile_id)
    if local_path is None and allow_download:
        downloaded = _download_elev_tile(elev_dir, tile_id)
        if downloaded is not None:
            local_path = str(downloaded)
            _find_local_elev_tile.cache_clear()
    return local_path


@lru_cache(maxsize=128)
def _open_elev_tile(tile_path: str):
    src = rasterio.open(tile_path)
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
        return out_path

    try:
        with requests.get(f"https://prd-tnm.s3.amazonaws.com/{key}", stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(out_path)
    except Exception:
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

        src, is_geographic, nodata = _open_elev_tile(local_path)
        xs = lons[idx]
        ys = lats[idx]
        if not is_geographic:
            xs, ys = rio_transform("EPSG:4326", src.crs, xs.tolist(), ys.tolist())
        samples = np.fromiter((val[0] for val in src.sample(list(zip(xs, ys)))), dtype=np.float32, count=idx.size)
        if nodata is not None:
            samples[np.isclose(samples, nodata)] = np.nan
        elev[idx] = samples

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
        "elev_mean": {"units": "m", "long_name": "forward moving-mean elevation above mean sea level"},
        "elev_std": {"units": "m", "long_name": "forward moving-window elevation standard deviation"},
        "elev_var": {"units": "m^2", "long_name": "forward moving-window elevation variance"},
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
        "wspd": (0.0, 120.0),
        "wspd_mean": (0.0, 120.0),
        "wspd_unbiased": (0.0, 120.0),
        "wdir": (0.0, 360.0),
        "wdir_mean": (0.0, 360.0),
        "wdir_unbiased": (0.0, 360.0),
        "u": (-120.0, 120.0),
        "u_mean": (-120.0, 120.0),
        "u_unbiased": (-120.0, 120.0),
        "v": (-120.0, 120.0),
        "v_mean": (-120.0, 120.0),
        "v_unbiased": (-120.0, 120.0),
        "elev": (-500.0, 5000.0),
        "elev_mean": (-500.0, 5000.0),
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


def _tile_ids_for_bbox(lon_min: float, lat_min: float, lon_max: float, lat_max: float):
    """Return USGS 1-degree tile IDs intersecting a lon/lat bbox.

    This follows the same tile convention used in preprocessing:
    lat 34.5 -> n35, lon -101.8 -> w102.
    """
    vals = [lon_min, lat_min, lon_max, lat_max]
    if not np.all(np.isfinite(vals)):
        return []
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    lat_ids = range(int(np.ceil(lat_min)), int(np.ceil(lat_max)) + 1)
    lon_ids = range(int(-np.floor(lon_max)), int(-np.floor(lon_min)) + 1)
    out = []
    for lat_id in lat_ids:
        for lon_id in lon_ids:
            lat_prefix = "n" if lat_id >= 0 else "s"
            lon_prefix = "w" if lon_id >= 0 else "e"
            out.append(f"{lat_prefix}{abs(lat_id):02d}{lon_prefix}{abs(lon_id):03d}")
    return sorted(set(out))


def _dem_mean_in_bbox(elev_dir: Path, lon_min: float, lat_min: float, lon_max: float, lat_max: float):
    """Mean USGS/TNM DEM elevation in a lon/lat bbox.

    Missing 3DEP/TNM GeoTIFF tiles are downloaded using the same S3 path used
    by preprocessing. No platform-elevation fallback is used: if DEM coverage is
    unavailable, the returned z_ref_m is NaN.
    """
    elev_dir = Path(elev_dir)
    elev_dir.mkdir(parents=True, exist_ok=True)

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
                    xs, ys = rio_transform("EPSG:4326", src.crs, xs, ys)
                left, right = min(xs), max(xs)
                bottom, top = min(ys), max(ys)

                try:
                    window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
                    window = window.round_offsets().round_lengths()
                    # Clip the window to the raster extent.
                    full = rasterio.windows.Window(0, 0, src.width, src.height)
                    window = window.intersection(full)
                except Exception:
                    continue

                if window.width <= 0 or window.height <= 0:
                    continue
                arr = src.read(1, window=window, masked=True)
                if np.ma.isMaskedArray(arr):
                    data = arr.filled(np.nan).astype(float)
                else:
                    data = np.asarray(arr, dtype=float)
                nodata = src.nodata
                if nodata is not None:
                    data[np.isclose(data, nodata)] = np.nan
                data[(data < -500.0) | (data > 5000.0)] = np.nan
                finite = data[np.isfinite(data)]
                if finite.size:
                    vals.append(finite)
                    used_files.append(str(local_path))
                    n_pixels += int(finite.size)
        except Exception as exc:
            print(f"[WARN] failed reading DEM tile {local_path} for bbox mean: {exc!r}", flush=True)
            continue

    if vals:
        z_ref_m = float(np.nanmean(np.concatenate(vals)))
    else:
        z_ref_m = np.nan

    bbox_string = f"{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}"
    return {
        "z_ref_m": z_ref_m,
        "bbox": (float(lon_min), float(lat_min), float(lon_max), float(lat_max)),
        "bbox_string": bbox_string,
        "dem_files": used_files,
        "dem_files_string": ";".join(used_files),
        "n_dem_pixels": int(n_pixels),
        "tile_ids": tile_ids,
    }



def _load_analysis_elevation_cache(cache_path):
    """Load analysis_elevation_cache_tsc*.pkl and return its by_key map.

    Supports the object written by build_analysis_elevation_cache.py:
      {"by_key": {...}, "by_mode": {...}, "records": [...]}

    The synthesis script only needs z_ref_m + provenance for a case/time.
    """
    if cache_path is None:
        return None
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"analysis elevation cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "by_key" in obj:
        return obj
    raise ValueError(f"Unrecognized analysis elevation cache structure: {cache_path}")


def _cache_lookup_reference_elevation(elevation_cache, caseid, mode, atime):
    """Return z_ref info from analysis elevation cache or None if unavailable."""
    if not elevation_cache:
        return None

    key_time = atime.strftime("%Y%m%d%H%M")
    caseid = str(caseid).strip()

    candidates = [
        f"{caseid}_{key_time}",   # key_format=case_time
        key_time,                 # key_format=time
    ]

    by_key = elevation_cache.get("by_key", {}) if isinstance(elevation_cache, dict) else {}
    rec = None
    for key in candidates:
        if key in by_key:
            rec = by_key[key]
            break

    # Fallback to by_mode maps if available.
    if rec is None and isinstance(elevation_cache, dict):
        by_mode = elevation_cache.get("by_mode", {})
        mode_map = by_mode.get("T" if str(mode).upper().startswith("T") else "N", {})
        for key in candidates:
            if key in mode_map:
                rec = mode_map[key]
                break

    # Last-resort fallback through records. This handles odd cache key formats.
    if rec is None and isinstance(elevation_cache, dict):
        for r in elevation_cache.get("records", []) or []:
            try:
                same_case = str(r.get("case_id", "")).strip() == caseid
                same_time = pd.Timestamp(r.get("atime")).strftime("%Y%m%d%H%M") == key_time
            except Exception:
                same_case = False
                same_time = False
            if same_case and same_time:
                rec = r
                break

    if rec is None:
        return None

    z_ref = float(rec.get("z_ref_m", np.nan))
    bbox = rec.get("bbox", None)
    if isinstance(bbox, str):
        bbox_string = bbox
    elif bbox is None:
        bbox_string = ""
    else:
        try:
            bbox_string = ",".join(f"{float(v):.6f}" for v in bbox)
        except Exception:
            bbox_string = str(bbox)

    dem_files = rec.get("dem_files", "")
    if isinstance(dem_files, (list, tuple)):
        dem_files_string = ";".join(str(v) for v in dem_files)
        dem_files_list = [str(v) for v in dem_files]
    else:
        dem_files_string = str(dem_files or "")
        dem_files_list = [v for v in dem_files_string.split(";") if v]

    n_dem_pixels = rec.get("n_dem_pixels", rec.get("n_pixels", 0))
    try:
        n_dem_pixels = int(n_dem_pixels)
    except Exception:
        n_dem_pixels = 0

    return {
        "z_ref_m": z_ref,
        "bbox": bbox,
        "bbox_string": bbox_string,
        "dem_files": dem_files_list,
        "dem_files_string": dem_files_string,
        "n_dem_pixels": n_dem_pixels,
        "tile_ids": [],
        "method": "analysis_elevation_cache",
        "status": rec.get("status", ""),
    }


def _analysis_reference_elevation(elev_file_path, dir_mesonet, atime, time_to_space_conversion=5, elevation_cache=None, caseid=None, mode=None):
    """DEM mean elevation over the mobile-mesonet footprint for one analysis time.

    The mobile mesonet footprint is the min/max lon/lat of valid mobile mesonet
    positions in the current time-to-space analysis window. The reference
    elevation is then computed from USGS 3DEP/TNM GeoTIFF pixels inside that
    bbox. Missing tiles are downloaded automatically. No platform-elevation
    fallback is used.
    """
    cached = _cache_lookup_reference_elevation(elevation_cache, caseid, mode, atime)
    if cached is not None:
        return cached

    mmfs = get_mm_files_cached(dir_mesonet, atime)
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    lats_all = []
    lons_all = []

    for fi in mmfs:
        try:
            with xr.open_dataset(fi, decode_times=False) as m:
                if "time" not in m or "lat" not in m or "lon" not in m:
                    continue
                tm_num = m.time.values
                mask = (tm_num >= date2num(atime - time_window, "seconds since 1970-01-01")) & (
                    tm_num <= date2num(atime + time_window, "seconds since 1970-01-01")
                )
                if not np.any(mask):
                    continue
                lat = np.asarray(m.lat.sel(time=mask).values, dtype=float)
                lon = np.asarray(m.lon.sel(time=mask).values, dtype=float)
                lon = np.where(lon > 0, -lon, lon)
                keep = np.isfinite(lat) & np.isfinite(lon) & (lon < 0)
                if np.any(keep):
                    lats_all.append(lat[keep])
                    lons_all.append(lon[keep])
        except Exception:
            continue

    if not lats_all:
        return {
            "z_ref_m": np.nan,
            "bbox": (np.nan, np.nan, np.nan, np.nan),
            "bbox_string": "",
            "dem_files": [],
            "dem_files_string": "",
            "n_dem_pixels": 0,
            "tile_ids": [],
        }

    lats = np.concatenate(lats_all)
    lons = np.concatenate(lons_all)
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))

    # Avoid zero-area raster windows when all samples share an identical lat/lon.
    pad = 1.0e-4
    if np.isclose(lon_min, lon_max):
        lon_min -= pad
        lon_max += pad
    if np.isclose(lat_min, lat_max):
        lat_min -= pad
        lat_max += pad

    info = _dem_mean_in_bbox(Path(elev_file_path), lon_min, lat_min, lon_max, lat_max)
    if not np.isfinite(info["z_ref_m"]):
        print(
            f"[WARN] no finite DEM pixels for analysis time {atime} "
            f"bbox={info['bbox_string']} tiles={info.get('tile_ids', [])}",
            flush=True,
        )
    return info

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

    data = np.full((temp.size, 8), np.nan, dtype=np.float32)
    data[:, 0] = temp
    data[:, 1] = pres
    data[:, 2] = rh
    data[:, 3] = wspd
    data[:, 4] = wdir
    data[:, 5] = u
    data[:, 6] = v
    data[:, 7] = elev
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
        "elev", "elev_mean", "elev_std", "elev_var",
    ]
    ncdata = [
        data[:, 0], mean_data[:, 0], std_data[:, 0], var_data[:, 0],
        data[:, 1], mean_data[:, 1], std_data[:, 1], var_data[:, 1],
        data[:, 2], mean_data[:, 2], std_data[:, 2], var_data[:, 2],
        data[:, 3], mean_data[:, 3], std_data[:, 3], var_data[:, 3],
        data[:, 4], mean_data[:, 4], std_data[:, 4], var_data[:, 4],
        data[:, 5], mean_data[:, 5], std_data[:, 5], var_data[:, 5],
        data[:, 6], mean_data[:, 6], std_data[:, 6], var_data[:, 6],
        elev, mean_data[:, 7], std_data[:, 7], var_data[:, 7],
    ]

    ds = xr.Dataset(coords={"time": ("time", time_num)})
    ds["lat"] = ("time", lat)
    ds["lon"] = ("time", lon)
    for name, values in zip(var_names, ncdata):
        ds[name] = ("time", np.asarray(values, dtype=np.float32))

    if source_ds is not None and "time" in source_ds:
        source_time = np.asarray(source_ds["time"].values)
        rebuilt_time = np.asarray(ds["time"].values)
        source_time_int = np.rint(source_time).astype(np.int64)
        rebuilt_time_int = np.rint(rebuilt_time).astype(np.int64)
        if source_time.shape != rebuilt_time.shape:
            rebuilt_df = pd.DataFrame({"time_int": rebuilt_time_int})
            rebuilt_df["occ"] = rebuilt_df.groupby("time_int").cumcount()
            rebuilt_df["idx"] = np.arange(rebuilt_df.shape[0], dtype=np.int64)

            source_df = pd.DataFrame({"time_int": source_time_int})
            source_df["occ"] = source_df.groupby("time_int").cumcount()

            merged = source_df.merge(rebuilt_df, on=["time_int", "occ"], how="left", sort=False)
            if merged["idx"].notna().all():
                ds = ds.isel(time=merged["idx"].astype(np.int64).to_numpy())

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
    ds.attrs["vehicle_dv_thresh_mps"] = float(vehicle_dv_thresh_mps)
    ds.attrs["vehicle_dv_window_sec"] = float(vehicle_dv_window_sec)
    ds.attrs["vehicle_dv_use_vector"] = "true" if vehicle_dv_use_vector else "false"
    ds.attrs["moving_statistics_window_seconds"] = int(window_seconds)
    ds.attrs["pressure_reduction_elevation_field"] = "elev_mean"
    if source_ds is not None:
        source_ds.close()
    return ds


def load_storm_track_window(case_idx, tracking_dir: Path, mode: str, start_time: datetime, end_time: datetime, dt_tracking: float = 1 / 60):
    """Load/interpolate a storm track by literal CaseID for the rebias window."""
    mtimes, srlat, srlon = _load_interp_storm_track(str(case_idx).strip(), str(tracking_dir), mode, float(dt_tracking))
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

            temp = m.temp_mean[itm].values
            pres = m.pres_mean[itm].values
            rh = m.rh_mean[itm].values
            wspd = m.wspd_mean[itm].values
            wdir = m.wdir_mean[itm].values
            elev = (m.elev_mean if "elev_mean" in m else m.elev)[itm].values
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
                & (wspd <= 100)
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

        case_label = str(case_ids[x]).strip()
        track_times, track_lats, track_lons = load_storm_track_window(
            case_label,
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
        comm.Barrier()

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

    nc_paths = sorted(prepared_dir.glob("*.nc"))
    for nc_path in nc_paths[rank::size]:
        with xr.open_dataset(nc_path) as ds:
            trimmed = _trim_dataset_to_analysis_windows(ds.load(), get_file_date_mm(str(nc_path)), transects_csv, time_to_space_minutes)
        if trimmed is None:
            nc_path.unlink(missing_ok=True)
            continue
        trimmed = _apply_variable_metadata(trimmed)
        _validate_dataset_units(trimmed, nc_path.name)
        _write_atomic_netcdf(trimmed, nc_path, overwrite=True)
        trimmed.close()
    if comm is not None:
        comm.Barrier()

    # Reduced pressure is computed per analysis time in get_mesonet_data().



def _parse_args():
    p = argparse.ArgumentParser(description='Build mobile mesonet pickle files from an already processed mobile_mesonet_processed/ NetCDF archive.')
    p.add_argument('--workdir', default='.', help='Workspace containing transects, Z files, soundings, radar cache, and storm tracks.')
    p.add_argument('--mesonet-dir', default='mobile_mesonet_processed', help='Directory containing cleaned/rebiased mobile mesonet NetCDF files.')
    p.add_argument('--transects', default='transect_times.csv', help='Transect timing table (.csv, .xlsx, or .xls).')
    p.add_argument('--tracking-dir', default='storm_tracks', help='Storm track directory.')
    p.add_argument('--radar-dir', default='nexrad', help='Radar cache directory.')
    p.add_argument('--elev-dir', default='elevation', help='Directory of USGS elevation GeoTIFF tiles.')
    p.add_argument('--elevation-cache', default=None, help='Optional analysis_elevation_cache_tsc*.pkl. If provided, z_ref_m is read from this cache instead of reopening DEM GeoTIFFs during synthesis.')
    p.add_argument('--sticknet-dir', default='sticknets', help='StickNet directory. Currently not used in final output path.')
    p.add_argument('--soundings', default='soundings_RAP_use_this.pkl', help='Observed+RAP sounding pickle consumed by reconciliation.')
    p.add_argument('--modes', nargs='+', default=['T', 'N'], choices=['T', 'N'], help='Modes to build.')
    p.add_argument('--years', type=int, nargs='*', default=None,
                   help='Optional Tstart years to include, e.g., --years 1995 1999 2024 2025.')
    p.add_argument('--time-to-space', type=int, default=5, choices=[5, 10], help='Time-to-space conversion in minutes.')
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
    p.add_argument('--output', default=None, help='Final merged pickle path.')
    p.add_argument('--keep-rank-files', action='store_true', help='Keep per-rank intermediate pickles.')
    p.add_argument('--analysis-keys', nargs='*', default=None, help='Optional list of YYYYMMDDHHMM analysis keys to run as a subset smoke test.')
    p.add_argument('--skip-radar', action='store_true',
                   help='Skip radar download/read and set reflectivity to NaN/ql to 0. Useful for thermodynamic and pressure-correction tests.')
    p.add_argument('--max-candidate-radars', type=int, default=8,
                   help='Maximum radar IDs to try per analysis window. Spreadsheet radars are tried first, then nearest WSR-88Ds are appended until this count is reached.')
    return p.parse_args()


def _resolve_path(workdir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(workdir, path_value)


def _analysis_times_for_tsc(time_to_space_conversion):
    if time_to_space_conversion == 5:
        return np.arange(-20, 20 + 1, time_to_space_conversion)
    if time_to_space_conversion == 10:
        return np.array([-15, -5, 5, 15])
    raise ValueError(f'Unsupported time_to_space_conversion: {time_to_space_conversion}')


def _build_shear_layers():
    levels = np.arange(0, 6500, 500)
    shear_layers = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            shear_layers.append((levels[i], levels[j]))
    shear_layermin = [val[0] for val in shear_layers]
    shear_layermax = [val[1] for val in shear_layers]
    return shear_layermin, shear_layermax


def _parse_transect_datetime(value):
    """Parse common transect timestamp formats; return None for missing values."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "nat", "none"}:
        return None
    for fmt in ("%Y%m%d-%H%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    except Exception:
        return None


def _analysis_window(atime, time_to_space_conversion):
    half = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    return atime - half, atime + half


def _window_overlaps(a0, a1, b0, b1):
    return (a0 <= b1) and (a1 >= b0)


def _safe_candidate_radars(row):
    for col in ("Merged Radars", "Radar", "Nearest WSR-88D"):
        if col in row.index:
            val = row[col]
            if not pd.isna(val):
                return val
    return None


def _build_tasks(tr, t0, amode, analysis_times, time_to_space_conversion):
    """Build analysis tasks, retaining only windows overlapping Observation Start/End.

    This mirrors the sounding-pipeline behavior: the nominal grid is still
    Tstart + analysis offset, but we skip windows where no mobile mesonet
    observations were collected.
    """
    tasks = []
    skipped_no_obs_overlap = 0
    has_obs_bounds = {"Observation Start", "Observation End"}.issubset(set(tr.columns))

    for analysis_time in analysis_times:
        for row_idx in range(len(t0)):
            row = tr.iloc[row_idx]
            atime = t0[row_idx] + timedelta(minutes=int(analysis_time))
            win_start, win_end = _analysis_window(atime, time_to_space_conversion)

            obs_start = _parse_transect_datetime(row["Observation Start"]) if has_obs_bounds else None
            obs_end = _parse_transect_datetime(row["Observation End"]) if has_obs_bounds else None
            if obs_start is not None and obs_end is not None:
                if obs_end < obs_start:
                    raise ValueError(
                        f"Observation End precedes Observation Start for ID={row.get('ID', row_idx)}: "
                        f"{obs_start} to {obs_end}"
                    )
                if not _window_overlaps(win_start, win_end, obs_start, obs_end):
                    skipped_no_obs_overlap += 1
                    continue

            caseid = str(row['CaseID']).strip()
            tstart_dt = t0[row_idx]
            tasks.append(
                {
                    'key': atime.strftime('%Y%m%d%H%M'),
                    'atime': atime,
                    'caseid': caseid,
                    'offset_min': int(analysis_time),
                    'transect_id': str(row.get('ID', row_idx)),
                    'case_label': str(row['CaseID']),
                    'candidate_radars': _safe_candidate_radars(row),
                    'window_start': win_start,
                    'window_end': win_end,
                    'tstart': tstart_dt,
                    'year': int(tstart_dt.year),
                }
            )
    return tasks, skipped_no_obs_overlap


def _filter_tasks(tasks, analysis_keys: Optional[set[str]] = None, years: Optional[set[int]] = None):
    """Filter analysis tasks by explicit YYYYMMDDHHMM keys and/or Tstart years."""
    out = tasks
    if years:
        out = [task for task in out if int(task.get('year', task['atime'].year)) in years]
    if analysis_keys:
        out = [task for task in out if task['key'] in analysis_keys]
    return out


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = _parse_args()
    workdir = Path(os.path.abspath(args.workdir))
    dir_mesonet = _resolve(workdir, args.mesonet_dir or os.environ.get('MM_DIR', 'mobile_mesonet_processed'))
    transects_path = _resolve(workdir, args.transects)
    tracking_dir = _resolve(workdir, args.tracking_dir)
    radar_file_dir = _resolve(workdir, args.radar_dir)
    elev_file_path = _resolve(workdir, args.elev_dir)
    sticknet_dir = _resolve(workdir, args.sticknet_dir)
    soundings_path = _resolve(workdir, args.soundings)
    output_path = _resolve(workdir, args.output or f'mobile_mesonet_data_tsc_{args.time_to_space}_min.pkl')
    analysis_keys = set(args.analysis_keys or [])
    selected_years = set(int(y) for y in args.years) if args.years else None

    required_paths = [
        (transects_path, 'transect_times table'),
        (tracking_dir, 'storm_tracks directory'),
        (radar_file_dir, 'nexrad directory'),
        (soundings_path, 'soundings_RAP_use_this.pkl'),
        (dir_mesonet, 'processed mesonet NetCDF directory'),
    ]
    for path_value, label in required_paths:
        if not os.path.exists(path_value):
            raise FileNotFoundError(f'Missing {label}: {path_value}')

    os.chdir(str(workdir))

    dir_mesonet = str(dir_mesonet)
    tracking_dir = str(tracking_dir)
    radar_file_dir = str(radar_file_dir)
    elev_file_path = str(elev_file_path)
    sticknet_dir = str(sticknet_dir)
    os.environ['MM_DIR'] = dir_mesonet

    analysis_times = _analysis_times_for_tsc(args.time_to_space)
    shear_layermin, shear_layermax = _build_shear_layers()

    with open(soundings_path, 'rb') as f:
        sound, sound_lat, sound_lon = pickle.load(f)

    elevation_cache = None
    if args.elevation_cache:
        elevation_cache_path = _resolve(workdir, args.elevation_cache)
        elevation_cache = _load_analysis_elevation_cache(elevation_cache_path)
        if rank == 0:
            ncache = len(elevation_cache.get('by_key', {})) if isinstance(elevation_cache, dict) else 0
            print(f'[INFO] using analysis elevation cache: {elevation_cache_path} entries={ncache}', flush=True)

    merged_modes = {}
    for mode in args.modes:
        t0, amode, tr = get_t0s(transects_path, mode)
        tasks, skipped_no_obs_overlap = _build_tasks(tr, t0, amode, analysis_times, args.time_to_space)
        tasks = _filter_tasks(tasks, analysis_keys=analysis_keys, years=selected_years)
        my_idx = np.arange(len(tasks))[rank::size]
        local_cases = {}
        if rank == 0:
            years_msg = f' years={sorted(selected_years)}' if selected_years else ''
            print(f'[INFO] mode={mode} total_tasks={len(tasks)} skipped_no_obs_overlap={skipped_no_obs_overlap}{years_msg} ranks={size}')
        for ix in tqdm(my_idx, disable=not show_pbar):
            task = tasks[ix]
            dat = parse_storm_tracks(
                task['caseid'],
                tracking_dir,
                amode,
                task['atime'],
                dt_tracking=1/60,
                time_to_space_conversion=args.time_to_space,
            )
            if dat is None:
                if rank == 0:
                    print(
                        f"[WARN] no storm-track coverage for mode={mode} ID={task.get('transect_id')} "
                        f"CaseID={task.get('case_label')} offset={task.get('offset_min')} "
                        f"window={task.get('window_start')}..{task.get('window_end')}",
                        flush=True,
                    )
                continue
            meso_time, meso_lat, meso_lon, u_storm, v_storm = dat
            case_data = process_case(
                dir_mesonet,
                sticknet_dir,
                elev_file_path,
                sound,
                shear_layermin,
                shear_layermax,
                task['atime'],
                u_storm,
                v_storm,
                radar_file_dir,
                amode,
                task['caseid'],
                meso_time,
                meso_lat,
                meso_lon,
                candidate_radars=task['candidate_radars'],
                time_to_space_conversion=args.time_to_space,
                filter_accelerating_winds=args.filter_accelerating_winds,
                vehicle_dv_thresh_mps=args.vehicle_dv_thresh_mps,
                vehicle_dv_window_sec=args.vehicle_dv_window_sec,
                vehicle_dv_use_vector=args.vehicle_dv_use_vector,
                skip_radar=args.skip_radar,
                elevation_cache=elevation_cache,
                max_candidate_radars=args.max_candidate_radars,
            )
            local_cases[task['key']] = case_data

        rank_file = workdir / f'mobile_mesonet_rank{rank}_tsc{args.time_to_space}_{mode}.pkl'
        with open(rank_file, 'wb') as f:
            pickle.dump(local_cases, f, protocol=pickle.HIGHEST_PROTOCOL)
        comm.Barrier()

    if rank != 0:
        return

    for mode in args.modes:
        merged_modes[mode] = {}
        for r in range(size):
            rank_file = workdir / f'mobile_mesonet_rank{r}_tsc{args.time_to_space}_{mode}.pkl'
            if not os.path.exists(rank_file):
                continue
            with open(rank_file, 'rb') as f:
                part = pickle.load(f)
            merged_modes[mode].update(part)
            if not args.keep_rank_files:
                os.remove(rank_file)

    with open(output_path, 'wb') as f:
        pickle.dump(merged_modes, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'[INFO] Wrote {output_path}')
    for mode in args.modes:
        print(f'[INFO] {mode} cases: {len(merged_modes[mode])}')


if __name__ == '__main__':
    main()
