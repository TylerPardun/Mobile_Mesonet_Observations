#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from glob import glob
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
np.round_ = np.round
import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import transform as rio_transform
import requests
from herbie import Herbie
from netCDF4 import Dataset, date2num, num2date
import metpy.calc as mpcalc
import metpy.constants as const
from metpy.calc import mixing_ratio_from_relative_humidity, relative_humidity_from_mixing_ratio
from metpy.units import units
from geopy.distance import distance
from geopy import Point
from sharppy.sharptab import interp, params, profile, winds
import sharppy.sharptab.profile as profile
from tqdm import tqdm

from utils import get_dx_dy, haversine, interpolate_var, rotate_grid, get_mm_files

warnings.filterwarnings("ignore")
logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("eccodes").setLevel(logging.ERROR)

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

        @staticmethod
        def scatter(data, root=0):
            return data[0] if isinstance(data, list) else data

        @staticmethod
        def gather(data, root=0):
            return [data]

    class _SerialMPI:
        COMM_WORLD = _SerialComm()

    MPI = _SerialMPI()


KTS_PER_MPS = 1.94384
KNOTS_TO_MS = 0.5144444444444445
DEFAULT_PRESSURE_LEVELS = np.array([1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0])

# 2025 NSSL Lidar sonde .txt files do not carry launch lat/lon in the
# %TITLE%/%RAW% format. Use the National Weather Center/NSSL Norman location
# by default, but allow clean overrides without changing the script.
DEFAULT_TXT_SOUNDING_LAT = float(os.environ.get('NSSL_LIDAR_LAT', '35.1813'))
DEFAULT_TXT_SOUNDING_LON = float(os.environ.get('NSSL_LIDAR_LON', '-97.4395'))


def atomic_pickle_dump(obj, output_path):
    output_path = os.path.abspath(output_path)
    outdir = os.path.dirname(output_path) or '.'
    os.makedirs(outdir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix='.pkl', dir=outdir)
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def parse_args():
    p = argparse.ArgumentParser(description='Build sounding products from raw soundings through RAP-finished layered pickles.')
    p.add_argument('--stage', choices=['raw', 'finish', 'layers', 'all'], default='all', help='Pipeline stage to run.')
    p.add_argument('--soundings-dir', default='soundings', help='Directory of raw sounding files.')
    p.add_argument('--raw-output', default='soundings_raw.pkl', help='Output path for raw observed sounding pickle.')
    p.add_argument('--observed-output', default='soundings.pkl', help='Output path for observed sounding pickle consumed by RAP finishing.')
    p.add_argument('--finished-output', default='soundings_RAP_use_this.pkl', help='Output path for RAP-finished sounding pickle.')
    p.add_argument('--gui-output', default='soundings_RAP.pkl',
                   help='Output path for GUI-ready neutral RAP/RUC sounding pickle with raw earth-relative winds.')
    p.add_argument('--bunkers-output', default='soundings_RAP_bunkers.pkl', help='Output path for Bunkers diagnostics pickle.')
    p.add_argument('--layered-output', default='soundings_RAP_use_this_with_layers.pkl', help='Output path for layered sounding diagnostics pickle.')
    p.add_argument('--dir-transects', default='.', help='Directory containing transect_times.csv or transect_times.xlsx, or a direct path to the transect_times CSV/XLSX file.')
    p.add_argument('--mesonet-dir', default='mobile_mesonet_processed', help='Directory containing processed mobile mesonet NetCDF files used to derive the mean elevation Z at each analysis time.')
    p.add_argument('--elevation-max-distance', type=float, default=20.0, help='Deprecated compatibility option; Z is now derived from the DEM over the mobile-mesonet footprint, not sampled platform distance.')
    p.add_argument('--tracking-dir', default='storm_tracks', help='Storm tracks directory.')
    p.add_argument('--elev-dir', default='elevation', help='Directory of USGS 3DEP/TNM GeoTIFF elevation tiles used to derive DEM regional mean elevation Z.')
    p.add_argument('--rap-dir', default='RAP', help='RAP cache/download directory.')
    p.add_argument('--time-to-space', type=int, default=5, help='Time-to-space conversion in minutes.')
    p.add_argument('--modes', nargs='+', default=['T', 'N'], help='Modes to process in the RAP finishing stage.')
    p.add_argument('--compute-eil', action='store_true', help='Compute effective inflow layer fields in the observed sounding pickle. This is slow and not required by the RAP finishing stage.')
    p.add_argument('--cwd', default='.', help='Working directory for finish/layer subprocess stages.')
    return p.parse_args()


def string_to_datetime(timestamp_str):
    timestamp_clean = str(timestamp_str).strip('[]')
    if '.' in timestamp_clean:
        base, frac = timestamp_clean.split('.')
        timestamp_clean = f'{base}.{frac[:6]}'
    return datetime.fromisoformat(timestamp_clean)


def sanitize_arrays(*arrays):
    return [np.asarray(arr, dtype=float).copy() for arr in arrays]


def _dedupe_sorted(values, *companions):
    if values.size == 0:
        return (values, *companions)
    _, keep = np.unique(values, return_index=True)
    keep = np.sort(keep)
    out = [values[keep]]
    for arr in companions:
        out.append(arr[keep])
    return tuple(out)


def apply_common_qc(height, pres, temp, dew, wind_a, wind_b):
    height, pres, temp, dew, wind_a, wind_b = sanitize_arrays(height, pres, temp, dew, wind_a, wind_b)
    wind_a[(wind_a > 150) | (wind_a < -150)] = np.nan
    wind_b[(wind_b > 150) | (wind_b < -150)] = np.nan
    temp[(temp > 55) | (temp < -100)] = np.nan
    dew[(dew > 55) | (dew < -120)] = np.nan
    pres[(pres > 1100) | (pres < 5)] = np.nan
    height[(height > 50000) | (height < -500)] = np.nan
    dew = np.minimum(dew, temp)

    keep = np.where(
        np.isfinite(height) & np.isfinite(pres) & np.isfinite(temp) &
        np.isfinite(dew) & np.isfinite(wind_a) & np.isfinite(wind_b)
    )[0]
    if keep.size < 10:
        return None

    height = height[keep]
    pres = pres[keep]
    temp = temp[keep]
    dew = dew[keep]
    wind_a = wind_a[keep]
    wind_b = wind_b[keep]

    order = np.argsort(height)
    height = height[order]
    pres = pres[order]
    temp = temp[order]
    dew = dew[order]
    wind_a = wind_a[order]
    wind_b = wind_b[order]

    monotonic = np.ones(height.size, dtype=bool)
    monotonic[1:] &= np.diff(height) > 0
    monotonic[1:] &= np.diff(pres) < 0
    if monotonic.sum() < 10:
        return None

    height = height[monotonic]
    pres = pres[monotonic]
    temp = temp[monotonic]
    dew = dew[monotonic]
    wind_a = wind_a[monotonic]
    wind_b = wind_b[monotonic]

    height, pres, temp, dew, wind_a, wind_b = _dedupe_sorted(height, pres, temp, dew, wind_a, wind_b)
    pres, height, temp, dew, wind_a, wind_b = _dedupe_sorted(-pres, height, temp, dew, wind_a, wind_b)
    pres = -pres

    if height.size < 10:
        return None

    height = height - height[0]
    return height, pres, temp, dew, wind_a, wind_b


def build_profile_from_wdir(height, pres, temp, dew, wdir, wspd_kts):
    # QC the wind direction/speed pair before converting to components.
    wdir = np.asarray(wdir, dtype=float)
    wspd_kts = np.asarray(wspd_kts, dtype=float)
    wdir[(wdir < 0) | (wdir > 360)] = np.nan
    wspd_kts[(wspd_kts < 0) | (wspd_kts > 300)] = np.nan
    u_kts, v_kts = mpcalc.wind_components(wspd_kts * units.knots, wdir * units.degrees)
    qc = apply_common_qc(height, pres, temp, dew, u_kts.m, v_kts.m)
    if qc is None:
        return None
    height, pres, temp, dew, u_kts, v_kts = qc
    return profile.create_profile(
        profile='default', pres=pres, hght=height, tmpc=temp, dwpc=dew,
        u=u_kts, v=v_kts, missing=-9999, strictQC=False
    )


def build_profile_from_uv(height, pres, temp, dew, u_kts, v_kts):
    qc = apply_common_qc(height, pres, temp, dew, u_kts, v_kts)
    if qc is None:
        return None
    height, pres, temp, dew, u_kts, v_kts = qc
    return profile.create_profile(
        profile='default', pres=pres, hght=height, tmpc=temp, dwpc=dew,
        u=u_kts, v=v_kts, missing=-9999, strictQC=False
    )


def compute_eil_fields(profs):
    n = len(profs)
    eil_thickness = np.full(n, np.nan)
    eil_bottom = np.full(n, np.nan)
    eil_top = np.full(n, np.nan)
    for i, prof in enumerate(profs):
        try:
            eil_pb, eil_pt = params.effective_inflow_layer(prof)
            if eil_pb:
                heights = np.array(prof.hght)[np.where((np.array(prof.pres) >= eil_pt) & (np.array(prof.pres) <= eil_pb))[0]]
                eil_bottom[i] = np.nanmin(heights)
                eil_top[i] = np.nanmax(heights)
                eil_thickness[i] = eil_top[i] - eil_bottom[i]
        except Exception:
            pass
    return eil_thickness, eil_top, eil_bottom


def parse_csv_soundings(soundings_dir, camp_dict):
    out = []
    for sfile in tqdm(sorted(glob(os.path.join(soundings_dir, '*.csv'))), total=len(sorted(glob(os.path.join(soundings_dir, '*.csv')))), desc='csv'):
        try:
            s = pd.read_csv(sfile, skiprows=[1])
        except Exception:
            continue
        if s.empty:
            continue
        try:
            s.columns = s.iloc[0]
            s = s.iloc[1:]
        except Exception:
            continue
        required = {
            'Date-Time',
            'Filtered Altitude (m)',
            'Filtered Pressure (mb)',
            'Filtered Temperature (K)',
            'Filtered Dewpoint (K)',
            'Filtered Wind Dir',
            'Filtered Wind Spd (m/s)',
            'Filtered Latitude',
            'Filtered Longitude',
        }
        if not required.issubset(set(s.columns)):
            continue
        date = string_to_datetime(s['Date-Time'].values[0])
        latitude = float(s['Filtered Latitude'].values[0])
        longitude = float(s['Filtered Longitude'].values[0])
        camp = camp_dict.get(date.year, 'Unknown')
        prof = build_profile_from_wdir(
            s['Filtered Altitude (m)'].values.astype(float),
            s['Filtered Pressure (mb)'].values.astype(float),
            s['Filtered Temperature (K)'].values.astype(float) - 273.15,
            s['Filtered Dewpoint (K)'].values.astype(float) - 273.15,
            s['Filtered Wind Dir'].values.astype(float),
            s['Filtered Wind Spd (m/s)'].values.astype(float) * KTS_PER_MPS,
        )
        if prof is not None:
            out.append((date, latitude, longitude, prof, camp))
    return out


def parse_nc_soundings(soundings_dir, camp_dict):
    out = []
    files = sorted(glob(os.path.join(soundings_dir, '*.nc')))
    for sfile in tqdm(files, total=len(files), desc='nc'):
        s = Dataset(sfile)
        date = num2date(s['base_time'][0] + s['time_offset'][0], 'seconds since 1970-01-01')
        date = datetime(date.year, date.month, date.day, date.hour, date.minute, date.second)
        latitude = float(s['lat'][0])
        longitude = float(s['lon'][0])
        camp = camp_dict.get(date.year, 'Unknown')
        prof = build_profile_from_uv(
            s['alt'][:],
            s['pres'][:],
            s['tdry'][:],
            s['dewpoint'][:],
            s['u'][:] * KTS_PER_MPS,
            s['v'][:] * KTS_PER_MPS,
        )
        s.close()
        if prof is not None:
            out.append((date, latitude, longitude, prof, camp))
    return out


def parse_cls_soundings(soundings_dir, camp_dict):
    out = []
    files = sorted(glob(os.path.join(soundings_dir, '*.cls')))
    for sfile in tqdm(files, total=len(files), desc='cls'):
        with open(sfile) as f:
            lines = f.readlines()
        starts = [iline for iline, line in enumerate(lines) if 'Nominal' in line]
        for sx in range(len(starts)):
            subfile = lines[starts[sx]:] if sx == len(starts) - 1 else lines[starts[sx]:starts[sx + 1]]
            date_str = subfile[0].split('Nominal Release Time (y,m,d,h,m,s):')[1].split('\n')[0]
            date = datetime.strptime(date_str, '%Y, %m, %d, %H:%M:%S')
            camp = camp_dict.get(date.year, 'Unknown')
            latitude, longitude = float(subfile[4].split()[11]), float(subfile[4].split()[10])
            pres, temp, dew, u, v, height = [], [], [], [], [], []
            for sln in subfile[4:]:
                pts = sln.split()
                if len(pts) == 21:
                    try:
                        pres.append(float(pts[1]))
                        temp.append(float(pts[2]))
                        dew.append(float(pts[3]))
                        u.append(float(pts[5]) * KTS_PER_MPS)
                        v.append(float(pts[6]) * KTS_PER_MPS)
                        height.append(float(pts[14]))
                    except ValueError:
                        pass
            prof = build_profile_from_uv(height, pres, temp, dew, u, v)
            if prof is not None:
                out.append((date, latitude, longitude, prof, camp))
    return out



def parse_txt_soundings(soundings_dir, camp_dict):
    """
    Parse 2025-style NSSL Lidar sonde Skew-T text files.

    Expected format:
        %TITLE%
        NSSL_LIDAR YYMMDD/HHMM
        %RAW%
        pressure_mb,height_m,temp_c,dewpoint_c,wind_dir_deg,wind_speed_kt

    The sample 2025 files do not include launch latitude/longitude, so this
    parser assigns DEFAULT_TXT_SOUNDING_LAT/LON. Override those by setting the
    environment variables NSSL_LIDAR_LAT and NSSL_LIDAR_LON before running.
    """

    out = []
    files = sorted(glob(os.path.join(soundings_dir, '*.txt')))

    for sfile in tqdm(files, total=len(files), desc='txt'):
        try:
            with open(sfile, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f'[txt parser] could not read {sfile}: {e}')
            continue

        if not lines:
            continue

        # Locate title/date metadata.
        title_date = None
        for iline, line in enumerate(lines):
            if line.upper() == '%TITLE%' and iline + 1 < len(lines):
                title_line = lines[iline + 1].strip()
                m = re.search(r'(\d{6})/(\d{4})', title_line)
                if m:
                    yy = int(m.group(1)[:2])
                    century = 2000 if yy < 70 else 1900
                    yymmdd_hhmm = f'{century + yy:04d}{m.group(1)[2:]}{m.group(2)}'
                    title_date = datetime.strptime(yymmdd_hhmm, '%Y%m%d%H%M')
                    break

        # Fallback to filename timestamp, e.g. upperair.NSSL_Lidar_sonde.202504262032.skewT.txt
        if title_date is None:
            m = re.search(r'(20\d{10}|19\d{10})', os.path.basename(sfile))
            if m:
                title_date = datetime.strptime(m.group(1), '%Y%m%d%H%M')
            else:
                print(f'[txt parser] no valid date found in {sfile}')
                continue

        try:
            raw_start = next(i for i, line in enumerate(lines) if line.upper() == '%RAW%') + 1
        except StopIteration:
            print(f'[txt parser] no %RAW% block found in {sfile}')
            continue

        rows = []
        for line in lines[raw_start:]:
            if not line or line.startswith('%'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                continue
            try:
                vals = [float(p) for p in parts[:6]]
            except ValueError:
                continue
            rows.append(vals)

        if len(rows) < 10:
            print(f'[txt parser] too few data rows in {sfile}: {len(rows)}')
            continue

        arr = np.asarray(rows, dtype=float)
        pres = arr[:, 0]
        height = arr[:, 1]
        temp = arr[:, 2]
        dew = arr[:, 3]
        wdir = arr[:, 4]
        wspd_kts = arr[:, 5]

        camp = camp_dict.get(title_date.year, 'Unknown')
        prof = build_profile_from_wdir(height, pres, temp, dew, wdir, wspd_kts)

        if prof is not None:
            out.append((title_date, DEFAULT_TXT_SOUNDING_LAT, DEFAULT_TXT_SOUNDING_LON, prof, camp))
        else:
            print(f'[txt parser] profile QC failed for {sfile}')

    return out

def build_observed_soundings(soundings_dir, raw_output, observed_output, compute_eil=False):
    camp_dict = {
        2009: 'V2S_09', 2010: 'V2S_10', 2017: 'RiVorS', 2019: 'TORUS_19',
        2022: 'TORUS_22', 2023: 'TORUS_LiTE', 2024: 'LIFT', 2025: 'LIFT'
    }
    rows = []
    rows.extend(parse_csv_soundings(soundings_dir, camp_dict))
    rows.extend(parse_nc_soundings(soundings_dir, camp_dict))
    rows.extend(parse_cls_soundings(soundings_dir, camp_dict))
    rows.extend(parse_txt_soundings(soundings_dir, camp_dict))
    rows.sort(key=lambda item: item[0])

    time = np.array([r[0] for r in rows], dtype=object)
    lat = np.array([r[1] for r in rows], dtype=float)
    lon = np.array([r[2] for r in rows], dtype=float)
    profs = np.array([r[3] for r in rows], dtype=object)
    campaign = np.array([r[4] for r in rows], dtype=object)

    sounding_raw = {'time': time, 'lat': lat, 'lon': lon, 'profs': profs, 'campagin': campaign}
    atomic_pickle_dump(sounding_raw, raw_output)

    if compute_eil:
        eil_thickness, eil_top, eil_bottom = compute_eil_fields(profs)
    else:
        nprof = len(profs)
        eil_thickness = np.full(nprof, np.nan)
        eil_top = np.full(nprof, np.nan)
        eil_bottom = np.full(nprof, np.nan)
    sounding_obs = {
        'time': time,
        'lat': lat,
        'lon': lon,
        'profs': profs,
        'campagin': campaign,
        'eil_thickness': eil_thickness,
        'eil_top': eil_top,
        'eil_bottom': eil_bottom,
    }
    atomic_pickle_dump(sounding_obs, observed_output)
    return len(rows)


def resolve_transect_times_path(path_like):
    """
    Resolve transect_times input from either:
      - a directory containing transect_times.csv or transect_times.xlsx
      - a direct path to transect_times.csv
      - a direct path to transect_times.xlsx

    Preference is CSV first if both files exist, because that preserves the
    previous behavior exactly.
    """
    pth = Path(path_like)

    if pth.is_file():
        return pth

    if not pth.exists():
        raise FileNotFoundError(
            f'Transect path does not exist: {pth}. Pass the directory containing '
            f'transect_times.csv/transect_times.xlsx with --dir-transects, or set --cwd '
            f'to the project root so the relative --dir-transects path resolves correctly.'
        )

    candidates = [
        pth / 'transect_times.csv',
        pth / 'transect_times.xlsx',
        pth / 'transect_times.xls',
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f'Could not find transect_times.csv, transect_times.xlsx, or transect_times.xls in {pth}. '
        f'Pass the correct directory with --dir-transects or pass the file directly.'
    )


def read_transect_times(path_like):
    """Read transect_times from CSV or Excel and normalize key columns."""
    fpath = resolve_transect_times_path(path_like)
    suffix = fpath.suffix.lower()

    if suffix == '.csv':
        tr = pd.read_csv(fpath)
    elif suffix in {'.xlsx', '.xls'}:
        # sheet_name=0 reads the first sheet. The uploaded workbook uses
        # a sheet named 'transect_times', but first-sheet reading is safer.
        tr = pd.read_excel(fpath, sheet_name=0)
    else:
        raise ValueError(f'Unsupported transect_times file type: {fpath}')

    # Strip whitespace that sometimes appears when spreadsheets are edited.
    tr.columns = [str(c).strip() for c in tr.columns]

    required = {'Mode', 'Tstart', 'CaseID'}
    missing = required.difference(tr.columns)
    if missing:
        raise ValueError(f'{fpath} is missing required columns: {sorted(missing)}')

    # Keep CaseID/Tstart as strings for the existing downstream parsing logic.
    tr['CaseID'] = tr['CaseID'].astype(str).str.strip()
    tr['Tstart'] = tr['Tstart'].astype(str).str.strip()

    return tr


def get_t0s(file_path, mode, long_track_length=None, wide_thresh=None):
    tr = read_transect_times(file_path)
    if mode == 'T':
        tr = tr.loc[tr.Mode == 'T']
        amode = 'T'
    elif mode == 'N':
        tr = tr.loc[tr.Mode == 'N0']
        amode = 'N'
    elif mode == 'W':
        tr = tr.loc[(tr.Mode == 'T') & (tr['Width [m]'] >= wide_thresh)]
        amode = 'T'
    elif mode == 'Sk':
        tr = tr.loc[(tr.Mode == 'T') & (tr['Width [m]'] < wide_thresh)]
        amode = 'T'
    elif mode == 'L':
        tr = tr.loc[(tr.Mode == 'T') & (tr['Length [km]'] >= long_track_length)]
        amode = 'T'
    elif mode == 'S':
        tr = tr.loc[(tr.Mode == 'T') & (tr['Length [km]'] < long_track_length)]
        amode = 'T'
    else:
        raise ValueError(f'Unsupported mode: {mode}')
    t0 = np.array([datetime.strptime(val, '%Y%m%d-%H%M') for val in tr['Tstart'].values], dtype=object)
    tr = tr.reset_index(drop=True)
    return t0, amode, tr


def _parse_transect_datetime(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {'nan', 'nat', 'none'}:
        return None
    for fmt in ('%Y%m%d-%H%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
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


def _analysis_window(atime, time_to_space):
    half = timedelta(seconds=int((time_to_space * 60) / 2))
    return atime - half, atime + half


def _window_overlaps(start_a, end_a, start_b, end_b):
    return (start_a <= end_b) and (end_a >= start_b)


def _build_analysis_tasks(tr, t0, amode, analysis_times, time_to_space):
    """Build per-analysis-time tasks, honoring Observation Start/End when present.

    The nominal analysis grid is still Tstart + offsets, but an analysis task is
    retained only when its finite-width time-to-space window overlaps the row's
    Observation Start/Observation End interval. This prevents the sounding
    finishing step from trying to derive storm motion, elevation, or RAP/RUC
    profiles for times where no mesonet observations were collected.
    """
    tasks = []
    has_obs_bounds = {'Observation Start', 'Observation End'}.issubset(set(tr.columns))
    skipped_no_obs_overlap = 0

    for row_idx in range(len(t0)):
        row = tr.iloc[row_idx]
        # Use the literal CaseID from transect_times.xlsx (e.g., T9, N4).
        # The storm-track files were renamed to CaseID_yyyymmdd_HHMM_*.csv,
        # so converting CaseID to the old zero-based numeric index silently
        # points at the wrong/missing track after case renumbering.
        case_id = str(row['CaseID']).strip()
        obs_start = _parse_transect_datetime(row['Observation Start']) if has_obs_bounds else None
        obs_end = _parse_transect_datetime(row['Observation End']) if has_obs_bounds else None
        if obs_start is not None and obs_end is not None and obs_end < obs_start:
            raise ValueError(
                f"Observation End precedes Observation Start for ID={row.get('ID', row_idx)}: "
                f"{obs_start} to {obs_end}"
            )

        for offset in analysis_times:
            atime = t0[row_idx] + timedelta(minutes=int(offset))
            win_start, win_end = _analysis_window(atime, time_to_space)
            if obs_start is not None and obs_end is not None:
                if not _window_overlaps(win_start, win_end, obs_start, obs_end):
                    skipped_no_obs_overlap += 1
                    continue
            tasks.append({
                'atime': atime,
                'key': atime.strftime('%Y%m%d%H%M%S'),
                'case_id': case_id,
                'row_idx': row_idx,
                'offset_min': int(offset),
                'transect_id': str(row.get('ID', row_idx)),
                'case_label': str(row['CaseID']),
                'obs_start': obs_start,
                'obs_end': obs_end,
                'window_start': win_start,
                'window_end': win_end,
            })
    return tasks, skipped_no_obs_overlap


def _storm_track_time_column(df):
    for col in ("time", "datetime", "Time", "DateTime"):
        if col in df.columns:
            return col
    return None


def _storm_track_latlon_columns(df):
    lat_col = None
    lon_col = None
    for col in ("lat", "latitude", "Lat", "Latitude"):
        if col in df.columns:
            lat_col = col
            break
    for col in ("lon", "longitude", "Lon", "Longitude"):
        if col in df.columns:
            lon_col = col
            break
    return lat_col, lon_col


def _track_subdir_for_case(case_id, mode):
    label = str(case_id).strip()
    if label.upper().startswith('T'):
        return 'tornadic'
    if label.upper().startswith('N'):
        return 'nontornadic'
    if mode in {'T', 'L', 'S', 'W', 'Sk'}:
        return 'tornadic'
    return 'nontornadic'


def _find_storm_track_file(case_id, tracking_dir, mode):
    """Find exactly one storm-track file for the literal CaseID.

    New storm-track names are CaseID_yyyymmdd_HHMM_autosave_edited.csv.
    This intentionally does not convert CaseID to an old numeric index.
    """
    label = str(case_id).strip()
    tracking_dir = Path(tracking_dir)
    subdir = _track_subdir_for_case(label, mode)
    search_dir = tracking_dir / subdir
    matches = sorted(search_dir.glob(f'{label}_*.csv'))

    # Compatibility fallback for legacy files with no .csv suffix glob match.
    if not matches:
        matches = sorted(search_dir.glob(f'{label}_*'))

    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple storm-track files matched CaseID={label} in {search_dir}: "
            + ", ".join(str(p) for p in matches)
        )
    return matches[0]


def parse_storm_tracks(case_id, tracking_dir, mode, atime, dt_tracking=1/60, time_to_space_conversion=10):
    track_file = _find_storm_track_file(case_id, tracking_dir, mode)
    if track_file is None:
        return None

    strk = pd.read_csv(track_file)
    if strk.empty:
        return None

    time_col = _storm_track_time_column(strk)
    lat_col, lon_col = _storm_track_latlon_columns(strk)
    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError(
            f"Storm-track file {track_file} is missing required columns. "
            f"Expected time/datetime plus lat/lon or latitude/longitude; got {list(strk.columns)}"
        )

    mtimes = pd.to_datetime(strk[time_col].values, errors='coerce', utc=False)
    good = ~pd.isna(mtimes)
    if good.sum() < 2:
        return None

    srlat = pd.to_numeric(strk.loc[good, lat_col], errors='coerce').values.astype(float)
    srlon = pd.to_numeric(strk.loc[good, lon_col], errors='coerce').values.astype(float)
    mtimes = mtimes[good]
    good_pos = np.isfinite(srlat) & np.isfinite(srlon)
    if good_pos.sum() < 2:
        return None

    mtimes = np.array(
        [datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in mtimes[good_pos]],
        dtype=object,
    )
    srlat = srlat[good_pos]
    srlon = srlon[good_pos]

    isort = np.argsort(mtimes)
    mtimes, srlat, srlon = mtimes[isort], srlat[isort], srlon[isort]

    # Drop duplicate timestamps before interpolation. Interpolating with duplicate
    # independent coordinates can fail or produce ambiguous storm positions.
    tnum_raw = date2num(mtimes, 'seconds since 1970-01-01')
    _, unique_idx = np.unique(tnum_raw, return_index=True)
    unique_idx = np.sort(unique_idx)
    mtimes = mtimes[unique_idx]
    srlat = srlat[unique_idx]
    srlon = srlon[unique_idx]
    if mtimes.size < 2:
        return None

    t0_num = date2num(mtimes[0], 'seconds since 1970-01-01')
    t1_num = date2num(mtimes[-1], 'seconds since 1970-01-01')
    if t1_num <= t0_num:
        return None

    time_grid = np.arange(t0_num, t1_num + dt_tracking * 60, dt_tracking * 60)
    if time_grid.size < 2:
        return None

    srlat = interpolate_var(srlat, date2num(mtimes, 'seconds since 1970-01-01'), time_grid)
    srlon = interpolate_var(srlon, date2num(mtimes, 'seconds since 1970-01-01'), time_grid)
    mtimes = num2date(time_grid, 'seconds since 1970-01-01')
    mtimes = np.array([datetime(val.year, val.month, val.day, val.hour, val.minute, val.second) for val in mtimes], dtype=object)

    win_start, win_end = _analysis_window(atime, time_to_space_conversion)
    imeso = np.where((mtimes >= win_start) & (mtimes <= win_end))[0]
    if imeso.size < 2:
        return None

    mtimes, srlat, srlon = mtimes[imeso], srlat[imeso], srlon[imeso]

    isort = np.argsort(mtimes)
    mtimes, srlat, srlon = mtimes[isort], srlat[isort], srlon[isort]

    dx_storm, dy_storm = get_dx_dy(srlon[0], srlat[0], srlon[-1], srlat[-1])
    dt_storm = (mtimes[-1] - mtimes[0]).total_seconds()
    if dt_storm <= 0:
        return None
    u_storm, v_storm = dx_storm / dt_storm, dy_storm / dt_storm
    return mtimes, srlat, srlon, u_storm, v_storm

def _srh(prof, zbot_m, ztop_m, stu, stv):
    _, srh, _ = winds.helicity(prof, zbot_m, ztop_m, stu=stu, stv=stv)
    return float(srh)


def _nan_if_masked(x):
    arr = np.asanyarray(x)
    if np.ma.isMaskedArray(arr):
        return np.nan if arr.mask.any() else float(arr)
    try:
        val = float(arr)
    except Exception:
        return np.nan
    return val if np.isfinite(val) else np.nan


def bunkers_from_raw(prof):
    out = {
        "BNK_RM_u": np.nan, "BNK_RM_v": np.nan, "BNK_RM_spd": np.nan, "BNK_RM_dir_deg": np.nan,
        "BNK_LM_u": np.nan, "BNK_LM_v": np.nan, "BNK_LM_spd": np.nan, "BNK_LM_dir_deg": np.nan,
        "SRH_05_bunkers": np.nan, "SRH_01_bunkers": np.nan, "SRH_03_bunkers": np.nan,
        "0_6km_mean_u": np.nan, "0_6km_mean_v": np.nan, "0_6km_mean_spd": np.nan, "0_6km_mean_dir": np.nan,
        "SRH_effective_bunkers": np.nan, "0_6km_mean_wind": np.nan,
    }
    try:
        urm, vrm, ulm, vlm = params.bunkers_storm_motion(prof)
    except Exception:
        return out

    urm = _nan_if_masked(urm)
    vrm = _nan_if_masked(vrm)
    ulm = _nan_if_masked(ulm)
    vlm = _nan_if_masked(vlm)
    if not np.isfinite(urm) or not np.isfinite(vrm) or not np.isfinite(ulm) or not np.isfinite(vlm):
        return out

    rm_spd = np.hypot(urm, vrm)
    lm_spd = np.hypot(ulm, vlm)
    rm_dir = float(np.asarray(mpcalc.wind_direction(urm * units('m/s'), vrm * units('m/s')).m))
    lm_dir = float(np.asarray(mpcalc.wind_direction(ulm * units('m/s'), vlm * units('m/s')).m))

    try:
        srh05 = _srh(prof, 0, 500, urm, vrm)
        srh01 = _srh(prof, 0, 1000, urm, vrm)
        srh03 = _srh(prof, 0, 3000, urm, vrm)
    except Exception:
        srh05 = srh01 = srh03 = np.nan

    try:
        pbot, ptop = params.effective_inflow_layer(prof)
        if np.isnan(pbot) or np.isnan(ptop):
            srh_eff = np.nan
        else:
            ebot_h = interp.to_agl(prof, interp.hght(prof, pbot))
            etop_h = interp.to_agl(prof, interp.hght(prof, ptop))
            _, srh_eff, _ = winds.helicity(prof, ebot_h, etop_h, stu=urm, stv=vrm)
    except Exception:
        srh_eff = np.nan

    try:
        p0, p6 = interp.pres(prof, 0), interp.pres(prof, 6000)
        u0, v0 = interp.components(prof, p0)
        u6, v6 = interp.components(prof, p6)
        mean06_u, mean06_v = u6 - u0, v6 - v0
        mean06_spd = np.sqrt((mean06_u ** 2) + (mean06_v ** 2))
        mean06_dir = mpcalc.wind_direction(mean06_u * units('m/s'), mean06_v * units('m/s')).m
    except Exception:
        mean06_u = mean06_v = mean06_spd = mean06_dir = np.nan

    out.update({
        "BNK_RM_u": urm, "BNK_RM_v": vrm, "BNK_RM_spd": rm_spd, "BNK_RM_dir_deg": rm_dir,
        "BNK_LM_u": ulm, "BNK_LM_v": vlm, "BNK_LM_spd": lm_spd, "BNK_LM_dir_deg": lm_dir,
        "SRH_05_bunkers": srh05, "SRH_01_bunkers": srh01, "SRH_03_bunkers": srh03,
        "SRH_effective_bunkers": srh_eff, "0_6km_mean_u": mean06_u, "0_6km_mean_v": mean06_v,
        "0_6km_mean_spd": mean06_spd, "0_6km_mean_dir": mean06_dir,
    })
    return out


def point_from_dist_bearing(start_lat, start_lon, distance_km, bearing):
    start_point = Point(start_lat, start_lon)
    destination = distance(kilometers=distance_km).destination(point=start_point, bearing=bearing)
    return destination.latitude, destination.longitude


def hour_rounder(t):
    return t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(hours=t.minute // 30)


def subset_data_km(clat, clon, lat2d, lon2d, data2d, dlat_km, dlon_km):
    lat2d = np.asarray(lat2d)
    lon2d = np.asarray(lon2d)
    data2d = np.asarray(data2d)

    def wrap_like(lon, ref):
        if np.nanmax(lon) > 180:
            lon = lon % 360.0
            ref = ref % 360.0
        else:
            lon = ((lon + 180) % 360) - 180
            ref = ((ref + 180) % 360) - 180
        return lon, ref

    lon2d, clon = wrap_like(lon2d, clon)
    deg2km = 111.32
    coslat = np.cos(np.deg2rad(clat))
    dx_km = ((((lon2d - clon + 180) % 360) - 180) * coslat * deg2km)
    dy_km = (lat2d - clat) * deg2km
    m = (np.abs(dx_km) <= dlon_km) & (np.abs(dy_km) <= dlat_km)
    if not np.any(m):
        raise ValueError('No grid points inside requested km box.')
    i_idx, j_idx = np.where(m)
    i0, i1 = i_idx.min(), i_idx.max()
    j0, j1 = j_idx.min(), j_idx.max()
    return lat2d[i0:i1+1, j0:j1+1], lon2d[i0:i1+1, j0:j1+1], data2d[i0:i1+1, j0:j1+1]


def pressure_at_height(p_lev_hPa, z3d_m, Z_m, fill_value=np.nan, extrapolate="nearest"):
    z3d = np.asarray(z3d_m, dtype=float)
    p1d = np.asarray(p_lev_hPa, dtype=float)
    nz, ny, nx = z3d.shape
    Z = np.broadcast_to(np.asarray(Z_m, dtype=float), (ny, nx))
    if np.nanmean(np.diff(z3d[:, 0, 0])) < 0:
        z3d = z3d[::-1, ...]
        p1d = p1d[::-1]
    z0 = z3d[:-1]
    z1 = z3d[1:]
    cross = (z0 <= Z) & (z1 >= Z)
    k_cross = cross.argmax(axis=0)
    has_cross = cross.any(axis=0)
    below = Z < np.nanmin(z3d, axis=0)
    above = Z > np.nanmax(z3d, axis=0)
    if extrapolate == "nearest":
        k = np.where(has_cross, k_cross, np.where(below, 0, np.where(above, nz - 2, k_cross)))
    else:
        k = np.where(has_cross, k_cross, -1)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    pZ = np.full((ny, nx), fill_value, dtype=float)
    valid = (k >= 0)
    if not np.any(valid):
        return pZ
    kv = k[valid]
    jv = jj[valid]
    iv = ii[valid]
    Zv = Z[valid]
    zk = z3d[kv, jv, iv]
    zk1 = z3d[kv + 1, jv, iv]
    denom = (zk1 - zk)
    w = np.where(np.abs(denom) > 0, (Zv - zk) / denom, 0.0)
    ln_pk = np.log(p1d[kv])
    ln_pk1 = np.log(p1d[kv + 1])
    pZ[valid] = np.exp((1.0 - w) * ln_pk + w * ln_pk1)
    return pZ


def state_at_height(p_lev_hPa, z3d_m, T3d_C, RH3d_pct, U3d, V3d, Z_m, fill_value=np.nan, extrapolate="nearest"):
    z3d = np.asarray(z3d_m, dtype=float)
    T3d = np.asarray(T3d_C, dtype=float)
    RH3d = np.asarray(RH3d_pct, dtype=float)
    U3d = np.asarray(U3d, dtype=float)
    V3d = np.asarray(V3d, dtype=float)
    nz, ny, nx = z3d.shape
    Z = np.broadcast_to(np.asarray(Z_m, dtype=float), (ny, nx))
    p1d = np.asarray(p_lev_hPa, dtype=float)
    if np.nanmean(np.diff(z3d[:, 0, 0])) < 0:
        z3d = z3d[::-1, ...]
        T3d = T3d[::-1, ...]
        RH3d = RH3d[::-1, ...]
        U3d = U3d[::-1, ...]
        V3d = V3d[::-1, ...]
        p1d = p1d[::-1]
    p3d = p1d[:, None, None] * np.ones_like(z3d)
    w3d = mixing_ratio_from_relative_humidity(
        p3d * units.hectopascal,
        T3d * units.degC,
        RH3d * units.percent,
    ).m
    z0 = z3d[:-1]
    z1 = z3d[1:]
    cross = (z0 <= Z) & (z1 >= Z)
    k_cross = cross.argmax(axis=0)
    has_cross = cross.any(axis=0)
    below = Z < np.nanmin(z3d, axis=0)
    above = Z > np.nanmax(z3d, axis=0)
    if extrapolate == "nearest":
        k = np.where(has_cross, k_cross, np.where(below, 0, np.where(above, nz - 2, k_cross)))
    else:
        k = np.where(has_cross, k_cross, -1)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    TZ = np.full((ny, nx), fill_value, float)
    wZ = np.full((ny, nx), fill_value, float)
    UZ = np.full((ny, nx), fill_value, float)
    VZ = np.full((ny, nx), fill_value, float)
    valid = (k >= 0)
    if np.any(valid):
        kv = k[valid]
        jv = jj[valid]
        iv = ii[valid]
        Zv = Z[valid]
        def lerp(f3d):
            fk = f3d[kv, jv, iv]
            fk1 = f3d[kv + 1, jv, iv]
            zk = z3d[kv, jv, iv]
            zk1 = z3d[kv + 1, jv, iv]
            w = np.where(np.abs(zk1 - zk) > 0, (Zv - zk) / (zk1 - zk), 0.0)
            return fk + w * (fk1 - fk)
        TZ[valid] = lerp(T3d)
        wZ[valid] = lerp(w3d)
        UZ[valid] = lerp(U3d)
        VZ[valid] = lerp(V3d)
    pZ = pressure_at_height(p1d, z3d, Z, fill_value=fill_value, extrapolate=extrapolate)
    RHZ = np.full((ny, nx), fill_value, float)
    ok = np.isfinite(pZ) & np.isfinite(TZ) & np.isfinite(wZ)
    if np.any(ok):
        RHZ[ok] = relative_humidity_from_mixing_ratio(
            pZ[ok] * units.hectopascal,
            TZ[ok] * units.degC,
            wZ[ok] * units.dimensionless,
        ).to('percent').m
    return pZ, TZ, UZ, VZ, RHZ


def download_read_file(date, download_pth, levels, grid=130):
    Path(download_pth).mkdir(parents=True, exist_ok=True)
    if date.year > 2012:
        filename = f"rap_{grid}_{date:%Y%m%d_%H%M}_000.grb2"
        url = f"https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/{date:%Y%m}/{date:%Y%m%d}/{filename}"
    else:
        filename = f"ruc2anl_{grid}_{date:%Y%m%d_%H%M}_000.grb2"
        url = f"https://www.ncei.noaa.gov/thredds/fileServer/model-ruc{grid}anl/{date:%Y%m}/{date:%Y%m%d}/{filename}"
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
        try:
            download_atomic(url, target)
            return True
        except Exception:
            return False

    if not ensure_local_file():
        return None

    def open_level(path, level, short_name=None):
        indexless = {"indexpath": ""}
        fkeys = {"typeOfLevel": "isobaricInhPa", "level": int(level)}
        if short_name is not None:
            fkeys["shortName"] = short_name
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(filter_by_keys=fkeys, **indexless))

    def read_iso_dataset(path):
        ds_t = xr.concat([open_level(path, L, "t") for L in levels], dim="isobaricInhPa")
        ds_u = xr.concat([open_level(path, L, "u") for L in levels], dim="isobaricInhPa")
        ds_v = xr.concat([open_level(path, L, "v") for L in levels], dim="isobaricInhPa")
        ds_gh = xr.concat([open_level(path, L, "gh") for L in levels], dim="isobaricInhPa")
        ds_r = xr.concat([open_level(path, L, "r") for L in levels], dim="isobaricInhPa")
        return xr.merge([ds_t, ds_u, ds_v, ds_gh, ds_r], compat="override")

    try:
        ds_iso = read_iso_dataset(str(target))
    except Exception:
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass
        if not ensure_local_file():
            return None
        ds_iso = read_iso_dataset(str(target))
    return ds_iso, str(target)


def download_rap_from_source(atime, download_pth, clat, clon, levels, grid=130, dx=50, dy=50):
    out = download_read_file(hour_rounder(atime), download_pth, levels, grid=grid)
    if out is None:
        return None
    ds_iso, filename = out
    for al, level in enumerate(levels):
        ds = ds_iso.sel(isobaricInhPa=level)
        latitudes = ds.latitude.values
        longitudes = ds.longitude.values - 360
        lat_slice, lon_slice, at = subset_data_km(clat, clon, latitudes, longitudes, ds.t.values, dy, dx)
        _, _, au = subset_data_km(clat, clon, latitudes, longitudes, ds.u.values, dy, dx)
        _, _, av = subset_data_km(clat, clon, latitudes, longitudes, ds.v.values, dy, dx)
        _, _, agh = subset_data_km(clat, clon, latitudes, longitudes, ds.gh.values, dy, dx)
        _, _, ar = subset_data_km(clat, clon, latitudes, longitudes, ds.r.values, dy, dx)
        if al == 0:
            shape = (len(levels), lat_slice.shape[0], lat_slice.shape[1])
            t = np.full(shape, np.nan)
            u = np.full(shape, np.nan)
            v = np.full(shape, np.nan)
            gh = np.full(shape, np.nan)
            r = np.full(shape, np.nan)
        t[al, :, :] = at - 273.15
        u[al, :, :] = au
        v[al, :, :] = av
        gh[al, :, :] = agh
        r[al, :, :] = ar
    return t, u, v, gh, r, levels, lat_slice, lon_slice, filename


def get_RAP_RUC_data(atime, download_pth, clat, clon, sounding_distance, levels):
    try:
        H = Herbie(hour_rounder(atime).strftime('%Y-%m-%d %H:%M'), model='rap', fxx=0, save_dir=download_pth)
        if H.grib is None:
            raise FileNotFoundError
        for al, level in enumerate(levels):
            level_str = f':{int(level)} mb' if level != 1013.2 else f':{level} mb'
            ds = H.xarray(level_str, backend_kwargs={'indexpath': 'memory'})
            latitudes, longitudes = ds.latitude.values, ds.longitude.values - 360
            lat_slice, lon_slice, at = subset_data_km(clat, clon, latitudes, longitudes, ds.t.values, sounding_distance, sounding_distance)
            _, _, au = subset_data_km(clat, clon, latitudes, longitudes, ds.u.values, sounding_distance, sounding_distance)
            _, _, av = subset_data_km(clat, clon, latitudes, longitudes, ds.v.values, sounding_distance, sounding_distance)
            _, _, agh = subset_data_km(clat, clon, latitudes, longitudes, ds.gh.values, sounding_distance, sounding_distance)
            _, _, ar = subset_data_km(clat, clon, latitudes, longitudes, ds.r.values, sounding_distance, sounding_distance)
            if al == 0:
                shape = (len(levels), lat_slice.shape[0], lat_slice.shape[1])
                t = np.full(shape, np.nan)
                u = np.full(shape, np.nan)
                v = np.full(shape, np.nan)
                gh = np.full(shape, np.nan)
                r = np.full(shape, np.nan)
            t[al, :, :] = at - 273.15
            u[al, :, :] = au
            v[al, :, :] = av
            gh[al, :, :] = agh
            r[al, :, :] = ar
        return t, u, v, gh, r, levels, lat_slice, lon_slice
    except Exception:
        out = download_rap_from_source(atime, download_pth, clat, clon, levels, grid=130, dx=sounding_distance, dy=sounding_distance)
        if out is None:
            return None
        t, u, v, gh, r, levels, lat_slice, lon_slice, _filename = out
        return t, u, v, gh, r, levels, lat_slice, lon_slice



# -----------------------------------------------------------------------------
# DEM-derived regional reference elevation helpers
# -----------------------------------------------------------------------------
_ELEV_WARNED_TILES = set()


def _elev_tile_id(lat, lon):
    """Return USGS 1-degree tile ID for a lat/lon point.

    This follows the same convention used by the mobile-mesonet processing
    script: lat 34.5 -> n35, lon -101.8 -> w102.
    """
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None
    lon = -abs(float(lon)) if lon > 0 else float(lon)
    lat_id = int(np.ceil(float(lat)))
    lon_id = int(-np.floor(float(lon)))
    lat_prefix = "n" if lat_id >= 0 else "s"
    lon_prefix = "w" if lon_id >= 0 else "e"
    return f"{lat_prefix}{abs(lat_id):02d}{lon_prefix}{abs(lon_id):03d}"


def _tile_ids_for_bbox(lon_min, lat_min, lon_max, lat_max):
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


@lru_cache(maxsize=None)
def _find_local_elev_tile(elev_dir, tile_id):
    matches = sorted(
        glob(os.path.join(str(elev_dir), f"*{tile_id}*.tif"))
        + glob(os.path.join(str(elev_dir), f"*{tile_id}*.tiff"))
    )
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
        if tile_id not in _ELEV_WARNED_TILES:
            print(f"[WARN] elevation tile {tile_id} not found at https://prd-tnm.s3.amazonaws.com/{prefix}", flush=True)
            _ELEV_WARNED_TILES.add(tile_id)
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
    except Exception:
        if tile_id not in _ELEV_WARNED_TILES:
            print(f"[WARN] failed to download elevation tile {tile_id} from https://prd-tnm.s3.amazonaws.com/{key}", flush=True)
            _ELEV_WARNED_TILES.add(tile_id)
        return None

    print(f"[INFO] downloaded elevation tile https://prd-tnm.s3.amazonaws.com/{key}", flush=True)
    _find_local_elev_tile.cache_clear()
    return out_path


def _resolve_elev_tile_path(elev_dir, tile_id, allow_download=True):
    local_path = _find_local_elev_tile(str(elev_dir), tile_id)
    if local_path is None and allow_download:
        downloaded = _download_elev_tile(Path(elev_dir), tile_id)
        if downloaded is not None:
            local_path = str(downloaded)
            _find_local_elev_tile.cache_clear()
    return local_path



def _expand_bbox_min_size(lon_min, lat_min, lon_max, lat_max, *, min_size_deg=0.002):
    """Expand very small lon/lat boxes so rasterio reads at least a few DEM cells.

    Some analysis windows contain nearly stationary/single-point mesonet
    footprints. A near-zero bounding box can round to a zero-size raster window,
    causing the DEM mean elevation to return NaN even when the correct tile
    exists. This expansion is small (~200 m in latitude) and only affects
    degenerate footprints.
    """
    lon_min = float(lon_min); lat_min = float(lat_min)
    lon_max = float(lon_max); lat_max = float(lat_max)
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    cx = 0.5 * (lon_min + lon_max)
    cy = 0.5 * (lat_min + lat_max)
    width = lon_max - lon_min
    height = lat_max - lat_min
    if not np.isfinite(width) or not np.isfinite(height):
        return lon_min, lat_min, lon_max, lat_max
    if width < min_size_deg:
        lon_min = cx - 0.5 * min_size_deg
        lon_max = cx + 0.5 * min_size_deg
    if height < min_size_deg:
        lat_min = cy - 0.5 * min_size_deg
        lat_max = cy + 0.5 * min_size_deg
    return lon_min, lat_min, lon_max, lat_max

def _dem_mean_in_bbox(elev_dir, lon_min, lat_min, lon_max, lat_max):
    elev_dir = Path(elev_dir)
    elev_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = _expand_bbox_min_size(lon_min, lat_min, lon_max, lat_max)
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
            print(f"[WARN] failed reading DEM tile {local_path} for bbox mean: {exc!r}", flush=True)

    z_ref_m = float(np.nanmean(np.concatenate(vals))) if vals else np.nan
    return {
        "z_ref_m": z_ref_m,
        "bbox": (float(lon_min), float(lat_min), float(lon_max), float(lat_max)),
        "dem_files": used_files,
        "n_dem_pixels": int(n_pixels),
        "tile_ids": tile_ids,
    }


def derive_dem_bbox_elev(dir_mesonet, elev_dir, atime, time_to_space_conversion=5):
    """DEM mean elevation over the mobile-mesonet footprint for one analysis time.

    The footprint is the min/max lon/lat of all valid mobile mesonet samples in
    the analysis window. Missing USGS/TNM 3DEP tiles are downloaded as needed.
    This is used as the representative regional surface elevation Z for RAP/RUC
    proximity sounding extraction.
    """
    mmfs = get_mm_files(dir_mesonet, atime)
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
        return np.nan, []

    lats = np.concatenate(lats_all)
    lons = np.concatenate(lons_all)
    info = _dem_mean_in_bbox(Path(elev_dir), float(np.nanmin(lons)), float(np.nanmin(lats)), float(np.nanmax(lons)), float(np.nanmax(lats)))
    return float(info.get("z_ref_m", np.nan)), mmfs


def _inflow_proximity_point(meso_lat, meso_lon, u_storm, v_storm,
                            dist_km=20.0, normalized_storm_motion=90.0,
                            rel_bearing_deg=135.0):
    """Return (lat, lon) of a proximity point in the storm-relative inflow sector.

    The point is placed `dist_km` from the mesocyclone along `rel_bearing_deg`,
    where the bearing is measured in the rotated, storm-relative frame used by
    rotate_grid (target_angle=normalized_storm_motion). With the storm motion
    vector rotated to `normalized_storm_motion`, a bearing of 135 deg lands ahead
    and to the right of storm motion (rdx>0, rdy<0) -- the inflow sector
    (quadrant IV) used by the downstream composite analysis. The rotated offset
    is de-rotated to geographic coordinates so the standard
    get_dx_dy -> rotate_grid pipeline recovers the intended inflow quadrant
    regardless of the storm-motion direction.
    """
    if not (np.isfinite(meso_lat) and np.isfinite(meso_lon)
            and np.isfinite(u_storm) and np.isfinite(v_storm)):
        return np.nan, np.nan
    br = np.radians(rel_bearing_deg)
    rdx = dist_km * np.sin(br)   # km; +x aligned with storm motion
    rdy = dist_km * np.cos(br)   # km; 135 deg -> rdx>0, rdy<0 (inflow)
    wind_angle = float(np.array(mpcalc.wind_direction(
        u_storm * (units.meter / units.second),
        v_storm * (units.meter / units.second),
        convention='to')))
    ra = np.radians(wind_angle) - np.radians(normalized_storm_motion)
    cos_a, sin_a = np.cos(ra), np.sin(ra)
    # Inverse of rotate_grid: geographic = R(-ra) * rotated
    dx = cos_a * rdx + sin_a * rdy   # km east
    dy = -sin_a * rdx + cos_a * rdy  # km north
    dist = float(np.hypot(dx, dy))
    bearing = float(np.degrees(np.arctan2(dx, dy)) % 360.0)
    return point_from_dist_bearing(meso_lat, meso_lon, dist, bearing)


def _is_inflow_point(meso_lat, meso_lon, ob_lat, ob_lon, u_storm, v_storm,
                     normalized_storm_motion=90.0):
    """True if (ob_lat, ob_lon) lies in the storm-relative inflow sector.

    Uses the same get_dx_dy -> rotate_grid convention as the downstream composite
    analysis; inflow is quadrant IV (rdx>0, rdy<0).
    """
    if not (np.isfinite(ob_lat) and np.isfinite(ob_lon)
            and np.isfinite(meso_lat) and np.isfinite(meso_lon)
            and np.isfinite(u_storm) and np.isfinite(v_storm)):
        return False
    dx, dy = get_dx_dy(meso_lon, meso_lat, ob_lon, ob_lat)
    rdx, rdy = rotate_grid(dx, dy, u_storm, v_storm, pivot_x=0, pivot_y=0,
                           target_angle=normalized_storm_motion)
    return bool((rdx > 0) and (rdy < 0))


def make_RAP_RUC_profile(atime, mtime, mlat, mlon, levels, download_pth, u_storm, v_storm, Z,
                         proximity_sounding_dist=20, proximity_sounding_bearing=135,
                         sounding_distance=50, normalized_storm_motion=90):
    if not np.isfinite(Z):
        return None, None, None, None, None, None
    # Place the proximity point in the storm-relative inflow sector (quadrant IV:
    # rdx>0, rdy<0 after rotating storm motion to `normalized_storm_motion`).
    # `proximity_sounding_bearing` is interpreted as a storm-relative bearing in
    # the rotated frame (135 deg -> right-front/inflow), making the placement
    # robust to the storm-motion direction rather than a fixed geographic bearing.
    clat, clon = _inflow_proximity_point(
        mlat[int(mtime.size / 2)],
        mlon[int(mtime.size / 2)],
        u_storm,
        v_storm,
        dist_km=proximity_sounding_dist,
        normalized_storm_motion=normalized_storm_motion,
        rel_bearing_deg=proximity_sounding_bearing,
    )
    if not (np.isfinite(clat) and np.isfinite(clon)):
        return None, None, None, None, None, None
    out = get_RAP_RUC_data(atime, download_pth, clat, clon, sounding_distance, levels)
    if out is None:
        return None, None, None, None, None, None
    t, u, v, gh, r, levels, lat_slice, lon_slice = out
    j, i = np.unravel_index(np.nanargmin((lat_slice - clat) ** 2 + (lon_slice - clon) ** 2), lat_slice.shape)
    dz = 100.0
    heights = np.arange(Z, 16000.0 + dz, dz)
    spres = np.full(heights.size, np.nan)
    stemp = np.full_like(spres, np.nan)
    sdew = np.full_like(spres, np.nan)
    su = np.full_like(spres, np.nan)
    sv = np.full_like(spres, np.nan)
    for k, zt in enumerate(heights):
        ppres, ptemp, pu, pv, prh = state_at_height(levels, gh, t, r, u, v, zt, fill_value=np.nan, extrapolate="nearest")
        spres[k] = ppres[j, i]
        stemp[k] = ptemp[j, i]
        su[k] = pu[j, i]
        sv[k] = pv[j, i]
        rh_frac = (prh[j, i] / 100.0) * units.dimensionless
        sdew[k] = mpcalc.dewpoint_from_relative_humidity((ptemp[j, i] * units.degC).to('kelvin'), rh_frac).to('degC').m
    bprof = profile.create_profile(profile='default', pres=spres, hght=heights - heights[0], tmpc=stemp, dwpc=sdew, u=su * KTS_PER_MPS, v=sv * KTS_PER_MPS, missing=-9999, strictQC=False)
    bunk_obs = bunkers_from_raw(bprof)
    su_sr = su - u_storm
    sv_sr = sv - v_storm
    ru_sr, rv_sr = rotate_grid(su_sr, sv_sr, u_storm, v_storm, pivot_x=0, pivot_y=0, target_angle=normalized_storm_motion)
    prof_sr_rot = profile.create_profile(profile='default', pres=spres, hght=heights - heights[0], tmpc=stemp, dwpc=sdew, u=ru_sr * KTS_PER_MPS, v=rv_sr * KTS_PER_MPS, missing=-9999, strictQC=False)
    return prof_sr_rot, clat, clon, bunk_obs, su, sv


def _float_list(values):
    arr = np.asarray(values, dtype=float)
    out = []
    for val in arr:
        out.append(float(val) if np.isfinite(val) else np.nan)
    return out


def _wind_dir_from_uv_ms(u_ms, v_ms):
    u_ms = np.asarray(u_ms, dtype=float)
    v_ms = np.asarray(v_ms, dtype=float)
    return (270.0 - np.degrees(np.arctan2(v_ms, u_ms))) % 360.0


def _neutral_sounding_record(pres, hght, tmpc, dwpc, u_ms, v_ms, *, source_model, u_storm=np.nan, v_storm=np.nan, wind_note="earth_relative"):
    """Return a SHARPpy-free profile record for GUI/bundle consumption."""
    pres = np.asarray(pres, dtype=float)
    hght = np.asarray(hght, dtype=float)
    tmpc = np.asarray(tmpc, dtype=float)
    dwpc = np.asarray(dwpc, dtype=float)
    u_ms = np.asarray(u_ms, dtype=float)
    v_ms = np.asarray(v_ms, dtype=float)
    wdir = _wind_dir_from_uv_ms(u_ms, v_ms)
    wspd = np.hypot(u_ms, v_ms)
    return {
        "schema": "meso_view_neutral_sounding_v1",
        "source_model": str(source_model).upper(),
        "wind_frame": "earth_relative",
        "wind_note": wind_note,
        "storm_motion_u_ms": float(u_storm) if np.isfinite(u_storm) else np.nan,
        "storm_motion_v_ms": float(v_storm) if np.isfinite(v_storm) else np.nan,
        "pres": _float_list(pres),
        "hght": _float_list(hght),
        "tmpc": _float_list(tmpc),
        "dwpc": _float_list(dwpc),
        "u_ms": _float_list(u_ms),
        "v_ms": _float_list(v_ms),
        "wspd_ms": _float_list(wspd),
        "wdir_deg": _float_list(wdir),
    }


def _neutral_record_from_profile(prof, u_ms, v_ms, *, source_model, u_storm=np.nan, v_storm=np.nan, wind_note="earth_relative"):
    if prof is None or not hasattr(prof, "pres"):
        return np.nan
    return _neutral_sounding_record(
        np.asarray(prof.pres, dtype=float),
        np.asarray(prof.hght, dtype=float),
        np.asarray(prof.tmpc, dtype=float),
        np.asarray(prof.dwpc, dtype=float),
        u_ms,
        v_ms,
        source_model=source_model,
        u_storm=u_storm,
        v_storm=v_storm,
        wind_note=wind_note,
    )


def _source_model_for_time(atime):
    return "RAP" if atime.year > 2012 else "RUC"


def run_finish_soundings(sound, dir_transects, tracking_dir, rap_dir, rap_output, bunkers_output,
                         dir_mesonet, elev_dir, gui_output=None,
                         normalized_storm_motion=90.0, time_to_space=5, analysis_start=-20, analysis_end=20,
                         sounding_timeframe=60.0, sounding_distance=50.0, proximity_sounding_dist=20.0,
                         proximity_sounding_bearing=135.0, elevation_max_distance=20.0, modes=('T', 'N'), comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rap_dir = os.path.abspath(rap_dir)
    rank_rap_dir = os.path.join(rap_dir, f'rank{rank}')
    os.makedirs(rank_rap_dir, exist_ok=True)
    output_dir = os.path.dirname(os.path.abspath(rap_output)) or '.'
    os.makedirs(output_dir, exist_ok=True)
    if gui_output:
        gui_dir = os.path.dirname(os.path.abspath(gui_output)) or '.'
        os.makedirs(gui_dir, exist_ok=True)
    analysis_times = np.arange(analysis_start, analysis_end + 1, time_to_space)
    levels = np.arange(100, 1025, 25)[::-1]
    dt_tracking = 1/60
    merged_profs_local, merged_lat_local, merged_lon_local = {}, {}, {}
    merged_gui_local = {}
    merged_bunk_local = {mode: {} for mode in modes}

    for mode in modes:
        t0, amode, tr = get_t0s(dir_transects, mode)
        tasks, skipped_no_obs_overlap = _build_analysis_tasks(tr, t0, amode, analysis_times, time_to_space)
        my_idx = list(range(rank, len(tasks), size))
        if rank == 0:
            print(
                f'[INFO] mode={mode} total_tasks={len(tasks)} '
                f'skipped_no_obs_overlap={skipped_no_obs_overlap} ranks={size}',
                flush=True,
            )
        for i in tqdm(my_idx, total=len(my_idx), disable=(rank != 0)):
            task = tasks[i]
            atime = task['atime']
            key = task['key']
            prof_sr_rot = np.nan
            gui_profile = np.nan
            clat, clon = np.nan, np.nan
            merged_bunk_local[mode][key] = None

            dat = parse_storm_tracks(
                task['case_id'],
                tracking_dir,
                amode,
                atime,
                dt_tracking=dt_tracking,
                time_to_space_conversion=time_to_space,
            )
            if dat is None:
                if rank == 0:
                    print(
                        f"[WARN] no storm-track coverage for mode={mode} ID={task['transect_id']} "
                        f"CaseID={task['case_label']} offset={task['offset_min']} "
                        f"window={task['window_start']}..{task['window_end']}",
                        flush=True,
                    )
                merged_profs_local[key] = prof_sr_rot
                merged_gui_local[key] = gui_profile
                merged_lat_local[key] = clat
                merged_lon_local[key] = clon
                continue

            meso_tm, meso_la, meso_lo, u_stm, v_stm = dat
            u_stm = float(u_stm)
            v_stm = float(v_stm)
            Z, _mmfs = derive_dem_bbox_elev(
                dir_mesonet=dir_mesonet,
                elev_dir=elev_dir,
                atime=atime,
                time_to_space_conversion=time_to_space,
            )
            if not np.isfinite(Z):
                if rank == 0:
                    print(
                        f"[WARN] no finite DEM regional elevation Z for mode={mode} ID={task['transect_id']} "
                        f"CaseID={task['case_label']} offset={task['offset_min']} atime={atime}",
                        flush=True,
                    )
            prof_sr_rot = np.nan
            gui_profile = np.nan
            clat, clon = np.nan, np.nan
            merged_bunk_local[mode][key] = None
            source_model = _source_model_for_time(atime)
            itm = np.where(
                (sound['time'] >= atime - timedelta(seconds=int((sounding_timeframe * 60) / 2))) &
                (sound['time'] <= atime + timedelta(seconds=int((sounding_timeframe * 60) / 2)))
            )[0]
            have_obs = False
            if itm.size > 0:
                soundlat = np.asarray(sound['lat'][itm], dtype=float)
                soundlon = np.asarray(sound['lon'][itm], dtype=float)
                meso_la = np.asarray(meso_la, dtype=float)
                meso_lo = np.asarray(meso_lo, dtype=float)
                dists = np.array([np.nanmin(haversine(soundlon[j], soundlat[j], meso_lo, meso_la)) for j in range(itm.size)], dtype=float)
                _mc = int(np.asarray(meso_la).size / 2)
                _cand = itm[int(np.nanargmin(dists))] if np.isfinite(np.nanmin(dists)) else None
                # Accept the observed sounding only if it is valid AND lies in the
                # storm-relative inflow sector. Otherwise fall through to the
                # RAP/RUC inflow proximity sounding below so that every case has an
                # inflow sounding (previously, a non-inflow observed sounding was
                # kept and then dropped by the downstream inflow-only filter).
                _obs_inflow = (
                    _cand is not None
                    and np.nanmin(dists) <= sounding_distance
                    and hasattr(sound['profs'][_cand], 'pres')
                    and _is_inflow_point(meso_la[_mc], meso_lo[_mc],
                                         sound['lat'][_cand], sound['lon'][_cand],
                                         u_stm, v_stm, normalized_storm_motion)
                )
                if _obs_inflow:
                    isel = itm[np.nanargmin(dists)]
                    fprof = sound['profs'][isel]
                    if not hasattr(fprof, 'pres'):
                        merged_profs_local[key] = prof_sr_rot
                        merged_gui_local[key] = gui_profile
                        merged_lat_local[key] = clat
                        merged_lon_local[key] = clon
                        continue
                    clat = sound['lat'][isel]
                    clon = sound['lon'][isel]
                    have_obs = True
                    try:
                        _ = np.isnan(fprof)
                        prof_sr_rot, clat, clon, bunk_obs, raw_u_ms, raw_v_ms = make_RAP_RUC_profile(
                            atime, meso_tm, meso_la, meso_lo, levels, rank_rap_dir, u_stm, v_stm, Z,
                            proximity_sounding_dist=proximity_sounding_dist,
                            proximity_sounding_bearing=proximity_sounding_bearing,
                            sounding_distance=sounding_distance,
                            normalized_storm_motion=normalized_storm_motion,
                        )
                        gui_profile = _neutral_record_from_profile(
                            prof_sr_rot,
                            raw_u_ms,
                            raw_v_ms,
                            source_model=source_model,
                            u_storm=u_stm,
                            v_storm=v_stm,
                            wind_note="model_earth_relative",
                        )
                        merged_bunk_local[mode][key] = bunk_obs
                    except TypeError:
                        ftemp = np.array(fprof.tmpc)
                        fdew = np.array(fprof.dwpc)
                        fhght = np.array(fprof.hght)
                        fpres = np.array(fprof.pres)
                        fu = np.array(fprof.u)
                        fv = np.array(fprof.v)
                        fu_sr = fu * KNOTS_TO_MS - u_stm
                        fv_sr = fv * KNOTS_TO_MS - v_stm
                        rfu_sr, rfv_sr = rotate_grid(fu_sr, fv_sr, u_stm, v_stm, pivot_x=0, pivot_y=0, target_angle=normalized_storm_motion)
                        if fpres[-1] > 100:
                            rap_sr_rot, _, _, _, rapu, rapv = make_RAP_RUC_profile(
                                atime, meso_tm, meso_la, meso_lo, levels, rank_rap_dir, u_stm, v_stm, Z,
                                proximity_sounding_dist=proximity_sounding_dist,
                                proximity_sounding_bearing=proximity_sounding_bearing,
                                sounding_distance=sounding_distance,
                                normalized_storm_motion=normalized_storm_motion,
                            )
                            if rap_sr_rot is None:
                                prof_sr_rot = profile.create_profile(profile='default', pres=fpres, hght=fhght, tmpc=ftemp, dwpc=fdew, u=rfu_sr * KTS_PER_MPS, v=rfv_sr * KTS_PER_MPS, missing=-9999, strictQC=False)
                                bprof = profile.create_profile(profile='default', pres=fpres, hght=fhght-fhght[0], tmpc=ftemp, dwpc=fdew, u=fu, v=fv, missing=-9999, strictQC=False)
                                gui_profile = _neutral_sounding_record(
                                    fpres, fhght, ftemp, fdew,
                                    fu * KNOTS_TO_MS, fv * KNOTS_TO_MS,
                                    source_model="OBSERVED",
                                    u_storm=u_stm,
                                    v_storm=v_stm,
                                    wind_note="observed_earth_relative",
                                )
                                merged_bunk_local[mode][key] = bunkers_from_raw(bprof)
                            else:
                                iend = (np.abs(np.array(rap_sr_rot.pres) - fpres[-1]).argmin())
                                ftemp = np.append(ftemp, np.array(rap_sr_rot.tmpc)[iend:])
                                fdew = np.append(fdew, np.array(rap_sr_rot.dwpc)[iend:])
                                fhght = np.append(fhght, np.array(rap_sr_rot.hght)[iend:])
                                fpres = np.append(fpres, np.array(rap_sr_rot.pres)[iend:])
                                fu_k = np.append(rfu_sr * KTS_PER_MPS, np.array(rap_sr_rot.u)[iend:])
                                fv_k = np.append(rfv_sr * KTS_PER_MPS, np.array(rap_sr_rot.v)[iend:])
                                fu_raw_k = np.append(fu, np.array(rapu)[iend:] * KTS_PER_MPS)
                                fv_raw_k = np.append(fv, np.array(rapv)[iend:] * KTS_PER_MPS)
                                prof_sr_rot = profile.create_profile(profile='default', pres=fpres, hght=fhght, tmpc=ftemp, dwpc=fdew, u=fu_k, v=fv_k, missing=-9999, strictQC=False)
                                bprof = profile.create_profile(profile='default', pres=fpres, hght=fhght-fhght[0], tmpc=ftemp, dwpc=fdew, u=fu_raw_k, v=fv_raw_k, missing=-9999, strictQC=False)
                                gui_profile = _neutral_sounding_record(
                                    fpres, fhght, ftemp, fdew,
                                    fu_raw_k * KNOTS_TO_MS, fv_raw_k * KNOTS_TO_MS,
                                    source_model=f"OBSERVED+{source_model}",
                                    u_storm=u_stm,
                                    v_storm=v_stm,
                                    wind_note="observed_with_model_extension_earth_relative",
                                )
                                merged_bunk_local[mode][key] = bunkers_from_raw(bprof)
                        else:
                            prof_sr_rot = profile.create_profile(profile='default', pres=fpres, hght=fhght, tmpc=ftemp, dwpc=fdew, u=rfu_sr * KTS_PER_MPS, v=rfv_sr * KTS_PER_MPS, missing=-9999, strictQC=False)
                            bprof = profile.create_profile(profile='default', pres=fpres, hght=fhght-fhght[0], tmpc=ftemp, dwpc=fdew, u=fu, v=fv, missing=-9999, strictQC=False)
                            gui_profile = _neutral_sounding_record(
                                fpres, fhght, ftemp, fdew,
                                fu * KNOTS_TO_MS, fv * KNOTS_TO_MS,
                                source_model="OBSERVED",
                                u_storm=u_stm,
                                v_storm=v_stm,
                                wind_note="observed_earth_relative",
                            )
                            merged_bunk_local[mode][key] = bunkers_from_raw(bprof)
            if not have_obs:
                prof_sr_rot, clat, clon, bunk_obs, raw_u_ms, raw_v_ms = make_RAP_RUC_profile(
                    atime, meso_tm, meso_la, meso_lo, levels, rank_rap_dir, u_stm, v_stm, Z,
                    proximity_sounding_dist=proximity_sounding_dist,
                    proximity_sounding_bearing=proximity_sounding_bearing,
                    sounding_distance=sounding_distance,
                    normalized_storm_motion=normalized_storm_motion,
                )
                merged_bunk_local[mode][key] = bunk_obs
                if prof_sr_rot is None:
                    prof_sr_rot = np.nan
                    clat, clon = np.nan, np.nan
                    gui_profile = np.nan
                else:
                    gui_profile = _neutral_record_from_profile(
                        prof_sr_rot,
                        raw_u_ms,
                        raw_v_ms,
                        source_model=source_model,
                        u_storm=u_stm,
                        v_storm=v_stm,
                        wind_note="model_earth_relative",
                    )
            merged_profs_local[key] = prof_sr_rot
            merged_gui_local[key] = gui_profile
            merged_lat_local[key] = clat
            merged_lon_local[key] = clon

    gathered_profiles = comm.gather((merged_profs_local, merged_lat_local, merged_lon_local, merged_gui_local), root=0)
    gathered_bunk = comm.gather(merged_bunk_local, root=0)
    if rank == 0:
        merged_profs, merged_lat, merged_lon, merged_gui = {}, {}, {}, {}
        merged_bunk = {mode: {} for mode in modes}
        for part in gathered_profiles:
            profs_part, lat_part, lon_part = part[0], part[1], part[2]
            gui_part = part[3] if len(part) > 3 else {}
            merged_profs.update(profs_part)
            merged_lat.update(lat_part)
            merged_lon.update(lon_part)
            merged_gui.update(gui_part)
        for bunk_part in gathered_bunk:
            for mode in modes:
                merged_bunk[mode].update(bunk_part.get(mode, {}))
        atomic_pickle_dump([merged_profs, merged_lat, merged_lon], rap_output)
        if gui_output:
            atomic_pickle_dump([merged_gui, merged_lat, merged_lon], gui_output)
        atomic_pickle_dump(merged_bunk, bunkers_output)


def compute_cape_cin_chunk(keys_chunk, profs_chunk, shear_layermin, shear_layermax, progress=False):
    nlay = len(shear_layermin)
    cape_dict, cin_dict = {}, {}
    tdd_mean_dict, qv_mean_dict = {}, {}
    shear_mag_dict, shear_dir_dict = {}, {}
    lapse_rate_dict, srh_dict = {}, {}
    iterate = tqdm(zip(keys_chunk, profs_chunk), total=len(keys_chunk)) if progress else zip(keys_chunk, profs_chunk)
    for key, prof in iterate:
        if getattr(prof, 'hght', None) is None or len(prof.hght) == 0:
            nanarr = np.full(nlay, np.nan)
            cape_dict[key] = cin_dict[key] = tdd_mean_dict[key] = qv_mean_dict[key] = nanarr.copy()
            shear_mag_dict[key] = shear_dir_dict[key] = lapse_rate_dict[key] = srh_dict[key] = nanarr.copy()
            continue
        pres = np.asarray(prof.pres)
        hgt = np.asarray(prof.hght)
        tmpc = np.asarray(prof.tmpc)
        dwpc = np.asarray(prof.dwpc)
        wspd = np.asarray(prof.wspd)
        wdir = np.asarray(prof.wdir)
        hgt_agl = hgt - hgt[0]
        acape = np.full(nlay, np.nan)
        acin = np.full(nlay, np.nan)
        tdd_mean = np.full(nlay, np.nan)
        qv_mean = np.full(nlay, np.nan)
        shear_mag = np.full(nlay, np.nan)
        shear_dir = np.full(nlay, np.nan)
        lapse_rate = np.full(nlay, np.nan)
        srh_layered = np.full(nlay, np.nan)
        for ish in range(nlay):
            try:
                z0 = shear_layermin[ish]
                z1 = shear_layermax[ish]
                iZ0 = int(np.argmin(np.abs(hgt_agl - z0)))
                iZ1 = int(np.argmin(np.abs(hgt_agl - z1)))
                if iZ1 <= iZ0:
                    continue
                aprof = profile.create_profile(profile='default', pres=pres[iZ0:iZ1], hght=hgt_agl[iZ0:iZ1], tmpc=tmpc[iZ0:iZ1], dwpc=dwpc[iZ0:iZ1], wspd=wspd[iZ0:iZ1], wdir=wdir[iZ0:iZ1], missing=-9999., strictQC=False)
                pcl = params.parcelx(aprof, flag=1)
                acape[ish], acin[ish] = pcl.bplus, pcl.bminus
                tdd_layer = tmpc[iZ0:iZ1] - dwpc[iZ0:iZ1]
                tdd_mean[ish] = np.nanmean(tdd_layer) if np.isfinite(tdd_layer).any() else np.nan
                qv_layer = np.array(aprof.wvmr)
                qv_mean[ish] = np.nanmean(qv_layer) if np.isfinite(qv_layer).any() else np.nan
                ub, vb = np.array(aprof.u)[0] * KNOTS_TO_MS, np.array(aprof.v)[0] * KNOTS_TO_MS
                ut, vt = np.array(aprof.u)[-1] * KNOTS_TO_MS, np.array(aprof.v)[-1] * KNOTS_TO_MS
                du, dv = (ut - ub), (vt - vb)
                shear_mag[ish] = np.hypot(du, dv)
                shear_dir[ish] = mpcalc.wind_direction(du * units('m/s'), dv * units('m/s')).m
                srh_layered[ish] = _srh(prof, z0, z1, 0, 0)
                dz_m = (hgt_agl[iZ1] - hgt_agl[iZ0])
                if dz_m > 0 and np.isfinite(tmpc[iZ0]) and np.isfinite(tmpc[iZ1]):
                    lapse_rate[ish] = (tmpc[iZ1] - tmpc[iZ0]) / (dz_m / 1000.0)
            except Exception:
                pass
        cape_dict[key] = acape
        cin_dict[key] = acin
        tdd_mean_dict[key] = tdd_mean
        qv_mean_dict[key] = qv_mean
        shear_mag_dict[key] = shear_mag
        shear_dir_dict[key] = shear_dir
        lapse_rate_dict[key] = lapse_rate
        srh_dict[key] = srh_layered
    return cape_dict, cin_dict, tdd_mean_dict, shear_mag_dict, shear_dir_dict, lapse_rate_dict, srh_dict, qv_mean_dict


def run_layer_integration(input_path, output_path, comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    levels = np.arange(0, 6500, 500)
    shear_layers = [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))]
    shear_layermin = [b for b, _ in shear_layers]
    shear_layermax = [t for _, t in shear_layers]
    if rank == 0:
        with open(input_path, 'rb') as f:
            sound, sound_lat, sound_lon = pickle.load(f)
        keys_all = np.array(list(sound.keys()))
        profs_all = np.array([sound[k] for k in keys_all], dtype=object)
        n = len(keys_all)
        counts = [n // size + (1 if i < (n % size) else 0) for i in range(size)]
        starts = np.cumsum([0] + counts[:-1])
        slices = [(starts[i], starts[i] + counts[i]) for i in range(size)]
        chunks_keys = [keys_all[s:e] for s, e in slices]
        chunks_profs = [profs_all[s:e] for s, e in slices]
    else:
        sound = sound_lat = sound_lon = None
        chunks_keys = chunks_profs = None
    my_keys = comm.scatter(chunks_keys, root=0)
    my_profs = comm.scatter(chunks_profs, root=0)
    all_parts = comm.gather(compute_cape_cin_chunk(my_keys, my_profs, shear_layermin, shear_layermax), root=0)
    if rank == 0:
        def merge_dicts(dicts):
            out = {}
            for d in dicts:
                out.update(d)
            return out
        cape_dict = merge_dicts([p[0] for p in all_parts])
        cin_dict = merge_dicts([p[1] for p in all_parts])
        tdd_dict = merge_dicts([p[2] for p in all_parts])
        shmag_dict = merge_dicts([p[3] for p in all_parts])
        shdir_dict = merge_dicts([p[4] for p in all_parts])
        lapse_dict = merge_dicts([p[5] for p in all_parts])
        srh_dict = merge_dicts([p[6] for p in all_parts])
        qv_dict = merge_dicts([p[7] for p in all_parts])
        nlay = len(shear_layermin)
        for k in sound.keys():
            for D in (cape_dict, cin_dict, tdd_dict, shmag_dict, shdir_dict, lapse_dict, srh_dict, qv_dict):
                if k not in D:
                    D[k] = np.full(nlay, np.nan)
        atomic_pickle_dump([sound, sound_lat, sound_lon, cape_dict, cin_dict, tdd_dict, shmag_dict, shdir_dict, lapse_dict, srh_dict, qv_dict], output_path)


def _count_valid_finished_profiles(finished_path):
    """Return simple validity counts for the RAP/RUC-finished sounding pickle."""
    with open(finished_path, 'rb') as f:
        profs, lats, lons = pickle.load(f)
    keys = list(profs.keys())
    n_total = len(keys)
    n_valid = sum(hasattr(profs[k], 'pres') for k in keys)
    n_finite_latlon = sum(
        np.isfinite(lats.get(k, np.nan)) and np.isfinite(lons.get(k, np.nan))
        for k in keys
    )
    return n_total, n_valid, n_finite_latlon


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cwd = Path(args.cwd).resolve()
    soundings_dir = Path(args.soundings_dir)
    if not soundings_dir.is_absolute():
        soundings_dir = (cwd / soundings_dir).resolve()
    raw_output = Path(args.raw_output)
    if not raw_output.is_absolute():
        raw_output = (cwd / raw_output).resolve()
    observed_output = Path(args.observed_output)
    if not observed_output.is_absolute():
        observed_output = (cwd / observed_output).resolve()
    finished_output = Path(args.finished_output)
    if not finished_output.is_absolute():
        finished_output = (cwd / finished_output).resolve()
    gui_output = Path(args.gui_output) if args.gui_output else None
    if gui_output is not None and not gui_output.is_absolute():
        gui_output = (cwd / gui_output).resolve()
    if gui_output is not None and gui_output == finished_output:
        raise ValueError(
            "--gui-output must differ from --finished-output. "
            "Use --finished-output soundings_RAP_use_this.pkl --gui-output soundings_RAP.pkl."
        )
    bunkers_output = Path(args.bunkers_output)
    if not bunkers_output.is_absolute():
        bunkers_output = (cwd / bunkers_output).resolve()
    layered_output = Path(args.layered_output)
    if not layered_output.is_absolute():
        layered_output = (cwd / layered_output).resolve()
    dir_transects = Path(args.dir_transects)
    if not dir_transects.is_absolute():
        dir_transects = (cwd / dir_transects).resolve()
    tracking_dir = Path(args.tracking_dir)
    if not tracking_dir.is_absolute():
        tracking_dir = (cwd / tracking_dir).resolve()
    elev_dir = Path(args.elev_dir)
    if not elev_dir.is_absolute():
        elev_dir = (cwd / elev_dir).resolve()
    mesonet_dir = Path(args.mesonet_dir)
    if not mesonet_dir.is_absolute():
        mesonet_dir = (cwd / mesonet_dir).resolve()
    rap_dir = Path(args.rap_dir)
    if not rap_dir.is_absolute():
        rap_dir = (cwd / rap_dir).resolve()

    if rank == 0:
        print(
            f'[INFO] MPI size={size} stage={args.stage} cwd={cwd}',
            flush=True,
        )

    # ------------------------------------------------------------------
    # RAW STAGE
    # ------------------------------------------------------------------
    # Raw sounding parsing is serial I/O. Only rank 0 should write
    # soundings_raw.pkl and soundings.pkl. Other ranks wait until the files
    # are fully written before entering the finish stage.
    if args.stage in ('raw', 'all'):
        if rank == 0:
            n = build_observed_soundings(
                str(soundings_dir),
                str(raw_output),
                str(observed_output),
                compute_eil=args.compute_eil,
            )
            print(f'wrote {raw_output}', flush=True)
            print(f'wrote {observed_output}', flush=True)
            print(f'profiles {n}', flush=True)
        comm.Barrier()

    # ------------------------------------------------------------------
    # FINISH STAGE
    # ------------------------------------------------------------------
    # This stage is MPI-parallel. All ranks read the completed observed
    # sounding pickle, divide the retained analysis windows, and rank 0
    # writes soundings_RAP.pkl and soundings_RAP_bunkers.pkl after gather.
    if args.stage in ('finish', 'all'):
        comm.Barrier()
        try:
            with open(observed_output, 'rb') as f:
                sound = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            raise RuntimeError(
                f'Failed to load observed sounding pickle {observed_output}. '
                f'The file is corrupt or only partially written. '
                f'Delete it and rerun this MPI-safe pipeline from scratch.'
            ) from e

        run_finish_soundings(
            sound=sound,
            dir_transects=str(dir_transects),
            tracking_dir=str(tracking_dir),
            rap_dir=str(rap_dir),
            rap_output=str(finished_output),
            bunkers_output=str(bunkers_output),
            dir_mesonet=str(mesonet_dir),
            elev_dir=str(elev_dir),
            gui_output=str(gui_output) if gui_output is not None else None,
            time_to_space=args.time_to_space,
            elevation_max_distance=args.elevation_max_distance,
            modes=args.modes,
            comm=comm,
        )
        comm.Barrier()
        if rank == 0:
            print(f'wrote {finished_output}', flush=True)
            if gui_output is not None:
                print(f'wrote {gui_output}', flush=True)
            print(f'wrote {bunkers_output}', flush=True)
            try:
                n_total, n_valid, n_finite_latlon = _count_valid_finished_profiles(finished_output)
                print(
                    f'[INFO] finished profile coverage: total={n_total} '
                    f'valid_profiles={n_valid} finite_latlon={n_finite_latlon}',
                    flush=True,
                )
                if n_total > 0 and n_valid == 0:
                    raise RuntimeError(
                        'Finish stage completed but produced zero valid SHARPpy profiles. '
                        'Stopping before layer integration so an all-NaN layered pickle is not written.'
                    )
            except Exception:
                # Re-raise on rank 0 and abort collectively below.
                raise
        comm.Barrier()

    # ------------------------------------------------------------------
    # LAYER STAGE
    # ------------------------------------------------------------------
    # This stage is also MPI-parallel. It must run only after rank 0 has
    # completely written the finished RAP/RUC pickle.
    if args.stage in ('layers', 'all'):
        comm.Barrier()
        run_layer_integration(str(finished_output), str(layered_output), comm=comm)
        comm.Barrier()
        if rank == 0:
            print(f'wrote {layered_output}', flush=True)


if __name__ == '__main__':
    main()
