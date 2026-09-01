#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_base_state_surface_network.py

Compute storm-environment base states from standardized ASOS/ISD and Oklahoma
Mesonet surface observations.

This replaces make_base_state_using_ASOS.py. It keeps the Markowski-style logic
(case-level surface base state, superobbing, Barnes weighting, pressure reduced
to a representative regional elevation), but updates these pieces:

1. Reads merged surface_obs_YYYY.nc files created by download_qc_surface_obs.py.
2. Uses ASOS + Oklahoma Mesonet where available.
3. Removes convectively contaminated stations using nearest NEXRAD reflectivity
   where radar data are available.
4. Computes Z_ref from USGS 3DEP DEM over the mobile-mesonet footprint, downloading
   missing 1-degree DEM tiles as needed.
5. Uses observed station pressure for thermodynamics and DEM-reduced pressure only
   for pressure/base-state perturbation fields.

Outputs
-------
Rank pickles:
  base_state_rank{rank}_tsc{time_to_space}_{mode}.pkl
  case_base_state_rank{rank}_tsc{time_to_space}_{mode}.pkl
  base_state_summary_rank{rank}_tsc{time_to_space}_{mode}.csv

Optional combined outputs with --write-combined:
  base_state_data_tsc{time_to_space}_{mode}.pkl
  case_base_state_tsc{time_to_space}_{mode}.pkl
  base_state_summary_tsc{time_to_space}_{mode}.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
np.round_ = np.round
import pandas as pd
import requests
import xarray as xr
from netCDF4 import date2num, num2date

import metpy.calc as mpcalc
import metpy.constants as const
from metpy.units import units

from sklearn.cluster import DBSCAN

try:
    from mpi4py import MPI
except Exception:
    MPI = None

try:
    import nexradaws
except Exception:
    nexradaws = None

try:
    import pyart
    from pyart.io.nexrad_common import NEXRAD_LOCATIONS
    from pyart.core.transforms import geographic_to_cartesian_aeqd as geo2cart_aeqd
except Exception:
    pyart = None
    NEXRAD_LOCATIONS = {}
    geo2cart_aeqd = None

try:
    import rasterio
    from rasterio.warp import transform_bounds
except Exception:
    rasterio = None
    transform_bounds = None

from utils import get_dx_dy, get_mm_files, compute_thw, rotate_grid


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute ASOS/Oklahoma Mesonet-derived base states.")
    p.add_argument("--transects", type=Path, default=Path("./transect_times.xlsx"))
    p.add_argument("--storm-track-root", type=Path, default=Path("./storm_tracks"))
    p.add_argument("--surface-dir", type=Path, default=Path("./surface_obs"), help="Directory containing surface_obs_YYYY.nc.")
    p.add_argument("--dir-mesonet", type=Path, default=Path("./mobile_mesonet_processed"), help="Processed mobile mesonet NetCDF directory.")
    p.add_argument("--elev-dir", type=Path, default=Path("./elevation"), help="USGS 3DEP GeoTIFF directory.")
    p.add_argument("--radar-dir", type=Path, default=Path("./nexrad"), help="NEXRAD cache/root directory.")
    p.add_argument("--out-dir", type=Path, default=Path("./base_states"))

    p.add_argument("--modes", nargs="*", default=["T", "N"], help="Modes to process: T and/or N.")
    p.add_argument("--years", type=int, nargs="*", default=None)
    p.add_argument("--case-id", nargs="*", default=None)
    p.add_argument("--analysis-start", type=int, default=-20)
    p.add_argument("--analysis-end", type=int, default=20)
    p.add_argument("--analysis-step", type=int, default=5)

    p.add_argument("--surface-windows-min", type=int, nargs="*", default=[20, 30, 45, 60], help="Adaptive half-windows for surface stations.")
    p.add_argument("--min-surface-stations", type=int, default=5)
    p.add_argument("--surface-distance", type=float, default=400.0, help="Max station distance from mesocyclone in km.")
    p.add_argument("--superob-eps-km", type=float, default=30.0)
    p.add_argument("--time-to-space-conversion", type=int, default=5)
    p.add_argument("--normalized-storm-motion", type=float, default=90.0)

    p.add_argument("--radar-contam-radius-km", type=float, default=10.0)
    p.add_argument("--radar-ref-threshold-dbz", type=float, default=5.0, help="Stations with percentile reflectivity above this are removed.")
    p.add_argument("--radar-ref-percentile", type=float, default=75.0)
    p.add_argument("--radar-time-tolerance-min", type=float, default=10.0)
    p.add_argument("--skip-radar-contamination", action="store_true")
    p.add_argument("--no-radar-download", action="store_true", help="Use local radar files only.")

    p.add_argument("--key-format", choices=["time", "case_time"], default="time")
    p.add_argument("--write-combined", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# =============================================================================
# Generic utilities
# =============================================================================

def wmean(arr, w):
    arr = np.asarray(arr, dtype=float)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return np.nan
    return float(np.nansum(w[good] * arr[good]) / np.nansum(w[good]))


def parse_yyyymmdd_hhmm(value) -> datetime:
    if pd.isna(value):
        raise ValueError("missing timestamp")
    return datetime.strptime(str(value).strip(), "%Y%m%d-%H%M")


def mode_to_sheet_mode(mode: str) -> str:
    return "T" if mode.upper().startswith("T") else "N0"


def case_zero_index(case_id: str) -> Optional[int]:
    m = re.search(r"(\d+)", str(case_id))
    return int(m.group(1)) - 1 if m else None


def read_transects(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        tr = pd.read_excel(path, sheet_name="transect_times")
    else:
        tr = pd.read_csv(path)
    tr.columns = [str(c).strip() for c in tr.columns]
    tr["obs_start_dt"] = tr["Observation Start"].apply(parse_yyyymmdd_hhmm)
    tr["obs_end_dt"] = tr["Observation End"].apply(parse_yyyymmdd_hhmm)
    tr["Tstart_dt"] = tr["Tstart"].apply(parse_yyyymmdd_hhmm)
    return tr


def filter_transects(tr: pd.DataFrame, mode: str, years=None, case_ids=None) -> Tuple[pd.DataFrame, str]:
    if mode.upper().startswith("T"):
        out = tr.loc[tr["Mode"].astype(str) == "T"].copy()
        amode = "T"
    else:
        out = tr.loc[tr["Mode"].astype(str) == "N0"].copy()
        amode = "N"
    if years is not None:
        out = out.loc[out["obs_start_dt"].dt.year.isin(set(int(y) for y in years))].copy()
    if case_ids is not None:
        out = out.loc[out["CaseID"].astype(str).isin(set(str(x) for x in case_ids))].copy()
    return out.sort_values("Tstart_dt").reset_index(drop=True), amode


def read_track_file(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    time_col = next((cols[c] for c in ("datetime", "time", "time_utc", "date_time") if c in cols), None)
    lat_col = next((cols[c] for c in ("latitude", "lat", "mesocyclone_latitude") if c in cols), None)
    lon_col = next((cols[c] for c in ("longitude", "lon", "mesocyclone_longitude") if c in cols), None)
    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"Track file has unrecognized columns: {path} columns={df.columns.tolist()}")
    t = pd.to_datetime(df[time_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(float)
    good = (~pd.isna(t)).to_numpy() & np.isfinite(lat) & np.isfinite(lon)
    times = np.asarray([datetime(v.year, v.month, v.day, v.hour, v.minute, v.second) for v in t[good]], dtype=object)
    lat = lat[good]
    lon = lon[good]
    order = np.argsort(times)
    return times[order], lat[order], lon[order]


def find_track_file(row: pd.Series, track_root: Path, amode: str, verbose: bool = False) -> Path:
    subdir = track_root / ("tornadic" if amode == "T" else "nontornadic")
    if not subdir.exists():
        raise FileNotFoundError(f"Missing storm-track directory: {subdir}")
    idx0 = case_zero_index(row["CaseID"])
    date_hint = row["Tstart_dt"].strftime("%Y%m%d")
    time_hint = row["Tstart_dt"].strftime("%Y%m%d%H%M")
    candidates: List[Path] = []
    if idx0 is not None:
        candidates += sorted(subdir.glob(f"{idx0}_*"))
        candidates += sorted(subdir.glob(f"{idx0 + 1}_*"))
    candidates += sorted(subdir.glob(f"*{time_hint}*"))
    candidates += sorted(subdir.glob(f"*{date_hint}*"))
    seen, uniq = set(), []
    for c in candidates:
        if c in seen or not c.is_file() or c.name.startswith((".", "._")):
            continue
        seen.add(c)
        uniq.append(c)
    for c in uniq:
        try:
            times, lat, lon = read_track_file(c)
            if len(times) == 0:
                continue
            if max(times) >= row["obs_start_dt"] - timedelta(hours=2) and min(times) <= row["obs_end_dt"] + timedelta(hours=2):
                if verbose:
                    log(f"  {row['CaseID']} track: {c}")
                return c
        except Exception as exc:
            if verbose:
                log(f"  rejected track {c}: {exc}")
    raise FileNotFoundError(f"Could not find storm-track file for {row['CaseID']} in {subdir}")


def _time_seconds(times):
    return np.array([(t - datetime(1970, 1, 1)).total_seconds() for t in times], dtype=float)


def interp_track_at_time(track_times, track_lat, track_lon, atime: datetime):
    secs = _time_seconds(track_times)
    target = (atime - datetime(1970, 1, 1)).total_seconds()
    order = np.argsort(secs)
    secs = secs[order]
    lat = np.asarray(track_lat, float)[order]
    lon = np.asarray(track_lon, float)[order]
    secs, keep = np.unique(secs, return_index=True)
    lat = lat[keep]
    lon = lon[keep]
    if target <= secs.min():
        return float(lat[0]), float(lon[0])
    if target >= secs.max():
        return float(lat[-1]), float(lon[-1])
    return float(np.interp(target, secs, lat)), float(np.interp(target, secs, lon))


def storm_motion_around_time(track_times, track_lat, track_lon, atime: datetime, half_window_min: float):
    t0 = atime - timedelta(minutes=half_window_min)
    t1 = atime + timedelta(minutes=half_window_min)
    lat0, lon0 = interp_track_at_time(track_times, track_lat, track_lon, t0)
    lat1, lon1 = interp_track_at_time(track_times, track_lat, track_lon, t1)
    dx, dy = get_dx_dy(lon0, lat0, lon1, lat1)
    dt = max((t1 - t0).total_seconds(), 1.0)
    return dx / dt, dy / dt


# =============================================================================
# DEM/TNM reference elevation helpers
# =============================================================================

def _dem_tile_tokens(lat_floor: int, lon_floor: int) -> List[str]:
    """Return plausible USGS 3DEP tile tokens for a 1-degree cell.

    USGS 3DEP files for the conterminous U.S. commonly use the *north*
    latitude edge and west longitude edge in the filename. For example, a
    raster covering 34--35 N and 103--102 W is named n35w103, not n34w103.
    Older/local files can vary, so keep the old floor-based token as a
    fallback.
    """
    ew = "e" if lon_floor >= 0 else "w"
    lon_label = abs(int(lon_floor))

    tokens = []
    if lat_floor >= 0:
        # Cell [lat_floor, lat_floor + 1] in the northern hemisphere.
        tokens.append(f"n{abs(int(lat_floor + 1)):02d}{ew}{lon_label:03d}")
        tokens.append(f"n{abs(int(lat_floor)):02d}{ew}{lon_label:03d}")
    else:
        # Defensive southern-hemisphere support; not used for this project.
        tokens.append(f"s{abs(int(lat_floor)):02d}{ew}{lon_label:03d}")
        tokens.append(f"s{abs(int(lat_floor + 1)):02d}{ew}{lon_label:03d}")

    # Preserve order but remove duplicates.
    out = []
    for t in tokens:
        if t not in out:
            out.append(t)
    return out


def _dem_tile_name(lat_floor: int, lon_floor: int) -> str:
    token = _dem_tile_tokens(lat_floor, lon_floor)[0]
    return f"USGS_13_{token}.tif"


def _find_local_dem_tile(elev_dir: Path, lat_floor: int, lon_floor: int) -> Optional[Path]:
    """Find the best local DEM tile for a 1-degree cell.

    Prefer the USGS 3DEP north-edge naming convention returned first by
    _dem_tile_tokens(), e.g., n35w103 for the 34--35 N, 103--102 W cell.
    Do not sort all candidate tokens together because n34... sorts before
    n35... alphabetically and can select the adjacent southern tile.
    """
    for token in _dem_tile_tokens(lat_floor, lon_floor):
        candidates = []
        candidates.extend(elev_dir.glob(f"*{token}*.tif"))
        candidates.extend(elev_dir.glob(f"*{token}*.tiff"))
        candidates = sorted({c for c in candidates if c.exists() and c.stat().st_size > 0})
        if candidates:
            return candidates[0]
    return None


def _download_tnm_dem_tile(elev_dir: Path, lat_floor: int, lon_floor: int, timeout=120) -> Optional[Path]:
    elev_dir.mkdir(parents=True, exist_ok=True)
    local = _find_local_dem_tile(elev_dir, lat_floor, lon_floor)
    if local is not None and local.exists() and local.stat().st_size > 0:
        return local

    # TNM 1/3 arc-sec naming is not perfectly stable across vintages, so query TNM API by bbox.
    west, east = lon_floor, lon_floor + 1
    south, north = lat_floor, lat_floor + 1
    url = (
        "https://tnmaccess.nationalmap.gov/api/v1/products?"
        f"bbox={west},{south},{east},{north}&datasets=National%20Elevation%20Dataset%20(NED)%201/3%20arc-second&prodFormats=GeoTIFF"
    )
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as exc:
        log(f"[WARN] DEM TNM query failed for tile lat={lat_floor} lon={lon_floor}: {exc}")
        return None

    if not items:
        log(f"[WARN] DEM TNM query returned no items for tile lat={lat_floor} lon={lon_floor}")
        return None

    # Prefer direct download URLs containing .tif/.zip; TNM sometimes returns zip links.
    best_url = None
    title = ""
    for item in items:
        urls = []
        if isinstance(item.get("urls"), dict):
            urls.extend(item["urls"].values())
        for k in ("downloadURL", "url"):
            if item.get(k):
                urls.append(item[k])
        for u in urls:
            if u and ("tif" in u.lower() or "zip" in u.lower()):
                best_url = u
                title = str(item.get("title", ""))
                break
        if best_url:
            break

    if best_url is None:
        log(f"[WARN] no usable DEM download URL for tile lat={lat_floor} lon={lon_floor}")
        return None

    suffix = ".zip" if best_url.lower().endswith(".zip") else ".tif"
    out = elev_dir / (_dem_tile_name(lat_floor, lon_floor).replace(".tif", suffix))
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        with requests.get(best_url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.replace(out)
        log(f"[INFO] downloaded DEM tile {out.name} from TNM")
        if out.suffix.lower() == ".zip":
            # Avoid adding zip extraction dependency. Users can pre-unzip if TNM only serves zip.
            log(f"[WARN] DEM tile downloaded as zip; unzip needed before raster use: {out}")
            return None
        return out if out.exists() else None
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        log(f"[WARN] DEM download failed for tile lat={lat_floor} lon={lon_floor}: {exc}")
        return None


def _tiles_for_bbox(lon_min, lat_min, lon_max, lat_max):
    lons = range(int(np.floor(lon_min)), int(np.floor(lon_max)) + 1)
    lats = range(int(np.floor(lat_min)), int(np.floor(lat_max)) + 1)
    return [(lat, lon) for lat in lats for lon in lons]


def _expand_degenerate_bbox(lon_min, lat_min, lon_max, lat_max, min_size_deg=0.01):
    """Expand very small DEM query boxes so rasterio reads at least a few cells.

    Mobile mesonet footprints can collapse to an almost point-like box when only
    one platform contributes in an analysis window. A zero/sub-pixel raster
    window returns no DEM cells even when the correct tile exists. This expansion
    is only a DEM sampling safeguard; normal larger footprints are unchanged.
    """
    lon_min = float(lon_min)
    lat_min = float(lat_min)
    lon_max = float(lon_max)
    lat_max = float(lat_max)

    if lon_max < lon_min:
        lon_min, lon_max = lon_max, lon_min
    if lat_max < lat_min:
        lat_min, lat_max = lat_max, lat_min

    cx = 0.5 * (lon_min + lon_max)
    cy = 0.5 * (lat_min + lat_max)
    width = lon_max - lon_min
    height = lat_max - lat_min

    if width < float(min_size_deg):
        half = 0.5 * float(min_size_deg)
        lon_min, lon_max = cx - half, cx + half
    if height < float(min_size_deg):
        half = 0.5 * float(min_size_deg)
        lat_min, lat_max = cy - half, cy + half

    return lon_min, lat_min, lon_max, lat_max


def dem_mean_for_bbox(elev_dir: Path, lon_min, lat_min, lon_max, lat_max, *, download_missing=True):
    if rasterio is None:
        raise RuntimeError("rasterio is required for DEM-based Z_ref")

    # Expand point-like/single-platform footprints before tile lookup and raster
    # reading. Without this, rasterio.windows.from_bounds can yield a zero-size
    # window even when the matching 3DEP tile exists. Larger footprints are
    # unchanged.
    lon_min, lat_min, lon_max, lat_max = _expand_degenerate_bbox(
        lon_min, lat_min, lon_max, lat_max, min_size_deg=0.01
    )

    files_used = []
    vals = []
    n_pixels = 0
    for lat_floor, lon_floor in _tiles_for_bbox(lon_min, lat_min, lon_max, lat_max):
        f = _find_local_dem_tile(elev_dir, lat_floor, lon_floor)
        if f is None and download_missing:
            f = _download_tnm_dem_tile(elev_dir, lat_floor, lon_floor)
        if f is None:
            continue
        try:
            with rasterio.open(f) as src:
                # Transform bbox from lon/lat to raster CRS if needed.
                if src.crs and src.crs.to_string().upper() not in {"EPSG:4326", "OGC:CRS84"}:
                    if transform_bounds is None:
                        continue
                    b = transform_bounds("EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max, densify_pts=21)
                else:
                    b = (lon_min, lat_min, lon_max, lat_max)
                window = rasterio.windows.from_bounds(*b, transform=src.transform)
                arr = src.read(1, window=window, boundless=False, masked=True)
                if arr.size == 0:
                    continue
                data = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
                nodata = src.nodata
                data = data.astype(float)
                good = np.isfinite(data)
                if nodata is not None:
                    good &= data != float(nodata)
                data = data[good]
                if data.size:
                    vals.append(data)
                    n_pixels += int(data.size)
                    files_used.append(str(f))
        except Exception as exc:
            log(f"[WARN] failed reading DEM tile {f}: {exc}")
            continue
    if not vals:
        return np.nan, "", 0
    allv = np.concatenate(vals)
    return float(np.nanmean(allv)), ";".join(sorted(set(files_used))), n_pixels


def mesonet_bbox_for_time(dir_mesonet: Path, atime: datetime, *, time_to_space_conversion: int):
    try:
        mmfs = get_mm_files(str(dir_mesonet) + "/", atime)
    except Exception:
        return None, []
    if len(mmfs) == 0:
        return None, []
    half = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    tmin = date2num(atime - half, "seconds since 1970-01-01")
    tmax = date2num(atime + half, "seconds since 1970-01-01")
    lats, lons = [], []
    for fi in mmfs:
        try:
            with xr.open_dataset(fi, decode_times=False) as ds:
                if "time" not in ds or "lat" not in ds or "lon" not in ds:
                    continue
                tm = ds["time"].values
                mask = (tm >= tmin) & (tm <= tmax)
                if not np.any(mask):
                    continue
                lat = np.asarray(ds["lat"].values[mask], dtype=float)
                lon = np.asarray(ds["lon"].values[mask], dtype=float)
                lon = np.where(lon > 0, -lon, lon)
                good = np.isfinite(lat) & np.isfinite(lon)
                if np.any(good):
                    lats.append(lat[good])
                    lons.append(lon[good])
        except Exception:
            continue
    if not lats:
        return None, mmfs
    lat_all = np.concatenate(lats)
    lon_all = np.concatenate(lons)
    if lat_all.size == 0:
        return None, mmfs
    return (float(np.nanmin(lon_all)), float(np.nanmin(lat_all)), float(np.nanmax(lon_all)), float(np.nanmax(lat_all))), mmfs


# =============================================================================
# Surface selection and base state
# =============================================================================

def open_surface_year(surface_dir: Path, year: int):
    path = surface_dir / f"surface_obs_{year}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing surface observation NetCDF: {path}")
    return xr.open_dataset(path, decode_times=False)


def select_nearest_surface_obs(ds: xr.Dataset, center_time: datetime, meso_lat: float, meso_lon: float, *, half_window_min: float, surface_distance: float) -> pd.DataFrame:
    times = pd.to_datetime(ds["time"].values, unit="s", origin="unix")
    station_ids = ds["station_id"].values.astype(str)
    networks = ds["network"].values.astype(str) if "network" in ds else np.array(["UNKNOWN"] * len(station_ids))
    tmin = pd.Timestamp(center_time - timedelta(minutes=half_window_min))
    tmax = pd.Timestamp(center_time + timedelta(minutes=half_window_min))
    mask = (times >= tmin) & (times <= tmax)
    if not np.any(mask):
        return pd.DataFrame()
    idx_all = np.where(mask)[0]
    rows = []
    # Unique by network + station.
    keys = sorted(set(zip(networks[idx_all], station_ids[idx_all])))
    for net, st in keys:
        idx = idx_all[(networks[idx_all] == net) & (station_ids[idx_all] == st)]
        if len(idx) == 0:
            continue
        dt = np.abs((times[idx] - pd.Timestamp(center_time)).total_seconds())
        ii = idx[int(np.argmin(dt))]
        lat = float(ds.latitude.values[ii])
        lon = float(ds.longitude.values[ii])
        dx, dy = get_dx_dy(meso_lon, meso_lat, lon, lat)
        dist = float(np.hypot(dx, dy) / 1000.0)
        if dist > surface_distance:
            continue
        row = {
            "index": int(ii), "time": times[ii].to_pydatetime(), "network": str(net), "station_id": str(st),
            "lat": lat, "lon": lon, "elev": float(ds.elev.values[ii]),
            "temp": float(ds.temperature.values[ii]), "dew": float(ds.dewpoint.values[ii]) if "dewpoint" in ds else np.nan,
            "rh": float(ds.rh.values[ii]), "pres": float(ds.pressure.values[ii]),
            "wspd": float(ds.wspd.values[ii]), "wdir": float(ds.wdir.values[ii]),
            "dist_km": dist, "dx_km": dx / 1000.0, "dy_km": dy / 1000.0,
            "abs_time_offset_min": float(dt.min() / 60.0),
        }
        if "pressure_source" in ds:
            row["pressure_source"] = str(ds.pressure_source.values[ii])
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    good = (
        df["temp"].between(-50, 55) & df["rh"].between(0, 100) & df["pres"].between(700, 1050) &
        df["wspd"].between(0, 75) & df["wdir"].between(0, 360) & np.isfinite(df["lat"]) & np.isfinite(df["lon"]) & np.isfinite(df["elev"])
    )
    return df.loc[good].reset_index(drop=True)


# =============================================================================
# NEXRAD contamination filtering
# =============================================================================

def _extract_radar_id_from_path(path_value):
    base = os.path.basename(str(path_value))
    if len(base) >= 4 and base[:4].upper().startswith("K"):
        return base[:4].upper()
    for part in base.replace("-", "_").split("_"):
        pu = part.upper()
        if len(pu) == 4 and pu.startswith("K"):
            return pu
    return None


def _radar_file_datetime(fname):
    base = os.path.basename(str(fname))
    parts = base.replace("-", "_").split("_")
    for p in range(len(parts) - 1):
        d, t = parts[p], parts[p + 1]
        if len(d) == 8 and len(t) == 6 and d.isdigit() and t.isdigit():
            return datetime.strptime(d + t, "%Y%m%d%H%M%S")
    try:
        d = base.split("_")[0][-8:]
        t = base.split("_")[1][:6]
        return datetime.strptime(d + t, "%Y%m%d%H%M%S")
    except Exception:
        return None


def _bucket_list_radar_day(radar_id, day_key, timeout=60):
    prefix = f"{day_key[:4]}/{day_key[4:6]}/{day_key[6:8]}/{radar_id}/"
    urls = [
        f"https://unidata-nexrad-level2.s3.amazonaws.com?prefix={prefix}",
        f"https://noaa-nexrad-level2.s3.amazonaws.com?prefix={prefix}",
    ]
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
                dt = _radar_file_datetime(os.path.basename(key))
                if dt is not None:
                    scans.append((key, dt.replace(tzinfo=timezone.utc)))
            return sorted(scans, key=lambda x: x[1])
        except Exception:
            continue
    return []


def _download_radar_volume(scan_key, radar_dir: Path, timeout=180) -> Optional[Path]:
    out = radar_dir / scan_key
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://unidata-nexrad-level2.s3.amazonaws.com/{scan_key}",
        f"https://noaa-nexrad-level2.s3.amazonaws.com/{scan_key}",
    ]
    for url in urls:
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out)
            return out
        except Exception:
            tmp.unlink(missing_ok=True)
            continue
    return None


def _local_radar_files(radar_dir: Path, radar_id: str, atime: datetime):
    day_key = atime.strftime("%Y%m%d")
    files = [p for p in radar_dir.glob("**/*") if p.is_file()]
    out = []
    for f in files:
        if _extract_radar_id_from_path(f) != radar_id:
            continue
        dt = _radar_file_datetime(f)
        if dt is not None and dt.strftime("%Y%m%d") == day_key:
            out.append((f, dt.replace(tzinfo=timezone.utc)))
    return sorted(out, key=lambda x: x[1])


def _nearest_radar_ids(lat, lon, n=4):
    ids = []
    for rid, meta in NEXRAD_LOCATIONS.items():
        try:
            d = np.hypot(float(meta["lat"]) - lat, (float(meta["lon"]) - lon) * np.cos(np.deg2rad(lat)))
            ids.append((d, rid))
        except Exception:
            pass
    return [rid for _, rid in sorted(ids)[:n]]


def radar_reflectivity_near_station(lat, lon, atime: datetime, radar_dir: Path, *, radius_km=10.0, percentile=75.0, time_tolerance_min=10.0, allow_download=True) -> Tuple[float, str]:
    if pyart is None or geo2cart_aeqd is None or not NEXRAD_LOCATIONS:
        return np.nan, "no_pyart"
    atime_utc = atime.replace(tzinfo=timezone.utc)
    for rid in _nearest_radar_ids(lat, lon, n=4):
        candidates = _local_radar_files(radar_dir, rid, atime)
        if not candidates and allow_download:
            scans = _bucket_list_radar_day(rid, atime.strftime("%Y%m%d"))
            if scans:
                # download only scans near the desired time.
                for key, dt in scans:
                    if abs((dt - atime_utc).total_seconds()) <= time_tolerance_min * 60:
                        path = _download_radar_volume(key, radar_dir)
                        if path is not None:
                            candidates.append((path, dt))
        if not candidates:
            continue
        # nearest volume
        path, dt = min(candidates, key=lambda x: abs((x[1] - atime_utc).total_seconds()))
        if abs((dt - atime_utc).total_seconds()) > time_tolerance_min * 60:
            continue
        try:
            radar = pyart.io.read_nexrad_archive(str(path), delay_field_loading=True)
        except Exception:
            continue
        try:
            # lowest reflectivity-like sweep.
            field = next((f for f in ("reflectivity", "DBZ", "REF", "CZ") if f in radar.fields), None)
            if field is None:
                continue
            # choose sweep nearest 0.5 deg
            sweeps = []
            for si in range(radar.nsweeps):
                istart = radar.sweep_start_ray_index["data"][si]
                iend = radar.sweep_end_ray_index["data"][si] + 1
                elev = np.asarray(radar.elevation["data"][istart:iend], dtype=float)
                sweeps.append((abs(np.nanmean(elev) - 0.5), si))
            si = sorted(sweeps)[0][1]
            ref = radar.get_field(si, field)
            gx, gy, _gz = radar.get_gate_x_y_z(si)
            rlon = float(radar.longitude["data"][0])
            rlat = float(radar.latitude["data"][0])
            px, py = geo2cart_aeqd(np.array([lon]), np.array([lat]), rlon, rlat)
            dist = np.hypot(gx.ravel() - px[0], gy.ravel() - py[0])
            vals = np.asanyarray(ref).filled(np.nan).ravel() if np.ma.isMaskedArray(ref) else np.asarray(ref).ravel()
            vals = vals[(dist <= radius_km * 1000.0) & np.isfinite(vals)]
            vals = vals[(vals > -32) & (vals < 100)]
            if vals.size == 0:
                return np.nan, f"no_gates:{rid}:{path.name}"
            return float(np.nanpercentile(vals, percentile)), f"ok:{rid}:{path.name}"
        finally:
            try:
                del radar
            except Exception:
                pass
    return np.nan, "no_radar"


def filter_convective_contamination(obs: pd.DataFrame, args) -> pd.DataFrame:
    if obs.empty or args.skip_radar_contamination:
        obs = obs.copy()
        obs["radar_ref_dbz"] = np.nan
        obs["radar_status"] = "skipped"
        obs["convective_contaminated"] = False
        return obs
    rows = []
    for _, row in obs.iterrows():
        dbz, status = radar_reflectivity_near_station(
            row["lat"], row["lon"], row["time"], args.radar_dir,
            radius_km=args.radar_contam_radius_km,
            percentile=args.radar_ref_percentile,
            time_tolerance_min=args.radar_time_tolerance_min,
            allow_download=not args.no_radar_download,
        )
        rr = row.to_dict()
        rr["radar_ref_dbz"] = dbz
        rr["radar_status"] = status
        rr["convective_contaminated"] = bool(np.isfinite(dbz) and dbz >= args.radar_ref_threshold_dbz)
        rows.append(rr)
    out = pd.DataFrame(rows)
    return out.loc[~out["convective_contaminated"]].reset_index(drop=True)


def compute_base_from_surface_obs(obs: pd.DataFrame, *, meso_lat: float, meso_lon: float, Z: float, surface_distance: float, superob_eps_km: float, Rd: float, g: float):
    if obs.empty:
        return None
    if not np.isfinite(Z):
        return None
    wspd = obs["wspd"].to_numpy(float)
    wdir = obs["wdir"].to_numpy(float)
    u, v = mpcalc.wind_components(wspd * units("m/s"), wdir * units.degrees)
    u, v = u.m, v.m
    temp = obs["temp"].to_numpy(float)
    rh = obs["rh"].to_numpy(float)
    pres = obs["pres"].to_numpy(float)
    elev = obs["elev"].to_numpy(float)
    dew = mpcalc.dewpoint_from_relative_humidity(temp * units.degC, rh * units.percent).m
    qv = mpcalc.mixing_ratio_from_relative_humidity(pres * units.hPa, temp * units.degC, rh * units.percent).m
    tv = mpcalc.virtual_temperature(temp * units.degC, qv * units.dimensionless).m
    # Pressure correction only; thermodynamics below stay on observed pressure.
    p_corr = pres * np.exp(((elev - Z) / (Rd * (tv + 273.15))) * g)
    th = mpcalc.potential_temperature(pres * units.hPa, temp * units.degC).m
    thv = mpcalc.virtual_potential_temperature(pres * units.hPa, temp * units.degC, qv * units.dimensionless).m
    the = mpcalc.equivalent_potential_temperature(pres * units.hPa, temp * units.degC, dew * units.degC).m
    thw = np.array([compute_thw(temp[i], pres[i], rh[i]) for i in range(len(temp))])
    dx = obs["dx_km"].to_numpy(float)
    dy = obs["dy_km"].to_numpy(float)
    X = np.column_stack([dx, dy])
    labels = DBSCAN(eps=float(superob_eps_km), min_samples=1).fit(X).labels_
    labs = np.unique(labels)
    vars_in = {"pres": pres, "pres_corr": p_corr, "temp": temp, "rh": rh, "u": u, "v": v, "qv": qv, "dew": dew, "th": th, "thv": thv, "the": the, "thw": thw, "elev": elev, "dx": dx, "dy": dy}
    super_vars = {k: [] for k in vars_in}
    for lab in labs:
        ii = np.where(labels == lab)[0]
        for k, arr in vars_in.items():
            super_vars[k].append(np.nanmean(arr[ii]))
    super_vars = {k: np.asarray(v, float) for k, v in super_vars.items()}
    n_super = len(super_vars["dx"])
    if n_super == 0:
        return None
    r_km = np.hypot(super_vars["dx"], super_vars["dy"])
    R = float(surface_distance)
    Dn = np.sqrt(np.pi * R * R / n_super)
    k0 = (2.0 * Dn / np.pi) ** 2
    weights = np.exp(-(r_km ** 2) / (2.0 * k0))
    base = {k: wmean(v, weights) for k, v in super_vars.items() if k not in {"dx", "dy", "elev"}}
    base.update({
        "n_surface_obs": int(len(obs)),
        "n_surface_stations": int(obs[["network", "station_id"]].drop_duplicates().shape[0]),
        "n_superobs": int(n_super),
        "Z": float(Z),
        "median_station_time_offset_min": float(np.nanmedian(obs["abs_time_offset_min"].to_numpy(float))),
        "max_station_time_offset_min": float(np.nanmax(obs["abs_time_offset_min"].to_numpy(float))),
        "n_radar_filtered": int(obs.get("convective_contaminated", pd.Series([], dtype=bool)).sum()) if "convective_contaminated" in obs else 0,
        "n_networks": int(obs["network"].nunique()),
        "networks": ";".join(sorted(obs["network"].unique())),
    })
    return base


def compute_case_base_state(row: pd.Series, track_times, track_lat, track_lon, ds_surface, args: argparse.Namespace, *, Rd: float, g: float):
    center = row["Tstart_dt"]
    meso_lat, meso_lon = interp_track_at_time(track_times, track_lat, track_lon, center)

    bbox, mmfs = mesonet_bbox_for_time(args.dir_mesonet, center, time_to_space_conversion=args.time_to_space_conversion)
    if bbox is None:
        Z, z_files, z_npixels = np.nan, "", 0
        bbox_str = ""
    else:
        lon_min, lat_min, lon_max, lat_max = bbox
        Z, z_files, z_npixels = dem_mean_for_bbox(args.elev_dir, lon_min, lat_min, lon_max, lat_max, download_missing=True)
        bbox_str = f"{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}"

    tried = []
    for win in args.surface_windows_min:
        obs = select_nearest_surface_obs(ds_surface, center, meso_lat, meso_lon, half_window_min=float(win), surface_distance=args.surface_distance)
        obs0_count = len(obs)
        if not obs.empty:
            obs = filter_convective_contamination(obs, args)
        tried.append((win, obs0_count, len(obs), obs[["network", "station_id"]].drop_duplicates().shape[0] if not obs.empty else 0))
        if (not obs.empty) and obs[["network", "station_id"]].drop_duplicates().shape[0] >= args.min_surface_stations:
            base = compute_base_from_surface_obs(obs, meso_lat=meso_lat, meso_lon=meso_lon, Z=Z, surface_distance=args.surface_distance, superob_eps_km=args.superob_eps_km, Rd=Rd, g=g)
            if base is not None:
                base.update({
                    "case_id": str(row["CaseID"]), "mode": str(row["Mode"]), "base_center_time": center,
                    "meso_lat": meso_lat, "meso_lon": meso_lon, "surface_window_minutes_used": int(win),
                    "surface_windows_tried": str(tried), "status": "ok", "Z_ref_method": "dem_mobile_mesonet_bbox",
                    "Z_ref_bbox": bbox_str, "Z_ref_dem_files": z_files, "Z_ref_n_dem_pixels": int(z_npixels),
                })
                return base

    # Weak fallback if any observations remain in largest window.
    win = int(args.surface_windows_min[-1])
    obs = select_nearest_surface_obs(ds_surface, center, meso_lat, meso_lon, half_window_min=float(win), surface_distance=args.surface_distance)
    obs0_count = len(obs)
    if not obs.empty:
        obs = filter_convective_contamination(obs, args)
    if not obs.empty:
        base = compute_base_from_surface_obs(obs, meso_lat=meso_lat, meso_lon=meso_lon, Z=Z, surface_distance=args.surface_distance, superob_eps_km=args.superob_eps_km, Rd=Rd, g=g)
        if base is not None:
            base.update({
                "case_id": str(row["CaseID"]), "mode": str(row["Mode"]), "base_center_time": center,
                "meso_lat": meso_lat, "meso_lon": meso_lon, "surface_window_minutes_used": int(win),
                "surface_windows_tried": str(tried), "status": f"weak_n_stations_below_{args.min_surface_stations}",
                "Z_ref_method": "dem_mobile_mesonet_bbox", "Z_ref_bbox": bbox_str, "Z_ref_dem_files": z_files,
                "Z_ref_n_dem_pixels": int(z_npixels),
            })
            return base

    return {"case_id": str(row["CaseID"]), "mode": str(row["Mode"]), "base_center_time": center, "meso_lat": meso_lat, "meso_lon": meso_lon, "surface_windows_tried": str(tried), "status": "missing_no_surface_base", "Z": Z, "Z_ref_bbox": bbox_str, "Z_ref_dem_files": z_files, "Z_ref_n_dem_pixels": int(z_npixels)}


def build_output_for_analysis_time(base: Dict, atime: datetime, track_times, track_lat, track_lon, args: argparse.Namespace):
    if base is None or base.get("status", "").startswith("missing"):
        return np.nan
    required = ["pres", "pres_corr", "temp", "rh", "u", "v", "qv", "dew", "th", "thv", "the", "thw"]
    if any(k not in base or not np.isfinite(base[k]) for k in required):
        return np.nan
    half = args.time_to_space_conversion / 2.0
    u_storm, v_storm = storm_motion_around_time(track_times, track_lat, track_lon, atime, half)
    try:
        ru, rv = rotate_grid(base["u"], base["v"], u_storm, v_storm, pivot_x=0, pivot_y=0, target_angle=args.normalized_storm_motion)
    except Exception:
        ru, rv = np.nan, np.nan
    out = {k: base[k] for k in required}
    out.update({
        "time": atime, "case_id": base.get("case_id"), "mode": base.get("mode"), "base_center_time": base.get("base_center_time"),
        "status": base.get("status"), "ru": ru, "rv": rv, "n_surface_obs": base.get("n_surface_obs", np.nan),
        "n_surface_stations": base.get("n_surface_stations", np.nan), "n_superobs": base.get("n_superobs", np.nan),
        "surface_window_minutes_used": base.get("surface_window_minutes_used", np.nan),
        "median_station_time_offset_min": base.get("median_station_time_offset_min", np.nan),
        "max_station_time_offset_min": base.get("max_station_time_offset_min", np.nan),
        "Z": base.get("Z", np.nan), "Z_ref_method": base.get("Z_ref_method", ""), "Z_ref_bbox": base.get("Z_ref_bbox", ""),
        "Z_ref_dem_files": base.get("Z_ref_dem_files", ""), "Z_ref_n_dem_pixels": base.get("Z_ref_n_dem_pixels", np.nan),
        "meso_lat": base.get("meso_lat", np.nan), "meso_lon": base.get("meso_lon", np.nan),
        "networks": base.get("networks", ""), "n_networks": base.get("n_networks", np.nan),
    })
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.elev_dir.mkdir(parents=True, exist_ok=True)
    args.radar_dir.mkdir(parents=True, exist_ok=True)
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm, rank, size = None, 0, 1
    Rd = float(np.asarray(const.dry_air_gas_constant))
    g = float(np.asarray(const.earth_gravity))
    tr_all = read_transects(args.transects)
    case_filter = set(str(c) for c in args.case_id) if args.case_id is not None else None

    for mode in args.modes:
        tr, amode = filter_transects(tr_all, mode, years=args.years, case_ids=case_filter)
        if tr.empty:
            if rank == 0:
                log(f"No cases selected for mode={mode}")
            continue
        offsets = np.arange(args.analysis_start, args.analysis_end + 1, args.analysis_step)
        case_records = []
        for _, row in tr.iterrows():
            tf = find_track_file(row, args.storm_track_root, amode, verbose=(args.verbose and rank == 0))
            tt, tl, to = read_track_file(tf)
            case_records.append({"row": row, "track_file": tf, "track_times": tt, "track_lat": tl, "track_lon": to})
        my_indices = np.array_split(np.arange(len(case_records)), size)[rank]
        local_base_by_case, local_output, summary_rows = {}, {}, []
        for ci in my_indices:
            rec = case_records[int(ci)]
            row = rec["row"]
            cid = str(row["CaseID"])
            year = int(row["Tstart_dt"].year)
            if args.verbose:
                log(f"[rank {rank}] Processing {cid} ({year})")
            try:
                ds = open_surface_year(args.surface_dir, year)
                base = compute_case_base_state(row, rec["track_times"], rec["track_lat"], rec["track_lon"], ds, args, Rd=Rd, g=g)
                ds.close()
            except Exception as exc:
                base = {"case_id": cid, "mode": str(row["Mode"]), "base_center_time": row["Tstart_dt"], "status": f"error:{type(exc).__name__}:{exc}"}
            local_base_by_case[cid] = base
            summary_rows.append({
                "case_id": cid, "mode": str(row["Mode"]), "year": year, "base_center_time": row["Tstart_dt"],
                "status": base.get("status", "unknown") if isinstance(base, dict) else "not_dict",
                "n_surface_stations": base.get("n_surface_stations", np.nan) if isinstance(base, dict) else np.nan,
                "n_superobs": base.get("n_superobs", np.nan) if isinstance(base, dict) else np.nan,
                "surface_window_minutes_used": base.get("surface_window_minutes_used", np.nan) if isinstance(base, dict) else np.nan,
                "median_station_time_offset_min": base.get("median_station_time_offset_min", np.nan) if isinstance(base, dict) else np.nan,
                "max_station_time_offset_min": base.get("max_station_time_offset_min", np.nan) if isinstance(base, dict) else np.nan,
                "Z": base.get("Z", np.nan) if isinstance(base, dict) else np.nan,
                "Z_ref_bbox": base.get("Z_ref_bbox", "") if isinstance(base, dict) else "",
                "Z_ref_n_dem_pixels": base.get("Z_ref_n_dem_pixels", np.nan) if isinstance(base, dict) else np.nan,
                "networks": base.get("networks", "") if isinstance(base, dict) else "",
                "track_file": str(rec["track_file"]),
            })
            for off in offsets:
                atime = row["Tstart_dt"] + timedelta(minutes=int(off))
                case_data = build_output_for_analysis_time(base, atime, rec["track_times"], rec["track_lat"], rec["track_lon"], args)
                key = f"{cid}_{atime:%Y%m%d%H%M}" if args.key_format == "case_time" else atime.strftime("%Y%m%d%H%M")
                local_output[key] = case_data
        rank_pickle = args.out_dir / f"base_state_rank{rank}_tsc{args.time_to_space_conversion}_{mode}.pkl"
        with open(rank_pickle, "wb") as f:
            pickle.dump(local_output, f)
        case_pickle = args.out_dir / f"case_base_state_rank{rank}_tsc{args.time_to_space_conversion}_{mode}.pkl"
        with open(case_pickle, "wb") as f:
            pickle.dump(local_base_by_case, f)
        summary_path = args.out_dir / f"base_state_summary_rank{rank}_tsc{args.time_to_space_conversion}_{mode}.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        if comm is not None:
            comm.Barrier()
        if rank == 0 and args.write_combined:
            combined, combined_case, summaries = {}, {}, []
            for rp in sorted(args.out_dir.glob(f"base_state_rank*_tsc{args.time_to_space_conversion}_{mode}.pkl")):
                with open(rp, "rb") as f:
                    combined.update(pickle.load(f))
            for cp in sorted(args.out_dir.glob(f"case_base_state_rank*_tsc{args.time_to_space_conversion}_{mode}.pkl")):
                with open(cp, "rb") as f:
                    combined_case.update(pickle.load(f))
            for sp in sorted(args.out_dir.glob(f"base_state_summary_rank*_tsc{args.time_to_space_conversion}_{mode}.csv")):
                try:
                    summaries.append(pd.read_csv(sp))
                except Exception:
                    pass
            with open(args.out_dir / f"base_state_data_tsc{args.time_to_space_conversion}_{mode}.pkl", "wb") as f:
                pickle.dump(combined, f)
            with open(args.out_dir / f"case_base_state_tsc{args.time_to_space_conversion}_{mode}.pkl", "wb") as f:
                pickle.dump(combined_case, f)
            if summaries:
                pd.concat(summaries, ignore_index=True).to_csv(args.out_dir / f"base_state_summary_tsc{args.time_to_space_conversion}_{mode}.csv", index=False)
            log(f"Wrote combined base-state outputs for mode={mode}: {len(combined)} analysis keys")
        if comm is not None:
            comm.Barrier()


if __name__ == "__main__":
    main()
