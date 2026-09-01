#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_analysis_elevation_cache.py

Build a reusable analysis-time elevation cache for mobile mesonet workflows.

For each retained analysis time, this script:
  1. Finds processed mobile mesonet files for the analysis window.
  2. Computes the mobile-mesonet lat/lon footprint in that window.
  3. Computes the mean USGS 3DEP DEM elevation inside that footprint.
  4. Writes a pickle + CSV so synthesis/base-state scripts can read Z_ref
     without repeatedly opening large GeoTIFF files.

The intended Z_ref convention is the one settled for the mesonet/base-state
workflow:
  Z_ref = mean USGS 3DEP DEM elevation over the mobile-mesonet footprint
          for the finite-width analysis window.

This is not platform-mean elevation. Per-observation platform elevation remains
stored in processed mesonet NetCDFs and is used separately as elev_obs.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr
from netCDF4 import date2num

try:
    from mpi4py import MPI
except Exception:
    MPI = None

try:
    import rasterio
    from rasterio.warp import transform_bounds
except Exception as exc:
    rasterio = None
    transform_bounds = None

from utils import get_mm_files


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build analysis-time DEM elevation cache.")
    p.add_argument("--transects", type=Path, required=True, help="Path to transect_times.xlsx/csv.")
    p.add_argument("--mesonet-dir", type=Path, required=True, help="Processed mobile mesonet NetCDF directory.")
    p.add_argument("--elev-dir", type=Path, required=True, help="USGS 3DEP GeoTIFF directory.")
    p.add_argument("--out", type=Path, default=Path("analysis_elevation_cache.pkl"), help="Output pickle path.")
    p.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path. Default: same stem as --out plus .csv")
    p.add_argument("--modes", nargs="*", default=["T", "N"], help="Modes to process: T and/or N. Default: T N")
    p.add_argument("--years", type=int, nargs="*", default=None, help="Optional years to include, based on Tstart year.")
    p.add_argument("--case-id", nargs="*", default=None, help="Optional CaseID values to include, e.g., T51 N57.")
    p.add_argument("--analysis-start", type=int, default=-20, help="First offset in minutes.")
    p.add_argument("--analysis-end", type=int, default=20, help="Last offset in minutes.")
    p.add_argument("--analysis-step", type=int, default=5, help="Offset step in minutes.")
    p.add_argument("--time-to-space", type=int, default=5, help="Analysis window width in minutes.")
    p.add_argument("--key-format", choices=["time", "case_time"], default="time",
                   help="Key format. time=YYYYMMDDHHMM; case_time=CaseID_YYYYMMDDHHMM")
    p.add_argument("--no-download-dem", action="store_true", help="Do not download missing DEM tiles.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def atomic_pickle_dump(obj, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".pkl", dir=str(output_path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def parse_yyyymmdd_hhmm(value) -> datetime:
    if pd.isna(value):
        raise ValueError("missing timestamp")
    s = str(value).strip().replace(".0", "")
    return datetime.strptime(s, "%Y%m%d-%H%M")


def read_transects(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name="transect_times")
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ID", "CaseID", "Mode", "Tstart", "Observation Start", "Observation End"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing required columns: {sorted(missing)}")
    df["Tstart_dt"] = df["Tstart"].apply(parse_yyyymmdd_hhmm)
    df["obs_start_dt"] = df["Observation Start"].apply(parse_yyyymmdd_hhmm)
    df["obs_end_dt"] = df["Observation End"].apply(parse_yyyymmdd_hhmm)
    return df


def mode_filter(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode.upper().startswith("T"):
        return df.loc[df["Mode"].astype(str) == "T"].copy()
    return df.loc[df["Mode"].astype(str) == "N0"].copy()


def analysis_window(atime: datetime, time_to_space: int) -> Tuple[datetime, datetime]:
    half = timedelta(seconds=int((float(time_to_space) * 60.0) / 2.0))
    return atime - half, atime + half


def overlaps(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    return (a0 <= b1) and (a1 >= b0)


# -----------------------------------------------------------------------------
# DEM helpers
# -----------------------------------------------------------------------------

def _dem_tile_tokens(lat_floor: int, lon_floor: int) -> List[str]:
    """Return preferred USGS 3DEP tile tokens for a 1-degree cell.

    USGS 3DEP 1-degree tiles are usually named by the north/west corner for
    latitude and west/east corner for longitude, e.g. the cell 34-35N, 103-102W
    is named n35w103. Some local collections may include lower-edge names, so
    a lower-edge fallback is retained but never preferred.
    """
    def token(lat_val: int, lon_val: int) -> str:
        ns = "n" if lat_val >= 0 else "s"
        ew = "e" if lon_val >= 0 else "w"
        return f"{ns}{abs(lat_val):02d}{ew}{abs(lon_val):03d}"

    primary = token(lat_floor + 1, lon_floor)
    fallback = token(lat_floor, lon_floor)
    return [primary, fallback] if fallback != primary else [primary]


def _is_valid_dem_tile(tile_path: Path) -> bool:
    try:
        if (not tile_path.exists()) or tile_path.stat().st_size <= 0:
            return False
        with rasterio.open(tile_path) as src:
            _ = src.width, src.height, src.count, src.crs
        return True
    except Exception:
        return False


def _remove_bad_dem_tile(tile_path: Path, reason: str = "invalid") -> None:
    try:
        if tile_path.exists():
            tile_path.unlink()
            log(f"[WARN] removed {reason} DEM tile {tile_path}")
    except Exception as exc:
        log(f"[WARN] could not remove bad DEM tile {tile_path}: {exc}")


def _find_local_dem_tile(elev_dir: Path, lat_floor: int, lon_floor: int) -> Optional[Path]:
    for tok in _dem_tile_tokens(lat_floor, lon_floor):
        candidates = list(elev_dir.glob(f"*{tok}*.tif")) + list(elev_dir.glob(f"*{tok}*.tiff"))
        candidates = sorted([c for c in candidates if c.is_file() and c.stat().st_size > 0])
        for cand in candidates:
            if _is_valid_dem_tile(cand):
                return cand
            _remove_bad_dem_tile(cand, reason="corrupt/stale")
    return None


def _dem_tile_name(lat_floor: int, lon_floor: int) -> str:
    tok = _dem_tile_tokens(lat_floor, lon_floor)[0]
    return f"USGS_13_{tok}.tif"


def _tiles_for_bbox(lon_min, lat_min, lon_max, lat_max):
    lon0, lon1 = sorted([float(lon_min), float(lon_max)])
    lat0, lat1 = sorted([float(lat_min), float(lat_max)])
    # Include every one-degree cell intersecting the bbox. Use nextafter so
    # exact integer upper bounds do not accidentally include the next tile.
    lon_stop = np.nextafter(lon1, -np.inf)
    lat_stop = np.nextafter(lat1, -np.inf)
    lons = range(int(np.floor(lon0)), int(np.floor(lon_stop)) + 1)
    lats = range(int(np.floor(lat0)), int(np.floor(lat_stop)) + 1)
    return [(lat, lon) for lat in lats for lon in lons]


def _expand_degenerate_bbox(lon_min, lat_min, lon_max, lat_max, min_width_deg=0.002):
    lon0, lon1 = sorted([float(lon_min), float(lon_max)])
    lat0, lat1 = sorted([float(lat_min), float(lat_max)])
    if (lon1 - lon0) < min_width_deg:
        c = 0.5 * (lon0 + lon1)
        lon0, lon1 = c - min_width_deg / 2.0, c + min_width_deg / 2.0
    if (lat1 - lat0) < min_width_deg:
        c = 0.5 * (lat0 + lat1)
        lat0, lat1 = c - min_width_deg / 2.0, c + min_width_deg / 2.0
    return lon0, lat0, lon1, lat1


def _download_tnm_dem_tile(elev_dir: Path, lat_floor: int, lon_floor: int, timeout=120) -> Optional[Path]:
    elev_dir.mkdir(parents=True, exist_ok=True)
    local = _find_local_dem_tile(elev_dir, lat_floor, lon_floor)
    if local is not None:
        return local

    west, east = lon_floor, lon_floor + 1
    south, north = lat_floor, lat_floor + 1
    url = (
        "https://tnmaccess.nationalmap.gov/api/v1/products?"
        f"bbox={west},{south},{east},{north}"
        "&datasets=National%20Elevation%20Dataset%20(NED)%201/3%20arc-second"
        "&prodFormats=GeoTIFF"
    )
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as exc:
        log(f"[WARN] DEM TNM query failed for {lat_floor},{lon_floor}: {exc}")
        return None

    best_url = None
    for item in items:
        urls = []
        if isinstance(item.get("urls"), dict):
            urls.extend(item["urls"].values())
        for k in ("downloadURL", "url"):
            if item.get(k):
                urls.append(item[k])
        for u in urls:
            if u and ("tif" in u.lower()) and not u.lower().endswith(".zip"):
                best_url = u
                break
        if best_url:
            break
    if best_url is None:
        log(f"[WARN] no GeoTIFF DEM URL for {lat_floor},{lon_floor}")
        return None

    out = elev_dir / _dem_tile_name(lat_floor, lon_floor)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        with requests.get(best_url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
        if not _is_valid_dem_tile(tmp):
            _remove_bad_dem_tile(tmp, reason="invalid downloaded")
            log(f"[WARN] DEM download for {lat_floor},{lon_floor} did not produce a readable GeoTIFF")
            return None
        tmp.replace(out)
        log(f"[INFO] downloaded DEM tile {out}")
        return out
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        log(f"[WARN] DEM download failed for {lat_floor},{lon_floor}: {exc}")
        return None


def dem_mean_for_bbox(elev_dir: Path, lon_min, lat_min, lon_max, lat_max, *, download_missing=True):
    if rasterio is None:
        raise RuntimeError("rasterio is required for DEM-based elevation cache")

    lon0_orig, lat0_orig, lon1_orig, lat1_orig = sorted([float(lon_min), float(lon_max)])[0], sorted([float(lat_min), float(lat_max)])[0], sorted([float(lon_min), float(lon_max)])[1], sorted([float(lat_min), float(lat_max)])[1]
    lon0, lat0, lon1, lat1 = _expand_degenerate_bbox(lon_min, lat_min, lon_max, lat_max)

    vals = []
    files_used = []
    n_pixels = 0
    missing_tiles = []
    for lat_floor, lon_floor in _tiles_for_bbox(lon0, lat0, lon1, lat1):
        f = _find_local_dem_tile(elev_dir, lat_floor, lon_floor)
        if f is None and download_missing:
            f = _download_tnm_dem_tile(elev_dir, lat_floor, lon_floor)
        if f is None:
            missing_tiles.append((lat_floor, lon_floor))
            continue
        try:
            with rasterio.open(f) as src:
                if src.crs and src.crs.to_string().upper() not in {"EPSG:4326", "OGC:CRS84"}:
                    if transform_bounds is None:
                        continue
                    b = transform_bounds("EPSG:4326", src.crs, lon0, lat0, lon1, lat1, densify_pts=21)
                else:
                    b = (lon0, lat0, lon1, lat1)
                window = rasterio.windows.from_bounds(*b, transform=src.transform)
                arr = src.read(1, window=window, boundless=False, masked=True)
                if arr.size == 0:
                    continue
                data = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
                data = np.asarray(data, dtype=float)
                good = np.isfinite(data)
                if src.nodata is not None:
                    good &= data != float(src.nodata)
                data = data[good]
                if data.size:
                    vals.append(data)
                    n_pixels += int(data.size)
                    files_used.append(str(f))
        except Exception as exc:
            log(f"[WARN] failed reading DEM tile {f}: {exc}")
            _remove_bad_dem_tile(Path(f), reason="unreadable")
            continue

    if not vals:
        return np.nan, "", 0, sorted(set(missing_tiles)), (lon0, lat0, lon1, lat1)
    allv = np.concatenate(vals)
    return float(np.nanmean(allv)), ";".join(sorted(set(files_used))), int(n_pixels), sorted(set(missing_tiles)), (lon0, lat0, lon1, lat1)


# -----------------------------------------------------------------------------
# Mesonet footprint
# -----------------------------------------------------------------------------

def mesonet_bbox_for_time(mesonet_dir: Path, atime: datetime, *, time_to_space: int):
    try:
        mmfs = get_mm_files(str(mesonet_dir) + "/", atime)
    except Exception:
        return None, []
    if len(mmfs) == 0:
        return None, []

    w0, w1 = analysis_window(atime, time_to_space)
    tmin = date2num(w0, "seconds since 1970-01-01")
    tmax = date2num(w1, "seconds since 1970-01-01")
    lats, lons = [], []
    for fi in mmfs:
        try:
            with xr.open_dataset(fi, decode_times=False) as ds:
                if not {"time", "lat", "lon"}.issubset(ds.variables):
                    continue
                tm = np.asarray(ds["time"].values, dtype=float)
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


def build_tasks(df: pd.DataFrame, modes: List[str], years, case_ids, analysis_offsets, time_to_space: int):
    rows = []
    wanted_case = set(str(c) for c in case_ids) if case_ids is not None else None
    wanted_years = set(int(y) for y in years) if years is not None else None
    for mode in modes:
        sub = mode_filter(df, mode)
        for _, row in sub.iterrows():
            if wanted_case is not None and str(row["CaseID"]) not in wanted_case:
                continue
            if wanted_years is not None and int(row["Tstart_dt"].year) not in wanted_years:
                continue
            for off in analysis_offsets:
                atime = row["Tstart_dt"] + timedelta(minutes=int(off))
                w0, w1 = analysis_window(atime, time_to_space)
                if not overlaps(w0, w1, row["obs_start_dt"], row["obs_end_dt"]):
                    continue
                if mode.upper().startswith("T"):
                    outmode = "T"
                else:
                    outmode = "N"
                key_time = atime.strftime("%Y%m%d%H%M")
                rows.append({
                    "mode": outmode,
                    "case_id": str(row["CaseID"]),
                    "transect_id": str(row["ID"]),
                    "atime": atime,
                    "key_time": key_time,
                    "offset_min": int(off),
                    "obs_start": row["obs_start_dt"],
                    "obs_end": row["obs_end_dt"],
                    "analysis_start": w0,
                    "analysis_end": w1,
                })
    return rows


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out}. Use --overwrite.")

    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    df = read_transects(args.transects)
    offsets = np.arange(args.analysis_start, args.analysis_end + 1, args.analysis_step, dtype=int)
    tasks = build_tasks(df, args.modes, args.years, args.case_id, offsets, args.time_to_space)

    if rank == 0:
        log(f"[INFO] total retained analysis tasks: {len(tasks)}")
        log(f"[INFO] ranks: {size}")

    my_indices = np.arange(len(tasks))[rank::size]
    local_records = []
    local_by_key = {}

    iterator = my_indices
    for idx in iterator:
        t = tasks[int(idx)]
        atime = t["atime"]
        key = t["key_time"] if args.key_format == "time" else f"{t['case_id']}_{t['key_time']}"
        bbox, mmfs = mesonet_bbox_for_time(args.mesonet_dir, atime, time_to_space=args.time_to_space)
        status = "ok"
        if bbox is None:
            z_ref, files, npix, missing_tiles, expanded_bbox = np.nan, "", 0, [], None
            status = "missing_no_mesonet_bbox"
        else:
            lon_min, lat_min, lon_max, lat_max = bbox
            z_ref, files, npix, missing_tiles, expanded_bbox = dem_mean_for_bbox(
                args.elev_dir,
                lon_min, lat_min, lon_max, lat_max,
                download_missing=not args.no_download_dem,
            )
            if not np.isfinite(z_ref):
                status = "missing_dem_z"

        rec = {
            **t,
            "key": key,
            "bbox": bbox,
            "expanded_bbox": expanded_bbox,
            "z_ref_m": z_ref,
            "dem_files": files,
            "n_dem_pixels": int(npix),
            "missing_dem_tiles": str(missing_tiles),
            "n_mesonet_files": len(mmfs) if mmfs is not None else 0,
            "status": status,
        }
        local_records.append(rec)
        local_by_key[key] = rec
        if args.verbose:
            log(f"[rank {rank}] {key} {t['case_id']} {status} z={z_ref} npix={npix}")

    if comm is not None:
        gathered_records = comm.gather(local_records, root=0)
        gathered_maps = comm.gather(local_by_key, root=0)
    else:
        gathered_records = [local_records]
        gathered_maps = [local_by_key]

    if rank != 0:
        return

    records = [r for part in gathered_records for r in part]
    by_key: Dict[str, Dict] = {}
    by_mode: Dict[str, Dict[str, Dict]] = {"T": {}, "N": {}}
    for part in gathered_maps:
        by_key.update(part)
    for k, rec in by_key.items():
        by_mode.setdefault(rec["mode"], {})[k] = rec

    out_obj = {
        "meta": {
            "created_by": "build_analysis_elevation_cache.py",
            "transects": str(args.transects),
            "mesonet_dir": str(args.mesonet_dir),
            "elev_dir": str(args.elev_dir),
            "time_to_space_min": args.time_to_space,
            "analysis_offsets_min": list(map(int, offsets)),
            "z_ref_definition": "mean USGS 3DEP DEM elevation inside the mobile-mesonet lat/lon footprint for each analysis window",
        },
        "by_key": by_key,
        "by_mode": by_mode,
        "records": records,
    }
    atomic_pickle_dump(out_obj, args.out)
    log(f"[INFO] wrote {args.out}")

    csv_out = args.csv_out if args.csv_out is not None else args.out.with_suffix(".csv")
    dfout = pd.DataFrame(records)
    # Make timestamps CSV-friendly.
    for col in ["atime", "obs_start", "obs_end", "analysis_start", "analysis_end"]:
        if col in dfout.columns:
            dfout[col] = pd.to_datetime(dfout[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["bbox", "expanded_bbox"]:
        if col in dfout.columns:
            dfout[col] = dfout[col].apply(lambda x: "" if x is None else ",".join(f"{float(v):.8f}" for v in x))
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    dfout.to_csv(csv_out, index=False)
    log(f"[INFO] wrote {csv_out}")
    log("[INFO] status counts:\n" + dfout["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
