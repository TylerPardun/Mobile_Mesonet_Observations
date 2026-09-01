#!/usr/bin/env python3
"""Build per-case elevation overlay PNGs and cursor lookup grids.

Outputs are written into each case bundle:

    elevation/elevation.png
    elevation/elevation_bounds.json
    elevation/elevation_grid.json

The PNG is a colorized/hillshaded DEM image that MapLibre can use as an
ImageSource.  The grid JSON is intentionally smaller and is used by the browser
for cursor elevation readout without calling back into Python on every mousemove.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
except Exception as exc:  # pragma: no cover - environment failure path
    raise SystemExit(f"rasterio is required: {exc}")

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Pillow is required: {exc}")

try:
    from matplotlib import colormaps
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")

from src.io import load_config  # noqa: E402
from scripts._script_utils import bundle_root_from_config, data_root_from_config, resolve_data_path  # noqa: E402

try:
    from build_analysis_elevation_cache import (  # type: ignore
        _download_tnm_dem_tile,
        _find_local_dem_tile,
        _tiles_for_bbox,
    )
except Exception:  # pragma: no cover - fallback keeps this script standalone
    _download_tnm_dem_tile = None
    _find_local_dem_tile = None
    _tiles_for_bbox = None


def _to_float(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _read_case_bounds(case_dir: Path, pad_km: float) -> Optional[dict]:
    lats: list[float] = []
    lons: list[float] = []

    meta_path = case_dir / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            bbox = meta.get("bbox") if isinstance(meta, dict) else None
            if isinstance(bbox, dict):
                for key in ("min_lat", "max_lat"):
                    v = _to_float(bbox.get(key))
                    if v is not None:
                        lats.append(v)
                for key in ("min_lon", "max_lon"):
                    v = _to_float(bbox.get(key))
                    if v is not None:
                        lons.append(v)
            for lat_key, lon_key in (("center_lat", "center_lon"), ("begin_lat", "begin_lon")):
                lat = _to_float(meta.get(lat_key))
                lon = _to_float(meta.get(lon_key))
                if lat is not None and lon is not None:
                    lats.append(lat)
                    lons.append(lon)
        except Exception:
            pass

    for name in ("mobile_mesonet.parquet", "storm_track.parquet", "surface_obs.parquet"):
        path = case_dir / name
        if not path.is_file():
            continue
        try:
            df = pd.read_parquet(path, columns=None)
        except Exception:
            continue
        if df.empty or "lat" not in df.columns or "lon" not in df.columns:
            continue
        lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(lat) & np.isfinite(lon)
        if mask.any():
            lats.extend([float(np.nanmin(lat[mask])), float(np.nanmax(lat[mask]))])
            lons.extend([float(np.nanmin(lon[mask])), float(np.nanmax(lon[mask]))])

    if not lats or not lons:
        return None

    min_lat, max_lat = float(np.nanmin(lats)), float(np.nanmax(lats))
    min_lon, max_lon = float(np.nanmin(lons)), float(np.nanmax(lons))
    cen_lat = 0.5 * (min_lat + max_lat)
    pad_lat = float(pad_km) / 111.0
    pad_lon = float(pad_km) / max(111.0 * math.cos(math.radians(cen_lat)), 10.0)
    return {
        "min_lon": min_lon - pad_lon,
        "min_lat": min_lat - pad_lat,
        "max_lon": max_lon + pad_lon,
        "max_lat": max_lat + pad_lat,
    }


def _find_tiles(elev_dir: Path, bounds: dict, *, download_missing: bool) -> list[Path]:
    if _tiles_for_bbox is None or _find_local_dem_tile is None:
        raise RuntimeError("build_analysis_elevation_cache helpers are required for DEM tile lookup")

    out: list[Path] = []
    for lat_floor, lon_floor in _tiles_for_bbox(
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"],
    ):
        tile = _find_local_dem_tile(elev_dir, lat_floor, lon_floor)
        if tile is None and download_missing and _download_tnm_dem_tile is not None:
            tile = _download_tnm_dem_tile(elev_dir, lat_floor, lon_floor)
        if tile is not None and tile.is_file() and tile.stat().st_size > 0:
            out.append(tile)
    return sorted(set(out))


def _read_dem(bounds: dict, elev_tiles: list[Path], width: int, height: int) -> np.ndarray:
    dst_transform = from_bounds(
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"],
        width, height,
    )
    dest = np.full((height, width), np.nan, dtype=np.float32)
    for tile in elev_tiles:
        with rasterio.open(tile) as src:
            tmp = np.full((height, width), np.nan, dtype=np.float32)
            src_nodata = src.nodata
            reproject(
                source=rasterio.band(src, 1),
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            mask = np.isfinite(tmp)
            dest[mask] = tmp[mask]
    return dest


def _hillshade(z: np.ndarray) -> np.ndarray:
    filled = z.copy()
    finite = np.isfinite(filled)
    if not finite.any():
        return np.zeros_like(filled, dtype=np.float32)
    median = float(np.nanmedian(filled))
    filled[~finite] = median
    gy, gx = np.gradient(filled)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = math.radians(315.0)
    alt = math.radians(45.0)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = (shade - np.nanmin(shade)) / max(float(np.nanmax(shade) - np.nanmin(shade)), 1e-6)
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def _write_png(z: np.ndarray, out_png: Path, *, alpha: float, cmap_name: str) -> tuple[float, float]:
    finite = np.isfinite(z)
    if not finite.any():
        raise RuntimeError("DEM output has no finite elevation values")
    z_min = float(np.nanpercentile(z[finite], 2.0))
    z_max = float(np.nanpercentile(z[finite], 98.0))
    if z_min >= z_max:
        z_min -= 1.0
        z_max += 1.0
    norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    rgba = colormaps.get_cmap(cmap_name)(norm)
    shade = _hillshade(z)
    rgb = rgba[..., :3] * (0.55 + 0.45 * shade[..., None])
    out = np.zeros((*z.shape, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(finite, int(np.clip(alpha, 0.0, 1.0) * 255), 0).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGBA").save(out_png)
    return z_min, z_max


def _resample_grid(z: np.ndarray, width: int, height: int) -> np.ndarray:
    img = Image.fromarray(np.where(np.isfinite(z), z, np.nan).astype(np.float32), mode="F")
    img = img.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=".tmp_", suffix=path.suffix) as f:
        f.write(text)
        tmp = Path(f.name)
    tmp.replace(path)


def build_case(case_dir: Path, elev_dir: Path, args: argparse.Namespace) -> bool:
    out_dir = case_dir / "elevation"
    png_path = out_dir / "elevation.png"
    bounds_path = out_dir / "elevation_bounds.json"
    grid_path = out_dir / "elevation_grid.json"
    if png_path.exists() and bounds_path.exists() and grid_path.exists() and not args.force:
        print(f"SKIP {case_dir.name}: elevation overlay exists")
        return True

    bounds = _read_case_bounds(case_dir, args.pad_km)
    if bounds is None:
        print(f"FAIL {case_dir.name}: no usable case bounds")
        return False

    tiles = _find_tiles(elev_dir, bounds, download_missing=not args.no_download_dem)
    if not tiles:
        print(f"FAIL {case_dir.name}: no DEM tiles found for bounds {bounds}")
        return False

    z = _read_dem(bounds, tiles, args.size, args.size)
    finite = np.isfinite(z)
    if not finite.any():
        print(f"FAIL {case_dir.name}: DEM tiles produced no finite values")
        return False

    z_min, z_max = _write_png(z, png_path, alpha=args.alpha, cmap_name=args.cmap)
    grid = _resample_grid(z, args.grid_size, args.grid_size)
    grid_values = [
        None if not np.isfinite(v) else round(float(v), 1)
        for v in grid.reshape(-1).tolist()
    ]
    bounds_payload = {
        **bounds,
        "png": "elevation.png",
        "grid": "elevation_grid.json",
        "z_min_m": z_min,
        "z_max_m": z_max,
        "width": int(args.size),
        "height": int(args.size),
        "dem_files": [str(p) for p in tiles],
    }
    grid_payload = {
        **bounds,
        "width": int(args.grid_size),
        "height": int(args.grid_size),
        "units": "m",
        "values": grid_values,
    }
    _atomic_write_text(bounds_path, json.dumps(bounds_payload, indent=2))
    _atomic_write_text(grid_path, json.dumps(grid_payload, separators=(",", ":")))
    print(f"OK   {case_dir.name}: {png_path}  z={z_min:.1f}..{z_max:.1f} m")
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "local_config.yaml"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--bundle-root", default=None)
    ap.add_argument("--elev-dir", default=None, help="Directory of USGS 3DEP/TNM GeoTIFF tiles")
    ap.add_argument("--cases", default=None, help="Comma-separated case IDs. Default: all bundle dirs.")
    ap.add_argument("--size", type=int, default=1024, help="PNG width/height in pixels")
    ap.add_argument("--grid-size", type=int, default=256, help="Cursor lookup grid width/height")
    ap.add_argument("--pad-km", type=float, default=35.0, help="Padding around case data bounds")
    ap.add_argument("--alpha", type=float, default=0.58, help="PNG alpha 0..1")
    ap.add_argument("--cmap", default="terrain", help="Matplotlib colormap")
    ap.add_argument("--no-download-dem", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = data_root_from_config(cfg, args.data_root)
    bundle_root = Path(args.bundle_root).expanduser().resolve() if args.bundle_root else bundle_root_from_config(cfg)
    elev_dir = (
        Path(args.elev_dir).expanduser().resolve()
        if args.elev_dir
        else resolve_data_path(cfg, "elevation_dir", "elevation", data_root=data_root)
    )

    if args.cases:
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_ids = sorted(p.name for p in bundle_root.iterdir() if p.is_dir())

    ok = 0
    fail = 0
    print(f"bundle_root: {bundle_root}")
    print(f"elev_dir   : {elev_dir}")
    for case_id in case_ids:
        if build_case(bundle_root / case_id, elev_dir, args):
            ok += 1
        else:
            fail += 1
    print(f"\ncomplete: ok={ok} fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
