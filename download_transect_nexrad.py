#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_transect_nexrad_2025.py

Download NEXRAD Level II files for 2025 mobile-mesonet transects listed in an
Excel/CSV transect table.

Expected spreadsheet columns:
    CaseID              folder/case label, e.g., T_0, N_12
    Observation Start   exact format YYYYMMDD-HHMM, e.g., 20250404-1936
    Observation End     exact format YYYYMMDD-HHMM, e.g., 20250404-1952
    Radar               single radar ID, e.g., KSHV
    Mode                optional if CaseID begins with T or N

Default directory assumption:
    script lives in: phd/python/
    output lives in: phd/data/nexrad/

Default output structure:
    ../data/nexrad/T/<CaseID>/<NEXRAD files>
    ../data/nexrad/N/<CaseID>/<NEXRAD files>

Examples:
    CaseID=T_0  -> ../data/nexrad/T/T_0/<NEXRAD files>
    CaseID=N_12 -> ../data/nexrad/N/N_12/<NEXRAD files>

The original NEXRAD Level II filenames are preserved for GR2Analyst use.

Examples:
    python download_transect_nexrad_2025.py \
        --transects ../data/transect_times.xlsx \
        --out-root ../data/nexrad \
        --year 2025 \
        --pad-minutes 10

    python download_transect_nexrad_2025.py \
        --transects transect_times.xlsx \
        --out-root ../data/nexrad \
        --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

try:
    import s3fs
except ImportError as e:
    raise SystemExit(
        "s3fs is required. Install with: mamba install -c conda-forge s3fs"
    ) from e


@dataclass(frozen=True)
class Transect:
    row_number: int
    transect_id: str
    mode: str
    group: str
    start: datetime
    end: datetime
    radar: str
    out_dir: Path


BAD_SUBSTRINGS = ("MDM", ".tar")


def parse_yyyymmdd_hhmm(value) -> datetime:
    """Parse exact transect timestamp strings like 20250404-1936.

    The spreadsheet already contains the full date for both Observation Start
    and Observation End. Do not infer midnight crossings. If a transect crosses
    00Z, Observation End must explicitly use the next YYYYMMDD date.
    """
    if pd.isna(value):
        raise ValueError("missing datetime value")

    s = str(value).strip()

    # Excel sometimes reads text-looking values as numeric if a column is edited.
    # Clean only the accidental .0 suffix; otherwise enforce YYYYMMDD-HHMM.
    if s.endswith(".0"):
        s = s[:-2]

    try:
        return datetime.strptime(s, "%Y%m%d-%H%M")
    except ValueError as exc:
        raise ValueError(
            f"Could not parse datetime value {value!r}; expected exact format YYYYMMDD-HHMM"
        ) from exc


def normalize_window(start_raw, end_raw) -> Tuple[datetime, datetime]:
    """Parse an observation window without guessing date rollover.

    Observation Start and Observation End must both be complete timestamps in
    YYYYMMDD-HHMM format. If end <= start, the spreadsheet is inconsistent
    and the script raises an error instead of adding a day.
    """
    start = parse_yyyymmdd_hhmm(start_raw)
    end = parse_yyyymmdd_hhmm(end_raw)

    if end <= start:
        raise ValueError(
            f"Observation End ({end:%Y%m%d-%H%M}) must be later than "
            f"Observation Start ({start:%Y%m%d-%H%M}). "
            "If the transect crossed 00Z, use the next calendar date in Observation End."
        )

    return start, end


def parse_radar(value) -> str:
    if pd.isna(value):
        raise ValueError("missing radar")
    radar = str(value).strip().upper()
    if not radar:
        raise ValueError("empty radar")
    if not radar.startswith("K") or len(radar) != 4:
        # Keep this as a warning-worthy value but don't hard fail; some radars
        # could be non-CONUS or have odd formatting.
        radar = radar[:4]
    return radar


def mode_group(mode: str) -> str:
    m = str(mode).strip().upper()
    if m.startswith("T"):
        return "T"
    if m.startswith("N"):
        return "N"
    return "UNKNOWN"


def parse_case_id(value) -> str:
    """Return the CaseID exactly as a folder-safe string."""
    if pd.isna(value):
        raise ValueError("missing CaseID")

    case_id = str(value).strip()

    # Excel can occasionally read integer-like IDs as floats. This keeps
    # labels such as T_0/N_12 unchanged while cleaning accidental .0 suffixes.
    if case_id.endswith(".0"):
        case_id = case_id[:-2]

    if not case_id:
        raise ValueError("empty CaseID")

    # Keep the case label itself intact, but prevent accidental nested paths.
    case_id = case_id.replace("/", "_").replace("\\", "_")
    return case_id


def group_from_case_id(case_id: str, mode: str = "") -> str:
    """Return top-level T/N group from CaseID, falling back to Mode."""
    cid = str(case_id).strip().upper()
    if cid.startswith("T"):
        return "T"
    if cid.startswith("N"):
        return "N"

    # Fallback for unexpected CaseID values.
    return mode_group(mode)


def make_out_dir(out_root: Path, group: str, case_id: str) -> Path:
    """Build output directory using the CaseID folder name.

    Examples:
        CaseID=T_0  -> out_root/T/T_0
        CaseID=N_12 -> out_root/N/N_12
    """
    return out_root / group / case_id


def read_transects(
    transects_file: Path,
    out_root: Path,
    year: int = 2025,
) -> List[Transect]:
    if not transects_file.exists():
        raise FileNotFoundError(f"Missing transect file: {transects_file}")

    suffix = transects_file.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(transects_file)
    elif suffix == ".csv":
        df = pd.read_csv(transects_file)
    else:
        raise ValueError("Transect file must be .xlsx, .xls, or .csv")

    required = ["CaseID", "Observation Start", "Observation End", "Radar"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    has_mode = "Mode" in df.columns
    transects: List[Transect] = []

    for idx, row in df.iterrows():
        try:
            start, end = normalize_window(row["Observation Start"], row["Observation End"])
        except Exception as e:
            raise ValueError(f"Row {int(idx) + 2}: invalid observation window: {e}") from e

        if start.year != year and end.year != year:
            continue

        transect_id = parse_case_id(row["CaseID"])
        mode = str(row["Mode"]).strip().upper() if has_mode and not pd.isna(row["Mode"]) else ""
        group = group_from_case_id(transect_id, mode)
        radar = parse_radar(row["Radar"])

        out_dir = make_out_dir(out_root, group, transect_id)

        transects.append(
            Transect(
                row_number=int(idx) + 2,  # +2 because Excel is 1-based and row 1 is header
                transect_id=transect_id,
                mode=mode,
                group=group,
                start=start,
                end=end,
                radar=radar,
                out_dir=out_dir,
            )
        )

    return transects


def parse_nexrad_time_from_key(key: str, radar: str) -> Optional[datetime]:
    """Parse NEXRAD Level II object time from S3 key or filename."""
    name = Path(key).name
    if any(bad in name for bad in BAD_SUBSTRINGS):
        return None

    try:
        suffix = name.split(radar, 1)[1]
        date_part = suffix.split("_")[0]
        time_part = suffix.split("_")[1][:6]
        return datetime.strptime(f"{date_part} {time_part}", "%Y%m%d %H%M%S")
    except Exception:
        return None


def list_s3_keys_for_day(fs: s3fs.S3FileSystem, radar: str, day: datetime) -> List[str]:
    bucket = "unidata-nexrad-level2"
    prefix = f"{bucket}/{day:%Y/%m/%d}/{radar}"

    try:
        keys = fs.ls(prefix)
    except FileNotFoundError:
        return []
    except Exception as e:
        # Some s3fs versions raise generic exceptions for absent prefixes.
        msg = str(e)
        if prefix in msg or "No such file" in msg or "not found" in msg.lower():
            return []
        raise

    return [k for k in keys if not any(bad in Path(k).name for bad in BAD_SUBSTRINGS)]


def list_scan_keys(
    fs: s3fs.S3FileSystem,
    radar: str,
    start: datetime,
    end: datetime,
    pad_minutes: int = 10,
) -> List[Tuple[datetime, str]]:
    window_start = start - timedelta(minutes=pad_minutes)
    window_end = end + timedelta(minutes=pad_minutes)

    days = []
    d = datetime(window_start.year, window_start.month, window_start.day)
    last_day = datetime(window_end.year, window_end.month, window_end.day)
    while d <= last_day:
        days.append(d)
        d += timedelta(days=1)

    scans: List[Tuple[datetime, str]] = []
    for day in days:
        for key in list_s3_keys_for_day(fs, radar, day):
            t = parse_nexrad_time_from_key(key, radar)
            if t is None:
                continue
            if window_start <= t <= window_end:
                scans.append((t, key))

    scans = sorted(set(scans), key=lambda x: x[0])
    return scans


def download_key(fs: s3fs.S3FileSystem, key: str, out_dir: Path, overwrite: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / Path(key).name

    if outfile.exists() and outfile.stat().st_size > 0 and not overwrite:
        return outfile

    tmpfile = outfile.with_suffix(outfile.suffix + ".part")
    if tmpfile.exists():
        tmpfile.unlink()

    fs.get(key, str(tmpfile))

    if not tmpfile.exists() or tmpfile.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is missing/empty: {tmpfile}")

    shutil.move(str(tmpfile), str(outfile))
    return outfile


def write_manifest(manifest_rows: List[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "download_manifest_2025.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download NEXRAD Level II files for 2025 mobile-mesonet transect windows."
    )
    parser.add_argument("--transects", type=Path, required=True, help="Path to transect_times.xlsx/csv")
    parser.add_argument("--out-root", type=Path, default=Path("../data/nexrad"), help="Output root, default ../data/nexrad")
    parser.add_argument("--year", type=int, default=2025, help="Year to download, default 2025")
    parser.add_argument("--pad-minutes", type=int, default=10, help="Minutes before/after window to include, default 10")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files even if already present")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded without downloading")
    parser.add_argument("--max-transects", type=int, default=None, help="Limit transects for testing")
    args = parser.parse_args()

    transects = read_transects(
        args.transects,
        out_root=args.out_root,
        year=args.year,
    )

    if args.max_transects is not None:
        transects = transects[: args.max_transects]

    if not transects:
        raise SystemExit(f"No transects found for year={args.year}")

    print(f"Found {len(transects)} transects for {args.year}")
    print(f"Output root: {args.out_root.resolve()}")

    fs = s3fs.S3FileSystem(anon=True)
    manifest_rows: List[dict] = []

    for i, tr in enumerate(transects, start=1):
        print("-" * 100)
        print(
            f"[{i}/{len(transects)}] {tr.transect_id} mode={tr.mode} group={tr.group} "
            f"radar={tr.radar} {tr.start:%Y-%m-%d %H:%M} to {tr.end:%Y-%m-%d %H:%M}"
        )
        print(f"Output: {tr.out_dir}")

        try:
            scans = list_scan_keys(fs, tr.radar, tr.start, tr.end, pad_minutes=args.pad_minutes)
        except Exception as e:
            print(f"ERROR listing scans: {type(e).__name__}: {e}")
            manifest_rows.append({
                "transect_id": tr.transect_id,
                "row_number": tr.row_number,
                "mode": tr.mode,
                "group": tr.group,
                "radar": tr.radar,
                "start": tr.start,
                "end": tr.end,
                "out_dir": str(tr.out_dir),
                "scan_time": None,
                "s3_key": None,
                "local_file": None,
                "status": "LIST_ERROR",
                "message": f"{type(e).__name__}: {e}",
            })
            continue

        print(f"Found {len(scans)} scans in padded window")

        if not scans:
            manifest_rows.append({
                "transect_id": tr.transect_id,
                "row_number": tr.row_number,
                "mode": tr.mode,
                "group": tr.group,
                "radar": tr.radar,
                "start": tr.start,
                "end": tr.end,
                "out_dir": str(tr.out_dir),
                "scan_time": None,
                "s3_key": None,
                "local_file": None,
                "status": "NO_SCANS",
                "message": "No scans found in padded window",
            })
            continue

        for scan_time, key in scans:
            if args.dry_run:
                print(f"  DRY {scan_time:%Y-%m-%d %H:%M:%S} s3://{key}")
                local_file = tr.out_dir / Path(key).name
                status = "DRY_RUN"
                message = "not downloaded"
            else:
                try:
                    local_file = download_key(fs, key, tr.out_dir, overwrite=args.overwrite)
                    status = "DOWNLOADED" if args.overwrite else "OK"
                    message = ""
                    print(f"  OK  {scan_time:%Y-%m-%d %H:%M:%S} -> {local_file.name}")
                except Exception as e:
                    local_file = None
                    status = "DOWNLOAD_ERROR"
                    message = f"{type(e).__name__}: {e}"
                    print(f"  ERR {scan_time:%Y-%m-%d %H:%M:%S} {message}")

            manifest_rows.append({
                "transect_id": tr.transect_id,
                "row_number": tr.row_number,
                "mode": tr.mode,
                "group": tr.group,
                "radar": tr.radar,
                "start": tr.start,
                "end": tr.end,
                "out_dir": str(tr.out_dir),
                "scan_time": scan_time,
                "s3_key": f"s3://{key}",
                "local_file": str(local_file) if local_file is not None else None,
                "status": status,
                "message": message,
            })

    write_manifest(manifest_rows, args.out_root)

    status_counts = pd.DataFrame(manifest_rows)["status"].value_counts(dropna=False)
    print("\nStatus counts:")
    print(status_counts.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
