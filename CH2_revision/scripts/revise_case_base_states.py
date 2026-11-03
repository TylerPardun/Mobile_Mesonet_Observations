#!/usr/bin/env python3
"""Build an ambient-focused, case-level base-state table.

The primary source is the regional RAP/RUC base-state field at the analysis
time nearest the case zero offset.  The associated build diagnostics confirm
that these products were generated without the optional inflow-sector mask.
Mobile-mesonet environmental medians and far-field surface stations are kept
as independent diagnostics; they do not set the revised values.

Pressure is always expressed at the fixed case DEM datum in metadata.json.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "data" / "case_review_bundles"
CURRENT_CSV = ROOT / "data" / "case_base_states.csv"
OUTPUT_CSV = ROOT / "data" / "case_base_states_revised.csv"
DEBUG_FILES = [
    ROOT / "data" / "base_states_RAP_RUC" / "debug" / "base_state_data_tsc5_N_debug.csv",
    ROOT / "data" / "base_states_RAP_RUC" / "debug" / "base_state_data_tsc5_T_debug.csv",
]

G = 9.80665
RD = 287.05


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def case_sort_key(case_id: str) -> tuple[str, int]:
    return case_id[0], int(case_id[1:])


def parse_key_time(key: str, values: dict | None = None) -> pd.Timestamp:
    if values and values.get("time") is not None:
        ts = pd.Timestamp(values["time"])
    else:
        ts = pd.to_datetime(str(key), format="%Y%m%d%H%M", utc=True)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def target_time(meta: dict) -> pd.Timestamp:
    value = meta.get("tstart") or meta.get("obs_start")
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def nearest_entry(values_by_key: dict, target: pd.Timestamp) -> tuple[str, dict, pd.Timestamp]:
    candidates = []
    for key, values in values_by_key.items():
        if not isinstance(values, dict):
            continue
        ts = parse_key_time(key, values)
        candidates.append((abs((ts - target).total_seconds()), str(key), values, ts))
    if not candidates:
        raise ValueError("no valid source entries")
    _, key, values, ts = min(candidates, key=lambda item: item[0])
    return key, values, ts


def virtual_temperature_k(temp_c: float, dew_c: float, pressure_hpa: float) -> float:
    """Virtual temperature from vapor pressure inferred from dewpoint."""
    e = 6.112 * math.exp(17.67 * dew_c / (dew_c + 243.5))
    mixing_ratio = 0.622 * e / max(pressure_hpa - e, 1.0)
    return (temp_c + 273.15) * (1.0 + 0.61 * mixing_ratio)


def reduce_pressure(
    pressure_hpa: float,
    source_z_m: float,
    target_z_m: float,
    temp_c: float,
    dew_c: float,
) -> float:
    tv = virtual_temperature_k(temp_c, dew_c, pressure_hpa)
    return pressure_hpa * math.exp(G * (source_z_m - target_z_m) / (RD * tv))


def consistent_rh(temp_c: float, dew_c: float) -> float:
    rh = relative_humidity_from_dewpoint(temp_c * units.degC, dew_c * units.degC)
    return float(np.clip(rh.to("dimensionless").m * 100.0, 0.0, 100.0))


def load_debug() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in DEBUG_FILES if path.exists()]
    if not frames:
        return pd.DataFrame()
    debug = pd.concat(frames, ignore_index=True)
    debug["case_id"] = debug["case_id"].astype(str)
    debug["time_key"] = debug["time"].astype(str).str.replace(r"\.0$", "", regex=True)
    return debug


def debug_row_for(debug: pd.DataFrame, case_id: str, key: str, valid_time: pd.Timestamp) -> pd.Series | None:
    if debug.empty:
        return None
    exact = debug[(debug["case_id"] == case_id) & (debug["time_key"] == str(key))]
    if len(exact):
        return exact.iloc[0]
    case_rows = debug[debug["case_id"] == case_id].copy()
    if case_rows.empty:
        return None
    case_rows["parsed_time"] = pd.to_datetime(case_rows["time_key"], format="%Y%m%d%H%M", utc=True, errors="coerce")
    idx = (case_rows["parsed_time"] - valid_time).abs().idxmin()
    return case_rows.loc[idx]


def interpolate_profile(profile: dict, z_ref_m: float) -> dict:
    z = np.asarray(profile["hgt_m"], dtype=float)
    order = np.argsort(z)
    z = z[order]

    # GUI model soundings use height AGL: the first level (0 m AGL) is already
    # the model surface at the case location/datum.  Treating hgt_m as MSL
    # would move the parcel hundreds of metres up the profile and corrupt
    # pressure by tens of hPa.
    if np.nanmin(z) == 0.0:
        surface_index = int(np.nanargmin(z))
        return {
            "temp_C": float(np.asarray(profile["T_C"], dtype=float)[surface_index]),
            "dew_C": float(np.asarray(profile["Td_C"], dtype=float)[surface_index]),
            "pres_hPa": float(np.asarray(profile["p_hPa"], dtype=float)[surface_index]),
            "u_ms": float(np.asarray(profile["u_ms"], dtype=float)[surface_index]),
            "v_ms": float(np.asarray(profile["v_ms"], dtype=float)[surface_index]),
        }

    def interp(key: str) -> float:
        arr = np.asarray(profile[key], dtype=float)[order]
        good = np.isfinite(z) & np.isfinite(arr)
        if not np.any(good):
            return np.nan
        return float(np.interp(z_ref_m, z[good], arr[good]))

    return {
        "temp_C": interp("T_C"),
        "dew_C": interp("Td_C"),
        "pres_hPa": interp("p_hPa"),
        "u_ms": interp("u_ms"),
        "v_ms": interp("v_ms"),
    }


def nearest_model_sounding(bundle: Path, target: pd.Timestamp, z_ref_m: float) -> tuple[dict, pd.Timestamp, str]:
    candidates = []
    for path in (bundle / "model_soundings").glob("*.json"):
        profile = load_json(path)
        ts = pd.to_datetime(str(profile["valid_time"]), format="%Y%m%d%H%M%S", utc=True)
        candidates.append((abs((ts - target).total_seconds()), path, profile, ts))
    if not candidates:
        raise ValueError(f"no model sounding JSON for {bundle.name}")
    _, path, profile, ts = min(candidates, key=lambda item: item[0])
    return interpolate_profile(profile, z_ref_m), ts, path.name


def n7_far_field_sounding(z_ref_m: float) -> tuple[dict, pd.Timestamp, str]:
    path = ROOT / "data" / "soundings" / "Far_Field_MW41_output_20190615_225029.csv"
    df = pd.read_csv(path, skiprows=2)
    z = pd.to_numeric(df["Filtered Altitude (m)"], errors="coerce")
    order = np.argsort(z.to_numpy(float))
    z_sorted = z.to_numpy(float)[order]

    def interp(column: str) -> float:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(float)[order]
        good = np.isfinite(z_sorted) & np.isfinite(values)
        return float(np.interp(z_ref_m, z_sorted[good], values[good]))

    temp_c = interp("Filtered Temperature (K)") - 273.15
    dew_c = interp("Filtered Dewpoint (K)") - 273.15
    pressure = interp("Filtered Pressure (mb)")
    wind_dir = interp("Filtered Wind Dir")
    wind_speed = interp("Filtered Wind Spd (m/s)")
    state = {
        "temp_C": temp_c,
        "dew_C": dew_c,
        "pres_hPa": pressure,
        "u_ms": -wind_speed * math.sin(math.radians(wind_dir)),
        "v_ms": -wind_speed * math.cos(math.radians(wind_dir)),
    }
    ts_text = str(df.iloc[0]["Date-Time"]).strip("[]")
    ts = pd.Timestamp(ts_text).tz_localize("UTC")
    return state, ts, path.name


def far_field_surface_check(bundle: Path, target: pd.Timestamp) -> dict:
    """Independent observed check; never used to set the revised state."""
    df = pd.read_parquet(bundle / "surface_obs.parquet").copy()
    if df.empty:
        return {}
    df["time"] = pd.to_datetime(df["time"], utc=True)
    dt_min = (df["time"] - target).abs().dt.total_seconds() / 60.0
    mask = dt_min <= 45.0
    if "range_km" in df:
        mask &= df["range_km"].between(40.0, 250.0, inclusive="both")
    df = df.loc[mask].copy()
    if df.empty:
        return {}
    df["dt_min"] = (df["time"] - target).abs().dt.total_seconds().abs() / 60.0
    df = df.sort_values("dt_min").groupby(["network", "station_id"], as_index=False).first()
    result = {
        "n_stations": int(len(df)),
        "temp_C": float(df["temp_C"].median()),
        "dew_C": float(df["dew_C"].median()),
        "pres_corr_hPa": float(df["pres_corr_hPa"].median()),
        "u_ms": float(df["u_ms"].median()),
        "v_ms": float(df["v_ms"].median()),
    }
    return result


def vector_delta(u1: float, v1: float, u2: float, v2: float) -> float:
    if not all(finite(v) for v in (u1, v1, u2, v2)):
        return np.nan
    return float(math.hypot(u1 - u2, v1 - v2))


def rounded(value, ndigits: int = 3):
    return round(float(value), ndigits) if finite(value) else np.nan


def main() -> None:
    current = pd.read_csv(CURRENT_CSV).set_index("case_id", drop=False)
    debug = load_debug()
    rows = []
    created = datetime.now(timezone.utc).isoformat()

    case_dirs = sorted((p for p in BUNDLE_ROOT.iterdir() if p.is_dir()), key=lambda p: case_sort_key(p.name))
    for bundle in case_dirs:
        case_id = bundle.name
        meta = load_json(bundle / "metadata.json")
        base_states = load_json(bundle / "base_states.json")
        suggested = load_json(bundle / "suggested_adjustment.json")
        original = current.loc[case_id]
        z_ref_m = float(meta["z_ref_m"])
        target = target_time(meta)

        rap = (base_states.get("RAP_RUC") or {}).get("values_by_key") or {}
        source_points = 0
        source_clusters = 0
        source_z_ref_m = z_ref_m
        source_file = ""
        source_limit = ""

        if rap:
            key, values, valid_time = nearest_entry(rap, target)
            dbg = debug_row_for(debug, case_id, key, valid_time)
            if dbg is None:
                raise ValueError(f"{case_id}: RAP/RUC source exists but debug provenance is missing")
            if bool(dbg.get("flag_inflow_mask_applied", False)):
                raise ValueError(f"{case_id}: source unexpectedly used inflow mask")
            source_z_ref_m = float(dbg["Z"]) if finite(dbg.get("Z")) else z_ref_m
            state = {
                "temp_C": float(values["temp"]),
                "dew_C": float(values["dew"]),
                "pres_hPa": float(values.get("pres_corr", values["pres"])),
                "u_ms": float(values["u"]),
                "v_ms": float(values["v"]),
            }
            state["pres_hPa"] = reduce_pressure(
                state["pres_hPa"], source_z_ref_m, z_ref_m, state["temp_C"], state["dew_C"]
            )
            base_source = "RAP_RUC_regional"
            method = "regional_model_zero_offset_no_inflow_mask"
            source_points = int(dbg.get("n_valid_points", 0)) if finite(dbg.get("n_valid_points")) else 0
            source_clusters = int(dbg.get("n_clusters", 0)) if finite(dbg.get("n_clusters")) else 0
            source_file = f"base_states.json:RAP_RUC:{key}"
            source_limit = "Regional model analysis can retain model bias and smooth real mesoscale gradients."
        elif case_id == "N7":
            state, valid_time, source_file = n7_far_field_sounding(z_ref_m)
            base_source = "far_field_observed_sounding"
            method = "far_field_sounding_interpolated_to_case_datum"
            source_points = 1
            source_limit = "Single far-field sounding; 49 minutes before case reference time."
        elif case_id in {"N17", "N57"}:
            state, valid_time, source_file = nearest_model_sounding(bundle, target, z_ref_m)
            base_source = "RAP_model_sounding"
            method = "model_sounding_interpolated_to_case_datum"
            source_points = 1
            source_limit = "Single-column model fallback because the regional gridded base-state record is absent."
        else:
            raise ValueError(f"{case_id}: no supported ambient source")

        if not all(finite(state[k]) for k in ("temp_C", "dew_C", "pres_hPa", "u_ms", "v_ms")):
            raise ValueError(f"{case_id}: selected state contains non-finite values: {state}")
        if state["dew_C"] > state["temp_C"]:
            raise ValueError(f"{case_id}: dewpoint exceeds temperature")

        state["rh_pct"] = consistent_rh(state["temp_C"], state["dew_C"])
        surface = far_field_surface_check(bundle, target)
        env = suggested.get("env_medians") or {}

        d_temp = state["temp_C"] - float(original["temp_C"])
        d_dew = state["dew_C"] - float(original["dew_C"])
        d_pres = state["pres_hPa"] - float(original["pres_corr_hPa"])
        d_wind = vector_delta(state["u_ms"], state["v_ms"], float(original["u_ms"]), float(original["v_ms"]))

        surface_support = False
        surface_summary = "No far-field surface check within 45 minutes."
        if surface:
            s_dt = state["temp_C"] - surface["temp_C"]
            s_dtd = state["dew_C"] - surface["dew_C"]
            s_dp = state["pres_hPa"] - surface["pres_corr_hPa"]
            s_dw = vector_delta(state["u_ms"], state["v_ms"], surface["u_ms"], surface["v_ms"])
            surface_support = abs(s_dt) <= 3.0 and abs(s_dtd) <= 4.0 and abs(s_dp) <= 3.5 and s_dw <= 6.0
            surface_summary = (
                f"Far-field surface check ({surface['n_stations']} stations): "
                f"model/source minus surface ΔT={s_dt:+.1f} C, ΔTd={s_dtd:+.1f} C, "
                f"Δp={s_dp:+.1f} hPa, vector-wind difference={s_dw:.1f} m s-1; "
                f"{'supports' if surface_support else 'does not tightly support'} the selected state."
            )

        current_looks_mobile_weighted = False
        mobile_note = "Mobile environmental median unavailable; it was not used."
        if all(finite(env.get(k)) for k in ("temp_C", "dew_C")):
            current_distance = math.hypot(
                (float(original["temp_C"]) - float(env["temp_C"])) / 2.0,
                (float(original["dew_C"]) - float(env["dew_C"])) / 2.0,
            )
            revised_distance = math.hypot(
                (state["temp_C"] - float(env["temp_C"])) / 2.0,
                (state["dew_C"] - float(env["dew_C"])) / 2.0,
            )
            current_looks_mobile_weighted = current_distance + 0.5 < revised_distance
            mobile_note = (
                "The original state is materially closer to the mobile pre-storm sample than the revised ambient source, "
                "which is consistent with possible inflow-parcel weighting."
                if current_looks_mobile_weighted
                else "The original state is not materially closer to the mobile pre-storm sample; the revision is source/datum standardization."
            )

        if base_source == "RAP_RUC_regional":
            quality = "high" if surface_support else "moderate"
            core_reason = (
                "Selected the regional RAP/RUC analysis at the case zero-offset time. "
                "Build diagnostics show the optional inflow-sector mask was not applied, so the value represents the "
                "Barnes-weighted regional model environment rather than the mobile inflow parcel."
            )
        elif case_id == "N7":
            quality = "moderate"
            core_reason = (
                "The gridded RAP/RUC base is missing. Used the observed Far_Field sounding launched before the case and "
                "interpolated it to the fixed case datum; its explicit far-field deployment is less inflow-specific than the mobile transect."
            )
        else:
            quality = "moderate" if surface_support else "low"
            core_reason = (
                "The gridded RAP/RUC base is missing. Used the nearest RAP model sounding and interpolated the surface "
                "state to the fixed case datum; this retains model ambient character without using the mobile inflow median."
            )

        datum_note = (
            f"Source pressure was hydrostatically reconciled from z={source_z_ref_m:.1f} m to the fixed case datum "
            f"using the source parcel virtual temperature. "
            if abs(source_z_ref_m - z_ref_m) > 0.05
            else "Source pressure already used the fixed case datum. "
        )
        reason = (
            f"{core_reason} {datum_note}Original-to-revised changes: ΔT={d_temp:+.1f} C, ΔTd={d_dew:+.1f} C, "
            f"Δp={d_pres:+.1f} hPa at z_ref={z_ref_m:.1f} m MSL, vector-wind change={d_wind:.1f} m s-1. "
            f"{surface_summary} {mobile_note}"
        )

        row = {
            "case_id": case_id,
            "mode": meta["mode"],
            "base_source": base_source,
            "method": method,
            "temp_C": rounded(state["temp_C"], 3),
            "rh_pct": rounded(state["rh_pct"], 3),
            "dew_C": rounded(state["dew_C"], 3),
            "pres_hPa": rounded(state["pres_hPa"], 3),
            "u_ms": rounded(state["u_ms"], 3),
            "v_ms": rounded(state["v_ms"], 3),
            "approved_pressure": False,
            "approved_wind": False,
            "quality": quality,
            "approved": False,
            "notes": reason,
            "n_env_obs": int(suggested.get("n_env_obs") or 0),
            "n_env_platforms": int(suggested.get("n_env_platforms") or 0),
            "sample_start_utc": suggested.get("sample_start_utc"),
            "sample_end_utc": suggested.get("sample_end_utc"),
            "saved_by_utc": created,
            "bundle_version": meta.get("bundle_version"),
            "pres_corr_hPa": rounded(state["pres_hPa"], 3),
            "z_ref_m": rounded(z_ref_m, 3),
            "z_ref_method": "case DEM datum from elevation-cache key nearest offset zero",
            "base_grain": "case",
            "base_valid_time_utc": valid_time.isoformat(),
            "source_z_ref_m": rounded(source_z_ref_m, 3),
            "source_file_or_key": source_file,
            "source_points": source_points,
            "source_clusters": source_clusters,
            "surface_check_n_stations": int(surface.get("n_stations", 0)) if surface else 0,
            "surface_check_supports": surface_support,
            "mobile_weighting_flag": current_looks_mobile_weighted,
            "original_temp_C": rounded(original["temp_C"], 3),
            "original_dew_C": rounded(original["dew_C"], 3),
            "original_rh_pct": rounded(original["rh_pct"], 3),
            "original_pres_corr_hPa": rounded(original["pres_corr_hPa"], 3),
            "original_u_ms": rounded(original["u_ms"], 3),
            "original_v_ms": rounded(original["v_ms"], 3),
            "delta_temp_C": rounded(d_temp, 3),
            "delta_dew_C": rounded(d_dew, 3),
            "delta_pres_corr_hPa": rounded(d_pres, 3),
            "delta_wind_vector_ms": rounded(d_wind, 3),
            "scientific_reason": reason,
            "source_limitation": source_limit,
            "pressure_compatibility_note": "Subtract only from observations reduced to this row's fixed z_ref_m.",
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    required_numeric = ["temp_C", "dew_C", "rh_pct", "pres_hPa", "pres_corr_hPa", "u_ms", "v_ms", "z_ref_m"]
    assert len(out) == len(current) == 109
    assert out["case_id"].is_unique
    assert set(out["case_id"]) == set(current.index)
    assert not out[required_numeric].isna().any().any()
    assert (out["dew_C"] <= out["temp_C"]).all()
    assert out["rh_pct"].between(0.0, 100.0).all()
    assert out["pres_hPa"].between(650.0, 1050.0).all()
    assert np.allclose(out["pres_hPa"], out["pres_corr_hPa"], atol=0.0, rtol=0.0)
    assert out["scientific_reason"].str.len().gt(100).all()
    assert out["pressure_compatibility_note"].str.contains("fixed z_ref_m", regex=False).all()
    out.to_csv(OUTPUT_CSV, index=False, float_format="%.3f")
    print(f"wrote {OUTPUT_CSV} ({len(out)} rows, {len(out.columns)} columns)")
    print(out["base_source"].value_counts().to_string())
    print(out["quality"].value_counts().to_string())
    print(f"mobile weighting flags: {int(out['mobile_weighting_flag'].sum())}")


if __name__ == "__main__":
    main()
