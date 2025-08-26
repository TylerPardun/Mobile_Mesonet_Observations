#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import pickle
from glob import glob
from mpi4py import MPI
import pandas as pd
from time import sleep
from utils import interpolate_var,get_dx_dy,get_mm_files,compute_thw,haversine,extract_within_radius,rotate_grid,get_elev_file_vectorized,get_elevation,compute_precipitable_water
from pyart.core.transforms import geographic_to_cartesian_aeqd as geo2cart_aeqd
import sharppy.sharptab.profile as profile
import sharppy.sharptab.params as params
from datetime import datetime, timedelta

import pyart
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator as _Interp  # optional, guarded below if you want
from netCDF4 import date2num, num2date
from tqdm import tqdm

import metpy.calc as mpcalc
from metpy.units import units
import metpy.constants as const

def get_t0s(file_path,mode):
    tr = pd.read_csv(file_path)
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
    if (mode == 'T') or (mode=='L') or (mode=='S') or (mode=='W') or (mode=='Sk'):
        mmfs = np.array(sorted(glob(tracking_dir+'/tornadic/{}_*'.format(x))))[0]
    else:
        mmfs = np.array(sorted(glob(tracking_dir+'/nontornadic/{}_*.csv'.format(x))))[0]

    strk = pd.read_csv(mmfs)
    
    mtimes = pd.to_datetime(strk['datetime'].values)
    mtimes = np.array([datetime(val.year,val.month,val.day,val.hour,val.minute,val.second) for val in mtimes])
    srlat,srlon = strk['latitude'].values,strk['longitude'].values
    
    #Sort for good measure
    isort = np.argsort(mtimes)
    mtimes,srlat,srlon = mtimes[isort],srlat[isort],srlon[isort]
    
    #Interpolate to match time of mtimes data
    time_grid = np.arange(date2num(mtimes[0],'seconds since 1970-01-01'),date2num(mtimes[-1],'seconds since 1970-01-01'),dt_tracking*60)
    srlat = interpolate_var(srlat,date2num(mtimes,'seconds since 1970-01-01'),time_grid)
    srlon = interpolate_var(srlon,date2num(mtimes,'seconds since 1970-01-01'),time_grid)
    mtimes = time_grid
    del time_grid
    
    #Convert back to datetime
    mtimes = num2date(mtimes,'seconds since 1970-01-01')
    mtimes = np.array([datetime(val.year,val.month,val.day,val.hour,val.minute,val.second) for val in mtimes])
    
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
    

def get_reflectivity_along_path(
    x, mtime, mlat, mlon, radar_file_dir, mode,
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

    # -- 1) Discover candidate radar files near the observation period
    nexrad_files = sorted(glob(os.path.join(radar_file_dir, f"{mode}_{x}", "*")))
    if not nexrad_files:
        return np.full(mtime.size, np.nan, dtype=float)

    # file datetime from filename: e.g., KXXX_YYYYMMDD_HHMMSS...
    def _file_dt(fname):
        base = os.path.basename(fname)
        # tolerate both "KXXXYYYYMMDD_HHMMSS" and "KXXX_YYYYMMDD_HHMMSS"
        parts = base.replace('-', '_').split('_')
        for p in range(len(parts)-1):
            d, t = parts[p], parts[p+1]
            if len(d) == 8 and len(t) == 6 and d.isdigit() and t.isdigit():
                return datetime.strptime(d + t, "%Y%m%d%H%M%S")
        # fallback: take first 2 tokens of pattern "YYYYMMDD HHMMSS"
        try:
            d = base.split('_')[0][-8:]
            t = base.split('_')[1][:6]
            return datetime.strptime(d + t, "%Y%m%d%H%M%S")
        except Exception:
            return None

    file_times = np.array([_file_dt(f) for f in nexrad_files], dtype=object)
    good_files = [f for f, ft in zip(nexrad_files, file_times) if ft is not None]
    if not good_files:
        return np.full(mtime.size, np.nan, dtype=float)

    # Pick a subset of files that bracket the obs times
    # (unique searchsorted indices gives a small representative set)
    ft = np.array([_file_dt(f) for f in good_files])
    ft_sort_idx = np.argsort(ft)
    good_files = list(np.array(good_files)[ft_sort_idx])
    ft = ft[ft_sort_idx]
    # map each obs to the previous file index
    obs_file_idx = np.clip(np.searchsorted(ft, mtime, side="right") - 1, 0, len(ft)-1)
    candidate_files = np.unique(obs_file_idx)
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

    if not sweep_times:
        return np.full(mtime.size, np.nan, dtype=float)

    sweep_times = np.array(sweep_times)
    sweep_nums  = np.array(sweep_nums, dtype=int)
    sweep_files = np.array(sweep_files, dtype=object)

    # -- 3) Match each observation time to nearest sweep time (within tolerance)
    sort_idx = np.argsort(sweep_times)
    st_sorted = sweep_times[sort_idx]
    sf_sorted = sweep_files[sort_idx]
    sn_sorted = sweep_nums[sort_idx]

    # find nearest index in sorted sweep times
    ii = np.searchsorted(st_sorted, mtime)
    ii = np.clip(ii, 1, len(st_sorted)-1)
    left  = ii - 1
    right = ii
    # pick closer of left/right
    use_right = (np.abs(mtime - st_sorted[left]) > np.abs(st_sorted[right] - mtime))
    nearest = np.where(use_right, right, left)

    # enforce time tolerance
    dt_sec = np.abs((st_sorted[nearest] - mtime) / timedelta(seconds=1))
    ok = dt_sec <= time_tolerance_sec

    matched_files = np.full(mtime.size, None, dtype=object)
    matched_sweeps = np.full(mtime.size, -1, dtype=int)
    matched_files[ok] = sf_sorted[nearest[ok]]
    matched_sweeps[ok] = sn_sorted[nearest[ok]]

    # -- 4) Group obs by (file, sweep) so we open each file once
    mesonet_ref = np.full(mtime.size, np.nan, dtype=float)
    # dictionary: (filepath, sweep) -> indices
    groups = {}
    for i in np.where(ok)[0]:
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
        gy, gx, _gz = radar.get_gate_x_y_z(sweep=si)  # gy/gx are 2D [ray, gate]
        gx_flat = gx.ravel().astype(float)
        gy_flat = gy.ravel().astype(float)

        # reflectivity values aligned with gates
        ref_flat = np.asanyarray(ref).ravel().astype(float)
        # mask out crazy values; keep <=0 if you want 0 instead of NaN
        ref_flat[ref_flat>100] = np.nan
        ref_flat[ref_flat<0] = np.nan 
        ref_flat[np.isfinite(ref_flat)] = np.nan 

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
                        try:
                            mesonet_ref[idxs[k]] = np.nanmedian(vals)
                        except:
                            mesonet_ref[idxs[k]] = 0
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

def compute_the_origin(prof, mdat, 
                       search_agl=(0., 6000.),   # meters AGL to search
                       delta_the=1.0,           # K tolerance for “near match”
                       prefer='low',            # 'low' (inflow) or 'high' (outflow)
                       smooth_window=0,         # e.g., 7 levels; 0 = no smoothing
                       evap_flag=None):         # optional boolean array: True -> choose upper crossing
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

    return out

def compute_sounding_layer_params(prof,mes,shear_layermin,shear_layermax,u_storm,v_storm):
    shear_mag,shear_dir = [],[]
    tdd,lr,cape,cin,pwat = [],[],[],[],[]
    for ish in range(len(shear_layermin)):
        Z = (np.array(prof.hght)-np.array(prof.hght)[0])[:-1] #m
    
        #Get the height indices of the layer
        iZ0,iZ1 = np.argmin(np.abs(Z-shear_layermin[ish])),np.argmin(np.abs(Z-shear_layermax[ish]))
        if Z[iZ0:iZ1].size > 0:
            su,sv,stemp,sdew,spres = np.array(prof.u)*0.514444,np.array(prof.v)*0.514444,np.array(prof.tmpc),np.array(prof.dwpc),np.array(prof.pres) #m/s
    
            #Rotate the hodograph to match the common storm motion vector
            su,sv = rotate_grid(su, sv, u_storm,v_storm, pivot_x=0,pivot_y=0,target_angle=normalized_storm_motion) #m/s
    
            #Compute SR bulk shear vector
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

def get_sticknet_data(sticknet_dir, elev_file_path, atime, Z, radar_file_dir, mode, caseid, time_to_space_conversion=5, elev_angle=0.5, time_tolerance_sec=360,search_radius_m=100.0,neighborhood_shape="circle",min_gates=1,fallback_to_nearest=True):
    '''
    This function will return TTU sticknet data from the time period specified. This function will correct pressure to the average height across all mesonets
    at this time period (details in derive_scale_height function)
    INPUTS:
        dir_mesonet -> location of mobile mesonet files
        atime -> datetime object of the analysis time period
        meso_lon,meso_lat -> coordinates of the meso location
        time_to_space_conversion -> int of the time to space conversion (minutes)
        P_0 -> float. Reference pressure to be used in the scale height calculation
    RETURNS:
        data -> dictionary housing all mobile mesonet observations within this time window
        H -> derived scale height
        Z -> average elevation of the mobile mesonet observations
    '''
    
    stick_files = sorted(glob(sticknet_dir+'/*{}*.nc*'.format(atime.strftime('%Y%m%d'))))
    
    # Initialize lists for all data
    all_data = {key: [] for key in ['time', 'lat', 'lon', 'pres', 'temp', 'dew', 'rh','pres_corr','ref','ql',
                                    'wspd', 'wdir', 'u', 'v','qv', 'th','thv_ql','tv_ql',
                                    'thv','the','thw','tv','elev', 'campaign', 'probe']}
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    for fi in stick_files:
        with xr.open_dataset(fi, decode_times=False) as m:
            tm = m.time.values
            mask = (tm >= date2num(atime - time_window, 'seconds since 1970-01-01')) & (tm <= date2num(atime + time_window,'seconds since 1970-01-01'))
    
            # Extract relevant data and filter
            temp,pres,rh,wspd,wdir = m.temp.sel(time=mask).values,m.pres.sel(time=mask).values,m.rh.sel(time=mask).values,m.wspd.sel(time=mask).values,m.wdir.sel(time=mask).values    
            lat,lon,tm = m.lat.sel(time=mask).values,m.lon.sel(time=mask).values,num2date(m.time.sel(time=mask).values,'seconds since 1970-01-01',only_use_cftime_datetimes=False)
            lon = np.array([-val if val>0 else val for val in lon])
            
            if temp.size > 1:
                #Unit check
                if rh[0] < 1:
                    rh = rh*100
                if temp[0] > 100:
                    temp = temp-273.15
                
                valid = (~np.isnan(lat)) & (~np.isnan(lon)) & (~np.isnan(temp)) & (~np.isnan(pres)) & (~np.isnan(rh)) & (~np.isnan(wspd)) & (~np.isnan(wdir)) & (lon<0) & (temp>-100) & (pres>-100) & (rh>-100) & (wspd>-100) & (wdir>-100)
                if np.any(valid):
                    temp,pres,rh,wspd,wdir = temp[valid],pres[valid],rh[valid],wspd[valid],wdir[valid]
                    lat,lon,tm = lat[valid],lon[valid],tm[valid]
                    
                    # Compute wind components
                    u, v = mpcalc.wind_components(wspd * (units.meters / units.seconds), wdir * units.degrees)
                    u, v = u.m, v.m
            
                    #Compute air density
                    rho = (pres*100)/(Rd*(temp+273.15))

                    elev_files = get_elev_file_vectorized(elev_file_path, lat, lon, progress=False)
                    elev  = get_elevation(elev_files, lat, lon, dlon=0.005, dlat=0.005, progress=False)
            
                    qv = mpcalc.mixing_ratio_from_relative_humidity(np.array(pres) * units.hPa, np.array(temp) * units.degC, np.array(rh) * units.percent).m
                    tv = mpcalc.virtual_temperature(np.array(temp)*units.degC,np.array(qv)*units.dimensionless).m
                    pres_corr = pres_corr = np.array(pres) * np.exp(((np.array(elev) - Z) / (Rd * (tv + 273.15))) * g)
    
                    ref = get_reflectivity_along_path(caseid, tm, lat, lon, radar_file_dir, amode, elev_angle=elev_angle,time_tolerance_sec=time_tolerance_sec,search_radius_m=search_radius_m,neighborhood_shape=neighborhood_shape,min_gates=min_gates,fallback_to_nearest=fallback_to_nearest)
                    ref[ref<0] = np.nan
                    ref[np.isnan(ref)] = 0
    
                    #Derive the liquid water mixing ratio [Rutledge & Hobbs (1984) vs. Hane & Ray (1985)]
                    ql = (10**((ref-42.2)/16.8))/1000 #g/g
                    alpha = 0.1*ref #dBZ
                    ql = ((1/rho) * (((10**alpha)/17300)**(4/7)))/1000
                    
                    dew = mpcalc.dewpoint_from_relative_humidity(np.array(temp) * units.degC, np.array(rh) * units.percent).m
                    qvs = mpcalc.saturation_mixing_ratio(np.array(pres) * units.hPa, np.array(temp) * units.degC).m
                    th = mpcalc.potential_temperature(np.array(pres) * units.hPa, np.array(temp) * units.degC).m
                    thv = th*(1+(0.61*qv))
                    thv_ql = th*(1+(0.61*qv)-ql)
                    tv_ql = ((temp+273.15)*(1+(0.61*qv)-ql)) - 273.15 #C
        
                    # Compute theta-e and theta-w
                    the = mpcalc.equivalent_potential_temperature(np.array(pres)*units.hPa,np.array(temp)*units.degC,np.array(dew)*units.degC).m
                    thw = np.array([compute_thw(temp[i], pres[i], rh[i]) for i in range(pres.size)])
        
                    # Identify campaign
                    year = tm[0].year if len(tm) > 0 else None
                    campaign = {
                        2009: 'V2_09', 2010: 'V2_10', 2017: 'RiVorS', 2019: 'TORUS_19',
                        2022: 'TORUS_22', 2023: 'TORUS_LiTE', 2024: 'LIFT'
                    }.get(year, 'Unknown')
            
                    # Append data
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
    

def get_mesonet_data(dir_mesonet, elev_file_path, atime, Z, radar_file_dir, mode, caseid, time_to_space_conversion=5, elev_angle=0.5, time_tolerance_sec=360,search_radius_m=100.0,neighborhood_shape="circle",min_gates=1,fallback_to_nearest=True):
    """
    Retrieves mobile mesonet data within a specified time period and corrects pressure to a common elevation.

    Parameters:
    - dir_mesonet (str): Path to mesonet data.
    - atime (datetime): Analysis time.
    - time_to_space_conversion (int, optional): Time window in minutes (default: 10).

    Returns:
    - data (dict): Dictionary with mesonet data.
    - Z (float): Mean elevation of the mesonet observations.
    """

    # Derive average elevation and fetch mesonet file paths
    mmfs = get_mm_files(dir_mesonet, atime)
    
    # Initialize lists for all data
    all_data = {key: [] for key in ['time', 'lat', 'lon', 'pres', 'temp', 'dew', 'rh','pres_corr','ref','ql',
                                    'wspd', 'wdir', 'u', 'v','qv', 'th','thv_ql','tv_ql',
                                    'thv','the','thw','tv','elev', 'campaign', 'probe']}
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    # Process each mesonet file
    for fi in mmfs:
        with xr.open_dataset(fi, decode_times=False) as m:
            tm = m.time.values
            mask = (tm >= date2num(atime - time_window, 'seconds since 1970-01-01')) & (tm <= date2num(atime + time_window,'seconds since 1970-01-01'))
    
            # Extract relevant data and filter
            try:
                temp,pres,rh,wspd,wdir = m.temp_unbiased.sel(time=mask).values,m.pres_unbiased.sel(time=mask).values,m.rh_unbiased.sel(time=mask).values,m.wspd_unbiased.sel(time=mask).values,m.wdir_unbiased.sel(time=mask).values
            except AttributeError: #Dead file
                continue
            lat,lon,tm = m.lat.sel(time=mask).values,m.lon.sel(time=mask).values,num2date(m.time.sel(time=mask).values,'seconds since 1970-01-01',only_use_cftime_datetimes=False)
            lon = np.array([-val if val>0 else val for val in lon])
            if temp.size > 1:
                #Unit check
                if rh[0] < 1:
                    rh = rh*100
                if temp[0] > 100:
                    temp = temp-273.15
                
                valid = (~np.isnan(lat)) & (~np.isnan(lon)) & (~np.isnan(temp)) & (~np.isnan(pres)) & (~np.isnan(rh)) & (~np.isnan(wspd)) & (~np.isnan(wdir)) & (lon<0) & (temp>-100) & (pres>-100) & (rh>-100) & (wspd>-100) & (wdir>-100)
                if np.any(valid):
                    temp,pres,rh,wspd,wdir = temp[valid],pres[valid],rh[valid],wspd[valid],wdir[valid]
                    lat,lon,tm = lat[valid],lon[valid],tm[valid]
                    
                    # Compute wind components
                    u, v = mpcalc.wind_components(wspd * (units.meters / units.seconds), wdir * units.degrees)
                    u, v = u.m, v.m
            
                    #Compute air density
                    rho = (pres*100)/(Rd*(temp+273.15))

                    elev_files = get_elev_file_vectorized(elev_file_path, lat, lon, progress=False)
                    elev  = get_elevation(elev_files, lat, lon, dlon=0.005, dlat=0.005, progress=False)
            
                    qv = mpcalc.mixing_ratio_from_relative_humidity(np.array(pres) * units.hPa, np.array(temp) * units.degC, np.array(rh) * units.percent).m
                    tv = mpcalc.virtual_temperature(np.array(temp)*units.degC,np.array(qv)*units.dimensionless).m
                    pres_corr = np.array(pres) * np.exp(((np.array(elev) - Z) / (Rd * (tv + 273.15))) * g)
        
                    #Remove stationary observations from 2009 VORTEX2 (Waugh and Fredrickson 2010)
                    if tm[0].year == 2009:
                        keep = keep_moving_points(lat, lon, tm, min_move_m=30, min_speed_mps=0.5)
                        tm   = tm[keep]; temp = temp[keep]; pres = pres[keep]; rh = rh[keep]
                        wspd = wspd[keep]; wdir = wdir[keep]; elev = elev[keep]
                        lat  = lat[keep]; lon  = lon[keep]; rho = rho[keep]; qv = qv[keep]; tv = tv[keep]
    
                    ref = get_reflectivity_along_path(caseid, tm, lat, lon, radar_file_dir, amode, elev_angle=elev_angle,time_tolerance_sec=time_tolerance_sec,search_radius_m=search_radius_m,neighborhood_shape=neighborhood_shape,min_gates=min_gates,fallback_to_nearest=fallback_to_nearest)
                    ref[ref<0] = np.nan
                    ref[np.isnan(ref)] = 0
    
                    #Derive the liquid water mixing ratio [Rutledge & Hobbs (1984) vs. Hane & Ray (1985)]
                    ql = (10**((ref-42.2)/16.8))/1000 #g/g
                    alpha = 0.1*ref #dBZ
                    ql = ((1/rho) * (((10**alpha)/17300)**(4/7)))/1000
                    
                    dew = mpcalc.dewpoint_from_relative_humidity(np.array(temp) * units.degC, np.array(rh) * units.percent).m
                    qvs = mpcalc.saturation_mixing_ratio(np.array(pres) * units.hPa, np.array(temp) * units.degC).m
                    th = mpcalc.potential_temperature(np.array(pres) * units.hPa, np.array(temp) * units.degC).m
                    thv = th*(1+(0.61*qv))
                    thv_ql = th*(1+(0.61*qv)-ql)
                    tv_ql = ((temp+273.15)*(1+(0.61*qv)-ql)) - 273.15 #C
        
                    # Compute theta-e and theta-w
                    the = mpcalc.equivalent_potential_temperature(np.array(pres)*units.hPa,np.array(temp)*units.degC,np.array(dew)*units.degC).m
                    thw = np.array([compute_thw(temp[i], pres[i], rh[i]) for i in range(pres.size)])
        
                    # Identify campaign
                    year = tm[0].year if len(tm) > 0 else None
                    campaign = {
                        2009: 'V2_09', 2010: 'V2_10', 2017: 'RiVorS', 2019: 'TORUS_19',
                        2022: 'TORUS_22', 2023: 'TORUS_LiTE', 2024: 'LIFT'
                    }.get(year, 'Unknown')
            
                    # Append data
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

def process_case(dir_mesonet, sticknet_dir, elev_file_path, sound, shear_layermin, shear_layermax, atime, Z, u_storm, v_storm, radar_file_dir, mode, caseid, meso_time, meso_lat, meso_lon, time_to_space_conversion=5):

    mes = get_mesonet_data(dir_mesonet, elev_file_path, atime, Z, radar_file_dir, mode, caseid, time_to_space_conversion=time_to_space_conversion)
    stick = get_sticknet_data(sticknet_dir, elev_file_path, atime, Z, radar_file_dir, mode, caseid, time_to_space_conversion=time_to_space_conversion)
    merged = {}
    for val in list(mes.keys()):
        merged[val] = np.append(mes[val],stick[val])
    
    del stick
    mes = merged
    del merged
    
    ######################################################
    # Get sounding-derived observations
    ######################################################
    
    #Interpolate the sounding
    try:
        prof = interpolate_profile_on_pressure(sound[atime.strftime('%Y%m%d%H%M%S')], dp_hPa=sounding_dpres, return_profile=True)
        qc = np.asarray(prof.pres, dtype=float)
        org_height = compute_the_origin(prof, mes)
        
        #Compute shear in various layers
        shear_layers = compute_sounding_layer_params(prof,mes,shear_layermin,shear_layermax,u_storm,v_storm)
        
    except:
        #No sounding :( make all nans
        prof = np.nan
        org_height = np.zeros((mes['time'].size))*np.nan
        shear_layers = {}
        keys = ['shear_mag','shear_dir','tdd','lr','cape','cin','pwat']
        for ky in keys:
            shear_layers[ky] = np.zeros((mes['time'].size))*np.nan
        
    
    #Append the data
    mes['prof'] = prof
    mes['org_height'] = org_height
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
    
    #Rotate the winds to match the normalized storm motion
    ru,rv = rotate_grid(mes['u'], mes['v'], u_storm,v_storm, pivot_x=0,pivot_y=0,target_angle=normalized_storm_motion)
    mes['ru'] = ru
    mes['rv'] = rv
    
    return mes
    

if __name__ == '__main__':
    # MPI Setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    dir_transects = './'
    tracking_dir = 'storm_tracks/'
    dir_mesonet = 'mobile_mesonet_processed2/'
    sticknet_dir = 'sticknets/'
    asos_dir = 'asos/'
    asos_rad_outdir = 'nexrad_asos/'
    radar_file_dir = 'nexrad/'
    elev_file_path = 'elevation/'    
    
    '''
    #Define paths to files
    dir_transects = '../data/'
    tracking_dir = '../data/storm_tracks/'
    dir_mesonet = '../data/mobile_mesonet_processed2/'
    sticknet_dir = '../data/sticknets/'
    asos_dir = '../data/asos/'
    asos_rad_outdir = '../data/nexrad_asos/'
    radar_file_dir = '../data/nexrad/'
    elev_file_path = '../data/elevation/'
    '''
    
    #Define some parameters
    time_to_space_conversion = 5 #min - Used to grab MM observations for analysis
    dt_tracking = 1/60 #min for interpolation of storm tracks
    max_distance_range = 100 #km for inclusion of sticknet elevation data relative to the mesocyclone
    asos_distance = 400 #km - Maximum radius to search for ASOS stations relative to the mesocyclone location
    ref_asos_dist = 10 #km - Radius to compute dBZ relative to an ASOS station to determine if uncontaminated
    asos_ref_threshold = 10 #dBZ - Used to as minimum to define convectively uncontaminated ASOS observations
    normalized_storm_motion = 90 #deg, westerly storm motion by default
    sounding_dpres = 0.2 #mb - interpolation of soundings used in CAPE/CIN computations
    
    #Define constants
    Rd = float(np.array(const.dry_air_gas_constant))
    g = float(np.array(const.earth_gravity))
    
    analysis_times = np.arange(-20,20+1,time_to_space_conversion)

    #Read in the sounding data
    with open('soundings_RAP.pkl','rb') as f:
        sound,sound_lat,sound_lon = pickle.load(f)
    
    # Define the atmospheric layers we want to retain data from
    levels = np.arange(0, 6500, 500)
    shear_layers = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            shear_layers.append((levels[i], levels[j]))
    
    #Get the bottom and top layers
    shear_layermin,shear_layermax = [val[0] for val in shear_layers],[val[1] for val in shear_layers]

    #Define some constants (don't need to change)
    modes = ['T','N']
    for mode in modes:

        #Start time of tornado time
        t0, amode, tr = get_t0s(dir_transects+'transect_times.csv',mode)

        #Make a list of all possible analysis times to geta base state for each one
        atimes = np.array([t0[x]+timedelta(minutes=int(analysis_time)) for analysis_time in analysis_times for x in range(len(t0))])
        caseIDs = np.array([int(tr['CaseID'].values[x].split(amode)[1])-1 for analysis_time in analysis_times for x in range(len(t0))])
        
        ############################################################
        # Get the metadata for each case
        ############################################################
        
        #Storm information
        dat = [parse_storm_tracks(caseIDs[x],tracking_dir,amode,atimes[x],dt_tracking=dt_tracking,time_to_space_conversion=time_to_space_conversion) for x in range(atimes.size)]
        meso_times,meso_lats,meso_lons,u_storms,v_storms = np.array([val[0] for val in dat]),np.array([val[1] for val in dat]),np.array([val[2] for val in dat]),np.array([val[3] for val in dat]),np.array([val[4] for val in dat])
        
        #Average elevation for each analysis time
        #Zs = np.array([derive_ave_elev(dir_mesonet, sticknet_dir, atimes[x], meso_times[x], meso_lats[x], meso_lons[x], time_to_space_conversion=time_to_space_conversion, max_distance_range=max_distance_range) for x in tqdm(range(atimes.size))])
        
        if amode == 'T':
            with open('Zs_tor.pkl','rb') as f:
                Zs = pickle.load(f)
        else:
            with open('Zs_ntor.pkl','rb') as f:
                Zs = pickle.load(f)

        # Distribute tasks across processes
        task_splits = np.array_split(atimes, size)
        local_tasks = task_splits[rank]
        
        # Local storage for this rank's results
        local_base = {}
        for x in tqdm(range(len(local_tasks))):
            sleep(np.random.choice(np.linspace(0,0.1,300))) #Prevents I/O issues
            ix = np.where((atimes==local_tasks[x]))[0][0]
            case_data = process_case(dir_mesonet, sticknet_dir, elev_file_path, sound, shear_layermin, shear_layermax, atimes[ix], Zs[ix], u_storms[ix], v_storms[ix], radar_file_dir, amode, caseIDs[ix], meso_times[ix], meso_lats[ix], meso_lons[ix], time_to_space_conversion=time_to_space_conversion)
                
            local_base[local_tasks[x].strftime('%Y%m%d%H%M')] = case_data

        #Write a pickle file for each rank
        filename = f'mobile_mesonet_rank{rank}_tsc{time_to_space_conversion}_{mode}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(local_base, f)

    '''
    # Only the root process saves the combined data
            if rank == 0:
        # Combine all gathered results
        all_base = {}
        for file in sorted(glob('mobile_mesonet_rank*_tsc*.pkl')):
            with open(file, 'rb') as f:
                all_base.update(pickle.load(f))
        
        with open('mobile_mesonet_data_tsc_{}_min.pkl'.format(time_to_space_conversion), 'wb') as f:
            pickle.dump(all_base, f)

        #Remove the rank pickle files
        #[os.remove(f) for f in sorted(glob('mobile_mesonet_rank*_tsc*.pkl'))]
    '''