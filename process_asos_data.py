#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from glob import glob
from datetime import datetime, timezone  # (timedelta not used)
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from tqdm import tqdm
import netCDF4 as nc
from netCDF4 import date2num

def haversine(lon1, lat1, lon2, lat2):
    """Vectorized Haversine formula for distance calculation."""
    R = 6371.0  # Earth radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c
    
def get_elev_file_vectorized(file_path, lats, lons, progress=False):
    """Vectorized version of get_elev_file to handle multiple lat/lon pairs."""
    
    # QC Checks (Vectorized)
    invalid_mask = ((np.ceil(lats) == 100) & (-np.floor(lons) == 100)) | (-np.floor(lons) <= 30)
    
    # Apply QC mask
    valid_lats = np.where(invalid_mask, np.nan, lats)
    valid_lons = np.where(invalid_mask, np.nan, lons)

    # Generate the formatted strings for file searching
    file_paths = np.full(valid_lats.shape, np.nan, dtype=object)  # Initialize array for file paths

    # Build file search patterns for both conditions (vectorized)
    iterator = tqdm(zip(valid_lats, valid_lons),total=len(valid_lats)) if progress else zip(valid_lats, valid_lons)
    search_patterns = np.array([
        file_path + f'*n{np.ceil(lat):.0f}w0{-np.floor(lon):.0f}*' 
        if -np.floor(lon) < 100 else 
        file_path + f'*n{np.ceil(lat):.0f}w{-np.floor(lon):.0f}*'
        for lat, lon in iterator
    ])

    # Vectorized file searching using glob
    iterator = tqdm(enumerate(search_patterns),total=len(search_patterns)) if progress else enumerate(search_patterns)
    for i, pattern in iterator:
        if not np.isnan(valid_lats[i]):  # Skip NaNs
            matched_files = glob(pattern)
            file_paths[i] = matched_files[0] if matched_files else np.nan

    return file_paths

def get_elevation(elev_files, lats, lons, dlon=0.005,dlat=0.005,progress=False):
    '''This function will return elevation data from the files input.
    Data will return array the size of the input coordinates arrays (must be 1D)
    '''
    
    #Organize the lat/lon points by their associate file
    elev = np.zeros((lats.size))*np.nan
    unique_files = np.unique(elev_files)
    
    ifile = [np.where(elev_files==val)[0] for val in unique_files]
    
    #Open each file individually and grab the data we need from it
    iterator = tqdm(enumerate(unique_files),total=len(unique_files)) if progress else enumerate(unique_files)
    
    for x,fs in iterator:
        #~13 m grid spacing
        d = rioxarray.open_rasterio(fs).to_dataset('band')
        longrid, latgrid = d.x.values, d.y.values
        d = d.rename({1: 'elev'})
        elev_data = d.elev.values
        d.close()
        #Get all of the coordinates that fall into this file
        lat_points,lon_points = lats[ifile[x]],lons[ifile[x]]
    
        for i in range(ifile[x].size):
            #Subset the grid
            ilon,ilat = np.where((longrid>=lon_points[i]-dlon)&(longrid<=lon_points[i]+dlon))[0],np.where((latgrid>=lat_points[i]-dlat)&(latgrid<=lat_points[i]+dlat))[0]
            alongrid,alatgrid = longrid[ilon], latgrid[ilat]
            
            X,Y = np.meshgrid(alongrid,alatgrid)
            min_dist = haversine(X.ravel(),Y.ravel(), lon_points[i], lat_points[i]).argmin()
            ielev = np.unravel_index(min_dist, X.shape)
            
            elev_plot = np.zeros((ilat.size,ilon.size))*np.nan
            for ix in range(ilat.size):
                elev_plot[ix,:] = elev_data[ilat[ix]][ilon]
            
            elev[ifile[x][i]] = float(elev_plot[int(ielev[0]), int(ielev[1])])
        
    return elev
    
def _parse_header_and_times(line):
    """
    Parse the fixed header: WBAN(5) + ICAO(4) + <space?> + CALLSIGN(3/4, left-justified)
    followed by YEAR(4) MONTH(2) DAY(2) HOURlocal(4) HOURutc(4).
    We detect the year as the first 4 consecutive digits after the first 9 chars.
    Returns dict and index where the 'DATA' tokens begin.
    """
    wban = line[0:5]
    icao = line[5:9]
    rest = line[13:].rstrip("\n")

    m = re.search(r'(\d{4})(\d{2})(\d{2})(\d{4})(\d{4})', rest)
    if not m:
        return None, None
    year, mon, day, tloc, tutc = m.groups()
    call = rest[:m.start()].strip()

    # Build UTC datetime
    hh_utc = int(tutc[:2]); mm_utc = int(tutc[2:])
    dt = datetime(int(year), int(mon), int(day), hh_utc, mm_utc, tzinfo=timezone.utc)

    # data tokens start after the matched block
    data_str = rest[m.end():].strip()
    return {
        "wban": wban.strip(),
        "icao": icao.strip(),
        "call": call,
        "time_utc": dt
    }, data_str

def _first_dir_spd_pair(tokens):
    """Find first (dir, spd_knots) pair where dir in [0,360], spd >=0."""
    for i in range(len(tokens) - 1):
        d = _to_int(tokens[i])
        s = _to_float(tokens[i+1])
        if np.isfinite(d) and np.isfinite(s) and (0 <= d <= 360) and (0 <= s < 250):
            return d, s
    return np.nan, np.nan

def _to_int(x, nan=np.nan):
    try:
        return int(x)
    except Exception:
        return nan

def _to_float(x, nan=np.nan):
    try:
        return float(x)
    except Exception:
        return nan

def _f_to_c(F):
    return (np.asarray(F, dtype=float) - 32.0) * 5.0/9.0

def _rh_from_t_td_C(Tc, TDc):
    # Simple Tetens method; good enough for ASOS 1-min
    # es, e in hPa
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    e  = 6.112 * np.exp(17.67 * TDc / (TDc + 243.5))
    rh = 100.0 * (e / es)
    return np.clip(rh, 0.0, 100.0)

def _median_station_pressure_hpa(tokens):
    """Pick ~3 pressure readings in inHg (around 25–35) and convert to hPa, median."""
    vals = []
    for tok in tokens:
        v = _to_float(tok)
        if np.isfinite(v) and (24.0 <= v <= 33.5):  # broad gate for inHg range
            vals.append(v)
    if not vals:
        return np.nan
    return np.nanmedian(vals) * INHG_TO_HPA
    
def parse_pg1_file(path):
    """
    PG1 (DSI-6405): extract UTC time, station, 2-min avg wind dir/speed (knots).
    Returns pandas DataFrame with ['station','time','wdir_deg','wspd_ms'].
    """
    rows = []
    station_from_name = os.path.basename(path).split("-")[3]  # asos-1min-pg1-KAAO-YYYYMM.dat -> KAAO
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if len(line) < 30:
                continue
            try:
                hdr, data = _parse_header_and_times(line)
            except ValueError:
                print(line)
            if hdr is None:
                continue
            # Sanity: station from header (ICAO or call) should match filename station
            stn = hdr["icao"] if hdr["icao"] else hdr["call"]
            if station_from_name and stn and (station_from_name.upper() != stn.upper()):
                # Different station in file name vs line; keep the line's station anyway
                pass
            toks = data.split()
            # From docs: after vis1, D/N, vis2, D/N, next are dir2min, spd2min, dir5s, spd5s, runway/RVR ... :contentReference[oaicite:2]{index=2}
            wdir_deg, wspd_kt = _first_dir_spd_pair(toks)
            wspd_ms = wspd_kt * KNOTS_TO_MS if np.isfinite(wspd_kt) else np.nan
            rows.append((stn, hdr["time_utc"], wdir_deg, wspd_ms))
    if not rows:
        return pd.DataFrame(columns=["station","time","wdir_deg","wspd_ms"])
    df = pd.DataFrame(rows, columns=["station","time","wdir_deg","wspd_ms"])
    # drop exact dup minutes; keep last
    df = df.drop_duplicates(subset=["station","time"], keep="last").sort_values("time")
    return df

def parse_pg2_file(path):
    """
    PG2 (DSI-6406): extract UTC time, station, 1-min dry-bulb (F), dewpoint (F), and station pressure (inHg from sensors 1-3).
    Convert to tempC, rh%, hPa. Returns DataFrame ['station','time','temp_C','rh_pct','pres_hPa'].
    """
    rows = []
    station_from_name = os.path.basename(path).split("-")[3]  # asos-1min-pg2-KAAO-YYYYMM.dat -> KAAO
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if len(line) < 30:
                continue
            hdr, data = _parse_header_and_times(line)
            if hdr is None:
                continue
            stn = hdr["icao"] if hdr["icao"] else hdr["call"]
            toks = data.split()
            if len(toks) < 2:
                continue

            # Temp and dewpoint are the final two items (Deg F) per the doc example. :contentReference[oaicite:3]{index=3}
            # Be defensive: attempt last 2 tokens; if needed, scan from the end.
            def _last_two_floats(ts):
                for k in range(len(ts)-2, -1, -1):
                    t1, t2 = _to_float(ts[k]), _to_float(ts[k+1]) if (k+1) < len(ts) else (np.nan)
                    if np.isfinite(t1) and np.isfinite(t2):
                        return t1, t2
                return np.nan, np.nan

            dryF, dewF = _last_two_floats(toks)
            tempC = _f_to_c(dryF) if np.isfinite(dryF) else np.nan
            dewC  = _f_to_c(dewF) if np.isfinite(dewF) else np.nan
            rh = _rh_from_t_td_C(tempC, dewC) if np.isfinite(tempC) and np.isfinite(dewC) else np.nan
            # Station pressure from up to three sensors (inHg) → median → hPa. :contentReference[oaicite:4]{index=4}
            pres_hPa = _median_station_pressure_hpa(toks)
            rows.append((stn, hdr["time_utc"], tempC, rh, pres_hPa))

    if not rows:
        return pd.DataFrame(columns=["station","time","temp_C","rh_pct","pres_hPa"])
    df = pd.DataFrame(rows, columns=["station","time","temp_C","rh_pct","pres_hPa"])
    df = df.drop_duplicates(subset=["station","time"], keep="last").sort_values("time")
    return df

def merge_pg1_pg2(pg1_df, pg2_df):
    """
    Inner-join on station + time (UTC-minute). Prefers PG1 wind and PG2 thermo.
    """
    if pg1_df.empty and pg2_df.empty:
        return pd.DataFrame(columns=["station","time","wdir_deg","wspd_ms","temp_C","rh_pct","pres_hPa"])
    if pg1_df.empty:
        out = pg2_df.copy()
        out["wdir_deg"] = np.nan
        out["wspd_ms"] = np.nan
        return out[["station","time","wdir_deg","wspd_ms","temp_C","rh_pct","pres_hPa"]]
    if pg2_df.empty:
        out = pg1_df.copy()
        out["temp_C"] = np.nan
        out["rh_pct"] = np.nan
        out["pres_hPa"] = np.nan
        return out[["station","time","wdir_deg","wspd_ms","temp_C","rh_pct","pres_hPa"]]

    out = pd.merge(pg1_df, pg2_df, on=["station","time"], how="inner")
    out = out.sort_values(["station","time"]).reset_index(drop=True)
    return out

def dt64ns_to_ncnum(
    dt64,
    units="seconds since 1970-01-01 00:00:00",
    calendar="standard"
):
    """
    Convert datetime64[ns]/pandas datetime -> numeric time for NetCDF.
    NaT -> NaN.
    """
    # Normalize to pandas DatetimeIndex (handles ns precision & NaT)
    ts = pd.to_datetime(np.asarray(dt64))

    # Boolean mask of NaT; this is already a numpy array
    mask = pd.isna(ts)

    out = np.full(ts.size, np.nan, dtype=float)
    if (~mask).any():
        # Convert only valid times to Python datetime (naive)
        py = ts[~mask].to_pydatetime()  # returns ndarray of datetime.datetime
        out[~mask] = date2num(py, units=units, calendar=calendar)
    return out

if __name__ == '__main__':
    #Define some constants
    KNOTS_TO_MS = 0.514444
    INHG_TO_HPA = 33.8638866667
    
    yrs = ['2010','2017','2019','2022','2023','2024']
    mnths = ['04','05','06']
    outfile_dir = '../data/asos/' #Directory to store large netCDF of all the data
    station_meta_csv = '../data/asos/asos-sites.csv' #CSV file of all ASOS station metadata
    input_dir = '../data/asos/' #Directory containing the ASOS files
    elev_file_path = '../data/elevation/' #Directory storing the TIF elevation file data
    nc_asos_outfile_path = '../data/asos/torus_nc_files' #Directory to store the individual NC files for each ASOS station site before merging
    
    for yr in yrs:
        for mnth in mnths:
            raw_asos_dir1,raw_asos_dir2 = input_dir+'/asos_{}_raw/ASOS_1min_pg1.{}{}/'.format(yr,yr,mnth),input_dir+'/asos_{}_raw/ASOS_1min_pg2.{}{}/'.format(yr,yr,mnth)
            asites = pd.read_csv(station_meta_csv)
            
            #Get all available ASOS station names
            pg1_files = sorted(glob(raw_asos_dir1+'/*pg1*'))
            asos_station_names = np.array([val.split('/')[-1].split('-')[3] for val in pg1_files])
            
            for x,val in tqdm(enumerate(asos_station_names),total=len(asos_station_names)):
                #x = 0
                #val = asos_station_names[x]
                try:
                    isite = np.where(asites.CALL.values==val[1:])[0][0]
                    site_lat,site_lon,site_elev = asites.LAT.values[isite],asites.LON.values[isite],asites.elev.values[isite]
                    if (site_lat>=26) & (site_lat<=49.3) & (site_lon>=-110.7) & (site_lon<=-87) & (~np.isnan(site_elev)):
                        #Get the corresponding pg1 and pg2 data from the ASOS station
                        p1f = sorted(glob(raw_asos_dir1 +'/*{}*'.format(val)))[0]
                        p2f = sorted(glob(raw_asos_dir2 + '/*{}*'.format(val)))[0]
                        
                        #print(p1f)
                        #Parse through pg1 and pg2
                        pg1_df,pg2_df = parse_pg1_file(p1f), parse_pg2_file(p2f)
                        big = merge_pg1_pg2(pg1_df, pg2_df)
            
                        big['elev'] = [site_elev]*len(big)
                        big['lat'] = [site_lat]*len(big)
                        big['lon'] = [site_lon]*len(big)
                        
                        tstamp = pd.to_datetime(big.time.values[0])
                        tstamp = datetime(tstamp.year,tstamp.month,tstamp.day,tstamp.hour,tstamp.minute)
                        out_nc = '../data/asos/torus_nc_files/{}_{}_{}.nc'.format(tstamp.strftime('%Y%m%d%H%M'),val,x)
                        
                        #Write to netCDF output
                        ds = xr.Dataset(
                                data_vars=dict(
                                    temp_C=("obs", big["temp_C"].values.astype("float32")),
                                    rh_pct=("obs", big["rh_pct"].values.astype("float32")),
                                    pres_hPa=("obs", big["pres_hPa"].values.astype("float32")),
                                    wspd_ms=("obs", big["wspd_ms"].values.astype("float32")),
                                    wdir_deg=("obs", big["wdir_deg"].values.astype("float32")),
                                    lat=("obs", big["lat"].values.astype("float32")),
                                    lon=("obs", big["lon"].values.astype("float32")),
                                    station=("obs", big["station"].astype("S8").values),
                                    elev=("obs", big["elev"].astype("float32").values),
                                ),
                                coords=dict(
                                    time=("obs", big["time"].values.astype("datetime64[ns]"))
                                ),
                                attrs=dict(
                                    title="ASOS 1-minute merged PG1+PG2",
                                    source_pg1="DSI-6405 (winds)",  # :contentReference[oaicite:5]{index=5}
                                    source_pg2="DSI-6406 (thermo/pressure)",  # :contentReference[oaicite:6]{index=6}
                                    units="temp_C [°C], rh_pct [%], pres_hPa [hPa], wspd_ms [m/s], wdir_deg [deg], lat/lon [deg]"
                                )
                            )
                        
                        # Compression + chunking (per-minute rows)
                        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
                        ds.to_netcdf(out_nc, encoding=encoding)
                        
                except IndexError: #Meaning no files to parse through
                    pass
    
    
        nc_files = sorted(glob(nc_asos_outfile_path+"/*{}*.nc".format(yr)))
        
        time,temp,pres,rh,wspd,wdir,elev = [],[],[],[],[],[],[]
        lat,lon = [],[]
        for x,f in tqdm(enumerate(nc_files),total=len(nc_files)):
            d = xr.open_dataset(f)
                    
            time.append(dt64ns_to_ncnum(d.time.values))
            temp.append(d.temp_C.values)
            rh.append(d.rh_pct.values)
            pres.append(d.pres_hPa.values)
            wspd.append(d.wspd_ms.values)
            wdir.append(d.wdir_deg.values)
            lat.append(d.lat.values)
            lon.append(d.lon.values)
            elev.append(d.elev.values)
                
            d.close()

        
        #Write a netCDF file from the other file
        oufile_name = outfile_dir + '/asos_{}.nc'.format(yr)
        ncfile = nc.Dataset(oufile_name, 'w', format='NETCDF4')
        
        # Add global attributes
        ncfile.description = 'ASOS Station Retrievals. Made by Tyler J. Pardun (tyler.pardun@noaa.gov)'
        ncfile.history = f'Created {datetime.now()}'
        
        # Define dimensions
        time = np.array([x for l in time for x in l])
        time_dim = ncfile.createDimension('time', time.size)
        
        # Create variables for latitude, longitude, and temperature
        var = ncfile.createVariable('time', 'f4', ('time',))
        var.units = 'seconds since 1970-01-01'
        var[:] = time
        
        var = ncfile.createVariable('latitude', 'f4', ('time',))
        var.units = 'degrees_north'
        var[:] = np.array([x for l in lat for x in l])
        
        var = ncfile.createVariable('longitude', 'f4', ('time',))
        var.units = 'degrees_east'
        var[:] = np.array([x for l in lon for x in l])
        
        var = ncfile.createVariable('elev', 'f4', ('time',))
        var.units = 'm'
        var[:] = np.array([x for l in elev for x in l])
        
        var = ncfile.createVariable('temperature', 'f4', ('time',))
        var.units = 'Celsius'
        var[:] = np.array([x for l in temp for x in l])
        
        var = ncfile.createVariable('pressure', 'f4', ('time',))
        var.units = 'hPa'
        var[:] = np.array([x for l in pres for x in l])
        
        var = ncfile.createVariable('rh', 'f4', ('time',))
        var.units = '%'
        var[:] = np.array([x for l in rh for x in l])
        
        var = ncfile.createVariable('wdir', 'f4', ('time',))
        var.units = 'deg'
        var[:] = np.array([x for l in wdir for x in l])
        
        var = ncfile.createVariable('wspd', 'f4', ('time',))
        var.units = 'm/s'
        var[:] = np.array([x for l in wspd for x in l])
        
        ncfile.close()
    
