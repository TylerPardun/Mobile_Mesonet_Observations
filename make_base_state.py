#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import interpolate_var,get_dx_dy,get_mm_files,compute_thw,haversine,extract_within_radius,rotate_grid
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from mpi4py import MPI
from time import sleep
from glob import glob
import xarray as xr
import metpy.constants as const
from datetime import datetime,timedelta,timezone
from netCDF4 import date2num,num2date
import metpy.calc as mpcalc
from metpy.units import units
import nexradaws
from scipy.spatial.distance import pdist
import pyart
import os
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN

def network_area_km2(super_dx, super_dy, R=None):
    """
    Return area in km^2 of the superobs network.
    super_dx, super_dy: arrays in km (storm-relative)
    R: optional analysis radius in km (e.g., 400) for fallback
    """
    XY = np.vstack([super_dx, super_dy]).T

    # Need at least 3 non-collinear points for ConvexHull
    if XY.shape[0] >= 3:
        try:
            hull = ConvexHull(XY)
            A_km2 = float(hull.volume)     # 2-D: 'volume' == area
            if A_km2 > 0.0:
                return A_km2
        except Exception:
            pass  # fall through to fallbacks

    # Fallbacks:
    if R is not None:
        # Use analysis disk area
        return float(np.pi * R * R)

    # Bounding-box fallback (very conservative)
    dx_span = np.ptp(super_dx) if super_dx.size else 0.0
    dy_span = np.ptp(super_dy) if super_dy.size else 0.0
    A_box = dx_span * dy_span
    return float(A_box) if A_box > 0.0 else 1.0  # last-resort epsilon

def wmean(arr,w): 
    return np.nansum(w * arr) / np.nansum(w)


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

def derive_ave_elev(dir_mesonet, sticknet_dir, atime, meso_time, meso_lat, meso_lon, time_to_space_conversion=10, max_distance_range=50):
    """
    Computes the average elevation of all mobile mesonets within the analysis period.

    Parameters:
    - dir_mesonet (str): Path to mobile mesonet files.
    - atime (datetime): Analysis time.
    - time_to_space_conversion (int, optional): Time window in minutes (default: 10).

    Returns:
    - H (float): Derived scale height.
    - Z (float): Mean elevation.
    """
    # Get mobile mesonet files
    mmfs = get_mm_files(dir_mesonet, atime)
    all_data = {key: [] for key in ['lat', 'lon', 'elev', 'dx','dy']}
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))

    for fi in mmfs:
        with xr.open_dataset(fi) as m:
            # Convert time and apply time filter
            tm = m.time.values
            mask = (tm >= date2num(atime - time_window, 'seconds since 1970-01-01')) & (tm <= date2num(atime + time_window,'seconds since 1970-01-01'))

            # Extract relevant data and filter
            elev,lat,lon,tm = m.elev.sel(time=mask).values,m.lat.sel(time=mask).values,m.lon.sel(time=mask).values,num2date(m.time.sel(time=mask).values,'seconds since 1970-01-01',only_use_cftime_datetimes=False)
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
    idist = np.where(dist<=max_distance_range)[0]
    all_elev = all_data['elev'][idist]

    #Sticknet elevation
    stick_elev = get_sticknet_elevation(atime,sticknet_dir,meso_time, meso_lat, meso_lon, max_distance_range=max_distance_range)

    Z = np.nanmean(np.append(all_elev,stick_elev))
    
    return Z

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


def asos_contamination(i, radar_dir, asos_rad_outdir, ucoords, atime, conn, ref_asos_dist=10, asos_ref_threshold=5, sweep=0):
    
    rad = pd.read_excel(radar_dir + 'NEXRAD_Locations.xlsx')
    dist = haversine(ucoords[i,1], ucoords[i,0], rad['lon'].values, rad['lat'].values)
    isort = np.argsort(dist)
    
    def get_available_scans(rval, atime):
        """Retrieve available radar scans for a given radar and time."""
        try:
            return conn.get_avail_scans(atime.strftime('%Y'), atime.strftime('%m'), atime.strftime('%d'), rval)
        except TypeError:
            return []
    
    def parse_file_dates(arad_files, rval):
        """Parse datetime from radar file names."""
        rad_file_dates = []
        for val in arad_files:
            filename = str(val).split(' - ')[-1].split('/')[-1]
            try:
                date_str = f"{filename.split(rval)[1].split('_')[0]} {filename.split(rval)[1].split('_')[1]}"
                rad_file_dates.append(datetime.strptime(date_str, '%Y%m%d %H%M%S'))
            except ValueError:
                rad_file_dates.append(datetime.strptime(date_str, ' %Y%m%d%H%M%S'))
        return np.array(rad_file_dates)
    
    def get_closest_radar_file(rval, atime):
        """Find the closest radar file to the given time."""
        arad_files = get_available_scans(rval, atime)
        if not arad_files:
            return None, None
        
        rad_file_dates = parse_file_dates(arad_files, rval)
        tot_seconds = np.abs(rad_file_dates - atime).astype('timedelta64[s]').astype(int)
        ifile = tot_seconds.argmin()
        return arad_files[ifile], rad_file_dates[ifile]
    
    def download_and_read_radar_file(rad_file, sweep=0):
        sleep(np.random.choice(np.linspace(0.01,0.1,100),1)[0])
        """Download and read the radar file, retrying with the next closest radar if necessary."""
        if not os.path.exists(asos_rad_outdir + '/' + rad_file.filename):
            conn.download(rad_file, asos_rad_outdir)
        
        try:
            return pyart.io.read_nexrad_archive(asos_rad_outdir + '/' + rad_file.filename)
        except Exception:
            return None
    
    
    # Iterate through sorted radar locations by distance
    for idist in range(len(isort)):
        rval = rad['STATION ID'].values[isort[idist]]
        rad_file, rad_file_date = get_closest_radar_file(rval, atime)
    
        # If no valid file is found, move to the next closest radar
        if rad_file is None or ('MDM' in rad_file.filename) or ('tar' in rad_file.filename):
            continue
    
        # Download and read the radar file
        radar = download_and_read_radar_file(rad_file, sweep=sweep)
        # If the radar file is unreadable, try the next closest radar
        if radar is None:
            continue
        
        # Extract reflectivity
        try:
            ref = radar.get_field(sweep=sweep, field_name='reflectivity')
        except IndexError:
            sweep = 1
            ref = radar.get_field(sweep=sweep, field_name='reflectivity')
    
        # Get lat/lon of radar gates
        y, x, _ = radar.get_gate_lat_lon_alt(sweep=sweep)
        asos_ref,rlat,rlon = extract_within_radius(y, x, ref, ucoords[i,0], ucoords[i,1], ref_asos_dist=ref_asos_dist)
    
        # Calculate 95th percentile of reflectivity
        mean_ref = np.nanpercentile(asos_ref, 95)
        contam = True if asos_ref_threshold <  mean_ref else False
        break

    return contam

def get_base_state_info(asos_dir,radar_dir,asos_rad_outdir,atime,meso_lon,meso_lat,meso_time,Z,time_to_space_conversion=10,asos_distance=400,asos_ref_threshold=5,ref_asos_dist=10):

    #Read in each file and find any observations
    files = sorted(glob(asos_dir+'*{}*.nc*'.format(atime.year)))
    files_okm = sorted(glob(asos_dir+'*oklahoma_mesonet*.nc*'))
    files = np.unique(np.append(files,files_okm))
    
    # Initialize lists for all data
    all_data = {key: [] for key in ['lat', 'lon', 'pres', 'pres_corr', 'temp', 'rh', 'u','v', 'qv', 'dew', 'th', 'thv', 'the', 'thw', 'dist', 'elev', 'time']}
    time_window = timedelta(seconds=int((time_to_space_conversion * 60) / 2))
    
    for fi in files:
        with xr.open_dataset(fi,decode_times=False) as m:
            tm = m.time.values
            
            #Index the observations within proximity(if any)
            mask = (tm >= date2num(atime - time_window, 'seconds since 1970-01-01')) & (tm <= date2num(atime + time_window,'seconds since 1970-01-01'))
    
            #Get the data we need
            lat,lon,time = m.latitude.sel(time=mask).values,m.longitude.sel(time=mask).values,m.time.sel(time=mask).values
            temp,rh,pres = m.temperature.sel(time=mask).values,m.rh.sel(time=mask).values,m.pressure.sel(time=mask).values #C, %, hPa
            wspd,wdir,elev = m.wspd.sel(time=mask).values,m.wdir.sel(time=mask).values,m.elev.sel(time=mask).values
            u,v = mpcalc.wind_components(np.array(wspd)*(units.meters/units.seconds), np.array(wdir)*units.degrees)
            u,v = u.m, v.m
            del wspd,wdir #only need u and v
    
            valid = (~np.isnan(elev)) & (~np.isnan(lat)) & (~np.isnan(lon)) & (lon<0) & (~np.isnan(temp)) & (~np.isnan(rh)) & (~np.isnan(pres)) & (~np.isnan(u)) & (~np.isnan(v)) & (temp>-100) & (rh>-100) & (pres>-100)
    
            if np.any(valid):
                lat,lon,time = lat[valid],lon[valid],time[valid]
                temp,pres,rh = temp[valid],pres[valid],rh[valid]
                u,v,elev = u[valid],v[valid],elev[valid]
                
                #Convert time to datetime
                time = num2date(time, 'seconds since 1970-01-01')
                time = np.array([datetime(v.year,v.month,v.day,v.hour,v.minute,v.second) for v in time])
                dist = compute_distances(meso_time, meso_lat, meso_lon, time, lat, lon)
                dx,dy = dist[:,0]/1000,dist[:,1]/1000
                dist = np.sqrt((dx**2)+(dy**2))
                idist = np.where(dist<=asos_distance)[0]
                
                lat,lon,time = lat[idist],lon[idist],time[idist]
                temp,pres,rh = temp[idist],pres[idist],rh[idist]
                u,v,elev,dist = u[idist],v[idist],elev[idist],dist[idist]
                dx,dy = dx[idist], dy[idist]

                #RH in percent
                if rh.size > 1:
                    if rh[0] < 1:
                        rh = rh*100
                else:
                    if rh < 1:
                        rh = rh*100
                        
    
                #Correct the pressure to some average altitude
                qv = mpcalc.mixing_ratio_from_relative_humidity(np.array(pres)*units.hPa,np.array(temp)*units.degC,np.array(rh)*units.percent).m
                tv = mpcalc.virtual_temperature(np.array(temp)*units.degC,qv*units.dimensionless).m
                p_corr = pres * np.exp(((elev-Z)/(Rd*(tv+273.15)))*g)
        
                #Compute all other variables we want to retain using the raw pressure
                dew = mpcalc.dewpoint_from_relative_humidity(np.array(temp)*units.degC, np.array(rh)*units.percent).m
                qvs = mpcalc.saturation_mixing_ratio(np.array(pres)*units.hPa,np.array(temp)*units.degC).m
                th = mpcalc.potential_temperature(np.array(pres)*units.hPa,np.array(temp)*units.degC).m
                thv = mpcalc.virtual_potential_temperature(np.array(pres)*units.hPa,np.array(temp)*units.degC,qv*units.dimensionless).m
                
                #Compute theta-e and theta-w
                the = mpcalc.equivalent_potential_temperature(np.array(pres)*units.hPa,np.array(temp)*units.degC,np.array(dew)*units.degC).m
                thw = np.array([compute_thw(temp[i], pres[i], rh[i]) for i in range(pres.size)])
    
                #Append the data
                all_data['lat'].append(lat)
                all_data['lon'].append(lon)
                all_data['pres'].append(pres)
                all_data['pres_corr'].append(p_corr)
                all_data['temp'].append(temp)
                all_data['rh'].append(rh)
                all_data['u'].append(u)
                all_data['v'].append(v)
                all_data['qv'].append(qv)
                all_data['dew'].append(dew)
                all_data['th'].append(th)
                all_data['thv'].append(thv)
                all_data['the'].append(the)
                all_data['thw'].append(thw)
                all_data['dist'].append(dist)
                all_data['elev'].append(elev)
                all_data['time'].append(time)
                
    
    #Flatten all arrays
    for ky in list(all_data.keys()):
        all_data[ky] = np.array([x for l in all_data[ky] for x in l])

    if all_data['pres'].size > 0:
        ############################################################
        # Retain convectively uncontaminated observations
        ############################################################
        
        #Get unique coordinates of ASOS stations
        coords = np.column_stack((all_data['lat'], all_data['lon']))
        ucoords,icoord = np.unique(coords,axis=0,return_index=True)
        ucoords_lat,ucoords_lon = ucoords[:,0], ucoords[:,1]
        
        #Download NEXRAD files (if needed)
        conn = nexradaws.NexradAwsInterface()
        
        iasos = np.array([asos_contamination(i,radar_dir,asos_rad_outdir,ucoords,atime,conn,ref_asos_dist=ref_asos_dist,asos_ref_threshold=asos_ref_threshold) for i in range(ucoords.shape[0])])
        
        #Use uncontaminated obs
        ikeep = np.where(~iasos)[0] #Returns True if contaminated
        
        if len(ikeep) > 0:
            #Translate the indices to all observations
            igood = [np.where((all_data['lat']==ucoords[ikeep[k]][0])&(all_data['lon']==ucoords[ikeep[k]][1]))[0] for k in range(len(ikeep))]
            igood = np.array([x for l in igood for x in l])
            
            #Retain only the uncontaminated observations
            for ky in list(all_data.keys()):
                all_data[ky] = all_data[ky][igood]
        
        ########################################################
        # Compute weighted average (Markowski (2002))
        ########################################################
        
        # 1) Build XY in km relative to the meso (all points)
        dist = compute_distances(meso_time, meso_lat, meso_lon,
                                 all_data['time'], all_data['lat'], all_data['lon'])
        dx, dy = dist[:, 0]/1000 , dist[:, 1]/1000
        X = np.vstack([dx, dy]).T  # shape (N_pts, 2)  -- ALL points
        
        # 2) Cluster ALL points to make superobs
        labels = DBSCAN(eps=30.0, min_samples=1).fit(X).labels_
        cluster_ids = np.unique(labels)
        nC = cluster_ids.size
        
        # 3) Allocate superobs
        super_lat  = np.empty(nC)
        super_lon  = np.empty(nC)
        super_dx   = np.empty(nC)
        super_dy   = np.empty(nC)
        super_vars = {k: np.empty(nC) for k in ['pres','temp','rh','u','v','qv','dew','th','thv','the','thw','lat','lon','elev','pres_corr']}
        
        # 4) Aggregate by cluster (means -> superobs)
        for j, lab in enumerate(cluster_ids):
            idx = np.where(labels == lab)[0]
            super_lat[j] = np.nanmean(all_data['lat'][idx])
            super_lon[j] = np.nanmean(all_data['lon'][idx])
            # centroid in storm-relative km
            super_dx[j], super_dy[j] = np.nanmean(X[idx, :], axis=0)
            for k in super_vars:
                super_vars[k][j] = np.nanmean(all_data[k][idx])
        
        # 5) Compute Dn and Barnes weights **using the superobs**
        # (area from convex hull; use projected km coords or a geodesic area function)
        # super_dx/super_dy are km; N is number of superobs
        # r_km must be distance from the mesocyclone for the SUPEROBS (not raw obs)
        r_km = np.hypot(super_dx, super_dy)  # km
        
        # Use the analysis radius to compute Dn (don't use the hull here)
        R = float(asos_distance)             # e.g., 400 km
        N = len(super_dx)                    # number of superobs
        Dn = np.sqrt(np.pi * R * R / N)      # km  (network spacing in the disk)
        
        k0 = (2.0 * Dn / np.pi) ** 2         # km^2   (Koch/Markowski)
        w  = np.exp(-(r_km**2) / (2.0 * k0)) # Barnes weights (note the 2*k0)
        
        base_pres_corr = wmean(super_vars['pres_corr'],w)
        base_pres = wmean(super_vars['pres'],w)
        base_temp = wmean(super_vars['temp'],w)
        base_rh = wmean(super_vars['rh'],w)
        base_u = wmean(super_vars['u'],w)
        base_v = wmean(super_vars['v'],w)
        base_qv = wmean(super_vars['qv'],w)
        base_dew = wmean(super_vars['dew'],w)
        base_th = wmean(super_vars['th'],w)
        base_thv = wmean(super_vars['thv'],w)
        base_the = wmean(super_vars['the'],w)
        base_thw = wmean(super_vars['thw'],w)
        base_dict = {'pres':base_pres,'pres_corr':base_pres_corr,'temp':base_temp,'rh':base_rh,'u':base_u,'v':base_v,'qv':base_qv,'dew':base_dew,'th':base_th,'thv':base_thv,'the':base_the, 'thw':base_thw}
        return base_dict
    else:
        return np.nan

def process_case(atime, caseID, meso_time,meso_lat,meso_lon,u_storm,v_storm, Z, dir_mesonet, sticknet_dir, asos_dir, radar_dir, asos_rad_outdir,tracking_dir, time_to_space_conversion, max_distance_range, dt_tracking, asos_distance, asos_ref_threshold, normalized_storm_motion):
    

    base = get_base_state_info(asos_dir,radar_dir,asos_rad_outdir,atime,meso_lon,meso_lat,meso_time,Z,time_to_space_conversion=time_to_space_conversion,asos_distance=asos_distance,asos_ref_threshold=asos_ref_threshold,ref_asos_dist=ref_asos_dist)

    #Make sure there aren't any nans
    try:
        qc = np.isnan(base)
        return np.nan
    except:

        #Rotate the base state winds to account for normalized storm motion
        base_ru,base_rv = rotate_grid(base['u'], base['v'], u_storm, v_storm, pivot_x=0,pivot_y=0,target_angle=normalized_storm_motion)
        base_return = {'time': atime, 'pres': base['pres'], 'pres_corr': base['pres_corr'], 'temp': base['temp'], 'rh': base['rh'], 'u': base['u'], 'v': base['v'], 'ru':base_ru, 'rv':base_rv,'qv':base['qv'], 'dew':base['dew'], 'th':base['th'], 'thv':base['thv'], 'the':base['the'], 'thw':base['thw']}
        return base_return 


    
if __name__ == '__main__':
    # MPI Setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #Define paths to files
    dir_transects = './'
    tracking_dir = 'storm_tracks/'
    dir_mesonet = 'mobile_mesonet_processed2/'
    sticknet_dir = 'sticknets/'
    asos_dir = 'asos/'
    radar_dir = './'
    asos_rad_outdir = 'nexrad_asos/'

    '''
    #Define paths to files
    dir_transects = '../data/'
    tracking_dir = '../data/storm_tracks/'
    dir_mesonet = '../data/mobile_mesonet_processed2/'
    sticknet_dir = '../data/sticknets/'
    asos_dir = '../data/asos/'
    radar_dir = '../data/'
    asos_rad_outdir = '../data/nexrad_asos/'
    '''
    
    #Define some parameters
    time_to_space_conversion = 5 #min - Used to grab MM observations for analysis
    dt_tracking = 1/60 #min for interpolation of storm tracks
    max_distance_range = 100 #km for inclusion of sticknet elevation data relative to the mesocyclone to capture mean elevation data
    asos_distance = 400 #km - Maximum radius to search for ASOS stations relative to the mesocyclone location
    ref_asos_dist = 10 #km - Radius to compute dBZ relative to an ASOS station to determine if uncontaminated
    asos_ref_threshold = 10 #dBZ - Used to as minimum to define convectively uncontaminated ASOS observations
    normalized_storm_motion = 90 #deg, westerly storm motion by default

    #Define constants
    Rd = float(np.array(const.dry_air_gas_constant))
    g = float(np.array(const.earth_gravity))
    
    analysis_times = np.arange(-20,20+1,time_to_space_conversion)
    
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
            #print(atimes[ix])
            case_data = process_case(atimes[ix],caseIDs[ix], meso_times[ix], meso_lats[ix] ,meso_lons[ix], u_storms[ix], v_storms[ix], Zs[ix], dir_mesonet, sticknet_dir, asos_dir, radar_dir, asos_rad_outdir, tracking_dir, time_to_space_conversion, max_distance_range, dt_tracking, asos_distance, asos_ref_threshold, normalized_storm_motion)
            
            try:
                local_base[case_data['time'].strftime('%Y%m%d%H%M')] = case_data
            except:
                local_base[local_tasks[x].strftime('%Y%m%d%H%M')] = np.nan

        #Write a pickle file for each rank
        filename = f'base_state_rank{rank}_tsc{time_to_space_conversion}_{mode}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(local_base, f)

    
    # Only the root process saves the combined data
    if rank == 0:
        # Combine all gathered results
        all_base = {}
        for file in sorted(glob('base_state_rank*_tsc*.pkl')):
            with open(file, 'rb') as f:
                all_base.update(pickle.load(f))
        
        with open('base_state_data_tsc_{}_min.pkl'.format(time_to_space_conversion), 'wb') as f:
            pickle.dump(all_base, f)

        #Remove the rank pickle files
        #[os.remove(f) for f in sorted(glob('base_state_rank*_tsc*.pkl'))]
