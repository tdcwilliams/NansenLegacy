#!/usr/bin/env python
import os
import sys
import argparse
import datetime as dt
from collections import OrderedDict

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage import distance_transform_edt

from pynextsim.projection_info import ProjectionInfo
import pynextsim.lib as nsl

AR_FILEMASK_RAW = '/Data/sim/data/AROME_barents_ensemble/raw/aro_eps_%Y%m%d00.nc'
AR_FILEMASK_NEW = '/Data/sim/data/AROME_barents_ensemble/processed/aro_eps_%Y%m%d.nc'
AR_TIME_RECS_PER_DAY = 8

DST_VARS = [
        'x_wind_10m',
        'y_wind_10m',
        'air_temperature_2m',
        'specific_humidity_2m',
        'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time',
        'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time',
        'air_pressure_at_sea_level',
        'precipitation_amount_acc',
        'integral_of_snowfall_amount_wrt_time',
        ]
#DST_VARS = [
#        'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time',
#        ]

def valid_date(date):
    return dt.datetime.strptime(date, '%Y%m%d')

def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
            description='''
            preprocess AROME ensemble
            - want 1st record to be at 00:00, and no more than 1 day per file
            - also need to deaccumulate the integrated variables (they are relative to 00:00 for
              the start date of the file)
            ''')
    parser.add_argument('date', type=valid_date,
            help='input date (YYYYMMDD)')
    parser.add_argument('-o', '--outdir', default='.',
            help='Where to save the generated netcdf files and figures')
    return parser.parse_args(args)

def open_arome(date):
    """ Read AROME geolocation

    Parameters
    ----------
    date : datetime.datetime
        date of AROME file

    Returns
    -------
    ar_ds : netCDF4.Dataset
        AROME dataset
    ar_proj : Pyproj
        AROME projection
    ar_pts : tuple with three 1D-ndarrays
        AROME time, y, x coordinates

    """
    ar_filename = date.strftime(AR_FILEMASK_RAW)
    yday = date - dt.timedelta(1) #get 00:00 record from previous day
    ar_filename2 = yday.strftime(AR_FILEMASK_RAW)

    print('\nOpening AROME file %s' %ar_filename)
    ar_ds = Dataset(ar_filename)
    print('Opening AROME file %s' %ar_filename2)
    ar_ds2 = Dataset(ar_filename2)
    ar_proj = pyproj.Proj(ar_ds.variables['projection_lambert'].proj4)
    ar_x_vec = ar_ds.variables['x'][:].data
    ar_y_vec = ar_ds.variables['y'][:].data
    ar_t_vec = np.linspace(0, 24, AR_TIME_RECS_PER_DAY+1)[:-1]

    return ar_ds, ar_ds2, ar_proj, (ar_t_vec, ar_y_vec, ar_x_vec)

def set_destination_coordinates(ar_proj, ar_ds, ar_ds2):
    """ Generate coordinates on the destination grid

    Parameters
    ----------
    ar_proj : Pyproj
        AROME projection
    ar_ds : netCDF4.Dataset
        main file
    ar_ds2 : netCDF4.Dataset
        previous day's file to get 00:00 time record

    Returns
    -------
    dst_vec : dict
        three vectors with destination coordinates, time, y, x
    dst_shape : tuple
        shape of destination grid
    """

    time = np.zeros((AR_TIME_RECS_PER_DAY,))
    time[0] = ar_ds2.variables['time'][AR_TIME_RECS_PER_DAY-1]
    time[1:] = ar_ds.variables['time'][:AR_TIME_RECS_PER_DAY-1]

    # coordinates on destination grid
    # X,Y (NEXTSIM)
    dst_vec = OrderedDict(
        time=time,
        ensemble_member=ar_ds.variables['ensemble_member'][:],
        y=ar_ds.variables['y'][:],
        x=ar_ds.variables['x'][:],
    )
    dst_shape = [len(v) for v in dst_vec.values()]
    return dst_vec, dst_shape

def proc_ar_var(vname, array):
    if vname in [
            'x_wind_10m',
            'y_wind_10m',
            'air_temperature_2m',
            'specific_humidity_2m',
            'air_pressure_at_sea_level',
            ]:
        # instaneous variable
        print('Getting %s (instantaneous)' %vname)
        return array[1:]

    # deaccumulate
    print('Deaccumulating %s' %vname)
    shp = list(array.shape)
    shp[0] -= 1
    out = np.zeros(shp)
    out[0] = np.diff(array[:2], axis=0) # take diff of 1st 2
    out[1] = array[2] # just grab the next one (03:00 - diff to 00:00)
    out[2:] = np.diff(array[2:], axis=0)
    return out

def export(outfile, ar_ds, ar_ds2, dst_vec, dst_shape):
    """ Export blended product

    Parameters
    ----------
    outfile : str
        netcdf output filename
    ar_ds : netCDF4.Dataset
        source AROME dataset
    dst_vec : dict
        three vectors with destination coordinates (time, ensemble_member, y, x)
    dst_shape : tuple
        shape of destination grid

    """
    # Create dataset for output
    skip_var_attr = ['_FillValue', 'grid_mapping']
    # create dataset
    print('Exporting %s' %outfile)
    dst_ds = Dataset(outfile, 'w')
    # add dimensions
    for dim_name, dim_vec in dst_vec.items():
        dst_dim = dst_ds.createDimension(dim_name, len(dim_vec))
        dst_var = dst_ds.createVariable(dim_name, 'f8', (dim_name,))
        dst_var[:] = dim_vec
        ar_var = ar_ds.variables[dim_name]
        for ncattr in ar_var.ncattrs():
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))

    # add array variables
    for dst_var_name in DST_VARS:
        dst_var = dst_ds.createVariable(dst_var_name, 'f4',
                ('time', 'ensemble_member', 'y', 'x',))
        shp = list(dst_shape)
        shp.insert(1, 1) # extra dimension: height above sea level
        shp[0] += 1 # get the time from before to deaccumulate
        array = np.zeros(shp)
        # get 03:00, 06:00, ..., 21:00 for current day
        array[2:AR_TIME_RECS_PER_DAY+2,:,:,:] = ar_ds.variables[
                    dst_var_name][:AR_TIME_RECS_PER_DAY-1,:,:,:]
        # 21:00 from day before and 00:00 of current day
        array[0:2,:,:,:] = ar_ds2.variables[
                dst_var_name][AR_TIME_RECS_PER_DAY-2:AR_TIME_RECS_PER_DAY,:,:,:]
        if 0:
            print('save %s.npz' %dst_var_name)
            np.savez('%s.npz' %dst_var_name,
                    orig=array, deacc=proc_ar_var(dst_var_name, array))
        dst_var[:] = np.reshape(
                proc_ar_var(dst_var_name, array), dst_shape)
        ar_var = ar_ds.variables[dst_var_name]
        for ncattr in ar_var.ncattrs():
            if ncattr in skip_var_attr:
                continue
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var.setncattr('grid_mapping', 'projection_lambert')

    # add projection variable
    dst_var_name = 'projection_lambert'
    dst_var = dst_ds.createVariable(dst_var_name, 'i4',)
    ar_var = ar_ds.variables[dst_var_name]
    for ncattr in ar_var.ncattrs():
        dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))

    # add lon/lat
    for var_name in ['longitude', 'latitude']:
        dst_var = dst_ds.createVariable(var_name, 'f8', ('y', 'x',))
        ar_var = ar_ds.variables[var_name]
        for ncattr in ar_var.ncattrs():
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var[:] = ar_var[:]

    dst_ds.close()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    nsl.make_dir(args.outdir)

    ar_ds, ar_ds2, ar_proj, ar_pts = open_arome(args.date)
    dst_vec, dst_shape = set_destination_coordinates(ar_proj, ar_ds, ar_ds2)

    # save the output
    outfile = os.path.join(args.outdir,
            args.date.strftime(AR_FILEMASK_NEW))
    export(outfile, ar_ds, ar_ds2, dst_vec, dst_shape)
