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
KW_COMPRESSION = dict(zlib=True)

DST_VARS = [
        ('x_wind_10m', 'instantaneous'),
        ('y_wind_10m', 'instantaneous'),
        ('air_temperature_2m', 'instantaneous'),
        ('specific_humidity_2m', 'instantaneous'),
        ('integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time', 'accumulated'),
        ('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time', 'accumulated'),
        ('air_pressure_at_sea_level', 'instantaneous'),
        ('precipitation_amount_acc', 'accumulated'),
        ('integral_of_snowfall_amount_wrt_time', 'accumulated'),
        ]
VALID_DATE = lambda x : dt.datetime.strptime(x, '%Y%m%d')

def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
            description='''
            preprocess AROME ensemble
            - want 1st record to be at 00:00, and no more than 1 day per file
            - also need to deaccumulate the integrated variables (they are relative to 00:00 for
              the start date of the file)
            ''')
    parser.add_argument('date', type=VALID_DATE,
            help='input date (YYYYMMDD)')
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

def get_ar_inst_var(ar_ds, ar_ds2, dst_var_name, shp):
    '''
    get instantaneous variable
    Need 2 files (00:00 is 24:00 from the day before)

    Parameters:
    -----------
    ar_ds : netCDF4.Dataset
        source AROME dataset
    ar_ds : netCDF4.Dataset
        source AROME dataset from day before
    var_name : str
        variable name
    shp : tuple
        shape of destination variable

    Returns:
    --------
    array : numpy.ndarray
    '''
    array = np.zeros(shp)
    # get 03:00, 06:00, ..., 21:00 for current day
    # from current day's file
    array[1:AR_TIME_RECS_PER_DAY+1,:,:,:] = ar_ds.variables[
                dst_var_name][:AR_TIME_RECS_PER_DAY-1,0,:,:,:]
    # 00:00 of current day from previous day's file
    array[0,:,:,:] = ar_ds2.variables[
            dst_var_name][AR_TIME_RECS_PER_DAY-1:AR_TIME_RECS_PER_DAY,0,:,:,:]
    return array

def get_ar_accum_var(ar_ds, ar_ds2, dst_var_name, shp):
    '''
    get accumulated variable
    Need 2 files (00:00 is 24:00 from the day before)

    Parameters:
    -----------
    ar_ds : netCDF4.Dataset
        source AROME dataset
    ar_ds : netCDF4.Dataset
        source AROME dataset from day before
    var_name : str
        variable name
    shp : tuple
        shape of destination variable

    Returns:
    --------
    array : numpy.ndarray
    '''
    array = np.zeros(shp)
    # get 03:00, 06:00, ..., 21:00 for current day
    # from current day's file
    tmp = ar_ds.variables[
            dst_var_name][:AR_TIME_RECS_PER_DAY-1,0,:,:,:]
    array[1:AR_TIME_RECS_PER_DAY+1,:,:,:] = np.gradient(tmp, axis=0)
    # 00:00 of current day from previous day's file
    tmp = ar_ds2.variables[
            dst_var_name][AR_TIME_RECS_PER_DAY-2:AR_TIME_RECS_PER_DAY,0,:,:,:]
    array[0,:,:,:] = np.diff(tmp, axis=0)
    return array

def create_dimensions(ar_ds, dst_ds, dst_vec):
    '''
    add dimensions, lon, lat and projection

    Parameters:
    -----------
    ar_ds : netCDF4.Dataset
        source AROME dataset
    dst_ds : netCDF4.Dataset
        target AROME dataset
    dst_vec : dict
        three vectors with destination coordinates (time, ensemble_member, y, x)
    '''
    # add the dimensions
    for dim_name, dim_vec in dst_vec.items():
        dlen = {'time': None}.get(dim_name, len(dim_vec)) # time should be unlimited
        dst_dim = dst_ds.createDimension(dim_name, dlen)
        dst_var = dst_ds.createVariable(
                dim_name, 'f8', (dim_name,), **KW_COMPRESSION)
        ar_var = ar_ds.variables[dim_name]
        for ncattr in ar_var.ncattrs():
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var[:] = dim_vec

    # add projection variable
    dst_var_name = 'projection_lambert'
    dst_var = dst_ds.createVariable(dst_var_name, 'i4',)
    ar_var = ar_ds.variables[dst_var_name]
    for ncattr in ar_var.ncattrs():
        dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))

    # add lon/lat
    for var_name in ['longitude', 'latitude']:
        dst_var = dst_ds.createVariable(
                var_name, 'f8', ('y', 'x',), **KW_COMPRESSION)
        ar_var = ar_ds.variables[var_name]
        for ncattr in ar_var.ncattrs():
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var[:] = ar_var[:]

def create_variables(ar_ds, ar_ds2, dst_ds, dst_vec, dst_shape):
    '''
    create the time-dependent variables

    Parameters:
    -----------
    ar_ds : netCDF4.Dataset
        source AROME dataset
    ar_ds2 : netCDF4.Dataset
        source AROME dataset from previous day
        (we need the 24:00 record for that day)
    dst_ds : netCDF4.Dataset
        target AROME dataset
    dst_vec : dict
        three vectors with destination coordinates (time, ensemble_member, y, x)
    dst_shape : tuple
        shape of destination grid
    '''
    get_var_funs = dict(
            instantaneous=get_ar_inst_var,
            accumulated=get_ar_accum_var,
            )
    dims = ('time', 'ensemble_member', 'y', 'x',)
    for dst_var_name, vtype in DST_VARS:
        print(dst_var_name, vtype, sep=': ')
        dst_var = dst_ds.createVariable(
                dst_var_name, 'f4', dims, **KW_COMPRESSION)
        array = get_var_funs[vtype](
                ar_ds, ar_ds2, dst_var_name, dst_shape)
        ar_var = ar_ds.variables[dst_var_name]
        for ncattr in ar_var.ncattrs():
            if ncattr in ['_FillValue', 'grid_mapping']:
                continue
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var.setncattr('grid_mapping', 'projection_lambert')
        dst_var[:] = np.reshape(array, dst_shape)

def run(args):
    '''
    process one day's AROME file

    Parameters:
    -----------
    args : argparse.Namespace
        parsed command line arguments
    '''

    ar_ds, ar_ds2, ar_proj, ar_pts = open_arome(args.date)
    dst_vec, dst_shape = set_destination_coordinates(ar_proj, ar_ds, ar_ds2)

    # save the output
    outfile = args.date.strftime(AR_FILEMASK_NEW)
    outdir = os.path.split(outfile)[0]
    nsl.make_dir(outdir)
    print('Exporting %s' %outfile)
    with Dataset(outfile, 'w') as dst_ds:
        create_dimensions(ar_ds, dst_ds, dst_vec)
        create_variables(ar_ds, ar_ds2, dst_ds,
                dst_vec, dst_shape)

    for ds in [ar_ds, ar_ds2]:
        ds.close()

if __name__ == '__main__':
    run(parse_args(sys.argv[1:]))
