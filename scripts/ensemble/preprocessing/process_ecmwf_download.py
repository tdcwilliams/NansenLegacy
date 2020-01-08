#!/usr/bin/env python
'''
downloaded files are for March 2018
- legacy_201803_3h.nc has 1st day of forecast for each start date
- legacy_201803_3h.[2,3,4].nc has 2nd/3rd/4th day of forecast
'''

import os
import sys
import argparse
import datetime as dt

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from pynextsim.projection_info import ProjectionInfo
import pynextsim.lib as nsl

# containers for interpolated data
DST_DATA = {}

# Celsius to Kelvin conversion
_KELVIN = 273.15 # [C]

# filenames
EC_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/legacy_201803_3h.%i.nc'
EC_REFDATE = dt.datetime(1900, 1, 1) #ref date in downloaded file
DST_REFDATE = dt.datetime(1950, 1, 1) #ref date hard-coded into neXtSIM
EC_ACC_VARS = ['ssrd', 'strd', 'tp', 'sf']
EC_RECS_PER_DAY = 8 #number of recs per day in downloaded file
NDAYS = 3

outdir = '/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/3h'
if not os.path.exists(outdir):
    os.mkdir(outdir)
if 0:
    #6h resolution
    TIME_RECS_PER_DAY = 4 #number of recs per day in output file
    NEW_FILEMASK = os.path.join(outdir, '6h', 'ec2_start%Y%m%d.nc')
else:
    #3h resolution
    TIME_RECS_PER_DAY = 8 #number of recs per day in output file
    NEW_FILEMASK = os.path.join(outdir, '3h', 'ec2_start%Y%m%d.nc')
TIME_RES_RATIO = int(EC_RECS_PER_DAY/TIME_RECS_PER_DAY)

# Destination variables
DST_VARS = {
    '10U' : 'u10',
    '10V' : 'v10',
    '2T': 't2m',
    '2D': 'd2m',
    'SSRD': 'ssrd',
    'STRD' : 'strd',
    'MSL': 'msl',
    'TP': 'tp',
    'TCC': 'tcc',
    'SF': 'sf',
    }
if 0:
    # test on smaller subset of variables
    #dst_var = 'TP'
    dst_var = '2T'
    DST_VARS = {dst_var: DST_VARS[dst_var]}

DST_DIMS = {
        'time' : 'time',
        'lat': 'latitude',
        'lon': 'longitude',
        }

VALID_DATE = lambda x : dt.datetime.strptime(x, '%Y%m%d')
KW_COMPRESSION = dict(zlib=True)

def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
            description="""
            Split downloaded ECMWF file into daily file
            and deaccumulate the required variables""")
    parser.add_argument('date', type=VALID_DATE,
            help='input date (YYYYMMDD)')
    return parser.parse_args(args)

def open_ecmwf():
    """ open downloaded ECMWF file

    Returns
    -------
    ec_ds : netCDF4.Dataset
        ECMWF dataset
    ec_pts : tuple with three 1D-ndarrays
        ECMWF time, latitude, longitude coordinates
    """
    ec_filename = EC_FILEMASK
    return {i-1: Dataset(EC_FILEMASK %i) for i in range(1,NDAYS+1)}

def get_ec2_var(ec_ds, ec_var_name, in_ec2_time_range):
    '''
    need to flip lat, get every 2nd record

    Parameters:
    -----------
    ec_ds : netCDF4.Dataset
    vname : str
        name of ec2 variable
    in_ec2_time_range : np.ndarray(bool)
        tells which time indices to use

    Returns:
    --------
    ec2_var : np.ndarray(float)
    '''
    v = ec_ds.variables[
                ec_var_name][in_ec2_time_range,::-1,:]#flip lat
    return v[::TIME_RES_RATIO] #flip lat and get every 2nd rec

def test_ec2_time_range(ec_ds, date):
    '''
    Parameters:
    -----------
    ec_ds : netCDF4.Dataset
    date : datetime.datetime

    Returns:
    --------
    in_range : numpy.ndarray (bool)
    '''
    ec_time_raw = ec_ds.variables['time'][:].astype(float)
    ec_time = np.array([
        EC_REFDATE + dt.timedelta(hours=h)
        for h in ec_time_raw])
    in_range = (ec_time >= date)*(ec_time < date + dt.timedelta(1))
    return in_range

def set_destination_coordinates(ec_ds, in_ec2_time_range):
    """ Generate coordinates on the destination grid

    Parameters
    ----------
    ec_ds : netCDF4.Dataset
    in_ec2_time_range: np.ndarray(bool)
        times in range for the day's date

    Returns
    -------
    dst_vec : dict
        three vectors with destination coordinates: time, lat, lon
    """
    # coordinates on destination grid
    # X,Y (NEXTSIM)
    time = ec_ds.variables['time'][in_ec2_time_range][::TIME_RES_RATIO]
    time_shift = (DST_REFDATE - EC_REFDATE).days*24.
    return {
        'time': time - time_shift,
        'lat': ec_ds.variables['latitude'][::-1],
        'lon': ec_ds.variables['longitude'][:],
    }

def export(outfile, ec_ds, dst_vec):
    """ Export split product

    Parameters
    ----------
    outfile : str
        netcdf output filename
    ec_ds : netCDF4.Dataset
        source ECMWF dataset
    dst_vec : dict
        three vectors with destination coordinates (time, lat, lon)
    """
    # Create dataset for output
    skip_var_attr = ['_FillValue', 'grid_mapping']
    # create dataset
    print('Exporting %s' %outfile)
    with Dataset(outfile, 'w') as dst_ds:
        # add dimensions
        for dim_name, dim_vec in dst_vec.items():
            dlen = {'time': None}.get(dim_name, len(dim_vec)) #time should be unlimited
            dtype = {'time': 'f8'}.get(dim_name, 'f4') #time should be double
            dst_dim = dst_ds.createDimension(dim_name, dlen)
            dst_var = dst_ds.createVariable(
                    dim_name, dtype, (dim_name,), **KW_COMPRESSION)
            ec_var = ec_ds.variables[DST_DIMS[dim_name]]
            for ncattr in ec_var.ncattrs():
                if [dim_name, ncattr] == ['time', 'units']:
                    units = DST_REFDATE.strftime(
                            'hours since %Y-%m-%d 00:00:00.0') #need to change ref time
                    dst_var.setncattr(ncattr, units)
                else:
                    dst_var.setncattr(ncattr, ec_var.getncattr(ncattr))
            dst_var[:] = dim_vec

        # add processed variables
        for dst_var_name, ec_var_name in DST_VARS.items():
            dst_var = dst_ds.createVariable(dst_var_name, 'f4',
                    ('time', 'lat', 'lon'),
                    **KW_COMPRESSION)
            ec_var = ec_ds.variables[ec_var_name]
            for ncattr in ec_var.ncattrs():
                if ncattr in skip_var_attr:
                    continue
                dst_var.setncattr(ncattr, ec_var.getncattr(ncattr))
            dst_var[:] = DST_DATA[dst_var_name]
def deaccumulate(arr):
    # in neXtSIM we want the rate,
    # and convert accumulated variables to rates
    # by dividing by the forcing resolution in seconds.
    # Hence we need to double the rate if we lower the resolution.
    v = np.gradient(arr, axis=0)
    v[v<0] = 0.
    return v

def merge_days(dst_data, dst_vecs):
    lat = dst_vecs[0]['lat']
    lon = dst_vecs[0]['lon']
    time = []
    for i in range(NDAYS):
        time += list(dst_vecs[i]['time'])
    dst_vec = dict(time=np.array(time), lat=lat, lon=lon)
    shp = (len(time), len(lat), len(lon))
    for v in DST_VARS:
        DST_DATA[v] = np.zeros(shp)
        for i, arr in enumerate(dst_data[v]):
            itime = np.array(
                    range(i*EC_RECS_PER_DAY, (i+1)*EC_RECS_PER_DAY)
                    )
            DST_DATA[v][itime,:,:] = arr

        # deaccumulate once we have all the time records to get a better
        # estimate for the accumulation rate
        if DST_VARS[v] in EC_ACC_VARS:
            print('deaccumulate %s' %v)
            DST_DATA[v] = deaccumulate(DST_DATA[v])
    return dst_vec

def run(args):
    '''
    make the file

    Parameters:
    -----------
    args : argparse.Namespace
    '''
    outdir = os.path.split(NEW_FILEMASK)[0]
    nsl.make_dir(outdir)
    outfile = os.path.join(outdir, args.date.strftime(NEW_FILEMASK))

    # open arome file and ecmwf file for each day of forecast
    ec_dsets = open_ecmwf()
    dst_data = {v: [] for v in DST_VARS}
    dst_vecs = []
    for i, ec_ds in ec_dsets.items():
        '''
        loop over each day of forecast:
        - eg 1st day of forecast is in legacy*.1.nc,
             2nd day of forecast is in legacy*.2.nc
        '''
        print('Day %i: Using file %s' %(i,ec_ds.filepath()))
        in_ec2_time_range = test_ec2_time_range(ec_ds, args.date + dt.timedelta(i))
        dst_vecs.append(
                set_destination_coordinates(ec_ds, in_ec2_time_range)
                )

        # fetch, interpolate and blend all variables from ECMWF and AROME
        for dst_var_name in DST_VARS:
            # Interpolate data from ECMWF
            ec_var_name = DST_VARS[dst_var_name]
            print('Read', ec_var_name)
            dst_data[dst_var_name].append(
                    get_ec2_var(ec_ds, ec_var_name, in_ec2_time_range)
                    )

    # set DST_DATA (combine days 1,2,... before exporting)
    dst_vec = merge_days(dst_data, dst_vecs)

    # save the output
    export(outfile, ec_ds, dst_vec)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    run(args)
