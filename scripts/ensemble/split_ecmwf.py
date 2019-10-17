#!/usr/bin/env python
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
EC_DATA = {}
DST_DATA = {}

# Celsius to Kelvin conversion
_KELVIN = 273.15 # [C]

# filenames
AR_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/processed/aro_eps_%Y%m%d.nc'
AR_MARGIN = 20 # pixels

EC_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/legacy_201803_3h.nc'
EC_REFDATE = dt.datetime(1900, 1, 1) #ref date in downloaded file
DST_REFDATE = dt.datetime(1950, 1, 1) #ref date hard-coded into neXtSIM
EC_ACC_VARS = ['ssrd', 'strd', 'tp', 'sf']

TIME_RECS_PER_DAY = 4
NEW_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/ec2_start%Y%m%d.nc'

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
    print('Opening ECMWF file %s\n' %ec_filename)
    return Dataset(ec_filename)

def get_ec2_var_accumulated(ec_ds, ec_var_name, in_ec2_time_range):
    '''
    deaccumulate accumulated var's
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
    print('deaccumulating...')
    v = ec_ds.variables[ec_var_name][
            in_ec2_time_range,::-1,:]
    # now deaccumulate and convert from 3h resolution to 6h
    # NB!! can't take central diff between different days as accumulation
    # is restarted each day
    return 2*np.gradient(v, axis=0)[::2,:,:]

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
    if ec_var_name in EC_ACC_VARS:
        return get_ec2_var_accumulated(ec_ds, ec_var_name, in_ec2_time_range)
    else:
        return ec_ds.variables[
                ec_var_name][in_ec2_time_range,::-1,:][::2] #flip lat and get every 2nd rec

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
        three vectors with destination coordinates, time, lat, lon
    """
    # coordinates on destination grid
    # X,Y (NEXTSIM)
    time = ec_ds.variables['time'][in_ec2_time_range][::2]
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
            dst_dim = dst_ds.createDimension(dim_name, dlen)
            dst_var = dst_ds.createVariable(
                    dim_name, 'f8', (dim_name,), **KW_COMPRESSION)
            ec_var = ec_ds.variables[DST_DIMS[dim_name]]
            for ncattr in ec_var.ncattrs():
                att = {
                        'time': DST_REFDATE.strftime('hours since %Y-%m-%d 00:00:00.0')
                        }.get(dim_name,
                                ec_var.getncattr(ncattr)) #need to change ref time
                dst_var.setncattr(ncattr, att)
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

    # open arome file and ecmwf file
    ec_ds = open_ecmwf()
    in_ec2_time_range = test_ec2_time_range(ec_ds, args.date)
    dst_vec = set_destination_coordinates(ec_ds, in_ec2_time_range)

    # fetch, interpolate and blend all variables from ECMWF and AROME
    for dst_var_name in DST_VARS:
        # Interpolate data from ECMWF
        ec_var_name = DST_VARS[dst_var_name]
        print('Read', ec_var_name)
        DST_DATA[dst_var_name] = get_ec2_var(ec_ds, ec_var_name, in_ec2_time_range)
        if 0:
            # test plots
            kw = dict(vmin=0, vmax=3e6)
            for n in range(8): 
                nsl.make_dir('figs0')
                figname = 'figs0/ec_%s_%i.png' %(dst_var_name, n)
                print('Saving %s' %figname)
                plt.imshow(dst_ecd_grd[n,:,:],
                        origin='upper', **kw)
                plt.colorbar()
                plt.title(dst_var_name)
                plt.savefig(figname)
                plt.close() 

                figname = 'figs0/ar_%s_%i.png' %(dst_var_name, n)
                print('Saving %s' %figname)
                plt.imshow(dst_ard_grd_all_members[0][n,:,:],
                        origin='upper', **kw)
                plt.colorbar()
                plt.title(dst_var_name)
                plt.savefig(figname)
                plt.close() 

    # save the output
    export(outfile, ec_ds, dst_vec)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    run(args)
