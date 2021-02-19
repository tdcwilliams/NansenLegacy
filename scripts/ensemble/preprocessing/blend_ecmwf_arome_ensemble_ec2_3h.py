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
from collections import defaultdict

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

EC_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/3h/ec2_start%Y%m%d.nc'
EC_REFDATE = dt.datetime(1950, 1, 1)
#EC_ACC_VARS = ['ssrd', 'strd', 'tp', 'sf'] #ec2 variables that need to be deaccumulated
EC_ACC_VARS = [] #ec2 variables are now already deaccumulated

NDAYS=2
AR_REFDATE = dt.datetime(1970, 1, 1)
TIME_RECS_PER_DAY = 8
TIME_VEC = np.linspace(0, NDAYS*24, NDAYS*TIME_RECS_PER_DAY+1)
NEW_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/blended_corrected/ec2_arome_blended_ensemble_%Y%m%d.nc'
GRID_FILE = NEW_FILEMASK.replace('%Y%m%d', 'grid')

# Destination grid
# neXtSIM default projection
DST_PI = ProjectionInfo()
EXPORT4D = True

# extent overlaps with AROME grid
DST_X_MIN = -300000
DST_X_MAX =  2800000
DST_Y_MIN = -2030000
DST_Y_MAX =  1500000
DST_X_RES, DST_Y_RES = 2500, 2500
#DST_X_RES, DST_Y_RES = 25000, 25000 #low res for testing

# extent overlaps with FRAM strait
#DST_X_MIN = -450000
#DST_X_MAX =  1300000
#DST_Y_MIN = -2000000
#DST_Y_MAX =  300000
#DST_X_RES, DST_Y_RES = 5000, 5000

VALID_DATE = lambda x : dt.datetime.strptime(x, '%Y%m%d')
KW_COMPRESSION = dict(zlib=True)


def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
            description="Blend ECMWF and AROME ensemble outputs on high resolution grid")
    parser.add_argument('date', type=VALID_DATE,
            help='input date (YYYYMMDD)')
    parser.add_argument('-p', '--plot', action='store_true',
            help='Generate plot of AROME and the new domains')
    #parser.add_argument('-g', '--save-grid', action='store_true',
    #        help='save the grid file')
    return parser.parse_args(args)

def precipitation_amount_acc(ec_TP):
    """ Calculate accumulated precipitation amount

    Parameters
    ----------
    ec_TP : numpy.ndarray
        precipitation_amount in 1 h [m]

    Returns
    ------
    precipit : numpy.ndarray
        Accumulated precipitation_amount [kg/m^2]
    """
    rho = 1000. # water density [kg / m3]
    return ec_TP * rho

def specific_humidity_2m(ec_2D, ec_MSL):
    """ Calculate specific humidity for the atmosphere

    Parameters
    ----------
    ec_2D : numpy.ndarray
        dew_point_temperature [K]
    ec_MSL : numpy.ndarray
        air_pressure_at_sea_level [Pa]

    Returns
    ------
    sphuma : numpy.ndarray
        specific humidity of the atmosphere [kg/kg]

    Note
    ----
    # from https://github.com/nansencenter/nextsim/blob/b700af5c67ead220eb42644207ce71109cd731b2/model/finiteelement.cpp#L5415

    const double aw=6.1121e2, bw=18.729, cw=257.87, dw=227.3;
    const double Aw=7.2e-4, Bw=3.20e-6, Cw=5.9e-10;
    const double alpha=0.62197, beta=0.37803;
    double fa     = 1. + Aw + M_mslp[i]*1e-2*( Bw + Cw*M_dair[i]*M_dair[i] );
    double esta   = fa*aw*std::exp( (bw-M_dair[i]/dw)*M_dair[i]/(M_dair[i]+cw) );
    sphuma = alpha*fa*esta/(M_mslp[i]-beta*fa*esta) ;

    """

    Aw = 7.2e-4
    Bw = 3.20e-6
    Cw = 5.9e-10
    aw = 6.1121e2
    bw = 18.729
    cw = 257.87
    dw = 227.3
    alpha=0.62197
    beta=0.37803

    # due point temperature [C]
    ec_DPT = ec_2D - _KELVIN
    fa = 1. + Aw + ec_MSL * 1e-2 * ( Bw + Cw * ec_DPT**2 )
    esta = fa * aw * np.exp( (bw - ec_DPT / dw) * ec_DPT / (ec_DPT + cw))
    sphuma = alpha * fa * esta / (ec_MSL - beta * fa * esta)
    return sphuma

def integral_of_snowfall_amount_wrt_time(ec_2T, ec_TP):
    """ Calculate integrate snowfall
    Integrate snowfall is equal to total precipitation <ec_TP> where air temperature <ec_2T>
    is below 0 .

    Parameters
    ----------
    ec_2T : numpy.ndarray
        air_temperature [K]
    ec_TP : numpy.ndarray
        precipitation_amount in 1 hour [m]

    Returns
    ------
    snowfall : numpy.ndarray
        Accumulated snowfall amount at surface in 1 hour [kg/m^2]

    """
    snowfall = np.zeros_like(ec_TP)
    freezing = ec_2T < _KELVIN
    snowfall[freezing] = precipitation_amount_acc(ec_TP)[freezing]
    return snowfall

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
    ar_filename = date.strftime(AR_FILEMASK)
    print('\nOpening AROME file %s' %ar_filename)
    ar_ds = Dataset(ar_filename)
    ar_proj = pyproj.Proj(ar_ds.variables['projection_lambert'].proj4)
    ar_x_vec = ar_ds.variables['x'][:].data
    ar_y_vec = ar_ds.variables['y'][:].data
    return ar_ds, ar_proj, (np.copy(TIME_VEC), ar_y_vec, ar_x_vec)

def open_ecmwf(date):
    """ Read ECMWF geolocation

    Parameters
    ----------
    date : datetime.datetime
        date of AROME file

    Returns
    -------
    ec_ds : netCDF4.Dataset
        ECMWF dataset
    ec_pts : tuple with three 1D-ndarrays
        ECMWF time, latitude, longitude coordinates
    """
    ec_filename = date.strftime(EC_FILEMASK)
    print('Opening ECMWF file %s\n' %ec_filename)
    ec_ds = Dataset(ec_filename)
    # make lon cyclic
    ec_lon_vec = np.array(
            list(ec_ds.variables['lon'][:]) + [180])
    # flip lat
    ec_lat_vec = ec_ds.variables['lat'][:]
    return ec_ds, (np.copy(TIME_VEC), ec_lat_vec, ec_lon_vec)

def get_ec2_var(ec_ds, ec_var_name, in_ec2_time_range):
    '''
    need to flip lat, and make lon cyclic

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
            ec_var_name][in_ec2_time_range,:,:]
    nt, nlat, nlon = v.shape
    v2 = np.zeros((nt, nlat, nlon+1))
    v2[:,:,:-1] = v
    v2[:,:,-1] = v2[:,:,0]

    if ec_var_name in EC_ACC_VARS:
        # need to deaccumulate
        return np.gradient(v2, axis=0)
    return v2

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
    in_range = (ec_time >= date)*(ec_time <= date + dt.timedelta(NDAYS))
    return in_range

def set_destination_coordinates(ar_proj, ar_ds):
    """ Generate coordinates on the destination grid

    Parameters
    ----------
    ar_proj : pyproj.Proj
        AROME projection

    Returns
    -------
    dst_vec : dict
        three vectors with destination coordinates, time, y, x
    dst_ecp : Nx3 ndarray
        destination coordinates for ECMWF (time, lat, lon)
    dst_arp : Nx3 ndarray
        destination coordinates for AROME (time, y, x)
    dst_shape : tuple
        shape of destination grid
    """
    # coordinates on destination grid
    # X,Y (NEXTSIM)
    dst_vec = {
        'x': np.arange(DST_X_MIN, DST_X_MAX, DST_X_RES),
        'y': np.arange(DST_Y_MAX, DST_Y_MIN, -DST_Y_RES),
        'time': np.copy(TIME_VEC),
        'ensemble_member': np.arange(ar_ds.variables['ensemble_member'].size),
    }

    dst_t_grd, dst_y_grd, dst_x_grd  = np.meshgrid(dst_vec['time'], dst_vec['y'], dst_vec['x'], indexing='ij')
    dst_shape = dst_t_grd.shape

    # lon,lat (ECMWF)
    dst_lon_grd, dst_lat_grd = DST_PI.pyproj(dst_x_grd, dst_y_grd, inverse=True)
    dst_ecp = np.array([dst_t_grd.flatten(), dst_lat_grd.flatten(), dst_lon_grd.flatten()]).T

    # X,Y (AROME)
    dst_arx_grd, dst_ary_grd = ar_proj(dst_lon_grd, dst_lat_grd, inverse=False)
    dst_arp = np.array([dst_t_grd.flatten(), dst_ary_grd.flatten(), dst_arx_grd.flatten()]).T

    return dst_vec, dst_ecp, dst_arp, dst_shape

def interpolate(src_array, src_x, dst_p, dst_shape):
    """ Interpolate AROME or ECMWF grids onto destination grid

    Parameters
    ----------
    src_array : 3D ndarray
        source array with data on original grid
    src_x : tuple
        three arrays with source coordinates
    dst_p : Nx3 ndarray
        destination coordinates for AROME (time, y, x)
    dst_shape : tuple
        shape of destination grid

    Returns
    -------
    dst_array : 3D ndarray
        interpolated array on new grid

    """
    # limit input array to the size of coordinates
    src_array = src_array[:len(src_x[0]), :len(src_x[1]), :len(src_x[2])]
    # train grid interpolator
    rgi = RegularGridInterpolator(src_x, src_array, bounds_error=False, fill_value=np.nan)
    # apply grid interpolator
    dst_array = rgi(dst_p).reshape(dst_shape)

    return dst_array

def plot_destination_grid(figname, ar_proj, ar_pts):
    """ Plot map with AROME and a destination domains """
    ar_x_grd, ar_y_grd = np.meshgrid(ar_pts[2], ar_pts[1])
    ar_lon_grd, ar_lat_grd = ar_proj(ar_x_grd, ar_y_grd, inverse=True)
    ar_lon_brd = np.hstack([ar_lon_grd[0,:], ar_lon_grd[:,-1], ar_lon_grd[-1,::-1], ar_lon_grd[::-1,0]])
    ar_lat_brd = np.hstack([ar_lat_grd[0,:], ar_lat_grd[:,-1], ar_lat_grd[-1,::-1], ar_lat_grd[::-1,0]])

    dst_extent = np.array([DST_X_MIN, DST_X_MAX, DST_Y_MIN, DST_Y_MAX])
    crs = DST_PI.crs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=crs)
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.set_extent(dst_extent*1.5, crs=crs)
    plt.plot(ar_lon_brd, ar_lat_brd, '.-', transform=ccrs.Geodetic())
    plt.plot([DST_X_MIN, DST_X_MIN, DST_X_MAX, DST_X_MAX, DST_X_MIN],
             [DST_Y_MIN, DST_Y_MAX, DST_Y_MAX, DST_Y_MIN, DST_Y_MIN], 'o-r')

    print('Saving %s' %figname)
    plt.savefig(figname)
    plt.close()

def create_distance(ar_shp):
    """ Create 3D matrix (same shape as AROME data) with distance from the border in pixels

    Parameters
    ----------
    ar_shp : list
        shape of AROME variables (without ensemble dimension)

    Returns
    -------
    distance : numpy.ndarray
        3D array with euclidian distance from the border (pix)

    """
    mask = np.ones(ar_shp[1:], bool)
    # put False into all border pixels (on a 2D matrix)
    mask[0,:] = False
    mask[-1,:] = False
    mask[:,0] = False
    mask[:,-1] = False
    # calculate distance from the border (from False pixels)
    dist2d = distance_transform_edt(mask, return_distances=True, return_indices=False)
    # convert 2D into 3D
    return np.array([dist2d]*len(TIME_VEC))

def blend(ar_data, ec_data, distance):
    """ Blend 2 3D matrices with AROME and ECMWF products into one using weighted average.
    Weight is proportional to the distance from the AROME border.

    Parameters
    ----------
    ar_data : numpy.ndarray
        AROME variable
    ec_data : numpy.ndarray
        ECMWF variable
    distance : numpy.ndarray
        distance from the AROME border (np.nan outside the AROME domain)

    Returns
    -------
    result : numpy.ndarray
        blended AROME and ECMWF

    """
    ar_nan_pix = np.isnan(ar_data)
    ar_weight = distance / AR_MARGIN
    ar_weight[ar_weight > 1] = 1
    ar_weight[ar_nan_pix] = 0
    ar_data[ar_nan_pix] = 0
    return ar_data * ar_weight + ec_data * (1 - ar_weight)

def add_time_to_file(dst_ds, ar_ds):
    dim_name = 'time'
    dst_dim = dst_ds.createDimension(dim_name, None) #time should be unlimited
    dst_var = dst_ds.createVariable(
            dim_name, 'f8', (dim_name,), **KW_COMPRESSION)
    ar_var = ar_ds.variables[dim_name]
    for ncattr in ar_var.ncattrs():
        dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
    # copy arome time
    dst_ds.variables['time'][:] = ar_ds.variables['time'][:]

def add_grid_to_file(dst_ds, ar_ds, dst_ecp, dst_vec, dst_shape):
    # add x,y dimensions
    dtypes = dict(x='f8', y='f8', ensemble_member='i4')
    for dim_name, dim_vec in dst_vec.items():
        dst_dim = dst_ds.createDimension(dim_name, len(dim_vec))
        dst_var = dst_ds.createVariable(
                dim_name, dtypes[dim_name], (dim_name,), **KW_COMPRESSION)
        ar_var = ar_ds.variables[dim_name]
        for ncattr in ar_var.ncattrs():
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var[:] = dim_vec

    # add projection variable
    projection_stereo = dict(
        grid_mapping_name = "polar_stereographic",
        standard_parallel = 60.,
        longitude_of_central_meridian = -45.,
        straight_vertical_longitude_from_pole = -45.,
        latitude_of_projection_origin = 90.,
        earth_radius = 6378.273e3,
        proj4 = DST_PI.pyproj.srs
    )
    dst_var = dst_ds.createVariable('projection_stereo', 'i1')
    dst_var.setncatts(projection_stereo)

    # add lon/lat
    for var_name in ['longitude', 'latitude']:
        dst_var = dst_ds.createVariable(
                var_name, 'f8', ('y', 'x',), **KW_COMPRESSION)
        ar_var = ar_ds.variables[var_name]
        for ncattr in ar_var.ncattrs():
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var[:] = {
                'longitude': dst_ecp[:, 2].reshape(dst_shape)[0],
                'latitude' : dst_ecp[:, 1].reshape(dst_shape)[0],
                }[var_name]

def export3d(outfile0, ar_ds, dst_ecp, dst_vec0, dst_shape):
    """ Export blended product - one file for each ensemble member

    Parameters
    ----------
    outfile : str
        netcdf output filename
    ar_ds : netCDF4.Dataset
        source AROME dataset
    dst_ecp : Nx3 ndarray
        destination coordinates for ECMWF (time, lat, lon)
    dst_vec : dict
        three vectors with destination coordinates (time, y, x)
    dst_shape : tuple
        shape of destination grid

    """
    # Create dataset for output
    skip_var_attr = []#['_FillValue', 'grid_mapping']
    num_ens_mems = len(dst_vec0["ensemble_member"])
    dst_vec = dict(**dst_vec0)
    del(dst_vec["ensemble_member"])
    del(dst_vec["time"])
    for i_ens in range(num_ens_mems):
        outfile = outfile0.replace('.nc', '.mem%.3i.nc' %(i_ens+1))
        print('Exporting %s' %outfile)
        if os.path.exists(outfile):
            print('%s exists - skipping' %outfile)
            continue
        # create dataset
        with Dataset(outfile, 'w') as dst_ds:

            # add dimensions, lon/lat, stereo
            add_time_to_file(dst_ds, ar_ds)
            add_grid_to_file(dst_ds, ar_ds, dst_ecp, dst_vec, dst_shape)

            # add blended variables
            for dst_var_name in DST_VARS:
                dst_var = dst_ds.createVariable(dst_var_name, 'f4',
                        ('time', 'y', 'x',),
                        **KW_COMPRESSION)
                ar_var = ar_ds.variables[dst_var_name]
                for ncattr in ar_var.ncattrs():
                    if ncattr in skip_var_attr:
                        continue
                    dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
                dst_var.setncattr('grid_mapping', 'projection_stereo')
                dst_var[:] = DST_DATA[dst_var_name][:,i_ens,:,:]

def export4d(outfile, ar_ds, dst_ecp, dst_vec0, dst_shape):
    """ Export blended product - single file with ensemble_member dimension

    Parameters
    ----------
    outfile : str
        netcdf output filename
    ar_ds : netCDF4.Dataset
        source AROME dataset
    dst_ecp : Nx3 ndarray
        destination coordinates for ECMWF (time, lat, lon)
    dst_vec : dict
        three vectors with destination coordinates (time, ensemble_member, y, x)
    dst_shape : tuple
        shape of destination grid

    """
    if os.path.exists(outfile):
        print('%s exists - skipping' %outfile)
        return
    # Create dataset for output
    skip_var_attr = []#['_FillValue', 'grid_mapping']
    # create dataset
    print('Exporting %s' %outfile)
    dst_vec = dict(**dst_vec0)
    del(dst_vec["time"])
    with Dataset(outfile, 'w') as dst_ds:
        add_time_to_file(dst_ds, ar_ds)
        add_grid_to_file(dst_ds, ar_ds, dst_ecp, dst_vec, dst_shape)

        # add blended variables
        for dst_var_name in DST_VARS:
            dst_var = dst_ds.createVariable(dst_var_name, 'f4',
                    ('time', 'ensemble_member', 'y', 'x',),
                    **KW_COMPRESSION)
            ar_var = ar_ds.variables[dst_var_name]
            for ncattr in ar_var.ncattrs():
                if ncattr in skip_var_attr:
                    continue
                dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
            dst_var.setncattr('grid_mapping', 'projection_stereo')
            dst_var[:] = DST_DATA[dst_var_name]


# Destination variables
DST_VARS = {
    'x_wind_10m' : {
        'ec_vars': ['10U'],
        'ec_func': lambda x: x,
        },
    'y_wind_10m' : {
        'ec_vars': ['10V'],
        'ec_func': lambda x: x,
        },
    'air_temperature_2m': {
        'ec_vars': ['2T'],
        'ec_func': lambda x: x,
        },
    'specific_humidity_2m': {
        'ec_vars': ['2D', 'MSL'],
        'ec_func': specific_humidity_2m,
        },
    'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time': {
        'ec_vars': ['SSRD'],
        'ec_func': lambda x: x,
        },
    'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time' : {
        'ec_vars': ['STRD'],
        'ec_func': lambda x: x,
        },
    'air_pressure_at_sea_level': {
        'ec_vars': ['MSL'],
        'ec_func': lambda x: x,
        },
    'precipitation_amount_acc': {
        'ec_vars': ['TP'],
        'ec_func': precipitation_amount_acc,
        },
    'integral_of_snowfall_amount_wrt_time' : {
        'ec_vars': ['2T', 'TP'],
        'ec_func': integral_of_snowfall_amount_wrt_time,
        },
    }
if 0:
    # test on smaller subset of variables
    dst_var = 'air_temperature_2m'
    #dst_var = 'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time'
    #dst_var = 'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time'
    DST_VARS = {dst_var: DST_VARS[dst_var]}

def rotate_winds(xg, yg, u, v):
    u, v = data[uname], data[vname]
    nt = u.shape[0]
    for i in range(nt):
        u[i], v[i] = nsl.rotate_velocities(
                DST_PI, xg, yg, u[i], v[i], fill_polar_hole=True)
        assert(np.all(np.isfinite(u[i])))
        assert(np.all(np.isfinite(v[i])))
    return u, v

def load_transformed_ECMWF(ec_ds, dst_vec, in_ec2_time_range, ec_pts, dst_ecp, dst_shape):
    # fetch, interpolate all variables (and rotate velocities) from ECMWF
    ec_data = dict()
    for dst_var_name in DST_VARS:
        ec_var_names = DST_VARS[dst_var_name]['ec_vars']
        for ec_var_name in ec_var_names:
            if ec_var_name in ec_data:
                continue
            print('Interpolate', ec_var_name)
            ec_var = get_ec2_var(ec_ds, ec_var_name, in_ec2_time_range)
            dst_ecd_grd = interpolate(ec_var, ec_pts, dst_ecp, dst_shape)
            ec_data[ec_var_name] = dst_ecd_grd

    # rotate winds
    print('Rotate winds')
    xg, yg = np.meshgrid(dst_vec['x'], dst_vec['y'])
    for i in range(len(in_ec2_time_range)):
        ec_data['10U'], ec_data['10V'] = rotate_winds(
                xg, yg, ec_data['10U'], ec_data['10V'])

    # convert to AROME variables
    for dst_var_name in DST_VARS:
        ec_var_names = DST_VARS[dst_var_name]['ec_vars']
        ec_args = [ec_data[ec_var_name] for ec_var_name in ec_var_names]
        EC_DATA[dst_var_name] = DST_VARS[dst_var_name]['ec_func'](*ec_args)

def load_transformed_AROME(ar_ds, i_ens, dst_vec, ar_proj, ar_shp, ar_pts, dst_arp, dst_shape):

    ar_data = dict()
    for dst_var_name in DST_VARS:
        # Interpolate data from AROME
        print('- Interpolate', dst_var_name)
        ar_var = np.zeros(ar_shp) #convert to 3d array
        ar_var[:] = ar_ds[dst_var_name][:, i_ens, :, :]
        ar_data[dst_var_name] = interpolate(ar_var, ar_pts, dst_arp, dst_shape)

    # rotate winds
    print('Rotate winds')
    xg, yg = np.meshgrid(dst_vec['x'], dst_vec['y'])
    for i in range(len(in_ec2_time_range)):
        ar_data['x_wind_10m'][i], ar_data['y_wind_10m'][i] = nsl.rotate_velocities(
                DST_PI, xg, yg,
                ar_data['x_wind_10m'][i], ar_data['y_wind_10m'][i],
                dst_proj=ar_proj,
                )
        assert(np.all(np.isfinite(ar_data['x_wind_10m'][i])))
        assert(np.all(np.isfinite(ar_data['y_wind_10m'][i])))
    return ar_data


def run(args):
    '''
    make the file

    Parameters:
    -----------
    args : argparse.Namespace
    '''
    os.makedirs(os.path.dirname(NEW_FILEMASK), exist_ok=True)
    outfile = args.date.strftime(NEW_FILEMASK)

    # open arome file and ecmwf file
    ar_ds, ar_proj, ar_pts = open_arome(args.date)
    ec_ds, ec_pts = open_ecmwf(args.date)
    in_ec2_time_range = test_ec2_time_range(ec_ds, args.date)

    if args.plot:
        # plot the grid if desired
        figname = outfile.replace('.nc', '.png')
        plot_destination_grid(figname, ar_proj, ar_pts)

    # set target coords
    (dst_vec, dst_ecp, dst_arp,
            dst_shape) = set_destination_coordinates(ar_proj, ar_ds)

    # distance to AROME border
    ar_shp = [len(v) for v in ar_pts]
    ar_dist = create_distance(ar_shp)
    dst_ardist_grd = interpolate(ar_dist, ar_pts, dst_arp, dst_shape)

    #if args.save_grid:
    #    # save the grid
    #    print('Exporting %s' %GRID_FILE)
    #    with Dataset(GRID_FILE, 'w') as dst_ds:
    #        add_grid_to_file(dst_ds, ar_ds, dst_ecp, dst_vec, dst_shape)

    sz = list(dst_ardist_grd.shape)
    num_ens_mems = 2 #ar_ds.dimensions['ensemble_member'].size
    sz.insert(1, num_ens_mems)
    DST_DATA = defaultdict(lambda : np.zeros(sz))


    # fetch, interpolate and blend all variables from ECMWF and AROME
    load_transformed_ECMWF(ec_ds, dst_vec, in_ec2_time_range, ec_pts, dst_ecp, dst_shape)
    for i_ens in range(num_ens_mems):
        print('AROME ensemble member: %i' %i_ens)
        ar_data = load_transformed_AROME(ar_ds, i_ens, dst_vec, ar_proj, ar_shp, ar_pts, dst_arp, dst_shape)

        # Compute destination product from ECMWF data
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
                plt.imshow(ar_data[dst_var_name][n,:,:],
                        origin='upper', **kw)
                plt.colorbar()
                plt.title(dst_var_name)
                plt.savefig(figname)
                plt.close()

        # blend and add to destination data
        for dst_var_name in ar_data:
            DST_DATA[dst_var_name][:, i_ens, :, :] = blend(
                    ar_data[dst_var_name], ar_data[dst_var_name], dst_ardist_grd)

    #export the files
    if EXPORT4D:
        export4d(outfile, ar_ds, dst_ecp, dst_vec, dst_shape)
    else:
        export3d(outfile, ar_ds, dst_ecp, dst_vec, dst_shape)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    run(args)
