#!/usr/bin/env python
import os
import sys
import argparse

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

# containers for interpolated data
EC_DATA = {}
DST_DATA = {}

# Celsius to Kelvin conversion
_KELVIN = 273.15 # [C]

# filenames
AR_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/processed/aro_eps_%s.nc'
AR_MARGIN = 20 # pixels
TIME_RECS_PER_DAY = 8

EC_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/legacy_201803_3h.nc'
NEW_FILEMASK = '/Data/sim/data/AROME_barents_ensemble/blended/ec2_arome_blended_ensemble_%s.nc'

# Destination grid
# neXtSIM default projection
DST_PI = ProjectionInfo()

# extent overlaps with AROME grid
DST_X_MIN = -300000
DST_X_MAX =  2800000
DST_Y_MIN = -2000000
DST_Y_MAX =  1500000
DST_X_RES, DST_Y_RES = 2500, 2500
#DST_X_RES, DST_Y_RES = 25000, 25000 #low res for testing

# extent overlaps with FRAM strait
#DST_X_MIN = -450000
#DST_X_MAX =  1300000
#DST_Y_MIN = -2000000
#DST_Y_MAX =  300000
#DST_X_RES, DST_Y_RES = 5000, 5000


def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
            description="Blend ECMWF and AROME ensemble outputs on high resolution grid")
    parser.add_argument('date', type=str,
            help='input date (YYYYMMDD)')
    parser.add_argument('-p', '--plot', action='store_true',
            help='Generate plot of AROME and the new domains')
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
    date : str
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
    ar_filename = AR_FILEMASK % date
    print('\nOpening AROME file %s' %ar_filename)
    ar_ds = Dataset(ar_filename)
    ar_proj = pyproj.Proj(ar_ds.variables['projection_lambert'].proj4)
    ar_x_vec = ar_ds.variables['x'][:].data
    ar_y_vec = ar_ds.variables['y'][:].data
    ar_t_vec = np.linspace(0, 24, TIME_RECS_PER_DAY+1)[:-1]

    return ar_ds, ar_proj, (ar_t_vec, ar_y_vec, ar_x_vec)

def open_ecmwf(date):
    """ Read ECMWF geolocation

    Parameters
    ----------
    date : str
        date of ECMWF file in YYYYMMDD format

    Returns
    -------
    ec_ds : netCDF4.Dataset
        ECMWF dataset
    ec_pts : tuple with three 1D-ndarrays
        ECMWF time, latitude, longitude coordinates

    """
    ec_filename = EC_FILEMASK % date
    print('Opening ECMWF file %s\n' %ec_filename)
    ec_ds = Dataset(ec_filename)
    ec_lon_vec = ec_ds.variables['lon'][:]
    # a hack to interpolate EC values also onto -180 degrees bins
    ec_lon_vec[0] = -180 #
    ec_lat_vec = ec_ds.variables['lat'][:]
    ec_time_vec = np.linspace(0, 24, TIME_RECS_PER_DAY+1)
    return ec_ds, (ec_time_vec, ec_lat_vec, ec_lon_vec)

def test_ec2_time_range(ec_ds, date):
    '''
    Parameters:
    -----------
    ec_ds : netCDF4.Dataset
    date : str
        date in YYYYMMDD format

    Returns:
    --------
    in_range : numpy.ndarray (bool)
    '''
    dto = dt.datetime.strptime(date, '%Y%m%d')
    ec_time_raw = ec_ds.variables['time'][:].astype(float)
    ec_time = np.array([
        _EC_REFDATE + dt.timedelta(hours=h)
        for h in ec_time_raw])
    in_range = (ec_time >= dto)*(ec_time < dto + dt.timedelta(1))
    return in_range

def set_destination_coordinates(ar_proj, ar_ds):
    """ Generate coordinates on the destination grid

    Parameters
    ----------
    ar_proj : Pyproj
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
        'time': np.linspace(0, 24, TIME_RECS_PER_DAY+1)[:-1],
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

def create_distance(ar_data):
    """ Create 3D matrix (same shape as AROME data) with distance from the border in pixels

    Parameters
    ----------
    ar_data : numpy.ndarray
        any 3D array (from AROME dataset)

    Returns
    -------
    distance : numpy.ndarray
        3D array with euclidian distance from the border (pix)

    """
    mask = np.ones(ar_var[0].shape, bool)
    # put False into all border pixels (on a 2D matrix)
    mask[0,:] = False
    mask[-1,:] = False
    mask[:,0] = False
    mask[:,-1] = False
    # calculate distance from the border (from False pixels)
    dist2d = distance_transform_edt(mask, return_distances=True, return_indices=False)
    # convert 2D into 3D
    return np.array([dist2d]*24)

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

def export(outfile, ar_ds, dst_ecp, dst_vec, dst_shape):
    """ Export blended product

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
    # copy arome time for the day
    dst_ds.variables['time'][:] = ar_ds.variables['time'][:TIME_RECS_PER_DAY]

    # add blended variables
    for dst_var_name in DST_VARS:
        dst_var = dst_ds.createVariable(dst_var_name, 'f4',
                ('time', 'ensemble_member', 'y', 'x',))
        dst_var[:] = DST_DATA[dst_var_name]
        ar_var = ar_ds.variables[dst_var_name]
        for ncattr in ar_var.ncattrs():
            if ncattr in skip_var_attr:
                continue
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var.setncattr('grid_mapping', 'projection_stereo')

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
    dst_lon_grd = dst_ecp[:, 2].reshape(dst_shape)[0]
    dst_lat_grd = dst_ecp[:, 1].reshape(dst_shape)[0]
    for var_name in ['longitude', 'latitude']:
        dst_var = dst_ds.createVariable(var_name, 'f8', ('y', 'x',))
        ar_var = ar_ds.variables[var_name]
        for ncattr in ar_var.ncattrs():
            dst_var.setncattr(ncattr, ar_var.getncattr(ncattr))
        dst_var[:] = {'longitude': dst_lon_grd, 'latitude': dst_lat_grd}[var_name]

    dst_ds.close()

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
#dst_var = 'air_temperature_2m'
#dst_var = 'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time'
#dst_var = 'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time'
#DST_VARS = {dst_var: DST_VARS[dst_var]}

def run(args):
    '''
    make the file

    Parameters:
    -----------
    args : argparse.Namespace
    '''
    outdir = os.path.split(NEW_FILEMASK)[0]
    nsl.make_dir(outdir)
    outfile = os.path.join(outdir, NEW_FILEMASK % args.date)

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

    dst_ardist_grd = None
    ar_shp = [len(ar_pts[i]) for i in range(3)]

    # fetch, interpolate and blend all variables from ECMWF and AROME
    num_ens_mems = ar_ds.dimensions['ensemble_member'].size
    for dst_var_name in DST_VARS:
        # Interpolate data from AROME
        print('Interpolate', dst_var_name)
        dst_ard_grd_all_members = []
        for i_ens in range(num_ens_mems):
            ar_var = np.zeros(ar_shp) #convert to 3d array
            #ar_var[:] = ar_ds[dst_var_name][:TIME_RECS_PER_DAY, 0, i_ens, :, :]
            ar_var[:] = ar_ds[dst_var_name][:TIME_RECS_PER_DAY, i_ens, :, :]
            dst_ard_grd_all_members.append(
                    interpolate(ar_var, ar_pts, dst_arp, dst_shape))

        # distance to AROME border
        if dst_ardist_grd is None:
            ar_dist = create_distance(ar_var)
            dst_ardist_grd = interpolate(ar_dist, ar_pts, dst_arp, dst_shape)

        # Interpolate data from ECMWF
        ec_var_names = DST_VARS[dst_var_name]['ec_vars']
        for ec_var_name in ec_var_names:
            if ec_var_name in EC_DATA:
                continue
            print('Interpolate', ec_var_name)
            ec_var = ec_ds.variables[
                    ec_var_name][in_ec2_time_range,:,:] #no more preproc needed
            dst_ecd_grd = interpolate(ec_var, ec_pts, dst_ecp, dst_shape)
            EC_DATA[ec_var_name] = dst_ecd_grd

        # Compute destination product from ECMWF data
        ec_args = [EC_DATA[ec_var_name] for ec_var_name in ec_var_names]
        dst_ecd_grd = DST_VARS[dst_var_name]['ec_func'](*ec_args)
        for n in []: #range(8): 
            figname = 'figs0/ec_%s_%i.png' %(dst_var_name, n)
            print('Saving %s' %figname)
            plt.imshow(dst_ecd_grd[n,:,:], origin='upper')
            plt.colorbar()
            plt.title(dst_var_name)
            plt.savefig(figname)
            plt.close() 
            figname = 'figs0/ar_%s_%i.png' %(dst_var_name, n)
            print('Saving %s' %figname)
            plt.imshow(dst_ard_grd_all_members[0][n,:,:], origin='upper')
            plt.colorbar()
            plt.title(dst_var_name)
            plt.savefig(figname)
            plt.close() 

        # blend and add to destination data
        sz = list(dst_ardist_grd.shape)
        sz.insert(1, num_ens_mems)
        DST_DATA[dst_var_name] = np.zeros(sz)
        for i_ens, dst_ard_grd in enumerate(dst_ard_grd_all_members):
            DST_DATA[dst_var_name][:, i_ens, :, :] = blend(
                    dst_ard_grd, dst_ecd_grd, dst_ardist_grd)

    # save the output
    export(outfile, ar_ds, dst_ecp, dst_vec, dst_shape)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    run(args)
