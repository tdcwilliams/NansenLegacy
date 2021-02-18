#!/usr/bin/env python
import os
import sys
import argparse
import datetime as dt
import numpy as np

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

from pynextsim.projection_info import ProjectionInfo
from pynextsim.gmshlib import GmshMesh
import pynextsim.lib as nsl

# containers for interpolated data
EC_DATA = dict()
DST_DATA = dict()

# nextsim ref date
NEXTSIM_REF_DATE = dt.datetime(1900, 1, 1)

# Celsius to Kelvin conversion
_KELVIN = 273.15 # [C]

# filenames
EC_FILEMASK = '/Data/sim/data/ECMWF_forecast_arctic/ec2_start%Y%m%d.nc'
NEW_FILEMASK = '/Data/sim/data/ECMWF_forecast_arctic_stereographic/generic_atm_%Y%m%d.nc'

# Destination grid
MSH_FILE = '/Data/sim/data/mesh/unref/large_arctic_10km.msh' # grid should cover this mesh file
DST_PROJ = ProjectionInfo() # neXtSIM default projection
DST_RES = 10e3 #10km grid
DST_GRID_BORDER = 100e3 #100km buffer so model doesn't crash if any mesh points are outside the grid


def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
            description="Reproject ECMWF to polar stereographic grid")
    parser.add_argument('date', type=nsl.valid_date,
            help='input date (YYYYMMDD)')
    #parser.add_argument('-p', '--plot', action='store_true',
    #        help='Generate plot of AROME and the new domains')
    return parser.parse_args(args)


def deaccumulate_ecmwf(var):
    """ Calculate de-accumulated ECMWF variable
    Convert accumulated variables to rates

    Parameters
    ----------
    var : numpy.ndarray
        ECMWF var accumulated over 6h

    Returns
    ------
    var : numpy.ndarray
        ECMWF var accumulated over one second
    """
    return var/(6*3600.)


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
    sphuma = alpha * fa * esta / (ec_MSL - beta * fa * esta) ;

    return sphuma


def derivative_of_precipitation_amount_wrt_time(ec_TP):
    """ Calculate total precipitation rate

    Parameters
    ----------
    ec_TP : numpy.ndarray
        precipitation_amount in 6h [Mg/m^2]
    Returns
    ------
    precipit : numpy.ndarray
        Total precipitation rate [kg/m^2/s]
    """
    return deaccumulate_ecmwf(1e3*ec_TP) #convert from Mg/m^2 to kg/m^2 and get rate


def derivative_of_snowfall_amount_wrt_time(ec_2T, ec_TP):
    """ Calculate rate of snowfall

    Parameters
    ----------
    ec_2T : numpy.ndarray
        air_temperature [K]
    ec_TP : numpy.ndarray
        precipitation_amount in 1 hour [m]

    Returns
    ------
    snowfall : numpy.ndarray
        Snowfall rate at surface [kg/m^2/s]
    """
    snowfall = derivative_of_precipitation_amount_wrt_time(ec_TP) #kg/m^2/s
    snowfall[ ec_2T >= _KELVIN ] == 0. #precip is not snow if air temp above freezing
    return snowfall


def get_ec_var(ec_ds, var_name, ec_pts, dst_ecp, dst_shape):
    """
    load ECMWF variable and interpolate it to the destination points

    Parameters:
    -----------
    ec_ds : netCDF4.Dataset
        ECMWF dataset
    var_name : str
        name of variable in ECMWF file
    ec_pts : tuple
        three arrays with source coordinates
    dst_ecp : numpy.ndarray
        Nx3 array with destination coordinates for ECMWF (time, y, x)
    dst_shape : tuple
        shape of destination grid
    """
    data = ec_ds.variables[ec_var_name][:, ::-1, :] #flip lat
    data = np.concatenate([data[:,:,-1:], data], axis=2) #make cyclic in lon
    return interpolate(data, ec_pts, dst_ecp, dst_shape)


def get_ec_pts(ec_ds):
    """ Read ECMWF geolocation

    Parameters
    ----------
    ec_ds : netCDF4.Dataset
        ECMWF dataset

    Returns
    -------
    ec_pts : tuple with three 1D-ndarrays
        ECMWF time, latitude, longitude coordinates

    """
    ec_lon_vec = np.array([-180.]+ list(ec_ds.variables['lon'][:]))# make lon cyclic
    ec_lat_vec = ec_ds.variables['lat'][::-1] #make lat increase
    ec_time_vec = np.arange(0, 24, 6)
    return ec_time_vec, ec_lat_vec, ec_lon_vec

def set_destination_coordinates():
    """ Generate coordinates on the destination grid

    Parameters
    ----------
    ar_proj : Pyproj
        AROME projection

    Returns
    -------
    dst_grid : pynextsim.gridding.Grid
        grid corresponding to a given mesh file
    dst_vec : dict
        three vectors with destination coordinates, time, y, x
    dst_ecp : Nx3 ndarray
        destination coordinates for ECMWF (time, lat, lon)
    dst_shape : tuple
        shape of destination grid

    """
    # coordinates on destination grid
    # X,Y (NEXTSIM)
    dst_grid = GmshMesh(MSH_FILE, projection=DST_PROJ).boundary.get_grid(
            resolution=DST_RES, border=DST_GRID_BORDER)
    dst_vec = {
        'x': dst_grid.xy[0][0],
        'y': dst_grid.xy[1][:,0],
        'time': np.arange(0, 24, 6),
        }
    dst_t_grd, dst_y_grd, dst_x_grd  = np.meshgrid(dst_vec['time'], dst_vec['y'], dst_vec['x'], indexing='ij')
    dst_shape = dst_t_grd.shape

    # lon,lat (ECMWF)
    dst_lon_grd, dst_lat_grd = DST_PROJ.pyproj(dst_x_grd, dst_y_grd, inverse=True)
    dst_ecp = np.array([dst_t_grd.flatten(), dst_lat_grd.flatten(), dst_lon_grd.flatten()]).T

    return dst_grid, dst_vec, dst_ecp, dst_shape


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


def plot_destination_grid(date, ar_proj, ar_pts):
    """ Plot map with AROME and a destination domains """
    ar_x_grd, ar_y_grd = np.meshgrid(ar_pts[2], ar_pts[1])
    ar_lon_grd, ar_lat_grd = ar_proj(ar_x_grd, ar_y_grd, inverse=True)
    ar_lon_brd = np.hstack([ar_lon_grd[0,:], ar_lon_grd[:,-1], ar_lon_grd[-1,::-1], ar_lon_grd[::-1,0]])
    ar_lat_brd = np.hstack([ar_lat_grd[0,:], ar_lat_grd[:,-1], ar_lat_grd[-1,::-1], ar_lat_grd[::-1,0]])

    dst_extent = np.array([DST_X_MIN, DST_X_MAX, DST_Y_MIN, DST_Y_MAX])
    crs = DST_PROJ.crs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=crs)
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.set_extent(dst_extent*1.5, crs=crs)
    plt.plot(ar_lon_brd, ar_lat_brd, '.-', transform=ccrs.Geodetic())
    plt.plot([DST_X_MIN, DST_X_MIN, DST_X_MAX, DST_X_MAX, DST_X_MIN],
             [DST_Y_MIN, DST_Y_MAX, DST_Y_MAX, DST_Y_MIN, DST_Y_MIN], 'o-r')
    plt.savefig(NEW_FILEMASK % date + '.png')
    plt.close()


def export(date, dst_vec, dst_ecp, dst_shape):
    """ Export blended product

    Parameters
    ----------
    date : datetime.datetime
        date to add into output filename
    dst_ecp : Nx3 ndarray
        destination coordinates for ECMWF (time, lat, lon)
    dst_vec : dict
        three vectors with destination coordinates (time, y, x)
    dst_shape : tuple
        shape of destination grid

    """
    # Create dataset for output
    dst_file = date.strftime(NEW_FILEMASK)
    dst_dir = os.path.dirname(dst_file)
    os.makedirs(dst_dir, exist_ok=True)

    print(f'Saving {dst_file}')
    with Dataset(dst_file, 'w') as dst_ds:
        # add dimensions
        dim_attrs = dict(
                time=dict(
                    standard_name = "time" ,
                    long_name = "time" ,
                    units = "days since 1900-01-01 00:00:00" ,
                    calendar = "standard" ,
                    ),#don't worry about time_bnds
                y=dict(
                    standard_name = "projection_y_coordinate" ,
                    units = "m" ,
                    axis = "Y" ,
                    ),
                x=dict(
                    standard_name = "projection_x_coordinate" ,
                    units = "m" ,
                    axis = "X" ,
                    ),
                )
        for dim_name in ['time', 'y', 'x']:
            data = dst_vec[dim_name]
            len_ = len(data)
            if dim_name == 'time':
                len_ = None
                t0 = (date - NEXTSIM_REF_DATE).days
                data = np.array([t0 + h/24 for h in data]) #now time is days since NEXTSIM_REF_DATE
            dst_dim = dst_ds.createDimension(dim_name, len_)
            dst_var = dst_ds.createVariable(dim_name, 'f8', (dim_name,), zlib=True)
            dst_var.setncatts(dim_attrs[dim_name])
            dst_var[:] = data

        # add interpolated/rotated variables
        for dst_var_name, info in DST_VARS.items():
            dst_var = dst_ds.createVariable(dst_var_name, 'f4', ('time', 'y', 'x',), zlib=True)
            dst_var.setncatts(info['ncatts'])
            dst_var[:] = DST_DATA[dst_var_name]
            dst_var.setncattr('grid_mapping', 'projection_stereo')

        # add projection variable
        dst_var = dst_ds.createVariable('projection_stereo', 'i1')
        dst_var.setncatts(DST_PROJ.ncattrs("polar_stereographic"))

        # add lon/lat
        dst_lon_grd = dst_ecp[:, 2].reshape(dst_shape)[0]
        dst_lat_grd = dst_ecp[:, 1].reshape(dst_shape)[0]
        for var_name, var_data, units in zip(
                ['longitude', 'latitude'],
                [dst_lon_grd, dst_lat_grd],
                ['degree_east', 'degree_north'],
                ):
            dst_var = dst_ds.createVariable(var_name, 'f8', ('y', 'x',), zlib=True)
            dst_var.setncattr('standard_name', var_name)
            dst_var.setncattr('long_name', var_name)
            dst_var.setncattr('units', units)
            dst_var[:] = var_data


# Destination variables
DST_VARS = {
    'x_wind_10m' : {
        'ec_vars': ['10U'],
        'ec_func': lambda x : x,
        'ncatts': dict(
            long_name = "Zonal 10 metre wind (U10M)" ,
            standard_name = "u_wind_10m" ,
            units = "m/s" ,
            ),
    },
    'y_wind_10m' : {
        'ec_vars': ['10V'],
        'ec_func': lambda x : x,
        'ncatts': dict(
            long_name = "Meridional 10 metre wind (V10M)" ,
            standard_name = "v_wind_10m" ,
            units = "m/s" ,
            ),
    },
    'air_temperature_2m': {
        'ec_vars': ['2T'],
        'ec_func': lambda x : x,
        'ncatts': dict(
            long_name = "Screen level temperature (T2M)" ,
            standard_name = "air_temperature" ,
            units = "K" ,
            )
    },
    'dew_point_temperature_2m': {
        'ec_vars': ['2D'],
        'ec_func': lambda x : x,
        'ncatts': dict(
            long_name = "Screen level dew point temperature (D2M)" ,
            standard_name = "dew_point_temperature" ,
            units = "K" ,
            )
    },
    'derivative_of_surface_downwelling_shortwave_flux_in_air_wrt_time': {
        'ec_vars': ['SSRD'],
        'ec_func': deaccumulate_ecmwf,
        'ncatts': dict(
            long_name = "Surface SW downwelling radiation rate" ,
	    standard_name = "derivative_of_surface_downwelling_shortwave_flux_in_air_wrt_time" ,
	    units = "W/m^2",
            ),
    },
    'derivative_of_surface_downwelling_longwave_flux_in_air_wrt_time' : {
        'ec_vars': ['STRD'],
        'ec_func': deaccumulate_ecmwf,
        'ncatts': dict(
            long_name = "Surface LW downwelling radiation rate" ,
	    standard_name = "derivative_of_surface_downwelling_longwave_flux_in_air_wrt_time" ,
	    units = "W/m^2",
            ),
    },
    'air_pressure_at_sea_level': {
        'ec_vars': ['MSL'],
        'ec_func': lambda x : x,
        'ncatts': dict(
            long_name = "Mean Sea Level Pressure (MSLP)" ,
	    standard_name = "air_pressure_at_sea_level" ,
	    units = "Pa" ,
            ),
    },
    'derivative_of_precipitation_amount_wrt_time': {
        'ec_vars': ['TP'],
        'ec_func': derivative_of_precipitation_amount_wrt_time,
        'ncatts': dict(
	    long_name = "Total precipitation rate",
	    standard_name = "precipitation_rate",
	    units = "kg/m^2/s",
            ),
    },
    'derivative_of_snowfall_amount_wrt_time' : {
        'ec_vars': ['2T', 'TP'],
        'ec_func': derivative_of_snowfall_amount_wrt_time,
        'ncatts': dict(
	    long_name = "Snowfall rate",
	    standard_name = "snowfall_rate",
	    units = "kg/m^2/s",
            ),
    },
}


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])
    ec2_file = args.date.strftime(EC_FILEMASK)
    print(f'Opening {ec2_file}')
    with Dataset(ec2_file, 'r') as ec_ds:
        ec_pts = get_ec_pts(ec_ds)
        dst_grid, dst_vec, dst_ecp, dst_shape = set_destination_coordinates()

        # fetch, interpolate and blend all variables from ECMWF
        for dst_var_name in DST_VARS:

            # Interpolate data from ECMWF
            ec_var_names = DST_VARS[dst_var_name]['ec_vars']
            for ec_var_name in ec_var_names:
                if ec_var_name in EC_DATA:
                    continue
                print('Interpolate', ec_var_name)
                EC_DATA[ec_var_name] = get_ec_var(ec_ds, ec_var_name, ec_pts, dst_ecp, dst_shape)

            # Compute destination product from ECMWF data
            ec_args = [EC_DATA[ec_var_name] for ec_var_name in ec_var_names]
            DST_DATA[dst_var_name] = DST_VARS[dst_var_name]['ec_func'](*ec_args)

    # rotate winds
    print('Rotate winds')
    for i in range(4):
        DST_DATA['x_wind_10m'][i], DST_DATA['y_wind_10m'][i] = nsl.rotate_velocities(
                DST_PROJ, *dst_grid.xy,
                DST_DATA['x_wind_10m'][i], DST_DATA['y_wind_10m'][i],
                fill_polar_hole=True,
                )
        assert(np.all(np.isfinite(DST_DATA['x_wind_10m'][i])))
        assert(np.all(np.isfinite(DST_DATA['y_wind_10m'][i])))

    export(args.date, dst_vec, dst_ecp, dst_shape)
