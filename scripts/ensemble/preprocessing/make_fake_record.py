#!/usr/bin/env python
'''
repeat last time record to avoid crashing at last time step in forecast mode
(need 2 days + 3h to stop crash at t=2days)
'''
import os, sys
import shutil
from netCDF4 import Dataset
import numpy as np
import argparse
import datetime as dt
import pynextsim.lib as nsl

#ROOTDIR = '/cluster/projects/nn2993k/sim/data/AROME_barents_ensemble'
ROOTDIR = '/Data/sim/data/AROME_barents_ensemble'
DST_VARS = [
    'x_wind_10m',
    'y_wind_10m',
    'air_temperature_2m',
    'specific_humidity_2m',
    'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time',
    'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time' ,
    'air_pressure_at_sea_level',
    'precipitation_amount_acc',
    'integral_of_snowfall_amount_wrt_time',
    ] #fix these variables
VALID_DATE = lambda x : dt.datetime.strptime(x, '%Y%m%d')

def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
            description="Blend ECMWF and AROME ensemble outputs on high resolution grid")
    parser.add_argument('date', type=VALID_DATE,
            help='input date (YYYYMMDD)')
    return parser.parse_args(args)

def append_last_record(args):
    dir1 = os.path.join(ROOTDIR, 'blended')
    dir2 = os.path.join(ROOTDIR, 'blended_with_fake_record')
    nsl.make_dir(dir2)
    f1 = args.date.strftime(os.path.join(dir1, 'ec2_arome_blended_ensemble_%Y%m%d.nc'))
    f2 = args.date.strftime(os.path.join(dir2, 'ec2_arome_blended_ensemble_%Y%m%d.nc'))
    print('Making %s' %f2)
    shutil.copy2(f1, f2)

    with Dataset(f2, 'r+') as ds:
        time = list(ds.variables['time'][:])
        nt = len(time)
        time.append(time[-1] + 3*3600)
        ds.variables['time'][:] = time
        for v in DST_VARS:
            ds.variables[v][nt,:,:,:] = ds.variables[v][nt-1,:,:,:]

append_last_record(parse_args(sys.argv[1:]))
