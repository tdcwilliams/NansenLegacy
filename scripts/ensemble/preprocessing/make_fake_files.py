#!/usr/bin/env python
'''
script to make fake files to stp nextsim crashing due to 'no file for previous/following day'
'''
import os
import shutil
from netCDF4 import Dataset
import numpy as np

ROOTDIR = '/cluster/projects/nn2993k/sim/data/AROME_barents_ensemble'
DIR1 = os.path.join(ROOTDIR, 'blended')
DIR2 = os.path.join(ROOTDIR, 'blended_with_fake_record', 'fake_files')
os.makedirs(DIR2, exist_ok=True)

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
DAYS_IN_SEC = 24*60*60. # time units are seconds

def fix_first_day():
    f1 = os.path.join(DIR1, 'ec2_arome_blended_ensemble_20180309.nc')
    f2 = os.path.join(DIR2, 'ec2_arome_blended_ensemble_20180308.nc')
    print('Making %s' %f2)
    shutil.copy2(f1, f2)

    with Dataset(f2, 'r+') as ds:
        time = ds.variables['time'][:] - DAYS_IN_SEC
        ds.variables['time'][:] = time
        nt = time.size
        for v in DST_VARS:
            for i in range(1, nt):
                # set all values to same as 1st time rec (20180309 00:00:00)
                ds.variables[v][i,:,:,:] = ds.variables[v][0,:,:,:]

def fix_last_day():
    f1 = os.path.join(DIR1, 'ec2_arome_blended_ensemble_20180331.nc')
    f2 = os.path.join(DIR2, 'ec2_arome_blended_ensemble_20180401.nc')
    print('Making %s' %f2)
    shutil.copy2(f1, f2)

    with Dataset(f2, 'r+') as ds:
        time = ds.variables['time'][:] + DAYS_IN_SEC
        ds.variables['time'] = time
        nt = time.size
        for v in DST_VARS:
            for i in range(nt-1):
                # set all values to same as last time rec (20180331 21:00:00)
                ds.variables[v][i,:,:,:] = ds.variables[v][nt-1,:,:,:]

fix_first_day()
#fix_last_day()
