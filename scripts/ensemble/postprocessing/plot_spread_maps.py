#! /usr/bin/env python

import os
import glob
import numpy as np
from datetime import datetime
from collections import defaultdict
from matplotlib import pyplot as plt

from geodataset.geodataset import GeoDatasetRead

from pynextsim.gridding import Grid
from pynextsim.projection_info import ProjectionInfo
from pynextsim.gmshlib import GmshMesh


ROOT_DIR = ('/cluster/work/users/timill/nextsimf_forecasts/'
            'arome_3km_ec2_arome_ensemble.fram/new_version')

PATTERN = (f'{ROOT_DIR}/ensemble_spreads/'
                'nextsim_ecmwf_arome_ensemble_forecast_spread_*')

AROME_GRID_FILE = f'{ROOT_DIR}/arome_grid.npz'

OUTDIR = 'out'

NPZ_FILE = f'{OUTDIR}/spread_maps.npz'

try:
    MESH_FILE = os.path.join(
            os.getenv('NEXTSIM_MESH_DIR'),
            'arome_3km.msh')
except:
    MESH_FILE = None


def parse_file_dates(f):
    fdate = datetime.strptime(f[-21:-13], '%Y%m%d')
    bdate = datetime.strptime(f[-11:-3], '%Y%m%d')
    return fdate, bdate


def sort_files(pattern):
    fsort = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        _, bdate = parse_file_dates(f)
        fsort[bdate] += [f]
    return fsort


def get_grid():
    f = glob.glob(PATTERN)[0]
    print(f"Getting grid from {f}")
    with GeoDatasetRead(f) as ds:
        return Grid(*ds.get_lonlat_arrays(), latlon=True)


def get_arome_corners():
    """
    Returns:
    --------
    xa_cnr : numpy.ndarray
        x-coords of grid corners
    ya_cnr : numpy.ndarray
        y-coords of grid corners
    """
    print(f"Getting AROME grid from {AROME_GRID_FILE}")
    with np.load(AROME_GRID_FILE) as f:
        lon = f['lon']
        lat = f['lat']
    return [[a[0,0], a[-1,0], a[-1,-1], a[0,-1], a[0,0]]
            for a in ProjectionInfo().pyproj(lon, lat)]


def get_spreads(fsort):
    """
    Returns:
    --------
    time : numpy.ndarray
        1d array of length nt
    variances : numpy.ndarray
        3d array of length
    """
    sum_spd_squared = {}
    n_speeds = defaultdict(float)
    #for cnt, (bdate, flist) in enumerate(fsort.items()):
    #    if cnt == 2:
    #        break
    for bdate, flist in fsort.items():
        #for f in flist[:1]:
        for f in flist:
            print(f)
            with GeoDatasetRead(f) as ds:
                for i,dto in enumerate(ds.datetimes):
                    #if i == 2:
                    #    break
                    time = (dto - bdate).total_seconds()
                    d_siu = ds.variables['siu'][i].filled(np.nan)
                    d_siv = ds.variables['siv'][i].filled(np.nan)
                    d_spd_squared = d_siu**2 + d_siv**2
                    n_ens, ny, nx = d_siu.shape
                    n_speeds[time] += n_ens
                    if time not in sum_spd_squared:
                        sum_spd_squared[time] = np.zeros((ny,nx))
                    sum_spd_squared[time] += np.sum(d_spd_squared,  axis=0)
    variances = {t : sss/(n_speeds[t] - 1)
            for t,sss in sum_spd_squared.items()}
    time = np.array(list(variances)) / 3600.
    variances = np.array(list(variances.values()))
    return time, variances


def save_variances():
    if os.path.exists(NPZ_FILE):
        return
    os.makedirs(OUTDIR, exist_ok=True)
    fsort = sort_files(PATTERN)
    time, variances = get_spreads(fsort)
    np.savez(NPZ_FILE, time=time, variances=variances)


def load_variances():
    with np.load(NPZ_FILE) as fid:
        return fid['time'], fid['variances']


def plot_variances(time, variances):
    grid = get_grid()
    xa_cnr, ya_cnr = get_arome_corners()
    if MESH_FILE is not None:
        msh = GmshMesh(MESH_FILE)
    for (time, var_t) in zip(time, variances):
        tstr = '%4.1fh' %time
        fig, ax = grid.plot(np.sqrt(var_t) * 100,
                title='Spread after %s' %tstr.replace(' ', ''),
                clim=[0,5],
                format='%3.1f',
                clabel='Std dev of speed anomalies, cm s$^{-1}$',
                )
        if MESH_FILE is not None:
            msh.boundary.plot(ax)
        else:
            ax.coastlines()
        ax.plot(xa_cnr, ya_cnr, 'k', linewidth=.5)
        fig.tight_layout()
        f = f"%s/spread_map_%s.png" %(OUTDIR, tstr.replace(' ', '0'))
        fig.savefig(f, dpi=200)
        print(f'Saved {f}')


def run():
    save_variances()
    time, variances = load_variances()
    plot_variances(time, variances)


if __name__ == "__main__":
    run()
