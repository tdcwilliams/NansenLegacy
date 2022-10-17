#! /usr/bin/env python

import os
import glob
import numpy as np
from datetime import datetime
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.path import Path

from geodataset.geodataset import GeoDatasetRead

from pynextsim.projection_info import ProjectionInfo


ROOT_DIR = ('/cluster/work/users/timill/nextsimf_forecasts/'
            'arome_3km_ec2_arome_ensemble.fram/new_version')

PATTERNS = dict(
        spreads=(f'{ROOT_DIR}/ensemble_spreads/'
                'nextsim_ecmwf_arome_ensemble_forecast_spread_*'),
        means=(f'{ROOT_DIR}/ensemble_means/'
                        'nextsim_ecmwf_arome_ensemble_forecast_mean_*'),
        )

OUTDIR = 'out'

NPZ_FILES = {k : f'{OUTDIR}/{k}.npz' for k in PATTERNS}

AROME_GRID_FILE = f'{ROOT_DIR}/arome_grid.npz'

PROJ = ProjectionInfo()


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
            for a in PROJ.pyproj(lon, lat)]


def get_grid_xy():
    """
    Returns:
    --------
    x : numpy.ndarray(float)
        2D array with x coords for nextsim output grid
    y : numpy.ndarray(float)
        2D array with y coords for nextsim output grid
    """
    f = glob.glob(PATTERNS['spreads'])[0]
    print(f"Getting grid from {f}")
    with GeoDatasetRead(f) as ds:
        return PROJ.pyproj(*ds.get_lonlat_arrays())


def get_arome_mask():
    """
    Returns:
    --------
    inside_arome : numpy.ndarray(bool)
        mask for nextsim output grid - True for points
        that are inside the arome domain
    """
    x, y = get_grid_xy()
    xcnr, ycnr = get_arome_corners()
    bpath = Path(np.array([xcnr, ycnr]).T)
    xy = np.array([x.flatten(), y.flatten()]).T
    mask = np.array(bpath.contains_points(xy), dtype=bool)
    return mask.reshape(x.shape)


def get_spreads(fsort):
    # store results for full domain
    sum_spd_squared = defaultdict(float)
    n_speeds = defaultdict(float)
    # store results for arome domain
    arome_mask = get_arome_mask()
    np.savez('out/arome_mask.npz', arome_mask=arome_mask)
    sum_spd_squared_arome = defaultdict(float)
    n_speeds_arome = defaultdict(float)
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
                    # full domain
                    good = np.isfinite(d_spd_squared)
                    sum_spd_squared[time] += np.sum(d_spd_squared[good])
                    n_speeds[time] += good.sum()
                    # arome domain
                    good *= arome_mask
                    sum_spd_squared_arome[time] += np.sum(d_spd_squared[good])
                    n_speeds_arome[time] += good.sum()
    variances = {t : sss/(n_speeds[t] - 1)
            for t,sss in sum_spd_squared.items()}
    variances_arome = {t : sss/(n_speeds_arome[t] - 1)
            for t,sss in sum_spd_squared_arome.items()}
    time = np.array(list(variances)) / 3600. # seconds to hours
    variances = np.array(list(variances.values()))
    variances_arome = np.array(list(variances_arome.values()))
    return time, variances, variances_arome


def save_variances():
    os.makedirs(OUTDIR, exist_ok=True)
    variances = {}
    for k,pat in PATTERNS.items():
        npz = NPZ_FILES[k]
        if os.path.exists(npz):
            continue
        fsort = sort_files(pat)
        time, variances, variances_arome = get_spreads(fsort)
        np.savez(npz, time=time, variances=variances,
                variances_arome=variances_arome)


def load_variances():
    variances = {}
    for k,f in NPZ_FILES.items():
        with np.load(f) as fid:
            variances[k] = (fid['time'], fid['variances'],
                    fid['variances_arome'])
    return variances


def plot_variances(variances):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)
    for k, (time, vars_k, vars_k_arome) in variances.items():
        ax.plot(time, 100 * np.sqrt(vars_k), label=k)
        ax.plot(time, 100 * np.sqrt(vars_k_arome), label=f'{k} (AROME domain)')
    ax.set_xticks(np.linspace(0,48,9))
    ax.set_xlabel('time, h')
    ax.set_ylabel('Std dev of speed anomalies, cm s$^{-1}$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f := f"{OUTDIR}/spreads.png", dpi=150)
    print(f'Saved {f}')


def run():
    save_variances()
    variances = load_variances()
    plot_variances(variances)


if __name__ == "__main__":
    run()
