#! /usr/bin/env python

import os
import glob
from argparse import ArgumentParser
import numpy as np
from datetime import datetime
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.path import Path
from sklearn.linear_model import LinearRegression

from geodataset.geodataset import GeoDatasetRead

from pynextsim.projection_info import ProjectionInfo

from scale_util import pwrspec2d


ROOT_DIR = ('/cluster/work/users/timill/nextsimf_forecasts/'
            'arome_3km_ec2_arome_ensemble.fram/new_version')

PATTERN = (f'{ROOT_DIR}/merged_files/'
                'nextsim_ecmwf_arome_ensemble_forecast_*')

AROME_GRID_FILE = f'{ROOT_DIR}/arome_grid.npz'

OUTFILE_PATTERN = 'spectra_%4.1fh' # one file for each lead time

PROJ = ProjectionInfo()

DX = 2500 # res in m


def parse_args():
    p = ArgumentParser('Plot spectra')
    p.add_argument('-o', '--outdir', type=str, default='out',
            help='where to save the output files')
    p.add_argument('-lts', '--lead-time-step', type=int, default=None,
            choices=list(range(16)),
            help='forecast lead time step (0=1.5h, 1=4.5h, ..., 15=46.5h)')
    p.add_argument('-s', '--snapshot', action='store_true', help=(
        'if not given we take mean of all forecasts for a given lead-time,'
        ' otherwise we just take the first example'))
    p.add_argument('-a', '--arome-domain', action='store_true', help=(
        'if not given we calculate the spectra over the whole domain,'
        ' otherwise we only calculate it for the AROME domain'))
    return vars(p.parse_args())


def sort_files():
    fsort = defaultdict(list)
    for f in sorted(glob.glob(PATTERN)):
        bdate = datetime.strptime(f[-11:-3], '%Y%m%d')
        fsort[bdate] += [f]
    return fsort


def get_velocity_spectra(u, v, arome_mask=None):
    n_ens = u.shape[0]
    good = np.isfinite(u[0])
    if arome_mask is not None:
        good *= arome_mask
    mask = ~good
    sum_pwr_vel = 0
    for i in range(n_ens):
        print(f'ensemble member {i+1}')
        u_ = np.array(u[i])
        v_ = np.array(v[i])
        u_[mask] = 0
        v_[mask] = 0
        wn, pwr_u = pwrspec2d(u_)
        wn, pwr_v = pwrspec2d(v_)
        wn /= DX
        sum_pwr_vel += .5*(pwr_u + pwr_v)
    return wn, sum_pwr_vel


def get_deformation(u, v):
    '''
    Get total deformation rate

    Parameters:
    -----------
    u : numpy.ndarray
        x-component of vector to plot
    v : numpy.ndarray
        y-component of vector to plot

    Returns:
    --------
    total : numpy.ndarray
        total deformation rate in %/day
    '''
    unit_fac = 100*24*3600 # convert from 1/s to %/day
    u_x, v_x = [unit_fac*np.gradient(a, axis=1) / DX for a in (u,v)]
    u_y, v_y = [unit_fac*np.gradient(a, axis=0) / DX for a in (u,v)]
    shear = np.hypot(u_x - v_y, u_y + v_x)
    div = u_x + v_y
    return np.hypot(shear, div)


def get_deformation_spectra(defor, mask2=None):
    n_ens = defor.shape[0]
    sum_pwr_defor = 0
    good = np.isfinite(defor[0])
    if mask2 is not None:
        good *= mask2
    mask = ~good
    for i in range(n_ens):
        print(f'ensemble member {i+1}')
        defor_ = np.array(defor[i])
        defor_[mask] = 0
        wn, pwr_defor = pwrspec2d(defor_)
        sum_pwr_defor += pwr_defor
    wn /= DX
    return wn, sum_pwr_defor


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


def get_leadtimes(fsort):
    """
    Returns:
    --------
    times : numpy.ndarray(float)
        lead times in seconds
    """
    times = []
    for bdate, flist in fsort.items():
        for f in flist:
            with GeoDatasetRead(f) as ds:
                times += [int((dto - bdate).total_seconds())
                        for dto in ds.datetimes]
        return np.array(sorted(times))


def select_uv(time, bdate, flist):
    for f in flist:
        with GeoDatasetRead(f) as ds:
            times = [int((dto - bdate).total_seconds())
                    for dto in ds.datetimes]
            if time not in times:
                print(f'Skipping {f} ({time = }')
                continue
            else:
                i = times.index(time)
                print(bdate, time / 3600, f, i)
            return (
                    ds.variables['siu'][i].filled(np.nan),
                    ds.variables['siv'][i].filled(np.nan),
                    )


def get_summed_spectra_snapshot(u, v, **kwargs):
    # spectrum of velocity
    print('get spectra of u,v')
    (wn, sum_pwr_vel_,
            ) = get_velocity_spectra(u, v, **kwargs)

    # spectrum of velocity spread (subtract ensemble mean)
    print('get spectra of u,v spread')
    umean = np.mean(u, axis=0)
    vmean = np.mean(v, axis=0)
    (wn, sum_pwr_dvel_,
            ) = get_velocity_spectra(
                    np.array([a - umean for a in u]),
                    np.array([a - vmean for a in v]),
                    **kwargs)

    # spectrum of deformation
    print('get spectra of deformation')
    defor = np.array([get_deformation(u_, v_)
        for u_,v_ in zip(u,v)])
    (wn, sum_pwr_defor_,
        ) = get_deformation_spectra(defor, **kwargs)

    # spectrum of deformation spread
    print('get spectra of deformation spread')
    defor_mean = np.mean(defor, axis=0)
    (wn, sum_pwr_defor_sprd_,
        ) = get_deformation_spectra(
                np.array([a - defor_mean for a in defor]),
                **kwargs)

    return (wn, sum_pwr_vel_, sum_pwr_dvel_,
            sum_pwr_defor_, sum_pwr_defor_sprd_)


def get_spectra(fsort, time, arome_mask=None, snapshot=False, **kwargs):
    # store results for full domain
    pwr_vel = 0
    pwr_vel_sprd = 0
    pwr_defor = 0
    pwr_defor_sprd = 0

    for n_times, (bdate, flist) in enumerate(fsort.items()):
        u,v = select_uv(time, bdate, flist)
        n_ens = u.shape[0]
        (wn, sum_pwr_vel_, sum_pwr_dvel_,
                sum_pwr_defor_, sum_pwr_defor_sprd_,
                ) = get_summed_spectra_snapshot(u, v, **kwargs)
        pwr_vel += sum_pwr_vel_
        pwr_vel_sprd += sum_pwr_dvel_
        pwr_defor += sum_pwr_defor_
        pwr_defor_sprd += sum_pwr_defor_sprd_
        if snapshot:
            break
    n_times += 1 # count started at 0 instead of 1

    # convert from sums to means
    n = n_times * n_ens
    pwr_vel /= n
    pwr_defor /= n

    # spreads have the ensemble means removed already
    n = n_times * (n_ens - 1)
    pwr_vel_sprd /= n
    pwr_defor_sprd /= n

    return dict(
            wn=wn,
            pwr_vel=pwr_vel,
            pwr_vel_sprd=pwr_vel_sprd,
            pwr_defor=pwr_defor,
            pwr_defor_sprd=pwr_defor_sprd,
            )


def add_asymptote(ax, wn, pwr):
    k0, k1 = wn[0], wn[-1]
    kmin = k0 + .5 * (k1 - k0)
    b = wn >= kmin
    x = np.log(wn[b]).reshape(-1,1)
    y = np.log(pwr[b])
    reg = LinearRegression().fit(x, y)
    pwr_ = np.exp(reg.predict(x))
    n = '%4.2f' %reg.coef_[0]
    ax.loglog(wn[b], pwr_, '--', label=f'n={n}')


def plot_spectra(figname, time, wn,
        pwr_vel, pwr_vel_sprd, pwr_defor, pwr_defor_sprd):
    """
    Parameters:
    -----------
    time : int
    pwr_vel : numpy.ndarray
    pwr_vel_sprd : numpy.ndarray
    pwr_defor : numpy.ndarray
    pwr_defor_sprd : numpy.ndarray
    """
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)
    ax.loglog(wn, pwr_vel, label="vel")
    ax.loglog(wn, pwr_vel_sprd, label="vel (spread)")
    ax.loglog(wn, pwr_defor, label="defor")
    ax.loglog(wn, pwr_defor_sprd, label="defor (spread)")
    add_asymptote(ax, wn, pwr_vel)
    add_asymptote(ax, wn, pwr_defor)
    ax.legend()
    ax.set_title(f"Spectra for lead time {time/3600}h")
    #ax.set_xticks(np.linspace(0,48,9))
    ax.set_ylim([1e-9, 50])
    ax.set_xlabel('Wave number, m$^{-1}$')
    ax.set_ylabel('Power spectral density, ?')
    fig.tight_layout()
    fig.savefig(figname, dpi=150)
    print(f'Saved {figname}')


def process_one_time(fsort, time, outdir, **kwargs):
    basename = (OUTFILE_PATTERN %(time/3600)).replace(' ', '0')
    if kwargs['snapshot']:
        basename += '_snapshot'
    npz = os.path.join(outdir, f'{basename}.npz')

    if os.path.exists(npz):
        print(f'Loading {npz}')
        with np.load(npz) as f:
            kw = dict(f)
    else:
        print(f'Making {npz}')
        kw = get_spectra(fsort, time, **kwargs)
        os.makedirs(outdir, exist_ok=True)
        np.savez(npz, **kw)
        print(f'Saved {npz}')

    figname = npz.replace('.npz', '.png')
    plot_spectra(figname, time, **kw)


def run(lead_time_step=None, arome_domain=False, **kwargs):
    fsort = sort_files()
    times = get_leadtimes(fsort)
    if lead_time_step is not None:
        times = [times[lead_time_step]]

    arome_mask = get_arome_mask() if arome_domain else None
    for time in times:
        process_one_time(fsort, time, arome_mask=arome_mask, **kwargs)


if __name__ == "__main__":
    run(**parse_args())
