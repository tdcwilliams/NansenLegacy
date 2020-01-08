#! /usr/bin/env python
import os, sys
import argparse
import datetime as dt
from matplotlib import pyplot as plt

from pynextsim.nextsim_bin import NextsimBin
import pynextsim.lib as nsl

VALID_DATE = lambda s: dt.datetime.strptime(s, '%Y%m%d')

class PlotInnovations(object):
    def __init__(self, root_forecast_dir, date1, date2):
        self.root_forecast_dir = root_forecast_dir
        self.date1 = date1
        self.date2 = date2

    @classmethod
    def init_from_cli(cls, cli):
        parser = argparse.ArgumentParser("Plot forecast innovations after assimilation")
        parser.add_argument("root_forecast_dir", type=str,
                help="root forecast dir")
        parser.add_argument("date1", type=VALID_DATE,
                help="first date to plot")
        parser.add_argument("date2", type=VALID_DATE,
                help="last date to plot")
        args = parser.parse_args(cli)
        
        self = cls.__new__(cls) # empty object
        super(cls, self).__init__()
        self.__init__(args.root_forecast_dir, args.date1, args.date2)
        return self

    def plot_one_date(self, plot_date, **kwargs):
        datestr = plot_date.strftime('%Y%m%dT000000Z')
        basename = 'field_%s' %datestr
        prev_date = plot_date - dt.timedelta(1)
        rf1 = os.path.join(
                self.root_forecast_dir,
                prev_date.strftime('%Y%m%d'),
                'restart',
                basename)# restart after 1 day of previous forecast
        rf2 = os.path.join(
                self.root_forecast_dir,
                plot_date.strftime('%Y%m%d'),
                'inputs',
                basename)# restart after assimilation
        nb1 = NextsimBin(rf1, add_gmsh=True,
                logfile = os.path.join(
                    self.root_forecast_dir,
                    prev_date.strftime('%Y%m%d'),
                    'nextsim.log'),
                **kwargs)
        nb2 = NextsimBin(rf2)
        plot_vars = [
                'M_conc',
                'M_thick',
                'M_snow_thick',
                'M_ridge_ratio',
                'M_tice_0',
                'M_tice_1',
                'M_tice_2',
                'M_sst',
                'M_sss',
                'M_tsurf_thin',
                'M_h_thin',
                'M_hs_thin',
                'M_conc_thin',
                ]
        innovs = dict()
        vtypes = dict()
        for v in plot_vars:
            v1 = nb1.get_var(v)
            v2 = nb2.get_var(v)
            k = 'innov-'+v
            innovs[k] = v2 - v1
            vtypes[k] = 'f'
        nb1.add_vars(innovs, vtypes)
        gridded_vars = nb1.interp_vars(list(innovs))

        outdir = os.path.join(
                self.root_forecast_dir,
                plot_date.strftime('%Y%m%d'),
                'innovation_plots')
        nsl.make_dir(outdir)
        for v, arr in gridded_vars.items():
            fig = nb1.get_gridded_figure(
                    arr.T,
                    land_color=[1,1,1],
                    clim=[-.5,.5], cmap='balance')
            figname = '%s/%s-%s.png' %(outdir, v, datestr)
            print('Saving %s' %figname)
            fig.savefig(figname)
            plt.close(fig)

        return nb1.mesh_info.gmsh_mesh
        
    def run(self):
        plot_date = self.date1
        gmo = None #want to reuse the GMSH mesh object
        while plot_date <= self.date2:
            gmo = self.plot_one_date(plot_date, gmsh_obj=gmo)
            plot_date += dt.timedelta(1)

if __name__ == '__main__':
    obj = PlotInnovations.init_from_cli(sys.argv[1:])
    obj.run()
