#! /usr/bin/env python
import os
import numpy as np
import glob
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def get_parser():
    ap = ArgumentParser("Script to plot spread and RMSE")
    ap.add_argument("root_dir", type=str,
            help="Root directory with *.mem_??? folders.")
    ap.add_argument("forecast_name", type=str,
            help="Results for each ensemble member will be in "
            "[ROOT_DIR]/[FORECAST_NAME].mem_??? folders.")
    ap.add_argument('-v', "--variable-name", default='Concentration',
            type=str, choices=['Concentration', 'Thickness'],
            help="Variable name. Default is Concentration.")
    ap.add_argument('-d', "--spread-dir", default='eval-ref-conc',
            type=str, help="Results of comparison to analysis should be in "
            "[ROOT_DIR]/[FORECAST_NAME].mem_???/[SPREAD_DIR]/20??????."
            "\nDefault is eval-ref-conc.")
    return ap


class PlotSpreadRMSE:

    def __init__(self, root_dir, forecast_name,
            variable_name='Concentration',
            spread_dir='eval-ref-conc',
            ):
        self.root_dir = root_dir
        self.forecast_name = forecast_name
        self.variable_name = variable_name
        self.spread_dir = spread_dir
        self.outdir = os.path.join(self.root_dir, 'stats')


    @staticmethod
    def date_parser(s):
        return dt.datetime.strptime(s, '%Y%m%dT%H%M%SZ')


    @staticmethod
    def get_start_date(date_dir):
        return dt.datetime.strptime(
                os.path.basename(date_dir), '%Y%m%d')


    def get_df_1mem(self, mem, pattern, fields):
        df_out = None
        for date_file in sorted(glob.glob(pattern)):
            df = pd.read_csv(date_file, sep='\t',
                    parse_dates=['Date'], date_parser=self.date_parser)
            fcdate = self.get_start_date(os.path.dirname(date_file))
            ind =pd.MultiIndex.from_tuples(
                    [(mem, fcdate, (d - fcdate).total_seconds()/3600)
                        for d in df['Date']],
                    names=['EnsembleMember', 'ForecastDate', 'LeadTime'])
            df = df.set_index(ind)[list(fields)].rename(columns=fields)
            #print(df)
            if df_out is None:
                df_out = df
            else:
                df_out = pd.concat([df_out, df])
            #print(df_out)
        return df_out


    def get_spread_1mem(self, mem_dir):
        mem = int(mem_dir[-3:])
        pattern = os.path.join(mem_dir, 'eval-ref-conc', '20??????',
                'eval_*_errors.txt')
        fields = {
                'Bias_F' : 'Bias',
                'RMSE_F' : 'Spread',
                f'Mean{self.variable_name}_F' : 'Mean',
                f'Mean{self.variable_name}_O' : 'AnalysisMean',
                }
        return self.get_df_1mem(mem, pattern, fields)


    def get_rmse_1mem(self, mem_dir):
        mem = int(mem_dir[-3:])
        pattern = os.path.join(mem_dir, 'eval-osisaf-conc', '20??????',
                'eval_*_errors.txt')
        fields = {
                'Bias_F' : 'Bias',
                'RMSE_F' : 'RMSE',
                f'Mean{self.variable_name}_F' : f'Model{self.variable_name}',
                f'Mean{self.variable_name}_O' : f'Observed{self.variable_name}',
                }
        return self.get_df_1mem(mem, pattern, fields)


    def get_spread_rmse(self):
        pattern = os.path.join(self.root_dir, f'{self.forecast_name}.mem_???')
        df_spread = None
        df_rmse = None
        for mem_dir in sorted(glob.glob(pattern)):
            print(mem_dir)
            # get spread
            df = self.get_spread_1mem(mem_dir)
            if df_spread is None:
                df_spread = df
            else:
                df_spread = pd.concat([df_spread, df])
            # get RMSE
            df = self.get_rmse_1mem(mem_dir)
            if df_rmse is None:
                df_rmse = df
            else:
                df_rmse = pd.concat([df_rmse, df])
        return df_spread, df_rmse

    
    def get_rms_vs_leadtime(self, df_in, vname):
        df = df_in.reset_index()[['LeadTime', vname]]
        df["Dummy"] = df[vname]**2
        df = df.groupby("LeadTime").mean()
        print(df)
        df[vname] = np.sqrt(df["Dummy"])
        return df[[vname]]


    def plot_spread_vs_lead_time(self, df):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(df.index, df['Spread'])
        ax.set_ylabel("Spread")
        ax.set_xlabel("Lead time, hours")
        ofil = os.path.join(self.outdir, "spread_vs_leadtime.png")
        print(f'Saving {ofil}')
        fig.savefig(ofil)


    def plot_rmse_vs_lead_time(self, df):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(df.index, df['RMSE'])
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Lead time, hours")
        ofil = os.path.join(self.outdir, "rmse_vs_leadtime.png")
        print(f'Saving {ofil}')
        fig.savefig(ofil)


    def run(self):
        df_spread, df_rmse = self.get_spread_rmse()
        os.makedirs(self.outdir, exist_ok=True)
        ofil = os.path.join(self.outdir, 'spread.csv')
        print(f'Saving {ofil}')
        df_spread.to_csv(ofil)
        ofil = os.path.join(self.outdir, 'rmse.csv')
        print(f'Saving {ofil}')
        df_rmse.to_csv(ofil)

        # spread vs lead time
        df_lt = self.get_rms_vs_leadtime(
                df_spread, "Spread")
        ofil = os.path.join(self.outdir, 'spread_vs_leadtime.csv')
        print(f'Saving {ofil}')
        df_lt.to_csv(ofil)
        self.plot_spread_vs_lead_time(df_lt)

        # RMSE vs lead time
        df_lt = self.get_rms_vs_leadtime(
                df_rmse, "RMSE")
        ofil = os.path.join(self.outdir, 'rmse_vs_leadtime.csv')
        print(f'Saving {ofil}')
        df_lt.to_csv(ofil)
        self.plot_rmse_vs_lead_time(df_lt)


if __name__ == "__main__":
    args = vars(get_parser().parse_args())
    print(args)
    obj = PlotSpreadRMSE(**args)
    obj.run()
