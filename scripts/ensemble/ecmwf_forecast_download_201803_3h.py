#! /usr/bin/env python
import sys
from ecmwfapi import ECMWFService
import datetime as dt
import json
from multiprocessing.pool import Pool

def download(pair):
    step, output = pair
    request = {
                "class" : "od",
                "date": '2018-02-01/to/2018-04-01',
                "expver": 1,
                "grid": "0.125/0.125",
                "resol" : "av",
                "area": "90/-180/40/180",
                "levtype": "sfc",
                "param": "10u/10v/2t/2d/tcc/tp/msl/ssrd/strd/sf",
                "step": step,
                "stream": "oper",
                "time": "00",
                "type": "fc",
                "format" : "netcdf",
                }
    print(json.dumps(request, indent=4))
    print('Output file: %s' %output)
    server.execute(request, output)

server = ECMWFService('mars')
pairs = [
        ("0/to/21/by/3" , 'legacy_201803_3h.1.nc'),
        ("24/to/45/by/3", 'legacy_201803_3h.2.nc'),
        ("48/to/69/by/3", 'legacy_201803_3h.3.nc'),
        ("72/to/93/by/3", 'legacy_201803_3h.4.nc'),
        ]
p = Pool(4)
p.map(download, pairs)
