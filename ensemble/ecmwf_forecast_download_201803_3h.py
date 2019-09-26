#! /usr/bin/env python
import sys
from ecmwfapi import ECMWFService
import datetime as dt
import json

server = ECMWFService('mars')
request = {
            "class" : "od",
            "date": '2018-03-01/to/2018-04-01',
            "expver": 1,
            "grid": "0.125/0.125",
            "resol" : "av",
            "area": "90/-180/40/180",
            "levtype": "sfc",
            "param": "10u/10v/2t/2d/tcc/tp/msl/ssrd/strd/sf",
            "step": "0/to/21/by/3",
            "stream": "oper",
            "time": "00",
            "type": "fc",
            "format" : "netcdf",
            }
output = 'legacy_201803_3h.nc'
print(json.dumps(request, indent=4))
print('Output file: %s' %output)
server.execute(request, output)
