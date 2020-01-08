#! /bin/bash
dir0=/Data/sim/data/ECMWF_forecast_arctic_3h/raw_files
dir1=/Data/sim/data/ECMWF_forecast_arctic_3h
date0=20180202
date1=20180331
dt=$date0

while [ $dt -le $date1 ]
do
    cmd="./process_ecmwf_download.py $dt $dir0 $dir1"
    echo $cmd
    $cmd
    #sem -j 4 $cmd &
    dt=`date -d "$dt +1days" "+%Y%m%d"`
done
