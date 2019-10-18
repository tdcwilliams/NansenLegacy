#! /bin/bash

for d in `seq -w 9 31`
do
    dt=201803$d
    cmd="./split_ecmwf.py $dt"
    echo $cmd
    sem -j 4 $cmd &
done
