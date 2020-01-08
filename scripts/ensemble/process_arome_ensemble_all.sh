#! /bin/bash

for d in `seq -w 10 30`
do
    dt=201803$d
    cmd="./process_arome_ensemble.py $dt"
    echo $cmd
    sem -j 4 $cmd &
done
