#! /bin/bash

if [ 1 -eq 0 ]
then
    # do 20180309 by itself for testing and save the grid
    cmd="./blend_ecmwf_arome_ensemble_ec2_3h.py 20180309 -g"
    echo $cmd
    $cmd
    exit 0
fi

for d in `seq -w 10 13`
do
    # loop through 1st few dates so we can have a few sequential days
    # for testing inside the model
    dt=201803$d
    cmd="./blend_ecmwf_arome_ensemble_ec2_3h.py $dt"
    #echo $cmd
    #$cmd
    sem -j 4 $cmd &
done

sem --wait
for d in `seq -w 14 30`
do
    # loop through the rest of the dates
    dt=201803$d
    cmd="./blend_ecmwf_arome_ensemble_ec2_3h.py $dt"
    #echo $cmd
    #$cmd
    sem -j 4 $cmd &
done
