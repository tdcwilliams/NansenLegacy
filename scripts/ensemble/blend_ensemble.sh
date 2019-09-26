#! /bin/bash

for d in `seq -w 9 30`
do
    dt=201803$d
    cmd="./blend_ecmwf_arome_ensemble_ec2_6h.py $dt"
    echo $cmd
    $cmd
done
