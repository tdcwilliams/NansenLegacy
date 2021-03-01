#! /bin/bash

# # do 20180309 by itself for testing and save the grid
# cmd="./blend_ecmwf_arome_ensemble_ec2_3h.py 20180309 -g"
# echo $cmd
# $cmd || exit 1

for d in `seq -w 10 31`
do
    dt=201803$d
    cmd="./blend_ecmwf_arome_ensemble_ec2_3h.py $dt"
    echo $cmd
    $cmd
    cmd="./make_fake_record.py $dt"
    echo $cmd
    $cmd
done
