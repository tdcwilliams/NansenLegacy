#! /bin/bash -x

#opts="-g" #uncomment to save grid 1st iteration
for d in `seq -w 9 31`
do
    ./blend_ecmwf_arome_ensemble_ec2_3h.py 201803$d $opts
    opts=""
done
