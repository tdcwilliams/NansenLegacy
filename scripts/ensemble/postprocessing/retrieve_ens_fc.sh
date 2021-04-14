#! /bin/bash -x
# script to copy old forecast back to work from NIRD
fcname=arome_3km_ec2_arome_ensemble.fram
nird_dir=/nird/projects/nird/NS2993K/NORSTORE_OSL_DISK/NS2993K/timill/Model-Results/nextsimf-ensemble/arome_3km/$fcname
target_dir=/cluster/work/users/timill/nextsimf_forecasts/$fcname/old_version/
mkdir -p $target_dir
cd $target_dir
for tfil in $nird_dir/arome_3km_ec2_arome_ensemble.fram.mem_0??.tar.gz
do
    tfil2=$(basename $tfil)
    cp $tfil .
    tar -zxvf $tfil2
    rm $tfil2
done
