#! /bin/bash -x
fcname=arome_3km_ec2_3h.fram
ndir=/nird/projects/nird/NS2993K/NORSTORE_OSL_DISK/NS2993K/timill/nextsimf_forecasts/$fcname/2018
target_dir=/cluster/work/users/timill/nextsimf_forecasts/arome_3km_ec2_arome_ensemble.fram/old_version
fc_root_dir=$target_dir/$fcname
mkdir -p $fc_root_dir
cd $fc_root_dir
for tfil in $ndir/??/*.tar.gz
do
    tfil2=$fc_root_dir/$(basename $tfil)
    cp $tfil $tfil2
    tar -zxf $tfil2
    rm $tfil2
done
