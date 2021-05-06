#! /bin/bash -x
root_dir=/cluster/work/users/timill/nextsimf_forecasts/

#expt_dir="arome_3km_ec2_arome_ensemble_smos.fram"
expt_dir="arome_3km_ec2_arome_ensemble.fram/new_version"
#expt_dir="arome_3km_ec2_arome_ensemble.fram/old_version"
#expt_dir="arome_3km_ec2_arome_ensemble.fram/old_version_transposed"
my_storage="/nird/projects/nird/NS2993K/NORSTORE_OSL_DISK/NS2993K/timill"
backup_dir="$my_storage/Model-Results/nextsimf-ensemble/arome_3km/"
outdir=$backup_dir/$expt_dir
mkdir -p $outdir

cd $root_dir/$expt_dir
tbad=() #track errors in tar
cpbad=() #track errors in cp
for subdir in *.mem_??? merged_files analysis arome_3km_ec2_3h.fram
do
    tfil=${subdir}.tar.gz
    tar -zcf $tfil $subdir || { tbad+=($tfil); continue; }
    sum1=(`md5sum $tfil`)

    tfil2=$outdir/$tfil
    cp $tfil $tfil2
    sum2=(`md5sum $tfil2`)
    [[ "${sum1[0]}" == "${sum2[0]}" ]] && rm $tfil || cpbad+=($tfil)
done

ecode=0
for tfil in ${tbad[@]}
do
    echo tar error: $tfil
    ecode=1
done
for tfil in ${cpbad[@]}
do
    echo cp error: $tfil
    ecode=1
done
exit $ecode
