#! /bin/bash -x
root_dir=/cluster/work/users/timill/nextsimf_forecasts/arome_3km_ec2_arome_ensemble.fram/
#root_dir=/cluster/work/users/timill/nextsimf_forecasts/arome_3km_ec2_arome_ensemble_smos.fram

expt_dir=$root_dir/new_version
#expt_dir=$root_dir/old_version
#expt_dir=$root_dir/old_version_transposed
backup_dir=/nird/projects/nird/NS2993K/NORSTORE_OSL_DISK/NS2993K/timill/Model-Results/nextsimf-ensemble/arome_3km/

ename=`basename $expt_dir`
outdir=$backup_dir/$ename
mkdir -p $outdir

cd $expt_dir
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
