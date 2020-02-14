#! /bin/bash
expt_dir=$1
backup_dir=$2

ename=`basename $expt_dir`
outdir=$backup_dir/$ename
mkdir -p $outdir

cd $expt_dir
bad=() #track errors
for subdir in mem*
do
    tfil=${subdir}.tar.gz
    tar -zcf $tfil $subdir
    sum1=`md5sum tfil`

    tfil2=$outdir/$tfil
    cp $tfil $tfil2
    sum2=`md5sum $tfil2`
    [[ "$sum1" != "$sum2" ]] && bad+=($tfil)
done

ecode=0
for tfil in ${bad[@]}
do
    echo corrupted file: $tfil
    ecode=1
done
exit $ecode
