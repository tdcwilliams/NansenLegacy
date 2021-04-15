#! /bin/bash -x
set -e

# get netcdf modules with
#   ml load NCO/4.7.9-intel-2018b
#   ml load ncview/2.1.7-intel-2018b
function usage {
    echo "Usage: $this_file ROOT_DIR"
    echo "Collects 1st day of one member of an ensemble forecast to make an analysis"
    exit 1
}

this_file=$(basename $0)
[[ $# -ne 1 ]] && usage

ROOT_DIR=$1
outdir=$ROOT_DIR/analysis/outputs
mkdir -p $outdir

for fcdir in $ROOT_DIR/20??????
do
    fcdate=$(basename $fcdir)
    [[ $fcdate == "20180308" ]] && continue
    nc2=$outdir/Moorings_${fcdate}.nc
    [[ -f $nc2 ]] && continue
    # can we just link?
    nc1=$fcdir/Moorings_$(date -d $fcdate "+%Yd%j").nc
    [[ -f $nc1 ]] && ln -s $nc1 $nc2 && continue
    # else need to extract 1st day
    nc1=$fcdir/Moorings.nc
    [[ -f $nc1 ]] && ncks -d time,0,7 $nc1 -o $nc2 && continue
done
