#! /bin/bash -x
function usage {
    echo "Usage: $this_file ROOT_DIR"
    exit 1
}

this_file=$(basename $0)
[[ $# -ne 1 ]] && usage

ROOT_DIR=$1
outdir=$ROOT_DIR/analysis
mkdir -p $outdir

for fcdir in $ROOT_DIR/20??????
do
    fcdate=$(basename $fcdir)
    [[ $fcdate == "20180308" ]] && continue
    nc1=$fcdir/Moorings.nc
    nc2=$outdir/$(date -d "$fcdate" "+Moorings_%Yd%j.nc")
    ncks -d time,0,7 $nc1 -o $nc2
done
rm $f
