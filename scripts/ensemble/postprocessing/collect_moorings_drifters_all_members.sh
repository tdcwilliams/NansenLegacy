#! /bin/bash -x
set -e

# get netcdf modules with
#   ml load NCO/4.7.9-intel-2018b
#   ml load ncview/2.1.7-intel-2018b
function usage {
    echo "Usage: $(basename $this_file) ROOT_DIR FCNAME [STEP]"
    echo "Loops over ensemble members of a forecast to make an analysis for each"
    echo "ROOT_DIR: where individual members are located"
    echo "  eg /cluster/work/users/timill/nextsimf_forecasts/arome_3km_ec2_arome_ensemble.fram/new_version"
    echo "FCNAME: name of forecast"
    echo "  eg arome_3km_ec2_arome_ensemble.fram"
    exit 1
}

this_file=$(readlink -f $0)
this_dir=$(dirname $this_file)
[[ $# -lt 2 ]] && usage

ROOT_DIR=$1
FCNAME=$2
STEP=${3-1}
dir_prefix="${ROOT_DIR}/${FCNAME}.mem_"
outdir=$ROOT_DIR/analysis/outputs
mkdir -p $outdir

if [ $STEP -eq 1 ]
then
    # collect "analysis" for each ensemble (1st day of each ensemble)
    for ens_dir in ${dir_prefix}???
    do
        $this_dir/collect_moorings.sh $ens_dir
        $this_dir/collect_drifters.sh $ens_dir $ens_dir/analysis/outputs
    done
    exit 0
fi


ens_dir=${dir_prefix}001
for fcdir in $ens_dir/20??????
do
    fcdate=$(basename $fcdir)
    [[ $fcdate == "20180308" ]] && continue
    ofil=$outdir/analysis_${fcdate}.nc 
    [[ -f $ofil ]] && continue
    ncea ${dir_prefix}???/analysis/outputs/Moorings_${fcdate}.nc -o $ofil
done
