#! /bin/bash -x
set -e

function usage {
    echo "evaluate_forecasts.sh ROOT_DIR FCNAME [MOORINGS_MASK]"
}
function run {
    echo $1
    sbatch \
        ${sbatch_opts[@]} \
        --export=COMMAND="$1" \
        slurm_batch_job.sh
}
function check_and_run {
    [[ -d $2 ]] && { echo "$2 exists"; return; } # don't repeat evaluations
    run "$1 -o $2"
}

ROOT_DIR=$1
FCNAME=$2
MOORINGS_MASK=${3-"Moorings.nc"}
dir_prefix="${ROOT_DIR}/${FCNAME}.mem_"

sbatch_opts=()
sbatch_opts+=("--job-name=eval-fc")
sbatch_opts+=("--time=00:15:00")

for fcdir in ${dir_prefix}???
do
    # evaluate "analysis" for each member
    ./eval_freerun.sh $fcdir/analysis "Moorings_%Y%m%d.nc" 1 1 1 1
done

# evaluate true analysis
# - no drift eval implemented yet
#./eval_freerun.sh $ROOT_DIR/analysis "Moorings_%Y%m%d.nc" 1 0 1 1

for fcdir in ${dir_prefix}???/201?????
do
    fcopts="$fcdir -mm $MOORINGS_MASK"
    fcopts2="-s OsisafConc -sig 10"
    check_and_run "evaluate_forecast.py $fcopts $fcopts2" "$fcdir/eval-osisaf-conc"
        #sig=0 better for looking at ice edge
    fcopts2="-s SmosThick -sig 10"
    check_and_run "evaluate_forecast.py $fcopts $fcopts2" "$fcdir/eval-smos"
        #sig>0 doesn't add anything
done
