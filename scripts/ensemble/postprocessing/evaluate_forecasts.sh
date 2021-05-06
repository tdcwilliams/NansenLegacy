#! /bin/bash -x
set -e

function usage {
    echo "evaluate_forecasts.sh ROOT_DIR FCNAME [MOORINGS_MASK]"
    exit 1
}
function run {
    cmd="singularity exec --cleanenv $PYNEXTSIM_SIF $1"
    echo $cmd
    sbatch \
        ${sbatch_opts[@]} \
        --export=COMMAND="$cmd" \
        slurm_batch_job.sh
}
function check_and_run {
    [[ -d $2 ]] && { echo "$2 exists"; return; } # don't repeat evaluations
    run "$1 -o $2"
}

[[ $# -lt 2 ]] && usage
ROOT_DIR=$1
FCNAME=$2
MOORINGS_MASK=${3-"Moorings.nc"}
dir_prefix="${ROOT_DIR}/${FCNAME}.mem_"
control_dir=$ROOT_DIR/arome_3km_ec2_3h.fram
an_dir=$ROOT_DIR/analysis
an_mm="analysis_%Y%m%d.nc"

sbatch_opts=()
sbatch_opts+=("--job-name=eval-fc")
sbatch_opts+=("--time=00:30:00")

eval_mem_analyses=0
eval_true_analysis=0
eval_forecasts=0
comp_forecasts_an=1

if [ "$eval_mem_analyses" == "1" ]
then
    for fcdir in $control_dir ${dir_prefix}???
    do
        # evaluate "analysis" for each member
        ./eval_freerun.sh $fcdir/analysis "Moorings_%Y%m%d.nc" 1 1 1 1
    done
fi

if [ "$eval_mem_analyses" == "1" ]
then
    # evaluate true analysis
    # - no drift eval implemented yet
    ./eval_freerun.sh $an_dir $an_mm 1 0 1 1
fi

if [ "$eval_forecasts" == "1" ] || [ "$comp_forecasts_an" == "1" ]
then
    for fcdir in $control_dir/201????? ${dir_prefix}???/201?????
    do
        # check for bad link
        [[ ! -d $fcdir ]] && continue
        fcdate=$(basename $fcdir)
        root_dir=$(dirname $fcdir)
        fcopts="$fcdir -mm $MOORINGS_MASK"

        if [ "$eval_forecasts" == "1" ]
        then
            # osisaf conc
            fcopts2="-s OsisafConc -sig 10"
            check_and_run "evaluate_forecast.py $fcopts $fcopts2" "$root_dir/eval-osisaf-conc/$fcdate"
            check_and_run "evaluate_forecast.py $fcopts $fcopts2 -ee" "$root_dir/eval-osisaf-extent/$fcdate"

            # smos thickness
            fcopts2="-s SmosThick -sig 10"
            check_and_run "evaluate_forecast.py $fcopts $fcopts2" "$root_dir/eval-smos/$fcdate"
        fi

        [[ "$root_dir" == "$control_dir" ]] && continue
        [[ "$comp_forecasts_an" != "1" ]] && continue
        # compare ensemble forecasts against analysis
        fcopts2="-s NextsimRefConc -rmd $an_dir/outputs -rmm $an_mm"
        check_and_run "evaluate_forecast.py $fcopts $fcopts2" "$root_dir/eval-ref-conc/$fcdate"
        fcopts2="-s NextsimRefThick -rmd $an_dir/outputs -rmm $an_mm"
        check_and_run "evaluate_forecast.py $fcopts $fcopts2" "$root_dir/eval-ref-thick/$fcdate"
    done
fi
