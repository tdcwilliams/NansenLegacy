#! /bin/bash
function usage {
    echo $0 FORECAST_NAME
}
function check_and_run_usage {
    echo check_and_run COMMAND OUTPUTDIR
}
function run_usage {
    echo run COMMAND
}
function check_and_run {
    [[ $# -ne 2 ]] && { check_and_run_usage; exit 1; }
    [[ -d $2 ]] && { echo "$2 exists"; return; } # don't repeat evaluations
    run "$1 -o $2"
}
function run {
    [[ $# -ne 1 ]] && { run_usage; exit 1; }
    echo $1
    sbatch \
        ${sbatch_opts[@]} \
        --export=COMMAND="$1" \
        slurm_batch_job.sh
}

[[ $# -ne 1 ]] && { usage; exit 0; }
FORECAST_NAME=$1
sbatch_opts=()
sbatch_opts+=("--job-name=eval-fc")
sbatch_opts+=("--time=00:15:00")

prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble
for smos in "" "_smos"
do
    for fcdir in ${prefix}${smos}.fram.mem_???/201?????
    do
        check_and_run "evaluate_forecast.py $fcdir -s OsisafConc" $fcdir/eval-osisaf-conc
            #sf=0 better for looking at ice edge
        check_and_run "evaluate_forecast.py $fcdir -s SmosThick" $fcdir/eval-smos
            #sf>0 doesn't add anything
    done
done
