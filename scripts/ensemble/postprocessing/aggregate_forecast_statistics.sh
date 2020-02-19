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
function run {
    [[ $# -ne 1 ]] && { run_usage; exit 1; }
    echo $1
    sbatch \
        ${sbatch_opts[@]} \
        --export=COMMAND="$1" \
        slurm_batch_job.sh
}

[[ $# -ne 1 ]] && { usage; exit 0; }
sbatch_opts=()
sbatch_opts+=("--job-name=agg_fc_stats")
sbatch_opts+=("--time=00:20:00")

prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble
for smos in "" "_smos"
do
    for fcdir in ${prefix}${smos}.fram.mem_???
    do
        fcname=`basename $fcdir`
        run "aggregate_evaluation_statistics.py \
            $fcname -sp eval-osisaf-conc"
        run "aggregate_evaluation_statistics.py \
            $fcname -sp eval-smos"
    done
done
