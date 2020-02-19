#! /bin/bash

sbatch_opts=()
sbatch_opts+=("--job-name=arome_plots")
sbatch_opts+=("--time=00:30:00")
#sbatch_opts+=("--mem-per-cpu=10G")

prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble
for smos in "" "_smos"
do
    for fcdir in ${prefix}${smos}.fram.mem_???/201?????
    do
        cmd="plot_nextsim_output.py \
            $fcdir plot.cfg \
            -o $fcdir/figs --no-im"
        sbatch \
            ${sbatch_opts[@]} \
            --export=COMMAND="$cmd" \
            slurm_batch_job.sh
    done
done
