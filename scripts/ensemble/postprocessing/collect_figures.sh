#! /bin/bash
mkdir -p logs
sbatch_opts=()
sbatch_opts+=("--job-name=collect_figures")
sbatch_opts+=("--time=00:40:00")
#sbatch_opts+=("--mem-per-cpu=10G")

[[ -z $OUTPUT_FORECAST_DATA_DIR ]] && { echo "OUTPUT_FORECAST_DATA_DIR"; exit 1; }
prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble
for smos in "" "_smos"
do
    for fcdir in ${prefix}${smos}.fram.mem_???
    do
        cmd="./collect_figures_1fc.sh $fcdir"
        echo $cmd
        #continue
        sbatch \
            ${sbatch_opts[@]} \
            --export=COMMAND="$cmd" \
            slurm_batch_job.sh
    done
done
