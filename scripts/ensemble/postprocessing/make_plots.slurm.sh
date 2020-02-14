#!/bin/sh
#SBATCH --account=nn2993k
#SBATCH --job-name=make_arome_plots
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --qos=devel
#SBATCH --time=0-0:30:00
##SBATCH --qos=short
##SBATCH --time=0-2:00:00

cd $SLURM_SUBMIT_DIR
source $HOME/nextsimf.ensemble.src

function plot {
    plot_nextsim_output.py $1 plot.cfg -o $fcdir/figs --no-im
}


for smos in "" "_smos"
do
    fcdir_prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble${smos}.fram
    for i in `seq 1 11`
    do
        s=`printf '%.3i' $i`
        root_dir=${fcdir_prefix}.mem_$s
        for fcdir in $root_dir/201?????
        do
            srun --ntasks=1 --exclusive ./make_plots.sh $fcdir &
        done
    done
done

wait
