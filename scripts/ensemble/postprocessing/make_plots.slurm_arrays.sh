#!/bin/bash -x
#SBATCH --account=nn2993k
#SBATCH --job-name=make_arome_plots
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --qos=devel
#SBATCH --time=0-0:30:00
##SBATCH --qos=short
##SBATCH --time=0-2:00:00
#SBATCH --array=1-264
#SBATCH --output=logs/slurm.%A_%a.out

cd $SLURM_SUBMIT_DIR
source $HOME/nextsimf.ensemble.src

function plot {
    plot_nextsim_output.py $1 plot.cfg -o $1/figs --no-im
}

#smos=""
smos="_smos"
fcdir_prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble${smos}.fram
fcdirs=($fcdir_prefix.mem_???/201?????)
[[ $SLURM_ARRAY_TASK_ID -lt ${#fcdirs[@]} ]] \
    && plot ${fcdirs[$SLURM_ARRAY_TASK_ID]}
