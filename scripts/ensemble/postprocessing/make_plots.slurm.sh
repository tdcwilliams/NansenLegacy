#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=make_arome_plots
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --qos=devel
#SBATCH --time=0-0:30:00
##SBATCH --qos=short
##SBATCH --time=0-2:00:00
#SBATCH --output=logs/slurm.%j.out

cd $SLURM_SUBMIT_DIR
source $HOME/nextsimf.ensemble.src
function plot {
    figdir=$1/figs
    [[ -d $figdir ]] && return
    plot_nextsim_output.py $1 plot.cfg -o $figdir --no-im
}

prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble
for smos in "" "_smos"
do
    for fcdir in ${prefix}${smos}.fram.mem_???/201?????
    do
        srun --ntasks=1 plot $fcdir &
    done
done

wait
