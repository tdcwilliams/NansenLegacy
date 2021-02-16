#!/bin/bash -x
## Project:
#SBATCH --account=ACCOUNT_NUMBER
## Job name:
#SBATCH --job-name=JOB_NAME
## Output file
#SBATCH --output=FORECAST_DIR/logs/slurm.JOB_NAME.%j.log         # Stdout & stderr
## Wall time limit:
#SBATCH --time=WALL_TIME
## Number of nodes:
#SBATCH --nodes=NUM_NODES
## Number of tasks (total)
#SBATCH --ntasks=NUM_TASKS
## Set OMP_NUM_THREADS
#SBATCH --cpus-per-task=1

## Queue Option (preproc for 1 node; normal for >4 nodes; also can be bigmem)
#SBATCH --qos=QOS

## Email info
#SBATCH --mail-type=ALL   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=EMAIL # email to the user

source $HOME/nextsimf.ensemble.src

# set options for run_forecast.py
# - capitalised names to be substituted in run_forecast_hpc.py
forecast_dir="FORECAST_DIR"
forecast_name="FORECAST_NAME"
date_begin="DATE_BEGIN"
duration="DURATION"
allow_no_restart="ALLOW_NO_RESTART"
date_shift="DATE_SHIFT"
no_post_proc="NO_POST_PROC"
exec_name="EXEC_NAME"
#mumps_memory="MUMPS_MEMORY"
mumps_memory="2000"

#run forecast
logdir=$forecast_dir/logs
cd $SCRATCH
cp -a $exec_name .
pwd
ls

forecast_preproc.py $forecast_name $date_begin $duration \
        $allow_no_restart \
        --assimilation-date-shift $date_shift \
        $no_post_proc \
        > $logdir/forecast_preproc.log || exit 1

mpirun ./`basename $exec_name` -mat_mumps_icntl_23 $mumps_memory \
    --config-files=$forecast_dir/inputs/nextsim.cfg \
    &> $logdir/nextsim.${SLURM_JOB_ID}.log || exit 1

forecast_postproc.py $forecast_name $date_begin $duration \
        $allow_no_restart \
        --assimilation-date-shift $date_shift \
        $no_post_proc \
        > $logdir/forecast_postproc.log || exit 1
