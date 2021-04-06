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
#QOS_LINE

## Ensure we have a clean environment
#SBATCH --export=NONE

## Email info
#SBATCH --mail-type=ALL   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=EMAIL # email to the user

source $HOME/pynextsimf.src

# set options for run_forecast.py
# - capitalised names to be substituted in submit_forecast_hpc.py
forecast_dir="FORECAST_DIR"
pyargs="PYARGS"

#run forecast
logdir=$forecast_dir/logs
mkdir -p $logdir
export NEXTSIM_DATA_DIR=$forecast_dir/data
run_forecast.py $pyargs &> $logdir/run_forecast.log || exit 1
