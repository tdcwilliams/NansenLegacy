#! /bin/bash -x
#SBATCH --account=nn2993k
#SBATCH --job-name=JOBNAME
#SBATCH --time 30:00:00
#SBATCH --partition=bigmem
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/slurm.%j.out

[[ -z $COMMAND ]] \
    && { echo "enter command to run with --export=COMMAND=..."; exit 1; }
source $HOME/nextsimf.ensemble.src
cd $SLURM_SUBMIT_DIR
$COMMAND
