#! /bin/bash -x

OUTDIR=${1-"out"}
SNAPSHOT=${2-0}
AROME=${3-0}

opts+=("-o $OUTDIR")
[[ $SNAPSHOT -eq 1 ]] && opts+=("-s")
[[ $AROME -eq 1 ]] && opts+=("-a")

for lts in $(seq 0 15)
do
    cmd="./plot_spectrum.py ${opts[@]} -lts $lts"
    #continue
    sbatch --export=COMMAND="$cmd" \
        --time="24:00:00" \
        --job-name="spec_$lts" \
        slurm_batch_job.sh
done 
