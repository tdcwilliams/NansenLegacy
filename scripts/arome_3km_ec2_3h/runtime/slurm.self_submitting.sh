#!/bin/bash -x
#SBATCH --account=nn2993k
#SBATCH --job-name=sens_ec2_arome_000
#SBATCH --output=logs/slurm.%j.log   # Stdout & stderr
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1

## Email info
#SBATCH --mail-type=ALL   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=timothy.williams@nersc.no

cd $SLURM_SUBMIT_DIR
source /cluster/home/timill/pynextsimf.src
split_length=7
date_begin=20180210
date_final=20180331
fcname=arome_3km_ec2_3h.fram

# get $START and check it
START=${START-"$date_begin"}
[[ $START -lt $date_begin ]] && \
    { echo "Bad input start date (<$date_begin)"; exit 1; }
[[ $START -ge $date_final ]] &&  \
    { echo "Bad input start date (>$date_final)"; exit 1; }

# set date_end
date_end=`date -d "$START + $split_length days" "+%Y%m%d"`
[[ $date_end -gt $date_final ]] && date_end=$date_final

# not using -nr/--allow-no-restart since init CS2-SMOS not working
# - we make a restart on fram.ad.nersc.no and put it in 20180209/restart
exe=$OUTPUT_FORECAST_DATA_DIR/$fcname/nextsim.exec
rerun_forecast.py $fcname $START $date_end 1 -c -e $exe -ds -1

# Exit if we have reached $date_final
[[ $date_end -eq $date_final ]] && exit 0

# Otherwise resubmit and exit
start=$(date -d "$date_end + 1day" "+%Y%m%d")
sbatch --export=START=$start $0
exit $?
