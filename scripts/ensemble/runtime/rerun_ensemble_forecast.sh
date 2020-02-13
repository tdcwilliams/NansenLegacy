#! /bin/bash
# creates the config files for the member ensemble forecasts
# NB no interaction between forecasts
if [ $# -ne 4 ]
then
    echo Usage: $0 FCNAME DATE1 DATE2 DURATION
    exit 1
fi

FCNAME=$1
DATE1=$2
DATE2=$3
DURATION=$4
NUM_ENS=11
#anr="--allow-no-restart"

for i_ens in `seq 1 $NUM_ENS`
do
    ens_str=`printf "mem_%.3d" "$i_ens"` #eg mem_001
    rerun_forecast.py ${FCNAME}.${ens_str} $DATE1 $DATE2 $DURATION \
        -np --hpc $anr
done
