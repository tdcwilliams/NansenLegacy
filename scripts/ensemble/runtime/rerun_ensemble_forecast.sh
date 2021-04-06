#! /bin/bash
# creates the config files for the member ensemble forecasts
# NB no interaction between forecasts
function usage {
    echo Usage: $0 FCNAME DATE1 DATE2 DURATION
    exit 1
}
[[ $# -ne 4 ]] && usage

FCNAME=$1
DATE1=$2
DATE2=$3
DURATION=$4
NUM_ENS=11
#anr="--allow-no-restart"

for i_ens in $(seq 1 $NUM_ENS)
do
    ens_str=$(printf "mem_%.3d" "$i_ens") #eg mem_001
    fcname="${FCNAME}.${ens_str}"
    exe=$OUTPUT_FORECAST_DATA_DIR/$fcname/nextsim.exec
    rerun_forecast.py $fcname $DATE1 $DATE2 $DURATION \
        -ds -1 -c -e $exe --hpc $anr
done
