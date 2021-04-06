#! /bin/bash
# creates the config files for the member ensemble forecasts
# NB no interaction between forecasts
function usage {
    echo "Usage: $0 CONFIG_FILE [RESTART_FORECAST_DATE_DIR]"
    exit 1
}

[[ $# -eq 0 ]] && usage
CONFIG_FILE=$1
RESTART_FORECAST_DATE_DIR=$2

cfg_base=`basename $CONFIG_FILE`
fcname=${cfg_base%*.cfg}
cfg_root=${NEXTSIM_ENV_ROOT_DIR}/config_files/$fcname
NUM_ENS=11

for i_ens in $(seq 1 $NUM_ENS)
do
    # copy the template .cfg file and modify the required fields
    ens_str=$(printf "mem_%.3d" "$i_ens") #eg mem_001
    cfg=${cfg_root}.${ens_str}.cfg
    echo Making $cfg
    cp $CONFIG_FILE $cfg
    sed -i "s|%%{ENS_MEM}|$i_ens|g" $cfg

    # if we have a restart, we make a fake forecast for the date
    # before we want to start and link into the forecast dir
    # -then we find the restart in the usual way
    [[ -z "$RESTART_FORECAST_DATE_DIR" ]] && continue
    mkdir -p $OUTPUT_FORECAST_DATA_DIR
    fcdir=$OUTPUT_FORECAST_DATA_DIR/${fcname}.${ens_str}
    mkdir -p $fcdir
    cmd="ln -s $RESTART_FORECAST_DATE_DIR $fcdir"
    echo $cmd
    $cmd
done
