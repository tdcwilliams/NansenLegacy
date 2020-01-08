#! /bin/bash
# creates the config files for the member ensemble forecasts
# NB no interaction between forecasts
if [ $# -ne 1 ]
then
    echo Usage: $0 CONFIG_FILE
    exit 1
fi

CONFIG_FILE=$1
cfg_base=`basename $CONFIG_FILE`
cfg_root=${NEXTSIM_ENV_ROOT_DIR}/config_files/${cfg_base%*.cfg}
NUM_ENS=11

for i_ens in `seq 1 $NUM_ENS`
do
    # copy the template .cfg file and modify the required fields
    ens_str=`printf "mem_%.3d" "$i_ens"` #eg mem_001
    cfg=${cfg_root}.${ens_str}.cfg
    echo Making $cfg
    cp $CONFIG_FILE $cfg
    sed -i "s|ENS_MEM|$i_ens|g" $cfg
done
