#! /bin/bash -x
set -e

# script to merge ensemble members into 1 file for easier browsing and sharing

# required module
# ml load NCO/4.7.9-intel-2018b

function usage {
    echo "merge_members.sh ROOT_DIR FCNAME OUTDIR [MOORINGS_MASK]"
    echo "ROOT_DIR: dir with all the member forecasts"
    echo "  eg /cluster/work/users/timill/nextsimf_forecasts/arome_3km_ec2_arome_ensemble.fram/old_version"
    echo "FCNAME: name of forecast"
    echo "  eg arome_3km_ec2_arome_ensemble.fram"
    echo "OUTDIR: dir to put all the merged files into"
    echo "MOORINGS_MASK: date format of individual members' moorings files"
    echo "  eg 'Moorings.nc' (default) or 'Moorings_%Yd%j.nc'"
    exit 1
}

function merge_one_set {
    [[ -f $ofil ]] && return
    ofil0="$OUTDIR/tmp/tmp.nc" #intermediate file

    # merge ensemble members (not special vars)
    vars="latitude,longitude,Polar_Stereographic_Grid"
    ncecat -O -x -v $vars ${flist[@]} -o $ofil0
    # add the special vars back
    ncks -A -v $vars ${flist[0]} $ofil0
    # rename "record" to "ensemble_member"
    ncrename -O -d "record","ensemble_member" $ofil0
    # reorder dimensions
    ncpdq -O -a "time","ensemble_member" $ofil0 -o $ofil

    # set some other attributes
    # - grid_mapping_name (not needed for time vars)
    for v in "sic" "sit" "snt" "siu" "siv" "longitude" "latitude"
    do
        ncatted -a grid_mapping_name,$v,c,c,polar_stereographic $ofil
    done
    rm $ofil0
}

[[ $# -lt 3 ]] && usage
ROOT_DIR=$1
FCNAME=$2
OUTDIR=$3
MOORINGS_MASK=${4-"Moorings.nc"}
mkdir -p $OUTDIR/tmp

mem1_dirs=($ROOT_DIR/${FCNAME}.mem_001/20??????)
for fcdir in "${mem1_dirs[@]}"
do
    bdate=$(basename $fcdir)
    [[ $bdate == "20180308" ]] && continue
    if [ $MOORINGS_MASK == "Moorings.nc" ]
    then
        flist=($ROOT_DIR/${FCNAME}.mem_???/$bdate/Moorings.nc)
        ofil="$OUTDIR/nextsim_ecmwf_arome_ensemble_forecast-b${bdate}.nc" # output file
        merge_one_set
    else
        for n in $(seq 0 1)
        do
            fcdate=$(date -d "$bdate +${n}days" "+%Y%m%d")
            mfil=$(date -d "$fcdate" "+$MOORINGS_MASK")
            flist=($ROOT_DIR/${FCNAME}.mem_???/$bdate/$mfil)
            ofil="$OUTDIR/nextsim_ecmwf_arome_ensemble_forecast_${fcdate}-b${bdate}.nc" # output file
            merge_one_set
        done
    fi
done
