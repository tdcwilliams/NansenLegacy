#! /bin/bash -x
set -e

# script to merge ensemble members into 1 file for easier browsing and sharing

# required module
# ml load NCO/4.7.9-intel-2018b

function usage {
    echo "merge_members.sh ROOT_DIR"
    echo "ROOT_DIR: dir with all the member forecasts"
    echo "  eg
    /cluster/work/users/timill/nextsimf_forecasts/arome_3km_ec2_arome_ensemble.fram/new_version"
    echo "FCNAME: name of forecast"
    echo "  eg arome_3km_ec2_arome_ensemble.fram"
    echo "OUTDIR: dir to put all the merged files into"
    echo "MOORINGS_MASK: date format of individual members' moorings files"
    echo "  eg 'Moorings.nc' (default) or 'Moorings_%Yd%j.nc'"
    exit 1
}

[[ $# -lt 1 ]] && usage
ROOT_DIR=$1

# output dirs
merged_dir=$ROOT_DIR/merged_files
mean_dir=$ROOT_DIR/ensemble_means
spread_dir=$ROOT_DIR/ensemble_spreads
anom_dir=$ROOT_DIR/anomalies_det
det_dir=$ROOT_DIR/arome_3km_ec2_3h.fram/analysis/outputs
mkdir -p $mean_dir $spread_dir $anom_dir

prefix="nextsim_ecmwf_arome_ensemble_forecast_"
for bday in $(seq -w 9 31)
do
    bdate="201803$bday"
    bdir_prefix="$ROOT_DIR/arome_3km_ec2_arome_ensemble.fram.mem_"
    for lead_time in 1 0
    do
        fdate=$(date -d "$bdate + ${lead_time}days" "+%Y%m%d")
        fdate_j=$(date -d "$bdate + ${lead_time}days" "+%Yd%j")
        flist=(${bdir_prefix}???/$bdate/Moorings_${fdate_j}.nc)

        # save the ensemble average
        fname_mean="$mean_dir/${prefix}mean_${fdate}-b${bdate}.nc"
        if [ ! -f $fname_mean ]
        then
            ncea ${flist[@]} -o $fname_mean
        fi

        # save the ensemble spread
        fname="$merged_dir/${prefix}${fdate}-b${bdate}.nc"
        fname_spread="$spread_dir/${prefix}spread_${fdate}-b${bdate}.nc"
        if [ -f $fname ] && [ ! -f $fname_spread ]
        then
            ncdiff $fname $fname_mean -o $fname_spread
        fi
    done

    # save the anomaly from the deterministic run
    fname_det="$det_dir/Moorings_${bdate}.nc"
    fname_anom="$anom_dir/${prefix}diff_det_${fdate}-b${bdate}.nc"
    if [ -f $fname ] && [ ! -f $fname_anom ]
    then
        ncdiff $fname $fname_det -o $fname_anom
    fi
done
