#! /bin/bash
for smos in "" "_smos"
do
    fcdir=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_3h${smos}.fram
    nird_dir=/nird/projects/nird/NS2993K/NERSC_2_METNO/
    for subdir in "" "nextsim_arome" "forecasts" "nextsim_ec2$smos"
    do
        nird_dir+="/$subdir"
        mkdir -p $nird_dir
    done

    echo "Copying forecasts to:"
    echo "$nird_dir"
    lines=`printf '=%.0s' {1..175}`

    for fc_date_dir in $fcdir/201?????
    do
        nc1=$fc_date_dir/Moorings.nc
        fcdate=`basename $fc_date_dir`
        nc2=$nird_dir/nextsim_ice.ecmwf_atmosphere.${fcdate}.nc
        echo cp $nc1 $nc2
        cp $nc1 $nc2
    done
done
