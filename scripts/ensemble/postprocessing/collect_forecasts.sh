#! /bin/bash
#smos=""
smos="_smos"
fcdir_prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble${smos}.fram
nird_dir=/nird/projects/nird/NS2993K/NERSC_2_METNO/
for subdir in "" nextsim_arome forecasts `basename $fcdir_prefix`
do
    nird_dir+="/$subdir"
    mkdir -p $nird_dir
done

echo "Copying forecasts to:"
echo "$nird_dir"
echo ""
for i in `seq 1 175`
do
    lines+="-"
done

for i in `seq 1 11`
do
    echo ""
    echo $lines
    mem_str="mem_`printf '%.3i' $i`"
    fcdir=${fcdir_prefix}.$mem_str
    for fc_date_dir in $fcdir/201?????
    do
        nc1=$fc_date_dir/Moorings.nc
        fcdate=`basename $fc_date_dir`
        nc2=$nird_dir/nextsim_ice.ecmwf_arome_atmosphere.${fcdate}.${mem_str}.nc
        echo cp $nc1 $nc2
        cp $nc1 $nc2
    done
    echo $lines
    echo ""
done
