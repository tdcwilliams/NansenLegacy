smos=""
#smos="_smos"
fcdir_prefix=$OUTPUT_FORECAST_DATA_DIR/arome_3km_ec2_arome_ensemble${smos}.fram

for i in `seq 1 11`
do
    s=`printf '%.3i' $i`
    fcdir=${fcdir_prefix}.mem_$s
    jn=nsf_ar3_mem${i}_15s${smos}_
    echo ""
    echo ./check_forecast.sh $fcdir $jn
    ./check_forecast.sh $fcdir $jn
done

if [ -f failed.txt ]
then
    echo ""
    echo "Failed experiments:"
    echo "-------------------"
    failed=(`cat failed.txt`)
    for fcdir in ${failed[@]}
    do
        f=$fcdir/failed.txt
        [[ -f $f ]] && failed_dates=(`cat $f`)
        for fcdate in ${failed_dates[@]}
        do
            echo $fcdir/$fcdate
        done
    done
fi

if [ -f moorings_to_fix.txt ]
then
    echo ""
    echo "Moorings to fix:"
    echo "----------------"
    failed=(`cat moorings_to_fix.txt`)
    for fcdir in ${failed[@]}
    do
        f=$fcdir/moorings_to_fix.txt
        [[ -f $f ]] && failed_dates=(`cat $f`)
        for fcdate in ${failed_dates[@]}
        do
            echo $fcdir/$fcdate
        done
    done

    echo ""
    echo "Fix after checking with:"
    echo "------------------------"
    for fcdir in ${failed[@]}
    do
        echo ./fix_moorings.sh $fcdir
    done
fi
