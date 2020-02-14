#! /bin/bash
[[ $# -gt 2 ]] && { echo "Usage: $0 FCDIR [JOBNAME]"; exit 1; }
FCDIR=$1
JOBNAME=$2
rm -f failed.txt moorings_to_fix.txt

for ddir in $FCDIR/????????
do
    echo $ddir
    dt1=`basename $ddir`
    dt2=`date -d "$dt1 +2days" "+%Y%m%d"`

    if [ ! -z $JOBNAME ]
    then
        # don't raise error if job still running
        sq=`squeue -h -n ${JOBNAME}_${dt1}`
        [[ ! -z $sq ]] && continue
    fi

    # check output for final time
    [[ ! -f $ddir/field_${dt2}T000000Z.dat ]] \
        && { failed+=($dt1); echo Failed; continue; } \
        || echo "Finished OK"

    # check right number of time records in moorings file
    tinfo=`ncdump -h $ddir/Moorings.nc | grep UNLIMITED`
    # grab stuff after "(" eg "16 currently)"
    tw=(${tinfo#*\(})
    # 1st word is number of recs
    nrecs=${tw[0]}
    [[ $nrecs -ne 16 ]] \
        && { moorings_to_fix+=($dt1=$nrecs); echo "Need to fix moorings"; } \
        || echo "Moorings OK"
done

if [ ${#failed[@]} -gt 0 ]
then
    f=$FCDIR/failed.txt
    echo "Some failed forecasts in $FCDIR"
    echo "- see $f"
    printf '%s\n' "${failed[@]}" > $f
    f=failed.txt
    touch $f
    echo $FCDIR >> $f
fi
if [ ${#moorings_to_fix[@]} -gt 0 ]
then
    f=$FCDIR/moorings_to_fix.txt
    echo "Some erroneous moorings in $FCDIR"
    echo "- see $f"
    printf '%s\n' "${moorings_to_fix[@]}" > $f
    f=moorings_to_fix.txt
    touch $f
    echo $FCDIR >> $f
fi
