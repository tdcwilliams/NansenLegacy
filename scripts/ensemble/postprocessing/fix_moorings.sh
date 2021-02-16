#! /bin/bash
fcdir=$1
f=$fcdir/moorings_to_fix.txt
[[ ! -f $f ]] && { echo Nothing to do; exit 0; }
nrecs_req=16
list=(`cat $f`)
for el in ${list[@]}
do
    fcdate=${el%=*}
    nrecs=${el#*=}
    ncfil=$fcdir/$fcdate/Moorings.nc
    mv $ncfil ${ncfil}.bak
    n0=$((nrecs-nrecs_req))
    n1=$((nrecs-1))
    echo ncks -d time,$n0,$n1 ${ncfil}.bak -o $ncfil
    ncks -d time,$n0,$n1 ${ncfil}.bak -o $ncfil
done
rm $f
