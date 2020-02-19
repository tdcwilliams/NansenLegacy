#! /bin/bash
[[ $# -eq 0 ]] && exit 1
fcdir=$1
mkdir $fcdir/figs-day1 $fcdir/figs-day2
for n in `seq 0 7`
do
    ln -s $fcdir/*/figs/*_`printf "%03g" $n`.png $fcdir/figs-day1
done
for n in `seq 8 15`
do
    ln -s $fcdir/*/figs/*_`printf "%03g" $n`.png $fcdir/figs-day2
done

for odir in $fcdir/figs-day?
do
    for v in concentration thickness
    do
        convert +map -delay 20 -loop 0 $odir/${v}*.png $odir/${v}.gif
    done
done
