#! /bin/bash
date0=20190901
date1=20200501
dt=$date0
while [ $dt -le $date1 ]
do
    cmd="./reproject_ec2.py $dt"
    echo $cmd
    sem -j8 $cmd &
    dt=$(date -d "$dt +1day" "+%Y%m%d")
done
