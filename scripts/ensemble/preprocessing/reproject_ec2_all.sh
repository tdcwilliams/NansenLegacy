#! /bin/bash -x
date0=20190901
date1=20200501
dt=$date0
while [ $dt -le $date1 ]
do
    sem -j8 ./reproject_ec2.py $dt &
    dt=$(date -d "$dt +1day" "+%Y%m%d")
done
