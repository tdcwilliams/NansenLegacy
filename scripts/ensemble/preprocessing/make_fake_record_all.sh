#! /bin/bash -x

for d in `seq -w 9 31`
do
    ./make_fake_record.py 201803$d
done
