#!/bin/bash

START=0
END=1000
STEP=1000
MAX=19801

while [ $END -le $MAX ]
do
    echo "Running rows $START to $END ..."
    python3 ag_Claude.py -s $START -e $END

    # รอ 5 ชั่วโมง 2 นาที (5*3600 + 120 = 18120 วินาที)
    sleep 18120

    # เพิ่มค่าไปอีก 200
    START=$((START+STEP))
    END=$((END+STEP))
done
