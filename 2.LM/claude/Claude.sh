#!/bin/bash

# ตั้งค่า model
claude config set model claude-sonnet-4-5-20250929

# ตรวจสอบ model ที่ใช้งาน
echo "Current model:"
claude config get model
echo ""

START=0
END=1000
STEP=1000
MAX=19801

while [ $END -le $MAX ]
do
    echo "Running rows $START to $END ..."
    python3 Claude.py -s $START -e $END

    # รอ 5 ชั่วโมง 2 นาที (5*3600 + 120 = 18120 วินาที)
    sleep 18120

    # เพิ่มค่าไปอีก 200
    START=$((START+STEP))
    END=$((END+STEP))
done
