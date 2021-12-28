#!/bin/env bash
set -ex

cd class
# ./launch.sh /checkpoint/apratik/data_dec/baseline apartment_0/class/baseline/no_noise 10 2 10trajrun_baseline
# ./launch.sh /checkpoint/apratik/data_dec/circle apartment_0/class/circle/no_noise 10 2 10trajrun_circle
./launch.sh /checkpoint/apratik/data_dec/straightline apartment_0/class/baseline/no_noise 10 2 10trajrun_straightline

cd ../instance
./launch.sh /checkpoint/apratik/data_dec/baseline apartment_0/instance/baseline/no_noise 10 2 10trajrun_baseline
./launch.sh /checkpoint/apratik/data_dec/circle apartment_0/instance/circle/no_noise 10 2 10trajrun_circle
./launch.sh /checkpoint/apratik/data_dec/straightline apartment_0/instance/straightline/no_noise 10 2 10trajrun_straightline