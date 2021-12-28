#!/bin/env bash
set -ex

# # Setting 1
# ./launch.sh /checkpoint/apratik/data_dec/baseline apartment_0/instance/baseline/no_noise 10 2 10trajrun_baseline
# ./launch.sh /checkpoint/apratik/data_dec/circle apartment_0/instance/circle/no_noise 10 2 10trajrun_circle
# ./launch.sh /checkpoint/apratik/data_dec/straightline apartment_0/instance/straightline/no_noise 10 2 10trajrun_straightline

# ./launch.sh /checkpoint/apratik/data_dec/baseline_noisy2 apartment_0/instance/baseline/noise 10 2 10trajrun_baseline
# ./launch.sh /checkpoint/apratik/data_dec/circle_noisy2 apartment_0/instance/circle/noise 10 2 10trajrun_circle
# ./launch.sh /checkpoint/apratik/data_dec/st_noisy2 apartment_0/instance/straightline/noise 10 2 10trajrun_straightline

# Setting 2
# ./launch.sh /checkpoint/apratik/data_dec/baseline apartment_0/class/baseline/no_noise 10 2 10trajrun_baseline
./launch.sh /checkpoint/apratik/data_dec/combined_examines/circle apartment_0/instance/circle/no_noise 10 2 "10trajrun circle no noise setting 2"
./launch.sh /checkpoint/apratik/data_dec/combined_examines/straightline apartment_0/instance/straightline/no_noise 10 2 "10trajrun straightline no noise setting 2"

# ./launch.sh /checkpoint/apratik/data_dec/combined_examines/baseline_noisy2 apartment_0/class/baseline/noise 10 2 10trajrun_baseline
./launch.sh /checkpoint/apratik/data_dec/combined_examines/circle_noisy2 apartment_0/instance/circle/noise 10 2 "10trajrun circle noise setting 2"
./launch.sh /checkpoint/apratik/data_dec/combined_examines/st_noisy2 apartment_0/instance/straightline/noise 10 2 "10trajrun straightline noise setting 2"