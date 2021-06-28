#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

xacro xacro/iiwa7.urdf.xacro -o urdf/iiwa7.urdf
xacro xacro/iiwa7.urdf.xacro gripper:=robotiq_85 -o urdf/iiwa7_robotiq-2f.urdf
xacro xacro/iiwa7.urdf.xacro transparent:=true -o urdf/iiwa7_transparent.urdf
xacro xacro/iiwa7.urdf.xacro force_torque_sensor:=true -o urdf/iiwa7_ft.urdf
xacro xacro/iiwa7.urdf.xacro end_effector:=ee_peg force_torque_sensor:=true -o urdf/iiwa7_ft_peg.urdf
