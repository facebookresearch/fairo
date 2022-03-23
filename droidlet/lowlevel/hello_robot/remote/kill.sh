#!/bin/env bash
echo "Killing hello droidlet"

kill_pattern () {
    ps -exo "pid,args" |grep "$1" | grep "$2" | grep -v grep | tr -s " " | sed "s/^[ \t]*//" | cut -f 1 -d" " | xargs kill -9 >/dev/null 2>&1 || true 
}

kill_pattern python navigation_service.py
kill_pattern python planning_service.py
kill_pattern python slam_service.py
kill_pattern python remote_hello_realsense.py
kill_pattern python remote_hello_robot.py
kill_pattern python remote_hello_saver.py
kill_pattern python Pyro4.naming
