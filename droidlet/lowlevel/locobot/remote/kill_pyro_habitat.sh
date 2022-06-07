#!/usr/bin/env bash
echo "Killing navigation, planning, slam, remote and naming processes"

kill_pattern () {
    ps -exo "pid,args" |grep "$1" | grep "$2" | grep -v grep | tr -s " " | sed "s/^[ \t]*//" | cut -f 1 -d" " | xargs kill -9 >/dev/null 2>&1 || true 
}

kill_pattern python navigation_service.py
kill_pattern python planning_service.py
kill_pattern python slam_service.py
kill_pattern python remote_locobot.py
kill_pattern python Pyro4.naming

# sleep 3
