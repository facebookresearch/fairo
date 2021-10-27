#!/bin/env bash
echo "Killing navigation, planning, slam, remote and naming processes"

kill_pattern () {
    ps -elf|grep "$1" | grep -v grep | tr -s " " | cut -f 4 -d" " | xargs kill -9 >/dev/null 2>&1 || true 
}

kill_pattern navigation_service.py
kill_pattern planning_service.py
kill_pattern slam_service.py
kill_pattern remote_locobot.py
kill_pattern Pyro4.naming

# sleep 3
