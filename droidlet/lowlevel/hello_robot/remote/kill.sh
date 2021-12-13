#!/bin/env bash
echo "Killing hello droidlet"

kill_pattern () {
    ps -elf|grep "$1" | grep "$2" | grep -v grep | tr -s " " | cut -f 4 -d" " | xargs kill -9 >/dev/null 2>&1 || true
}

kill_pattern python remote_hello_realsense.py
kill_pattern python remote_hello_robot.py
kill_pattern python remote_hello_saver.py
kill_pattern python Pyro4.naming
