ps -elf|grep slam_service.py | grep -v grep | tr -s " " | cut -f 4 -d" " | xargs kill -9 >/dev/null 2>&1
ps -elf|grep planning_service.py | grep -v grep | tr -s " " | cut -f 4 -d" " | xargs kill -9 >/dev/null 2>&1
ps -elf|grep navigation_service.py | grep -v grep | tr -s " " | cut -f 4 -d" " | xargs kill -9 >/dev/null 2>&1


sleep 0.5

echo "Killed all"

python slam_service.py &
sleep 3
python planning_service.py &
sleep 3
python navigation_service.py
