ps -elf|grep _service.py | grep -v grep | tr -s " " | cut -f 4 -d" " | xargs kill -9

sleep 0.5

echo "Killed all"

python slam_service.py &
sleep 1
python planning_service.py &
sleep 1
python navigation_service.py
