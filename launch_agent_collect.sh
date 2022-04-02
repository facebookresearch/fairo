default_ip=$(hostname -I | cut -f1 -d" ")
ip=${ROBOT_IP:-$default_ip}
echo "default_ip" $default_ip
export ROBOT_IP=$ip
export SAVE_EXPLORATION=True
export DATA_PATH=straightline_test
export HEURISTIC=straightline
export VISUALIZE_EXAMINE=True
export CONTINUOUS_EXPLORE=False
source activate /private/home/apratik/miniconda3/envs/droidlet
python agents/robot/robot_agent.py --dev
