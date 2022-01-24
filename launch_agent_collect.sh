default_ip=$(hostname -I | cut -f1 -d" ")
ip=${LOCOBOT_IP:-$default_ip}
echo "default_ip" $default_ip
export LOCOBOT_IP=$ip
export SAVE_EXPLORATION=True
export DATA_PATH=straightline_test
export HEURISTIC=straightline
export VISUALIZE_EXAMINE=True
export CONTINUOUS_EXPLORE=False
source activate /private/home/apratik/miniconda3/envs/droidlet
python agents/locobot/locobot_agent.py --dev