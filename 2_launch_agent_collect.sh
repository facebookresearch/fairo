export HEURISTIC=default #default straightline
export LOCOBOT_IP=172.17.0.2
source activate ~/.conda/envs/locobot_env
python agents/locobot/locobot_agent.py --dev