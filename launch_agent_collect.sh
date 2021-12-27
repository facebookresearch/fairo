export LOCOBOT_IP=$1
export SAVE_EXPLORATION=True
export DATA_PATH=straightline_noisy
export HEURISTIC=straightlinet
export VISUALIZE_EXAMINE=True
export CONTINUOUS_EXPLORE=True
source activate /private/home/apratik/miniconda3/envs/droidlet
python agents/locobot/locobot_agent.py --dev