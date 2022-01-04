export HEURISTIC=$1 #default straightline
export LOCOBOT_IP=$2
# export DB_FILE=locodb.db
source activate /private/home/apratik/.conda/envs/denv3
python agents/locobot/locobot_agent.py --dev