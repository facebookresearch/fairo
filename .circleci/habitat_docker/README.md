
1. Build main docker image

```
cd .circleci/habitat_docker
docker run --rm -v $(pwd):/remote -w /remote theh1ghwayman/locobot-assistant:8.0 cp -r /Replica-Dataset /remote/

awk '{print}' ../../conda.txt ../../agents/locobot/conda.txt >conda.txt
awk '{print}' ../../requirements.txt ../../agents/locobot/requirements.txt | grep -v requirements.txt >requirements.txt

docker build -t theh1ghwayman/locobot-assistant:10.0 .
```