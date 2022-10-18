
1. Build Craftassist base docker image

```
cd tools/docker
awk '{print}' ../../conda.txt ../../agents/craftassist/conda.txt >conda.txt
awk '{print}' ../../requirements.txt ../../agents/craftassist/requirements.txt | grep -v requirements.txt >requirements.txt

docker build -t shaomai/craftassist:1.0 -f ./craftassist_docker/Dockerfile .
```