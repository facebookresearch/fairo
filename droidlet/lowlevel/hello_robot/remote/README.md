# Getting the Remote service up and running

On the Hello Robot Stretch, run the following from the directory where the readme exists

## 1. Install venv and requirements

```bash
python3 -m venv droidlet
. droidlet/bin/activate
pip install -r requirements.txt
```

## 2. Install droidlet in develop mode

```bash
pushd ../../../../
python setup.py develop
popd
```

## 3. Start the services

```bash
./launch.sh
```