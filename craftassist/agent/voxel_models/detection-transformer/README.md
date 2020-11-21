End-to-End Object Detection with Transformers (DEV)
========

## Installation

Install PyTorch and torchvision
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Should be good to go.

## Train locally
For debugging purposes:
```
python detection.py --dataset_file="house"
```