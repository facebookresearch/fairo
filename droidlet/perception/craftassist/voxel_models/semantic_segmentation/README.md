# Training Semantic Segmentation Models

## Download datasets from S3

```
curl https://craftassist.s3-us-west-2.amazonaws.com/pubr/instance_segmentation_data.tar.gz -o datasets.tar.gz
tar -xvzf datasets.tar.gz -C .
```

## Train the model
```
python train_semantic_segmentation.py --data_dir="instance_segmentation_data/"
```