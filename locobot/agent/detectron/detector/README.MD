Prepare dataset
```
curl https://locobot-bucket.s3-us-west-2.amazonaws.com/astorplace.tar.gz | tar -xzv -C datasets/
```

Run training
```
train_net.py --data-dir /scratch/apratik/minecraft/python/locobot/datasets/astorplace --dataset-name astorplace --annotations-json /scratch/apratik/minecraft/python/locobot/detectron/detector/other_annotations9.json --config-file /scratch/apratik/minecraft/python/locobot/detectron/detector/configs/mask_rcnn_R_101_FPN_1x.yaml
```