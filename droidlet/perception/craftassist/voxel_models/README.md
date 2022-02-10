To get a really good semantic segmentation model, which could then be used as backbone in DETR/MDETR, follow those steps:

# Generate Data #


```
cd droidlet/perception/craftassist/voxel_models/semantic_segmentation
mkdir /vision_data
python generate_data_from_iglu.py --NUM_SCENES 500  --save_data_path /vision_data/training_data.pkl
python generate_data_from_iglu.py --NUM_SCENES 50  --save_data_path /vision_data/validation_data.pkl
```

Make sure you generate both `training_data.pkl` and `validation.pkl` files with exact these two names.


# Train backbone #

```
python train_semantic_segmentation.py --data_dir /vision_data/  --cuda --batchsize 32 --sample_empty_prob 0.0001
```
