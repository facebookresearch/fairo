# Label propogation

## Label propogation based on robot trajectory
Note: Assumed data storage format
```
├── scene_path
│   ├── apartment_0
│   │   ├── rgb
|   │   │   ├── 00000.jpg
|   │   │   ├── 00001.jpg
            .
            .
│   │   ├── seg
|   │   │   ├── 00000.npy
|   │   │   ├── 00001.npy
            .
            .
│   │   ├── out_dir (will be genrated after running the code)
|   │   │   ├── 00000.npy
|   │   │   ├── 00001.npy
            .
            .
│   │   ├── data.json (robot state information with corresponding image id)
│   │   ├── train_img_id.json (will be genrated after running the code)

    .
    .
│   ├── apartment_1 
    .
    .     
```
To run the label prpogation might need to install
```
python label_propogation_parallel.py --scene_path /checkpoint/dhirajgandhi/active_vision/replica_random_exploration_data --freq 60 --propogation_step 30 --out_dir pred_label_using_traj 
```

## Create COCO dataset 
```
python create_coco_data.py --img_root_path /checkpoint/dhirajgandhi/active_vision/habitat_data_with_seg/rgb --img_annot_root_path /checkpoint/dhirajgandhi/active_vision/habitat_data_with_seg/pred_label --train_json /checkpoint/apratik/ActiveVision/train.json
```

## Some computed COCO dateset are under `json` folder
| file name                                  | test/train | gt/pred | propogation step | skip  | exclude              |
|--------------------------------------------|------------|---------|------------------|-------|----------------------|
| test_gt_e_floor,ceiling,wall.json          | test       | gt      | -                | -     | floor, ceiling, wall |
| train_gt_p_0_s_1_e_floor,ceiling,wall.json | train      | gt      | 0                | 1     | floor, ceiling, wall |
| train_p_0_s_1_e_.json                      | train      | pred    | 0                | 1     | -                    |
| train_p_0_s_1_e_floor,ceiling,wall.json    | train      | pred    | 0                | 1     | floor, ceiling, wall |
| train_p_5_s_1.json                         | train      | pred    | 5                | 1     | -                    |
| train_p_5_s_1_e_floor,ceiling,wall.json    | train      | pred    | 5                | 1     | floor, ceiling, wall |

* propogation = 0 , means only used image index mentioned in provided `train_json` in `create_coco_data.py`

* propogation = `x`, skip = `y` ,  means for every image index `i` mentioned in provided `train_json` in `create_coco_data.py` we included `range(i-x,i+x+1, skip)` indexes while creating annotated json

for how to use them during training refere to [ipython notebook](../notebooks/train_detector.ipynb)
