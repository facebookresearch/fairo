Each notebook here corresponds to a step in the data pipeline
* 0_prep_turk_csv.ipynb - prepares a batch of input files for the instance segmentation task
* 1_visualize_masks.ipynb - visualizes masks from the instance segmentation task
* 2_create_properties_turk_job.ipynb - converts masks annotations to bounding boxes for collecting class and property labels.
* 3_parse_labels_create_coco.ipynb - converts mask annotation and text labels into a COCO annotations file