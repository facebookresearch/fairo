The notebooks here guide you through each step of the mturk data collection process for the locobot detector, which is trained to output **instance segmentation, class, bounding boxe labels and object properties**.

We collect mturk annotations for the locobot detector using a 2-stage process
* Stage 1: Collect mask annotations without any object or property labels 
![image](https://user-images.githubusercontent.com/57542204/110144116-f812f580-7da5-11eb-8e9d-c0d3551f71f9.png)

* Stage 2: For each mask annotation from above, collect class labels and properties.
![image](https://user-images.githubusercontent.com/57542204/110144166-07923e80-7da6-11eb-9e6a-c267ff4474f3.png)

Here's a quick summary of how each notebook helps in this 2-step process.
* `0_prep_turk_csv.ipynb` - prepares a batch of input files for the instance segmentation task (stage 1 prep)
* `1_visualize_masks.ipynb` - visualizes masks from the instance segmentation task (stage 1 results)
* `2_create_properties_turk_job.ipynb` - converts masks annotations to bounding boxes for collecting class and property labels (stage 2 prep)
* `3_parse_labels_create_coco.ipynb` - converts mask annotation and text labels into a COCO annotations file (stage 2 results)
