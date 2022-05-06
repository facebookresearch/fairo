## Vision Labeling and Annotation Pipeline ##

This folder contains the scripts for improving the voxel-based perception model.  There are two versions of the pipeline that use these files:
 - The main interaction pipeline will generate a vision annotation job when a vision error occurs during interaction with a crowdworker
 - There is a vision labeling data generator that will generate scenes and labels of the objects within those scenes, and then send them to be annotated via the same vision annotation job as above.

Vision annotation jobs are generated via either a VisionListener (in the case of the interaction job) or VisionLabelingListener (in the case of a labeling job).


### Launching a vision labeling job ###

To launch a batch of HITs that will label objects present in IGLU voxel scenes, run `vision_labeling_jobs.py` with the desired arguments for what the scenes should look like, how many HITs to generate, and what the timeouts should be.