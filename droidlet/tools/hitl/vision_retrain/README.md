## Vision Labeling and Annotation Pipeline ##

This folder contains the scripts for improving the voxel-based perception model.  There are two versions of the pipeline that use these files:
 - The main interaction pipeline will generate a vision annotation job when a vision error occurs during interaction with a crowdworker
 - There is a vision labeling data generator that will generate scenes and labels of the objects within those scenes, and then send them to be annotated via the same vision annotation job as above.

Vision annotation jobs are generated via either a VisionListener (in the case of the interaction job) or VisionLabelingListener (in the case of a labeling job).


### Launching a vision labeling job ###

To launch a batch of HITs that will label objects present in IGLU voxel scenes, run `vision_labeling_jobs.py` with the desired arguments for what the scenes should look like, how many HITs to generate, and what the timeouts should be.

### Filtering uannotated scenes from annotated scene list ###

Yield on Turk jobs is never 100%, so not all scenes that are requested will be labeled, and not all scenes that are labeled will be annotated.  The former is fine, nothing is lost.  In the second case, we want to 1) clean unannotated scenes from the annotated scene list so they are not used for training and 2) recover those scenes and send them off for annotation in a different job.  To do this, run `recover_unannotated_scenes.py` with the --batch_id.

### Combining and annotating labeled scene lists ###

If you ever have lists of 1 or more labeled scene lists and want to combine them and run them as a new annotation job, you may do so using `annotate_labeled_scenes.py`.  The script will look for scene_lists with the suffix "_unannotated.json" in the provided batch_id directories, combine them, and launch an annotation job with them.