Instructions for each stage of the pipeline

Clone the droidlet repo to your home directory, so that it is accessible to jobs on the cluster. You'll need to wait for each set of jobs to finish before queueing the next. 
```
git clone git@github.com:facebookresearch/fairo.git
git checkout ap/reex_soumith
cd tools/active_vision
```

1. Find candidates

`./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_pt2/ av_sm 50 class`

2. Launch Reexplore

`./launch_reexplore.sh /checkpoint/soumith/jobs/reexplore/av_sm/av300_pt2 class`

3. Launch Label Prop

`./launch_label_prop.sh /checkpoint/soumith/jobs/reexplore/av_sm/av300_pt2 av_sm class`

4. Launch Training

`./launch_training.sh /checkpoint/soumith/jobs/reexplore/labelprop/av_sm class`
