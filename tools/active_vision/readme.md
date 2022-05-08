Instructions for each stage of the pipeline

Clone the droidlet repo to your home directory, so that it is accessible to jobs on the cluster. You'll need to wait for each set of jobs to finish before queueing the next. 
```
git clone git@github.com:facebookresearch/fairo.git
git checkout ap/reex_arthur
cd tools/active_vision
```

1. Find candidates

`./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_pt2/ av_sm_pfix 50 instance`

Time estimate - ~10-15 minutes. This just finds respawn location and runs on devlab. 

2. Launch Reexplore

`./launch_reexplore.sh /checkpoint/aszlam/jobs/reexplore/av_sm_pfix/av300_pt2 instance`

Time estimate - ~a few hours, you can run this overnight. It goes and reexplores each object for all the trajectories. Run on `learnfair`

3. Launch Label Prop

`./launch_label_prop.sh /checkpoint/aszlam/jobs/reexplore/av_sm_pfix/av300_pt2 av_sm_pfix instance`

Time estimate - ~1-2 hours.

4. Launch Training

`./launch_training.sh /checkpoint/aszlam/jobs/reexplore/labelprop/av_sm_pfix instance`

Time estimate - ~7-8 hours. 
