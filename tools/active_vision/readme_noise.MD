Instructions for each stage of the pipeline

Clone the droidlet repo to your home directory, so that it is accessible to jobs on the cluster. You'll need to wait for each set of jobs to finish before queueing the next. 
```
git clone git@github.com:facebookresearch/fairo.git
git checkout ap/reex_soumith
cd tools/active_vision
```

1. Find candidates

`./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_noise_pt2 av_sm_noise 60 class`

Time estimate - ~10-15 minutes. This just finds respawn location and runs on devlab.

- Job DIR: ``

2. Launch Reexplore

`./launch_reexplore.sh /checkpoint/soumith/jobs/reexplore/av_sm_noise/av300_noise_pt2 class --noise`

Time estimate - ~a few hours, you can run this overnight. It goes and reexplores each object for all the trajectories. Runs on learnfair

- Job DIR: ``

3. Launch Label Prop

`./launch_label_prop.sh /checkpoint/soumith/jobs/reexplore/av_sm_noise/av300_noise_pt2 av_sm_noise class`

Time estimate - ~1-2 hours.

- Job DIR: ``

4. Launch Training

`./launch_training.sh /checkpoint/soumith/jobs/reexplore/labelprop/av_sm_noise class`

Time estimate - ~7-8 hours.

- Job DIR: ``
