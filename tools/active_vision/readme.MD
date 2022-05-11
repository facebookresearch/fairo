Instructions for each stage of the pipeline

Clone the droidlet repo to your home directory, so that it is accessible to jobs on the cluster. You'll need to wait for each set of jobs to finish before queueing the next. 
```
git clone git@github.com:facebookresearch/fairo.git
git checkout ap/reex_soumith
cd tools/active_vision
```

1. Find candidates

`./launch_candidate_selection.sh /checkpoint/apratik/data_reexplore/av300_pt2/ av_sm 50 class`

Time estimate - ~10-15 minutes. This just finds respawn location and runs on devlab.

- Job DIR: `/checkpoint/soumith/jobs/reexplore/candidates/09-05-2022/07:53:51`

2. Launch Reexplore

`./launch_reexplore.sh /checkpoint/soumith/jobs/reexplore/av_sm/av300_pt2 class`

Time estimate - ~a few hours, you can run this overnight. It goes and reexplores each object for all the trajectories. Runs on learnfair

- Job DIR: `/checkpoint/soumith/jobs/reexplore/recollect/10-05-2022/11:09:43`

3. Launch Label Prop

`./launch_label_prop.sh /checkpoint/soumith/jobs/reexplore/av_sm/av300_pt2 av_sm class`

Time estimate - ~1-2 hours.

- Job DIR: `/checkpoint/soumith/jobs/reexplore/labelprop/10-05-2022/17:34:34`

4. Launch Training

`./launch_training.sh /checkpoint/soumith/jobs/reexplore/labelprop/av_sm class`

Time estimate - ~7-8 hours.

- Job DIR: `/checkpoint/soumith/jobs/reexplore/training/11-05-2022/09:51:35`
