We translate between two separate coordinate systems. The locobot co-ordinate axes is the standard on by [PyRobot](https://github.com/facebookresearch/pyrobot/blob/master/docs/website/docs/ex_navigation.md#frames-of-reference), and the droidlet coordinate axes is the standard we follow.

<p align="center">
   <img src="https://locobot-bucket.s3-us-west-2.amazonaws.com/documentation/coordinate_systems.png", width="800" height="400" />
</p>

The conventions for yaw and pitch are the same across both - `yaw` is assumed to be in the range `[-pi, pi]`, and increasing `yaw` moves *counterclockwise*. `pitch` is assumed to be in the range `[-pi/2, pi/2]`.  `pi/2` is down, `-pi/2` is up.
  
This means we do the following transformations -
* the point cloud `(x_l, y_l, z_l)` is fetched in locobot world coords, but [saved in droidlet world coords](https://github.com/fairinternal/minecraft/blob/73b6769328ff2d53c41b2d66683ea7547cb176e8/locobot/agent/perception/perception_objects.py#L88). 
`(x_d = -y_l, y_d = z_l, z_d = x_l)`
* the state of the base of the robot `(x_l, y_l, yaw_l)` is fetched in locobot world coords and [transformed to droidlet world coords](https://github.com/fairinternal/minecraft/blob/73b6769328ff2d53c41b2d66683ea7547cb176e8/locobot/agent/locobot_mover.py#L134).
    `(x_d = -y_l, y_d = z_l, yaw_d = yaw_l)`
