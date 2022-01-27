# Mask Output Viewing and Comparison Tool

This tool allows the simultaneous viewing of the predicted segment mask alongside the ground truth mask.

To launch the tool, point Servez (https://greggman.github.io/servez/) or your preferred node server at this folder.  If using Servez, it should load index.html automatically, otherwise select it to load the tool.

You will be prompted to enter paths to the model output scene and the ground truth scene **json** files, respectively.  Here is an example URL for testing the tool: https://craftassist.s3.us-west-2.amazonaws.com/pubr/scenes/scene20220113202201.json

The data format is expected to match the output of `small_scenes_with_shapes.py`, also below:

```js
[{
    "avatarInfo": {"pos": (x,y,z), "look": (yaw, pitch)}
    "agentInfo": {"pos": (x,y,z), "look": (yaw, pitch)}
    "inst_seg_tags": [{"tags": ["shape"], "locs": [(x,y,z) ... (x,y,z)]} ...]
    "blocks": [(x,y,z,bid) ... (x,y,z,bid)]
}]
```