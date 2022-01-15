# Mask Output Viewing and Comparison Tool

This tool allows the simultaneous viewing of the predicted segment mask alongside the ground truth mask.

Before starting, run `npm i` in this folder to install dependencies

To launch the tool, point Servez (https://greggman.github.io/servez/) or your preferred node server at this folder.  If using Servez, it should load index.html automatically, otherwise select it to load the tool.

You will be prompted to enter paths to the model output scene and the ground truth scene **json** files, respectively.  These can be local paths relative to this directory or a URL.

The data format is expected to match the output of small_scenes_with_shapes.py in the same directory as this folder