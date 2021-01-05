# Geoscorer

## Introduction

Geoscore is a model which takes a `context` (currently a 32x32x32 voxel) and a `segment` to place in that `context` (currently an 8x8x8 voxel). It produces the target position to place the `segment` within the `context`.

The goal is to be able to resolve requests like "build a bench next to the tree" more naturally than simple heuristics would allow.

Currently the training setup (and therefore the usage in the bot) allows for only directional commands ("ABOVE"/"LEFT"/etc) for building an schematic (the `segment`) in a space (the `context`), but there are many interesting extensions to this setup.

## Usage

You can run the Minecraft agent using a geoscorer model as the default placement method for directional `build` commands by passing the model path as a flag:

```
python ./agent/craftassist_agent.py --geoscorer_model_path <model_path>
```

The `Geoscorer` object defined in `../geoscorer.py` loads the model and provides methods to determine whether the call conditions are supported (`use(...)`) and to use the model to output positions (`produce_object_positions(...)`).  It handles all of the conversion between Minecraft world format and the geoscorer utils format.

The `Geoscorer` object uses a wrapper around the training/eval code, `ContextSegmentMergerWrapper`, which is defined in `geoscorer_wrapper.py`.  Note that currently the parameters of the model setup are all hardcoded in this file, so the model file selected by the `--geoscorer_model_path` flag must match.

## Geoscorer Training

### Training Loop
The training loop lives in `train_spatial_emb.py` and can be run as follows:

```
python train_spatial_emb.py --checkpoint checkpoint_path.bin --useid --nepoch 5000 \
    --seg_direction_net --seg_use_direction --cont_use_direction --cont_use_xyz_from_viewer_look \
    --dataset_config dataset_configs/all_good_split.json
```

Some notes:
- If you omit the `--checkpoint` then no checkpoint will be saved
- `--useid` determines whether the blockids are passed to the model or just the binary 0/1 of block existence
- The middle set of flags (`--seg_direction_net`, ...) control model configuration & features used
- You can specify the datasets to train on either with a dataset config (as shown in the example) or with other flags (see dataset section)

There is also a `--visualize_epochs` option which uses visdom to display a few examples each epoch of training (good for debugging but includes a sleep and slows down training dramatically).

Finally, alot of the utils used in the training loop are defined in `training_utils.py` including checkpointing, arg handling, dataloading, etc.

### Model Setup

The modules of the model are defined in `models.py`.

The model works by producing an embedding for each of the locations in the context, and another embedding for the segment, and then taking the dot product of the segment against each of the locations in the context to choose the best position to return.

A batch element, passed into each of these modules, is a dictionary containing:
- **`context`**: NxNxN voxel of block ids (if usedid is set, or 0/1 if it is not)
- **`segment`**: MxMxM voxel of block ids (same about useid)
- **`viewer_pos`**: The position of the viewer (used to calculate the absolute directional meaning of the relative directional commands, "LEFT" for example)
- **`viewer_look`**: The point the viewer is looking, currently assumed to be the center of any main context object, like the "tree" in "build a bench to the left of the tree")
- **`direction`**: An embedding describing the directional command, currently it is a 5 element vector where the first 3 provide the dimension and the last 2 provide the direction in that dimension.

In more detail, there are 4 parts of the model (where the separation between these modules is mainly a historical artifact):

**ContextEmbeddingNet**
- Takes the batch elements and produces a batch of DxNxNxN where D is the size of the embedding for each of the points in the context voxel.
- Opts can be set to include additional features in the context for each block beyond just the blockid
  - `--cont_use_direction` includes the directional embedding
  - `--cont_use_xyz_from_viewer_look` includes the relative vector from the viewer_look

**SegmentEmbeddingNet**
- Takes the batch elements and produces a batch of Dx1x1x1 where D matches the context embedding size
- No additional features are used beyond the MxMxM blockids of the segment

**SegmentDirectionEmbeddingNet**
- An optional part of the model which combines the seg embeddings from the `SegmentEmbeddingNet` with the `direction` embedding
- To use, specify `--seg_direction_net` and `--seg_use_direction` (TODO: fix this duplication in the code)

**ContextSegmentScoringModule**
- Takes a dictionary with the context and segment embeddings, `c_embeds` and `s_embeds`, and does the dot product to produce the final scores.

### Datasets

The different datasets are designed to be consumed through the CombinedData dataset (found in `combined_dataset.py`) which takes either a dataset config, so that a specific dataset can be used multiple times with different parameters, of dataset ratios dict (eg. `{"shape_pair": 0.5, "shape_piece": 0.4"}`).

The currently implemented datsets are:
- **ShapePairData**: takes two shapes, one context and one segment, and orients them relative to each other based on given direction. Parameters control the max distance between the two shapes, the size of the shapes, the type of the shapes, and ground below the shapes.
- **ShapePieceData**: takes a single shape and removes a rectangular subsection as the segment, calculating the direction based on the removed segment position.  Parameters control the ground below the shape.
- **InstanceSegData**: takes the segmented house data and chooses one segement to be the segment (using the rest of the segments as the context).  Parameters control how many of the rest of the segments are used for the context and the ground below the shape.

Note that a good configuration for simple test training is to use ShapePairData with `max_shift` of 0, `fixed_size` of 3 and `shape_type` of "cube" which will train a model to attach two cubes together according to the sampled direction (easy to visually debug).

There are many util fxns used in the datasets, they are split into `directional_utils.py`, which contains utils related to the direction vectors and calculating directions, and `spatial_utils.py` which contains all non-directional spatial utils.

### Evaluation

`visualization_utils.py` contains a few visualization classes that handle standard debugging visualization setups.

`eval_model.py` is a script which will take a model and optionally run visualizations on a given dataset and also collects some metrics:
- `overlap_count`: number of non-empty blocks that overlap between the segment and the context (should be low or zero)
- `out_of_bounds_count`: number of segment blocks that are outside of the context voxel (should be low or zero)
- `min_dist_above_{}`: number of datapoints where the segment is farther than some threshold from any block in the context
- `dir_wrong_count`: number of datapoints where the model places the segment in the wrong direction (should be zero)
- `dist_from_target`: distance of the segment location from the target location (should be low, but some datasets aren't deterministic so it shouldn't necessarily be zero)
