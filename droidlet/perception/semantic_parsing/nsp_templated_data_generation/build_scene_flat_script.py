"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
if __name__ == "__main__":
    import argparse
    import pickle
    import os
    from tqdm import tqdm
    from build_scene import *
    from block_data import COLOR_BID_MAP

    BLOCK_DATA = pickle.load(
        open("/private/home/aszlam/minecraft_specs/block_images/block_data", "rb")
    )

    allowed_blocktypes = []
    count = 0
    for c, l in COLOR_BID_MAP.items():
        for idm in l:
            allowed_blocktypes.append(BLOCK_DATA["bid_to_name"][idm])
            count += 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="/checkpoint/aszlam/minecraft/inverse_model/flat_ads/")
    parser.add_argument("--N", type=int, default=10000000)
    #    parser.add_argument("--num_per_chunk", type=int, default=10000000)
    args = parser.parse_args()

    template_attributes = {"count": range(1, 5)}
    template_attributes["step"] = range(1, 10)
    template_attributes["non_shape_names"] = ["triangle", "circle", "disk", "rectangle"]
    template_attributes["mob_names"] = ["pig", "sheep", "cow", "chicken"]
    template_attributes["allowed_blocktypes"] = allowed_blocktypes
    template_attributes["distribution"] = {
        "MOVE": 1.0,
        "BUILD": 1.0,
        "DESTROY": 1.0,
        "DIG": 0.8,
        "COPY": 0.8,
        "FILL": 0.8,
        "SPAWN": 0.1,
        "DANCE": 0.8,
    }

    scenes = []
    for i in tqdm(range(args.N)):
        S = build_scene(template_attributes, sl=16, flat=True)
        scenes.append(S)
    f = open(os.path.join(args.target, "flat_scenes_dump.pk"), "wb")
    pickle.dump(scenes, f)
    f.close()
