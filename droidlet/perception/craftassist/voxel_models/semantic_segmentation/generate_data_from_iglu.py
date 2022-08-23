import numpy as np
from droidlet.lowlevel.minecraft.small_scenes_with_shapes import build_shape_scene, GROUND_DEPTH, SL, H
from droidlet.lowlevel.minecraft.shape_util import SHAPE_NAMES

IDX2NAME = ["nothing"] + SHAPE_NAMES
NAME2IDX = {IDX2NAME[i]: i for i in range(len(IDX2NAME))}


def json_to_segdata(J):
    data = [np.zeros((SL, H, SL), dtype="int32"),
            np.zeros((SL, H, SL), dtype="int32"),
            IDX2NAME]
    for l in J["blocks"]:
        # print(l)
        data[0][l[0], l[1], l[2]] = l[3]
    for t in J["inst_seg_tags"]:
        name_idx = NAME2IDX[t["tags"][0]]
        locs = t["locs"]
        for l in locs:
            data[1][l[0], l[1], l[2]] = name_idx
    return data


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--SL", type=int, default=SL)
    parser.add_argument("--H", type=int, default=H)
    parser.add_argument("--GROUND_DEPTH", type=int, default=GROUND_DEPTH)
    parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
    parser.add_argument("--NUM_SCENES", type=int, default=3)
    parser.add_argument("--fence", action="store_true", default=False)
    parser.add_argument("--cuberite_x_offset", type=int, default=-SL // 2)
    parser.add_argument("--cuberite_y_offset", type=int, default=63 - GROUND_DEPTH)
    parser.add_argument("--cuberite_z_offset", type=int, default=-SL // 2)
    parser.add_argument("--save_data_path", default="")
    args = parser.parse_args()

    data = []
    data_cnt = 0
    while data_cnt < args.NUM_SCENES:
        # try:
        J = build_shape_scene(args)
        data.append(json_to_segdata(J))
        data_cnt += 1
            # print(f"OK, {data_cnt}")
        # except Exception as e:
        #     print(f"DATA ERROR, {e}")
    # for i in range(args.NUM_SCENES):
    #     try:
    #         J = build_shape_scene(args)
    #         data.append(json_to_segdata(J))
    #     except Exception as e:
    #         err_data_cnt += 1
    #         print(f"DATA ERROR, CNT: {err_data_cnt}, e: {e}")
    if args.save_data_path:
        with open(args.save_data_path, "wb") as f:
            pickle.dump(data, f)
