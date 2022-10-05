# import numpy as np
# import json
# import pickle
# import torch

# from vision import SemSegWrapper
# from data_loaders import SemSegData

# from droidlet.lowlevel.minecraft.small_scenes_with_shapes import build_shape_scene, GROUND_DEPTH, SL, H

# from droidlet.lowlevel.minecraft.shape_util import SHAPE_NAMES

# import argparse


# SL = 17 
# H = 13

# # model_path = "/checkpoint/yuxuans/jobs/hitl_vision/data_shapes_6kind_6000_nfbid_0_nepochs_500_lr_0.001_batchsz_256_sampleEmptyProb_0.05_hiddenDim_128_noTargetProb_0.3_probThreshold_0.3_queryEmbed_bert_runName_SWEEP2/model.pt"
# model_path = "/checkpoint/yuxuans/models/hitl_vision/v1.pt"
# data_path = "/checkpoint/yuxuans/datasets/inst_seg/D5_TEST/validation_data.pkl"
# if __name__ == "__main__":
#     fake_data = np.zeros((SL, H, SL), dtype="int64")
#     json_data = {
#         "inst_seg_tags": [],
#         "blocks": []
#     }
#     cube = []
#     bar = []
    
#     # a cube
#     (x, y, z) = (0, 0, 0)
#     for ix in range(4):
#         for iy in range(4):
#             for iz in range(4):
#                 fake_data[x + ix, y + iy, z + iz] = 50
#                 cube.append((x + ix, y + iy, z + iz))
#                 json_data["blocks"].append((x+ix, y+iy, z+iz, 50))


#     # a bar
#     (x, y, z) = (7, 0, 7)
#     for ix in range(1):
#         for iy in range(10):
#             for iz in range(1):
#                 fake_data[x + ix, y + iy, z + iz] = 51
#                 bar.append((x + ix, y + iy, z + iz))
#                 json_data["blocks"].append((x+ix, y+iy, z+iz, 51))

#     json_data["inst_seg_tags"].append({"cube": cube})
#     json_data["inst_seg_tags"].append({"bar": bar})

#     # a sphere
#     # N = 6
#     # c = N / 2 - 1 / 2
#     # CNT = 0
#     # for r in range(N):
#     #     for s in range(N):
#     #         for t in range(N):
#     #             w = ((r - c) ** 2 + (s - c) ** 2 + (t - c) ** 2) ** 0.5
#     #             if w < N / 2:
#     #                 fake_data[r, s, t] = 50
#     #                 # S.append(((r, s, t), bid))
#     #                 CNT += 1
#     # print(f"CNT: {CNT}")
    
#     for ix in range(17):
#         for iy in range(10, 12):
#             for iz in range(17):
#                 fake_data[ix, iy, iz] = 46
#                 json_data["blocks"].append((ix, iy, iz, 46))

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--SL", type=int, default=SL)
#     parser.add_argument("--H", type=int, default=H)
#     parser.add_argument("--GROUND_DEPTH", type=int, default=GROUND_DEPTH)
#     parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
#     parser.add_argument("--NUM_SCENES", type=int, default=3)
#     parser.add_argument("--fence", action="store_true", default=False)
#     parser.add_argument("--cuberite_x_offset", type=int, default=-SL // 2)
#     parser.add_argument("--cuberite_y_offset", type=int, default=63 - GROUND_DEPTH)
#     parser.add_argument("--cuberite_z_offset", type=int, default=-SL // 2)
#     parser.add_argument("--save_data_path", default="")
#     args = parser.parse_args()

#     text = "where is the cube"

#     # scene = build_shape_scene(args)

#     # print(scene)

#     model = SemSegWrapper(model=model_path, cuda=True)

#     data = pickle.load(open(data_path, "rb"))
#     # print(data[0])
#     idx = 0

#     in_data = []
#     out_data = []
#     for d in data[1:6]:
#         print(f"Data point: {idx}:")
#         blocks = d[0]
#         text = d[3]
#         gt = d[4]
#         text_embed = d[5]
#         pred = model.perceive(blocks, text_embed=text_embed, offset=None)
#         # print(f"gt: {torch.from_numpy(gt).nonzero()}")
#         # print(f"pred:{pred}")




#         json_inp_data = {
#             "blocks":[],
#             "inst_seg_tags": []
#         }

#         for ix in range(blocks.shape[0]):
#             for iy in range(blocks.shape[1]):
#                 for iz in range(blocks.shape[2]):
#                     json_inp_data["blocks"].append((ix, iy, iz, int(blocks[ix, iy, iz])))
#         for i in range(d[4].shape[3]):
#             locs = []
#             (xs, ys, zs) = np.nonzero(d[4][:, :, :, i])
#             for j in range(len(xs)):
#                 locs.append((int(xs[j]), int(ys[j]), int(zs[j])))
#             payload = {"tags": [text[i]], "locs": locs}
#             json_inp_data["inst_seg_tags"].append(payload)

#         # print(json_inp_data)

#         json_out_data = {
#             "blocks":[],
#             "inst_seg_tags": []
#         }
#         for ix in range(blocks.shape[0]):
#             for iy in range(blocks.shape[1]):
#                 for iz in range(blocks.shape[2]):
#                     json_out_data["blocks"].append((ix, iy, iz, int(blocks[ix, iy, iz])))

#         for i, p in enumerate(pred):
#             payload2 = {"tags": [text[i]], "locs": p}
#             json_out_data["inst_seg_tags"].append(payload2)

#         in_data.append(json_inp_data)
#         out_data.append(json_out_data)
#         for i in range(len(pred)):
#             locs = json_inp_data["inst_seg_tags"][i]["locs"]
#             p = pred[i]
#         print(f"GT: \n{locs}\n PRED: \n{p}\n")
#         idx += 1
#     # print(in_data)
#     with open("/checkpoint/yuxuans/data/demo_in_2.json", "w") as f:
#         json.dump(in_data, f)
#     with open("/checkpoint/yuxuans/data/demo_out_2.json", "w") as f:
#         json.dump(out_data, f)
#     # pred = model.perceive(fake_data, text_span=text, offset=None)

#     # print(pred)

#     # with open("/checkpoint/yuxuans/data/demo.json", "w") as f:
#     #     json.dump(json_data, f)


#################################################################################
    
import numpy as np
import json
import pickle
import torch

from vision import SemSegWrapper
from data_loaders import SemSegData

from droidlet.lowlevel.minecraft.small_scenes_with_shapes import build_shape_scene, GROUND_DEPTH, SL, H

from droidlet.lowlevel.minecraft.shape_util import SHAPE_NAMES

import argparse


SL = 17 
H = 13

# model_path = "/checkpoint/yuxuans/jobs/hitl_vision/data_shapes_6kind_6000_nfbid_0_nepochs_500_lr_0.001_batchsz_256_sampleEmptyProb_0.05_hiddenDim_128_noTargetProb_0.3_probThreshold_0.3_queryEmbed_bert_runName_SWEEP2/model.pt"
model_path = "/checkpoint/yuxuans/models/hitl_vision/v5.pt"
data_path = "/checkpoint/yuxuans/datasets/inst_seg/D11_test/training_data.pkl"
if __name__ == "__main__":
    fake_data = np.zeros((SL, H, SL), dtype="int64")
    json_data = {
        "inst_seg_tags": [],
        "blocks": []
    }
    cube = []
    bar = []
    
    # a cube
    (x, y, z) = (0, 0, 0)
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                fake_data[x + ix, y + iy, z + iz] = 50
                cube.append((x + ix, y + iy, z + iz))
                json_data["blocks"].append((x+ix, y+iy, z+iz, 50))


    # a bar
    (x, y, z) = (7, 0, 7)
    for ix in range(1):
        for iy in range(10):
            for iz in range(1):
                fake_data[x + ix, y + iy, z + iz] = 51
                bar.append((x + ix, y + iy, z + iz))
                json_data["blocks"].append((x+ix, y+iy, z+iz, 51))

    json_data["inst_seg_tags"].append({"cube": cube})
    json_data["inst_seg_tags"].append({"bar": bar})

    # a sphere
    # N = 6
    # c = N / 2 - 1 / 2
    # CNT = 0
    # for r in range(N):
    #     for s in range(N):
    #         for t in range(N):
    #             w = ((r - c) ** 2 + (s - c) ** 2 + (t - c) ** 2) ** 0.5
    #             if w < N / 2:
    #                 fake_data[r, s, t] = 50
    #                 # S.append(((r, s, t), bid))
    #                 CNT += 1
    # print(f"CNT: {CNT}")
    
    for ix in range(17):
        for iy in range(10, 12):
            for iz in range(17):
                fake_data[ix, iy, iz] = 46
                json_data["blocks"].append((ix, iy, iz, 46))

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


    # scene = build_shape_scene(args)

    # print(scene)

    model = SemSegWrapper(model=model_path, cuda=True)

    data = pickle.load(open(data_path, "rb"))
    # print(data[0])
    idx = 0

    in_data = []
    out_data = []
    for d in data[1:]:
        print(f"Data point: {idx}:")
        blocks = d[0]
        text = d[3]
        gt = d[4]
        text_embed = d[5]
        pred = model.perceive(blocks, text_embed=text_embed, offset=None)
        # print(f"gt: {torch.from_numpy(gt).nonzero()}")
        # print(f"pred:{pred}")

        for i in range(d[4].shape[3]):
            json_inp_data = {
                "blocks":[],
                "inst_seg_tags": []
            }

            for ix in range(blocks.shape[0]):
                for iy in range(blocks.shape[1]):
                    for iz in range(blocks.shape[2]):
                        json_inp_data["blocks"].append((ix, iy, iz, int(blocks[ix, iy, iz])))

            locs = []
            print(f"Tag: {text[i]}, gt: \n{d[4][:, :, :, i]}")
            (xs, ys, zs) = np.nonzero(d[4][:, :, :, i])
            for j in range(len(xs)):
                locs.append((int(xs[j]), int(ys[j]), int(zs[j])))
            payload = {"tags": [text[i]], "locs": locs}
            json_inp_data["inst_seg_tags"].append(payload)


            json_out_data = {
                "blocks":[],
                "inst_seg_tags": []
            }
            for ix in range(blocks.shape[0]):
                for iy in range(blocks.shape[1]):
                    for iz in range(blocks.shape[2]):
                        json_out_data["blocks"].append((ix, iy, iz, int(blocks[ix, iy, iz])))
            print(f"gt len: {d[4].shape[3]}, pred len: {len(pred)}")
            p = pred[i]
            payload2 = {"tags": [text[i]], "locs": p}
            json_out_data["inst_seg_tags"].append(payload2)

            in_data.append(json_inp_data)
            out_data.append(json_out_data)
            print(f"GT: \n{locs}\n PRED: \n{p}\n")

    # print(in_data)
    with open("/checkpoint/yuxuans/data/test_in_11.json", "w") as f:
        json.dump(in_data, f)
    with open("/checkpoint/yuxuans/data/test_out_11.json", "w") as f:
        json.dump(out_data, f)
    # pred = model.perceive(fake_data, text_span=text, offset=None)

    # print(pred)

    # with open("/checkpoint/yuxuans/data/demo.json", "w") as f:
    #     json.dump(json_data, f)



    
