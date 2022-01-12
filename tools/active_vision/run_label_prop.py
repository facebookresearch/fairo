import argparse
import submitit
import glob
from pathlib import Path
import os
import subprocess
import json
import logging
from shutil import copyfile, rmtree
import ray
import numpy as np
import cv2
from droidlet.perception.robot import LabelPropagate
from PIL import Image
import matplotlib.pyplot as plt

set_keys = {
    'e1r1r2': ['e1', 'r1', 'r2'],
    'e1s1r2': ['e1', 's1', 'r2'],
    'e1c1sr2': ['e1', 'c1s', 'r2'],
    'e1c1lr2': ['e1', 'c1l', 'r2'],
    'e1s1c1s': ['e1', 's1', 'c1s'],
    'e1s1c1l': ['e1', 's1', 'c1l'],
}

d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
) 

def prop_and_combine(pth, out_name, f_name, prop_length):
    # f takes values output by reexplore - e1, s1, ... etc
    # There will be multiple folders named f however - as many as there are spawn locations / objects 
    # create a foldernamed px/out_name
    out_dir = os.path.join(pth, f'p{prop_length}', out_name)
    if os.path.isdir(out_dir):
        rmtree(out_dir)

    folders = Path(pth).rglob(f_name)
    print(f'pth {pth} out_dir {out_dir} fname {f_name}')
    for x in folders:
        print(x)
        acopydir(os.path.join(x, 'rgb'), os.path.join(out_dir, 'rgb'), '.jpg') # only want to copy the propagated files!
        # do label prop of length p on the GT frame (which is the first frame)
        # create folder named px
        # copy rgb over to px/rgb
        # do label prop into px/seg
        # out_dir = os.path.join(pth)
        # pass

def save_propagated_visual(semantic1, semantic2, save_dir, out_indx):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    arr = []
    for semantic_obs in [semantic1, semantic2]:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)

    titles = ['gt', 'propagated']
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 2, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.savefig("{}/{:05d}.jpg".format(save_dir, out_indx))

def calculate_accuracy(act, pred):
    assert act.shape == pred.shape    
    correct = np.sum(act[pred != 0] == pred[pred != 0])
    total = np.sum(pred != 0)
    return correct/total

# @ray.remote
def propogate_label(
    root_path: str,
    src_img_indx: int,
    propogation_step: int,
    base_pose_data: np.ndarray,
    out_dir: str,
    frame_range_begin: int
):
    """Take the label for src_img_indx and propogate it to [src_img_indx - propogation_step, src_img_indx + propogation_step]
    Args:
        root_path (str): root path where images are stored
        src_img_indx (int): source image index
        src_label (np.ndarray): array with labeled images are stored (hwc format)
        propogation_step (int): number of steps to progate the label
        base_pose_data(np.ndarray): (x,y,theta)
        out_dir (str): path to store labeled propogation image
        frame_range_begin (int): filename indx to begin dumping out files from
    """

     
    print(f" root {root_path}, out {out_dir}, p {propogation_step}")

    ### load the inputs ###
    # load robot trajecotry data which has pose information coreesponding to each img observation taken
    with open(os.path.join(root_path, "data.json"), "r") as f:
        base_pose_data = json.load(f)
    # load img
    try:
        src_img = cv2.imread(os.path.join(root_path, "rgb/{:05d}.jpg".format(src_img_indx)))
        # load depth in mm
        src_depth = np.load(os.path.join(root_path, "depth/{:05d}.npy".format(src_img_indx)))
        # load robot pose for img index
        src_pose = base_pose_data["{}".format(src_img_indx)]
        src_label = np.load(os.path.join(root_path, "seg/{:05d}.npy".format(src_img_indx)))
    except:
        print(f"Couldn't load index {src_img_indx} from {root_path}")
        return
   
    # images in which in which we want to label propogation based on the provided gt seg label
    image_range = [max(src_img_indx - propogation_step, 0), src_img_indx + propogation_step]
    out_indx = frame_range_begin

    acc_json = {}

    for img_indx in range(image_range[0], image_range[1] + 1):
        print("img_index = {}".format(img_indx))
        ### create point cloud in wolrd frame for img_indx ###
        
        try:
            # get the robot pose value
            base_pose = base_pose_data[str(img_indx)]
            # get the depth
            cur_depth = np.load(os.path.join(root_path, "depth/{:05d}.npy".format(img_indx)))
        except:
            print(f'{img_indx} out of bounds! Total images {len(os.listdir(os.path.join(root_path, "rgb")))}')
            continue
        
        lp = LabelPropagate()

        annot_img = lp(src_img, src_depth, src_label, src_pose, base_pose, cur_depth)
        
        # store the annotation file
        np.save(os.path.join(out_dir, "{:05d}.npy".format(out_indx)), annot_img.astype(np.uint32))
        
        # store the annotation + rgb visual
        gt_label = np.load(os.path.join(root_path, "seg/{:05d}.npy".format(img_indx)))
        save_propagated_visual(
            gt_label, 
            annot_img, 
            os.path.join(out_dir, 'lp_visuals'), out_indx
        )

        # calculate metrics
        acc = calculate_accuracy(gt_label, annot_img)
        acc_json[img_indx-src_img_indx] = acc

        out_indx += 1

    with open(os.path.join(out_dir, 'lp_accuracy.json'), 'w') as f:
        json.dump(acc_json, f)


def propagate_dir(reex_dir):
    # should have folders in [r1, r2, s1, c1s, c1l]
    for fold in ['r1', 'r2', 's1', 'c1s', 'c1l']:
        # prop lengths
        prop_f = os.path.join(reex_dir, fold)
        print(f'creating {prop_f}')
        src_img_indx = 0
        json_path = os.path.join(prop_f, 'data.json')
        assert os.path.isfile(json_path)
        with open(json_path, "r") as f:
            base_pose_data = json.load(f)

        for p in range(0, 10, 2):
            out_dir = os.path.join(prop_f, f'pred_label_p{p}')
            if os.path.isdir(out_dir):
                rmtree(out_dir)
            os.makedirs(out_dir)

            propogate_label(
                prop_f, src_img_indx, 
                p,
                base_pose_data,
                out_dir,
                0
            ) 

def acopydir(src, dst, pred_f):
    # print(f'acopydir {src} {dst} {pred_f}')
    og_rgb = os.path.join(src, 'rgb')
    og_rgb_dbg = os.path.join(src, 'rgb_dbg')
    og_visuals = os.path.join(src, pred_f, 'lp_visuals')
    
    out_dir = os.path.join(dst, pred_f)
    rgb_dir = os.path.join(out_dir, 'rgb')
    rgb_dbg_dir = os.path.join(out_dir, 'rgb_dbg')
    seg_dir = os.path.join(out_dir, 'seg')
    vis_dir = os.path.join(out_dir, 'lp_visuals')

    if not os.path.isdir(out_dir):
        for x in [out_dir, rgb_dir, rgb_dbg_dir, seg_dir, vis_dir]:
            os.makedirs(x)
    
    # copy seg files and their corresponding rgb files, numbered appropriately

    fsa = list(glob.glob(os.path.join(seg_dir, '*.npy')))
    ctr = len(fsa)
    for x in glob.glob(os.path.join(src, pred_f, '*.npy')):
        og_indx = int(x.split('/')[-1].split('.')[0])
        # copy seg
        fname_seg = "{:05d}{}".format(ctr, '.npy')
        copyfile(x, os.path.join(seg_dir, fname_seg))
        # copy rgb
        fname_rgb = "{:05d}{}".format(ctr, '.jpg')
        copyfile(
            os.path.join(og_rgb, "{:05d}{}".format(og_indx, '.jpg')),
            os.path.join(rgb_dir, fname_rgb)
        )
        # copy rgb_dbg
        fname_rgb = "{:05d}{}".format(ctr, '.jpg')
        copyfile(
            os.path.join(og_rgb_dbg, "{:05d}{}".format(og_indx, '.jpg')),
            os.path.join(rgb_dbg_dir, fname_rgb)
        )
        # copy visuals
        fname_vis = "{:05d}{}".format(ctr, '.jpg')
        copyfile(
            os.path.join(og_visuals, "{:05d}{}".format(og_indx, '.jpg')),
            os.path.join(vis_dir, fname_vis)
        )
        ctr += 1

def combine(src, dst, input_folds):
    """
    \src (0)
        \input_folds (r1, s1 ..)
            \pred_label_px
    \dst (..\a)
        \pred_label_px
            \rgb
            \seg
            coco.json
            metrics.json

    """
    print(f'src {src} dst {dst}, input_folds {input_folds}')
    for x in input_folds:
        # print(f'combining {os.path.join(src, x)} into {dst}')
        for p in Path(os.path.join(src, x)).rglob('pred_label*'):
            pred_f = str(p).split('/')[-1]
            # print(pred_f, p.parent)
            # want to put src/pred_label into dst/pred_label
            acopydir(p.parent, dst, pred_f)

def run_label_prop(data_dir, job_dir):
    print(f'data_dir {data_dir}')
    jobs = []

    with executor.batch():
        for path in Path(data_dir).rglob('reexplore_data.json'):
            with open(os.path.join(path.parent, 'reexplore_data.json'), 'r') as f:
                data = json.load(f)
                if len(data.keys()) > 0:
                    print(f'processing {path.parent}')
                    for eid in os.listdir(path.parent):
                        if eid.isdigit():
                            # do prop on each 
                            # propagate_dir(os.path.join(path.parent, eid))

                            # combine all propagated based on combinations
                            for out_name, input_folds in set_keys.items():
                                out_dir = os.path.join(path.parent, out_name)
                                if os.path.isdir(out_dir):
                                    print(f'rmtree {out_dir}')
                                    rmtree(out_dir)
                                combine(
                                    os.path.join(path.parent, eid), 
                                    out_dir, 
                                    input_folds
                                )
                            return

                    # def job_unit(pth, out_name, input_fs):
                    #     # now, for each setkey, write a fn to output the rgb, seg folders 
                    #     # combine all c1, s1, r1, r2 into one rgb

                    #     for p in range(2, 10, 2):
                    #         for f in input_fs:
                    #             propagate(pth, p)
                    #             # prop_and_combine(pth, out_name, f, p)
                        
                    #     # TODO: move E1 to rgb, seg 

                    # # job_unit(path.parent, noise)


                    # for out_name, input_fs in set_keys.items():
                    #     # job = executor.submit(job_unit, path.parent, out_name, input_fs)
                    #     # jobs.append(job)
                    #     job_unit(path.parent, out_name, input_fs)
                    #     return

    if len(jobs) > 0:
        print(f"Job Id {jobs[0].job_id.split('_')[0]}, num jobs {len(jobs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for running active vision pipeline")
    parser.add_argument(
        "--data_dir",
        help="path where scene data is being stored",
        type=str,
    )
    parser.add_argument("--job_dir", type=str, default="", help="")
    parser.add_argument("--comment", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")
    parser.add_argument("--noise", action="store_true", default=False, help="Spawn habitat with noise")

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, 'slurm_logs/%j'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="learnfair", #"learnfair", #scavenge
        timeout_min=30,
        mem_gb=256,
        gpus_per_node=4,
        tasks_per_node=1, 
        cpus_per_task=8,
        additional_parameters={
            "mail-user": f"{os.environ['USER']}@fb.com",
            "mail-type": "all",
        },
        slurm_comment="Droidlet Active Vision Pipeline"
    )

    run_label_prop(args.data_dir, args.job_dir)