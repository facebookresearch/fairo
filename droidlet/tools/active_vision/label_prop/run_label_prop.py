import argparse
import submitit
import glob
from pathlib import Path
import os
import json
from shutil import copyfile, rmtree
import numpy as np
import cv2
import time
from droidlet.perception.robot import LabelPropagate
from PIL import Image
import matplotlib.pyplot as plt
from common_utils import log_time
from math import isnan
from common_utils import heuristics as heus, combinations as set_keys, prop_lengths

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
    frame_range_begin: int,
    label_to_save: int,
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
            cur_pose = base_pose_data[str(img_indx)]
            # get the depth
            cur_depth = np.load(os.path.join(root_path, "depth/{:05d}.npy".format(img_indx)))
        except:
            print(f'{img_indx} out of bounds! Total images {len(os.listdir(os.path.join(root_path, "rgb")))}')
            continue
        
        lp = LabelPropagate()

        annot_img = lp(src_img, src_depth, src_label, src_pose, cur_pose, cur_depth)
        
        # filter label
        annot_img[annot_img != label_to_save] = 0

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


def propagate_dir(reex_dir, out_dir):
    print(f'propagate_dir {reex_dir}')
    # should have folders in [r1, r2, s1, c1s, c1l]
    for fold in heus:
        # prop lengths
        prop_f = os.path.join(reex_dir, fold)
        out_f = os.path.join(out_dir, fold)

        print(f'input {prop_f}, output {out_f}')
        src_img_indx = 0
        json_path = os.path.join(prop_f, 'data.json')
        assert os.path.isfile(json_path), f'{json_path} does not exist!'
        with open(json_path, "r") as f:
            base_pose_data = json.load(f)

        task_json_path = os.path.join(reex_dir, 'task_data.json')
        assert os.path.isfile(task_json_path), f'{task_json_path} does not exist!'
        with open(task_json_path, "r") as f:
            task_data = json.load(f)
            print(f'loaded task data {task_data}')
        
        pred_dir = os.path.join(out_f, 'pred')
        if os.path.isdir(pred_dir):
            rmtree(pred_dir)
        os.makedirs(pred_dir)

        start = time.time()
        propogate_label(
            prop_f, src_img_indx, 
            18,
            base_pose_data,
            pred_dir,
            0,
            int(task_data['label']),
        )
        end = time.time()
        print(f'total one time propagation time {end - start}')

        for p in prop_lengths:
            od = os.path.join(out_f, f'pred_label_p{p}')
            if os.path.isdir(od):
                rmtree(od)
            os.makedirs(od)

            def copyover(in_dir, out_dir, p):
                for x in range(p+1):
                    if os.path.isfile(os.path.join(in_dir, f'{x:05d}.npy')):
                        copyfile(os.path.join(in_dir, f'{x:05d}.npy'), os.path.join(out_dir, f'{x:05d}.npy'))

                # calculate average accuracy
                acc_file = os.path.join(in_dir, "lp_accuracy.json")
                avg_acc, total = 0, 0
                if os.path.isfile(acc_file):
                    with open(acc_file, "r") as fp:
                        accuracies = json.load(fp)
                        print(f'accuracies {accuracies}')
                        for x in range(p+1):
                            if str(x) in accuracies and not isnan(accuracies[str(x)]):
                                avg_acc += accuracies[str(x)]
                                total += 1
                avg_acc /= total
                print(f'average accuracy {avg_acc}')
                with open(os.path.join(out_dir, 'lp_accuracy.txt'), 'w') as f:
                    f.write(str(avg_acc))

            copyover(in_dir=pred_dir, out_dir=od, p=p)
        end2 = time.time()
        print(f'total time to copyover {end2-end}')

def acopydir(src, dst, og_data, pred_f):
    print(f'acopydir {src} {dst} {pred_f}, og_data {og_data}')
    og_rgb = os.path.join(og_data, 'rgb')
    og_seg = os.path.join(og_data, 'seg')
    og_rgb_dbg = os.path.join(og_data, 'rgb_dbg')
    og_visuals = os.path.join(src, 'pred', 'lp_visuals')
    
    out_dir = os.path.join(dst, pred_f)
    rgb_dir = os.path.join(out_dir, 'rgb')
    rgb_dbg_dir = os.path.join(out_dir, 'rgb_dbg')
    seg_gt_dir = os.path.join(out_dir, 'seg_gt')
    seg_dir = os.path.join(out_dir, 'seg')
    vis_dir = os.path.join(out_dir, 'lp_visuals')

    if not os.path.isdir(out_dir):
        for x in [out_dir, rgb_dir, rgb_dbg_dir, seg_dir, seg_gt_dir, vis_dir]:
            os.makedirs(x)
    
    # copy seg files and their corresponding rgb files, numbered appropriately

    fsa = list(glob.glob(os.path.join(seg_dir, '*.npy')))
    ctr = len(fsa)
    print(f'initial counter value {ctr}')
    for x in glob.glob(os.path.join(src, pred_f, '*.npy')):
        og_indx = int(x.split('/')[-1].split('.')[0])
        # copy seg
        fname_seg = "{:05d}{}".format(ctr, '.npy')
        copyfile(x, os.path.join(seg_dir, fname_seg))

        # copy seg_gt
        fname_seg_gt = "{:05d}{}".format(ctr, '.npy')
        copyfile(
            os.path.join(og_seg, "{:05d}{}".format(og_indx, '.npy')),
            os.path.join(seg_gt_dir, fname_seg_gt)
        )

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

def combine(src, dst, og_data, input_folds):
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
    print(f'combine src {src} dst {dst}, input_folds {input_folds}')
    for x in input_folds:
        print(f'combining {os.path.join(src, x)} into {dst}')
        for p in Path(os.path.join(src, x)).rglob('pred_label*'):
            pred_f = str(p).split('/')[-1]
            # only copy if lp accuracy > 90
            print(f'checking lp accuracy for {str(p)}')
            with open(os.path.join(str(p), 'lp_accuracy.txt'), 'r') as f:
                acc = f.readline()
                print (f'average accuracy {acc}')
                if float(acc)*100 < 90:
                    print(f'InsufficientAccuracyError! could skip since average accuracy < 90 ..')
                    # continue

            # want to put src/pred_label into dst/pred_label
            acopydir(p.parent, dst, os.path.join(og_data, x), pred_f)


def sanity_check_traj(x):
    is_valid = True
    tid = x.split('/')[-3]
    # if int(tid) not in [33, 89, 1, 4, 35]:
    #     return False
    print(f'traj {x}')
    gt = x.split('/')[-1]
    reex_objects = [x for x in os.listdir(x) if x.isdigit()]  
    # print(reex_objects)
    valid_objects = 0
    for obj in reex_objects:
        valid = True
        obj_path = os.path.join(x, obj)
        children = os.listdir(obj_path)
        print(f'children {children}')
        for h in heus: #['r1', 'r2', 's1', 'c1l', 'c1s']:
            if h not in children:
                print(f'{h} missing in {obj_path}')
                valid = False
        if valid:
            valid_objects += 1

    if int(gt) != len(reex_objects) or int(gt) != valid_objects:
        print(f'traj {tid} expected objects {gt}, actual {len(reex_objects)}, valid_objects {valid_objects}')
        is_valid = False
    print(f'sanity check {is_valid} for {x}')
    return is_valid

def run_label_prop(data_dir, job_dir, job_out_dir, setting):
    print(f'data_dir {data_dir}')
    jobs = []

    def get_lookuplist(valid_trajs):
        return [f'{x}/{setting}/5' for x in valid_trajs]

    with executor.batch():
        for path in Path(data_dir).rglob('reexplore_data.json'):
            with open(os.path.join(path.parent, 'reexplore_data.json'), 'r') as f:
                data = json.load(f)
                if len(data.keys()) > 0:
                    if sanity_check_traj(str(path.parent)):
                        print(f'processing {path.parent}')
                        for out_name, input_folds in set_keys.items():
                            print(f'extract outdir from path {path}')
                            rel_path = '/'.join(str(path.parent).split('/')[-3:])
                            out_dir = os.path.join(job_out_dir, rel_path, out_name)
                            print(f'out_dir {out_dir}')

                            if os.path.isdir(out_dir):
                                rmtree(out_dir)
                                print(f'rmtree {out_dir}')
                        
                        for eid in os.listdir(path.parent):
                            if eid.isdigit():
                                print(f'\neid {eid}, path.parent {path.parent}')
                                rel_path = '/'.join(str(path.parent).split('/')[-3:])
                                out_dir = os.path.join(job_out_dir, rel_path)
                                print(f'out_dir {out_dir}, rel_path {rel_path}')
                                
                                @log_time(os.path.join(job_dir, 'job_log.txt'))
                                def job_unit(path, eid, set_keys, od, jod, rp):
                                    print(f'eid {eid}, path {path}')
                                    # do label prop on each reexplore subtrajectory 
                                    propagate_dir(os.path.join(path.parent, eid), os.path.join(od, eid))
                                    
                                    # combine all propagated based on combinations
                                    for out_name, input_folds in set_keys.items():
                                        combine(
                                            os.path.join(jod, rp, eid), 
                                            os.path.join(od, out_name), 
                                            os.path.join(data_dir, rp, eid),
                                            input_folds
                                        )

                                # job_unit(path, eid, set_keys, out_dir, job_out_dir, rel_path)
                                job = executor.submit(job_unit, path, eid, set_keys, out_dir, job_out_dir, rel_path)
                                jobs.append(job)
                                print(f"num jobs {len(jobs)}")

    if len(jobs) > 0:
        print(f"Job Id {jobs[0].job_id.split('_')[0]}, num jobs {len(jobs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for running active vision pipeline")
    parser.add_argument(
        "--data_dir",
        help="path where scene data is being stored",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        help="path where output scene data is being stored",
        type=str,
    )
    parser.add_argument("--job_dir", type=str, default="", help="")
    parser.add_argument("--setting", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")
    parser.add_argument("--noise", action="store_true", default=False, help="Spawn habitat with noise")

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, 'slurm_logs/%j'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="learnfair", #"learnfair", #scavenge
        timeout_min=1000,
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

    run_label_prop(args.data_dir, args.job_dir, args.out_dir, args.setting)