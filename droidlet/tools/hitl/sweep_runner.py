import argparse
import os
import sys
import shutil
from datetime import datetime, date
import glob
from random import shuffle
import re
import stat
import subprocess
import torch

def write_data_chunk_to_file(data, target_fname, commands_only=False, mask=None):
    count = 0
    os.makedirs(os.path.dirname(target_fname), exist_ok=True)
    print(target_fname)
    with open(target_fname, "w") as fd:
        for line in data:
            if mask is None or mask[count]:
                if commands_only:
                    command, action_dict = line.split("|")
                    fd.write(command + "\n")
                else:
                    fd.write(line)
            count = count + 1




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--droidlet_dir", default="/private/home/aszlam/fairinternal/droidlet/")
    parser.add_argument("--checkpoint_dir", default="/checkpoint/aszlam/nsp/")
    parser.add_argument("--full_data_dir", default="agents/craftassist/datasets/full_data/")
    parser.add_argument("--sweep_config_folder", default="", help="path to sweep config")
    parser.add_argument("--sweep_scripts_output_dir", default="", help="where to put script")
    parser.add_argument("--sweep_name", default="", help="name of sweep")
    parser.add_argument("--output_dir", default="", help="where to put job_output")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--no_sweep", action="store_true", default=False, help="just set up files, no run")
    parser.add_argument(
        "--append_date", action="store_false", help="append date to output dir and job name"
    )
    parser.add_argument("--partition", default="learnfair", help="name of partition")
    opts = parser.parse_args()    

    ###############################################
    # copy data, make splits, write data splits:
    ###############################################
    full_data_folder = os.path.join(opts.droidlet_dir, opts.full_data_dir)
    target_data_folder = os.path.join(opts.checkpoint_dir, date.today().strftime("%b_%d_%Y"))
    print("copying data to " + target_data_folder + " from " + full_data_folder)
    try:
        shutil.rmtree(target_data_folder)
    except:
        pass
    # this is not srictly necessary, going to have individual folders for train/val/test:
    shutil.copytree(full_data_folder, target_data_folder)
    # get data splits:
    split_masks_path = os.path.join(opts.sweep_config_folder, "split_masks.pth")
    print("reading splits from " + split_masks_path)
    # split_masks are boolean torch tensors arranged in
    # into dicts of the form split_masks[data_type][train/val/test]
    # if a mask is None or has length 0, that data will not be written

    """
    split_masks = {"annotated":{}}
    a = torch.multinomial(torch.Tensor([.8,.1,.1]), len(L), replacement=True)
    splits = ["train", "valid", "test"]
    dtype = ["annotated"]
    for i in range(len(splits)):
        split_masks["annotated"][splits[i]] = a==i
    """

    split_masks = torch.load(split_masks_path)
    print("writing splits:")
    for dtype, masks in split_masks.items():
        data_file = os.path.join(full_data_folder, dtype + ".txt")
        with open(data_file, "r") as fd:
            data = fd.readlines()
        for k in ["train", "valid", "test"]:
            mask = masks.get(k)
            if mask is not None and len(mask)>0:
                fname = os.path.join(os.path.join(target_data_folder, k+"/"), dtype + ".txt")
                print("writing {} split of {}".format(k, dtype))
                write_data_chunk_to_file(data, fname, mask=mask)
    print("done writing splits")

    ###################################################
    # generate sweep scripts from configs
    # TODO do this with submitit instead of generating
    # bash scripts
    ###################################################

    assert opts.sweep_scripts_output_dir != ""

    now = datetime.now()
    nowstr = now.strftime("_%m_%d_%H_%M")
    job_name = opts.sweep_name
    output_dir = opts.output_dir
    scripts_dir = opts.sweep_scripts_output_dir
    if opts.append_date:
        job_name = job_name + nowstr
        output_dir = os.path.join(output_dir, job_name)
        scripts_dir = os.path.join(scripts_dir, job_name)
    print("making job dirs {} and script dirs {}".format(output_dir,  scripts_dir))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    sweep_config_path = os.path.join(opts.sweep_config_folder, "sweep_config.txt")
          
    with open(sweep_config_path) as f:
        args = {}
        for l in f.readlines():
            if len(l) > 0 and l[0] != "#":
                w = l.split("=")
                argname = w[0].strip()
                if len(w) == 1:
                    argvals_text = [""]
                else:
                    argvals_text = w[1].strip().split(",")
                args[argname] = [av.strip() for av in argvals_text]

    # grid search... TODO random
    varying_args = [""]
    static_args = " "
    for argname in args.keys():
        new_arglist = []
        if len(args[argname]) > 1:
            for a in varying_args:
                for argval in args[argname]:
                    new_arglist.append(a + " --" + argname + " " + argval)
            varying_args = new_arglist.copy()
        else:
            static_args = static_args + " --" + argname + " " + args[argname][0]
    all_arglists = []
    for a in varying_args:
        all_arglists.append(a + static_args)

    #train_command = os.path.join(opts.droidlet_path,"tools/nsp_scripts/train_model.py")


    # TODO call the data script automatically here
    model_out = os.path.join(output_dir, "model_out/")
    os.makedirs(model_out, exist_ok=True)

    print("the sweep will use data in " + target_data_folder)
    print("models will be written to " + model_out)

    train_command = "tools/nsp_scripts/train_model.py" + " --data_dir " + target_data_folder + " --output_dir " + model_out + " "

    errpaths = []
    outpaths = []
    modelpaths = []
    for i in range(len(all_arglists)):
        uid = job_name + "_P" + str(i)
        body = "#! /bin/bash \n"
        body += "#SBATCH --job-name=" + uid + "\n"
        body += "#SBATCH --output=" + os.path.join(output_dir, str(i) + ".out") + "\n"
        outpaths.append(os.path.join(output_dir, str(i) + ".out"))
        body += "#SBATCH --error=" + os.path.join(output_dir, str(i) + ".err") + "\n"
        errpaths.append(os.path.join(output_dir, str(i) + ".err"))
        body += "#SBATCH --partition=" + opts.partition + "\n"
        body += "#SBATCH --nodes=1 \n"
        body += "#SBATCH --ntasks-per-node=1 \n"
        body += "#SBATCH --gres=gpu:1 \n"
        body += "#SBATCH --cpus-per-task=10 \n"
        body += "#SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself \n"
        body += "#SBATCH --open-mode=append \n"
        body += "#SBATCH --time=4320 \n"
        body += "\n"
        #    body += st #FIXME
        body += "\n"
        body += "module purge\n"
        # TODO make a env loader...
        body += "module load NCCL/2.2.13-cuda.9.0 \n"
        body += "module load anaconda3/5.0.1 \n"
        body += "source activate /private/home/aszlam/.conda/envs/kitchen \n"
        #######
        body += "cd /private/home/aszlam/fairinternal/droidlet/ \n"
        body += "/private/home/aszlam/.conda/envs/kitchen/bin/ipython -- " +  train_command
        body += all_arglists[i]
        #body += " --sbatch --save_model_uid " + job_name + "_" + str(i)
        scriptname = os.path.join(scripts_dir, str(i) + ".sh")
        g = open(scriptname, "w")
        g.write(body)
        g.close()
        st = os.stat(scriptname)
        os.chmod(scriptname, st.st_mode | stat.S_IEXEC)

    mastername = os.path.join(scripts_dir, "master")
    g = open(mastername, "w")
    g.write("#! /bin/sh \n")
    for i in range(len(all_arglists)):
        g.write("# opts :: " + varying_args[i] + " :: " + outpaths[i] + "\n")
    for i in range(len(all_arglists)):
        g.write("echo " + "'" + varying_args[i] + " :: " + outpaths[i] + "'" + "\n")
    for i in range(len(all_arglists)):
        g.write("sbatch " + str(i) + ".sh &\n")
    g.close()
    st = os.stat(mastername)
    os.chdir(scripts_dir)
    os.chmod(mastername, st.st_mode | stat.S_IEXEC)
    os.system(mastername)


    this_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile(os.path.join(this_dir, "sweep_monitor.py"), os.path.join(model_out, "sweep_monitor.py"))
    for i in range(300):
        cmd = 'echo -e "cd {}\n python sweep_monitor.py" | at now +{} minutes'.format(model_out, (i+1)*10)
        os.system(cmd)
