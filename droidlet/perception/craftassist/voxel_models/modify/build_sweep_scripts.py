"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from datetime import datetime
import argparse
import os
import stat

parser = argparse.ArgumentParser()
parser.add_argument("--sweep_config_path", default="", help="path to sweep config")
parser.add_argument("--sweep_scripts_output_dir", default="", help="where to put script")
parser.add_argument("--sweep_name", default="", help="name of sweep")
parser.add_argument("--output_dir", default="", help="where to put job_output")
parser.add_argument(
    "--append_date", action="store_false", help="append date to output dir and job name"
)
parser.add_argument("--partition", default="learnfair", help="name of partition")
opts = parser.parse_args()

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
os.makedirs(output_dir, exist_ok=True)
os.makedirs(scripts_dir, exist_ok=True)


with open(opts.sweep_config_path) as f:
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
    body += "source activate /private/home/kavyasrinet/.conda/envs/minecraft_env\n"
    body += "cd /private/home/aszlam/fairinternal/minecraft/python/craftassist\n"
    #######
    body += "/private/home/kavyasrinet/.conda/envs/minecraft_env/bin/python voxel_models/modify/train_conv_model.py"
    body += all_arglists[i]
    body += " --sbatch --save_model_uid " + job_name + "_" + str(i)
    scriptname = os.path.join(scripts_dir, str(i) + ".sh")
    g = open(scriptname, "w")
    g.write(body)
    g.close()
    st = os.stat(scriptname)
    os.chmod(scriptname, st.st_mode | stat.S_IEXEC)

g = open(os.path.join(scripts_dir, "master"), "w")
g.write("#! /bin/sh \n")
for i in range(len(all_arglists)):
    g.write("# opts :: " + varying_args[i] + " :: " + outpaths[i] + "\n")
for i in range(len(all_arglists)):
    g.write("echo " + "'" + varying_args[i] + " :: " + outpaths[i] + "'" + "\n")
for i in range(len(all_arglists)):
    g.write("sbatch " + str(i) + ".sh &\n")
g.close()
