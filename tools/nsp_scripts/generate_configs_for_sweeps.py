"""
This script reads in a .txt file containing the grid search sweep config ranges,
and generates config files to be used in slurm jobs.
"""
import argparse
import os


def generate_config_pair(arr1, arr2):
    configs = []
    for x in arr1:
        for y in arr2:
            if type(x) == list:
                configs.append([*x, y])
            else:
                configs.append([x, y])
    return configs


def generate_configs_from_file(input_file):
    with open(input_file, "r") as fd:
        params = fd.read().splitlines()

    full_args = []
    for line in params:
        arg, values_range = line.split("=")
        values = values_range.split(":")
        arr1 = ["{}={}".format(arg, v) for v in values]
        full_args.append(arr1)

    configs = generate_config_pair(full_args[0], full_args[1])
    for i in range(2, len(full_args)):
        new_configs = generate_config_pair(configs, full_args[i])
        configs = new_configs

    print(configs)
    print(len(configs))
    return configs


def write_config_to_file(config, idx, output_dir, sweep_output_dir, data_dir):
    output_path = "{}config_{}.txt".format(output_dir, idx)
    with open(output_path, "w") as fd:
        for c in config:
            fd.write("--" + c + "\n")
        # additional for static
        fd.write("--output_dir={}/sweep{}/\n".format(sweep_output_dir, idx))
        fd.write("--data_dir={}\n".format(data_dir))
        fd.write("--tree_voc_file={}/caip_test_model_tree.json\n".format(sweep_output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to sweep config, i.e. which parameters to run and their ranges for grid search
    parser.add_argument(
        "--sweep_params_path", help="path to sweep params and values to search over"
    )
    # Name of sweep, to identify
    parser.add_argument(
        "--data_dir", help="path to training data directory containing train/test/valid split"
    )
    parser.add_argument("--output_dir", help="where to write configs to")
    parser.add_argument("--sweep_output_dir", help="where to save models to during sweeps")

    opts = parser.parse_args()
    input_file = opts.sweep_params_path
    configs = generate_configs_from_file(input_file)
    output_dir = opts.output_dir
    sweep_output_dir = opts.sweep_output_dir
    for i in range(len(configs)):
        write_config_to_file(configs[i], i, output_dir, sweep_output_dir, opts.data_dir)
        # create sweep directories
        sweep_dir_path = "{}sweep{}/".format(sweep_output_dir, i)
        if not os.path.isdir(sweep_dir_path):
            os.mkdir(sweep_dir_path)
