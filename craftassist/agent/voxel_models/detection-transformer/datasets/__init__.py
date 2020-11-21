import importlib


def build_dataset(image_set, args):
    # what a hack
    mod = importlib.import_module("datasets." + args.dataset_file)
    return mod.build(image_set, args)
