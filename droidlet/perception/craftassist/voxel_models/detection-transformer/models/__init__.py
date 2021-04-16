import importlib


def build_model(args):
    # what a hack
    mod = importlib.import_module("models." + args.model_file)
    return mod.build(args)
