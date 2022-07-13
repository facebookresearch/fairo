import torch


def load_model(model_fpath: str):
    """
    generates a model instance from the model file
    """
    return torch.load(model_fpath)


def get_keys(model):
    """
    helper method to get all keys for the model
    """
    return model.keys()


def get_value_by_key(model, key):
    """
    helper method for getting a value in the model dict
    """
    return model[key]
