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
    return list(model.keys())


def get_value_by_key(model, key):
    """
    helper method for getting a value in the model dict
    """

    value = model[key]

    # logic for preparing json
    if isinstance(value, argparse.Namespace) or isinstance(value, collections.OrderedDict):
        # for args and state_dict
        return json.dumps(value.__dict__)
    else:
        return json.dumps(value)


def get_complete_model(model):
    """
    helper method to get the complete model
    """
    model_dict = {}
    for key in model.keys():
        model_dict[key] = get_value_by_key(model, key)
    return json.dumps(model_dict)
