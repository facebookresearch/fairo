import pickle

def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except:
        return False
    return True

class empty_object():
    def __init__(self) -> None:
        pass

def safe_object(input_object):
    if isinstance(input_object, tuple):
        tuple_len = len(input_object)
        to_return = []
        for i in range(tuple_len):
            to_return.append(safe_single_object(input_object[i]))
        return tuple(to_return)
    else:
        return safe_single_object(input_object)
        
def safe_single_object(input_object):
    if is_picklable(input_object):
        return input_object       
    all_attrs = dir(input_object)
    return_obj = empty_object()
    for attr in all_attrs:
        if attr.startswith("__"):
            continue
        if is_picklable(getattr(input_object, attr)):
            setattr(return_obj, attr, getattr(input_object, attr))
    return return_obj

def get_safe_single_object_attr_dict(input_object):
    return_dict = {}
    all_attrs = vars(input_object)
    for attr in all_attrs:
        if attr.startswith("__"):
            continue
        if is_picklable(getattr(input_object, attr)):
            return_dict[attr] = all_attrs[attr]
    return return_dict