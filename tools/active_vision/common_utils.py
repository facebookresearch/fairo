import json
from datetime import datetime

instance_ids = [243,404,196,133,166,170,172]
class_labels = ['chair', 'cushion', 'door', 'indoor-plant', 'sofa', 'table']

def is_annot_validfn_inst(annot):
    if annot not in instance_ids:
        return False
    return True

def is_annot_validfn_class(annot):
    def load_semantic_json(scene):
        habitat_semantic_json = f'/checkpoint/apratik/replica/{scene}/habitat/info_semantic.json'
        with open(habitat_semantic_json, "r") as f:
            hsd = json.load(f)
        if hsd is None:
            print("Semantic json not found!")
        return hsd
    hsd = load_semantic_json('apartment_0')
    label_id_dict = {}
    for obj_cls in hsd["classes"]:
        if obj_cls["name"] in class_labels:
            label_id_dict[obj_cls["id"]] = obj_cls["name"]
    if hsd["id_to_label"][annot] < 1 or hsd["id_to_label"][annot] not in label_id_dict.keys():
        return False
    return True

def log_time(path_to_logfile: str) -> None:
    """Decorator that logs the start and end time of the wrapped function 
    """
    def log(func):
        def wrapped(*args, **kwargs):
            start_time_str = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            func(*args, **kwargs)
            end_time_str = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            with open(path_to_logfile, 'a') as f:
                f.write(f"\nstart {start_time_str} end {end_time_str}")
        return wrapped
    return log

class EarlyStoppingException(Exception):
    pass