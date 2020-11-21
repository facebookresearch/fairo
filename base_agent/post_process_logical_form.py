"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import copy

# move location inside reference_object for Fill and Destroy actions
def fix_fill_and_destroy_location(action_dict):
    action_name = action_dict["action_type"]
    if action_name in ["FILL", "DESTROY"]:
        if "location" in action_dict:
            if "reference_object" not in action_dict:
                action_dict["reference_object"] = {}
            action_dict["reference_object"]["location"] = action_dict["location"]
            action_dict.pop("location")
    return action_dict


# fix for location_type, adding them to reference object instead
def fix_location_type_in_location(d):
    new_d = copy.deepcopy(d)
    for key, value in d.items():
        if key == "location_type":
            if value in ["SPEAKER_LOOK", "AGENT_POS", "SPEAKER_POS", "COORDINATES"]:
                # keep value the same for SPEAKER_LOOK, update for all others.
                updated_value = value
                if value == "AGENT_POS":
                    updated_value = "AGENT"
                elif value == "SPEAKER_POS":
                    updated_value = "SPEAKER"
                elif value == "COORDINATES":
                    updated_value = {"coordinates_span": d["coordinates"]}

                # add to reference object instead
                if "reference_object" in d:
                    if not new_d.get("filters"):
                        new_d["filters"] = []
                    new_d["reference_object"]["special_reference"] = updated_value
                else:
                    new_d["reference_object"] = {"special_reference": updated_value}
            new_d.pop(key)
            new_d.pop("coordinates", None)

        if type(value) == dict:
            new_d[key] = fix_location_type_in_location(value)
    return new_d


# fix reference object to have properties inside 'filters'
def fix_reference_object_with_filters(d):
    new_d = copy.deepcopy(d)
    for key, value in d.items():
        if key in ["reference_object", "reference_object_1", "reference_object_2"]:
            val = d[key]
            if "special_reference" not in val:
                if "repeat" in val:
                    new_d[key] = {"repeat": val["repeat"]}
                    val.pop("repeat")
                    new_d[key]["filters"] = val
                elif "location" in val:
                    new_d[key] = fix_location_type_in_location(val)
                else:
                    new_d[key] = {"filters": val}

                if type(val) == dict:
                    new_d[key]["filters"] = fix_reference_object_with_filters(val)
        elif type(value) == dict:
            new_d[key] = fix_reference_object_with_filters(value)

    return new_d
