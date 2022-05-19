import argparse
from random import choice
from datetime import datetime
from copy import deepcopy
import os
import random
import json

SEED = 123

OBJ_ACTIONS = ["build", "destroy", "copy"]
OBJECTS = ["cube", "sphere", "house", "wall", "pyramid"]
RESPONSES = ["yes", "no"]

NOOP = { "dialogue_type": "NOOP" }
PUT_MEMORY = {
    "dialogue_type": "PUT_MEMORY",
    "filters": {
        "where_clause" : {
            "AND": {
                "pred_text": "has_tag", 
                "obj_text": "active_clarification"
            }
        }
    },
    "upsert" : {
        "memory_data": {
            "memory_type": "TRIPLE",
            "triples": [{
                "pred_text": "has_tag",
                "obj_text": []
            }]
        } 
    }
}


def build_first_turn():
    action = choice(OBJ_ACTIONS)
    ref_obj = choice(OBJECTS)
    user1 = f"User: {action} the {ref_obj}"
    agent1 = f"Agent: I'm not sure about something. I think you wanted me to {action} a {ref_obj}, is that right?"
    first_turn = user1 + " " + agent1
    
    return ref_obj, first_turn

def build_second_turn(first_turn: str):
    response = choice(RESPONSES)
    second_turn = first_turn + " " + f"User: {response}"

    return response, second_turn

def build_next_turn(ref_obj: str, prev_turn: str):
    # assumes that the user answered 'yes' to the check_parse question
    point_q = f"Agent: Is this the {ref_obj}? (Look for the flashing object)"
    response = choice(RESPONSES)
    next_turn = prev_turn + " " + point_q + " " + f"User: {response}"

    return response, next_turn

def build_put_memory(ref_obj: str, turn: str):
    lf = deepcopy(PUT_MEMORY)

    char_idx = turn.find(ref_obj)
    start_idx = len(turn[:char_idx].strip().split(" "))
    end_idx = start_idx + len(ref_obj.split(" ")) - 1

    lf["upsert"]["memory_data"]["triples"][0]["obj_text"] = [0, [start_idx, end_idx]]

    return json.dumps(lf)

def next_turn_wrapper(file, ref_obj: str, turn: str, turn_num: int):
            if turn_num > 5:
                return
            response, next_turn = build_next_turn(ref_obj, turn)
            if response == "no":
                file.write(f'{next_turn}|{json.dumps(NOOP)}\n')
                next_turn_wrapper(file, ref_obj, next_turn, turn_num+1)
            else:
                file.write(f'{next_turn}|{build_put_memory(ref_obj, turn)}\n')
                return

def main(num_cmds):

    random.seed(SEED)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "clarification.txt")
    with open(path, "w") as file:
        for i in range(num_cmds):
            ref_obj, first_turn = build_first_turn()
            initial_response, second_turn = build_second_turn(first_turn)
            file.write(f"{second_turn}|{json.dumps(NOOP)}\n")

            if initial_response == "yes":
                next_turn_wrapper(file, ref_obj, second_turn, 1)

    os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test"), exist_ok=True)
    
    with open(path, "r") as f1:
        all_data = f1.readlines()
        random.shuffle(all_data)
    
    line = 0
    path = os.path.join(save_dir, f"train/{filename}")
    with open(path, "w") as f2:
        while line < (0.8 * len(all_data)):
            f2.write(all_data[line])
            line += 1
    path = os.path.join(save_dir, f"valid/{filename}")
    with open(path, "w") as f3:
        while line < (0.9 * len(all_data)):
            f3.write(all_data[line])
            line += 1
    path = os.path.join(save_dir, f"test/{filename}")
    with open(path, "w") as f4:
        while line < len(all_data):
            f4.write(all_data[line])
            line += 1

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_cmds",
        type=int,
        default=600,
        help="The number of commands to generate (about 1/3 the number of training data points)"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine new templated data with the existing data set"
    )
    opts = parser.parse_args()

    if opts.combine:
        save_dir = "/private/home/ethancarlson/fairo/droidlet/artifacts/datasets/annotated_data"
        filename = "templated_clarification.txt"
    else:
        save_dir = f"/checkpoint/ethancarlson/nsp_pp/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        filename = "templated.txt"
    
    main(opts.num_cmds)