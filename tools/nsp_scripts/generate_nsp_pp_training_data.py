import argparse
from random import choice
from datetime import datetime
from copy import deepcopy

OBJ_ACTIONS = ["build", "destroy", "copy"]
OBJECTS = ["cube", "sphere", "house", "wall", "pyramid"]
RESPONSES = ["yes", "no"]

SAVE_PATH = f"/checkpoint/ethancarlson/nsp_pp/{round(datetime.timestamp(datetime.utcnow()))}.txt"

NOOP = { 'dialogue_type': 'NOOP' }
PUT_MEMORY = {
    "dialogue_type": "PUT_MEMORY",
    "filters": {
        "where_clause" : {
            "AND": {
                "pred_text": "has_tag", 
                "obj_text": {
                    "fixed_value": "active_clarification"
                }
            }
        }
    },
    "upsert" : {
        "memory_data": {
            "memory_type": "TRIPLE",
            "triples": [{
                "pred_text": "has_tag",
                "obj_text": ""
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

def build_put_memory(ref_obj: str):
    lf = deepcopy(PUT_MEMORY)
    # TODO Replace this with a search for the word index and format the SPAN appropriately
    lf["upsert"]["memory_data"]["triples"][0]["obj_text"] = ref_obj

    return lf

def next_turn_wrapper(file, ref_obj: str, turn: str, turn_num: int):
            if turn_num > 5:
                return
            response, next_turn = build_next_turn(ref_obj, turn)
            if response == "no":
                file.write(f"{next_turn}|{NOOP}\n")
                next_turn_wrapper(file, ref_obj, next_turn, turn_num+1)
            else:
                file.write(f"{next_turn}|{build_put_memory(ref_obj)}\n")
                return

def main(num_cmds):
    with open(SAVE_PATH, "w") as file:
        for i in range(num_cmds):
            ref_obj, first_turn = build_first_turn()
            initial_response, second_turn = build_second_turn(first_turn)
            file.write(f"{second_turn}|{NOOP}\n")

            if initial_response == "yes":
                next_turn_wrapper(file, ref_obj, second_turn, 1)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_cmds",
        type=int,
        default=100,
        help="The number of commands to generate (about 1/3 the number of training data points)"
    )
    opts = parser.parse_args()
    
    main(opts.num_cmds)