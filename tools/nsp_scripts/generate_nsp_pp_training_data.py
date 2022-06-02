import argparse
from random import choice
from datetime import datetime
from copy import deepcopy
import os
import random
import json

SEED = 123

FILLERS = [
    "",
    "",
    " please",
    " bot",
    " will you",
    " can you",
    " will you please",
]
OBJ_ACTIONS = [" destroy ", " copy "]
COLORS = [" blue", " red", " green"]
SIZES = [" big", " small", " tiny", " huge"]
ADJECTIVES = ["", "", " floating"] + COLORS + SIZES
ADJ_NOT_EMPTY = [adj for adj in ADJECTIVES if adj]
OBJECTS = [
    " cube",
    " sphere",
    " house",
    " wall",
    " pyramid",
    " square",
    " platform",
]
RESPONSES = ["yes", "no", "modification", "replacement"]
ITS = ["", "It's "]
MODIFICATION_FOLLOWUPS = ["adjective", "near"]
REPLACEMENT_PREFIXES = ["", " No,", " I mean"]
NEAR = ["near", "next to", "beside", "by"]

NOOP = {"dialogue_type": "NOOP"}
RESOLVE_POINT_LF = {
    "dialogue_type": "PUT_MEMORY",
    "filters": {
        "where_clause": {
            "AND": {"pred_text": "has_tag", "obj_text": "active_clarification"}
        }
    },
    "upsert": {
        "memory_data": {
            "memory_type": "TRIPLE",
            "triples": [{"pred_text": "has_tag", "obj_text": []}],
        }
    },
}
ORIG_CMD_LF = {
    "dialogue_type": "HUMAN_GIVE_COMMAND",
    "event_sequence": [
        {
            "action_type": "",
            "reference_object": {
                "filters": {"where_clause": {"AND": []}},
                "text_span": [],
            },
        }
    ],
}
NEAR_SELECTOR = {
    "location": {
        "reference_object": {
            "filters": {
                "where_clause": {
                    "AND": [{"pred_text": "has_name", "obj_text": []}]
                }
            }
        },
        "relative_direction": "NEAR",
        "text_span": [],
    }
}

# CC1 (there is no ___) Clarification followups:
# "destroy the cube" -> "I mean the sphere" (modification)
# "destroy the sphere" -> "I mean destroy the cube" (replacement)
# "destroy the sphere" -> "No, destroy the cube" (replacement)
# "destroy the sphere" -> "destroy the cube" (replacement)
# "destroy the sphere" -> "it's near the hole" (modification)

# CC2 (there is more than one ___) Clarification followups:
# "destroy the cube" -> "The one near the hole" (new parse, no memory update)
# "destroy the cube" -> "I mean the blue cube" (new parse, no memory update)
# "destroy the sphere" -> "the big one" (new parse, no memory update)


def build_first_turn():
    """
    Creates the first user command and agent clarification response (check_parse).
    Inital command may contain filler starter words an adjective that modifies
    the reference object, in addition to the required action and ref_obj
    """
    filler = choice(FILLERS)
    action = choice(OBJ_ACTIONS)
    ref_obj = choice(OBJECTS)
    adj = choice(ADJECTIVES)
    user1 = f"User:{filler}{action}the{adj}{ref_obj}"
    agent1 = f"Agent: I'm not sure about something. I think you wanted me to{action}a{ref_obj}, is that right?"
    first_turn = user1 + " " + agent1

    return action, adj, ref_obj, first_turn


def build_second_turn(
    action: str, adj: str, ref_obj: str, first_turn: str, file
):
    """
    Constructs the second turn of dialog, which is either a yes/no response
    or a clarification follow-up (either a modification or replacement command).
    Also routes the necessary information to write the consequent LF to the file
    """
    response = choice(RESPONSES)
    if response in ["yes", "no"]:
        second_turn = first_turn + " " + f"User: {response}"
    else:
        second_turn = build_follow_up(
            action, adj, ref_obj, first_turn, response, file
        )

    return response, second_turn


def build_follow_up(
    action: str,
    adj: str,
    ref_obj: str,
    first_turn: str,
    response_type: str,
    file,
):
    """
    Builds the user clarification follow-up, which is always the second thing
    said by the user. Writes the correct LF to the file.
    """
    new_adj = near_thing = near_phrase = new_ref_obj = None

    if response_type == "modification":
        modification = choice(MODIFICATION_FOLLOWUPS)
        if modification == "adjective":
            new_adj = choice([a for a in ADJ_NOT_EMPTY if a != adj])
            user2 = f"User: {choice(ITS)}the{new_adj} thing"
        elif modification == "near":
            near_thing = choice([o for o in OBJECTS if o != ref_obj])
            near_phrase = f"{choice(NEAR)} the{near_thing}"
            user2 = f"User: {choice(ITS)}{near_phrase}"

    elif response_type == "replacement":
        new_adj = choice(ADJECTIVES)
        new_ref_obj = choice([o for o in OBJECTS if o != ref_obj])
        user2 = f"User:{choice(REPLACEMENT_PREFIXES)}{choice([' ', action])}the{new_adj}{new_ref_obj}"

    second_turn = first_turn + " " + user2

    lf = build_followup_lf(
        action,
        adj,
        ref_obj,
        new_adj,
        near_thing,
        near_phrase,
        new_ref_obj,
        second_turn,
        response_type,
    )
    file.write(f"{second_turn}|{lf}\n")

    return second_turn


def build_followup_lf(
    action: str,
    adj: str,
    ref_obj: str,
    new_adj: str,
    near_thing: str,
    near_phrase: str,
    new_ref_obj: str,
    second_turn: str,
    response_type: str,
):
    lf = deepcopy(ORIG_CMD_LF)
    lf["event_sequence"][0]["action_type"] = action.strip()

    lf_ref_obj = new_ref_obj if new_ref_obj else ref_obj
    ref_obj_span = build_span(second_turn, lf_ref_obj)

    if response_type == "replacement":
        # in a modification we don't want the has_name key, just rely on the modification
        lf["event_sequence"][0]["reference_object"]["filters"]["where_clause"][
            "AND"
        ].append({"pred_text": "has_name", "obj_text": ref_obj_span})

    lf_adj = new_adj if new_adj else adj
    if lf_adj:
        if lf_adj in COLORS:
            pred_text = "has_colour"
        elif lf_adj in SIZES:
            pred_text = "has_size"
        else:
            pred_text = "has_tag"
        lf["event_sequence"][0]["reference_object"]["filters"]["where_clause"][
            "AND"
        ].append(
            {
                "pred_text": pred_text,
                "obj_text": build_span(second_turn, lf_adj),
            }
        )

        # if there is a modified adj, won't be able to create a contiguous span with it
        ref_obj_text = lf_adj.strip() + " " + lf_ref_obj.strip()
        try_span = build_span(second_turn, ref_obj_text)
        if try_span:
            lf["event_sequence"][0]["reference_object"]["text_span"] = try_span
        else:
            lf["event_sequence"][0]["reference_object"][
                "text_span"
            ] = ref_obj_span
    else:
        lf["event_sequence"][0]["reference_object"]["text_span"] = ref_obj_span

    # No empty where clause
    if (
        len(
            lf["event_sequence"][0]["reference_object"]["filters"][
                "where_clause"
            ]["AND"]
        )
        == 0
    ):
        lf["event_sequence"][0]["reference_object"]["filters"].pop(
            "where_clause", None
        )

    if near_thing:
        selector = deepcopy(NEAR_SELECTOR)
        selector["location"]["reference_object"]["filters"]["where_clause"][
            "AND"
        ][0]["obj_text"] = build_span(second_turn, near_thing)
        selector["location"]["text_span"] = build_span(
            second_turn, near_phrase
        )
        lf["event_sequence"][0]["reference_object"]["filters"][
            "selector"
        ] = selector

    return json.dumps(lf)


def build_next_turn(ref_obj: str, prev_turn: str):
    # assumes that the user answered 'yes' to the check_parse question
    point_q = f"Agent: Is this the{ref_obj}? (Look for the flashing object)"
    response = choice(["yes", "no"])
    next_turn = prev_turn + " " + point_q + " " + f"User: {response}"

    return response, next_turn


def build_resolve_point_lf(ref_obj: str, turn: str):
    lf = deepcopy(RESOLVE_POINT_LF)
    lf["upsert"]["memory_data"]["triples"][0]["obj_text"] = build_span(
        turn, ref_obj
    )

    return json.dumps(lf)


def build_span(turn: str, text: str):
    cleantext = text.strip()
    char_idx = turn.find(cleantext)
    if char_idx == -1:
        return False
    start_idx = len(turn[:char_idx].strip().split(" "))
    end_idx = start_idx + len(cleantext.split(" ")) - 1

    return [0, [start_idx, end_idx]]


def next_turn_wrapper(file, ref_obj: str, turn: str, turn_num: int):
    if turn_num > 5:
        return
    response, next_turn = build_next_turn(ref_obj, turn)
    if response == "no":
        file.write(f"{next_turn}|{json.dumps(NOOP)}\n")
        next_turn_wrapper(file, ref_obj, next_turn, turn_num + 1)
    else:
        file.write(f"{next_turn}|{build_resolve_point_lf(ref_obj, turn)}\n")
        return


def main(num_cmds):

    random.seed(SEED)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "clarification.txt")
    with open(path, "w") as file:
        for i in range(num_cmds):
            action, adj, ref_obj, first_turn = build_first_turn()
            initial_response, second_turn = build_second_turn(
                action, adj, ref_obj, first_turn, file
            )

            if initial_response in ["yes", "no"]:
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
        default=1000,
        help="The number of commands to generate (about 1/2 the number of training data points)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine new templated data with the existing data set",
    )
    opts = parser.parse_args()

    if opts.combine:
        save_dir = "/private/home/ethancarlson/fairo/droidlet/artifacts/datasets/annotated_data"
        filename = "templated_clarification.txt"
    else:
        save_dir = f"/checkpoint/ethancarlson/nsp_pp/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        filename = "templated.txt"

    main(opts.num_cmds)
