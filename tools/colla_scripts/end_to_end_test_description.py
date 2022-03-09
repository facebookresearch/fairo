import random

'''
    1. move <rel dir agent> to <ref object name> 
    2. move <rel dir> <n> steps
    3. move <rel dir of agent>
    4. dig <name> <rel dir> <ref obj name>
    5. dig <name> <n blocks deep>
    6. Fill <coref> <ref obj name> with <name>
    7. destroy <name> <rel dir> <ref obj name>
    8. spawn <ref obj name> <rel dir> <ref obj name>
    9. build <name> <rel dir> <ref obj name>
    10. build <size> <name>
'''
def generate_temp(option_num=0):
    if not option_num:
        option_num = random.randint(1, 10)

    # Move actions
    rel_dir_agent = {
        'LEFT': 'left',
        'RIGHT': 'right',
        'FRONT': 'forward',
        'BACK': 'back'
    }
    rel_dir_name = random.choice(list(rel_dir_agent.keys()))
    rel_dir_val = rel_dir_agent[rel_dir_name]
    ref_obj_names = [
        'square',
        'cube',
        'house',
        'pool'
    ]

    ref_obj_name = random.choice(ref_obj_names)
    begin_word = random.choice(['move', 'walk', 'run'])
    if option_num == 1:
        # move <rel dir agent> to <ref object name>
        phrase = " ".join([begin_word, rel_dir_val, "to", "the", ref_obj_name])
        output = {"template_number": option_num, "rel_dir": rel_dir_name, "location_ref_obj_name": ref_obj_name}
    elif option_num == 2:
        # move <rel dir> <n> steps
        n = random.randint(1, 10)
        phrase = " ".join([begin_word, rel_dir_val, str(n), "steps"])
        output = {"template_number": option_num, "rel_dir": rel_dir_name, "n": n}
    elif option_num == 3:
        # move <rel dir of agent>
        phrase = " ".join([begin_word, rel_dir_val])
        output = {"template_number": option_num, "rel_dir": rel_dir_name}
    elif option_num in [4, 5]:
        action_type = "DIG"
        schematic_name = random.choice(['hole', 'pool', 'dent'])
        # to the left of
        rel_dir_agent = {
            'LEFT': 'to the left of',
            'RIGHT': 'to the right of',
            'FRONT': 'in front of',
            'BACK': 'behind'
        }
        rel_dir_name = random.choice(list(rel_dir_agent.keys()))
        rel_dir_val = rel_dir_agent[rel_dir_name]

        if option_num == 4:
            # dig <name> <rel dir> <ref obj name>
            phrase = " ".join(["dig", "a", schematic_name, rel_dir_val, ref_obj_name])
            output = {
                "template_number": option_num,
                "rel_dir": rel_dir_name,
                "schematic_name": schematic_name,
                "location_ref_obj_name": ref_obj_name
            }
        elif option_num == 5:
            # dig <name> <n> blocks deep
            n = random.randint(1, 10)
            phrase = " ".join(["dig", "a", schematic_name, str(n), "blocks deep"])
            output = {
                "template_number": option_num,
                "schematic_name": schematic_name,
                "n": n
            }
    elif option_num == 6:
        # Fill <coref> <ref obj name> with <name>
        schematic_name = random.choice(['sand', 'water', 'grass', 'dirt', 'stones'])
        coref_val = random.choice(['this', 'that'])
        phrase = " ".join(["fill", coref_val, ref_obj_name, 'with', schematic_name])
        output = {
            "template_number": option_num,
            "contains_coreference": True,
            "ref_obj_name": ref_obj_name,
            "schematic_name": schematic_name,
        }
    elif option_num == 7:
        # destroy <name> <rel dir> <ref obj name>
        rel_dir_agent = {
            'LEFT': 'to the left of',
            'RIGHT': 'to the right of',
            'FRONT': 'in front of',
            'BACK': 'behind'
        }
        rel_dir_name = random.choice(list(rel_dir_agent.keys()))
        rel_dir_val = rel_dir_agent[rel_dir_name]
        location_ref_obj_name = random.choice(ref_obj_names)
        phrase = " ".join(["destroy the", ref_obj_name, "that is", rel_dir_val, "the", location_ref_obj_name])
        output = {
            "template_number": option_num,
            "ref_obj_name": ref_obj_name,
            "rel_dir": rel_dir_name,
            "location_ref_obj_name": location_ref_obj_name
        }
    elif option_num == 8:
        # spawn <ref obj name> <relative direction> <ref obj name>
        rel_dir_agent = {
            'LEFT': 'to the left of',
            'RIGHT': 'to the right of',
            'FRONT': 'in front of',
            'BACK': 'behind'
        }
        rel_dir_name = random.choice(list(rel_dir_agent.keys()))
        rel_dir_val = rel_dir_agent[rel_dir_name]
        location_ref_obj_name = random.choice(ref_obj_names)
        phrase = " ".join(["spawn", "a", ref_obj_name, rel_dir_val, "the", location_ref_obj_name])
        output = {
            "template_number": option_num,
            "ref_obj_name": ref_obj_name,
            "rel_dir": rel_dir_name,
            "location_ref_obj_name": location_ref_obj_name
        }
    elif option_num == 9:
        # build <name> <rel dir> <ref obj name>
        schematic_name = random.choice(['square', 'cube', 'cuboid', 'circle', 'tower'])
        rel_dir_agent = {
            'LEFT': 'to the left of',
            'RIGHT': 'to the right of',
            'FRONT': 'in front of',
            'BACK': 'behind'
        }
        rel_dir_name = random.choice(list(rel_dir_agent.keys()))
        rel_dir_val = rel_dir_agent[rel_dir_name]
        location_ref_obj_name = random.choice(ref_obj_names)
        phrase = " ".join(["build", "a", schematic_name, rel_dir_val, "the", location_ref_obj_name])
        output = {
            "template_number": option_num,
            "schematic_name": schematic_name,
            "rel_dir": rel_dir_name,
            "location_ref_obj_name": location_ref_obj_name
        }
    elif option_num == 10:
        # build <size> <name>
        size = random.choice(["big", "small", "tiny"])
        schematic_name = random.choice(['square', 'cube', 'cuboid', 'circle', 'tower'])
        phrase = " ".join(["build", "a", size, schematic_name])
        output = {
            "template_number": option_num,
            "schematic_name": schematic_name,
            "size": size
        }

    return phrase, output


for i in range(1, 4):
    print(generate_temp())