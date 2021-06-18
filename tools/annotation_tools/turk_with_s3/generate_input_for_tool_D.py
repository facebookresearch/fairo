"""
Construct input for annotation tools D from tool C outputs.

inpout for tool D is : 
text \t \text \t "reference_object" \t "comparison" \t highlighted_text
"""
from pprint import pprint
import ast
import json
import copy
import argparse
import os
from pprint import pprint
import copy

def merge_indices(indices):
    a, b = indices[0]
    for i in range(1, len(indices)):
        x, y = indices[i]
        if x != b + 1:
            return indices
        else:
            b = y
    if a==b:
        return a
    return [a, b]
import ast
import json


def construct_inputs_from_tool_C(tool_C_out_file, tool_D_input_file):
    all_actions = {}
    with open(tool_C_out_file, "r") as f1, open(tool_D_input_file, 'w') as f2:
        composites= []
        
        for line in f1.readlines():
            line  = line.strip()
            cmd, action_child_name , out = line.split("\t")
            words = cmd.split()
            
            words_copy = copy.deepcopy(words)
            #pprint(out)
            if action_child_name not in ['reference_object', 
                                        'receiver_reference_object',
                                        'source_reference_object']:
                print("Not a reference object!!!")
                continue
            out_dict = json.loads(out)
            ref_obj_dict = out_dict[action_child_name]
                
            # find children that need to be re-annotated
            for key, val in ref_obj_dict.items():
                child_name = None
                # child needs annotation
                if val[0]== 'no':
                    # insert "span"
                    words_copy = copy.deepcopy(words)
                    child_name = key
                    
                    write_line = ""
                    write_line += " ".join(words) + "\t"
                    
                    if len(val[1]) > 1:
                        indices = [merge_indices(val[1])]
                    else:
                        indices = val[1]
                    
                    line_output = "{}\t{}\t{}\t{}\t{}".format(" ".join(words), " ".join(words), action_child_name, child_name, str(indices).replace(", ","-"))
                    f2.write(line_output+ "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default to directory of script being run for writing inputs and outputs
    default_write_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--write_dir_path", type=str, default=default_write_dir)
    args = parser.parse_args()

    # This must exist since we are using tool A outputs
    folder_name_C = '{}/C/'.format(args.write_dir_path)
    # Directories for tool D may not exist yet
    folder_name_D = '{}/D/'.format(args.write_dir_path)

    # If the tool specific write directories do not exist, create them
    if not os.path.isdir(folder_name_D):
        os.mkdir(folder_name_D)

    tool_C_out_file = folder_name_C + 'all_agreements.txt'
    tool_D_input_file = folder_name_D + 'input.txt'

    construct_inputs_from_tool_C(tool_C_out_file, tool_D_input_file)