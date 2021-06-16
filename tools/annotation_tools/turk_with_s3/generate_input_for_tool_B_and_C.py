"""
Construct input for annotation tools B and C from tool A outputs.
"""
from pprint import pprint
import ast
import json
import copy
import argparse
import os


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


def construct_inputs_from_tool_A(tool_A_out_file, tool_B_input_file, tool_C_input_file):
    all_actions = {}
    composites= []
    with open(tool_A_out_file, "r") as f1, open(tool_B_input_file, 'w') as f2, open(tool_C_input_file, 'w') as f3:
        
        for line in f1.readlines():
            line  = line.strip()
            cmd, out = line.split("\t")
            words = cmd.split()
            
            words_copy = copy.deepcopy(words)
            action_dict = ast.literal_eval(out)
            action_type = action_dict['action_type'][1]
            
            all_actions[action_type] =  all_actions.get(action_type, 0)+ 1
            # write composite separately
            # NOTE: not currently using composites in annotations pipeline
            if action_type=='composite_action':
                composites.append(" ".join(words))
                continue
            
            # no need to annotate children of these two actions
            if action_type == 'noop':
                continue
                
            # find children that need to be re-annotated
            for key, val in action_dict.items():
                child_name = None
                # child needs annotation
                if val[0]== 'no':
                    # insert "span"
                    words_copy = copy.deepcopy(words)
                    child_name = key           
                    indices = merge_indices(val[1])
                    span_text = None
                    line_output = "{}\t{}\t{}\t{}\t{}".format(" ".join(words), " ".join(words), action_type, child_name, [indices])
                    if child_name in ['reference_object', 
                                    'receiver_reference_object', 
                                    'source_reference_object']:
                        # write for tool C 
                        f3.write(line_output + "\n")
                    else:
                        # write for tool B
                        f2.write(line_output + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default to directory of script being run for writing inputs and outputs
    default_write_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--write_dir_path", type=str, default=default_write_dir)
    args = parser.parse_args()

    # This must exist since we are using tool A outputs
    folder_name_A = '{}/A/'.format(args.write_dir_path)
    # Directories for tools B and C may not exist yet
    folder_name_B = '{}/B/'.format(args.write_dir_path)
    folder_name_C = '{}/C/'.format(args.write_dir_path)

    # If the tool specific write directories do not exist, create them
    if not os.path.isdir(folder_name_B):
        os.mkdir(folder_name_B)

    if not os.path.isdir(folder_name_C):
        os.mkdir(folder_name_C)

    tool_A_out_file = folder_name_A + 'all_agreements.txt'
    tool_B_input_file = folder_name_B + 'input.txt'
    tool_C_input_file = folder_name_C + 'input.txt'

    construct_inputs_from_tool_A(tool_A_out_file, tool_B_input_file, tool_C_input_file)