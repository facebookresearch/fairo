"""
Construct input for annotation tools B and C from tool A outputs.
"""
from pprint import pprint
import ast
import json
import copy
import argparse
import os


# construct input for tool D from tool C's output
# inpout for tool D is : 
# text \t highlighted_text \t "reference_object" \t "comparison"
# folder_name_C = '../../minecraft/python/craftassist/text_to_tree_tool/turk_data/new_dance_form_data/next_20/tool3/'
# folder_name_D = '../../minecraft/python/craftassist/text_to_tree_tool/turk_data/new_dance_form_data/next_20/tool4/'
folder_name_C = '/private/home/rebeccaqian/droidlet/tools/annotation_tools/turk_with_s3/C/' #'/Users/kavyasrinet/Desktop/other_actions/0/toolC/'
folder_name_D = '/private/home/rebeccaqian/droidlet/tools/annotation_tools/turk_with_s3/D/' #'/Users/kavyasrinet/Desktop/other_actions/0/toolD/'
tool_C_out_file = folder_name_C + 'all_agreements.txt'
tool_D_input_file = folder_name_D + 'input.txt'

from pprint import pprint
'''
command: Input.command

'''
import copy
example_d = {}

def merge_indices(indices):
    a, b = indices[0]
    for i in range(1, len(indices)):
        x, y = indices[i]
        if x != b + 1:
            return indices
            # raise NotImplementedError("Spans not continuous during merging!!!")
        else:
            b = y
    if a==b:
        return a
    return [a, b]
import ast
import json


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
#         print(out_dict)
#         print(ref_obj_dict)
#         break
            
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
                #print(words, child_name, action_type, val[1])
                
                indices = merge_indices(val[1])
                span_text = None
                if type(indices) == list:
                    if type(indices[0]) == list:
                        # this means that indices were scattered and disjoint
                        for idx in indices:
                            words_copy[idx[0]] = "<span style='background-color: #FFFF00'>" + words_copy[idx[0]]
                            words_copy[idx[1]] = words_copy[idx[1]] + "</span>"
                    else:
                        words_copy[indices[0]] = "<span style='background-color: #FFFF00'>" + words_copy[indices[0]]
                        words_copy[indices[1]] = words_copy[indices[1]] + "</span>"
                else:
                    words_copy[indices] = "<span style='background-color: #FFFF00'>" + words_copy[indices] + "</span>"
                write_line += " ".join(words_copy) + "\t" + action_child_name + "\t" + child_name
                f2.write(write_line+ "\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     # Default to directory of script being run for writing inputs and outputs
#     default_write_dir = os.path.dirname(os.path.abspath(__file__))
    
#     parser.add_argument("--write_dir_path", type=str, default=default_write_dir)
#     args = parser.parse_args()

#     # This must exist since we are using tool A outputs
#     folder_name_A = '{}/A/'.format(args.write_dir_path)
#     # Directories for tools B and C may not exist yet
#     folder_name_B = '{}/B/'.format(args.write_dir_path)
#     folder_name_C = '{}/C/'.format(args.write_dir_path)

#     # If the tool specific write directories do not exist, create them
#     if not os.path.isdir(folder_name_B):
#         os.mkdir(folder_name_B)

#     if not os.path.isdir(folder_name_C):
#         os.mkdir(folder_name_C)

#     tool_A_out_file = folder_name_A + 'all_agreements.txt'
#     tool_B_input_file = folder_name_B + 'input.txt'
#     tool_C_input_file = folder_name_C + 'input.txt'

#     construct_inputs_from_tool_A(tool_A_out_file, tool_B_input_file, tool_C_input_file)