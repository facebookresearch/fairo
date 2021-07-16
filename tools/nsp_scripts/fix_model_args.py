import pickle

"""
Set the args paths to relative to droidlet root in serialized args 
for loading semantic parsing models.
"""

model_args_path = (
    "../../craftassist/agent/models/semantic_parser/ttad_bert_updated/caip_test_model_args.pk"
)
model_args = pickle.load(open(model_args_path, "rb"))
print(model_args)
model_args.data_dir = "craftassist/agent/datasets/annotated_data/"
model_args.output_dir = "/"
model_args.tree_voc_file = (
    "craftassist/agent/models/semantic_parser/ttad_bert_updated/caip_test_model_tree.json"
)
pickle.dump(model_args, open(model_args_path, "wb"))
