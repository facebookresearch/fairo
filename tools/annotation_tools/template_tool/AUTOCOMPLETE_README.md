# Autocomplete Droidlet Data Annotator

The Autocomplete Data Annotator provides an easy way to create parse trees for natural language commands. This tool is in ongoing development.

# Set up and Installation
Running server:
```
cd ~/droidlet/tools/annotation_tools/template_tool/backend/
npm install && npm start
```

Running client:
```
cd ~/droidlet/tools/annotation_tools/template_tool/frontend/
npm install && npm start
```

In order to write to the Craftassist S3 bucket, you need to be in an environment with `boto3` installed and have valid AWS credentials. For instructions on how to configure awscli, see https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html .

# Usage
## Setting Up 
The tool loads from `annotated.txt`, `templated_locobot.txt` and `short_commands.txt` in `~/droidlet/craftassist/agent/datasets/ground_truth/datasets/` to create the initial data store in `~/droidlet/tools/annotation_tools/template_tool/backend/command_dict_pairs.json`. Commands provided for labelling are first checked against this set, to see if there is an existing parse tree.

Commands we want to label are in `~/droidlet/tools/annotation_tools/template_tool/backend/commands.txt`. Write one command for each line.

## Autocomplete

Currently the tool supports all of `HUMAN_GIVE_COMMAND`, `GET_MEMORY`, `PUT_MEMORY` and `NOOP` dialogue types.

The tool autocompletes children for tree nodes based on matches in the filters spec on key expansion, indicated by `<key>: [space] [space] [Enter]`.

Some examples of find and replace:
Input:
`"GET_MEMORY": \n"`
Match:
`{ "dialogue_type": "GET_MEMORY", "filters": "", "replace": "" }`

Input:
`{ "dialogue_type": "GET_MEMORY", "filters": \n, "replace": "" }`

Match:
`{ "dialogue_type": "GET_MEMORY", "filters": { "triples": "", "output": "", "contains_coreference": "", "memory_type": "", "argval": "", "comparator": "", "author": "", "location": "" }, "replace": "" }`

The tool also populates new triples, on `}, [space] [space] [Enter]`.

`"triples": [ { "pred_text": "", "obj_text": "", "subj_text": "" }, \n`
becomes
`"triples": [ { "pred_text": "", "obj_text": "", "subj_text": "" }, { "pred_text": "", "obj_text": "", "subj_text": "" } ]`

To pretty print a JSON valid dictionary, press Enter in the text box.

## Save and Upload
On `Save`, the current command and parse tree are saved to `command_dict_pairs.json` in `~/droidlet/tools/annotation_tools/template_tool/backend/`.

On `Upload to S3`, the new data pairs in `~/droidlet/tools/annotation_tools/template_tool/backend/command_dict_pairs.json` are first postprocessed into the format required for droidlet NLU components, and written to `~/droidlet/tools/annotation_tools/template_tool/backend/autocomplete_annotations.txt`. This is in the format

```
[command]|[action_dict]\n
[command]|[action_dict]\n
...
```

Then, this file is uploaded to our S3 bucket `s3://craftassist/pubr/`. It is now ready to be used in training, validation or ground truth actions!

