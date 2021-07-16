# Autocomplete Droidlet Data Annotator

The Autocomplete Data Annotator provides an easy way to create parse trees for natural language commands. This tool is in ongoing development.

# Set up and Installation
## Fetching Data
Make sure that you have the latest datasets downloaded from S3.
```
cd droidlet
./tools/data_scripts/try_download.sh
```
Note that our scripts and modules are tested in Linux environments on FAIR devservers. There may be differences running on Mac OSX.
For internal users, we recommend running our client and server on devfair and tunnelling ports 3000 and 9000 with Eternal Terminal, which persists the connections, eg.
```
et <your_devserver>:8080 -N -t 3000:3000 --jport 8080
```
and for the server,
```
et <your_devserver>:8080 -N -t 9000:9000 --jport 8080
```

## Preprocessing
First, create a file `~/droidlet/tools/annotation_tools/template_tool/backend/commands.txt` and write commands we want to annotate, one on each line.

To prepopulate the tool with annotated data, run the preprocessing script from the `backend` folder:
```
cd ~/droidlet/tools/annotation_tools/template_tool/backend/
python ~/droidlet/tools/data_processing/preprocess_datasets_for_autocomplete.py

Optional Args:
--annotations_dir_path: Path to directory containing existing labelled data.
--commands_path: Path to file with one command per line, which we want to annotate. Defaults to commands.txt
```

By default, the tool loads from `annotated.txt`, `locobot.txt`, `high_pri_commands.txt` and `short_commands.txt` in `~/droidlet/craftassist/agent/datasets/full_data/` to create the initial data store in `~/droidlet/tools/annotation_tools/template_tool/frontend/src/command_dict_pairs.json`. Commands provided for labelling are first checked against this set, to see if there is an existing parse tree.

Commands we want to label are in `~/droidlet/tools/annotation_tools/template_tool/backend/commands.txt`. Write one command for each line.


## Running the Autocomplete Tool
In two separate terminal sessions on devfair, run:
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

Then you are able to access the Autocomplete tool in your browser at `localhost:3000/autocomplete`.

In order to write to the Craftassist S3 bucket, you need to be in an environment with `boto3` installed and have valid AWS credentials. For instructions on how to configure awscli, see https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html . If you do not have this configured, you can still run the tool as usual and save annotations to disk.

# Usage
## Autocomplete

Currently the tool supports all of `HUMAN_GIVE_COMMAND`, `GET_MEMORY`, `PUT_MEMORY` and `NOOP` dialogue types.

The tool autocompletes children for tree nodes based on matches in the filters spec on key expansion, indicated by `<key>: [space] [space] [Enter]`.

For dialogue type autocomplete, type the dialogue type in lower case, eg. `"get_memory":[space][space]<enter>` -->
```
{
    "dialogue_type": "GET_MEMORY",
    "filters": "",
    "replace": ""
}
```

This also applies to action types, eg. eg. `{ "move":[space][space]<enter> }` -->
```
{
    "move": {
        "location": "",
        "replace": "",
        "action_type": "",
        "stop_condition": "",
        "remove_condition": ""
    }
}
```

For all other keys, type the key value and `[space][space]<enter>`, eg.
```
{
    "dialogue_type": "GET_MEMORY",
    "filters": [space][space]<enter>,
    "replace": ""
}
```
will autocomplete to
```
{
    "dialogue_type": "GET_MEMORY",
    "filters": {
        "output": "",
        "contains_coreference": "",
        "memory_type": "",
        "comparator": "",
        "triples": "",
        "author": "",
        "location": "",
        "repeat": "",
        "selector": ""
    },
    "replace": ""
}
```

Note that there is a comma after "filters", which ensures that the subtree inserted by autocomplete will complete to correct JSON. Otherwise, this will not work.

If labelling fragments, you need to start with an empty dictionary, eg.
```
{"filters":[space][space]<enter>}
```
instead of `"filters":[space][space]<enter>`. Again, this is to ensure correct JSON.

The tool also populates triples with the first array item's keys pre-filled.

`"triples": [ { "pred_text": "", "obj_text": "", "subj_text": "" }, \n`

To pretty print a JSON valid dictionary, press Enter in the text box.

## Save and Upload
On `Save Annotations`, the current command and parse tree are saved to `command_dict_pairs.json` in `~/droidlet/tools/annotation_tools/template_tool/frontend/src/`.

On `Create Dataset from Annotations`, the new data pairs in `~/droidlet/tools/annotation_tools/template_tool/frontend/src/command_dict_pairs.json` are first postprocessed into the format required for droidlet NLU components (fill span ranges, remove empty keys), using `~/droidlet/tools/data_processing/autocomplete_postprocess.py`. 

By default, the results are written to a new file under `~/droidlet/craftassist/agent/datasets/full_data/autocomplete_<DATE>.txt`. If you want to overwrite an existing data file, you need to specify the source file and output path in args, eg.

```
python3 /private/home/rebeccaqian/droidlet/tools/data_processing/autocomplete_postprocess.py --existing_annotations annotated.txt --output_file annotated_latest.txt
```

This above command would replace the commands in `annotated.txt` that have been relabelled, and write the new annotated dataset to `annotated_latest.txt`.

Final data is in the format

```
[command]|[action_dict]\n
[command]|[action_dict]\n
...
```

Then, you may rename and upload this file to the S3 URI `s3://craftassist/pubr/`. You can also submit a PR to update the datasets. It is now ready to be used in training, validation or ground truth actions!

