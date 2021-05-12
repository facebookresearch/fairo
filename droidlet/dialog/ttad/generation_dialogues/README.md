## Dataset generation ##

To generate the dataset run:

```
python generate_dialogues.py -n <number of actions> -s <seed> --noops_file <file with noop data>
--action_type <name of lowercase action>
```

The script takes in the following optional parameters:
1. `-n` : You can give the number of commands that need to be generated, default is 100
2. `-s` : The seed, default is 0
3. `--noops_file` : A text file containing the data from which Noop action commands are sampled.
Default is 100K lines from the Cornell Movie Dataset.
4. `--action_type` : The lowercase name of action to be generated. Default is all.

Example commands:
```
python generate_dialogues.py -n 100000 --action_type freebuild

python generate_dialogues.py -s 2 --action_type freebuild

```
This script gives the natural language command / dialogue followed by the action tree in the next line for every generated dialogue.

## Templates ##
All the templates are in the `templates/` folder.

The file `get_template` function in `templates.py` returns a random template for the specified action.

Each action's templates can be found in the `action_name_templates.py` where `action_name` is the lowercase name
of the action, e.g. : `move_templates.py`
