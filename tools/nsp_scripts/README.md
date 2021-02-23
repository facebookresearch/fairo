# NSP Scripts

This directory contains tools used to query and evaluate NSP models. `data_processing_scripts` contains scripts used to prepare and update datasets used by NLU components in Droidlet.
- `eval_model_on_dataset.py`: Evaluate model on a CAIP dataset.
- `test_model_script.py`: Query model using beam search.

## Data Processing Scripts
This is a suite of data processing scripts that are used to process datasets for training the semantic parser and ground truth lookup at agent runtime. This includes
- `process_templated_for_gt.py`: Deduplicates templated datasets for use in ground truth.
- `create_annotated_split.py`: Creates a train/test/valid split of a data type, i.e. annotated or templated.
- `process_templated_generations_for_train.py`: Creates a train/test/valid split of templated data from templated generations created using the generation script.
- `remove_static_valid_commands.py`: Sets aside commands to be used for validation.
- `update_valid_test_sets.py`: Updates the valid splits with updated action dictionaries.