# NSP Scripts

This directory contains tools used to query and evaluate NSP models. `data_processing_scripts` contains scripts used to prepare and update datasets used by NLU components in Droidlet.

## Data Processing Scripts
This is a suite of data processing scripts that are used to process datasets for training the semantic parser and ground truth lookup at agent runtime. This includes
- `process_templated_for_gt.py`: Deduplicates templated datasets for use in ground truth.
- `create_annotated_split.py`: Creates a train/test/valid split of a data type, i.e. annotated or templated.
- `generate_templated_data.py`: Generates templated training data using the generation script.
- `update_annotated_split.py`: Updates the data split with updated action dictionaries.