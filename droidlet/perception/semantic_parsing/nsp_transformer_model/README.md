# Training Semantic Parsing Models

## Setup

Training code for semantic parsing models is in
```
train_model.py
```

Depending on your GPU driver version, you may need to downgrade your pytorch and CUDA versions. To update your conda env with these versions, run
```
conda create -n droidlet_env python==3.7 pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 tensorboard -c pytorch
conda activate droidlet_env
```

For a list of pytorch and CUDA compatible versions, see:
https://pytorch.org/get-started/previous-versions/

## Parser Training Instructions

First, if you don't have dataset, you will need to run
```
$ python droidlet/tools/artifact_scripts/fetch_artifacts_from_aws.py --agent_name craftassist --artifact_name datasets --checksum_file datasets.txt
```

Then, to train NLU model, run
```
$ python droidlet/perception/semantic_parsing/nsp_transformer_model/train_model.py --batch_size 32 --data_dir droidlet/artifacts/datasets/annotated_data/ --dtype_samples 'annotated:1.0' --tree_voc_file droidlet/artifacts/models/nlu/ttad_bert_updated/caip_test_model_tree.json --output_dir $CHECKPOINT_PATH
```
Remember to modfiy ```CHECKPOINT_PATH``` to the directory where you want to store all saved models.

Feel free to experiment with the model parameters. The models and tree vocabulary files are saved under $CHECKPOINT_PATH, along with a log that contains training and validation accuracies after every epoch. Once you're done, you can choose which epoch you want the parameters for, and use that model.
```
$ cp $PATH_TO_BEST_CHECKPOINT_MODEL droidlet/artifacts/models/nlu/caip_test_model.pth
```

You can now use that model to run the agent.

## Parser Evaluating Instructions

We support interative way to evaluate or query semantic parser via ipython,
```
ipython
from droidlet.perception.semantic_parsing.nsp_transformer_model.test_model_script import *
```
Then run the following to parse input arguments, build the model, tokenzier and dataset,
```
model, tokenizer = model_configure(args)
dataset = dataset_configure(args, tokenizer)
```
For query model, you can run
```
query_model("hello", args, model, tokenizer, dataset)
```
For evaluate model, you can run
```
eval_model(args, model, tokenizer, dataset)
```

## List of scripts
1. [train_model.py](./train_model.py) - The main training script for NLU model.
2. [test_model_script.py](./test_model_script.py) - The evaluation script for NLU model, which supports query and evaluate modes.
3. [caip_dataset.py](./caip_dataset.py) - The CAIP dataset definition for NLU model.
4. [decoder_with_loss.py](./decoder_with_loss.py) - The definition of decoder part of NLU model.
5. [encoder_decoder.py](./encoder_decoder.py) - The definition of encoder-decoder model of NLU.
6. [modeling_bert.py](./modeling_bert.py) - The customized bert related modules.
7. [label_smoothing_loss.py](./label_smoothing_loss.py) - The definition of label smoothing loss.
8. [optimizer_warmup.py](./optimizer_warmup.py) - Custom wrapper for adam optimizer with warmup training.
9. [tokenization_utils.py](./tokenization_utils.py) - Dictionary between span and values.
10. [utils_caip.py](./utils_caip.py) - Utility for caip dataset.
11. [utils_model.py](./utils_model.py) - Utility for NLU model.
12. [utils_parsing.py](./utils_parsing.py) - Utility for semantic parsing. 
13. [query_model.py](./query_model.py) - The definition of NLU query model.

## Data Processing Scripts
This is a suite of data processing scripts that are used to process datasets for training the semantic parser and ground truth lookup at agent runtime. This includes
1. [process_templated_for_gt.py](./process_templated_for_gt.py) - Deduplicates templated datasets for use in ground truth.
2. [create_annotated_split.py](./create_annotated_split.py) - Creates a train/test/valid split of a data type, i.e. annotated or templated.
3. [process_templated_generations_for_train.py](./process_templated_generations_for_train.py) - Creates a train/test/valid split of templated data from templated generations created using the generation script.
4. [remove_static_valid_commands.py](./remove_static_valid_commands.py): Sets aside commands to be used for validation.
5. [update_valid_test_sets.py](./update_valid_test_sets.py) - Updates the valid splits with updated action dictionaries.

