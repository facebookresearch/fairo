# Training Semantic Parsing Models

### Setup

Training code for semantic parsing models is in
```
base_agent/ttad/ttad_transformer_model/train_model.py
```

Depending on your GPU driver version, you may need to downgrade your pytorch and CUDA versions. As of this writing, FAIR machines have installed NVIDIA driver version 10010, which is compatible with pytorch 1.5.1 and cudatoolkit 10.1. To update your conda env with these versions, run
```
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
```

For a list of pytorch and CUDA compatible versions, see:
https://pytorch.org/get-started/previous-versions/

### Parser Training Instructions

First, we need to pre-generate some templated data to train the model on. 500K examples should be a good start:
```
$ cd ~/minecraft/base_agent/ttad/generation_dialogues
$ python generate_dialogue.py -n 500000 > generated_dialogues.txt
```

This generates a text file. We next pre-process the data into the format required by the training script,
```
$ cd ../ttad_transformer_model/
$ python data_scripts/preprocess_templated.py \
--raw_data_path ../generation_dialogues/generated_dialogues.txt \
--output_path [OUTPUT_PATH (file must be named templated.txt)]
```

To create train/test/valid splits of the data, run
```
$ python data_scripts/create_annotated_split.py \
--raw_data_path [PATH_TO_DATA_DIR] \
--output_path [PATH_TO_SPLIT_FOLDERS] \
--filename "templated.txt" \
--split_ratio "0.7:0.2:0.1"
```

To create a split of annotated data too, simply run the above, but with filename "annotated.txt".

We are now ready to train the model with:
```
$ cd ~/minecraft
$ python base_agent/ttad/ttad_transformer_model/train_model.py \
--data_dir craftassist/agent/models/ttad_bert_updated/annotated_data/ \
--dtype_samples '[["templated", 0.35], ["templated_modify", 0.05], ["annotated", 0.6]]' \
--tree_voc_file craftassist/agent/models/ttad_bert_updated/models/caip_test_model_tree.json \
--output_dir $CHECKPOINT_PATH
```

Feel free to experiment with the model parameters. The models and tree vocabulary files are saved under $CHECKPOINT_PATH, along with a log that contains training and validation accuracies after every epoch. Once you're done, you can choose which epoch you want the parameters for, and use that model.
```
$ cp $PATH_TO_BEST_CHECKPOINT_MODEL craftassist/agent/models/caip_test_model.pth
```

You can now use that model to run the agent.
