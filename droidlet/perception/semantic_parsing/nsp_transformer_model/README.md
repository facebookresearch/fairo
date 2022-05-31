# Training Semantic Parsing Models

### Setup

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

### Parser Training Instructions

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
