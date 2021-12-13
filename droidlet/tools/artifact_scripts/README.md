# Scripts for handling droidlet artifacts

This folder has scripts that help keep the version of droidlet models and datasets in sync with what is 
the version on remote master.

## Fetching the artifacts
The current agent scripts ([locobot_agent](https://github.com/facebookresearch/fairo/blob/main/agents/locobot/locobot_agent.py) and [craftassist_agent](https://github.com/facebookresearch/fairo/blob/main/agents/craftassist/craftassist_agent.py)) check for inconsistency between local and 
remote and force a download if they differ. If you'd like to compare the diff manually (for all models and datasets) 
and download if they are different, do :
```
python try_download.py --agent_name <name of agent> --test_mode 
```
where `agent_name` is the name of agent you are downloading the artifacts for. Note that some perception models will be 
different for different agents like `craftassist` and `locobot`.
`test_mode` is a binary flag, and if passed, downloads the `perception_test_assets` for robot.

The above script first compares the checksum of your local artifacts folders and compares them against the checksums 
tracked in main, if they are different it will force download the updated version.


You can also force pull the remote artifact version by running:
```
python fetch_artifacts_from_aws.py --agent_name <name of agent type> --artifact_name <datasets or models> 
--model_name <nlu or perception>
```
The script above uses the agent name, artifact_name, model_name (if provided for models artifact_name) to fetch the 
right tar file from s3 and overwrites corresponding local directory.

## Uploading my local changes to artifacts
If you make changes to the any folders in [artifacts directory](https://github.com/facebookresearch/fairo/tree/main/droidlet/artifacts) 
and would like to update remote, run:
```
python upload_artifacts_to_aws.py --agent_name <name of agent type> --artifact_name <datasets or models> 
--model_name <nlu or perception>
```

The above script computes the checksum of your local artifact directory, tar's it, uploads it to S3 and updates the 
`tracked_checksums` folder with the respective file. This updated file content will need to be checked in to your PR
in order for the uploaded version to be tracked in main.

## Helpers
- `compute_checksum`: This script computes hashes for the given local artifact directories and saves them to 
  `artifact_scripts/tracked_checksums/`. The files in this folder are tracked by git, you will have to add them to your 
  PR.
  
## Tracked checksums
This folder tracks the checksums for artifacts in remote.
- `craftassist_perception.txt` : Remote checksum for `models/perception/craftassist` folder.
- `locobot_pereption.txt`: Remote checksum for `models/perception/locobot` folder.
- `nlu.txt`: Remote checksum for `models/nlu` folder.
- `datasets.txt`: Remote checksum for `artifacts/datasets` folder.