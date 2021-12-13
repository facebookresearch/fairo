## How to run the pipeline end-to-end

### Step 1: set up environment variables for credentials (e.g. AWS, Mturk, Cloudflare)

Here's a list of environment variables you should set before you run the pipeline:
1. MEPHISTO_AWS_ACCESS_KEY_ID
2. MEPHISTO_AWS_SECRET_ACCESS_KEY
3. MEPHISTO_REQUESTER
4. AWS_ACCESS_KEY_ID
5. AWS_SECRET_ACCESS_KEY
6. AWS_DEFAULT_REGION
7. CLOUDFLARE_TOKEN
8. CLOUDFLARE_ZONE_ID
9. CLOUDFLARE_EMAIL
10. MTURK_AWS_ACCESS_KEY_ID
11. MTURK_AWS_SECRET_ACCESS_KEY

1-2 is the AWS credentials you provided to Mephisto for mturk access purpose. In this pipeline, Mephisto will create Interaction Job HITs using these credentials. 
3 is the name of Mephisto requester you registered  
4-6 is the AWS info we use to spin up ECS instances for hosting dashboard, you should provide one with ECS,ECR access permission  
7-9 is the cloudflare info which we use to register domain names for dashboard urls  
10-11 is the AWS credentials we use to spin up Annotation Jobs through AWS mturk.

Note that both Interaction Job and Annotation Jobs are AWS mturk HITs. 1-2 & 10-11 would be the same if you are using the same AWS IAM users to spin up those jobs. You can also use different AWS IAM users to spin up those jobs and in that case, 1-2 and 10-11 would be different.

### Step 2: Tag the docker image and promote to ECR

For Interaction Job, we are serving a specific version of dashboard on ECS instances which are built using one of the docker images we promoted to ECR from each CI run. Please see [here](https://github.com/facebookresearch/fairo/tree/main/tools/servermgr#background-info) for some background info.

Here, you should:

1. First find the SHA1 of that commit (in CI you should see something like: docker push ECR_image_URI:<COMMIT SHA1>)
2. Tag that digest with a spcific TAG_NAME by running: `./tools/docker/promote.sh <COMMIT SHA1> <TAG_NAME>`


### Step 3: run the pipeline with one click
first change your directory:
```
cd droidlet/tools/hitl/nsp_retrain
```

then start the runner with:
```
python main.py --interaction_job_num 2 --image_tag <TAG_NAME> --task_name <TASK_NAME> --resample
```

Note that we are providing the following parameters:
- `interaction_job_num` specifies how many interaction jobs we would like to spin up, i.e. how many dashboard sessions we are creating for turk workers.
- `image_tag` you should provide <TAG_NAME> mentioned in step 2. It will specify which dashboard version (i.e. which commit) is used.
- `task_name` you can provide any name here. The ECS instances will be named after this.
- `resample` it specifies where or not we are retraining the model on resampling mode. The default mode is non-resampling, meaning that the model will be retrained using old+new data and evaluated only on old valid/test data. Resampling mode means all the old and new data will be combined and randomly splited into train/valid/test dataset.


you can also provide environment variables mentioned in step#1 here by running command like:
```
MEPHISTO_AWS_ACCESS_KEY_ID="xxxxx" MEPHISTO_AWS_SECRET_ACCESS_KEY="xxxx" .... python main.py --interaction_job_num 2 --image_tag <TAG_NAME> --task_name <TASK_NAME> --resample
```