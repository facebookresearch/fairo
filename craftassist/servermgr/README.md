## How servermgr Works

- servermgr itself is a python Flask app that runs on Heroku
- There is a Craftassist docker image
    - its Dockerfile is at [tools/docker/Dockerfile](https://github.com/facebookresearch/droidlet/blob/main/tools/docker/Dockerfile)
    - its Makefile is at [tools/docker/Makefile](https://github.com/facebookresearch/droidlet/blob/main/tools/docker/Makefile)
    - it is remotely stored using AWS ECR.

- When a servermgr user hits the big green button to launch a server, servermgr
  makes a request to AWS ECS to launch a container from this image, using this
  script: [craftassist/servermgr/run.withagent.sh](https://github.com/facebookresearch/droidlet/blob/main/craftassist/servermgr/run.withagent.sh) which
  launches a Cuberite server, launches an agent, waits for the end of the session, then
  bundles up the workdir and copies it to S3

## How To Deploy a new Craftassist Bot

### Background Info

- On every successful CircleCI run on master, a docker image is pushed to ECR
  and tagged with the master commit SHA1, see the "Push versioned docker
  containers" step in the CircleCI config at [.circleci/config.yml](https://github.com/facebookresearch/droidlet/blob/main/.circleci/config.yml)
- servermgr always deploys the image with the `latest` tag
- To cause servermgr to use a newer commit, the versioned docker image pushed
  by CircleCI must be tagged `latest`. No changes to the servermgr codebase are
  necessary.

### Actual How To

1. Verify that the commit passed CI successfully. If all is green, you should see under the "Push versioned docker containers" step a line like

```
docker push ECR_image_URI:tag
```

Notice that tag is the SHA1 of the latest master commit

2. Run [tools/docker/promote.sh](https://github.com/facebookresearch/droidlet/blob/main/tools/docker/promote.sh) using the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY credentials like this:

```
AWS_ACCESS_KEY_ID="key id here" AWS_SECRET_ACCESS_KEY="secret access key here" ./tools/docker/promote.sh <tag from above>
```

Replacing the <tag from above> with whatever commit you'd like to promote


## How to deploy a new servermgr

0. Do once ever: run `heroku login` and enter the credentials
and then run: 
```
git remote add <name of heorku git> <link to heroku git>
```
1. Make changes to the code at [app.py](app.py)
2. Commit and push to master
3. Run [deploy.sh](deploy.sh)
