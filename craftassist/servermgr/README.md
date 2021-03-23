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
<Commit SHA1>: digest: <DIGEST>: size: xxxx
```

Notice that tag is the SHA1 of the latest master commit

2. Run [tools/docker/promote.sh](https://github.com/facebookresearch/droidlet/blob/main/tools/docker/promote.sh) using the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY credentials like this:

```
AWS_ACCESS_KEY_ID="key id here" AWS_SECRET_ACCESS_KEY="secret access key here" ./tools/docker/promote.sh <DIGEST from above>
```

Replacing the <DIGEST from above> with whatever digest you'd like to promote


## How to deploy a new servermgr

These are instructions for updating `craftassist.io`.

### Set Up

Ensure that you have the Heroku credentials for `servermgr`, and Heroku CLI installed. To set up Heroku CLI, run `heroku login -i` and enter your Heroku credentials. You should be able to see the Heroku apps under your account, including `servermgr` which runs on `craftassist.io`.

### Deploy to craftassist.io

To get the Heroku Git URL, run
```
heroku info servermgr
```

In addition to the Git URL this will show you information about the servermgr app, such as link to the web console.

You want to register this Heroku Git URL as a remote for the Droidlet repo. Inside the Droidlet repo, run 
```
git remote add servermgr <Heroku Git URL>
```

Check that your remotes were configured properly with `git remote -v`.

To deploy a new servermgr app:
1. Make changes to the code inside [../servermgr/](`~/droidlet/craftassist/servermgr/`).
2. Commit and push to your feature branch
3. Run [deploy.sh](deploy.sh). By default this deploys your working `servermgr` directory to Heroku. (Note: this script needs to be run from within the `servermgr` directory.)
