# Crowdsourcing Tools

This directory contains crowdsourcing pipelines for droidlet.

# Mephisto Setup

Complete the onboarding tutorial at:
https://github.com/facebookresearch/mephisto/blob/master/docs/quickstart.md

This requires you to clone the `Mephisto` repo. You may need to periodically pull updates to `Mephisto` in order to get patches and latest Mephisto features.

## Running the Static HTML workflow
To start a mock task running locally:
`python droidlet_static_html_task/static_test_script.py`

This opens up a task in your browser on localhost, and allows you to view your tasks without MTurk or Heroku credentials. 

## Running Static HTML Tasks with Heroku

### Pre-reqs
You need to have a Heroku account and valid Heroku credentials. To install Heroku CLI, run `brew tap heroku/brew && brew install heroku`. Then log in with `heroku login`.

You also need an AWS IAM account with MTurk permissions. You also need AWS access keys with permissions to spin up ECS containers, which will be used to communicate with AWS in the flask app. These may or may not be the same access keys; contact your AWS system administrator if you do not have these credentials. 

During the Mephisto onboarding tutorial, you would've configured Mephisto with your AWS credentials. Our Heroku architect pipeline also accesses environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; ensure that these are set.

### Running tasks
Run the following command to start a `droidlet_static_html_task` with `heroku` architect and `flask` server type

```
python droidlet_static_html_task/static_test_script.py mephisto.architect.server_type=flask mephisto/architect=heroku mephisto.architect.server_source_path=<path_to_server>
```

You can replace `droidlet_static_html_task/static_test_script.py` with any mephisto task you would like to run and replace `<path_to_server>` with any heroku server setup you would like to use. For `servermgr` which serves our craftassist.io backend, the code is in `~/droidlet/crowdsourcing/servermgr`.
