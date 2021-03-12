# Crowdsourcing Tools

This directory contains crowdsourcing pipelines for droidlet.

## How to run

Run the following command to start a `droidlet_static_html_task` with `heroku` architect and `flask` server type

```
python droidlet_static_html_task/static_test_script.py mephisto.architect.server_type=flask mephisto/architect=heroku mephisto.architect.server_source_path=servermgr
```

You can replace `droidlet_static_html_task/static_test_script.py` with any mephisto task you would like to run and replace `./servermgr` with any heroku server setup you would like to use.

## Extra configs for using servermgr

Since we are requesting ECS instance in servermgr flask app. It's required to set up aws-specific information as environment variable passed to heroku. Add the following entries to your mephisto config file:

```
mephisto:
  architect:
    heroku_app_name: YOUR_HEROKU_APP_NAME
    heroku_config_args:
      AWS_ACCESS_KEY_ID: XXX
      AWS_SECRET_ACCESS_KEY: XXX
      AWS_DEFAULT_REGION: XXX (should be "us-west-1")
```
