# Crowdsourcing Tools

This directory contains crowdsourcing pipelines for droidlet.

# Mephisto Setup

Complete the onboarding tutorial at:
https://github.com/facebookresearch/mephisto/blob/master/docs/quickstart.md

## Running the Static HTML workflow
`python droidlet_static_html_task/static_test_script.py`

## How to run

Run the following command to start a `droidlet_static_html_task` with `heroku` architect and `flask` server type

```
python droidlet_static_html_task/static_test_script.py mephisto.architect.server_type=flask mephisto/architect=heroku mephisto.architect.server_source_path=servermgr
```

You can replace `droidlet_static_html_task/static_test_script.py` with any mephisto task you would like to run and replace `./servermgr` with any heroku server setup you would like to use.
