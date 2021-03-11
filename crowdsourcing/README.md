# Crowdsourcing Tools

This directory contains crowdsourcing pipelines for droidlet.

## How to run

Run the following command to start a `droidlet_static_html_task` with `heroku` architect and `flask` server type

```
python droidlet_static_html_task/static_test_script.py mephisto.architect.server_type=flask mephisto/architect=heroku mephisto.architect.server_source_path=servermgr
```

You can replace `droidlet_static_html_task/static_test_script.py` with any mephisto task you would like to run and replace `./servermgr` with any heroku server setup you would like to use.
