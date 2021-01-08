# Simple script to pull logs from sentry

## How to use

- set environment variable SENTRY_AUTH_TOKEN to the auth tokens of your sentry account
- run pull_logs.py with optional arguments:
-- org: your organization slug
-- project: project slug
-- keyword: search keyword
-- status: status of issues, can be one of [is:unresolved/is:resolved/is:ignored/is:unassigned/is:assigned]
-- save_to: output path to save retrieved results

e.g.
```
SENTRY_AUTH_TOKEN='{YOUR_SENTRY_AUTH_TOKEN}' python pull_logs.py --org='craftassist' --project='craftassist' --status='is:unresolved'--keyword='ttad_pre_coref' --save_to='./result.txt'
```
