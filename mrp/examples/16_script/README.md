# Example 16: Scripting msetup commands

All `mrp` commands are available in both CLI and script form.

In this example we demo a `launch.py` script that runs processes sequentially.

Running launch.py brings up a `redis` server, runs a command to set `foo=bar`, then runs another command to print the value of `foo`.
