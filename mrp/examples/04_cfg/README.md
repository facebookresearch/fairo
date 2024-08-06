# Example 04: Config

In this example we show how to define externally controllable variables.

In `msetup.py`, `proc` defines a variable named `prefix`, which is initialized with `"In the beginning... "`.

In `proc.py`, this variable is declared as a string:
```py
prefix = a0.cfg(a0.env.topic(), "/prefix", str)
```

We use the same alephzero/api image as the last example to allow easy edit from a web-ui.

Launch all the processes as before:
```sh
mrp up
```

Now open your browser to `file:///path/to/mrp/examples/04_api/cfgedit.html`

(make sure to update the path to the actual filepath on your machine)

You should see the different processes and their config variables and values. Change the value of `proc`'s prefix and press commit.

You'll see the new prefix by checking the logs:
```sh
mrp logs --old
```

Finally, to stop all processes:
```sh
mrp down
```
