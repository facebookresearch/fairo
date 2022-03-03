# Example 01: Basic

This is a very basic, single process example.

`fsetup.py` defines a process named `proc`.

To launch:
```sh
fbrp -v up
```

A Conda env will be created with Python 3.7+, and `python3 proc.py` will be executed inside that environment.

The process prints a sequence number every second. To see the output:
```sh
fbrp logs
```
ctrl+c to exit log view.

To see logs from the beginning,
```sh
fbrp logs --old
```

To check if the process is running:
```sh
fbrp ps
```

Finally, to stop the process:
```sh
fbrp down
```
