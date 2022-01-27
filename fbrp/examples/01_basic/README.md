# Example 01: Basic

This is a very basic, single process example.

`fsetup.py` defines a process named `proc`.

To launch:
```sh
python fsetup.py -v up
```

A Conda env will be created with Python 3.7+, and `python3 proc.py` will be executed inside that environment.

The process prints a sequence number every second. To see the output:
```sh
python fsetup.py logs
```
ctrl+c to exit log view.

To see logs from the beginning,
```sh
python fsetup.py logs --old
```

To check if the process is running:
```sh
python fsetup.py ps
```

Finally, to stop the process:
```sh
python fsetup.py down
```
