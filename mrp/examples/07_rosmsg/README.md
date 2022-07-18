# Example 07: ROS Msg

In this example we split off the traditional `msetup.py` into two files: `msetup.py` and `frun.py`.

`msetup.py` will be used to compile custom ros messages into Python libraries.

`frun.py` will run `proc`.

To run them in order,
```sh
mrp up && \
mrp wait && \
mrp -f frun.py up
```

To see the rosmsg in use:
```sh
mrp -f frun.py logs
```

To kill the running `proc`:
```sh
mrp -f frun.py down
```
