# Example 07: ROS Msg

In this example we split off the traditional `fsetup.py` into two files: `fsetup.py` and `frun.py`.

`fsetup.py` will be used to compile custom ros messages into Python libraries.

`frun.py` will run `proc`.

To run them in order,
```sh
fbrp -v up && \
fbrp wait && \
fbrp -f frun.py -v up
```

To see the rosmsg in use:
```sh
fbrp -f frun.py logs
```

To kill the running `proc`:
```sh
fbrp -f frun.py down
```
