# Example 15: Attach

In this example we run interactive processes.

Begin by launching the processes:
```
mrp up
```

Then attach to one, say Python 3.8, with:
```sh
mrp attach py38
```

To exit and kill the process, press ctrl+x.

To disconnect without killing the process, press ctrl+z.

You can also attach on `up`:
```sh
mrp up py38 --attach  # Python 3.8
mrp up py39 --attach  # Python 3.9
mrp up u20 --attach  # Ubuntu 20.04
mrp up u22 --attach  # Ubuntu 22.04
```
