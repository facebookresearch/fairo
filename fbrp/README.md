# Facebook Robotics Platform

Deploy, launch, manage, and orchestrate heterogeneous robots with ease!

## Install

Before installing, you'll need python3+pip and docker.

```sh
pip install .
docker build -t fbrp/base .
```

## Example

The example has three dummy processes, to show off different aspects of `fsetup.py`.

* mycamera: A docker-based process.
* myimu: A conda-based process.
* myimageproc: A docker-based process, with a dependency on mycamera.

`cd` into the example and run one of:
```sh
# To bring up all the processes:
python fsetup.py up
# To bring up the image processer and the camera dependency:
python fsetup.py up myimageproc
# To bring up the image processer without any dependencies:
python fsetup.py up myimageproc --nodeps
# To rebuild the process and run replace it within the running system:
python fsetup.py up myimageproc -f --nodeps
```

All the processes default to running in the background. To see the stdout:
```sh
python fsetup.py logs
```

To bring down one process in the running system:
```
python fsetup.py down <process>
```

To bring down the whole system:
```
python fsetup.py down
```

To debug a process in pdb:
```
python fsetup.py pdb
```

## TODO:

* Remote execution
* Extend api
  * Better list
  * ...
