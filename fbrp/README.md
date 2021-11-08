# Facebook Robotics Platform

Deploy, launch, manage, and orchestrate heteronigious robots with ease!

## Install

Before installing, you'll need python3+pip, conda, and docker. The later two only if using the runtime.

```sh
pip install fbrp
```

## Commands

### up

Starts all defined processes, or a given subset.
```sh
# To bring up all the processes:
python fsetup.py up
# To bring up myproc:
python fsetup.py up myproc
# Add -v for verbose building print-outs:
python fsetup.py up -v myproc
```

### down
Stops all defined processes, or a given subset.
```sh
# To bring down all the processes:
python fsetup.py down
# To bring down myproc:
python fsetup.py down myproc
```

### logs
All the processes default to running in the background. To see the stdout:
```sh
# Attach to the log stream of all processes:
python fsetup.py logs
# Attach to the log stream of myproc:
python fsetup.py logs myproc
# Attach to the log stream of all processes, starting from the beginning:
python fsetup.py logs --old
```

### ps
See the state of running processes:
```sh
python fsetup.py ps
```

## Runtime

### Conda

Run a process within a conda environment:
```py
fbrp.process(
    name="proc",
    runtime=fbrp.Conda(
        yaml="env.yml",
        run_command=["python3", "proc.py"],
    ),
)
```

The environment can be provided as a yaml file, local env name, or a list of channels and dependencies.

### Docker

```py
fbrp.process(
    name="api",
    runtime=fbrp.Docker(image="ghcr.io/alephzero/api:latest"),
)
```

The environment can be provided as a dockerfile or image name.

Mount points can be added with
```py
runtime=fbrp.Docker(
    image="ghcr.io/alephzero/log:latest",
    mount=["/tmp/logs:/tmp/logs"],
),
```

Other kwargs passed to Docker will be passed directly to the docker engine.
