# Facebook Robotics Platform

Deploy, launch, manage, and orchestrate heterogeneous robots with ease!

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
fbrp up
# To bring up myproc:
fbrp up myproc
# Add -v for verbose building print-outs:
fbrp up -v myproc
```

### down
Stops all defined processes, or a given subset.
```sh
# To bring down all the processes:
fbrp down
# To bring down myproc:
fbrp down myproc
```

### logs
All the processes default to running in the background. To see the stdout:
```sh
# Attach to the log stream of all processes:
fbrp logs
# Attach to the log stream of myproc:
fbrp logs myproc
# Attach to the log stream of all processes, starting from the beginning:
fbrp logs --old
```

### ps
See the state of running processes:
```sh
fbrp ps
```

## Runtime

### Host

Run a process without sandboxing:
```py
fbrp.process(
    name="proc",
    runtime=fbrp.Host(
        run_command=["python3", "proc.py"],
    ),
)
```

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

## CLI Auto-Complete

Add the following snippet to your `~/.bashrc` to get tab completion:

`eval "$(_FBRP_COMPLETE=bash_source fbrp)"`
