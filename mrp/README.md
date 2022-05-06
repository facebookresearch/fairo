# Meta Robotics Platform

Deploy, launch, manage, and orchestrate heterogeneous robots with ease!

## Install

Before installing, you'll need python3+pip, conda, and docker. The later two only if using the runtime.

```sh
pip install mrp
```

## Commands

### up

Starts all defined processes, or a given subset.
```sh
# To bring up all the processes:
mrp up
# To bring up myproc:
mrp up myproc
# Add -v for verbose building print-outs:
mrp up -v myproc
```

### down
Stops all defined processes, or a given subset.
```sh
# To bring down all the processes:
mrp down
# To bring down myproc:
mrp down myproc
```

### logs
All the processes default to running in the background. To see the stdout:
```sh
# Attach to the log stream of all processes:
mrp logs
# Attach to the log stream of myproc:
mrp logs myproc
# Attach to the log stream of all processes, starting from the beginning:
mrp logs --old
```

### ps
See the state of running processes:
```sh
mrp ps
```

## Runtime

### Host

Run a process without sandboxing:
```py
mrp.process(
    name="proc",
    runtime=mrp.Host(
        run_command=["python3", "proc.py"],
    ),
)
```

### Conda

Run a process within a conda environment:
```py
mrp.process(
    name="proc",
    runtime=mrp.Conda(
        yaml="env.yml",
        run_command=["python3", "proc.py"],
    ),
)
```

The environment can be provided as a yaml file, local env name, or a list of channels and dependencies.

### Docker

```py
mrp.process(
    name="api",
    runtime=mrp.Docker(image="ghcr.io/alephzero/api:latest"),
)
```

The environment can be provided as a dockerfile or image name.

Mount points can be added with
```py
runtime=mrp.Docker(
    image="ghcr.io/alephzero/log:latest",
    mount=["/tmp/logs:/tmp/logs"],
),
```

Other kwargs passed to Docker will be passed directly to the docker engine.

## CLI Auto-Complete

Add the following snippet to your `~/.bashrc` to get tab completion:

`eval "$(_MRP_COMPLETE=bash_source mrp)"`

## For Contributers

### Code Formatting
We use [black](https://github.com/psf/black) and [flake8](https://flake8.pycqa.org/en/latest/) to lint our code. The `formatting` test will not pass if your code does not conform.

To make this easy for yourself, you can either
- Add the formattings to your IDE
- Install the git [pre-commit](https://pre-commit.com/) hooks by running
    ```bash
    pip install pre-commit
    pre-commit install
    ```

To format manually, run: `black .`
