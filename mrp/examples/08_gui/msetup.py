import mrp
import os

mrp.process(
    name="conda_proc",
    runtime=mrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "proc.py"],
    ),
    env={
        "my_conda_path": "/tmp/",
        "my_conda_time_out": "2",
    },
)

mrp.process(
    name="docker_x11_proc",
    runtime=mrp.Docker(
        dockerfile="./Dockerfile",
        mount=["/tmp/.X11-unix:/tmp/.X11-unix"],
        run_kwargs={
            "Env": [f"DISPLAY={os.getenv('DISPLAY')}"],
        },
    ),
    env={
        "my_docker_path": "/tmp/",
        "my_docker_time_out": "2",
    },
)

mrp.process(
    name="docker_opengl_proc",
    runtime=mrp.Docker(
        dockerfile="./Dockerfile.opengl",
        mount=["/tmp/.X11-unix:/tmp/.X11-unix"],
        run_kwargs={
            "Env": [f"DISPLAY={os.getenv('DISPLAY')}"],
            "HostConfig": {"Runtime": "nvidia"},  #  default "runc"
        },
    ),
)

mrp.main()
