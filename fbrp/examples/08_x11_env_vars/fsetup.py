import fbrp
import os

fbrp.process(
    name="conda_proc",
    runtime=fbrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "proc.py"],
    ),
    env={
        "my_conda_path" : "/tmp/",
        "my_conda_time_out" : "2",
    },
)

fbrp.process(
    name="docker_proc",
    runtime=fbrp.Docker(
        dockerfile="./Dockerfile",
        mount=["/tmp/.X11-unix:/tmp/.X11-unix"],
        run_kwargs = {
            "Env": [f"DISPLAY={os.getenv('DISPLAY')}"],
        }
    ),
    env={
        "my_docker_path" : "/tmp/",
        "my_docker_time_out" : "2",
    },
)

fbrp.main()
