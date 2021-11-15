import fbrp
import os

fbrp.process(
    name="proc",
    runtime=fbrp.Docker(
        dockerfile="./Dockerfile",
        mount=["/tmp/.X11-unix:/tmp/.X11-unix"],
        run_kwargs = {
            "Env": [f"DISPLAY={os.environ['DISPLAY']}"],
        }
    ),
)

fbrp.main()
