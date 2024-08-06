import mrp

mrp.process(
    name="py38",
    runtime=mrp.Conda(
        dependencies=["python=3.8"],
        run_command=["python"],
    ),
)

mrp.process(
    name="py39",
    runtime=mrp.Conda(
        dependencies=["python=3.9"],
        run_command=["python"],
    ),
)

mrp.process(
    name="u20",
    runtime=mrp.Docker(image="ubuntu:20.04"),
)

mrp.process(
    name="u22",
    runtime=mrp.Docker(image="ubuntu:22.04"),
)

mrp.process(
    name="dpy38",
    runtime=mrp.Docker(image="python:3.8"),
)

mrp.process(
    name="dpy39",
    runtime=mrp.Docker(image="python:3.9"),
)

if __name__ == "__main__":
    mrp.main()
