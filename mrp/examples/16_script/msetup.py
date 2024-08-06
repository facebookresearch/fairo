import mrp

mrp.process(
    name="redis",
    runtime=mrp.Docker(image="redis"),
)

mrp.process(
    name="set_foo",
    runtime=mrp.Conda(
        channels=["conda-forge"],
        dependencies=["redis-py"],
        run_command=["python", "set_foo.py"],
    ),
)

mrp.process(
    name="get_foo",
    runtime=mrp.Conda(
        channels=["conda-forge"],
        dependencies=["redis-py"],
        run_command=["python", "get_foo.py"],
    ),
)

if __name__ == "__main__":
    mrp.main()
