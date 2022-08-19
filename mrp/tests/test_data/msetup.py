import mrp

mrp.process(
    name="proc1",
    runtime=mrp.Host(
        run_command=[],
    ),
)

mrp.process(
    name="proc2",
    runtime=mrp.Host(
        run_command=[],
    ),
)

if __name__ == "__main__":
    mrp.main()
