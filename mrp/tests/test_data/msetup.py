import mrp

mrp.process(
    name="proc1",
    runtime=mrp.Host(
        run_command=["echo", "data", ">", "./file1.txt"],
    ),
)

mrp.process(
    name="proc2",
    runtime=mrp.Host(
        run_command=["echo", "data", ">", "./file2.txt"],
    ),
)

if __name__ == "__main__":
    mrp.main()
