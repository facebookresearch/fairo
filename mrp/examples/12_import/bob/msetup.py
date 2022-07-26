import mrp

mrp.import_msetup("../alice")

mrp.process(
    name="bob",
    runtime=mrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "bob.py"],
    ),
)

if __name__ == "__main__":
    mrp.main()
