import mrp

mrp.process(
    name="alice",
    runtime=mrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "alice.py"],
    ),
)

if __name__ == "__main__":
    mrp.main()
