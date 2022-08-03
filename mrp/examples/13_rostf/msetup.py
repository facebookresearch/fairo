import mrp

mrp.process(
    name="rostf",
    runtime=mrp.Conda(
        channels=[
            "conda-forge",
            "robostack",
        ],
        dependencies=[
            "ros-noetic-tf2-ros",
        ],
        run_command=["python", "rostf.py"],
    ),
)

if __name__ == "__main__":
    mrp.main()
