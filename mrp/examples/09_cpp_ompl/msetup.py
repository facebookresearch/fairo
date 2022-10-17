import mrp

mrp.process(
    name="proc",
    runtime=mrp.Conda(
        channels=[
            "conda-forge",
            "robostack",
        ],
        dependencies=[
            "python>=3.7",
            "ros-foxy-ompl",
        ],
        setup_commands=[
            [
                "g++",
                "-o",
                "proc",
                mrp.NoEscape("-I${CONDA_PREFIX}/include"),
                mrp.NoEscape("-I${CONDA_PREFIX}/include/eigen3"),
                mrp.NoEscape("-I${CONDA_PREFIX}/include/ompl-1.5"),
                "./proc.cpp",
                mrp.NoEscape("-L${CONDA_PREFIX}/lib"),
                "-lompl",
            ],
        ],
        run_command=[mrp.NoEscape("LD_LIBRARY_PATH=${CONDA_PREFIX}/lib"), "./proc"],
    ),
)

if __name__ == "__main__":
    mrp.main()
