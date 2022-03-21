import fbrp

fbrp.process(
    name="proc",
    runtime=fbrp.Conda(
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
                fbrp.NoEscape("-I${CONDA_PREFIX}/include"),
                fbrp.NoEscape("-I${CONDA_PREFIX}/include/eigen3"),
                fbrp.NoEscape("-I${CONDA_PREFIX}/include/ompl-1.5"),
                "./proc.cpp",
                fbrp.NoEscape("-L${CONDA_PREFIX}/lib"),
                "-lompl",
            ],
        ],
        run_command=[fbrp.NoEscape("LD_LIBRARY_PATH=${CONDA_PREFIX}/lib"), "./proc"],
    ),
)

fbrp.main()
