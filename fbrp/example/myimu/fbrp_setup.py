from facebook_robotics_platform import setup as fbrp

fbrp.process(
    name="myimu",
    runtime=fbrp.Conda(
        # You can specify an:
        #  * existing env name
        #  * a yaml description file
        #  * the channels and dependencies
        # You can also specify multiple and you'll get a merge.
        #
        # env="myenv",
        yaml="myenv.yml",
        channels=[
            "conda-forge",
            "defaults",
        ],
        dependencies=[
            "numpy",
        ],
        setup_commands=[
            ["pip", "install", "-r", "requirements.txt"],
        ],
        run_command=["python3", "myimu.py"],
    ),
)

if __name__ == "__main__":
    fbrp.main()
