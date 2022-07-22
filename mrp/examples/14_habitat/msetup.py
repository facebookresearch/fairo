import mrp

mrp.process(
    name="viewer",
    runtime=mrp.Conda(
        channels=["conda-forge", "aihabitat"],
        dependencies=["habitat-sim"],
        setup_commands=[
            [
                "python3",
                "-m",
                "habitat_sim.utils.datasets_download",
                "--uids",
                "replica_cad_dataset",
                "--no-replace",
            ],
        ],
        run_command=[
            "habitat-viewer",
            "--dataset",
            "data/replica_cad/replicaCAD.scene_dataset_config.json",
            "--",
            "apt_1",
        ],
    ),
)

if __name__ == "__main__":
    mrp.main()
