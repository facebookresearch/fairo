import mrp

mrp.process(
    name="grasp_server",
    runtime=mrp.Host(
        run_command=["python", "src/polygrasp/grasp_rpc.py"],
    ),
)

mrp.process(
    name="pointcloud_server",
    runtime=mrp.Host(
        run_command=["python", "src/polygrasp/pointcloud_rpc.py"],
    ),
)

mrp.main()
