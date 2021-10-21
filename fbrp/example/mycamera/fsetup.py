import fbrp

fbrp.process(
    name="mycamera",
    runtime=fbrp.Docker(dockerfile="./Dockerfile"),
)

if __name__ == "__main__":
    fbrp.main()


fbrp.process(
   name="cam1_downsampler",
   runtime=fbrp.Conda(
       yaml="/path/to/env.yml",
       run_command=["python3", "myimu.py"],
   ),
   deps=["cam1"],
)

