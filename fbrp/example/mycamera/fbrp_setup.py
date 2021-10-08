from facebook_robotics_platform import setup as fbrp

fbrp.process(
    name="mycamera",
    runtime=fbrp.Docker(dockerfile="./Dockerfile"),
)

if __name__ == "__main__":
    fbrp.main()
