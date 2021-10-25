import fbrp

fbrp.process(
    name="mycamera",
    runtime=fbrp.Docker(dockerfile="./Dockerfile"),
)

if __name__ == "__main__":
    fbrp.process(
        name="main",
        deps=["mycamera"],
    )

    fbrp.main()
