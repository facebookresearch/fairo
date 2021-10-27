import fbrp
from mycamera import fsetup  # defines "mycamera"
from myimu import fsetup  # defines "myimu"

fbrp.process(
    name="myimageproc",
    root="./myimageproc",
    runtime=fbrp.Docker(
        dockerfile="./myimageproc/Dockerfile",
    ),
    cfg={
        "foo": {
            "bar": "abc",
        }
    },
    deps=[
        "mycamera",
    ],
)

if __name__ == "__main__":
    fbrp.process(
        name="main",
        deps=[
            "myimageproc",
            "myimu",
        ],
    )

    fbrp.main()
