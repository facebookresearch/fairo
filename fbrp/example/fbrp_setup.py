from facebook_robotics_platform import setup as fbrp
from mycamera import fbrp_setup  # defines "mycamera"
from myimu import fbrp_setup  # defines "myimu"

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
