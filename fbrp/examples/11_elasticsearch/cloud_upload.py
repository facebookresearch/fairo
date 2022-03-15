import a0
import boto3
import os
import signal
import sys


logpath = a0.cfg(a0.env.topic(), "/logpath", str)
cloud = a0.cfg(a0.env.topic(), "/cloud", str)
bucket = a0.cfg(a0.env.topic(), "/bucket", str)

if cloud != "aws":
    print("Only cloud=aws is supported at the moment.", file=sys.stderr)
    sys.exit(1)


def onlogfile(path):
    if os.path.basename(path).startswith("."):
        return

    relpath = os.path.relpath(path, str(logpath))
    dst = os.path.join(str(bucket), relpath)

    print(dst)
    # Upload to AWS.


def main():
    # Connect to AWS.

    pattern = os.path.join(str(logpath), "**", "*.a0")
    d = a0.Discovery(pattern, onlogfile)
    signal.pause()


if __name__ == "__main__":
    main()
