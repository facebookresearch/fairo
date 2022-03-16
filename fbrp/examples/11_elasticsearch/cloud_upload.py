import a0
import boto3
import os
import signal


logpath = a0.cfg(a0.env.topic(), "/logpath", str)
cloud = a0.cfg(a0.env.topic(), "/cloud", str)
bucket = a0.cfg(a0.env.topic(), "/bucket", str)
prefix = a0.cfg(a0.env.topic(), "/prefix", str)
credentail = a0.cfg(a0.env.topic(), "/credentail", str)

if cloud != "aws":
    print("Only cloud=aws is supported at the moment.", file=sys.stderr)
    sys.exit(1)

cloud_credential={}
with open(str(credentail),'r') as f:
    for line in f.readlines():
        if 'AWS_' in line:
            k, v=line.split()[1].split('=')[:2]
            cloud_credential[k]=v.strip('\'').strip('"')

s3 = boto3.resource('s3',
                       aws_access_key_id=cloud_credential['AWS_ACCESS_KEY_ID'],
                       aws_secret_access_key=cloud_credential['AWS_SECRET_ACCESS_KEY'],
                       aws_session_token=cloud_credential['AWS_SESSION_TOKEN'])
bucket = s3.Bucket(str(bucket))

def onlogfile(path):
    if os.path.basename(path).startswith("."):
        return

    relpath = os.path.relpath(path, str(logpath))
    dst = os.path.join(str(prefix), relpath)
    print(dst)
    # Upload to AWS.
    bucket.upload_file(Filename=path, Key=dst)


def main():
    # Connect to AWS.

    pattern = os.path.join(str(logpath), "**", "*.a0")
    d = a0.Discovery(pattern, onlogfile)
    signal.pause()


if __name__ == "__main__":
    main()
