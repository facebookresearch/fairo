import a0
import boto3
import os
import signal
import queue
import threading
import time
import logging
boto3.set_stream_logger('boto3.client', logging.CRITICAL)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


logpath = a0.cfg(a0.env.topic(), "/logpath", str)
cloud = a0.cfg(a0.env.topic(), "/cloud", str)
bucket = a0.cfg(a0.env.topic(), "/bucket", str)
prefix = a0.cfg(a0.env.topic(), "/prefix", str)
credential = a0.cfg(a0.env.topic(), "/credential", str)

if cloud != "aws":
    print("Only cloud=aws is supported at the moment.", file=sys.stderr)
    sys.exit(1)

cloud_credential={}
with open(str(credential), 'r') as f:
    for line in f.readlines():
        if 'AWS_' in line:
            k, v=line.split()[1].split('=')[:2]
            cloud_credential[k]=v.strip('\'').strip('"')

job_queue = queue.Queue()

def uploader():
    # create individual session for each thread for threadsafety
    session = boto3.session.Session(
                aws_access_key_id=cloud_credential['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=cloud_credential['AWS_SECRET_ACCESS_KEY'],
                aws_session_token=cloud_credential['AWS_SESSION_TOKEN'])
    s3 = session.client('s3')
    while True:
        (path, dst) = job_queue.get()
        print(f'starting {path} -> {dst}')
        s3.upload_file(Filename=path, Bucket=str(bucket), Key=dst)
        print(f'uploaded {path} -> {dst}')
    

num_worker = 1
for i in range(num_worker):
    threading.Thread(target=uploader).start()


def onlogfile(path):
    if os.path.basename(path).startswith("."):
        return

    relpath = os.path.relpath(path, str(logpath))
    dst = os.path.join(str(prefix), relpath)
    print(f'enque {path} -> {dst}')
    job_queue.put((path, dst))


def main():
    # Connect to AWS.

    pattern = os.path.join(str(logpath), "**", "*.a0")
    d = a0.Discovery(pattern, onlogfile)
    while True:
        time.sleep(10)
        print(f'queue size {job_queue.qsize()}')
    signal.pause()


if __name__ == "__main__":
    main()
