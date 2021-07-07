"""
This script collects S3 logs from Turk and postprocesses them
"""

import glob
import pandas as pd
import re
from datetime import datetime
import argparse
import boto3
import os

pd.set_option('display.max_rows', 10)

def read_s3_bucket(s3_logs_dir):
    print('{s3_logs_dir}/**/{csv_filename}'.format(s3_logs_dir=s3_logs_dir, csv_filename='logs.tar.gz'))
    for csv_path in glob.glob('{s3_logs_dir}/**/{csv_filename}'.format(s3_logs_dir=s3_logs_dir, csv_filename='logs.tar.gz')):
        print(csv_path)
#     # session = boto3.Session(profile_name=profile_name)
#     # s3 = session.client('s3')
#     # bucket_name, key = mpu.aws._s3_path_split(source)
#     # s3_object = s3.get_object(Bucket=bucket_name, Key=key)
#     # body = s3_object['Body']
#     s3 = boto3.client('s3',
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
#     )
#     # print("Set up boto3 s3 client, attempting to download internal resources.")

#     s3.download_file(
#         'craftassist/turk_interactions_with_agent/turk',
#         'private.txt',
#         'private.txt'
#     )

def read_turk_logs(turk_logs_directory, turk_output_directory, filename):
    # Crawl turk logs directory
    all_turk_interactions = None

    for csv_path in glob.glob('{turk_logs_dir}/**/{csv_filename}'.format(turk_logs_dir=turk_logs_directory, csv_filename=filename + '.csv')):
        print(csv_path)
        with open(csv_path) as fd:
            # collect the NSP outputs CSV
            csv_file = pd.read_csv(csv_path, delimiter="|")
            # add a column with the interaction log ID
            interaction_log_id = re.search(r'\/([^\/]*)\/nsp_outputs.csv', csv_path).group(1)
            csv_file["turk_log_id"] = interaction_log_id
            if all_turk_interactions is None:
                all_turk_interactions = csv_file
            else:
                all_turk_interactions = pd.concat([all_turk_interactions, csv_file], ignore_index=True)

    # Drop duplicates
    all_turk_interactions.drop_duplicates()
    print(all_turk_interactions.head())
    print(all_turk_interactions.shape)

    # Pipe CSV outputs to another file
    out_path = "{output_dir}/{filename}_{timestamp}.csv".format(output_dir=turk_output_directory, filename=filename, timestamp=datetime.now().strftime("%Y-%m-%d"))
    all_turk_interactions.to_csv(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--turk_logs_directory', help='where to read s3 logs from')
    parser.add_argument('--turk_output_directory', help='where to write the collated NSP outputs')
    parser.add_argument('--filename', help='name of the CSV file we want to read, eg. nsp_outputs')
    args = parser.parse_args()
    read_s3_bucket(args.turk_logs_directory)
    # read_turk_logs(args.turk_logs_directory, args.turk_output_directory, args.filename)
    