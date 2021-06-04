"""
This script collects S3 logs from Turk and postprocesses them
"""

import glob
import pandas as pd
import re
from datetime import datetime
import argparse

# TURK_LOGS_DIRECTORY = "~/s3_logs"
# TURK_OUTPUT_DIRECTORY = "~/s3_logs/processed_logs/"

pd.set_option('display.max_rows', 10)

def read_turk_logs(turk_logs_directory, turk_output_directory):
    # Crawl turk logs directory
    all_turk_interactions = None

    for csv_path in glob.glob('{turk_logs_dir}/**/{csv_filename}'.format(turk_logs_dir=turk_logs_directory, csv_filename='nsp_outputs.csv')):
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
    out_path = "{output_dir}/nsp_outputs_{timestamp}.csv".format(output_dir=turk_output_directory, timestamp=datetime.now().strftime("%Y-%m-%d"))
    all_turk_interactions.to_csv(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--turk_logs_directory', help='where to read s3 logs from')
    parser.add_argument('--turk_output_directory', help='where to write the collated NSP outputs')
    args = parser.parse_args()

    read_turk_logs(args.turk_logs_directory, args.turk_output_directory)
    