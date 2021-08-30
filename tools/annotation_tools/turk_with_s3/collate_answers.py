"""
Collate results from turk output and input
"""

import pandas as pd
import argparse

def collate_answers(turk_output_csv, collate_output_csv, job_spec_csv):
    # load Turk inputs CSV
    input_data = pd.read_csv(job_spec_csv, dtype=str)
    # load Turk outputs CSV
    output_data = pd.read_csv(turk_output_csv, dtype=str)
    merged_inner = pd.merge(left=input_data, right=output_data, left_on="HITId", right_on="HITId", sort=False)
    merged_inner.to_csv(collate_output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turk_output_csv", type=str, required=True)
    parser.add_argument("--job_spec_csv", type=str, required=True)
    parser.add_argument("--collate_output_csv", type=str, required=True)

    args = parser.parse_args()
    collate_answers(args.turk_output_csv, args.collate_output_csv, args.job_spec_csv)
