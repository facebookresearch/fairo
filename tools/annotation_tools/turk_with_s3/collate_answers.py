"""
Collate results from turk output and input
"""

import pandas as pd

# load Turk inputs CSV
input_data = pd.read_csv("turk_job_specs.csv")
# load Turk outputs CSV
output_data = pd.read_csv("turk_output.csv")
merged_inner = pd.merge(left=input_data, right=output_data, left_on="HITId", right_on="HITId")
merged_inner.to_csv("processed_outputs.csv", index=False)
