"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import csv


def any_two(a, b, c):
    return a == b or a == c or b == c


with open("Batch_3449808_batch_results.csv", "r") as f:
    r = csv.DictReader(f)
    r = [d for d in r]

whittled = [
    {k: v for k, v in d.items() if (k.startswith("Answer.") or k == "Input.command") and v != ""}
    for d in r
]

with open("results.tmp", "r") as f:
    processed = f.readlines()

assert len(processed) == len(whittled)

faulty_processed_idxs = []
for i in range(181):
    if not any_two(processed[3 * i], processed[3 * i + 1], processed[3 * i + 2]):
        print(i)
        print(whittled[3 * i])
        # print(processed[3*i], processed[3*i+1], processed[3*i+2], '', sep='\n')
        faulty_processed_idxs.append(i)

# for i in faulty_processed_idxs:
#     print(whittled[3*i], whittled[3*i], whittled[3*i], '', sep='\n')
