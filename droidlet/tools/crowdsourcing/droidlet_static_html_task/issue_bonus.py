#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser
from mephisto.data_model.worker import Worker

def main(run_id, bonus_rate, performance_dir) -> list:
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, help="integer mephisto run ID", required=True)
    parser.add_argument("--bonus_rate", type=float, help="dollars per performance point", required=True)
    parser.add_argument("--performance_dir", type=str, help="directory containing 'performance_scores.csv'", required=True)
    args = parser.parse_args()
    
    main(args.run_id, args.bonus_rate, args.performance_dir)