import re
import os
import pandas as pd

COL_CONTENT = "content"
COL_FREQ = "freq"


def process_log(path: str):
    # get log files in the path
    for fname in os.listdir(path):
        if fname.endswith(".log"):
            # found a log file
            fpath = os.path.join(path, fname)
            df = pd.DataFrame(columns=[COL_CONTENT, COL_FREQ])
            df = df.set_index(COL_CONTENT)

            with open(fpath) as file:
                content = ""
                for line in file:
                    # check if starts with YYYY-MM-DD
                    # or line starts with logging level
                    if (
                        re.match(r"^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])", line)
                        or line.startswith("DEBUG")
                        or line.startswith("INFO")
                        or line.startswith("WARNING")
                        or line.startswith("ERROR")
                        or line.startswith("CRITICAL")
                    ):
                        # if the content exists & starts with trace back, append to df
                        if content and content.startswith("Traceback"):
                            if content not in df.index:
                                df.loc[content] = 0
                            df.loc[content] += 1
                        content = ""
                    else:
                        content += line

            if len(df) > 0:
                # Dedup based on content column and save
                df.to_csv(f"{fpath}.traceback.csv")


process_log(
    "/private/home/chuxi/.hitl/tmp/20220224132033/interaction/2022-02-24T21:20:35.330930+00:00"
)
