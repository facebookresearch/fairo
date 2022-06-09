import re
import pandas as pd

date_line_reg = re.compile(r'/^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$/')
COL_ST = 'start'
COL_CON = 'content'
df = pd.DataFrame(columns = [COL_ST, COL_CON])

with open('/private/home/chuxi/.hitl/tmp/20220224132033/interaction/2022-02-24T21:30:54.916839+00:00/agent.log') as file:
    block = {COL_ST: '', COL_CON: ''}

    for line in file:
        # check if starts with YYYY-MM-DD
        if re.match(r'^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])', line):
            # if the block has content & starts with trace back, append to df
            if block[COL_CON] and block[COL_CON].startswith('Traceback'):
                df = df.append(block, ignore_index = True)
            block[COL_ST] = line
            block[COL_CON] = ''
        else:
            block[COL_CON] += line

df.to_csv('/private/home/chuxi/.hitl/tmp/20220224132033/interaction/2022-02-24T21:30:54.916839+00:00/agent.log.raw.csv')
df = df.groupby(COL_CON,as_index=False).size()
print(df.head(10))
df.to_csv('/private/home/chuxi/.hitl/tmp/20220224132033/interaction/2022-02-24T21:30:54.916839+00:00/agent.log.count.csv')



