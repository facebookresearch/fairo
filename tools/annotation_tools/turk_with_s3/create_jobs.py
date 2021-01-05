import boto3
import json
import pandas as pd
import os
import csv
from datetime import datetime

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
mturk = boto3.client('mturk',
   aws_access_key_id = access_key,
   aws_secret_access_key = secret_key,
   region_name='us-east-1',
   endpoint_url = MTURK_SANDBOX
)
print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my Sandbox account")

# Delete HITs
# For use in dev only
for item in mturk.list_hits()['HITs']:
    hit_id=item['HITId']
    print('HITId:', hit_id)

    # Get HIT status
    status=mturk.get_hit(HITId=hit_id)['HIT']['HITStatus']
    print('HITStatus:', status)

    # If HIT is active then set it to expire immediately
    if status=='Assignable' or status=='Reviewable':
        response = mturk.update_expiration_for_hit(
            HITId=hit_id,
            ExpireAt=datetime(2015, 1, 1)
        )        

    # Delete the HIT
    try:
        mturk.delete_hit(HITId=hit_id)
    except:
        print('Not deleted')
    else:
        print('Deleted')

# XML file containing ExternalQuestion object.
# See MTurk API docs for constraints.
with open('step_1.xml','r') as fd:
    question = fd.read()

# Where we will save the turk job parameters
turk_jobs_df = pd.DataFrame()

with open('turk_input.csv', newline='') as csvfile:
    turk_inputs = csv.reader(csvfile, delimiter=',')
    headers = next(turk_inputs, None)
    # Construct the URL query params
    for row in turk_inputs:
        query_params = ""
        job_spec = {}
        for i in range(len(headers)):
            if not row[i]:
                continue
            value = row[i].replace(' ', '%20')
            query_params += "{}={}&amp;".format(headers[i], value)
            # Save param info to job specs
            job_spec["Input.{}".format(headers[i])] = row[i]
        curr_question = question.format(query_params)

        new_hit = mturk.create_hit(
            Title = 'Match Sentences',
            Description = 'Given an original sentence, identify whether new sentences have the same structure as the original.',
            Keywords = 'text, categorization, quick',
            Reward = '0.15',
            MaxAssignments = 1,
            LifetimeInSeconds = 600,
            AssignmentDurationInSeconds = 600,
            AutoApprovalDelayInSeconds = 14400,
            Question = curr_question,
        )
        print("A new HIT has been created. You can preview it here:")
        print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
        print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
        job_spec['HITId'] = new_hit['HIT']['HITId']

        turk_jobs_df = turk_jobs_df.append(job_spec, ignore_index=True)

turk_jobs_df.to_csv("turk_job_specs.csv", index=False)

# Remember to modify the URL above when publishing
# HITs to the live marketplace.
# Use: https://worker.mturk.com/mturk/preview?groupId=
