import boto3
import json
import pandas as pd
import os
import csv
import argparse
import ast
from urllib.parse import quote
from datetime import datetime

def create_turk_job(xml_file_path: str, tool_num: int, input_csv: str, job_spec_csv: str, use_sandbox: bool):
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", default="us-east-1")

    if use_sandbox:
        MTURK_URL = "https://mturk-requester-sandbox.{}.amazonaws.com".format(aws_region)
    else:
        MTURK_URL = "https://mturk-requester.{}.amazonaws.com".format(aws_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
        endpoint_url=MTURK_URL,
    )
    print("I have $" + mturk.get_account_balance()["AvailableBalance"] + " in my Sandbox account")

    # Delete HITs - For use in dev only
    # NOTE: remove if not needed
    if use_sandbox:
        for item in mturk.list_hits()["HITs"]:
            hit_id = item["HITId"]
            print("HITId:", hit_id)

            # Get HIT status
            status = mturk.get_hit(HITId=hit_id)["HIT"]["HITStatus"]
            print("HITStatus:", status)

            # If HIT is active then set it to expire immediately
            if status == "Assignable" or status == "Reviewable":
                response = mturk.update_expiration_for_hit(HITId=hit_id, ExpireAt=datetime(2015, 1, 1))

            # Delete the HIT
            try:
                mturk.delete_hit(HITId=hit_id)
            except:
                print("Not deleted")
            else:
                print("Deleted")

    # XML file containing ExternalQuestion object.
    # See MTurk API docs for constraints.
    with open(xml_file_path, "r") as fd:
        question = fd.read()

    # Where we will save the turk job parameters
    # if there are existing jobs data, we will load those
    # The format is HITId followed by a list of features, eg.
    # HITId,Input.command,Input.word0,Input.word1...
    if os.path.exists(job_spec_csv) and os.path.getsize(job_spec_csv) > 0:
        turk_jobs_df = pd.read_csv(job_spec_csv)
    else:
        turk_jobs_df = pd.DataFrame()

    with open(input_csv, newline="") as csvfile:
        turk_inputs = csv.reader(csvfile, delimiter=",")
        headers = next(turk_inputs, None)
        # Construct the URL query params
        for row in turk_inputs:
            query_params = ""
            job_spec = {}
            for i in range(len(headers)):
                if not row[i]:
                    continue
                if headers[i] == "highlight_words":
                    format_row = row[i].replace("-", ",")
                    json_str = json.dumps(format_row)
                    value = quote(format_row)
                else:
                    value = row[i].replace(" ", "%20")
                query_params += "{}={}&amp;".format(headers[i], value)
                # Save param info to job specs
                job_spec["Input.{}".format(headers[i])] = row[i]
            curr_question = question.format(query_params)

            print(curr_question)
            if use_sandbox:
                new_hit = mturk.create_hit(
                    Title="CraftAssist Instruction Annotations",
                    Description="Given a sentence, provide information about its intent and highlight key words",
                    Keywords="text, categorization, quick",
                    Reward="0.3",
                    MaxAssignments=1,
                    LifetimeInSeconds=600,
                    AssignmentDurationInSeconds=600,
                    AutoApprovalDelayInSeconds=14400,
                    Question=curr_question
                )
            else:
                new_hit = mturk.create_hit(
                    Title="CraftAssist Instruction Annotations",
                    Description="Given a sentence, provide information about its intent and highlight key words",
                    Keywords="text, categorization, quick",
                    Reward="0.3",
                    MaxAssignments=1,
                    LifetimeInSeconds=600,
                    AssignmentDurationInSeconds=600,
                    AutoApprovalDelayInSeconds=14400,
                    Question=curr_question,
                    # TODO: consider making qualification configurable via JSON
                    QualificationRequirements=[{
                            'QualificationTypeId': '32Z2G9B76CN4NO5994JO5V24P3EAXC',
                            'Comparator': 'EqualTo',
                            'IntegerValues': [
                                100,
                            ],
                            'RequiredToPreview': False,
                            'ActionsGuarded': 'Accept'
                        },
                    ]
                )
            print("A new HIT has been created. You can preview it here:")
            if use_sandbox:
                print(
                    "https://workersandbox.mturk.com/mturk/preview?groupId="
                    + new_hit["HIT"]["HITGroupId"]
                )
            else:
                print(
                    "https://worker.mturk.com/mturk/preview?groupId="
                    + new_hit["HIT"]["HITGroupId"]
                )
            print("HITID = " + new_hit["HIT"]["HITId"] + " (Use to Get Results)")
            job_spec["HITId"] = new_hit["HIT"]["HITId"]

            turk_jobs_df = turk_jobs_df.append(job_spec, ignore_index=True)

    turk_jobs_df.to_csv(job_spec_csv, index=False)

    # Remember to modify the URL above when publishing
    # HITs to the live marketplace.
    # Use: https://worker.mturk.com/mturk/preview?groupId=


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_file", type=str, required=True)
    parser.add_argument("--tool_num", type=int, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--job_spec_csv", type=str, required=True)
    parser.add_argument("--dev", action="store_true")

    args = parser.parse_args()
    create_turk_job(args.xml_file, args.tool_num, args.input_csv, args.job_spec_csv, args.dev)
