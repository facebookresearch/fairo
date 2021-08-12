import boto3
import pandas as pd
import xmltodict
import os
import sys
import time
import argparse


def delete_hits(mturk):
    # Check if there are outstanding assignable or reviewable HITs
    all_hits = mturk.list_hits()["HITs"]
    hit_ids = [item["HITId"] for item in all_hits]
    # This is slow but there's no better way to get the status of pending HITs
    for hit_id in hit_ids:
        # Get HIT status
        mturk.delete_hit(HITId=hit_id)

def get_hit_list_status(mturk):
    # Check if there are outstanding assignable or reviewable HITs
    all_hits = mturk.list_hits()["HITs"]
    hit_ids = [item["HITId"] for item in all_hits]
    hit_status = {
        "assignable": [],
        "reviewable": [],
    }
    # This is slow but there's no better way to get the status of pending HITs
    for hit_id in hit_ids:
        # Get HIT status
        status = mturk.get_hit(HITId=hit_id)["HIT"]["HITStatus"]
        if status == "Assignable":
            hit_status["assignable"].append(hit_id)
        elif status == "Reviewable":
            hit_status["reviewable"].append(hit_id)
    print("HITStatus: {}".format(hit_status))
    return hit_status


def get_results(mturk, output_csv: str, use_sandbox: bool):
    # This will contain the answers
    if os.path.exists(output_csv):
        res = pd.read_csv(output_csv)
    else:
        res = pd.DataFrame()
    NUM_TRIES_REMAINING = 5
    curr_hit_status = get_hit_list_status(mturk)

    while curr_hit_status["assignable"] or curr_hit_status["reviewable"]:
        if NUM_TRIES_REMAINING == 0:
            break
        print("*** Fetching results ***".format(NUM_TRIES_REMAINING))
        # get reviewable hits, ie hits that have been completed
        # hits = [x['HITId'] for x in mturk.list_reviewable_hits()['HITs']]
        # If there are no reviewable HITs currently, wait 2 mins in between tries.
        if len(curr_hit_status["reviewable"]) == 0:
            NUM_TRIES_REMAINING -= 1
            time.sleep(30)
            curr_hit_status = get_hit_list_status(mturk)
            continue

        # If there are no assignable or reviewable HITs, the job is done!
        if len(curr_hit_status["reviewable"]) == 0 and len(curr_hit_status["assignable"]) == 0:
            print("*** No HITs pending or awaiting review. Exiting.")
            sys.exit()

        # Parse responses from each reviewable HIT
        for hit_id in curr_hit_status["reviewable"]:
            new_row = {"HITId": hit_id}
            worker_results = mturk.list_assignments_for_hit(
                HITId=hit_id, AssignmentStatuses=["Submitted"]
            )
            if worker_results["NumResults"] > 0:
                for assignment in worker_results["Assignments"]:
                    new_row["WorkerId"] = assignment["WorkerId"]
                    xml_doc = xmltodict.parse(assignment["Answer"])

                    print("Worker's answer was:")
                    if type(xml_doc["QuestionFormAnswers"]["Answer"]) is list:
                        # Multiple fields in HIT layout
                        for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
                            input_field = answer_field["QuestionIdentifier"]
                            answer = answer_field["FreeText"]
                            print("For input field: " + input_field)
                            print("Submitted answer: " + answer)
                            new_row["Answer.{}".format(input_field)] = answer

                        res = res.append(new_row, ignore_index=True)
                        res.to_csv(output_csv, index=False)
                    else:
                        # One field found in HIT layout
                        answer = xml_doc["QuestionFormAnswers"]["Answer"]["FreeText"]
                        input_field = xml_doc["QuestionFormAnswers"]["Answer"]["QuestionIdentifier"]
                        print("For input field: " + input_field)
                        print("Submitted answer: " + answer)
                        new_row["Answer.{}".format(input_field)] = answer
                        res = res.append(new_row, ignore_index=True)
                        res.to_csv(output_csv, index=False)

                    mturk.approve_assignment(AssignmentId=assignment["AssignmentId"])
                    mturk.delete_hit(HITId=hit_id)
            else:
                print("No results ready yet")
                # if returned assignment is empty,reject
                mturk.delete_hit(HITId=hit_id)
        curr_hit_status = get_hit_list_status(mturk)
        print(curr_hit_status)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--dev", action="store_true")

    args = parser.parse_args()
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", default="us-east-1")

    if args.dev:
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
    get_results(mturk, args.output_csv, args.dev)
