"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import os
import re

import CloudFlare
import boto3


def generate_batch_id():
    import datetime

    dt = datetime.datetime.now()
    return int(dt.strftime("%Y%m%d%H%M%S"))


def deregister_dashboard_subdomain(batch_id):
    if (
        os.getenv("CLOUDFLARE_TOKEN")
        and os.getenv("CLOUDFLARE_ZONE_ID")
        and os.getenv("CLOUDFLARE_EMAIL")
    ):
        logging.info("Deresigister subdomains on craftassist.io")
        cf_token = os.getenv("CLOUDFLARE_TOKEN")
        zone_id = os.getenv("CLOUDFLARE_ZONE_ID")
        cf_email = os.getenv("CLOUDFLARE_EMAIL")
        cf = CloudFlare.CloudFlare(email=cf_email, token=cf_token)
        dns_records = cf.zones.dns_records.get(zone_id)
            
        for record in dns_records:
            print(f'{record["name"]} pattern : {batch_id}')
            if re.match(fr"dashboard-{batch_id}-\d+.craftassist.io", record["name"]):
                print(f"matched cf record to be deleted: {record['name']}")
                cf.zones.dns_records.delete(zone_id, record["id"])
                logging.debug(f'Deleted cf dns record: {record["name"]}')
                print(f'Deleted cf dns record: {record["name"]}')

def dedup_commands(command_list):
    cmd_set = set()
    deduped_cmd_list = []
    for command in command_list:
        if command.lower() not in cmd_set:
            cmd_set.add(command.lower())
            deduped_cmd_list.append(command)
    return deduped_cmd_list

def examine_hit(hit_id):
    access_key = os.getenv("MTURK_AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("MTURK_AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("MTURK_AWS_REGION", default="us-east-1")
    dev_flag = None
    if dev_flag:
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

    worker_results = mturk.list_assignments_for_hit(
        HITId=hit_id, AssignmentStatuses=["Submitted"]
    )
    print(worker_results["NumResults"])
    print(worker_results["Assignments"])


def delete_all_mturk_hits():
    import os
    import boto3
    from datetime import datetime

    access_key = os.getenv("MTURK_AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("MTURK_AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("MTURK_AWS_REGION", default="us-east-1")

    MTURK_URL = "https://mturk-requester-sandbox.{}.amazonaws.com".format(aws_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
        endpoint_url=MTURK_URL,
    )

    all_hits = mturk.list_hits()["HITs"]
    hit_ids = [item["HITId"] for item in all_hits]
    # This is slow but there's no better way to get the status of pending HITs
    for hit_id in hit_ids:
        # Get HIT status
        status = mturk.get_hit(HITId=hit_id)["HIT"]["HITStatus"]
        try:
            response = mturk.update_expiration_for_hit(HITId=hit_id, ExpireAt=datetime(2015, 1, 1))
            mturk.delete_hit(HITId=hit_id)
        except:
            pass
        print(f"Hit {hit_id}, status: {status}")
    

if __name__ == "__main__":
    # pass
    for i in range(100):
        deregister_dashboard_subdomain(20211014214358)
    # examine_hit("34YWR3PJ2AD51SZWORZ4M41QBOG0XV")
