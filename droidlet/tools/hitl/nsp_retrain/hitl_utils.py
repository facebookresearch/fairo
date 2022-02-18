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
            # if re.match(fr"dashboard-{batch_id}-\d+.craftassist.io", record["name"]):
            if re.match(fr"dashboard-\d+-\d+.craftassist.io", record["name"]):
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
    
def check_account_all_hits():
    import os
    import boto3
    from datetime import datetime

    access_key = os.getenv("MTURK_AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("MTURK_AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("MTURK_AWS_REGION", default="us-east-1")

    MTURK_URL = "https://mturk-requester.{}.amazonaws.com".format(aws_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
        endpoint_url=MTURK_URL,
    )

    next_token = None

    response = mturk.list_hits()
    next_token = response['NextToken']
    all_hits = response["HITs"]
    while next_token is not None:
        response = mturk.list_hits(NextToken=next_token)
        all_hits.extend(response["HITs"])
        next_token = response['NextToken'] if 'NextToken' in response else None
    
    hit_ids = [item["HITId"] for item in all_hits]

    from datetime import datetime, timedelta
    import pytz

    now = pytz.utc.localize(datetime.now())
    prev_now = pytz.utc.localize(datetime.now() - timedelta(hours=3))
    print(f"now: {now}, prev: {prev_now}")
    # This is slow but there's no better way to get the status of pending HITs
    print(f"Total HIT Num: {len(hit_ids)}")
    this_run_hits = []
    tbd = []
    cnt = 0
    for hit_id in hit_ids:
        # Get HIT status
        hit = mturk.get_hit(HITId=hit_id)["HIT"]
        print(hit["Title"])
        if hit["Title"] == "Interact with our fun virtual assistant":
            cnt += 1
        # create_time = hit["CreationTime"]
        # if create_time > prev_now:
        #     this_run_hits.append(hit)
        #     tbd.append(hit_id)
    print(f"this run hits num: {len(this_run_hits)}")
    print(f"interaction job num left: {cnt}")


def check_all_worker_with_qual(qual_name):
    access_key = os.getenv("MTURK_AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("MTURK_AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("MTURK_AWS_REGION", default="us-east-1")

    MTURK_URL = "https://mturk-requester.{}.amazonaws.com".format(aws_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
        endpoint_url=MTURK_URL,
    )
    next_token = None
    response = mturk.list_workers_with_qualification_type(QualificationTypeId=qual_name, Status="Granted", MaxResults=100)
    workers = response["Qualifications"]
    next_token = response["NextToken"]

    while next_token is not None:
        response = mturk.list_workers_with_qualification_type(QualificationTypeId=qual_name, Status="Granted", NextToken=next_token, MaxResults=100)
        workers.extend(response["Qualifications"])
        next_token = response["NextToken"] if "NextToken" in response else None
    print(f"All worker did qual: {len(workers)}")
    worker_100 = []
    for worker in workers:
        # print(worker)
        # print("\n")
        if (worker['IntegerValue'] == 100):
            worker_100.append(worker)

    print(f"Qualfied worker num: {len(worker_100)}")


def grant_worker_with_qual(qual_name, worker_ids):
    access_key = os.getenv("MTURK_AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("MTURK_AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("MTURK_AWS_REGION", default="us-east-1")

    MTURK_URL = "https://mturk-requester.{}.amazonaws.com".format(aws_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
        endpoint_url=MTURK_URL,
    )

    for worker_id in worker_ids:
        mturk.associate_qualification_with_worker(
            QualificationTypeId=qual_name,
            WorkerId=worker_id,
            IntegerValue=100,
            SendNotification=False
        )
    
    
    

if __name__ == "__main__":
    # pass
    # check_account_all_hits()
    # check_all_worker_with_qual("32Z2G9B76CN4NO5994JO5V24P3EAXC")
    # worker_ids = [
    #     "A2JP9IKRHNLRPI",
    #     "A35C0II2FFV18S",
    #     "A1FA3QRISJ1RIP",
    #     "A1A3ML8ME7LSK",
    #     "A1JGRKKRD3ADYX",
    #     "A1ZPIPHM272LO8",
    #     "A132MSWBBVTOES",
    #     "A1SYMCOR29IWNA"
    # ]
    # grant_worker_with_qual("32Z2G9B76CN4NO5994JO5V24P3EAXC", worker_ids)
    # check_all_worker_with_qual("32Z2G9B76CN4NO5994JO5V24P3EAXC")
    for i in range(100):
        deregister_dashboard_subdomain(20211207020233)
    # examine_hit("34YWR3PJ2AD51SZWORZ4M41QBOG0XV")
