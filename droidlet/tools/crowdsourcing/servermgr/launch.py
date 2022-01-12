"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import boto3
import logging
import time

import ping_cuberite

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logging.getLogger().setLevel(logging.INFO)

ecs = boto3.client("ecs")
ec2 = boto3.resource("ec2")

SUBNET_IDS = ["subnet-bee9d9d9"]
SECURITY_GROUP_IDS = ["sg-04ec8fa6e1d91d460"]


def launch_task():
    """Launch one task, return task ARN"""
    r = ecs.run_task(
        cluster="craftassist",
        taskDefinition="craftassist",
        count=1,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": SUBNET_IDS,
                "securityGroups": SECURITY_GROUP_IDS,
                "assignPublicIp": "ENABLED",
            }
        },
    )
    print("LAUNCHED", r["tasks"][0])
    return r["tasks"][0]["taskArn"]


def describe_task(task_arn):
    x = ecs.describe_tasks(cluster="craftassist", tasks=[task_arn])
    logging.info(x)
    attachment_id = x["tasks"][0]["containers"][0]["networkInterfaces"][0]["attachmentId"]
    attachment = next(y for y in x["tasks"][0]["attachments"] if y["id"] == attachment_id)
    eni = next(y for y in attachment["details"] if y["name"] == "networkInterfaceId")["value"]
    ip = ec2.NetworkInterface(eni).private_ip_addresses[0]["Association"]["PublicIp"]
    return ip


if __name__ == "__main__":
    # task_arn = launch_task()
    # logging.info('Launched task {}'.format(task_arn))
    task_arn = "55394421-9b47-4fa1-aa8e-0fabb3b0765a"
    for _ in range(120):
        time.sleep(1)
        try:
            ip = describe_task(task_arn)
            logging.info("IP {}".format(ip))
            ping_cuberite.ping(ip, 25565, timeout=1)
            logging.info("DONE")
        except Exception as e:
            logging.info("Not yet... {}".format(e))
