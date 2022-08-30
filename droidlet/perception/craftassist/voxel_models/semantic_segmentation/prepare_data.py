import argparse
import boto3
import json
import os
import random
import pickle

from generate_data_from_iglu import json_to_segdata

DEFAULT_BUCKET_NAME = "droidlet-hitl"


AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
s3 = boto3.resource('s3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


def read_data_from_s3(fname, bucket=DEFAULT_BUCKET_NAME):
    response = s3.Object(f"{bucket}", f"{fname}").get()
    data = response["Body"].read().decode("utf-8")
    data = json.loads(data)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3_file_path", type=str, default="20220824150517/annotated_scenes/1661415655_clean.json"
    )
    
    parser.add_argument("--split_ratio", type=float, default=0.8, help="how much for training data, the rest are for validation")
    parser.add_argument("--seed", type=int, default=1234, help="random seed for shuffling, etc.")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)

    raw_data = read_data_from_s3(args.s3_file_path)
    processed_data = []
    for J in raw_data:
        processed_data.append(json_to_segdata(J))
    
    random.shuffle(processed_data)

    training_data_sz = int(len(processed_data) * args.split_ratio)
    training_data = processed_data[:training_data_sz]
    validation_data = processed_data[training_data_sz:]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, "training_data.pkl"), "wb") as f:
        pickle.dump(training_data, f)
    
    with open(os.path.join(output_dir, "validation_data.pkl"), "wb") as f:
        pickle.dump(validation_data, f)


    
    
