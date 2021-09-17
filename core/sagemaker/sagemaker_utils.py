# Citation:
# 1. https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-creating-buckets.html

import boto3
import os
import logging
from botocore.exceptions import ClientError


def upload_files_to_bucket(file_names, bucket_name, bucket_key_prefix):
    bucket = open_bucket(bucket_name=bucket_name)

    for file_name in file_names:
        actual_file_name = os.path.basename(file_name)
        filekey = os.path.join(bucket_key_prefix, actual_file_name)
        bucket.upload_file(Filename=file_name, Key=filekey)


def create_bucket(bucket_name):
    try:
        s3_client = boto3.client("s3")
        location = {'LocationConstraint': 'us-east-1'}
        s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def open_bucket(bucket_name):
    s3 = boto3.resource("s3")
    bucket_names = [bucket.name for bucket in s3.buckets.all()]

    if bucket_name not in bucket_names:
        assert create_bucket(bucket_name=bucket_name)

    bucket = s3.Bucket(bucket_name)
    return bucket
