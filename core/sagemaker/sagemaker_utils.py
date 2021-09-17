# general packages
import boto3
import os


def upload_files_to_bucket(file_names, bucket_name, bucket_key_prefix):
    bucket = open_bucket(bucket_name=bucket_name)

    for file_name in file_names:
        actual_file_name = os.path.basename(file_name)
        filekey = os.path.join(bucket_key_prefix, actual_file_name)
        bucket.upload_file(Filename=file_name, Key=filekey)


def open_bucket(bucket_name):
    s3 = boto3.resource("s3")
    bucket_names = [bucket.name for bucket in s3.buckets.all()]
    assert bucket_name in bucket_names

    bucket = s3.Bucket(bucket_name)
    return bucket
