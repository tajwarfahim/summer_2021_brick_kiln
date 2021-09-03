import boto3
import os

def upload_files_to_bucket(file_names, bucket_name, bucket_key_prefix):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    for file_name in file_names:
        actual_file_name = file_name.split("/")[-1]
        filekey = os.path.join(bucket_key_prefix, actual_file_name)
        bucket.upload_file(Filename=file_name, Key=filekey)
