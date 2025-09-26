#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : upload_files.py
"""
import pandas as pd
from pathlib import Path
import boto3
from datetime import datetime

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "output"
source_file = "content_based_recommendations.csv"
ts = datetime.now()

boto3_session = boto3.Session(
    botocore_session=dbutils.credentials.getServiceCredentialsProvider(
        'service-cred-nas-lifion_ml-sdq-dit'
    )
)
s3_client = boto3_session.client('s3') 

# Upload file to S3
bucket_name = "ml-models-bucket-appbuild-02"
s3_file = f"content_based_recommendations_{ts}.csv"
s3_path = f"recommended-actions/{s3_file}"

source_path = local_path / source_file
with open(source_path, "rb") as f:
    response = s3_client.put_object(
        Bucket=bucket_name,
        Body=f,
        Key=s3_path
    )
status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
if status == 200:
    print(f"Successful S3 put_object response. Status - {status}")