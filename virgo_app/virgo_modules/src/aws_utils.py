import yaml
import boto3
from pathlib import Path
from io import StringIO, BytesIO
import pandas as pd


def upload_file_to_aws(bucket,key,input_path, secret_path = 'secrets.yaml'):
    
    credentials = yaml.safe_load(Path(secret_path).read_text())
    session = boto3.Session(aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'],aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])
    bucket = credentials[bucket]
    s3 = session.resource('s3')
    s3.meta.client.upload_file(Filename=input_path , Bucket=bucket, Key=key)

def upload_pandas_to_s3(data_frame,bucket,key, secret_path = 'secrets.yaml'):

    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer)
    csv_buffer.seek(0)

    credentials = yaml.safe_load(Path(secret_path).read_text())
    s3 = boto3.client("s3",region_name=credentials['AWS_DEFAULT_REGION'],aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'],aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])
    bucket = credentials[bucket]
    s3.put_object(Bucket=bucket, Body=csv_buffer.getvalue(), Key= key)

def download_file_to_aws(bucket,key, secret_path = 'secrets.yaml'):
    
    credentials = yaml.safe_load(Path(secret_path).read_text())
    s3c = boto3.client(
            's3', 
            region_name = credentials['AWS_DEFAULT_REGION'],
            aws_access_key_id = credentials['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key = credentials['AWS_SECRET_ACCESS_KEY']
        )
    obj = s3c.get_object(Bucket= bucket , Key = key)
    df = pd.read_csv(BytesIO(obj['Body'].read()), encoding='utf8')
    return df