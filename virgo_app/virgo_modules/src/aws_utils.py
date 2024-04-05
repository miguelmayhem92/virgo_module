import yaml
import boto3
from pathlib import Path
from io import StringIO, BytesIO
import pandas as pd


def upload_file_to_aws(bucket,key,input_path, aws_credentials):
    '''
    upload file from a folder to an s3 folder

            Parameters:
                    bucket (str): bucket name
                    key (str): key pattern or folder in s3 e.g. path/to/upload/
                    input_path (str): input path of the file to upload e.g. path/to/upload.txt
                    aws_credentials (dict): aws credentials dictionary
                    
            Returns:
                    None
    '''
    session = boto3.Session(aws_access_key_id=aws_credentials['AWS_ACCESS_KEY_ID'],aws_secret_access_key=aws_credentials['AWS_SECRET_ACCESS_KEY'])
    bucket = aws_credentials[bucket]
    s3 = session.resource('s3')
    s3.meta.client.upload_file(Filename=input_path , Bucket=bucket, Key=key)

def upload_pandas_to_s3(data_frame,bucket,key, aws_credentials):
    '''
    upload dataframe as csv to an s3 folder

            Parameters:
                    data_frame (pd.DataFrame): data
                    bucket (str): bucket name
                    key (str): key pattern or folder in s3 e.g. path/to/upload/
                    aws_credentials (dict): aws credentials dictionary
                    
            Returns:
                    None
    '''
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer)
    csv_buffer.seek(0)

    s3 = boto3.client("s3",region_name=aws_credentials['AWS_DEFAULT_REGION'],aws_access_key_id=aws_credentials['AWS_ACCESS_KEY_ID'],aws_secret_access_key=aws_credentials['AWS_SECRET_ACCESS_KEY'])
    bucket = aws_credentials[bucket]
    s3.put_object(Bucket=bucket, Body=csv_buffer.getvalue(), Key= key)

def download_file_to_aws(bucket,key, aws_credentials):
    '''
    download csv file from s3 folder

            Parameters:
                    bucket (str): bucket name
                    key (str): key pattern or folder in s3 e.g. path/to/download/file.csv
                    aws_credentials (dict): aws credentials dictionary
                    
            Returns:
                    None
    '''
    s3c = boto3.client(
            's3', 
            region_name = aws_credentials['AWS_DEFAULT_REGION'],
            aws_access_key_id = aws_credentials['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key = aws_credentials['AWS_SECRET_ACCESS_KEY']
        )
    obj = s3c.get_object(Bucket= bucket , Key = key)
    df = pd.read_csv(BytesIO(obj['Body'].read()), encoding='utf8')
    return df