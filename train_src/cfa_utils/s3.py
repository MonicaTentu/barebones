import boto3
import os

# https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket
def download_dir(resource, dist, local, bucket, status_cb=lambda x: x, striplen=0):
    if not striplen:
        striplen = len(dist)
    client = resource.meta.client
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(resource, subdir.get('Prefix'), local, bucket, status_cb)
        for file in result.get('Contents', []):
            dest_pathname = os.path.join(local, file.get('Key')[striplen:])
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if not file.get('Key').endswith('/'):
               	status_cb(dest_pathname)
                client.download_file(bucket, file.get('Key'), dest_pathname)
