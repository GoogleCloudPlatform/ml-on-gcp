# Copyright 2017, Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import shutil
import pickle

from google.cloud import storage


def _make_gcs_uri(bucket_name, object_name):
    return 'gs://{}/{}'.format(bucket_name, object_name)


def _split_uri(gcs_uri):
    """Splits gs://bucket_name/object_name to (bucket_name, object_name)"""
    pattern = r'gs://([^/]+)/(.+)'
    match = re.match(pattern, gcs_uri)

    bucket_name = match.group(1)
    object_name = match.group(2)

    return bucket_name, object_name


def get_blob(bucket_name, object_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    return blob


def get_uri_blob(gcs_uri):
    bucket_name, object_name = _split_uri(gcs_uri)
    return get_blob(bucket_name, object_name)


def archive_and_upload(bucket_name, directory, extension='zip', object_name=None):
    """Archives a directory and upload to GCS.
    Returns the object's GCS uri.
    """
    storage_client = storage.Client()
    object_name = object_name or '{}.{}'.format(directory, extension)

    temp_filename = shutil.make_archive('_tmp', extension, directory)

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(temp_filename)

    os.remove(temp_filename)

    return _make_gcs_uri(bucket_name, object_name)


def pickle_and_upload(obj, bucket_name, object_name):
    """Returns the object's GCS uri."""
    pickle_str = pickle.dumps(obj)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_string(pickle_str)

    return _make_gcs_uri(bucket_name, object_name)


def download_and_unpickle(bucket_name, object_name):
    blob = get_blob(bucket_name, object_name)
    pickle_str = blob.download_as_string()

    obj = pickle.loads(pickle_str)
    return obj


def download_uri_and_unpickle(gcs_uri):
    bucket_name, object_name = _split_uri(gcs_uri)
    obj = download_and_unpickle(bucket_name, object_name)

    return obj


