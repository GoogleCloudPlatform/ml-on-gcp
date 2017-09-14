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
import shutil

from google.cloud import storage

def archive_and_upload(bucket_name, directory, extension='zip', object_name=None):
    """Archives a directory and upload to GCS"""
    sc = storage.Client()
    object_name = object_name or '{}.{}'.format(directory, extension)

    temp_filename = shutil.make_archive('_tmp', extension, directory)

    bucket = sc.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(temp_filename)

    os.remove(temp_filename)
