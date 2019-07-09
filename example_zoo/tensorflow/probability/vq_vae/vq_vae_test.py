# Copyright 2019 Google LLC
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
import subprocess
import time
import uuid

import pytest

from google.cloud import storage

WAIT_TIME = 240
ARTIFACTS_BUCKET = os.environ['EXAMPLE_ZOO_ARTIFACTS_BUCKET']
PROJECT_ID = os.environ['EXAMPLE_ZOO_PROJECT_ID']

SUBMIT_SCRIPTS = ['submit_27.sh', 'submit_35.sh']


@pytest.fixture(scope='session')
def gcs_bucket_prefix():
    # Create a temporary prefix for storing the artifacts.
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(ARTIFACTS_BUCKET)
    prefix = os.path.join('example_zoo_artifacts', str(uuid.uuid4()))

    yield (bucket, prefix)

    # Clean up after sleeping for another minute.
    time.sleep(120)
    for blob in bucket.list_blobs(prefix=prefix):
        blob.delete()


@pytest.mark.parametrize('submit_script', SUBMIT_SCRIPTS)
def test_vq_vae(gcs_bucket_prefix, submit_script):
    bucket, prefix = gcs_bucket_prefix

    subprocess_env = os.environ.copy()
    subprocess_env['EXAMPLE_ZOO_ARTIFACTS_BUCKET'] = 'gs://{}/{}'.format(os.environ['EXAMPLE_ZOO_ARTIFACTS_BUCKET'], prefix)

    out = subprocess.check_output(['bash', submit_script], env=subprocess_env)
    out_str = out.decode('ascii')

    assert 'QUEUED' in out_str, 'Job submission failed: {}'.format(out_str)

    # Get jobId so we can cancel the job easily.
    job_id = re.match(r'jobId: (.+)\b', out_str).group(1)

    time.sleep(WAIT_TIME)

    # Cancel the job.
    subprocess.check_call(['gcloud', 'ai-platform', 'jobs', 'cancel', job_id, '--project', PROJECT_ID])

    blob_names = [blob.name for blob in bucket.list_blobs(prefix=prefix)]
    out_str = ' '.join(blob_names)

    assert 'validation_reconstructions.png' in out_str, 'Artifact "validation_reconstructions.png" not found in bucket {} with prefix {} after {} seconds.'.format(bucket, prefix, WAIT_TIME)
