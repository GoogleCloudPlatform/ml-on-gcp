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

from contextlib import contextmanager
import os
import shutil
import tempfile
import yaml

from cmle_package import CMLEPackage
from git import Repo

CONFIG_FILENAMES = [
    'tf_probability_samples.yaml'
]
GITHUB_URL_TEMPLATE = 'https://github.com/{}/{}.git'


@contextmanager
def temp_clone(org, repository):
    temp_dir = tempfile.mkdtemp()
    github_url = GITHUB_URL_TEMPLATE.format(org, repository)
    print('Cloning from {}'.format(github_url))
    repo = Repo.clone_from(github_url, temp_dir)#, multi_options=['--depth 1'])

    try:
        yield repo
    finally:
        shutil.rmtree(repo.working_dir)


for filename in CONFIG_FILENAMES:
    with open(filename, 'r') as f:
        config = yaml.load(f.read())

    org = config['org']
    repository = config['repository']
    samples = config['samples']

    with temp_clone(org, repository) as repo:
        for sample_dict in samples:
            sample_dict['org'] = org
            sample_dict['repository'] = repository

            cmle_package = CMLEPackage(sample_dict, repo)

            cmle_package.generate()
