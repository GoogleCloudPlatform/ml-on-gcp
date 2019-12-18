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

import argparse
from contextlib import contextmanager
import glob
import os
import shutil
import tempfile
import yaml

from cmle_package import CMLEPackage
from git import Repo

CONFIG_FILENAMES = glob.glob('*_samples.yaml')
GITHUB_URL_TEMPLATE = 'https://github.com/{}/{}.git'


@contextmanager
def temp_clone(org, repository):
    temp_dir = tempfile.mkdtemp()
    github_url = GITHUB_URL_TEMPLATE.format(org, repository)
    print('Cloning from {}'.format(github_url))
    repo = Repo.clone_from(github_url, temp_dir, multi_options=['--depth 1', '--no-single-branch'])

    try:
        yield repo
    finally:
        shutil.rmtree(repo.working_dir)


def main(args):
    for filename in CONFIG_FILENAMES:
        with open(filename, 'r') as f:
            config = yaml.load(f.read())

        org = config['org']
        repository = config['repository']
        branch = config['branch']
        requires = config.get('requires', [])
        samples = config['samples']
        runtime_version = config['runtime_version']

        # filter samples
        if args.filter:
            print('Building samples with script_name containing: {}'.format(args.filter))
            samples = [s for s in samples if args.filter in s['script_name']]

        if not samples:
            continue

        with temp_clone(org, repository) as repo:
            for sample_dict in samples:
                sample_dict['org'] = org
                sample_dict['repository'] = repository

                # inherit from repo wide config
                sample_dict.setdefault('requires', []).extend(requires)
                sample_dict['runtime_version'] = runtime_version
                sample_dict['branch'] = branch

                cmle_package = CMLEPackage(sample_dict, repo)

                cmle_package.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, default='', help='process only samples whose script_name contains the filter string')

    args = parser.parse_args()

    main(args)
