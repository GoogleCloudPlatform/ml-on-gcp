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
import shutil
import urllib2

SOURCE_BASE = 'https://raw.githubusercontent.com/{org}/{repository}/{branch}/{source_path}/{source_name}'
WEB_BASE = 'https://github.com/{org}/{repository}/blob/{branch}/{source_path}/{source_name}'


class CMLEPackage(object):
    def __init__(self, sample_dict):
        self.org = sample_dict['org']
        self.repository = sample_dict['repository']
        self.branch = sample_dict['branch']
        self.source_path = sample_dict['source_path']
        self.source_name = sample_dict['source_name']

        # optional
        self.requires = sample_dict.get('requires', [])
        self.tfgfile_wrap = sample_dict.get('tfgfile_wrap', [])

        self._source_content = None

        self.name = self.source_name.split('.')[0]
        self.output_dir = os.path.join('..', self.org, self.repository, self.name)

        # prefix to output filename mapping.
        self.outputs = {
            '': ['setup.py', 'config.yaml', 'submit.sh', 'README.md'],
            'trainer': [self.source_name, '__init__.py', 'tfgfile_wrapper.py']
        }

        # clean up previously generated package
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, 'trainer'))

        self.format_dict = {
            'org': self.org,
            'repository': self.repository,
            'source_path': self.source_path,
            'source_name': self.source_name,
            'name': self.name,
            'requires': '' if not self.requires else ','.join("'{}'".format(req) for req in self.requires),
            'web_url': self.web_url
        }


    @property
    def source_url(self):
        source_url = SOURCE_BASE.format(
            org=self.org,
            repository=self.repository,
            branch=self.branch,
            source_path=self.source_path,
            source_name=self.source_name
        )
        return source_url


    @property
    def web_url(self):
        web_url = WEB_BASE.format(
            org=self.org,
            repository=self.repository,
            branch=self.branch,
            source_path=self.source_path,
            source_name=self.source_name
        )
        return web_url


    def get_and_modify_source(self):
        print('Getting source: {}'.format(self.source_name))
        
        response = urllib2.urlopen(self.source_url)

        if self.tfgfile_wrap:
            lines = []
            add_import = True
            for line in response:
                if add_import and 'import' in line and 'from __future__' not in line:
                    line = 'from trainer.tfgfile_wrapper import tfgfile_wrapper\n' + line
                    add_import = False

                for to_wrap in self.tfgfile_wrap:
                    if 'def {}'.format(to_wrap) in line:
                        line = '@tfgfile_wrapper\n' + line

                lines.append(line)

            self._source_content = ''.join(lines)

        else:
            self._source_content = response.read()


    @property
    def source_content(self):
        if self._source_content is None:
            self.get_and_modify_source()
        return self._source_content


    def generate(self):
        for prefix, filenames in self.outputs.items():
            for filename in filenames:
                output_path = os.path.join(self.output_dir, prefix, filename)

                if filename == self.source_name:
                    content = self.source_content
                else:
                    with open(os.path.join('templates', filename), 'r') as f:
                        template = f.read()

                    content = template.format(**self.format_dict)

                with open(output_path, 'w') as f:
                    f.write(content)
