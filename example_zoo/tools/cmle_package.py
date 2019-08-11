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


class Pipe(object):
    # a pipe is a triplet of (source path, destination path, list of transformations)
    def __init__(self, source, destination, transformations=[]):
        self.source = source
        self.destination = destination
        self.transformations = transformations


    def _handle_dir(self):
        # ignore tests and test data
        shutil.copytree(
            self.source,
            self.destination,
            ignore=shutil.ignore_patterns('*_test.py', '*testing*')
        )


    def _handle_file(self):
        with open(self.source, 'r') as source_file, open(self.destination, 'w') as destination_file:
                content = source_file.read()

                for transformation in self.transformations:
                    content = transformation(content)

                destination_file.write(content)


    def run(self):
        if not os.path.exists(self.source):
            raise ValueError('{} does not exist'.format(self.source))
        if os.path.isdir(self.source):
            self._handle_dir()
        elif os.path.isfile(self.source):
            self._handle_file()


class CMLEPackage(object):
    WEB_BASE = 'https://github.com/{org}/{repository}/blob/{branch}/{full_path}'
    TEMPLATE_FILENAMES = [
        'setup.py',
        'config.yaml',
        'submit_27.sh',
        'submit_35.sh',
        'README.md',
    ]

    def __init__(self, sample_dict, repo):
        self.org = sample_dict['org']
        self.repository = sample_dict['repository']
        self.branch = sample_dict['branch']
        self.module_path = sample_dict.get('module_path', '')
        self.script_path = sample_dict.get('script_path', '')
        self.script_name = sample_dict['script_name']
        self.artifact = sample_dict['artifact']
        self.wait_time = sample_dict['wait_time']

        # check out the specified branch
        self.repo = repo
        self.repo.git.checkout(self.branch)
        self.working_dir = self.repo.working_dir

        # optional configs
        self.other_sources = sample_dict.get('other_sources', [])

        if 'args' in sample_dict:
            sep = ' \\\n    '
            self.args = sep + sep.join(sample_dict['args'])
        else:
            self.args = ''

        if 'requires' in sample_dict:
            self.requires = ','.join("'{}'".format(req) for req in sample_dict['requires'])
        else:
            self.requires = ''

        self.tfgfile_wrap = sample_dict.get('tfgfile_wrap', [])

        self.pipes = []


    def format(self, content):
        return content.format(**self.format_dict)


    def add_tfgfile_wrapper(self, content):
        lines = []
        add_import = True
        for line in content.split('\n'):
            if add_import and 'import' in line and 'from __future__' not in line:
                lines.append(self.tfgfile_wrapper_import)
                add_import = False

            for to_wrap in self.tfgfile_wrap:
                if 'def {}'.format(to_wrap) in line:
                    lines.append('@tfgfile_wrapper')

            lines.append(line)

        return '\n'.join(lines)


    def add_job_dir_flag(self, content):
        flags_define = 'flags.DEFINE_string(name="job-dir", default="/tmp", help="AI Platform Training passes this to the training script.")'
        lines = []
        add_flags_define = False
        for line in content.split('\n'):
            # inject the flags define line right after the import
            if add_flags_define == True:
                lines.append(flags_define)
                add_flags_define = False

            if line == 'from absl import flags':
                add_flags_define = True                

            lines.append(line)

        return '\n'.join(lines)


    def build_pipes(self):
        for template_filename in self.TEMPLATE_FILENAMES:
            self.pipes.append(
                Pipe(
                    os.path.join('templates', template_filename),
                    os.path.join(self.output_dir, template_filename),
                    [self.format]
                )
            )

        # test
        self.pipes.append(
            Pipe(
                'templates/cmle_test.py',
                os.path.join(self.output_dir, self.test_name),
                [self.format]
            )
        )

        # add __init__.py at all levels
        parts = self.output_script_path.split('/')
        current_path = ''
        for part in parts:
            current_path = os.path.join(current_path, part)
            self.pipes.append(
                Pipe(
                    'templates/__init__.py',
                    os.path.join(self.output_dir, current_path, '__init__.py')
                )
            )

        # tfgfile_wrapper if needed
        if self.tfgfile_wrap:
            self.pipes.append(
                Pipe(
                    'templates/tfgfile_wrapper.py',
                    os.path.join(self.output_dir, self.output_script_path, 'tfgfile_wrapper.py')
                )
            )

        # source
        source_transformations = [self.add_job_dir_flag]

        if self.tfgfile_wrap:
            source_transformations.append(self.add_tfgfile_wrapper)

        self.pipes.append(
            Pipe(
                os.path.join(self.working_dir, self.module_path, self.script_path, self.script_name),
                os.path.join(self.output_dir, self.output_script_path, self.script_name),
                source_transformations
            )
        )

        # # other source files/directories
        # for other_source in self.other_sources:
        #     self.pipes.append(
        #         Pipe(
        #             os.path.join(self.working_dir, other_source),
        #             os.path.join(self.output_dir, other_source)
        #         )
        #     )

        # experimental: use source_finder to find minimally required other source files
        from source_finder import SourceFinder
        sf = SourceFinder(
            os.path.join(self.working_dir, self.module_path),
            os.path.join(self.working_dir, self.module_path, self.script_path, self.script_name)
        )
        sf.process()

        for module_path in sf.script_imports.keys():
            rel_path = sf.path_to_relative_path(module_path)

            self.pipes.append(
                Pipe(
                    module_path,
                    os.path.join(self.output_dir, rel_path)
                )
            )



    @property
    def name(self):
        return self.script_name.split('.')[0]


    @property
    def test_name(self):
        return 'cmle_{}_test.py'.format(self.name)
    

    # for the generated package, putting the script into a `trainer` directory if no script_path is specified
    @property
    def output_script_path(self):
        return self.script_path or 'trainer'
    

    @property
    def package_path(self):
        return self.output_script_path.split('/')[0]


    @property
    def module_parent(self):
        return self.output_script_path.replace('/', '.')


    @property
    def module_name(self):
        return '{}.{}'.format(self.module_parent, self.name)


    @property
    def output_dir(self):
        return os.path.join('..', self.org, self.repository, self.name)


    @property
    def web_url(self):
        web_url = self.WEB_BASE.format(
            org=self.org,
            repository=self.repository,
            branch=self.branch,
            full_path=self.full_path
        )
        return web_url


    @property
    def full_path(self):
        return os.path.join(self.module_path, self.script_path, self.script_name)


    @property
    def tfgfile_wrapper_import(self):
        return 'from {}.tfgfile_wrapper import tfgfile_wrapper'.format(self.module_parent)


    @property
    def format_dict(self):
        format_dict = {
            'org': self.org,
            'repository': self.repository,
            'name': self.name,
            'package_path': self.package_path,
            'module_name': self.module_name,
            'full_path': self.full_path,
            'requires': self.requires,
            'web_url': self.web_url,
            'artifact': self.artifact,
            'wait_time': self.wait_time,
            'args': self.args
        }
        return format_dict


    def generate(self):
        print('Building package for {}'.format(self.name))
        # clean up previously generated package
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, self.output_script_path))

        self.build_pipes()

        for pipe in self.pipes:
            pipe.run()
