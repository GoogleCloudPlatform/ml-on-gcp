# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Helper functions to find all local dependencies between modules within a root package.

import ast
import os
import re


class SourceFinder(object):
    def __init__(self, package_path, script_path):
        # root should be an absolute path
        # remove trailing '/'
        self.package_path = package_path.rstrip('/')

        # script_path should be absolute path to script
        self.script_path = script_path

        # for this to work properly self.root must not have a trailing '/'
        # parent is used to figure out the absolute path of dependency scripts
        self.parent, self.package_name = os.path.split(self.package_path)

        # for example:
        # script_path = /tmp/a/b/c/d.py
        # package_path = /tmp/a/b
        # parent = /tmp/a
        # package_name = b

        self.externals = set([])

        # keys are absolute paths, values are lists of module names
        self.script_imports = {}


    def process(self):
        # to_visit is a list of absolute paths
        to_visit = [self.script_path]

        while len(to_visit) > 0:
            path = to_visit.pop(0)

            module_names = self.process_script(path)

            for module_name in module_names:
                # turn this into absolute path
                path = os.path.join(self.parent, self.module_name_to_path(module_name))

                # sometimes a variable is imported, in which case we back track one level
                if not os.path.exists(path):
                    parent, _ = os.path.split(path)
                    path = parent + '.py'

                # at this point the file should exist
                if not os.path.exists(path):
                    raise FileNotFoundError(path)

                # add to the to_visit list if not yet visited
                if path not in self.script_imports:
                    to_visit.append(path)


    def process_script(self, path):
        # side effect: updates self.externals and self.script_imports
        # returns the script_imports of the processed script
        with open(path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        self.script_imports[path] = set([])
        for node in tree.body:
            if node.__class__ is ast.Import:
                module_names = [alias.name for alias in node.names]

            elif node.__class__ is ast.ImportFrom:
                parent_module_name = node.module

                module_names = ['{}.{}'.format(parent_module_name, alias.name) for alias in node.names]

            else:
                continue

            for module_name in module_names:
                if module_name.startswith(self.package_name):
                    self.script_imports[path].add(module_name)
                else:
                    self.externals.add(module_name)

        return self.script_imports[path]


    def module_name_to_path(self, module_name):
        # converts module = 'a.b.c' to 'a/b/c.py'
        path = module_name.replace('.', '/') + '.py'

        return path


    def path_to_relative_path(self, path):
        # returns path starting with self.package_name
        # this is used in cmle_package.py
        return re.sub('^{}/'.format(self.parent), '', path)

