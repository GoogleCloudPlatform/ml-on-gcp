# Copyright 2018 Google LLC
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


import tensorflow as tf
from subprocess import call
import inspect
import os
import nbformat
import re
import tempfile
from textwrap import dedent

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

credentials = GoogleCredentials.get_application_default()

# Source:
# https://cloud.google.com/ml-engine/docs/how-tos/online-predict
# https://cloud.google.com/ml-engine/docs/how-tos/deploying-models


def ipynb_to_py(source, export_base):
    notebook = nbformat.read('ipynb_to_py.ipynb', as_version=4)

    files = {} # filename: file object
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        lines = cell['source'].split('\n')
        
        # the first line should contain a file name followed by "##"
        if lines[0].startswith('##'):
            filename = lines[0].replace('#', '').strip()
            
            # open a new file object for the filename if it hasn't been created yet
            file_obj = files.setdefault(filename, open(os.path.join(export_base, filename), 'w'))
        else:
            # notebook only cells, ignore
            continue
            
        # handle lines that should be added only when exported to .py
        # only lines that immediate follow the "##" line that starts with "#" will have that "#" removed
        iter_lines = enumerate(lines)
        # skip the first line
        next(iter_lines)
        for i, line in iter_lines:
            if line.startswith('#'):
                code_line = line.replace('#', '', 1).strip()
                file_obj.write(code_line + '\n')
                
            else:
                # no more "#" lines
                break
        
        # other lines will just be copied over as is
        for line in lines[i:]:
            file_obj.write(line + '\n')

    for filename, file_obj in files.items():
        print('File written: {}'.format(os.path.join(export_base, filename)))
        file_obj.close()


def get_models(project):
    service = discovery.build('ml', 'v1', credentials=credentials)
    name = 'projects/{}'.format(project)

    request = service.projects().models().list(parent=name)

    return request.execute()


def get_model_versions(project, model):
    service = discovery.build('ml', 'v1', credentials=credentials)
    name = 'projects/{}/models/{}'.format(project, model)

    nextPageToken = True
    versions = []

    while nextPageToken:
        request = service.projects().models().versions().list(parent=name, pageToken=nextPageToken)
        response = request.execute()

        nextPageToken = response.get('nextPageToken', None)
        versions.extend(response['versions'])

    return versions


def build_and_upload_package(gcs_directory):
    dtemp = tempfile.mkdtemp()
    call(['python', 'setup.py', 'egg_info', '--egg-base={}'.format(dtemp), 'sdist', '--dist-dir={}'.format(dtemp)])
    package_fn = [fn for fn in tf.gfile.ListDirectory(dtemp) if fn.endswith('tar.gz')][0]
    package_local = os.path.join(dtemp, package_fn)
    package_gcs = os.path.join(gcs_directory, package_fn)

    tf.gfile.Copy(package_local, package_gcs, overwrite=True)

    return package_gcs


def create_package(model_fn, input_fn, parse_args, train):
    template_fn = 'trainer/task_template.py'
    output_fn = 'trainer/task.py'

    sub_dict = {
        '# <MODEL_FN>': model_fn,
        '# <INPUT_FN>': input_fn,
        '# <PARSE_ARGS>': parse_args,
        '# <TRAIN>': train
    }

    with open(template_fn, 'r') as f:
        code_content = f.read()

    for placeholder, function in sub_dict.iteritems():
        function_code = inspect.getsource(function)
        function_code = dedent(function_code)
        code_content = re.sub(placeholder, function_code, code_content)

    with open(output_fn, 'w') as f:
        f.write(code_content)


def train_model(project, job_spec):
    service = discovery.build('ml', 'v1', credentials=credentials)
    parent = 'projects/{}'.format(project)

    request = service.projects().jobs().create(body=job_spec, parent=parent)

    return request.execute()


def deploy_model(project, model, version, gcs_uri):
    service = discovery.build('ml', 'v1', credentials=credentials)

    parent = 'projects/{}/models/{}'.format(project, model)
    request_dict = {
        'name': version,
        'deploymentUri': gcs_uri
    }

    request = service.projects().models().versions().create(parent=parent, body=request_dict)

    return request.execute()


def set_default(project, model, version):
    service = discovery.build('ml', 'v1', credentials=credentials)
    name = 'projects/{}/models/{}/versions/{}'.format(project, model, version)

    # check if already is default
    req = service.projects().models().versions().get(name=name)
    res = req.execute()

    if 'isDefault' in res and res['isDefault'] is True:
        return res

    request_dict = {
        'name': version
    }

    request = service.projects().models().versions().setDefault(name=name, body=request_dict)

    return request.execute()


def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']
