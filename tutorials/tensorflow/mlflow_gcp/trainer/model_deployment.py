# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deploy a model in AI Platform."""

import logging
import json
import time
import subprocess

from googleapiclient import discovery
from googleapiclient import errors

_WAIT_FOR_COMPLETION_SLEEP_SECONDS = 10
_PYTHON_VERSION = '3.5'
_RUN_TIME_VERSION = '1.15'


def _create_service():
    """Gets service instance to start API searches.

    :return:
    """
    return discovery.build('ml', 'v1')


def copy_artifacts(source_path, destination_path):
    """

    :param source_path:
    :param destination_path:
    :return:
    """
    logging.info(
        'Moving model directory from {} to {}'.format(source_path,
                                                      destination_path))
    subprocess.call(
        "gsutil -m cp -r {} {}".format(source_path, destination_path),
        shell=True)


class AIPlatformModel(object):
    def __init__(self, project_id):
        self._project_id = project_id
        self._service = _create_service()

    def model_exists(self, model_name):
        """

        :param model_name:
        :return:
        """
        models = self._service.projects().models()
        try:
            response = models.list(
                parent='projects/{}'.format(self._project_id)).execute()
            if response:
                for model in response['models']:
                    if model['name'].rsplit('/', 1)[1] == model_name:
                        return True
                    else:
                        return False
        except errors.HttpError as err:
            logging.error('%s', json.loads(err.content)['error']['message'])

    def _list_model_versions(self, model_name):
        """Lists existing model versions in the project.

        Args:
          model_name: Model name to list versions for.

        Returns:
          Dictionary of model versions.
        """
        versions = self._service.projects().models().versions()

        try:
            return versions.list(
                parent='projects/{}/models/{}'.format(self._project_id,
                                                      model_name)).execute()
        except errors.HttpError as err:
            logging.error('%s', json.loads(err.content)['error']['message'])

    def create_model(self, model_name, model_region='us-central1'):
        """

        :param model_name:
        :param model_region:
        :return:
        """
        if not self.model_exists(model_name):
            body = {
                'name': model_name,
                'regions': model_region,
                'description': 'MLflow model'
            }
            parent = 'projects/{}'.format(self._project_id)
            try:
                self._service.projects().models().create(
                    parent=parent, body=body).execute()
                logging.info('Model "%s" has been created.', model_name)
            except errors.HttpError as err:
                logging.error('"%s". Skipping model creation.',
                              json.loads(err.content)['error']['message'])
        else:
            logging.warning('Model "%s" already exists.', model_name)

    def deploy_model(self, bucket_name, model_name, model_version,
                     runtime_version=_RUN_TIME_VERSION):
        """Deploys model on AI Platform.

        Args:
          bucket_name: Cloud Storage Bucket name that stores saved model.
          model_name: Model name to deploy.
          model_version: Model version.
          runtime_version: Runtime version.

        Raises:
          RuntimeError if deployment completes with errors.
        """
        # For details on request body, refer to:
        # https://cloud.google.com/ml-engine/reference/rest/v1/projects
        # .models.versions/create
        model_version_exists = False
        model_versions_list = self._list_model_versions(model_name)
        #  Field: version.name Error: A name should start with a letter and
        #  contain only letters, numbers and underscores
        model_version = 'mlflow_{}'.format(model_version)

        if model_versions_list:
            for version in model_versions_list['versions']:
                if version['name'].rsplit('/', 1)[1] == model_version:
                    model_version_exists = True

        if not model_version_exists:
            request_body = {
                'name': model_version,
                'deploymentUri': '{}'.format(bucket_name),
                'framework': 'TENSORFLOW',
                'runtimeVersion': runtime_version,
                'pythonVersion': _PYTHON_VERSION
            }
            parent = 'projects/{}/models/{}'.format(self._project_id,
                                                    model_name)
            response = self._service.projects().models().versions().create(
                parent=parent, body=request_body).execute()
            op_name = response['name']
            while True:
                deploy_status = (
                    self._service.projects().operations().get(
                        name=op_name).execute())
                if deploy_status.get('done'):
                    logging.info('Model "%s" with version "%s" deployed.',
                                 model_name,
                                 model_version)
                    break
                if deploy_status.get('error'):
                    logging.error(deploy_status['error'])
                    raise RuntimeError(
                        'Failed to deploy model for serving: {}'.format(
                            deploy_status['error']))
                logging.info(
                    'Waiting for %d seconds for "%s" with "%s" version to be '
                    'deployed.',
                    _WAIT_FOR_COMPLETION_SLEEP_SECONDS, model_name,
                    model_version)
                time.sleep(_WAIT_FOR_COMPLETION_SLEEP_SECONDS)

        else:
            logging.info('Model "%s" with version "%s" already exists.',
                         model_name,
                         model_version)
