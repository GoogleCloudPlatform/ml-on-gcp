import tensorflow as tf
from subprocess import call
import os
import tempfile

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

credentials = GoogleCredentials.get_application_default()

# Source:
# https://cloud.google.com/ml-engine/docs/how-tos/online-predict
# https://cloud.google.com/ml-engine/docs/how-tos/deploying-models


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


def package_and_upload_module(gcs_directory):
    dtemp = tempfile.mkdtemp()
    call(['python', 'setup.py', 'egg_info', '--egg-base={}'.format(dtemp), 'sdist', '--dist-dir={}'.format(dtemp)])
    package_fn = [fn for fn in tf.gfile.ListDirectory(dtemp) if fn.endswith('tar.gz')][0]
    package_local = os.path.join(dtemp, package_fn)
    package_gcs = os.path.join(gcs_directory, package_fn)

    tf.gfile.Copy(package_local, package_gcs, overwrite=True)

    return package_gcs


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
