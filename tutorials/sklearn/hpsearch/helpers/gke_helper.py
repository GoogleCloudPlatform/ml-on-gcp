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

"""Helpers for accessing Google Kubernetes Engine in Python code.

`create_cluster`: Creates a Kubernetes cluster with suitable access scopes
needed for the hpsearch notebooks.

For more information:
https://cloud.google.com/kubernetes-engine/
"""

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

credentials = GoogleCredentials.get_application_default()


def create_cluster(project_id, zone, cluster_id, n_nodes=1, machine_type='n1-highcpu-4'):
    """Documentation:
    https://cloud.google.com/sdk/gcloud/reference/container/clusters/create
    """
    service = discovery.build('container', 'v1', credentials=credentials)
    cluster = {
        'master_auth': {
            'username': 'admin'
        },
        'name': cluster_id,
        'node_pools': [
            {
                'name': 'default-pool',
                'initial_node_count': n_nodes,
                'config': {
                    'machine_type': machine_type,
                    'oauth_scopes': [
                        # Allowing write access to Google Cloud Storage in addition
                        # to the default scopes included in the gcloud command for
                        # creating clusters.
                        'https://www.googleapis.com/auth/devstorage.read_write',
                        'https://www.googleapis.com/auth/compute',
                        'https://www.googleapis.com/auth/devstorage.read_only',
                        'https://www.googleapis.com/auth/service.management.readonly',
                        'https://www.googleapis.com/auth/servicecontrol'
                    ]
                }
            }
        ]
    }

    body = {
        'cluster': cluster
    }
    create = service.projects().zones().clusters().create(body=body, zone=zone, projectId=project_id).execute()

    return create


def get_cluster(project_id, zone, cluster_id):
    service = discovery.build('container', 'v1', credentials=credentials)
    cluster = service.projects().zones().clusters().get(zone=zone, projectId=project_id, clusterId=cluster_id).execute()

    return cluster


def delete_cluster(project_id, zone, cluster_id):
    service = discovery.build('container', 'v1', credentials=credentials)
    delete = service.projects().zones().clusters().delete(zone=zone, projectId=project_id, clusterId=cluster_id).execute()

    return delete

