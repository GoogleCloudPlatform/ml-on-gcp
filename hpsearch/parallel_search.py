from sklearn.model_selection import GridSearchCV

from sklearn.externals.joblib._parallel_backends import ParallelBackendBase


#from google.cloud import storage

class GridSearchCVGCP(GridSearchCV):
    def __init__(self, n_worker, **kwargs):
        super(GridSearchCVGCP, self).__init__(**kwargs)

######

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

credentials = GoogleCredentials.get_application_default()

service = discovery.build('container', 'v1', credentials=credentials)


#####
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from google.cloud import storage

import shutil

shutil.make_archive('source', 'zip', 'source')

sc = storage.Client()

bucket = sc.get_bucket('hpsearch')
blob = bucket.blob('source.zip')
blob.upload_from_filename('source.zip')



credentials = GoogleCredentials.get_application_default()

project_id = 'rising-sea-112358'
service = discovery.build('cloudbuild', 'v1', credentials=credentials)
builds = service.projects().builds().list(projectId=project_id).execute()


body = {
    'source': {
        'storageSource': {
            'bucket': 'hpsearch',
            'object': 'source.zip'
        }
    },
    'steps': [
        {
            'name': 'gcr.io/cloud-builders/docker',
            'args': [
                'build',
                '-t',
                'gcr.io/$PROJECT_ID/hpsearch',
                '/workspace'
            ]
        }
    ],
    'images': [
        'gcr.io/$PROJECT_ID/hpsearch'
    ]
}

build = service.projects().builds().create(projectId=project_id, body=body).execute()

#####

# create a cluster
zone = 'us-central1-b'
project_id = 'rising-sea-112358'

cluster = {
    'name': 'test-cluster',
    'initial_node_count': 1
}

body = {
    'cluster': cluster
}

create = service.projects().zones().clusters().create(body=body, zone=zone, projectId=project_id).execute()

clusters = service.projects().zones().clusters().list(zone=zone, projectId=project_id).execute()

cluster_id = cluster['name']

cluster = service.projects().zones().clusters().get(zone=zone, projectId=project_id, clusterId=cluster_id).execute()

if False:
    delete = service.projects().zones().clusters().delete(zone=zone, projectId=project_id, clusterId=cluster_id).execute()



####
from kubernetes import client, config
# gcloud container get-server-config
# kubectl get credntial to make sure config loads
config.load_kube_config()


v1 = client.CoreV1Api()

# brew install python to get 2.7.13 which has updated openssl
# check openssl version with python -c "import ssl; print ssl.OPENSSL_VERSION"
# mkvirtualenv -p /usr/local/Cellar/python/2.7.13_1/bin/python2 hpsearch  

v1.list_node()

klient.get_api_resources_with_http_info()


klient.list_pod_for_all_namespaces(watch=False)




"""
steps:
    create wrapper SearchCVGCP object specifying GCS bucket (and estimator)
    [e.g. grid_search = GridSearchCV(); search = SearchGCP(grid_search, ...)]
    upload source (part of sample) to GCS
    cloudbuild docker image from source on GCS (.zip), passing in paths to estimator, X, y(LRO1)
    create cluster with prescribed number of nodes(LRO2)
    upload pickled estimator, X, y to GCS (in timestamp folder)
    (check LRO1 LRO2)
    deploy containers passing in partial parameter grids
    source sends output data to GCS (same bucket) [use callback to store partial results?]
    delete cluster when all jobs are done [how to tell?]
    wrapper SearchCVGCP object has method to retrieve the output data and merge them.
    allow now instance wrapper object to retrieve previous runs by specifying bucket name
"""
