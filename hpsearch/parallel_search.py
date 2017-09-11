from sklearn.model_selection import GridSearchCV

from sklearn.externals.joblib._parallel_backends import ParallelBackendBase


#from google.cloud import storage

class GridSearchCVGCP(GridSearchCV):
    def __init__(self, n_worker, **kwargs):
        super(GridSearchCVGCP, self).__init__(**kwargs)

######

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from kubernetes import client, config

credentials = GoogleCredentials.get_application_default()

service = discovery.build('container', 'v1', credentials=credentials)

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
config.kube_config.configuration.host = clusters['clusters'][0]['selfLink']

klient = client.CoreV1Api()

klient.get_api_resources_with_http_info()


klient.list_pod_for_all_namespaces(watch=False)
