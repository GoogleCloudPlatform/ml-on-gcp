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

import yaml
from kubernetes import client, config

# brew install python to get 2.7.13 which has updated openssl
# check openssl version with python -c "import ssl; print ssl.OPENSSL_VERSION"
# mkvirtualenv -p /usr/local/Cellar/python/2.7.13_1/bin/python2 hpsearch 


def get_nodes():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    nodes = v1.list_node()
    
    return nodes


def create_job(job_body, namespace='default'):
    config.load_kube_config()
    v1 = client.BatchV1Api()

    job = v1.create_namespaced_job(body=job_body, namespace=namespace)
    return job


def create_job_from_file(job_filename, namespace='default'):
    with open(job_filename, 'r') as f:
        job_body = yaml.load(f)
    
    job = create_job(job_body, namespace)
    return job


def get_pod_logs(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()

    pod_list = v1.list_namespaced_pod(namespace=namespace)

    result = {}
    for pod in pod_list.items:
        pod_name = pod.metadata.name
        print('getting logs from pod {}'.format(pod_name))

        result[pod_name] = v1.read_namespaced_pod_log(pod_name, namespace)

    return result


def delete_job(job_name, namespace='default'):
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()

    print('deleting job {} with namespace {}'.format(job_name, namespace))
    delete = batch_v1.delete_namespaced_job(name=job_name, body=client.V1DeleteOptions(), namespace=namespace)
    return delete


def delete_pod(pod_name, namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()

    print('deleting pod {} with namespace {}'.format(pod_name, namespace))
    delete = v1.delete_namespaced_pod(name=pod_name, body=client.V1DeleteOptions(), namespace=namespace)
    return delete


def delete_jobs_pods(job_names, namespace='default'):
    for job_name in job_names:
        delete_job(job_name, namespace)

    config.load_kube_config()
    v1 = client.CoreV1Api()

    pod_list = v1.list_namespaced_pod(namespace=namespace)

    for pod in pod_list.items:
        if pod.metadata.labels['job-name'] in job_names:
            delete_pod(pod.metadata.name, namespace)

