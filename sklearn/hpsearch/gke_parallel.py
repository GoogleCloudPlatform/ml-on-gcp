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


import logging
import time
from gke_helper import get_cluster
from gcs_helper import pickle_and_upload, get_uri_blob, download_uri_and_unpickle
from kubernetes_helper import create_job, delete_jobs_pods
from copy import deepcopy
from itertools import product


class GKEParallel(object):
    def __init__(self, search, project_id, zone, cluster_id, bucket_name, image_name, task_name=None):
        self.search = search
        self.project_id = project_id
        self.cluster_id = cluster_id
        self.bucket_name = bucket_name
        self.image_name = image_name
        self.task_name = task_name
        self.gcs_uri = None

        self.cluster = get_cluster(project_id, zone, cluster_id)
        self.n_nodes = self.cluster['currentNodeCount']

        self.task_name = None
        self.param_grids = {}
        self.job_names = {}
        self.output_uris = {}
        self.dones = {}
        self.results = {}

        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

        self._cancelled = False
        self._done = False


    def _expand(self, param_grid_dict, expand_keys):
        _param_grid_dict = deepcopy(param_grid_dict)

        expand_lists = [_param_grid_dict.pop(key) for key in expand_keys]
        
        expanded = []
        for prod in product(*expand_lists):
            lists = [[element] for element in prod]
            singleton = dict(zip(expand_keys, lists))
            singleton.update(_param_grid_dict)

            expanded.append(singleton)

        return expanded


    def _expand_param_grid(self, param_grid, target_fold=5):
        """Returns a list of param_grids.

        If param_grid is a dict:

        The implemented strategy attempts to expand the param_grid
        into at least target_fold smaller param_grids.
        """
        if type(param_grid) == list:
            return param_grid
        else:
            expand_keys = []
            n_fold = 1
            for key, lst in param_grid.items():
                expand_keys.append(key)
                n_fold *= len(lst)

                if n_fold >= target_fold:
                    break

            expanded = self._expand(param_grid, expand_keys)

            assert len(expanded) == n_fold

            return expanded


    def _make_job_name(self, worker_id):
        return '{}.worker.{}'.format(self.task_name, worker_id)


    def _make_job_body(self, worker_id, X_uri, y_uri):
        body = {
            'apiVersion': 'batch/v1',
            'kind': 'Job',
            'metadata': {
                'name': self._make_job_name(worker_id)
            },
            'spec': {
                'template': {
                    'spec': {
                        'containers': [
                            {
                                'image': 'gcr.io/{}/{}'.format(self.project_id, self.image_name),
                                'command': ['python'],
                                'args': ['worker.py', self.bucket_name, self.task_name, worker_id, X_uri, y_uri],
                                'name': 'worker'
                            }
                        ],
                'restartPolicy': 'OnFailure'}
                }
            }
        }

        return body


    def fit(self, X, y, per_node=2):
        """Returns an `operation` object that implements `done()` and `result()`.
        """
        timestamp = str(int(time.time()))
        self.task_name = self.task_name or '{}.{}.{}'.format(self.cluster_id, self.image_name, timestamp)
        self._done = False
        self._cancelled = False

        if not(type(X) == str and X.startswith('gs://')):
            X_uri = pickle_and_upload(X, self.bucket_name, '{}/X.pkl'.format(self.task_name))

        if not(type(y) == str and y.startswith('gs://')):
            y_uri = pickle_and_upload(y, self.bucket_name, '{}/y.pkl'.format(self.task_name))

        pickle_and_upload(self.search, self.bucket_name, '{}/search.pkl'.format(self.task_name))

        param_grids = self._expand_param_grid(self.search.param_grid, per_node * self.n_nodes)

        for i, param_grid in enumerate(param_grids):
            worker_id = str(i)

            self.param_grids[worker_id] = param_grid
            self.job_names[worker_id] = self._make_job_name(worker_id)
            self.output_uris[worker_id] = 'gs://{}/{}/{}/fitted_search.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.dones[worker_id] = False

            pickle_and_upload(param_grid, self.bucket_name, '{}/{}/param_grid.pkl'.format(self.task_name, worker_id))

            job_body = self._make_job_body(worker_id, X_uri, y_uri)

            logging.info('Deploying worker {}'.format(worker_id))
            create_job(job_body)


        self.persist()


    def persist(self):
        self.gcs_uri = pickle_and_upload(self, self.bucket_name, '{}/gke_search.pkl'.format(self.task_name))
        print('Persised the GKEParallel instance: {}'.format(self.gcs_uri))


    # Implement part of the concurrent.future.Future interface.
    def done(self):
        # TODO: consider using kubernetes API to check if pod completed
        if not self._done:
            for worker_id, output_uri in self.output_uris.items():
                logging.info('Checking if worker {} is done'.format(worker_id))
                self.dones[worker_id] = get_uri_blob(output_uri).exists()

            self._done = all(self.dones.values())

        return self._done


    def cancel(self):
        """Deletes the kubernetes jobs, but completed data remains."""
        if not self._cancelled:
            delete_jobs_pods(self.job_names.values())
            self._cancelled = True


    def cancelled(self):
        return self._cancelled


    def result(self):
        if not self.done():
            n_done = len(d for d in self.dones.values() if d)
            print('Not done: {} out of {} workers completed.'.format(n_done, len(self.dones)))
            return None

        if not self.results:
            for worker_id, output_uri in self.output_uris.items():
                logging.info('Getting result from worker {}'.format(worker_id))
                self.results[worker_id] = download_uri_and_unpickle(output_uri)

            self._aggregate_results()
            self.persist()

        return self.results


    def _aggregate_results(self):
        result = self.results.values()[0]
        
        self.best_score_ = result.best_score_
        self.best_params_ = result.best_params_
        self.best_estimator_ = result.best_estimator_

        for result in self.results.values()[1:]:
            if result.best_score_ > self.best_score_:
                self.best_score_ = result.best_score_
                self.best_params_ = result.best_params_
                self.best_estimator_ = result.best_estimator_


    # TODO: test and deligate other methods also to self.best_estimator_
    # such as predict_proba.
    def predict(self, *args, **kwargs):
        return self.best_estimator_.predict(*args, **kwargs)






