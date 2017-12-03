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


import time
from gke_helper import get_cluster
from gcs_helper import pickle_and_upload, get_uri_blob, download_uri_and_unpickle
from kubernetes_helper import create_job, delete_jobs_pods
from copy import deepcopy
from itertools import product
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class GKEParallel(object):
    SUPPORTED_SEARCH = [GridSearchCV, RandomizedSearchCV]

    def __init__(self, search, project_id, zone, cluster_id, bucket_name, image_name, task_name=None):
        if type(search) not in self.SUPPORTED_SEARCH:
            raise TypeError('Search type {} not supported.  Only supporting {}.'.format(type(search), [s.__name__ for s in self.SUPPORTED_SEARCH]))

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

        # For GridSearchCV
        self.param_grids = {}
        # For RandomizedSearchCV
        self.param_distributions = None
        self.n_iter = None

        self.job_names = {}
        self.output_uris = {}
        self.output_without_estimator_uris = {}
        self.dones = {}
        self.results = {}

        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_search_ = None

        self._cancelled = False
        self._done = False


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


    def _deploy_job(self, worker_id, X_uri, y_uri):
        job_body = self._make_job_body(worker_id, X_uri, y_uri)

        print('Deploying worker {}'.format(worker_id))
        create_job(job_body)


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


    def _handle_grid_search(self, X_uri, y_uri, per_node):
        param_grids = self._expand_param_grid(self.search.param_grid, per_node * self.n_nodes)

        for i, param_grid in enumerate(param_grids):
            worker_id = str(i)

            self.param_grids[worker_id] = param_grid
            self.job_names[worker_id] = self._make_job_name(worker_id)
            self.output_uris[worker_id] = 'gs://{}/{}/{}/fitted_search.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.output_without_estimator_uris[worker_id] = 'gs://{}/{}/{}/fitted_search_without_estimator.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.dones[worker_id] = False

            pickle_and_upload(param_grid, self.bucket_name, '{}/{}/param_grid.pkl'.format(self.task_name, worker_id))

            self._deploy_job(worker_id, X_uri, y_uri)


    def _handle_randomized_search(self, X_uri, y_uri, per_node):
        self.param_distributions = self.search.param_distributions
        self.n_iter = self.search.n_iter
        n_iter = self.n_iter / (per_node * self.n_nodes) + 1

        for i in xrange(per_node * self.n_nodes):
            worker_id = str(i)

            self.job_names[worker_id] = self._make_job_name(worker_id)
            self.output_uris[worker_id] = 'gs://{}/{}/{}/fitted_search.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.output_without_estimator_uris[worker_id] = 'gs://{}/{}/{}/fitted_search_without_estimator.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.dones[worker_id] = False

            pickle_and_upload(self.param_distributions, self.bucket_name, '{}/{}/param_distributions.pkl'.format(self.task_name, worker_id))
            pickle_and_upload(n_iter, self.bucket_name, '{}/{}/n_iter.pkl'.format(self.task_name, worker_id))

            self._deploy_job(worker_id, X_uri, y_uri)


    def _upload_data(self, X, y):
        if type(X) == str and X.startswith('gs://'):
            X_uri = X
        else:
            X_uri = pickle_and_upload(X, self.bucket_name, '{}/X.pkl'.format(self.task_name))

        if type(y) == str and y.startswith('gs://'):
            y_uri = y
        else:
            y_uri = pickle_and_upload(y, self.bucket_name, '{}/y.pkl'.format(self.task_name))

        search_uri = pickle_and_upload(self.search, self.bucket_name, '{}/search.pkl'.format(self.task_name))

        return X_uri, y_uri, search_uri


    def fit(self, X, y, per_node=2):
        """Returns an `operation` object that implements `done()` and `result()`.
        """
        timestamp = str(int(time.time()))
        self.task_name = self.task_name or '{}.{}.{}'.format(self.cluster_id, self.image_name, timestamp)
        self._done = False
        self._cancelled = False

        X_uri, y_uri, _ = self._upload_data(X, y)

        if type(self.search) == GridSearchCV:
            handler = self._handle_grid_search
        elif type(self.search) == RandomizedSearchCV:
            handler = self._handle_randomized_search

        print('Fitting {}'.format(type(self.search)))
        handler(X_uri, y_uri, per_node)

        self.persist()


    def persist(self):
        self.gcs_uri = pickle_and_upload(self, self.bucket_name, '{}/gke_search.pkl'.format(self.task_name))
        print('Persisted the GKEParallel instance: {}'.format(self.gcs_uri))


    # Implement part of the concurrent.future.Future interface.
    def done(self):
        # TODO: consider using kubernetes API to check if pod completed
        if not self._done:
            for worker_id, output_uri in self.output_uris.items():
                print('Checking if worker {} is done'.format(worker_id))
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


    # TODO: allow getting only the best result to save time
    def result(self, download=False):
        if not self.done():
            n_done = len(d for d in self.dones.values() if d)
            print('Not done: {} out of {} workers completed.'.format(n_done, len(self.dones)))
            return None

        if not self.results:
            for worker_id, output_uri in self.output_without_estimator_uris.items():
                print('Getting result from worker {}'.format(worker_id))
                self.results[worker_id] = download_uri_and_unpickle(output_uri)

            self._aggregate_results(download)

        return self.results


    def _aggregate_results(self, download):
        best_id = None
        for worker_id, result in self.results.items():
            if self.best_score_ is None or result.best_score_ > self.best_score_:

                self.best_score_ = result.best_score_
                self.best_params_ = result.best_params_

                best_id = worker_id

        if download:
            # Download only the best estimator among the workers.
            print('Downloading the best estimator (worker {}).'.format(best_id))
            output_uri = self.output_uris[best_id]
            self.best_search_ = download_uri_and_unpickle(output_uri)
            self.best_estimator_ = self.best_search_.best_estimator_


    # Implement part of SearchCV interface by delegation.
    def predict(self, *args, **kwargs):
        return self.best_estimator_.predict(*args, **kwargs)


    def predict_proba(self, *args, **kwargs):
        return self.best_estimator_.predict_proba(*args, **kwargs)


    def predict_log_proba(self, *args, **kwargs):
        return self.best_estimator_.predict_log_proba(*args, **kwargs)

