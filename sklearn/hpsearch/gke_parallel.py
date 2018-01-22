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
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


class GKEParallel(object):
    SUPPORTED_SEARCH = [
        GridSearchCV,
        RandomizedSearchCV,
        BayesSearchCV
    ]

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
        # For BayesSearchCV
        self.search_spaces = {}

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


    def _partition_grid(self, param_grid_dict, partition_keys):
        _param_grid_dict = deepcopy(param_grid_dict)

        partition_lists = [_param_grid_dict.pop(key) for key in partition_keys]
        
        partitioned = []
        for prod in product(*partition_lists):
            lists = [[element] for element in prod]
            singleton = dict(zip(partition_keys, lists))
            singleton.update(_param_grid_dict)

            partitioned.append(singleton)

        return partitioned


    def _partition_param_grid(self, param_grid, target_n_partition=5):
        """Returns a list of param_grids whose union is the input
        param_grid.

        If param_grid is a dict:

        The implemented strategy attempts to partition the param_grid
        into at least target_n_partition smaller param_grids.

        NOTE: The naive strategy implemented here does not distinguish
        between different types of parameters nor their impact on the
        running time.  The user of this module is encouraged to
        implement their own paritioning strategy based on their needs.
        """
        if type(param_grid) == list:
            # If the input is already a list of param_grids then just
            # use it as is.
            return param_grid
        else:
            # The strategy is to simply expand the grid fully with
            # respect to a parameter:
            # [1, 2, 3]x[4, 5] --> [1]x[4, 5], [2]x[4, 5], [3]x[4, 5]
            # until the target number of partitions is reached.
            partition_keys = []
            n_partition = 1
            for key, lst in param_grid.items():
                partition_keys.append(key)
                n_partition *= len(lst)

                if n_partition >= target_n_partition:
                    break

            partitioned = self._partition_grid(param_grid, partition_keys)

            return partitioned


    def _handle_grid_search(self, X_uri, y_uri):
        param_grids = self._partition_param_grid(self.search.param_grid, self.n_nodes)

        for i, param_grid in enumerate(param_grids):
            worker_id = str(i)

            self.param_grids[worker_id] = param_grid
            self.job_names[worker_id] = self._make_job_name(worker_id)
            self.output_uris[worker_id] = 'gs://{}/{}/{}/fitted_search.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.output_without_estimator_uris[worker_id] = 'gs://{}/{}/{}/fitted_search_without_estimator.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.dones[worker_id] = False

            pickle_and_upload(param_grid, self.bucket_name, '{}/{}/param_grid.pkl'.format(self.task_name, worker_id))

            # TODO: Make sure that each job is deployed to a different node.
            self._deploy_job(worker_id, X_uri, y_uri)


    def _handle_randomized_search(self, X_uri, y_uri):
        self.param_distributions = self.search.param_distributions
        self.n_iter = self.search.n_iter
        n_iter = self.n_iter / self.n_nodes + 1

        for i in xrange(self.n_nodes):
            worker_id = str(i)

            self.job_names[worker_id] = self._make_job_name(worker_id)
            self.output_uris[worker_id] = 'gs://{}/{}/{}/fitted_search.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.output_without_estimator_uris[worker_id] = 'gs://{}/{}/{}/fitted_search_without_estimator.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.dones[worker_id] = False

            pickle_and_upload(self.param_distributions, self.bucket_name, '{}/{}/param_distributions.pkl'.format(self.task_name, worker_id))
            pickle_and_upload(n_iter, self.bucket_name, '{}/{}/n_iter.pkl'.format(self.task_name, worker_id))

            self._deploy_job(worker_id, X_uri, y_uri)


    def _partition_space(space):
        """Partitions the space into two subspaces.  In the case of
        Real and Integer, the subspaces are not disjoint, but
        overlapping at an endpoint.
        """

        partitioned = [space]
        if type(space) == Categorical:
            if len(space.categories) >= 2:
                mid_index = len(space.categories) / 2
                left_categories = space.categories[:mid_index]
                right_categories = space.categories[mid_index:]

                if space.prior is not None:
                    left_prior = space.prior[:mid_index]
                    left_weight = sum(left_prior)
                    left_prior = [p/left_weight for p in left_prior]

                    right_prior = space.prior[mid_index:]
                    right_weight = sum(right_prior)
                    right_prior = [p/right_weight for p in right_prior]
                else:
                    left_prior = None
                    right_prior = None

                left = Categorical(left_categories, prior=left_prior, transform=space.transform, name=space.name)
                right = Categorical(right_categories, prior=right_prior, transform=space.transform, name=space.name)

        elif type(space) == Integer:
            mid = int((high - low) / 2)
            left = Integer(low, mid, transform=space.transform, name=space.name)
            right = Integer(mid, high, transform=space.transform, name=space.name)

            partitioned = [left, right]

        elif type(space) == Real:
            mid = (high - low) / 2
            left = Real(low, mid, prior=space.prior, transform=space.transform, name=space.name)
            right = Real(mid, high, prior=space.prior, transform=space.transform, name=space.name)

            partitioned = [left, right]

        return partitioned


    def _partition_search_spaces(self, search_spaces, target_n_partition=5):
        """Returns a list of search_spaces whose union is the input
        search_spaces.

        If search_spaces is a dict:

        The implemented strategy attempts to partition the search_spaces
        into at least target_n_partition smaller search_spaces.

        NOTE: The search_spaces format list(dict, int>0) is not supported
        by this implementation.

        NOTE: The naive strategy implemented here does not distinguish
        between different types of parameters nor their impact on the
        running time.  The user of this module is encouraged to
        implement their own paritioning strategy based on their needs.
        """
        if type(search_spaces) == list:
            # If the input is already a list of search_spaces then just
            # use it as is.
            return search_spaces
        else:
            # TODO: implement this
            partitioned = [search_spaces]
            while len(partitioned) < target_n_partition:
                break

            return partitioned


    def _handle_bayes_search(self, X_uri, y_uri):
        partitioned_search_spaces = self._partition_search_spaces(self.search.search_spaces_, self.n_nodes)

        for i, search_spaces in enumerate(partitioned_search_spaces):
            worker_id = str(i)

            self.search_spaces[worker_id] = search_spaces
            self.job_names[worker_id] = self._make_job_name(worker_id)
            self.output_uris[worker_id] = 'gs://{}/{}/{}/fitted_search.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.output_without_estimator_uris[worker_id] = 'gs://{}/{}/{}/fitted_search_without_estimator.pkl'.format(self.bucket_name, self.task_name, worker_id)
            self.dones[worker_id] = False

            pickle_and_upload(search_spaces, self.bucket_name, '{}/{}/search_spaces.pkl'.format(self.task_name, worker_id))

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


    def fit(self, X, y):
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
        elif type(self.search) == BayesSearchCV:
            handler = self._handle_bayes_search

        print('Fitting {}'.format(type(self.search)))
        handler(X_uri, y_uri)

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

        if not self.results or download:
            for worker_id, output_uri in self.output_without_estimator_uris.items():
                print('Getting result from worker {}'.format(worker_id))
                self.results[worker_id] = download_uri_and_unpickle(output_uri)

            self._aggregate_results(download)

            self.persist()

        return self.results


    def _aggregate_results(self, download):
        best_id = None
        for worker_id, result in self.results.items():
            if self.best_score_ is None or result.best_score_ > self.best_score_:

                self.best_score_ = result.best_score_
                self.best_params_ = result.best_params_

                best_id = worker_id

        if download and self.best_estimator_ is None:
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

