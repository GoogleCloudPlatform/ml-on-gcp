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

"""The worker code that will be deployed as Kubernetes jobs to a cluster.

`execute`: Gets data and a pickled copy of a SearchCV object from GCS, calls
the `fit` method on that object, and persist the fitted object to GCS.
"""

import argparse
import datetime
import logging
import pickle
import re
from google.cloud import storage
from gcs_helper import pickle_and_upload, download_and_unpickle, download_uri_and_unpickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV


def execute(bucket_name, task_name, worker_id, X_uri, y_uri):
    X = download_uri_and_unpickle(X_uri)
    y = download_uri_and_unpickle(y_uri)
    search = download_and_unpickle(bucket_name, '{}/search.pkl'.format(task_name))

    if type(search) == GridSearchCV:
        param_grid = download_and_unpickle(bucket_name, '{}/{}/param_grid.pkl'.format(task_name, worker_id))
        search.param_grid = param_grid

    elif type(search) == RandomizedSearchCV:
        param_distributions = download_and_unpickle(bucket_name, '{}/{}/param_distributions.pkl'.format(task_name, worker_id))
        n_iter = download_and_unpickle(bucket_name, '{}/{}/n_iter.pkl'.format(task_name, worker_id))
        search.param_distributions = param_distributions
        search.n_iter = n_iter

    elif type(search) == BayesSearchCV:
        search_spaces = download_and_unpickle(bucket_name, '{}/{}/search_spaces.pkl'.format(task_name, worker_id))
        search.search_spaces = search_spaces

    # Calling `search.fit` on the pickled of the SearchCV object, in particular
    # this will be using the same `n_jobs` as when the original copy of the
    # object created in the notebook.
    search.fit(X, y)

    pickle_and_upload(search, bucket_name, '{}/{}/fitted_search.pkl'.format(task_name, worker_id))

    # Save a copy of the search object without the estimator, useful when the
    # user only wants to examine the scores without having to download the
    # estimator, which can sometimes be large.
    del search.best_estimator_
    pickle_and_upload(search, bucket_name, '{}/{}/fitted_search_without_estimator.pkl'.format(task_name, worker_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # The order of arguments needs to be kept the same as the order specified
    # in `../gke_parallel.py`'s `_make_job_body` method.
    parser.add_argument('bucket_name', type=str)
    parser.add_argument('task_name', type=str)
    parser.add_argument('worker_id', type=str)

    parser.add_argument('X_uri', type=str)
    parser.add_argument('y_uri', type=str)

    args = parser.parse_args()

    execute(args.bucket_name, args.task_name, args.worker_id, args.X_uri, args.y_uri)
