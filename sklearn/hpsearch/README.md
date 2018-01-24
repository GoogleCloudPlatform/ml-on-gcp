# Parallelized Hyperparameter Search with Google Container Engine

## Introduction

This sample package helps you run [`scikit-learn`]'s [`GridSearchCV`] and [`RandomizedSearchCV`], and [`scikit-optimize`]'s [`BayesSearchCV`] on [Google Container Engine](https://cloud.google.com/container-engine/).

The design of the workflow is to entirely stay in a [Jupyter notebook], with necessary boilerplate codes abstracted away in the [helpers](helpers/).  Below we highlight some key steps of the workflow.  If you are ready to get started, skip over to [Requirements](#requirements).

For instance, to build the Docker image with Google Cloud Container Builder that will carry out the fitting tasks on a cluster:

```python
from helpers.cloudbuild_helper import build

build(project_id, source_dir, bucket_name, image_name)
```

To create a cluster on Google Cloud Kubernetes Engine:

```python
from helpers.gke_helper import create_cluster

create_cluster(project_id, zone, cluster_id, n_nodes=1, machine_type='n1-standard-64')
```

To better utilize resources, you should choose as few nodes as possible with the same total number of cores.  For example 1 node with 64 cores is beter than 2 nodes with 32 cores each, since in the latter case the fitting task needs to be deployed as two (or more) separate jobs, and it is likely for some of the nodes to idle if they finish their job first.

Once the Docker image is built and a cluster started, you can create a `SearchCV` object in the notebook:

```python
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real

rfc = RandomForestClassifier(n_jobs=-1)
search_spaces = {
    'max_features': Real(0.5, 1.0),
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(5, 45),
    'min_samples_split': Real(0.01, 0.1)
}
search = BayesSearchCV(estimator=rfc, search_spaces=search_spaces, n_jobs=-1, verbose=3, n_iter=100)
```

Calling `search.fit` would fit the `SearchCV` object on the local machine.  To fit the object on the cluster, we first wrap it in a helper object:

```python
from gke_parallel import GKEParallel

gke_search = GKEParallel(search, project_id, zone, cluster_id, bucket_name, image_name)
```

Now we can fit the object on the cluster:

```python
gke_search.fit(X_train, y_train)
```

To check whether the fitting tasks have completed:

```python
gke_search.done()
```

Once complete, the helper object can be used for some tasks supported by the original `SearchCV` object:

```python
y_predicted = gke_search.predict(X_test)
```


[`scikit-learn`]: http://scikit-learn.org/
[`GridSearchCV`]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
[`RandomizedSearchCV`]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
[`scikit-optimize`]: https://scikit-optimize.github.io/
[`BayesSearchCV`]: https://scikit-optimize.github.io/#skopt.BayesSearchCV
[Jupyter notebook]: https://jupyter.org/

## Requirements

You will need a Google Cloud Platform project which has the following products [enabled](https://support.google.com/cloud/answer/6158841?hl=en):

- [Google Container Registry](https://cloud.google.com/container-registry/)

- [Google Container Engine](https://cloud.google.com/container-engine/)

- [Google Cloud Storage](https://cloud.google.com/storage/)


In addition, to follow the steps of the sample we recommend you work in a [Jupyter notebook] running [Python](https://www.python.org/) v2.7.10 or newer.


## Before you start

1. Install [Google Cloud Platform SDK](https://cloud.google.com/sdk/downloads).

1. Install [kubectl](https://cloud.google.com/container-engine/docs/quickstart).

1. Run `git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git`

1. Run `cd ml-on-gcp/sklearn/hpsearch`

1. Run `pip install -r requirements.txt`

1. Follow the steps in one of the notebooks:

	- [GridSearchCV Notebook](gke_grid_search.ipynb)

	- [RandomizedSearchCV Notebook](gke_randomized_search.ipynb)

	- [BayesSearchCV Notebook](gke_bayes_search.ipynb).
