# Machine Learning on Google Cloud Platform

Guides to bringing your code from various Machine Learning frameworks
to Google Cloud Platform.

The goal is to present recipes and practices that will help you spend
less time wrangling with the various interfaces and more time exploring your
datasets, building your models, and in general solving the problems you
really care about.

- - -

## Blog posts

1. [Genomic ancestry inference with deep learning](https://cloud.google.com/blog/big-data/2017/09/genomic-ancestry-inference-with-deep-learning) - Ancestry inference on Google Cloud Platform using the [1000 Genomes dataset](https://cloud.google.com/genomics/data/1000-genomes)

- - -

## TensorFlow

1. [Estimators](tensorflow/tf-estimators.ipynb) - A guide to the Estimator
   interface.


- - -

## scikit-learn

1. [scikit-learn on GCE](sklearn/titanic) - Train a simple model with scikit-learn on a Google Compute Engine

2. [Model serve](sklearn/gae_serve) - Serve model with Google App Engine and Cloud Endpoints.

3. [Hyperparameter search](sklearn/hpsearch) - Hyperparameter search on a Google Kubernetes Engine cluster from a Jupyter notebook.

- - -

## Google Compute Engine

1. [Compute Engine survival training](gce/survival-training/README.md) - Introduces a framework for running resilient training jobs on Google Compute Engine.

1. [Compute Engine burst training](gce/burst-training/README.md) - A guide to
   using powerful VMs to quickly and cheaply perform computationally intensive
   training jobs. (The example training job in this guide uses
   [xgboost](https://github.com/dmlc/xgboost) as well as
   [scikit-learn](http://scikit-learn.org/stable/).)

- - -

## Cloud TPU

1. [Hyperparameter Tuning with tf.metrics](tpu/hptuning/resnet/README.md) - Run hyperparameter tuning jobs on Cloud Machine Learning Engine using `tf.metrics` for a ResNet model.

1. [Hyperparameter Tuning with cloudml-hypertune](tpu/hptuning/resnet-hypertune/README.md) - Run hyperparameter tuning jobs on Cloud Machine Learning Engine using `cloudml-hypertune` for a ResNet model.
