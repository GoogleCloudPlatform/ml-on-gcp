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

2. [Running TensorFlow inference workloads at scale with TensorRT 5 and NVIDIA T4 GPUs](https://cloud.google.com/blog/products/ai-machine-learning/running-tensorflow-inference-workloads-at-scale-with-tensorrt-5-and-nvidia-t4-gpus) - Creating a demo of ML inference using Tesla T4, TensorFlow, TensorRT, Load balancing and Auto-scale.

3. [NVIDIA’s RAPIDS joins our set of Deep Learning VM images for faster data science](https://cloud.google.com/blog/products/ai-machine-learning/nvidias-rapids-joins-our-set-of-deep-learning-vm-images-for-faster-data-science) - Google Cloud’s set of Deep Learning Virtual Machine (VM) images, which enable the one-click setup machine learning-focused development environments. But some data scientists still use a combination of pandas, Dask, scikit-learn, and Spark on traditional CPU-based instances. If you’d like to speed up your end-to-end pipeline through scale, Google Cloud’s Deep Learning VMs now include an experimental image with RAPIDS, NVIDIA’s open source and Python-based GPU-accelerated data processing and machine learning libraries that are a key part of NVIDIA’s larger collection of CUDA-X AI accelerated software. CUDA-X AI is the collection of NVIDIA's GPU acceleration libraries to accelerate deep learning, machine learning, and data analysis.

4. [Inferring Machine Learning Models from Google Cloud Functions](https://cloud.google.com/blog/products/ai-machine-learning/TBD) - Introduction to Inferring AI Platform models from Google Cloud Function endpoints.

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

2. [Compute Engine burst training](gce/burst-training/README.md) - A guide to
   using powerful VMs to quickly and cheaply perform computationally intensive
   training jobs. (The example training job in this guide uses
   [xgboost](https://github.com/dmlc/xgboost) as well as
   [scikit-learn](http://scikit-learn.org/stable/).)

- - -

## Google Cloud Function

1. [Google Cloud Function + AI Platform Example](gcf/gcf-ai-platform-example/README.md) - Example endpoints to infer AI Plaform models.

- - -

## Example Zoo

Collections of examples adapted to be runnable on [AI Platform](https://cloud.google.com/ai-platform/).

1. [tensorflow-probability examples](/example_zoo/tensorflow/probability).

