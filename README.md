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

4. [NVIDIA Achieves Breakthroughs in Language Understanding to Enable Real-Time Conversational AI](https://nvidianews.nvidia.com/news/nvidia-achieves-breakthroughs-in-language-understandingto-enable-real-time-conversational-ai?ncid=so-elev-49597#cid=organicSocial_en-us_Elevate_Deep-Learning-AI-for-Developers-DL13) - BERT Notebook in AI Hub and AI Platform Notebooks

- - -

## TensorFlow

1. [Estimators](tutorials/tensorflow/tf-estimators.ipynb) - A guide to the Estimator
   interface.


- - -

## scikit-learn

1. [scikit-learn on GCE](tutorials/sklearn/titanic) - Train a simple model with scikit-learn on a Google Compute Engine

2. [Model serve](tutorials/sklearn/gae_serve) - Serve model with Google App Engine and Cloud Endpoints.

3. [Hyperparameter search](tutorials/sklearn/hpsearch) - Hyperparameter search on a Google Kubernetes Engine cluster from a Jupyter notebook.

- - -

## Google Compute Engine

1. [Compute Engine survival training](gce/survival-training/README.md) - Introduces a framework for running resilient training jobs on Google Compute Engine.

2. [Compute Engine burst training](gce/burst-training/README.md) - A guide to
   using powerful VMs to quickly and cheaply perform computationally intensive
   training jobs. (The example training job in this guide uses
   [xgboost](https://github.com/dmlc/xgboost) as well as
   [scikit-learn](http://scikit-learn.org/stable/).)

- - -

## Google Cloud Functions

1. [Google Cloud Functions + AI Platform Example](gcf/gcf-ai-platform-example/README.md) - Example endpoints to infer AI Platform models.

- - -

## Example Zoo

Collections of examples adapted to be runnable on [AI Platform](https://cloud.google.com/ai-platform/).

1. [tensorflow-probability examples](/example_zoo/tensorflow/probability).

1. [tensorflow-models examples](/example_zoo/tensorflow/models).

## Google Machine Learning Repositories

If you’re looking for our guides on how to do Machine Learning on Google Cloud Platform (GCP) using other services, please checkout our other repositories: 

- [AI Platform samples](https://github.com/GoogleCloudPlatform/ai-platform-samples), which has guides on how to bring your code from various ML frameworks to Google Cloud AI Platform using different products such as AI Platform Training, Prediction, Notebooks and AI Hub.
- [Keras Idiomatic Programmer](https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer) This repository contains content produced by Google Cloud AI Developer Relations for machine learning and artificial intelligence. The content covers a wide spectrum from educational, training, and research, covering from novices, junior/intermediate to advanced.
- [Professional Services](https://github.com/GoogleCloudPlatform/professional-services), common solutions and tools developed by Google Cloud's Professional Services team.
