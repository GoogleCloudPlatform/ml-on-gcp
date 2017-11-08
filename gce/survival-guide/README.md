# The GCE survival guide

[Google Compute Engine](https://cloud.google.com/compute/docs/) (GCE) gives you direct access to the computational backbone of Google Cloud Platform. You can use GCE to create virtual machines provisioned with the resources of your choosing ([satisfying the constraints of any quotas that you define](https://cloud.google.com/compute/quotas)).

This has particular value when it comes to training machine learning models, as this gives you access to resources beyond those available to you physically. For example, GCE will soon offer preemptible GPU instances. This means that, if you designed your trainer appropriately, your cost per training step could be a fraction of what it would be on a dedicated instance.

Even if you want to train on a dedicated instance, GPU-enabled GCE instances have always had the issue that they go down about once a week for maintenance. This introduces a significant amount of overhead (in both time and money) to performing training on GCE instances.

However, GCE *does* expose primitives that allow you set up training jobs and your GCE instances in a manner that lets you train across instance preemptions and shutdowns.

This guide provides you with a template to do exactly that.


## Survival strategy

There are five components to our resilient training jobs:

1. [GCE custom images](https://cloud.google.com/compute/docs/images#custom_images)

1. A trainer command line interface

1. [GCE startup scripts](https://cloud.google.com/compute/docs/startupscript)

1. [GCE instance metadata](https://cloud.google.com/compute/docs/storing-retrieving-metadata)

1. [Cloud Source Repositories](https://cloud.google.com/source-repositories/)


As a part of this process, if you have not already done so, you will be prompted to enable both the Compute Engine API as well as the Cloud Source Repositories API for your Google Cloud Platform project.

We will be working through the [Google Cloud SDK](https://cloud.google.com/sdk/). If you have not already done so, it is worth installing.


### Custom images

Your training job may require specific versions of libraries, like [tensorflow](https://www.tensorflow.org/) or [scikit-learn](http://scikit-learn.org). Additionally, in order to take advantage of instances with attached GPUs, you will have to have hardware-specific drivers installed on the instance. [Custom images](https://cloud.google.com/compute/docs/images#custom_images) are a convenient solution to this problem of environment reproduction within a GCE virtual machine.

If you do have requirements (in addition to simply the OS) in your training environments, it is highly recommended that you create a VM image that you can use to spawn new instances without having to go through the tedious configuration process every time. The examples following this strategy section will demonstrate the creation of such images so, if this seems daunting, don't worry.


### Trainer CLI

Our strategy will have the GCE instance automatically kick off your training job on startup. As part of this kick off, it will have to provide the job with the appropriate parameters -- how many steps to train for, how often to checkpoint the model, which data to train the model with, what hyperparameters should be used in training the model, etc.

It is helpful, if you intend to do this more than once, to assert a common semantical system across all your trainers. We provide a boilerplate CLI in this repo. This interface is very similar to that provided by [TensorFlow estimators](https://www.tensorflow.org/programmers_guide/estimators), but can be used even with other frameworks. In the sections that follow, we will provide you with both TensorFlow and with scikit-learn examples that demonstrate its use.


### Startup scripts

GCE instances have an in-built mechanism that triggers a script on instance startup (for the appropriate definition of startup, but this need not concern us for now). This is the mechanism we will use to kick off a training job or to pick it up where it left off before a shutdown. We will make use of it through [startup scripts](https://cloud.google.com/compute/docs/startupscript#troubleshooting).

It is alright if you are not familiar with these things. What is more important at the moment is to know that they exist and have a general sense for how they fit into the training process.


### Instance metadata

You may want to deploy different parametrizations of a training job to multiple GCE instances at different times. For this purpose, [we will make use of the metadata server available to every GCE instance](https://cloud.google.com/compute/docs/storing-retrieving-metadata#custom).

Instance metadata is where we will store things like model hyperparameters and paths to training and evaluation data. We will store metadata on our instances before we start our training jobs, and our startup scripts will make use of this metadata on instance startup.

If you are using the [example startup script provided in this guide](./gce/startup.sh), you will only have to make minimal changes to do so. The important thing again is to understand where instance metadata fits into our picture and that GCE guarantees that it will be available to our startup script when it runs.


### Cloud Source Repositories

Finally, we need a mechanism by which we can get our trainer code into the GCE instance. There are several solutions available to us:

1. We could simply store the trainer application on the custom image from which we create the GCE instance.

1. We could store the code in a [GCS](https://cloud.google.com/storage/) bucket and have the startup script download it to the instance before doing anything else.

1. We could store the code in a [Cloud Source Repository](https://cloud.google.com/source-repositories/) and have the startup script clone it to the instance before doing anything else.

In this guide, in the interests of flexibility, we will go with the Cloud Source Repository option. The reason this is attractive is that it lets you apply `git` semantics to your training jobs -- for example, you can [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) different model architectures and specify them through the instance metadata. Or you could just use the trainer on some special branch of the repository which, again, would be specified at deployment time using instance metadata. This setup is much more powerful than the others.


## Examples

+ [TensorFlow estimator](#TensorFlow-estimator)

+ [sklearn model](#sklearn-model) - under development


### TensorFlow estimator

WIP


### sklearn model

WIP
