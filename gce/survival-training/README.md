# Compute Engine survival training

Training jobs that survive Compute Engine shutdowns

- - -

[Google Compute Engine](https://cloud.google.com/compute/docs/) (Compute Engine) gives you direct access to the computational backbone of Google Cloud Platform. You can use Compute Engine to create virtual machines provisioned with the resources of your choosing ([satisfying the constraints of any quotas that you define](https://cloud.google.com/compute/quotas)).

This has particular value when it comes to training machine learning models, as this gives you access to resources beyond those available to you physically. Now Compute Engine offers preemptible GPU instances. This means that, if you design your trainer appropriately, your cost per training step could be a fraction of what it would be on a dedicated instance.

Dedicated GPU-enabled Compute Engine instances go down regularly for maintenance and, by default, restart but without reinitializing the processes that were running on them before the shutdown. This introduces significant overhead in both time and cost to performing training on Compute Engine instances.

Similarly, preemptible VMs can be shut down at any time with very little advance warning, interrupting whatever processes were running on them. They also have a maximum uptime of 24 hours.

You can design a trainer to be robust against the behaviours of both dedicated GPU instances and preemptible instances using exactly the same semantics - the main issue you have to tackle is of your job surviving through a shutdown.

This guide provides you with a template to do exactly that.


## Survival strategy

There are five components to our resilient training jobs:

1. [Compute Engine custom images](https://cloud.google.com/compute/docs/images#custom_images)

1. A trainer command line interface

1. A delivery mechanism to inject your trainer into a Compute Engine instance

1. [Compute Engine startup scripts](https://cloud.google.com/compute/docs/startupscript)

1. [Compute Engine instance metadata](https://cloud.google.com/compute/docs/storing-retrieving-metadata)


### Prerequisites

As a part of this process, if you have not already done so, you will be prompted to enable the Compute Engine API.

We will be working in large part through the [Google Cloud SDK](https://cloud.google.com/sdk/). If you have not already done so, it is worth installing.


### Custom images

Your training job may require specific versions of libraries, like [tensorflow](https://www.tensorflow.org/) or [scikit-learn](http://scikit-learn.org). Additionally, in order to take advantage of instances with attached GPUs, you will have to have hardware-specific drivers installed on the instance. [Custom images](https://cloud.google.com/compute/docs/images#custom_images) are a convenient solution to this problem of environment reproduction within a Compute Engine virtual machine.

If you do have requirements (in addition to simply the OS) in your training environments, it is highly recommended that you create a VM image that you can use to spawn new instances without having to go through the tedious configuration process every time. The examples following this strategy section will demonstrate the creation of such images so, if this seems daunting, don't worry.


### Trainer CLI

Our strategy will have the Compute Engine instance automatically kick off your training job on startup. As part of this kick off, it will have to provide the job with the appropriate parameters -- how many steps to train for, how often to checkpoint the model, which data to train the model with, what hyperparameters should be used in training the model, etc.

It is helpful, if you intend to do this more than once, to assert a common semantical system across all your trainers. We provide a [boilerplate CLI](./dummy/train.py) in this repo. This interface is very similar to that provided by [TensorFlow estimators](https://www.tensorflow.org/programmers_guide/estimators), but can be used even with other frameworks. In the sections that follow, we will provide you with both TensorFlow and with scikit-learn examples that demonstrate its use.


### Delivery mechanism

We need a mechanism by which we can get our trainer code into the Compute Engine instance. There are several solutions available to us:

1. We could simply store the trainer application on the custom image from which we create the Compute Engine instance.

1. We could store the code in a [GCS](https://cloud.google.com/storage/) bucket and have the startup script download it to the instance before doing anything else.

1. We could store the code in a [Cloud Source Repository](https://cloud.google.com/source-repositories/) and have the startup script clone it to the instance before doing anything else.

1. We could simply clone an existing repository from [GitHub](https://github.com/) (be careful with this because, if working with private repos, this would require you to set up a git credentials helper in your VM image).

In our sample deployments (below), we will demonstrate some of these options. Changing between them will generally involve only minor tweaks to the startup scripts and instance metadata, so it should not be too challenging to generalize to any of the others. If you do have problems, please raise an issue.


### Startup script

Compute Engine instances have an in-built mechanism that triggers a script on instance startup (for the appropriate definition of startup, but this need not concern us for now). This is the mechanism we will use to kick off a training job or to pick it up where it left off before a shutdown. We will make use of it through [startup scripts](https://cloud.google.com/compute/docs/startupscript#troubleshooting).

It is alright if you are not familiar with these things. What is more important at the moment is to know that they exist and have a general sense for how they fit into the training process.


### Instance metadata

You may want to deploy different parametrizations of a training job to multiple Compute Engine instances at different times. For this purpose, [we will make use of the metadata server available to every Compute Engine instance](https://cloud.google.com/compute/docs/storing-retrieving-metadata#custom).

Instance metadata is where we will store things like model hyperparameters and paths to training and evaluation data. We will store metadata on our instances before we start our training jobs, and our startup scripts will make use of this metadata on instance startup.

If you are using the [example startup script provided in this guide](./Compute Engine/startup.sh), you will only have to make minimal changes to do so. The important thing again is to understand where instance metadata fits into our picture and that Compute Engine guarantees that it will be available to our startup script when it runs.


## Examples

The following examples demonstrate the use of the framework provided here:

+ [TensorFlow estimator](./README-tf-estimator.md)

+ [sklearn model](./README-sklearn.md) - work in progress

- - -

##### Notes and references

+ The trainer CLI suggested here is modeled on the [the Cloud ML Engine trainer interface](https://cloud.google.com/ml-engine/docs/packaging-trainer). This has the sizable benefit that, should you ever require distribution of any TensorFlow models that you are training through this process on Compute Engine, it should take very little work to move your training to ML Engine.
