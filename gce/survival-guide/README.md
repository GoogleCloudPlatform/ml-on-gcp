# The GCE survival guide

[Google Compute Engine](https://cloud.google.com/compute/docs/) (GCE) gives you direct access to the computational backbone of Google Cloud Platform. You can use GCE to create virtual machines provisioned with the resources of your choosing ([satisfying the constraints of any quotas that you define](https://cloud.google.com/compute/quotas)).

This has particular value when it comes to training machine learning models, as this gives you access to resources beyond those available to you physically. For example, GCE will soon offer preemptible GPU instances. This means that, if you designed your trainer appropriately, your cost per training step could be a fraction of what it would be on a dedicated instance.

Even if you want to train on a dedicated instance, GPU-enabled GCE instances have always had the issue that they go down about once a week for maintenance. This introduces a significant amount of overhead (in both time and money) to performing training on GCE instances.

However, GCE *does* expose primitives that allow you set up training jobs *and* your GCE instances so that you can train across instance preemptions and shutdowns.

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

Our strategy will have the GCE instance automatically kicking off your training job on startup. 
