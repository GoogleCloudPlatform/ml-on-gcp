# The GCE survival guide: TensorFlow Estimator

The nice thing about the [TensorFlow Estimator API](https://www.tensorflow.org/programmers_guide/estimators) is that it provides the same checkpointing semantics as the ones we are using as part of our survival process.

Moreover, the [Cloud ML Engine trainer interface](https://cloud.google.com/ml-engine/docs/packaging-trainer) also accepts model parameters and hyperparameters in the same fashion suggested here. So we should easily be able to take a model intended to run on Cloud ML Engine and run it on a single (but beefy) GCE instance.

Exactly such a model exists! We will make use of the tensorflow/models [CIFAR-10 estimator example](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator).


Let us follow each of the steps as outlined in [the main README](./README.md).

## Custom image

The CIFAR-10 example in question is designed to use multiple GPUs. We will need a GCE instance capable of supporting this. Let us start out by creating such a GCE instance once and freezing it into a custom image.

(Note: If you are following along but don't feel like covering the cost of a multi-GPU GCE instance, you can also use one without any GPUs.)

Before we can create a GPU instance, we must make sure that we have enough GPUs available to us in our project's quota and in our desired region. You can check this on [IAM and admin quota page](https://console.cloud.google.com/iam-admin/quotas). If you do not have enough GPUs in your quota, you will have to request them in the appropriate zone. [This page has a list of zones in which GPUs are available](https://cloud.google.com/compute/docs/gpus/). For the purposes of this guide, I will use four NVIDIA K80 GPUs in `us-west1-b`.

It is simplest to [create the GCE instance through the Cloud Console](https://console.cloud.google.com/compute/instancesAdd):

![woah](./img/tf-estimator-instance-creation.png)

Those are the settings I used, which amount to a cost of about $3.34 per hour (in USD).

The next step is to install 

- - -

[HOME](./README.md)
