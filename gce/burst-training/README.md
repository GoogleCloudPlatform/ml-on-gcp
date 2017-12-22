# Compute Engine burst training

[Google Compute Engine](https://cloud.google.com/compute/) gives you access to
very powerful virtual machines which you can use for as much or, more
importantly for most data science use cases, as little time as you need.

Common tasks that might take tens of minutes or even hours on your personal can
be completed in seconds on a machine with 64 cores and almost half a terabyte of
memory, and this can be done at a cost of under $1.

This makes Google Cloud Platform (or any equivalent cloud infrastructure for
that matter) a powerful tool for model training. The idea is simple:

1. Write a trainer locally, testing it on a small sample of your training data
   to make sure that the training job is sound in principle (that it has no
   syntax errors and such).

2. Package the trainer and deploy it to a virtual machine (VM) instance with
   many, many CPU cores and as much RAM as necessary.

3. Wait for a few minutes for the training job to complete, having it dump a
   model binary on distributed storage so that you can use it later.

4. Have the virtual machine instance shutdown on completion of the training job
   so that you are not charged for unused uptime.

This is particularly valuable if you want to train a model on a very large
dataset or if you want to perform a large-scale hyperparameter search.

This guide presents a framework for executing this process on Google Cloud
Platform. We will also see a concrete example of this framework in action. Once
you are done with this guide, you should be able to modify the scripts that
accompany it to accomodate your custom training jobs.

If you run into any problems, please create an issue on this repository and
[I](https://github.com/nkashy1) will gladly help you resolve them. If you would
like to improve this guide, do not hesitate to make a pull request.

- - -

## Cloud infrastructure

[Google Compute Engine](https://cloud.google.com/compute/) allows us to
provision and interact with virtual machines on Google Cloud Platform, and it
forms the basis of our burst training framework.

We will use [Google Cloud Storage](https://cloud.google.com/storage/) as the
common medium of storage, accessible both from our local environment as well as
from our Compute Engine virtual machines.


## Task

To demonstrate the burst training workflow, we will build an *xgboost* model to
address the [Census income
problem](https://archive.ics.uci.edu/ml/datasets/census+income). The problem
involves predicting a person's income level from their other Census information.

We will use *xgboost*'s *scikit-learn*-like API so that we can also perform a
hyperparameter search using
[RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
In this case, although the dataset itself is small, the value of burst training
is in allowing us to perform a much more comprehensive hyperparameter search
compared to what would be possible on even a powerful laptop.


## Code

There are three types of code in this directory:

1. Code facilitating the reproduction of your development environment on a
   Compute Engine virtual machine.

2. Code which defines your training job -- this is the kind of code that you
   write every day using frameworks like [pandas](https://pandas.pydata.org/),
   [scikit-learn](http://scikit-learn.org/stable/), and
   [xgboost](https://github.com/dmlc/xgboost). In the case of this guide, this
   is the code which directly pertains to the Census income problem.

3. Code which executes your training job in a Compute Engine instance running
   your development environment as specified in step 1.


### Reproducing your environment on Compute Engine

The very first step in the burst training process is making sure that Compute
Engine is set up to execute the code that you either already have or will be
writing locally. In the case of our Census example, we will be using Python 2.7
with the packages listed in [requirements.txt](./requirements.txt).

We do this by creating a virtual machine image that we can apply to any Compute
Engine instances we create. An image is, essentially, the frozen disk state of a
Compute Engine instance which, when used to create *another* instance, brings
that instance up with the same state. This means that, if your image already had
*scikit-learn* installed, then every virtual machine created with that image
would come with *scikit-learn* out of the box.

The burst training framework provides you with a script that you can run to
build your image - [build-image.sh](./build-image.sh). This script brings up a
VM instance, installs the appropriate libraries and packages, creates the image,
and then deletes the Compute Engine instance when it is done.

Packages are installed on the VM is done by means of a [startup
script](https://cloud.google.com/deployment-manager/docs/step-by-step-guide/setting-metadata-and-startup-scripts),
which is executed on the VM immediately after it starts. For the Census example,
we are using [image.sh](./image.sh). In your own work, it should suffice to make
the appropriate modifications to that file.

You can run [build-image.sh](./build-image.sh) from your terminal:
```
./build-image.sh <INSTANCE-NAME> <IMAGE-FAMILY>
```

Here `<INSTANCE-NAME>` is a name you would like to assign to the temporary GCE
instance on which we build the environment as specified in
[image.sh](./image.sh). This instance will be deleted once the image has built,
so you won't have to deal with it for very long.

`<IMAGE-FAMILY>` starts a family of images under which Compute Engine stores
your image. The value of this is that you can update your image over time and,
by referring to the image family rather than an individual image when you create
your burst training instances, the most up-to-date image in that family will be
used in your training jobs.

**Please replace `<INSTANCE-NAME>` and `<IMAGE-FAMILY>` with your own names.**


### Training code

[census-analysis.py](./census-analysis.py) contains the training code we will
run for the Census problem. The only important thing about this code, as far as
adapting it to your own use case, is the command line interface. This is the
means by which we will get our burst training instance to pass parameters to the
training job.

In the case of the Census code, we have to specify three parameters when we run
the job:

1. `--census-data-path` -- the Cloud Storage path to the training and test data

2. `--model-output-path` -- the Cloud Storage path at which the trainer should
   store its trained model binary

3. `--cv-iterations` -- the number of hyperparameter configurations that
   `RandomizedSearchCV` should try

That particular trainer, also allows you to specify a `--mode`, which defaults
to `train`. This `--mode` argument will be useful to us once the training job
has completed, as it will allow us to test out the trained model from our local
environment.

*Note: The [census-analysis.py](./census-analysis.py) trainer uses TensorFlow's
[tf.gfile](https://www.tensorflow.org/api_docs/python/tf/gfile) to store and
retrieve data from Cloud Storage. This is the only capacity in which it uses
TensorFlow. This is also a very useful module to know about -- it allows you to
interact with Cloud Storage objects just as you would interact with files on
your own filesystem.*


### Execution


