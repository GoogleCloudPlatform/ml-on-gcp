# Compute Engine burst training

[Google Compute Engine](https://cloud.google.com/compute/) gives you access to
very powerful virtual machines which you can use for as much or as little time
as you need.

Common tasks that might take tens of minutes or even hours on your personal
machine can be completed in seconds on a machine with up to 64 cores and
terabytes of memory, and this can be done at negligible cost given that the jobs
complete so quickly

This makes Google Cloud Platform a powerful tool for training parallelizable
models. The idea is simple:

1. Write a trainer locally, testing it on a small sample of your training data
   to make sure that the training job is sound in principle (that it has no
   syntax errors and such).

2. Package the trainer and deploy it to a virtual machine (VM) instance with
   many, many CPU cores and as much RAM as necessary.

3. Wait a little while for the training job to complete, having it dump a
   model binary on distributed storage so that you can use it later.

4. Have the virtual machine instance shutdown on completion of the training job
   so that you are not charged for unused uptime.

This is particularly valuable if you want to train a model on a large
dataset or if you want to perform a large-scale hyperparameter search.

This guide presents a framework for executing this process on Google Cloud
Platform. We will also see a concrete example of this framework in action. Once
you are done with this guide, you should be able to modify the scripts that
accompany it to accomodate your custom training jobs.

If you run into any problems, please create an issue on this repository.
If you would like to improve this guide, do not hesitate to make a pull request.

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
   Compute Engine virtual machine - [build-image.sh](./build-image.sh) and
   [image.sh](./image.sh).

2. Code which defines your training job -- this is the kind of code that you
   write every day using frameworks like [pandas](https://pandas.pydata.org/),
   [scikit-learn](http://scikit-learn.org/stable/), and
   [xgboost](https://github.com/dmlc/xgboost) -
   [census-analysis.py](./census-analysis.py)

3. Code which executes your training job in a Compute Engine instance running
   your development environment as specified in step 1 - [train.sh](./train.sh)
   and [census-startup.sh](./census-startup.sh)

The sections below describe the function of each component.


### Reproducing your environment on Compute Engine

The very first step in the burst training process is making sure that Compute
Engine is set up to execute the code that you either already have or will be
writing locally. In the case of our Census example, we will be using Python 2.7
with the packages listed in [requirements.txt](./requirements.txt).

(**Note:** [requirements.txt](./requirements.txt) is intended for local use only.
The installation of Python packages onto the VM image is handled by
[image.sh](./image.sh).)

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

Packages are installed on the VM by means of a [startup
script](https://cloud.google.com/deployment-manager/docs/step-by-step-guide/setting-metadata-and-startup-scripts),
which is executed on the VM immediately after it starts. For the Census example,
we are using [image.sh](./image.sh). In your own work, it should suffice to make
the appropriate modifications to that file.

To begin with, if your desired image family does not yet exist, run:
```
gcloud compute images create <BASE-IMAGE-NAME> --family <IMAGE-FAMILY>
--source-image-family ubuntu-1604-lts --source-image-project ubuntu-os-cloud
```

In this case, we are basing our images off of Compute Engine's official Ubuntu
16.04 image. You could also use any of the other images listed
[here](https://cloud.google.com/compute/docs/images) by replacing the
`--source-image-family` and `--source-image-project` arguments above.

Next you build the actual image you want by running
[build-image.sh](./build-image.sh) from your terminal:
```
./build-image.sh <BUILDING-INSTANCE-NAME> <IMAGE-FAMILY>
```

Here `<BUILDING-INSTANCE-NAME>` is a name you would like to assign to the temporary GCE
instance on which we build the environment as specified in
[image.sh](./image.sh). This instance will be deleted once the image has built,
so you won't have to deal with it for very long.

`<IMAGE-FAMILY>` starts a family of images under which Compute Engine stores
your image. The value of this is that you can update your image over time and,
by referring to the image family rather than an individual image when you create
your burst training instances, the most up-to-date image in that family will be
used in your training jobs.

**Please replace `<BASE-IMAGE-NAME>`, `<BUILDING-INSTANCE-NAME>` and
`<IMAGE-FAMILY>` with your own names.**


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

That particular trainer also allows you to specify a `--mode`, which defaults
to `train`. This `--mode` argument will be useful to us once the training job
has completed, as it will allow us to test out the trained model from our local
environment.

*Note: The [census-analysis.py](./census-analysis.py) trainer uses TensorFlow's
[tf.gfile](https://www.tensorflow.org/api_docs/python/tf/gfile) to store and
retrieve data from Cloud Storage. This is the only capacity in which it uses
TensorFlow. This is also a very useful module to know about -- it allows you to
interact with Cloud Storage objects through your Python environment just as you
would interact with files on your own filesystem.*


### Cloud Storage bucket

Before execution, you will have to place the training and evaluation data on
[Google Cloud Storage](https://cloud.google.com/storage/docs/).

The executor for the Census example expects a Cloud Storage Bucket of your
choice with the data stored under the path `gs://<BUCKET-NAME>/census/`. You can
[download the data from the UCI Machine Learning
Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/) -
the `adult.data` and `adult.test` files are all you need. Once you have them
available to you locally, you can copy them over to the desured Cloud Storage
path by running the following command from the directory containing those files:

```
gsutil -m cp adult.* gs://<BUCKET-NAME>/census/
```

You may have to create the bucket first - either using `gsutil mb
gs://<BUCKET-NAME>` or through the [Cloud Console Storage
browser](https://console.cloud.google.com/storage/browser).

Alternatively, you can also use the [Storage
browser](https://console.cloud.google.com/storage/browser) in the Cloud Console.

The executor, which we cover in more detail below, will:

1. Upload the training script to `gs://<BUCKET-NAME>`

2. Run the training script on the data in `gs://<BUCKET-NAME>/census`

3. Save the trained model to `gs://<BUCKET-NAME>/census-<TIMESTAMP>`, with the
   `<TIMESTAMP>` representing the time at which the job was submitted

**Please replace <BUCKET-NAME> with your own name.**


### Execution

#### Setup

The core of the execution step is a call to `python census-analysis.py`.
However, this call needs to take place on a Compute Engine instance on which the
image containing our environment has been loaded. Moreover:

1. The `python census-analysis.py` call must be made with the right command line
   arguments.

2. The Compute Engine instance should be shut down immediately after the `python
   censys-analysis.py` call has returned. This will keep Compute Engine charges
   to a minimum.

We use two Compute Engine mechanisms to achieve this behavior:

1. [Instance
   metadata](https://cloud.google.com/compute/docs/storing-retrieving-metadata)
   -- These are key-value pairs of strings that you can set on a Compute Engine
   VM. In burst training, we use them to set values for command line arguments
   as well as to deliver the training code to the instance. The additional
   benefit to using metadata like this is that they are visible on your
   instances either through the Cloud Console or through `gcloud compute
   instances describe <INSTANCE_NAME>`.

2. [Startup scripts](https://cloud.google.com/compute/docs/startupscript) --
   This is a script that is executed immediately once a Compute Engine instance
   has started. In burst training, as we are using a Linux image (although you
   can alter this as necessary), we use a
   `bash` script which:

   - Loads the appropriate instance metadata

   - Downloads the trainer code to the VM from Google Cloud Storage

   - Runs the training job specifying the appropriate command line arguments
     from the metadata

   - Shuts down the instance once the training is complete

The [training script](./train.sh) shows this process in action. It performs the
following operations:

1. Uploads [census.py](./census.py) to the Cloud Storage bucket specified by
   its second command line argument.

1. Creates an instance whose name is the first command line argument and which
   uses the most recent image under the image family specified by the third
   command line argument. This instance is created with metadata specifying the
   arguments we wish to run [census.py](./census.py) with when we begin training
   and with [census-startup.sh](./census-startup.sh) as the startup script,
   which handles the execution of the training job and shutting down of the VM
   once the training is either complete or has errored out.

Once you have built your VM image, you can run this script to execute your first
burst training job as follows:

```
./train.sh <TRAINING-INSTANCE-NAME> <BUCKET-NAME> <IMAGE-FAMILY>
```

Here, `<BUCKET-NAME>` and `<IMAGE-FAMILY>` should have the same values as above.
`<TRAINING-INSTANCE-NAME>` should be a fresh name for this new training
instance.

The checklist below describes the steps you should take to run the trainer for
the census income classifier.

#### Checklist

1. Make sure you have a base image in your desired image family:
```
gcloud compute images create <BASE-IMAGE-NAME> --family <IMAGE-FAMILY>
--source-image-family ubuntu-1604-lts --source-image-project ubuntu-os-cloud
```

1. Run [build-image.sh](./build-image.sh) to create a new image in your desired
   `IMAGE_FAMILY`.

1. Put the Census income data into a Cloud Storage bucket:
```
gsutil -m cp adult.* gs://<BUCKET-NAME>/census/
```

1. Run [train.sh](./train.sh).


#### Preemptibility

The [sample training script](./train.sh) specifies that training should be
performed on a [preemptible VM
instance](https://cloud.google.com/compute/docs/instances/preemptible). This
makes burst training even cheaper than it otherwise would be -- preemptible
Compute Engine VMs can be up to 80% cheaper than regular VMs.

In the case of a census sample, using a preemptible VM makes sense because, even
at scale, the job is short-lived. If you have a longer-running training job, you
have two options:

1. Specify that you do not want to use a preemptible VM for training by removing
   the `--preemptible` flag from `gcloud compute instances create` command in
   the training script.

2. Set your job up as per our [survival training
   guide](https://github.com/GoogleCloudPlatform/ml-on-gcp/tree/master/gce/survival-training)
   so that it can take advantage of preemptible VMs on Compute Engine.


## How to use burst training with our own code

You can adapt the procedure described here to perform burst training of your own
models. Moreover, you should be able to do so with only slight modifications to
the code by following these steps:

1. Modify [image.sh](./image.sh) to install whatever software you require.

1. Run [build-image.sh](./build-image.sh) to create a new image in your desired
   `IMAGE_FAMILY`.

1. Upload your data to Google Cloud Storage.

1. Decide on what instance metadata you will use for your burst training.

1. Modify [census-startup.sh](./census-startup.sh) to

   - Load this metadata

   - Download your trainer script or package

   - Run the trainer with the appropriate parameters

   Do not remove the final shutdown command unless you want the instance to keep
   running even after training has completed.

1. Modify [train.sh](./train.sh) to
   
   - Upload your own trainer to Cloud Storage. Although, in this example, our
     trainer consisted of a single Python file, you can also use a full Python
     package. Note that you may have to modify your trainer to accept parameters
     from the command line.

   - Create an instance of the appropriate [machine type](https://cloud.google.com/compute/docs/machine-types).
     Note that you can also specify a custom machine type.

   - Make sure that it is pointing at your modified startup script.

   - Set your desired metadata.

1. Run your modified [train.sh](./train.sh).
