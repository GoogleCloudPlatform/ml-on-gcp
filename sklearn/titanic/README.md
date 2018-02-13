## Introduction

This sample code illustrates how to start a GCE instance and train a model using scikit-learn on the instance. We will be using the [Titanic dataset](https://www.kaggle.com/c/titanic) from [Kaggle](https://www.kaggle.com/). The goal is to train a classifier to predict whether a given passenger on Titanic survived or not.

## Prerequisites

Before we start, we need to:

* login to [Kaggle](https://www.kaggle.com) and download the Titanic dataset, mainly [train.csv](https://www.kaggle.com/c/titanic/download/train.csv).

* have a GCP account, and [download](https://cloud.google.com/sdk/), install, and [configure](https://cloud.google.com/sdk/gcloud/reference/config/) the Google Cloud SDK on our local computer.

* [create a project](https://cloud.google.com/sdk/gcloud/reference/projects/create) on GCP and set the peoject and zone properties as instructed [here](https://cloud.google.com/sdk/gcloud/reference/config/set).

* set these environment variables on the local computer:
	* $MY_INSTANCE: the name of the new instance (e.g. titanic_trainer).
	* $MY_BUCKET: the bucket name to be created on GCS (e.g. titanic_bucket).
	* $MY_FOLDER: the folder name to be used in the bucket (e.g. titanic_folder).

## Steps:
We will create a Google Compute Engine instance, configure it, and use it to train a model. While some of the steps can be done using the web portal [here](https://pantheon.corp.google.com), we will try to accomplish this using the SDK and mainly the [gcloud](https://cloud.google.com/sdk/gcloud/) command. Unless otherwise specified as a comment, all the commands are to be run locally.

##### 1. Create a Google Compute Engine Instance
First, we should create a GCE instance that we can use to train our model on. In this case, we can rely on many of the default arguments and simply create the instance:

```bash
gcloud compute instances create $MY_INSTANCE
```
which creates an instance of type *n1-standard-1* (1 CPU, 3.75GB of RAM), and a *Debian* disk image. We can however specify a different machine type or image. For example, the following command creates an *Ubuntu* instance of type *n1-standard-2*:

```bash
gcloud compute instances create $MY_INSTANCE --machine-type=n1-standard-2 --image-family=ubuntu-1604-lts --image-project ubuntu-os-cloud
```
You may want to learn more about the [machine types](https://cloud.google.com/compute/docs/machine-types) and  [images](https://cloud.google.com/compute/docs/images), or see a list of available machine types and images with the following two commands:

```bash
gcloud compute machine-types list
gcloud compute images list
```

##### 2. Accessing and Configuring the Instance

Accessing the newly created image is quite simple:
```bash
gcloud compute ssh $MY_INSTANCE
```

Once we ssh to the instance, we will need to use *pip* to install *scikit-learn* and some other packages:
```bash
# Run on the new instance
sudo apt-get install -y python-pip
pip install numpy pandas sklearn scipy tensorflow
```

Please note that we are not using *tensorflow* to train the model in this sample. We will however use *gfile*, a python class implemented in *tensorflow* which unifies how we access the local and remote files. An alternative to using *gfile*  for accessing the files stored in *Google Cloud Storage (GCS)* is [Cloud Storage Client Libraries](https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python).

If we want to use *GCS* to store the dataset, the new instance will then require to obtain user access credentials. The following command will guide us through the process:
```bash
# Run on the new instance
gcloud auth application-default login
```

##### 3. Running the Code
We need to make the dataset available to the training code. Here are two ways for the code to access the dataset.

###### Copying it to the instance
We can copy the dataset into the new instance and use it directly inside the instance and read it as a local file. Fortunately, *gcloud* makes this an easy step:
```bash
gcloud compute scp ./train.csv $MY_INSTANCE:~/
```

Note that the same method can be used to copy any other file (e.g. the python code):
```bash
gcloud compute scp ./titanic.py $MY_INSTANCE:~/
```


We can then run our python code in the instance to create a model:
```bash
# Run on the new instance
python titanic.py --titanic-data-path ~/train.csv --model-output-path ~/model.pkl
```

which will generate the model and save it on the instance.

###### Reading it from GCS
Another solution is to read the dataset directly from a bucket in GCS. We use *gsutil* to create the bucket in GCS, and then copy the dataset into it:
```bash
gsutil mb $MY_BUCKET
gsutil cp ./train.csv  gs://$MY_BUCKET/$MY_FOLDER/train.csv
```
The advantage of using *gfile* in our python code is that we can use the exact same code and read the dataset from GCS:
```bash
# Run on the new instance
export $MY_BUCKET=THE_GIVEN_BUCKET_NAME_HERE
export $MY_FOLDER=THE_GIVEN_FOLDER_NAME_HERE
python titanic.py --titanic-data-path gs://$MY_BUCKET/$MY_FOLDER/train.csv --model-output-path gs://$MY_BUCKET/$MY_FOLDER/model.pkl
```

which will create the model and store it in the same location as our dataset in *GCS*.

That is it!! We have successfully trained a classifier using scikit-learn on a *GCE* instance.

##### 4. Stopping or Deleting the Instance
The *GCE* instance that we created will be running indefinitely and costing us, unless we either stop or delete it completely. Stopping the instance will allow us to start it again later on, should we require to use it again which may be the better choice:
```bash
gcloud compute instances stop $MY_INSTANCE
```

However, if we want to completely delete the instance, we can use the following command:
```bash
gcloud compute instances delete $MY_INSTANCE
```

## What is Next
##### Automating the Entire Process
It is possible to automate the entire process described in this tutorial by writing some scripts to be run periodically. The [burst training sample](https://github.com/GoogleCloudPlatform/ml-on-gcp/tree/master/gce/burst-training) has a detailed description of how to do this.

##### Lowering the Cost
For some machine learning cases, you may require an instance with a high number of CPU's or a large amount of RAM. These instances are typically more expensive. [Preemptible Instances](https://cloud.google.com/compute/docs/instances/preemptible) can be a great option to train models on at a much lower cost, and they are definitely worth considering if we are going to train certain models periodically.


