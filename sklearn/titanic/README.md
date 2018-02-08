## Introduction

This sample code illustrates how to start a GCE instance and train a model using scikit-learn on the instance. We will be using the [Titanic dataset](https://www.kaggle.com/c/titanic). The goal is to train a classifier to predict whether a given passenger on Titanic survived or not.

## Prerequisites

You need to download [train.csv](https://www.kaggle.com/c/titanic/download/train.csv) from [Kaggle](https://www.kaggle.com/).

The sample assumes that you already have a [GCP](https://cloud.google.com/) account and you have [downloaded](https://cloud.google.com/sdk/), installed, and [configured](https://cloud.google.com/sdk/gcloud/reference/config/) it on your computer.

Finally, you need to have a project created on GCP. For further instructions, please check [here](https://cloud.google.com/sdk/gcloud/reference/projects/create).


## Steps:
We will create a Google Compute Engine instance, configure it, and use it to train a model. While some of the steps can be done using the web portal [here](https://pantheon.corp.google.com), we will try to accomplish this using the SDK and mainly the [gcloud](https://cloud.google.com/sdk/gcloud/) command.

##### 1. Create a Google Compute Engine Instance
First, we should create a GCE instance that we can use to train our model on. In this case, we can use many of the default arguments and simply create the instance:

```
gcloud compute instances create MYINSTANCE
```

where MYINSTANCE is our given name to this instance. By default, this instance will be of type *n1-standard-1* with 1 CPU and 3.75GB of RAM. To see a list of available machine types, you can use:

```
gcloud compute machine-types list
```

By default, the instance is going to be built using a Debian image. But you can change this during the instance creation. To see a list of available images, you may use the following command:
```
gcloud compute images list
```

##### 2. Accessing and Configuring the Instance

Accessing the newly created image is quite simple:
```
gcloud compute ssh MYINSTANCE
```

Once we ssh to the instance, we will need to use *pip* to install scikit-learn and some other packages:
```
# Run on the new instance
sudo apt-get install python-pip
pip install numpy pandas sklearn scipy tensorflow
```
We are not using tensorflow to train the model in this sample. However we will be using GFile, a python class implemented in tensorflow which unifies how we access the local and remote files.

The new instance also requires to obtain user access credentials. The following command will guide you through the process:
```
# Run on the new instance
gcloud auth application-default login
```

##### 3. Accessing the dataset
In this sample, we will show two ways for our code to access the dataset.

###### Copying it to the instance
We can copy the dataset and use it directly inside the instance as a local file. Fortunately, *gcloud* make this an easy step:
```
gcloud compute scp ./train.csv  MYINSTANCE:~/
```

Note that the same method can be used to copy any other file (e.g. the python code).

We can then run our python code in the instance to create a model:
```
# Run on the new instance
python titanic.py --titanic-data-path ./train.csv --model-output-path ./model.pkl
```

###### Reading it from Google Cloud Storage
Another solution is to read the dataset directly from a Bucket in GCS. First, we need to create a Bucket. We will use *gsutil* to do it:
```
gsutil mb MYBUCKET
gsutil cp ./train.csv  gs://MYBUCKET/MYFOLDER/train.csv
```

The two commands above will create a Bucket in GCS, and then copy the dataset into it.

The advantage of using GFile in our python code is that we can use the exact same code and read the dataset from GCS:
```
# Run on the new instance
python titanic.py --titanic-data-path gs://MYBUCKET/MYFOLDER/train.csv --model-output-path gs://MYBUCKET/MYFOLDER/model.pkl
```

which will create the model and stores it in the same location as our dataset in GCS.

That is it!! We have successfully trained a classifier using scikit-learn on a GCE instance.

## What is Next
Depending on your needs, here are a few things you may consider:
* You may require more resources (more CPU's, more RAM) to train your model, in which case your instance should have a different type.
* You may also want another OS for your instance, which requires you to create it using a different image.
* It may be a good idea to stop or even delete your instance after your model is trained to minimize the cost.
* If you need to train your model periodically, it may be worth considering to automate the entire process.