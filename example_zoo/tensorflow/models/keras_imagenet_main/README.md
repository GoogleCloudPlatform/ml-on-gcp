# AI Platform Example Zoo: keras_imagenet_main

This is an automatically created sample based on [tensorflow/models/official/resnet/keras/keras_imagenet_main.py](https://github.com/tensorflow/models/blob/r1.13.0/official/resnet/keras/keras_imagenet_main.py).

To run the sample:


1. Update [submit_27.sh](submit_27.sh) (or [submit_35.sh](submit_35.sh) for Python 3.5) with GCS bucket and GCP project ID:

```
# in submit_27.sh

BUCKET=gs://your-bucket/your-
PROJECT_ID=your-gcp-project-id
```

1. Submit the job:

```
bash submit_27.sh
```
