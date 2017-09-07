
## Introduction

This is a simple sample that shows how to:

- Create a GAE service that loads a pickled sci-kit learn model from GCS, and use it to serve prediction requests through Cloud Endpoints.


[comment]: #(get Andrew to help set these up as the advanced usage section)
The benefits of this configuration include:

1. GAE's autoscaling and load balancing.
1. Cloud Endpoints' monitoring and access control.


## Requirements

Python.

GCP SDK.  gsutil.

A GCP project that has GAE, GCS, and Cloud Endpoints enabled.


## Setup

1. `git clone https://github.com/GoogleCloudPlatform/ml-on-gcp`

1. `cd ml-on-gcp`

1. `pip install -r requirements.txt`

1. The name `modelserve` is tentative which can be changed to something else, but you need to change all occurrences consistently.

1. Update `modelserve.yaml`:  Replace `PROJECT_ID` with your project's id in this line:

    `host: "modelserve.endpoints.PROJECT_ID.cloud.goog"`

    Note that this file defines the API specifying the input `X` to be an array of arrays of floats, and output `y` to be an array of floats.  The model included in this sample code is a linear regression model with 2-dimensional inputs.

1. Deploy the service endpoint:

    `gcloud service-management deploy modelserve.yaml`

    This step deploys a [Cloud Endpoint](https://cloud.google.com/endpoints/) service, which allows us to monitor the API usage.

1. Create a GCS bucket and copy the model file over:

    ```
    gsutil mb gs://BUCKET_NAME
    gsutil cp lr.pkl gs://BUCKET_NAME
    ```

1. Update `app.yaml`:  Replace `PROJECT_ID` with your project's id in this line:

    `  name: modelserve.endpoints.PROJECT_ID.cloud.goog`

    Replace `BUCKET_NAME` with the name of the bucket you created on GCS above.

1. Deploy the backend service:

    `gcloud app deploy`


1. If the deployment is successful, you can test it from the command line.  Firt go to your project's console's IAM page and create an API key `API_KEY`, and use it and your project's id in the following:

    `curl -H "Content-Type: application/json" -X POST -d '{"X": [[1, 2], [5, -1], [1, 0]]}' "https://modelserve-dot-PROJECT_ID.appspot.com/predict?key=API_KEY"`

    You should get the following response:

    `{"y": [0.6473534912754967, -0.7187842827829021, 0.3882338314071392]}`





