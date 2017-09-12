
## Introduction

This is a simple sample that shows how to:

- Create a Google App Engine (GAE) service on Google Cloud Platform (GCP) that loads a pickled scikit-learn model from Google Cloud Storage (GCS), and use it to serve prediction requests through Google Cloud Endpoints.

The benefits of this configuration include:

1. GAE's autoscaling and load balancing.
1. Cloud Endpoints' monitoring and access control.


## Requirements

- [Python](https://www.python.org/).  The version (2.7 or 3) used for local developement of the model should match the version used in the service, which is specified in the file `app.yaml`.

- [Google Cloud Platform SDK](https://cloud.google.com/sdk/).  The SDK includes the commandline tools `gcloud` for deploying the service and [`gsutil`](https://cloud.google.com/storage/docs/gsutil) for managing files on GCS.

- A Google Cloud Platform project which as the following products enabled:

    - [Google App Engine (GAE)](https://cloud.google.com/appengine/)

    - [Google Cloud Storage (GCS)](https://cloud.google.com/storage/)

    - [Cloud Endpoints (CE)](https://cloud.google.com/endpoints/)


## Setup

1. `git clone https://github.com/GoogleCloudPlatform/ml-on-gcp`

1. `cd ml-on-gcp/gae_serve`

1. The name `modelserve` (appearing in `app.yaml` and `modelserve.yaml`) is tentative which can be changed.  You should make the same change in the steps below.

1. Update `modelserve.yaml`:  Replace `PROJECT_ID` with your GCP project's id in this line:

    `host: "PROJECT_ID.appspot.com"`

    * Note that this file defines the API specifying the input `X` to be an array of arrays of floats, and output `y` to be an array of floats.  The model included in this sample code `lr.pkl` is a pickled [linear regression model](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with 2-dimensional inputs.

1. Deploy the service endpoint:

    `gcloud service-management deploy modelserve.yaml`

    This step deploys a [Cloud Endpoint](https://cloud.google.com/endpoints/) service, which allows us to monitor the API usage on the [Endpoints](https://console.cloud.google.com/endpoints) console page.

1. If the deployment is successful, get the deployment's config id either from the [Endpoints](https://console.cloud.google.com/endpoints) console page under the service's Deployment history tab, or you can find all the configuration IDs by running the following:

    `gcloud service-management configs list --service="PROJECT_ID.appspot.com"`

    The configuration ID should look like `2017-08-03r0`.  The `r0, r1, ...` part in the configuration IDs indicate the revision numbers, and you should use the highest (most recent) revision number.

1. Create a GCS bucket with your choice of a `BUCKET_NAME`, and copy the sample model file over:

    ```
    gsutil mb gs://BUCKET_NAME
    gsutil cp lr.pkl gs://BUCKET_NAME
    ```

1. Update `app.yaml`:

    * If you already have at least one GAE service in your GCP project:

        - Replace `PROJECT_ID` with your GCP project's id.

        - Replace `BUCKET_NAME` with the name of the bucket you created on GCS above.

        - Replace `CONFIG_ID` with the configuration ID you got from the service endpoint deployment.

    * If you have not deployed any service to GAE with this GCP project:

        - Replace `service: modelserve` with `service: default` in `app.yaml`.  The first service deployed to GAE must be named `default`.

        - Replace `PROJECT_ID` with your GCP project's id.

        - Replace `BUCKET_NAME` with the name of the bucket you created on GCS above.

        - Replace `CONFIG_ID` with the configuration ID you got from the service endpoint deployment.

        - **Note that you will not be able to delete the `default` service from your project.**

    See the [documentation](https://cloud.google.com/appengine/docs/flexible/python/configuring-your-app-with-app-yaml) for more information about `app.yaml`.

1. Deploy the backend service:

    `gcloud app deploy`

    **This step could take several minutes to complete.**


1. If the deployment is successful, you can test it from the command line:

    - Create an API key with the "Create credentials" button on the [Credentials](https://console.cloud.google.com/apis/credentials) page.  Make sure you switch to the correct GCP project first.

    - Now you can test the service endpoint: (Remember to replace `PROJECT_ID` and `API_KEY` with their actual values below.)

        * If you have `service: modelserve` in `app.yaml`:

            `curl -H "Content-Type: application/json" -X POST -d '{"X": [[1, 2], [5, -1], [1, 0]]}' "https://modelserve-dot-PROJECT_ID.appspot.com/predict?key=API_KEY"`

        * If you have `service: default` in `app.yaml`:

            `curl -H "Content-Type: application/json" -X POST -d '{"X": [[1, 2], [5, -1], [1, 0]]}' "https://PROJECT_ID.appspot.com/predict?key=API_KEY"`

        You should get the following response:

        `{"y": [0.6473534912754967, -0.7187842827829021, 0.3882338314071392]}`

        The deployed model `lr.pkl` is a simple [linear regression model](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with 2-dimensional inputs.


## Clean up

* If the service was deployed as the `default` service (that is, with `service: default` in `app.yaml`), then the service cannot be deleted from the project.

* If the service was deployed as the `modelserve` service (that is, with `service: modelserve` in `app.yaml`), then you can delete service by running:

    `gcloud app services delete modelserve`

    `gcloud service-management delete PROJECT_ID.appspot.com`


## Advanced usage

### Healthcheck

Note that in the `main.py` file, the model is not loaded until the first request has been received.  For the model


### Autoscaling

TODO
