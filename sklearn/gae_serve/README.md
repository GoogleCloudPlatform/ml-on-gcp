
## Introduction

This is a simple sample that shows how to:

- Create a Google App Engine service on Google Cloud Platform that loads a pickled scikit-learn model from Google Cloud Storage, and use it to serve prediction requests through Google Cloud Endpoints.

The benefits of this configuration include:

1. App Engine's autoscaling and load balancing.
1. Cloud Endpoints' monitoring and access control.


## Requirements

- [Python](https://www.python.org/).  The version (2.7 or 3) used for local developement of the model should match the version used in the service, which is specified in the file `app.yaml`.

- [Google Cloud Platform SDK](https://cloud.google.com/sdk/).  The SDK includes the commandline tools `gcloud` for deploying the service and [`gsutil`](https://cloud.google.com/storage/docs/gsutil) for managing files on Cloud Storage.

- A Google Cloud Platform project which as the following products enabled:

    - [Google App Engine](https://cloud.google.com/appengine/)

    - [Google Cloud Storage](https://cloud.google.com/storage/)

    - [Google Cloud Endpoints](https://cloud.google.com/endpoints/)


## Setup

1. `git clone https://github.com/GoogleCloudPlatform/ml-on-gcp`

1. `cd ml-on-gcp/gae_serve`

1. This samples demostrates how to deploy an App Engine service named `modelserve`.  If you prefer to deploy to the `default` service (for example, if this is the first App Engine service in your project, it must be named `default`), use the yaml files in the `default/` subdirectory by copying them over the yaml files in the root directory of this sample.

    - **Note that App Engine does not allow deleting the `default` service from your project.**

1. Update `modelserve.yaml`:  Replace `PROJECT_ID` with your Google Cloud Platform project's id in this line:

    `host: "modelserve-dot-PROJECT_ID.appspot.com"`

    * Note that this file defines the API specifying the input `X` to be an array of arrays of floats, and output `y` to be an array of floats.  The model included in this sample code `lr.pkl` is a pickled [linear regression model](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with 2-dimensional inputs.

1. Deploy the service endpoint:

    `gcloud service-management deploy modelserve.yaml`

    This step deploys a [Cloud Endpoint](https://cloud.google.com/endpoints/) service, which allows us to monitor the API usage on the [Endpoints](https://console.cloud.google.com/endpoints) console page.

1. If the deployment is successful, get the deployment's config id either from the [Endpoints](https://console.cloud.google.com/endpoints) console page under the service's Deployment history tab, or you can find all the configuration IDs by running the following:

    `gcloud service-management configs list --service="modelserve-dot-PROJECT_ID.appspot.com"`

    The configuration ID should look like `2017-08-03r0`.  The `r0, r1, ...` part in the configuration IDs indicate the revision numbers, and you should use the highest (most recent) revision number.

1. Create a Cloud Storage bucket with your choice of a `BUCKET_NAME`, and copy the sample model file over:

    ```
    gsutil mb gs://BUCKET_NAME
    gsutil cp lr.pkl gs://BUCKET_NAME
    ```

1. Update `app.yaml`:

    * If you already have at least one App Engine service in your Google Cloud Platform project:

        - Replace `PROJECT_ID` with your Google Cloud Platform project's id.

        - Replace `BUCKET_NAME` with the name of the bucket you created on Cloud Storage above.

        - Replace `CONFIG_ID` with the configuration ID you got from the service endpoint deployment.

    See the [documentation](https://cloud.google.com/appengine/docs/flexible/python/configuring-your-app-with-app-yaml) for more information about `app.yaml`.

1. Deploy the backend service:

    `gcloud app deploy`

    **This step could take several minutes to complete.**


1. If the deployment is successful, you can access it by first creating an API key with the "Create credentials" button on the [Credentials](https://console.cloud.google.com/apis/credentials) page.  Make sure you switch to the correct Google Cloud Platform project first.


1. You can access the deployed service in a few different ways: (Remember to replace `PROJECT_ID` and `API_KEY` with their actual values below.)

    * From the command line:

        `curl -H "Content-Type: application/json" -X POST -d '{"X": [[1, 2], [5, -1], [1, 0]]}' "https://modelserve-dot-PROJECT_ID.appspot.com/predict?key=API_KEY"`

        (Change the host URL to `PROJECT_ID.appspot.com` if you deployed the service as `default`.)

        You should get the following response:

        `{"y": [0.6473534912754967, -0.7187842827829021, 0.3882338314071392]}`

        The deployed model `lr.pkl` is a simple [linear regression model](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with 2-dimensional inputs.

    * With the simple python client included in this sample:

        ```python
        from client import ModelServiceClient

        model_service_client = ModelServiceClient(host='https://modelserve-dot-PROJECT_ID.appspot.com', api_key='API_KEY')

        model_service_client.predict([[1, 2], [5, -1], [1, 0]])

        # => [0.6473534912754967, -0.7187842827829021, 0.3882338314071392]
        ```

    * With the automatically generated swagger client ([instructions](https://github.com/swagger-api/swagger-codegen)):

        ```python
        import swagger_client

        swagger_client.configuration.api_key['key'] = 'API_KEY'
        api = swagger_client.DefaultApi()

        body = swagger_client.X([[1, 2], [5, -1], [1, 0]])

        response = api.predict(body)

        # response = {"y": [0.6473534912754967, -0.7187842827829021, 0.3882338314071392]}
        ```


## Clean up

Delete service by running:

```
gcloud app services delete modelserve
gcloud service-management delete modelserve-dot-PROJECT_ID.appspot.com
```


(If the service was deployed as the `default` service, it cannot be deleted.)


## Advanced usage

### Healthcheck

For information about configuring the service's healthcheck, see the [documentation](https://cloud.google.com/appengine/docs/flexible/nodejs/configuring-your-app-with-app-yaml#health_checks).


### Autoscaling

For information about configuring the service's autoscaling, see the [documentation](https://cloud.google.com/appengine/docs/flexible/nodejs/configuring-your-app-with-app-yaml#services).


### Quota

For information about configuring the service's quota, see the [documentation](https://cloud.google.com/endpoints/docs/openapi/quotas-configure).
