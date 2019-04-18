# Google Cloud Function Examples

Example code to infer Google Cloud AI Platform models from Google Cloud Functions

### Example Endpoints

main.py contains the example code for two endpoints:

* Inference Endpoint: get_demo_inference_endpoint
* Model Meta Information Endpoint: get_demo_inference_meta_endpoint


### Deployment

Set your project as default GCP project with
```
$ gcloud config set project <GCP_PROJECT_ID>
```

and then deploy the endpoints with
```
$ gcloud functions deploy <ENDPOINT_NAME> --entry-point <FUNCTION_NAME> --runtime python37 --trigger-http
```

#### Deployment of the Inference Endpoint

```
$ gcloud functions deploy demo_inference --entry-point get_demo_inference_endpoint --runtime python37 --trigger-http
```

#### Deployment of the Model Meta Information Endpoint

```
$ gcloud functions deploy demo_inference_meta --entry-point get_demo_inference_meta_endpoint --runtime python37 --trigger-http
```