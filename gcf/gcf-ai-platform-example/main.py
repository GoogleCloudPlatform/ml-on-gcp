import json

from flask import current_app as app

import googleapiclient.discovery


def _generate_payload(sentence):
    """
    Helper function to prep the payload data structure. Also a good
    place to manipulate the payload (e.g., lower-casing if needed)
    """
    return {"instances": [{"sentence": sentence}]}


def _get_model_meta(service, project, model='demo_model', version=None):
    """
    Helper function to gather the model meta information. Depending on
    the type of the model (default/other), we have to hit different
    service endpoints.
    """
    url = f'projects/{project}/models/{model}'

    if version:
        url += f'/versions/{version}'
        response = service.projects().models().versions().get(name=url).execute()
        meta = response
    else:
        response = service.projects().models().get(name=url).execute()
        meta = response['defaultVersion']

    model_id = meta['name']
    return meta, model_id


def _get_model_prediction(service, project, model='demo_model',
                          version=None, body=None):
    """
    Helper function to infer a model prediction
    """
    if body is None:
        raise NotImplementedError(
            f"_get_model_prediction didn't get any payload for model {model}")

    url = f'projects/{project}/models/{model}'
    if version:
        url += f'/versions/{version}'

    response = service.projects().predict(name=url, body=body).execute()
    return response


def _connect_service():
    """
    Helper function to load the API service. In case, API credentials are
    required, add them  to the kwargs
    """
    kwargs = {'serviceName': 'ml', 'version': 'v1'}
    return googleapiclient.discovery.build(**kwargs)


def get_demo_inference_endpoint(request):
    """
    Endpoint to demonstrate requesting an inference from a model
    hosted via Google's AI Platform.

    Args:
        request with an argument `sentence` ('application/json')
    Returns:
        Response with prediction results
    """
    # expected content type is
    request_json = request.get_json(silent=True)
    sentence = request_json['sentence']

    service = _connect_service()
    project = 'drydock'
    model = 'demo_model'
    response = _get_model_prediction(service, project,
                                     model=model,
                                     body=_generate_payload(sentence))
    return json.dumps(response)


def get_demo_inference_meta_endpoint():
    """
    Endpoint to demonstrate requesting the meta information from Google's AI Platform
    Args:
        None
    Returns:
        Response with model meta information
    """
    service = _connect_service()
    project = 'drydock'
    model = 'demo_model'
    response = _get_model_meta(service, project, model=model)
    return json.dumps(response)
