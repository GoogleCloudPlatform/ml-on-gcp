# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Converts images to JSON format using Base64, then uses AI Platform
prediction.

Converts images to JSON File.
Currently models support jpg format
Depending the model select the correct model type.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import googleapiclient.discovery

from ast import literal_eval

import time
import base64
import logging
import requests

INPUT_FILE = 'image.jpg'

# AI Platform
PROJECT = 'dpe-cloud-mle'
MODEL_NAME = 'pretrained_model'
MODEL_VERSION = 'cpu'

NUM_REQUESTS = 10
ENABLE_PREDICT = True


def get_classes():
    """Get classes for predictions.

    Returns:
      A dictionary with classes
    """
    url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw' \
          '/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5' \
          '/imagenet1000_clsidx_to_labels.txt'
    response = requests.get(url)
    classes = literal_eval(response.text)
    return classes


def convert_to_base64(image_file):
    """Open image and convert it to base64"""
    print('Converting Image to Base64')
    with open(image_file, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _get_service():
    """Gets service instance to start API searches.

    Returns:
      A Google API Service used to send requests.
    """
    # Create the AI Platform service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    return googleapiclient.discovery.build('ml', 'v1')


def _handle_response(classes, response):
    """Process response from AI Platfrom service."""

    if 'error' in response:
        raise RuntimeError(response['error'])
    if response:
        prediction_class = response.get('predictions')[0].get('classes') - 1
        prediction_probabilities = response.get('predictions')[0].get(
            'probabilities')
        return prediction_class
    return None


def _generate_prediction(service, classes, body):
    name = 'projects/{}/models/{}'.format(PROJECT, MODEL_NAME)
    start = time.time()
    response = service.projects().predict(
        name=name,
        body={"instances": {"image_bytes": {"b64": body}}}
    ).execute()
    end = time.time()
    print('Request took: %.3f seconds' % (end - start))
    return _handle_response(classes, response)


def main():
    base64_content = convert_to_base64(INPUT_FILE)
    if ENABLE_PREDICT:
        classes = get_classes()
        service = _get_service()
        try:
            for _ in range(0, NUM_REQUESTS):
                _generate_prediction(service, classes, base64_content)
        except requests.exceptions.HTTPError as err:
            logging.exception(err)


if __name__ == '__main__':
    main()