# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Converts images to json file

Converts images to JSON File.
Currently models support 2 formats, tensor or jpg
Depending the model select the correct model type.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ast import literal_eval
from PIL import Image

import base64
import codecs
import json
import logging
import requests
import numpy as np

INPUT_FILE = 'image.jpg'
OUTPUT_FILE = '/tmp/out.json'

LOAD_BALANCER = '' # Enter your Public Load Balancer IP Address.
URL = 'http://%s/v1/models/default:predict' % LOAD_BALANCER
UPLOAD_FOLDER = '/tmp/'
# Wait this long for outgoing HTTP connections to be established.
_CONNECT_TIMEOUT_SECONDS = 90
# Wait this long to read from an HTTP socket.
_READ_TIMEOUT_SECONDS = 120
MODEL_TYPE = 'jpg'  # tensor | jpg


def get_classes():
  url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw' \
        '/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5' \
        '/imagenet1000_clsidx_to_labels.txt'
  response = requests.get(url)
  classes = literal_eval(response.text)
  return classes


def convert_to_json(image_file):
  """Open image, convert it to numpy and create JSON request"""
  img = Image.open(image_file).resize((240, 240))
  img_array = np.array(img)
  predict_request = {"instances": [img_array.tolist()]}
  json.dump(predict_request, codecs.open(OUTPUT_FILE, 'w', encoding='utf-8'),
            separators=(',', ':'), sort_keys=True, indent=4)
  return predict_request


def convert_to_base64(image_file):
  """Open image and convert it to base64"""
  with open(image_file, 'rb') as f:
    jpeg_bytes = base64.b64encode(f.read()).decode('utf-8')
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
    # Write JSON to file
    with open(OUTPUT_FILE, 'w') as f:
      f.write(predict_request)
    return predict_request


def model_predict(predict_request):
  """Sends Image for prediction."""
  session = requests.Session()
  try:
    response = session.post(
      URL,
      data=predict_request,
      timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
      allow_redirects=False)
    response.raise_for_status()
    return response.json()
  except requests.exceptions.HTTPError as err:
    logging.exception(err)
    if err.response.status_code == 400:
      logging.exception('Server error %s', URL)
      return
    if err.response.status_code == 404:
      logging.exception('Page not found %s', URL)
      return


def main():
  if MODEL_TYPE == 'tensor':
    predict_request = convert_to_json(INPUT_FILE)
  elif MODEL_TYPE == 'jpg':
    predict_request = convert_to_base64(INPUT_FILE)
  else:
    logging.error('Invalid Model Type')
    return
  classes = get_classes()
  response = model_predict(predict_request)
  if response:
    prediction_class = response.get('predictions')[0].get('classes')
    print('Prediction: [%d] %s' % (prediction_class, classes[prediction_class]))


if __name__ == '__main__':
  main()
