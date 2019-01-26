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
"""Web server to access Prediction server in GCP.

A Flask server used for Predictions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ast import literal_eval
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from PIL import Image
from werkzeug.utils import secure_filename

import base64
import logging
import numpy as np
import os
import requests

app = Flask(__name__)

MODEL_TYPE = 'jpg'  # tensor | jpg
LOAD_BALANCER = ''
URL = 'http://%s/v1/models/default:predict' % LOAD_BALANCER
UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# Wait this long for outgoing HTTP connections to be established.
_CONNECT_TIMEOUT_SECONDS = 90
# Wait this long to read from an HTTP socket.
_READ_TIMEOUT_SECONDS = 120

app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_classes():
  """Get classes...

  Returns:
    A dictionary with class information: {1: 'cat'}
  """
  # Classes dictionary.
  url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw' \
        '/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5' \
        '/imagenet1000_clsidx_to_labels.txt'
  response = requests.get(url)
  response.raise_for_status()
  return literal_eval(response.text)


def convert_to_json(image_file):
  """Open image, convert it to numpy and create JSON request

  Args:
    image_file: A `str` with file path

  Returns:
    A dictionary used to get inference using Tensors.
  """
  img = Image.open(image_file).resize((240, 240))
  img_array = np.array(img)
  predict_request = {"instances": [img_array.tolist()]}
  return predict_request


def convert_to_base64(image_file):
  """Open image and convert it to base64

  Args:
    image_file: A `str` with file path

  Returns:
    A dictionary used to get inference using Image Base64.
  """
  with open(image_file, 'rb') as f:
    jpeg_bytes = base64.b64encode(f.read()).decode('utf-8')
    return '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes


def conversion_helper(model_type, filename):
  """

  :param model_type:
  :param filename:
  :return:
  """
  if model_type == 'jpg':
    return convert_to_base64(
      os.path.join(app.config['UPLOAD_FOLDER'], filename))
  elif model_type == 'tensor':
    return convert_to_json(
      os.path.join(app.config['UPLOAD_FOLDER'], filename))
  else:
    logging.error('Invalid model')
    return redirect(request.url)


def model_predict(predict_request):
  """Sends Image for prediction.

  Args:
    predict_request: A dictionary used for Inference

  Returns:
    A JSON object with inference response.
  """
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


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
  logging.info(request.files)
  if 'image' not in request.files:
    logging.error('Error. No file part')
    return redirect(request.url)
  file = request.files['image']
  if not file.filename:
    logging.error('Not selected file')
    return redirect(request.url)
  if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

  # Call API for prediction.
  predict_request = conversion_helper(MODEL_TYPE, filename)
  if predict_request:
    response = model_predict(predict_request)
    if response:
      prediction_class = response.get('predictions')[0].get('classes')
      return render_template('index.html', init=True,
                             prediction=classes[prediction_class])
    else:
      return render_template('index.html', init=True,
                             prediction=None)
  else:
    logging.error('Not a valid request')
    return redirect(request.url)


# Obtain classes before Server starts.
classes = get_classes()

if __name__ == '__main__':
  app.run(debug=True, port=8001)
