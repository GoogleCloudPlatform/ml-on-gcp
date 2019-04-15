# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Inference logic."""

import tempfile
import numpy as np
import PIL

from tfserve import TFServeApp

RESNET_MODEL = '/root/tftrt_int8_resnetv2_imagenet_frozen_graph.pb'
INPUT_TENSOR = 'import/input_tensor:0'
SOFTMAX_TENSOR = 'import/softmax_tensor:0'


def encode(request_data):
  with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg') as f:
    f.write(request_data)
    img = PIL.Image.open(f.name).resize((224, 224))
    img = np.asarray(img) / 255.
  return {INPUT_TENSOR: img}


def decode(outputs):
  p = outputs[SOFTMAX_TENSOR]
  index = np.argmax(p)
  return {'class': str(index), 'prob': str(float(p[index]))}


app = TFServeApp(RESNET_MODEL, [INPUT_TENSOR], [SOFTMAX_TENSOR], encode, decode)
app.run('0.0.0.0', 5000)
