#!/bin/bash

# Copyright 2017 Google Inc. All Rights Reserved.
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

### Metadata specification
# All this metadata is pulled from the Compute Engine instance metadata server

# GCS path to training and test data
CENSUS_DATA_PATH=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/CENSUS_DATA_PATH -H "Metadata-Flavor: Google")

# GCS path where the model should be stored
MODEL_OUTPUT_PATH=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/MODEL_OUTPUT_PATH -H "Metadata-Flavor: Google")

# Number of hyperparameter search iterations
CV_ITERATIONS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/CV_ITERATIONS -H "Metadata-Flavor: Google")



### 
